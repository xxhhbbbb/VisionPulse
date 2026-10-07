# SPDX-License-Identifier: Apache-2.0
"""
Triton post-anchor block-sparse attention.

This kernel is used by layers `> anchor_layer`.
It keeps *all* non-visual key blocks and only the selected visual blocks exported
by the anchor layer. The keep decision is query-row dependent, so the sparse mask
is stored as `kept_block_mask[b, q, block]`.

Implementation notes:
    - the kernel still processes the sequence block by block, in FlashAttention's
      online-softmax style, so numerical behavior remains aligned with dense attention
    - for blocks not selected by any row of the current query tile, the kernel skips
      all K/V loads for that block entirely
    - for blocks selected only by a subset of rows, the block is loaded once and rows
      not selecting it are forced to `-inf` before the online-softmax update
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_SUPPORTED_HEAD_DIMS = {16, 32, 64, 128}


@triton.jit
def _block_sparse_flash_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    mask_ptr,
    keep_ptr,
    out_ptr,
    stride_qb,
    stride_qh,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vk,
    stride_vd,
    stride_mb,
    stride_mq,
    stride_mk,
    stride_pb,
    stride_pq,
    stride_pk,
    stride_ob,
    stride_oq,
    stride_oh,
    stride_od,
    num_heads,
    q_len,
    k_len,
    num_key_blocks,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = (
        q_ptr
        + batch_idx * stride_qb
        + head_idx * stride_qh
        + offs_m[:, None] * stride_qq
        + offs_d[None, :] * stride_qd
    )
    q_mask = (offs_m[:, None] < q_len) & (offs_d[None, :] < BLOCK_DMODEL)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    for block_idx in range(0, num_key_blocks):
        keep_ptrs = keep_ptr + batch_idx * stride_pb + offs_m * stride_pq + block_idx * stride_pk
        row_keep = tl.load(keep_ptrs, mask=offs_m < q_len, other=0).to(tl.int1)
        keep_any = tl.sum(row_keep.to(tl.int32), axis=0)

        if keep_any > 0:
            start_n = block_idx * BLOCK_N
            curr_n = start_n + offs_n
            valid_q = offs_m < q_len
            valid_k = curr_n < k_len
            valid_tile = valid_q[:, None] & valid_k[None, :]

            kv_mask = valid_k[:, None] & (offs_d[None, :] < BLOCK_DMODEL)
            k_ptrs = (
                k_ptr
                + batch_idx * stride_kb
                + head_idx * stride_kh
                + curr_n[:, None] * stride_kk
                + offs_d[None, :] * stride_kd
            )
            v_ptrs = (
                v_ptr
                + batch_idx * stride_vb
                + head_idx * stride_vh
                + curr_n[:, None] * stride_vk
                + offs_d[None, :] * stride_vd
            )
            k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
            v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

            qk = tl.dot(q, tl.trans(k)) * sm_scale
            qk = tl.where(valid_tile, qk, float("-inf"))

            mask_ptrs = (
                mask_ptr
                + batch_idx * stride_mb
                + offs_m[:, None] * stride_mq
                + curr_n[None, :] * stride_mk
            )
            mask_vals = tl.load(mask_ptrs, mask=valid_tile, other=float("-inf")).to(tl.float32)
            qk = qk + mask_vals
            qk = tl.where(row_keep[:, None], qk, float("-inf"))

            block_row_max = tl.max(qk, axis=1)
            row_updates = row_keep & (block_row_max != float("-inf"))
            m_ij = tl.where(row_updates, tl.maximum(m_i, block_row_max), m_i)
            p = tl.where(row_updates[:, None], tl.exp(qk - m_ij[:, None]), 0.0)
            alpha = tl.where(row_updates, tl.exp(m_i - m_ij), 1.0)
            # For the generic q_len>1 sparse path we must preserve the same
            # row-by-row online softmax update as dense FlashAttention, but
            # still use a matrix multiply for the PV contraction. The previous
            # elementwise-broadcast implementation:
            #
            #   tl.sum(v * p[:, :, None], axis=1)
            #
            # materialized a [BLOCK_M, BLOCK_N, D] temporary and was the main
            # reason the q_len>1 post-anchor sparse kernel became drastically
            # slower than dense prefill. `tl.dot(p, v)` is mathematically
            # identical here and matches the dense kernel's contraction path.
            acc_update = tl.dot(p.to(v.dtype), v)
            acc = acc * alpha[:, None] + acc_update
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_ij

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    out = acc / l_safe[:, None]
    out_ptrs = (
        out_ptr
        + batch_idx * stride_ob
        + offs_m[:, None] * stride_oq
        + head_idx * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_mask = (offs_m[:, None] < q_len) & (offs_d[None, :] < BLOCK_DMODEL)
    tl.store(out_ptrs, out.to(tl.float32), mask=out_mask)


@triton.jit
def _block_sparse_flash_decode_q1_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    mask_ptr,
    keep_idx_ptr,
    keep_count_ptr,
    out_ptr,
    stride_qb,
    stride_qh,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vk,
    stride_vd,
    stride_mb,
    stride_mk,
    stride_ib,
    stride_ik,
    stride_cb,
    stride_ob,
    stride_oq,
    stride_oh,
    stride_od,
    num_heads,
    k_len,
    max_kept_blocks,
    sm_scale,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid_bh = tl.program_id(0)

    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = q_ptr + batch_idx * stride_qb + head_idx * stride_qh + offs_d * stride_qd
    q_mask = offs_d < BLOCK_DMODEL
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full((1,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((1,), dtype=tl.float32)
    acc = tl.zeros((1, BLOCK_DMODEL), dtype=tl.float32)

    keep_count = tl.load(keep_count_ptr + batch_idx * stride_cb).to(tl.int32)

    for keep_slot in range(0, max_kept_blocks):
        if keep_slot < keep_count:
            block_idx = tl.load(keep_idx_ptr + batch_idx * stride_ib + keep_slot * stride_ik).to(tl.int32)
            start_n = block_idx * BLOCK_N
            curr_n = start_n + offs_n
            valid_k = curr_n < k_len

            kv_mask = valid_k[:, None] & (offs_d[None, :] < BLOCK_DMODEL)
            k_ptrs = (
                k_ptr
                + batch_idx * stride_kb
                + head_idx * stride_kh
                + curr_n[:, None] * stride_kk
                + offs_d[None, :] * stride_kd
            )
            v_ptrs = (
                v_ptr
                + batch_idx * stride_vb
                + head_idx * stride_vh
                + curr_n[:, None] * stride_vk
                + offs_d[None, :] * stride_vd
            )
            k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
            v = tl.load(v_ptrs, mask=kv_mask, other=0.0)

            qk_row = tl.sum(k * q[None, :], axis=1) * sm_scale
            qk = tl.where(valid_k[None, :], qk_row[None, :], float("-inf"))

            mask_ptrs = mask_ptr + batch_idx * stride_mb + curr_n[None, :] * stride_mk
            mask_vals = tl.load(mask_ptrs, mask=valid_k[None, :], other=float("-inf")).to(tl.float32)
            qk = qk + mask_vals

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_ij[:, None])
            alpha = tl.exp(m_i - m_ij)
            acc_update = tl.sum(v.to(tl.float32) * p[:, :, None], axis=1)
            acc = acc * alpha[:, None] + acc_update
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_ij

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    out = acc / l_safe[:, None]
    out_ptrs = (
        out_ptr
        + batch_idx * stride_ob
        + head_idx * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_mask = offs_d[None, :] < BLOCK_DMODEL
    tl.store(out_ptrs, out.to(tl.float32), mask=out_mask)


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    kept_block_mask: torch.Tensor,
    block_size: int,
) -> None:
    if not query.is_cuda or not key.is_cuda or not value.is_cuda or not kept_block_mask.is_cuda:
        raise ValueError("Triton block-sparse attention requires CUDA tensors.")
    if attention_mask is None or not attention_mask.is_cuda:
        raise ValueError("Triton block-sparse attention requires a CUDA additive attention_mask tensor.")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Expected Q/K/V to have shape [B, H, T, D].")
    if kept_block_mask.ndim != 3:
        raise ValueError("kept_block_mask must have shape [B, Q, num_key_blocks].")
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError("Batch dimensions of Q/K/V must match.")
    if query.shape[1] != key.shape[1] or query.shape[1] != value.shape[1]:
        raise ValueError("Head dimensions of Q/K/V must match for Triton block-sparse attention.")
    if key.shape[2] != value.shape[2] or query.shape[3] != key.shape[3] or key.shape[3] != value.shape[3]:
        raise ValueError("Q/K/V shape mismatch.")
    if attention_mask.shape[0] != query.shape[0] or attention_mask.shape[2] != query.shape[2] or attention_mask.shape[3] != key.shape[2]:
        raise ValueError("attention_mask shape must be [B, 1, Q, K].")
    if kept_block_mask.shape[0] != query.shape[0] or kept_block_mask.shape[1] != query.shape[2]:
        raise ValueError("kept_block_mask batch/query dimensions must match Q.")
    expected_num_blocks = triton.cdiv(key.shape[2], block_size)
    if kept_block_mask.shape[2] != expected_num_blocks:
        raise ValueError(
            f"kept_block_mask last dim mismatch: expected {expected_num_blocks}, got {kept_block_mask.shape[2]}"
        )
    if query.shape[-1] not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(f"Unsupported head_dim={query.shape[-1]}; supported dims are {_SUPPORTED_HEAD_DIMS}.")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Triton block-sparse attention currently supports fp16 / bf16 inputs only.")
    if block_size not in (16, 32, 64, 128):
        raise ValueError("Triton block-sparse attention supports block_size in {16, 32, 64, 128}.")


def _validate_decode_q1_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    kept_block_indices: torch.Tensor,
    kept_block_counts: torch.Tensor,
    block_size: int,
) -> None:
    if not query.is_cuda or not key.is_cuda or not value.is_cuda or not kept_block_indices.is_cuda or not kept_block_counts.is_cuda:
        raise ValueError("Decode-q1 sparse attention requires CUDA tensors.")
    if attention_mask is None or not attention_mask.is_cuda:
        raise ValueError("Decode-q1 sparse attention requires a CUDA additive attention_mask tensor.")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Expected Q/K/V to have shape [B, H, T, D].")
    if query.shape[2] != 1:
        raise ValueError("Decode-q1 sparse attention requires query length 1.")
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError("Batch dimensions of Q/K/V must match.")
    if query.shape[1] != key.shape[1] or query.shape[1] != value.shape[1]:
        raise ValueError("Head dimensions of Q/K/V must match for decode-q1 sparse attention.")
    if key.shape[2] != value.shape[2] or query.shape[3] != key.shape[3] or key.shape[3] != value.shape[3]:
        raise ValueError("Q/K/V shape mismatch.")
    if attention_mask.ndim != 4:
        raise ValueError("Expected attention_mask to have shape [B, 1, Q, K].")
    if attention_mask.shape[0] != query.shape[0] or attention_mask.shape[2] != 1 or attention_mask.shape[3] != key.shape[2]:
        raise ValueError("attention_mask shape must be [B, 1, 1, K].")
    if query.shape[-1] not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(f"Unsupported head_dim={query.shape[-1]}; supported dims are {_SUPPORTED_HEAD_DIMS}.")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Decode-q1 sparse attention currently supports fp16 / bf16 inputs only.")
    if block_size not in (16, 32, 64, 128):
        raise ValueError("Decode-q1 sparse attention supports block_size in {16, 32, 64, 128}.")
    if kept_block_indices.ndim != 2:
        raise ValueError("kept_block_indices must have shape [B, max_kept_blocks].")
    if kept_block_counts.ndim != 1:
        raise ValueError("kept_block_counts must have shape [B].")
    if kept_block_indices.shape[0] != query.shape[0] or kept_block_counts.shape[0] != query.shape[0]:
        raise ValueError("kept_block_indices / kept_block_counts batch dimension mismatch.")
    if kept_block_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("kept_block_indices must be int32 or int64.")
    if kept_block_counts.dtype not in (torch.int32, torch.int64):
        raise ValueError("kept_block_counts must be int32 or int64.")


def triton_block_sparse_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    kept_block_mask: torch.Tensor,
    sm_scale: float,
    block_size: int,
    *,
    block_m: int = 32,
) -> torch.Tensor:
    """
    Block-sparse attention for layers after the anchor layer.

    Args:
        kept_block_mask: [B, Q, num_key_blocks] bool.
            True means the corresponding query row is allowed to attend to the key block.
            This mask must already include all non-visual blocks.
    Returns:
        out: [B, Q, H, D]
    """
    _validate_inputs(query, key, value, attention_mask, kept_block_mask, block_size)

    if block_m == 32 and query.shape[2] <= 16:
        block_m = 16

    out = torch.empty(
        query.shape[0],
        query.shape[2],
        query.shape[1],
        query.shape[3],
        device=query.device,
        dtype=torch.float32,
    )
    grid = (triton.cdiv(query.shape[2], block_m), query.shape[0] * query.shape[1])
    if query.shape[2] <= 16:
        num_warps = 2
    else:
        num_warps = 4 if query.shape[-1] <= 64 else 8

    _block_sparse_flash_fwd_kernel[grid](
        query,
        key,
        value,
        attention_mask,
        kept_block_mask,
        out,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        attention_mask.stride(0),
        attention_mask.stride(2),
        attention_mask.stride(3),
        kept_block_mask.stride(0),
        kept_block_mask.stride(1),
        kept_block_mask.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        query.shape[1],
        query.shape[2],
        key.shape[2],
        kept_block_mask.shape[2],
        sm_scale,
        BLOCK_M=block_m,
        BLOCK_N=block_size,
        BLOCK_DMODEL=query.shape[-1],
        num_warps=num_warps,
        num_stages=2,
    )
    return out.to(query.dtype)


def triton_block_sparse_attention_decode_q1_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    kept_block_indices: torch.Tensor,
    kept_block_counts: torch.Tensor,
    sm_scale: float,
    block_size: int,
) -> torch.Tensor:
    """
    Exact decode-time (`Q=1`) sparse attention.

    This specialized kernel is mathematically equivalent to the generic
    `triton_block_sparse_attention_forward` path with `kept_block_mask[:, 0, :]`,
    but it iterates directly over the packed kept-block index list instead of
    scanning every possible key block and testing whether it is kept.

    Equivalence guarantees:
        - the kept block set is identical to the boolean keep mask
        - kept blocks are processed in ascending block-index order
        - dropped blocks are skipped exactly as in the generic kernel
        - online softmax semantics are unchanged
    """
    _validate_decode_q1_inputs(
        query, key, value, attention_mask, kept_block_indices, kept_block_counts, block_size
    )

    out = torch.empty(
        query.shape[0],
        1,
        query.shape[1],
        query.shape[3],
        device=query.device,
        dtype=torch.float32,
    )
    max_kept_blocks = kept_block_indices.shape[1]
    grid = (query.shape[0] * query.shape[1],)

    _block_sparse_flash_decode_q1_fwd_kernel[grid](
        query,
        key,
        value,
        attention_mask,
        kept_block_indices,
        kept_block_counts,
        out,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        attention_mask.stride(0),
        attention_mask.stride(3),
        kept_block_indices.stride(0),
        kept_block_indices.stride(1),
        kept_block_counts.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        query.shape[1],
        key.shape[2],
        max_kept_blocks,
        sm_scale,
        BLOCK_N=block_size,
        BLOCK_DMODEL=query.shape[-1],
        num_warps=1 if query.shape[-1] <= 64 else 2,
        num_stages=2,
    )
    return out.to(query.dtype)
