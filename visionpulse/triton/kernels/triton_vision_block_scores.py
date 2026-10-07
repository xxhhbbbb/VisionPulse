# SPDX-License-Identifier: Apache-2.0
"""
Triton anchor-layer attention kernel.

The anchor layer performs *full* attention, but simultaneously exports block-level
score logits used by the sparse-vision budget policy.

Block-score definition:
    - attention output uses the standard attention scale `sm_scale`
    - block score logits use a separate score temperature `score_temperature`
    - for each query row and key block, the kernel first computes
          block_score_logit = max_{k in block}(QK^T * sm_scale / score_temperature + mask)
    - the kernel then performs an online normalization over the block dimension and writes
      the final normalized block score
          block_score = exp(block_score_logit - m_score) / l_score
      where `(m_score, l_score)` are the online softmax statistics over block logits
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

_SUPPORTED_HEAD_DIMS = {16, 32, 64, 128}


@triton.jit
def _anchor_flash_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    mask_ptr,
    out_ptr,
    score_ptr,
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
    stride_ob,
    stride_oq,
    stride_oh,
    stride_od,
    stride_sb,
    stride_sh,
    stride_sq,
    stride_sk,
    num_heads,
    q_len,
    k_len,
    sm_scale,
    score_scale,
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
    m_score_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_score_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    for start_n in range(0, k_len, BLOCK_N):
        curr_n = start_n + offs_n
        block_idx = start_n // BLOCK_N
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

        qk_raw = tl.dot(q, tl.trans(k))

        mask_ptrs = (
            mask_ptr
            + batch_idx * stride_mb
            + offs_m[:, None] * stride_mq
            + curr_n[None, :] * stride_mk
        )
        mask_vals = tl.load(mask_ptrs, mask=valid_tile, other=float("-inf")).to(tl.float32)

        attn_logits = qk_raw * sm_scale
        attn_logits = tl.where(valid_tile, attn_logits, float("-inf"))
        attn_logits = attn_logits + mask_vals

        score_logits = qk_raw * score_scale
        score_logits = tl.where(valid_tile, score_logits, float("-inf"))
        score_logits = score_logits + mask_vals

        block_score_logit = tl.max(score_logits, axis=1)
        score_row_ptrs = (
            score_ptr
            + batch_idx * stride_sb
            + head_idx * stride_sh
            + offs_m * stride_sq
            + block_idx * stride_sk
        )
        tl.store(score_row_ptrs, block_score_logit, mask=valid_q)

        m_score_new = tl.maximum(m_score_i, block_score_logit)
        alpha_score = tl.exp(m_score_i - m_score_new)
        contrib_score = tl.exp(block_score_logit - m_score_new)
        l_score_i = l_score_i * alpha_score + contrib_score
        m_score_i = m_score_new

        m_ij = tl.maximum(m_i, tl.max(attn_logits, axis=1))
        p = tl.exp(attn_logits - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
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

    l_score_safe = tl.where(l_score_i > 0, l_score_i, 1.0)
    num_key_blocks = tl.cdiv(k_len, BLOCK_N)
    for block_idx in range(0, num_key_blocks):
        score_row_ptrs = (
            score_ptr
            + batch_idx * stride_sb
            + head_idx * stride_sh
            + offs_m * stride_sq
            + block_idx * stride_sk
        )
        raw_block_score = tl.load(score_row_ptrs, mask=offs_m < q_len, other=float("-inf"))
        norm_block_score = tl.exp(raw_block_score - m_score_i) / l_score_safe
        norm_block_score = tl.where(offs_m < q_len, norm_block_score, 0.0)
        tl.store(score_row_ptrs, norm_block_score, mask=offs_m < q_len)


@triton.jit
def _anchor_flash_decode_q1_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    mask_ptr,
    out_ptr,
    score_ptr,
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
    stride_ob,
    stride_oq,
    stride_oh,
    stride_od,
    stride_sb,
    stride_sh,
    stride_sq,
    stride_sk,
    num_heads,
    k_len,
    sm_scale,
    score_scale,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid_bh = tl.program_id(0)

    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = q_ptr + batch_idx * stride_qb + head_idx * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptrs, mask=offs_d < BLOCK_DMODEL, other=0.0)

    m_i = tl.full((1,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((1,), dtype=tl.float32)
    m_score_i = tl.full((1,), float("-inf"), dtype=tl.float32)
    l_score_i = tl.zeros((1,), dtype=tl.float32)
    acc = tl.zeros((1, BLOCK_DMODEL), dtype=tl.float32)

    num_key_blocks = tl.cdiv(k_len, BLOCK_N)
    for block_idx in range(0, num_key_blocks):
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

        qk_row = tl.sum(k * q[None, :], axis=1)

        mask_ptrs = mask_ptr + batch_idx * stride_mb + curr_n[None, :] * stride_mk
        mask_vals = tl.load(mask_ptrs, mask=valid_k[None, :], other=float("-inf")).to(tl.float32)
        mask_row = tl.reshape(mask_vals, (BLOCK_N,))

        attn_logits = qk_row[None, :] * sm_scale
        attn_logits = tl.where(valid_k[None, :], attn_logits, float("-inf"))
        attn_logits = attn_logits + mask_vals

        score_logits = qk_row * score_scale
        score_logits = tl.where(valid_k, score_logits, float("-inf"))
        score_logits = score_logits + mask_row
        block_score_logit = tl.max(score_logits, axis=0)

        score_row_ptrs = (
            score_ptr
            + batch_idx * stride_sb
            + head_idx * stride_sh
            + block_idx * stride_sk
        )
        tl.store(score_row_ptrs, block_score_logit)

        m_score_new = tl.maximum(m_score_i, block_score_logit)
        alpha_score = tl.exp(m_score_i - m_score_new)
        contrib_score = tl.exp(block_score_logit - m_score_new)
        l_score_i = l_score_i * alpha_score + contrib_score
        m_score_i = m_score_new

        m_ij = tl.maximum(m_i, tl.max(attn_logits, axis=1))
        p = tl.exp(attn_logits - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        acc_update = tl.sum(v * p.to(v.dtype)[:, :, None], axis=1)
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
    tl.store(out_ptrs, out.to(tl.float32), mask=offs_d[None, :] < BLOCK_DMODEL)

    m_score_scalar = tl.max(m_score_i, axis=0)
    l_score_scalar = tl.max(l_score_i, axis=0)
    l_score_safe = tl.where(l_score_scalar > 0, l_score_scalar, 1.0)
    for block_idx in range(0, num_key_blocks):
        score_row_ptrs = (
            score_ptr
            + batch_idx * stride_sb
            + head_idx * stride_sh
            + block_idx * stride_sk
        )
        raw_block_score = tl.load(score_row_ptrs)
        norm_block_score = tl.exp(raw_block_score - m_score_scalar) / l_score_safe
        tl.store(score_row_ptrs, norm_block_score)


def _check_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    block_size: int,
) -> Tuple[int, int, int, int, int]:
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("Triton anchor attention requires CUDA tensors.")
    if attention_mask is None or not attention_mask.is_cuda:
        raise ValueError("Triton anchor attention requires a CUDA additive attention_mask tensor.")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Expected Q/K/V to have shape [B, H, T, D].")
    if attention_mask.ndim != 4:
        raise ValueError("Expected attention_mask to have shape [B, 1, Q, K].")
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError("Batch dimensions of Q/K/V must match.")
    if query.shape[1] != key.shape[1] or query.shape[1] != value.shape[1]:
        raise ValueError("Head dimensions of Q/K/V must match for Triton anchor attention.")
    if key.shape[2] != value.shape[2] or query.shape[3] != key.shape[3] or key.shape[3] != value.shape[3]:
        raise ValueError("Q/K/V shape mismatch.")
    if attention_mask.shape[0] != query.shape[0] or attention_mask.shape[2] != query.shape[2] or attention_mask.shape[3] != key.shape[2]:
        raise ValueError("attention_mask shape must be [B, 1, Q, K].")

    head_dim = query.shape[-1]
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(f"Unsupported head_dim={head_dim}; supported dims are {_SUPPORTED_HEAD_DIMS}.")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Triton anchor attention currently supports fp16 / bf16 inputs only.")
    if block_size not in (16, 32, 64, 128):
        raise ValueError("Triton anchor attention supports block_size in {16, 32, 64, 128}.")
    return query.shape[0], query.shape[1], query.shape[2], key.shape[2], head_dim


def triton_anchor_dense_attention_with_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    sm_scale: float,
    score_temperature: float,
    block_size: int,
    *,
    block_m: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Full attention at the sparse-vision anchor layer.

    Returns:
        out: [B, Q, H, D]
        block_scores: [B, H, Q, num_key_blocks] float32
    """
    _, _, q_len, k_len, _ = _check_inputs(query, key, value, attention_mask, block_size)
    if score_temperature <= 0:
        raise ValueError("score_temperature must be positive.")

    if block_m == 64:
        if q_len <= 16:
            block_m = 16
        elif q_len <= 32:
            block_m = 32

    num_key_blocks = triton.cdiv(k_len, block_size)
    out = torch.empty(
        query.shape[0],
        query.shape[2],
        query.shape[1],
        query.shape[3],
        device=query.device,
        dtype=torch.float32,
    )
    # Every valid [batch, head, query_row, block] entry is overwritten by the
    # kernel before being read back during the final normalization pass, so a
    # zero-fill / -inf-fill here is unnecessary launch-side work.
    block_scores = torch.empty(
        (query.shape[0], query.shape[1], query.shape[2], num_key_blocks),
        device=query.device,
        dtype=torch.float32,
    )

    score_scale = float(sm_scale) / float(score_temperature)
    if q_len == 1:
        grid = (query.shape[0] * query.shape[1],)
        _anchor_flash_decode_q1_fwd_kernel[grid](
            query,
            key,
            value,
            attention_mask,
            out,
            block_scores,
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
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            block_scores.stride(0),
            block_scores.stride(1),
            block_scores.stride(2),
            block_scores.stride(3),
            query.shape[1],
            k_len,
            sm_scale,
            score_scale,
            BLOCK_N=block_size,
            BLOCK_DMODEL=query.shape[-1],
            num_warps=1 if query.shape[-1] <= 64 else 2,
            num_stages=2,
        )
    else:
        grid = (triton.cdiv(q_len, block_m), query.shape[0] * query.shape[1])
        if q_len <= 16:
            num_warps = 2
        else:
            num_warps = 4 if query.shape[-1] <= 64 else 8

        _anchor_flash_fwd_kernel[grid](
            query,
            key,
            value,
            attention_mask,
            out,
            block_scores,
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
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            block_scores.stride(0),
            block_scores.stride(1),
            block_scores.stride(2),
            block_scores.stride(3),
            query.shape[1],
            q_len,
            k_len,
            sm_scale,
            score_scale,
            BLOCK_M=block_m,
            BLOCK_N=block_size,
            BLOCK_DMODEL=query.shape[-1],
            num_warps=num_warps,
            num_stages=2,
        )
    return out.to(query.dtype), block_scores
