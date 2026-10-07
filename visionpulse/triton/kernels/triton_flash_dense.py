# SPDX-License-Identifier: Apache-2.0
"""
Triton dense causal attention for Qwen3-VL sparse-vision inference.

This module implements the dense baseline path used by layers `< anchor_layer`.
It is intentionally inference-only: the sparse-vision pipeline is used for formal
inference / evaluation, and the kernel therefore focuses on numerically correct
forward execution with explicit additive masks from HuggingFace.

Tensor conventions used throughout this file:
    Q: [B, H, Q, D]
    K: [B, H, K, D]
    V: [B, H, K, D]
    attention_mask: [B, 1, Q, K] additive mask (0 for valid, -inf / very negative for invalid)
    O: [B, Q, H, D]  (matches the local attention wrappers in this repository)
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

_SUPPORTED_HEAD_DIMS = {16, 32, 64, 128}


@triton.jit
def _dense_flash_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    mask_ptr,
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
    stride_ob,
    stride_oq,
    stride_oh,
    stride_od,
    num_heads,
    q_len,
    k_len,
    sm_scale,
    HAS_MASK: tl.constexpr,
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
    # Keep Q/K/V in the original low-precision dtype so tl.dot can use tensor-core
    # paths; accumulation still happens in fp32 through the online-softmax state.
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    for start_n in range(0, k_len, BLOCK_N):
        curr_n = start_n + offs_n
        kv_mask = (curr_n[:, None] < k_len) & (offs_d[None, :] < BLOCK_DMODEL)

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

        valid_q = offs_m < q_len
        valid_k = curr_n < k_len
        qk = tl.where(valid_q[:, None] & valid_k[None, :], qk, float("-inf"))

        if HAS_MASK:
            mask_ptrs = (
                mask_ptr
                + batch_idx * stride_mb
                + offs_m[:, None] * stride_mq
                + curr_n[None, :] * stride_mk
            )
            mask_vals = tl.load(mask_ptrs, mask=valid_q[:, None] & valid_k[None, :], other=float("-inf"))
            qk = qk + mask_vals.to(tl.float32)

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
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


def _check_kernel_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[int, int, int, int]:
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("Triton dense attention requires CUDA tensors.")
    if attention_mask is None or not attention_mask.is_cuda:
        raise ValueError("Triton dense attention requires a CUDA additive attention_mask tensor.")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Expected Q/K/V to have shape [B, H, T, D].")
    if attention_mask.ndim != 4:
        raise ValueError("Expected attention_mask to have shape [B, 1, Q, K].")
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError("Batch dimensions of Q/K/V must match.")
    if query.shape[1] != key.shape[1] or query.shape[1] != value.shape[1]:
        raise ValueError("Head dimensions of Q/K/V must match for Triton dense attention.")
    if key.shape[2] != value.shape[2]:
        raise ValueError("Key/value sequence length mismatch.")
    if key.shape[3] != value.shape[3] or query.shape[3] != key.shape[3]:
        raise ValueError("Head dimension mismatch across Q/K/V.")
    if attention_mask.shape[0] != query.shape[0] or attention_mask.shape[2] != query.shape[2] or attention_mask.shape[3] != key.shape[2]:
        raise ValueError("attention_mask shape must be [B, 1, Q, K].")

    bsz, heads, q_len, head_dim = query.shape
    k_len = key.shape[2]
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(f"Unsupported head_dim={head_dim}; supported dims are {_SUPPORTED_HEAD_DIMS}.")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Triton dense attention currently supports fp16 / bf16 inputs only.")
    return bsz, heads, q_len, k_len


def triton_dense_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
    sm_scale: float,
    *,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    """
    Dense attention baseline used by layers before the sparse-vision anchor layer.

    Returns output with shape [B, Q, H, D] to match the existing attention wrapper.
    """
    _, _, q_len, _ = _check_kernel_inputs(query, key, value, attention_mask)

    out = torch.empty(
        query.shape[0],
        query.shape[2],
        query.shape[1],
        query.shape[3],
        device=query.device,
        dtype=torch.float32,
    )

    grid = (triton.cdiv(q_len, block_m), query.shape[0] * query.shape[1])
    num_warps = 4 if query.shape[-1] <= 64 else 8

    _dense_flash_fwd_kernel[grid](
        query,
        key,
        value,
        attention_mask,
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
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        query.shape[1],
        query.shape[2],
        key.shape[2],
        sm_scale,
        HAS_MASK=True,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_DMODEL=query.shape[-1],
        num_warps=num_warps,
        num_stages=2,
    )
    return out.to(query.dtype)
