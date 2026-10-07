# SPDX-License-Identifier: Apache-2.0
"""
Sparse-vision budget utilities.

This module is the single source of truth for the block-score → visual-mass → budget
→ selected-visual-block pipeline used by the Triton sparse-vision attention path.

Notation:
    block_scores: [B, H, Q, NB]
        block-level attention scores already normalized inside the anchor kernel.
    visual_block_mask: [B, NB] bool
        True only for fully-visual blocks that are eligible for pruning.
        Mixed visual/text boundary blocks are intentionally excluded and must stay kept.
"""

from __future__ import annotations

from typing import Optional

import torch


def _ensure_device_dtype(
    tensor: torch.Tensor, *, device: torch.device, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    target_dtype = tensor.dtype if dtype is None else dtype
    if tensor.device == device and tensor.dtype == target_dtype:
        return tensor
    return tensor.to(device=device, dtype=target_dtype)


def normalize_visual_token_mask(
    visual_token_mask: torch.Tensor,
    *,
    batch_size: int,
    seq_len_k: int,
    device: torch.device,
) -> torch.Tensor:
    """Pad / trim visual token mask to the current KV length."""
    mask = _ensure_device_dtype(visual_token_mask, device=device, dtype=torch.bool)
    if mask.shape[0] != batch_size:
        raise ValueError(
            f"visual_token_mask batch mismatch: expected {batch_size}, got {mask.shape[0]}"
        )
    if mask.shape[-1] < seq_len_k:
        pad = torch.zeros(
            batch_size, seq_len_k - mask.shape[-1], device=device, dtype=torch.bool
        )
        mask = torch.cat([mask, pad], dim=-1)
    elif mask.shape[-1] > seq_len_k:
        mask = mask[:, :seq_len_k]
    return mask


def visual_block_mask_from_token_mask(
    visual_token_mask: torch.Tensor, block_size: int
) -> torch.Tensor:
    """
    Convert a token-level visual mask [B, K] into the *prunable* visual-block mask [B, NB].

    Important semantic choice:
        mixed boundary blocks must be preserved exactly, so a block is marked as
        visual/prunable only when all tokens in that block are visual.

    Consequence:
        - pure visual blocks      -> True  (eligible for top-k pruning)
        - mixed visual/text block -> False (falls into the always-kept prefix)
        - pure non-visual block   -> False (always kept)
    """
    if visual_token_mask.ndim != 2:
        raise ValueError("visual_token_mask must have shape [B, K]")
    bsz, seq_len = visual_token_mask.shape
    num_blocks = (seq_len + block_size - 1) // block_size
    padded = torch.zeros(
        bsz, num_blocks * block_size, device=visual_token_mask.device, dtype=torch.bool
    )
    padded[:, :seq_len] = visual_token_mask
    block_mask = padded.view(bsz, num_blocks, block_size)
    return block_mask.all(dim=-1)


def pack_kept_block_indices(
    kept_block_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack a boolean kept-block mask into ascending kept-block indices.

    Args:
        kept_block_mask: [B, NB] bool

    Returns:
        kept_block_indices: [B, max_kept_blocks] int32, sorted by block index
        kept_block_counts : [B] int32
    """
    if kept_block_mask.ndim != 2:
        raise ValueError("kept_block_mask must have shape [B, NB]")

    bsz, num_blocks = kept_block_mask.shape
    kept_counts = kept_block_mask.sum(dim=-1, dtype=torch.int32)
    max_kept = int(kept_counts.max().item()) if kept_counts.numel() > 0 else 0
    if max_kept == 0:
        return (
            torch.empty(bsz, 0, device=kept_block_mask.device, dtype=torch.int32),
            kept_counts,
        )

    block_idx = (
        torch.arange(num_blocks, device=kept_block_mask.device, dtype=torch.int32)
        .view(1, num_blocks)
        .expand(bsz, -1)
    )
    sentinel = torch.full_like(block_idx, fill_value=num_blocks)
    packed = (
        torch.sort(torch.where(kept_block_mask, block_idx, sentinel), dim=-1)
        .values[:, :max_kept]
        .contiguous()
    )
    return packed, kept_counts


def pad_packed_block_indices(
    block_indices: torch.Tensor,
    block_counts: torch.Tensor,
    *,
    total_num_blocks: int,
) -> torch.Tensor:
    """
    Pad a packed ascending block-index list with the sentinel `total_num_blocks`.

    This helper is used by the decode q=1 fast path so non-visual block indices
    can be combined with selected visual indices using a fixed-width buffer.
    """
    if block_indices.ndim != 2:
        raise ValueError("block_indices must have shape [B, max_blocks].")
    if block_counts.ndim != 1:
        raise ValueError("block_counts must have shape [B].")
    if block_indices.shape[0] != block_counts.shape[0]:
        raise ValueError("block_indices / block_counts batch dimension mismatch.")

    sentinel = torch.full(
        (block_indices.shape[0],),
        fill_value=total_num_blocks,
        device=block_indices.device,
        dtype=torch.int32,
    )
    indices_i32 = block_indices.to(dtype=torch.int32)
    active = (
        torch.arange(block_indices.shape[1], device=block_indices.device).view(1, -1)
        < block_counts[:, None]
    )
    return torch.where(active, indices_i32, sentinel[:, None]).contiguous()


def decode_q1_exact_upper_bounds(
    num_visual_tokens: torch.Tensor,
    num_visual_blocks: torch.Tensor,
    non_visual_block_counts: torch.Tensor,
    *,
    fixed_budget: Optional[float],
    budget_max: float,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute exact-safe q=1 decode upper bounds.

    These bounds are only used to reduce storage / execution width. They do *not*
    change the top-k candidate set:

        - visual top-k is still performed over the full visual-block candidate set
        - ranking scores are unchanged
        - the final kept block set is unchanged

    Returns:
        visual_topk_upper_bound_per_batch: [B] int32
        kept_upper_bound_per_batch       : [B] int32
    """
    nvt = num_visual_tokens.to(dtype=torch.float32)
    nvb = num_visual_blocks.to(dtype=torch.int32)
    nvb_i64 = nvb.to(dtype=torch.int64)
    nvb_positive = nvb > 0

    if fixed_budget is not None:
        budget_ratio = torch.full_like(nvt, float(fixed_budget))
    else:
        budget_ratio = torch.full_like(nvt, float(budget_max))

    visual_upper = torch.ceil((budget_ratio * nvt) / float(block_size)).to(torch.int64)
    visual_upper = torch.where(
        nvb_positive,
        torch.minimum(torch.clamp(visual_upper, min=1), nvb_i64),
        torch.zeros_like(visual_upper),
    ).to(torch.int32)
    kept_upper = non_visual_block_counts.to(dtype=torch.int32) + visual_upper
    return visual_upper, kept_upper


def build_sparse_decode_indices_q1(
    block_scores: torch.Tensor,
    visual_block_mask: torch.Tensor,
    num_visual_tokens: torch.Tensor,
    *,
    budget_min: float,
    budget_max: float,
    fixed_budget: Optional[float],
    block_size: int,
    non_visual_block_indices: torch.Tensor,
    non_visual_block_counts: torch.Tensor,
    non_visual_block_indices_padded: Optional[torch.Tensor] = None,
    block_rank_template: Optional[torch.Tensor] = None,
    visual_topk_upper_bound: Optional[int] = None,
    kept_upper_bound: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Exact runtime-oriented q=1 builder that only materializes what the Triton decode
    fast path actually consumes.

    Emits packed kept indices directly without constructing full boolean
    selected/kept masks.

    Important implementation note:
        this function is on the decode critical path, so it must not trigger device→host
        synchronizations via `.item()`. We therefore keep the output width static:

            - top-k ranks all candidate blocks and returns the configured output width
            - `kept_block_counts` tells the sparse decode kernel how many prefix entries
              are actually valid

        This changes only the representation size, not the semantics:
            - the kept block *set* is unchanged
            - the kept blocks remain sorted in ascending block index order
            - dropped blocks are still represented only by padded sentinel slots
    """
    if block_scores.ndim != 4 or block_scores.shape[2] != 1:
        raise ValueError(
            "build_sparse_decode_indices_q1 requires block_scores with shape [B, H, 1, NB]."
        )
    if visual_block_mask.ndim != 2:
        raise ValueError("visual_block_mask must have shape [B, NB]")
    if non_visual_block_indices.ndim != 2 or non_visual_block_counts.ndim != 1:
        raise ValueError("non_visual_block_indices/counts have invalid rank.")
    if (
        non_visual_block_indices.shape[0] != block_scores.shape[0]
        or non_visual_block_counts.shape[0] != block_scores.shape[0]
    ):
        raise ValueError("non_visual_block_indices/counts batch dimension mismatch.")

    scores = block_scores.float().squeeze(2)  # [B, H, NB]
    visual_mask_f = visual_block_mask[:, None, :].float()
    per_head_visual_mass = (scores * visual_mask_f).sum(dim=-1)
    visual_mass_flat = per_head_visual_mass.amax(dim=1)

    num_visual_tokens_f = num_visual_tokens.to(
        device=block_scores.device, dtype=torch.float32
    )
    num_visual_blocks = visual_block_mask.sum(dim=-1)
    if fixed_budget is not None:
        budget_ratio = torch.full_like(visual_mass_flat, float(fixed_budget))
    else:
        budget_ratio = visual_mass_flat.clamp(min=budget_min, max=budget_max)
    k_blocks_flat = torch.ceil(
        (budget_ratio * num_visual_tokens_f) / float(block_size)
    ).to(torch.long)
    k_blocks_flat = torch.where(
        num_visual_blocks > 0,
        torch.minimum(torch.clamp(k_blocks_flat, min=1), num_visual_blocks),
        torch.zeros_like(k_blocks_flat),
    )

    num_blocks = visual_block_mask.shape[1]
    sentinel = torch.full(
        (visual_block_mask.shape[0],),
        fill_value=num_blocks,
        device=block_scores.device,
        dtype=torch.int32,
    )
    if non_visual_block_indices_padded is not None:
        non_visual_vals = _ensure_device_dtype(
            non_visual_block_indices_padded,
            device=block_scores.device,
            dtype=torch.int32,
        )
        if non_visual_vals.shape[0] != block_scores.shape[0]:
            raise ValueError(
                "non_visual_block_indices_padded batch dimension mismatch."
            )
    else:
        non_visual_vals = pad_packed_block_indices(
            _ensure_device_dtype(
                non_visual_block_indices, device=block_scores.device, dtype=torch.int32
            ),
            _ensure_device_dtype(
                non_visual_block_counts, device=block_scores.device, dtype=torch.int32
            ),
            total_num_blocks=num_blocks,
        )

    rank_scores = scores.mean(dim=1).masked_fill(
        ~visual_block_mask, float("-inf")
    )  # [B, NB]

    # The candidate set remains the full `[B, NB]` score tensor. We only reduce the
    # *output width* of top-k to an exact-safe upper bound that is guaranteed to be
    # >= the actual `k_blocks_flat` for every batch row.
    topk_width = (
        num_blocks if visual_topk_upper_bound is None else int(visual_topk_upper_bound)
    )
    topk_width = max(0, min(topk_width, num_blocks))
    top_idx = (
        torch.topk(rank_scores, k=topk_width, dim=-1).indices.to(torch.int32)
        if topk_width > 0
        else torch.empty(
            rank_scores.shape[0], 0, device=block_scores.device, dtype=torch.int32
        )
    )

    if block_rank_template is not None:
        rank_template = _ensure_device_dtype(
            block_rank_template, device=block_scores.device, dtype=torch.int32
        )
        if (
            rank_template.ndim != 2
            or rank_template.shape[0] != 1
            or rank_template.shape[1] < topk_width
        ):
            raise ValueError(
                "block_rank_template must have shape [1, width] with width >= topk_width."
            )
        rank_template = rank_template[:, :topk_width]
    else:
        rank_template = torch.arange(
            topk_width, device=block_scores.device, dtype=torch.int32
        ).view(1, -1)
    selected_active = rank_template < k_blocks_flat[:, None]
    selected_vals = torch.where(selected_active, top_idx, sentinel[:, None])
    combined = torch.cat([non_visual_vals, selected_vals], dim=-1)

    kept_counts = _ensure_device_dtype(
        non_visual_block_counts, device=block_scores.device, dtype=torch.int32
    ) + k_blocks_flat.to(torch.int32)
    # The exact number of valid kept entries is carried by `kept_counts`. Returning a
    # fixed-width buffer with a strict safe upper bound removes host sync without
    # changing the final kept set.
    kept_width = num_blocks if kept_upper_bound is None else int(kept_upper_bound)
    kept_width = max(0, min(kept_width, combined.shape[1]))
    kept_indices = torch.sort(combined, dim=-1).values[:, :kept_width].contiguous()
    visual_mass = visual_mass_flat.view(-1, 1, 1, 1).to(block_scores.dtype)
    k_blocks = k_blocks_flat.view(-1, 1, 1, 1)
    return visual_mass, k_blocks, kept_indices, kept_counts
