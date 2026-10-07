# SPDX-License-Identifier: Apache-2.0
"""Layer routing and block selection for Triton visual sparse attention."""
from typing import Any, Optional

import torch
from torch import nn

from . import budget as vb
from .kernels.triton_block_sparse_attn import (
    triton_block_sparse_attention_decode_q1_forward,
)
from .kernels.triton_flash_dense import triton_dense_attention_forward
from .kernels.triton_vision_block_scores import (
    triton_anchor_dense_attention_with_scores,
)

_SUPPORTED_TRITON_BLOCK_SIZES = {16, 32, 64, 128}


def _ensure_tensor_on_device(
    tensor: torch.Tensor, *, device: torch.device, dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    target_dtype = tensor.dtype if dtype is None else dtype
    if tensor.device == device and tensor.dtype == target_dtype:
        return tensor
    return tensor.to(device=device, dtype=target_dtype)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _prepare_visual_metadata(
    sparse_state: dict[str, Any],
    *,
    batch_size: int,
    seq_len_k: int,
    device: torch.device,
    block_size: int,
    need_token_mask: bool = True,
) -> tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Prepare visual masks and counts in the shared state of this forward.

    Only fully visual blocks are prunable. Mixed boundary blocks remain visible.
    The model creates a fresh state for each forward; selections are shared only
    between layers of the same decoding step.
    """
    cached_token_mask = sparse_state.get("visual_token_mask_cached")
    cached_block_mask = sparse_state.get("visual_block_mask")
    cached_num_visual_tokens = sparse_state.get("num_visual_tokens")
    expected_num_blocks = (seq_len_k + block_size - 1) // block_size

    if (
        not need_token_mask
        and cached_block_mask is not None
        and cached_num_visual_tokens is not None
        and cached_block_mask.shape[0] == batch_size
    ):
        visual_block_mask = _ensure_tensor_on_device(
            cached_block_mask, device=device, dtype=torch.bool
        )
        if visual_block_mask.shape[-1] < expected_num_blocks:
            pad = torch.zeros(
                batch_size,
                expected_num_blocks - visual_block_mask.shape[-1],
                device=device,
                dtype=torch.bool,
            )
            visual_block_mask = torch.cat([visual_block_mask, pad], dim=-1)
        elif visual_block_mask.shape[-1] > expected_num_blocks:
            visual_block_mask = visual_block_mask[:, :expected_num_blocks]
        sparse_state["visual_block_mask"] = visual_block_mask
        return (
            None,
            visual_block_mask,
            _ensure_tensor_on_device(cached_num_visual_tokens, device=device),
        )

    if (
        cached_token_mask is not None
        and cached_block_mask is not None
        and cached_num_visual_tokens is not None
        and cached_token_mask.shape[0] == batch_size
    ):
        visual_token_mask = vb.normalize_visual_token_mask(
            cached_token_mask, batch_size=batch_size, seq_len_k=seq_len_k, device=device
        )
        visual_block_mask = _ensure_tensor_on_device(
            cached_block_mask, device=device, dtype=torch.bool
        )
        if visual_block_mask.shape[-1] == expected_num_blocks:
            return (
                visual_token_mask,
                visual_block_mask,
                _ensure_tensor_on_device(cached_num_visual_tokens, device=device),
            )

    visual_token_mask = sparse_state.get("visual_token_mask")
    if visual_token_mask is None:
        raise ValueError("Sparse vision state must contain visual_token_mask.")
    visual_token_mask = vb.normalize_visual_token_mask(
        visual_token_mask, batch_size=batch_size, seq_len_k=seq_len_k, device=device
    )
    # `visual_block_mask` here means fully-visual / prunable blocks only; mixed boundary blocks stay in the always-kept prefix.
    visual_block_mask = vb.visual_block_mask_from_token_mask(
        visual_token_mask, block_size
    )
    num_visual_tokens = sparse_state.get("num_visual_tokens")
    if num_visual_tokens is None:
        num_visual_tokens = visual_token_mask.sum(dim=-1)
    else:
        num_visual_tokens = _ensure_tensor_on_device(num_visual_tokens, device=device)
    sparse_state["visual_token_mask_cached"] = visual_token_mask
    sparse_state["visual_block_mask"] = visual_block_mask
    sparse_state["num_visual_tokens"] = num_visual_tokens
    return visual_token_mask, visual_block_mask, num_visual_tokens


def _triton_supported(
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    block_size: int,
) -> bool:
    if not torch.cuda.is_available():
        return False
    if (not query.is_cuda) or (not key.is_cuda):
        return False
    if attention_mask is None or (not attention_mask.is_cuda):
        return False
    if query.dtype not in (torch.float16, torch.bfloat16):
        return False
    if query.shape[-1] not in (16, 32, 64, 128):
        return False
    if block_size not in _SUPPORTED_TRITON_BLOCK_SIZES:
        return False
    return True


def _triton_sparse_vision_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float,
    *,
    sparse_state: dict[str, Any],
    **kwargs: Any,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Triton implementation of the full sparse-vision inference pipeline.

    Layer routing:
        - layers 0..16 dense baseline
        - layer 17 dense anchor + block score export
        - layers 18..end block-sparse
    where `anchor_layer = 17` is interpreted as a 0-based layer index.
    """
    if module.training:
        raise RuntimeError(
            "The Triton sparse-vision path is inference-only and must not be used in training mode."
        )
    if dropout != 0.0:
        raise RuntimeError(
            "The Triton sparse-vision path expects dropout=0 during inference."
        )

    block_size = int(getattr(module.config, "sparse_vision_block_size", 64))
    if not _triton_supported(query, key, attention_mask, block_size):
        raise RuntimeError(
            "Current inputs are not supported by the Triton sparse-vision path."
        )

    anchor_layer = int(getattr(module.config, "sparse_vision_anchor_layer", 17))
    score_temperature = float(
        getattr(module.config, "sparse_vision_score_temperature", 0.4)
    )
    budget_min = float(getattr(module.config, "sparse_vision_budget_min", 0.05))
    budget_max = float(getattr(module.config, "sparse_vision_budget_max", 0.10))
    fixed_budget = getattr(module.config, "sparse_vision_budget", None)

    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)

    batch_size = query.shape[0]
    q_len = query.shape[2]
    seq_len_k = key_states.shape[2]

    if q_len > 1:
        for ephemeral_key in (
            "kept_block_indices",
            "kept_block_counts",
            "visual_mass",
        ):
            sparse_state.pop(ephemeral_key, None)
        attn_output = triton_dense_attention_forward(
            query,
            key_states,
            value_states,
            attention_mask,
            scaling,
            block_n=block_size,
        )
        return attn_output, None

    if module.layer_idx < anchor_layer:
        attn_output = triton_dense_attention_forward(
            query, key_states, value_states, attention_mask, scaling, block_n=block_size
        )
        return attn_output, None

    if module.layer_idx == anchor_layer:
        visual_token_mask, visual_block_mask, num_visual_tokens = (
            _prepare_visual_metadata(
                sparse_state,
                batch_size=batch_size,
                seq_len_k=seq_len_k,
                device=query.device,
                block_size=block_size,
                need_token_mask=False,
            )
        )
        if visual_token_mask is not None:
            sparse_state["visual_token_mask"] = visual_token_mask
        sparse_state["visual_block_mask"] = visual_block_mask
        sparse_state["num_visual_tokens"] = num_visual_tokens

        non_visual_block_indices = sparse_state.get("non_visual_block_indices")
        non_visual_block_counts = sparse_state.get("non_visual_block_counts")
        non_visual_block_indices_padded = sparse_state.get(
            "non_visual_block_indices_padded"
        )
        block_rank_template = sparse_state.get("block_rank_template")
        bounds_num_key_blocks = sparse_state.get("decode_q1_bounds_num_key_blocks")
        expected_non_visual_blocks = (~visual_block_mask).sum(dim=-1)
        expected_visual_blocks = visual_block_mask.sum(dim=-1)
        if (
            non_visual_block_indices is None
            or non_visual_block_counts is None
            or non_visual_block_indices.shape[0] != batch_size
            or not torch.equal(
                _ensure_tensor_on_device(
                    non_visual_block_counts, device=query.device, dtype=torch.int32
                ),
                expected_non_visual_blocks.to(torch.int32),
            )
        ):
            non_visual_block_indices, non_visual_block_counts = (
                vb.pack_kept_block_indices(~visual_block_mask)
            )
            sparse_state["non_visual_block_indices"] = non_visual_block_indices
            sparse_state["non_visual_block_counts"] = non_visual_block_counts
            non_visual_block_indices_padded = None
            block_rank_template = None

        num_key_blocks = visual_block_mask.shape[-1]
        if (
            non_visual_block_indices_padded is None
            or non_visual_block_indices_padded.shape[0] != batch_size
            or non_visual_block_indices_padded.shape[1]
            != non_visual_block_indices.shape[1]
        ):
            non_visual_block_indices_padded = vb.pad_packed_block_indices(
                non_visual_block_indices,
                _ensure_tensor_on_device(
                    non_visual_block_counts, device=query.device, dtype=torch.int32
                ),
                total_num_blocks=num_key_blocks,
            )
            sparse_state["non_visual_block_indices_padded"] = (
                non_visual_block_indices_padded
            )
        if block_rank_template is None or block_rank_template.shape != (
            1,
            num_key_blocks,
        ):
            block_rank_template = torch.arange(
                num_key_blocks, device=query.device, dtype=torch.int32
            ).view(1, -1)
            sparse_state["block_rank_template"] = block_rank_template
        visual_topk_upper_bound = sparse_state.get("decode_q1_visual_topk_upper_bound")
        kept_upper_bound = sparse_state.get("decode_q1_kept_upper_bound")
        if (
            visual_topk_upper_bound is None
            or kept_upper_bound is None
            or bounds_num_key_blocks is None
            or int(bounds_num_key_blocks) != num_key_blocks
        ):
            visual_upper_per_batch, kept_upper_per_batch = (
                vb.decode_q1_exact_upper_bounds(
                    num_visual_tokens,
                    expected_visual_blocks,
                    non_visual_block_counts.to(device=query.device, dtype=torch.int32),
                    fixed_budget=fixed_budget,
                    budget_max=budget_max,
                    block_size=block_size,
                )
            )
            visual_topk_upper_bound = (
                int(visual_upper_per_batch.max().item())
                if visual_upper_per_batch.numel() > 0
                else 0
            )
            kept_upper_bound = (
                int(kept_upper_per_batch.max().item())
                if kept_upper_per_batch.numel() > 0
                else 0
            )
            sparse_state["decode_q1_visual_topk_upper_bound"] = visual_topk_upper_bound
            sparse_state["decode_q1_kept_upper_bound"] = kept_upper_bound
            sparse_state["decode_q1_bounds_num_key_blocks"] = num_key_blocks

        attn_output, block_scores = triton_anchor_dense_attention_with_scores(
            query,
            key_states,
            value_states,
            attention_mask,
            scaling,
            score_temperature,
            block_size,
        )
        visual_mass, k_blocks, kept_block_indices, kept_block_counts = (
            vb.build_sparse_decode_indices_q1(
                block_scores,
                visual_block_mask,
                num_visual_tokens,
                budget_min=budget_min,
                budget_max=budget_max,
                fixed_budget=fixed_budget,
                block_size=block_size,
                non_visual_block_indices=non_visual_block_indices,
                non_visual_block_counts=non_visual_block_counts,
                non_visual_block_indices_padded=non_visual_block_indices_padded,
                block_rank_template=block_rank_template,
                visual_topk_upper_bound=visual_topk_upper_bound,
                kept_upper_bound=kept_upper_bound,
            )
        )
        sparse_state["kept_block_indices"] = kept_block_indices
        sparse_state["kept_block_counts"] = kept_block_counts

        sparse_state["visual_mass"] = visual_mass
        return attn_output, None

    kept_block_indices = sparse_state.get("kept_block_indices")
    kept_block_counts = sparse_state.get("kept_block_counts")

    num_blocks = (seq_len_k + block_size - 1) // block_size
    if kept_block_indices is not None and kept_block_counts is not None and q_len == 1:
        decode_indices = _ensure_tensor_on_device(
            kept_block_indices, device=query.device, dtype=torch.int32
        )
        decode_counts = _ensure_tensor_on_device(
            kept_block_counts, device=query.device, dtype=torch.int32
        )
        if torch.equal(decode_counts, torch.full_like(decode_counts, num_blocks)):
            attn_output = triton_dense_attention_forward(
                query,
                key_states,
                value_states,
                attention_mask,
                scaling,
                block_n=block_size,
            )
            return attn_output, None

        attn_output = triton_block_sparse_attention_decode_q1_forward(
            query,
            key_states,
            value_states,
            attention_mask,
            decode_indices,
            decode_counts,
            scaling,
            block_size,
        )
        return attn_output, None

    raise RuntimeError(
        "Post-anchor sparse attention requires packed block indices from the anchor layer."
    )
