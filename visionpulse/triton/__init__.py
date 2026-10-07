# SPDX-License-Identifier: Apache-2.0
"""Opt-in FlashAttention-style block-sparse Triton inference."""


def enable_triton(
    model,
    *,
    sparse=True,
    anchor_layer=17,
    block_size=64,
    score_temperature=0.4,
    budget_min=0.01,
    budget_max=0.05,
    budget=None
):
    """Enable the original block-level backend on the bundled Qwen3-VL model.

    Dense prefill and a dense anchor output are preserved. Subsequent decode
    layers attend to selected visual blocks and all non-visual/mixed blocks.
    Set sparse=False to use the same dense Triton baseline at every layer.
    """
    from ..modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    if not isinstance(model, Qwen3VLForConditionalGeneration):
        raise TypeError(
            "Load Qwen3VLForConditionalGeneration from visionpulse.modeling_qwen3_vl."
        )
    if block_size not in (16, 32, 64, 128):
        raise ValueError("block_size must be 16, 32, 64, or 128")
    if not 0 <= budget_min <= budget_max <= 1 or score_temperature <= 0:
        raise ValueError("Invalid budget bounds or score temperature")
    if budget is not None and not 0 <= budget <= 1:
        raise ValueError("budget must be in [0, 1]")
    config = model.config.text_config
    if not 0 <= anchor_layer < config.num_hidden_layers:
        raise ValueError("anchor_layer is outside the decoder")
    config._attn_implementation = "eager"
    config.visionpulse_triton = True
    config.sparse_vision_anchor_layer = (
        anchor_layer if sparse else config.num_hidden_layers
    )
    config.sparse_vision_block_size = block_size
    config.sparse_vision_score_temperature = score_temperature
    config.sparse_vision_budget_min = budget_min
    config.sparse_vision_budget_max = budget_max
    config.sparse_vision_budget = budget
    return model


def attention_forward(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
):
    from .attention import _triton_sparse_vision_attention_forward
    from .kernels.triton_flash_dense import triton_dense_attention_forward
    from .attention import _repeat_kv

    if module.training or dropout:
        raise RuntimeError("The Triton backend requires eval mode and zero dropout")
    # The existing decoder supplies a fresh shared namespace for each forward.
    step = kwargs["visionpulse_step_state"]
    visual_mask = kwargs.get("visual_mask")
    if not kwargs.get("is_pruned", False) or visual_mask is None:
        return (
            triton_dense_attention_forward(
                query,
                _repeat_kv(key, module.num_key_value_groups),
                _repeat_kv(value, module.num_key_value_groups),
                attention_mask,
                scaling,
                block_n=module.config.sparse_vision_block_size,
            ),
            None,
        )
    if not hasattr(step, "block_state"):
        step.block_state = {
            "visual_token_mask": visual_mask,
            "num_visual_tokens": visual_mask.sum(dim=-1),
        }
    return _triton_sparse_vision_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling,
        dropout,
        sparse_state=step.block_state,
    )
