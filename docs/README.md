# Documentation

| Guide | Contents |
|---|---|
| [Models](models/README.md) | Supported models, weights, and VisionPulse parameters |
| [Evaluation](evaluation/README.md) | Datasets, judge setup, evaluation, and saved results |
| [Analysis](../analysis/README.md) | Visual activations, attention mass correlation, layer similarity, and efficiency |

Start with [Quick Start](../README.md#-quick-start) for installation and the
main-table evaluation commands.

## Code entry points

Evaluation runs through `run.py`, the model registrations in `vlmeval/config.py`,
and the Qwen3-VL or InternVL adapter. The adapters load the VisionPulse model
implementations and return predictions to VLMEvalKit for scoring.

| Component | Source |
|---|---|
| Model registrations and parameters | [`vlmeval/config.py`](../vlmeval/config.py) |
| Qwen3-VL adapter | [`vlmeval/vlm/qwen3_vl/model.py`](../vlmeval/vlm/qwen3_vl/model.py) |
| Qwen3-VL attention | [`visionpulse/modeling_qwen3_vl.py`](../visionpulse/modeling_qwen3_vl.py) |
| InternVL adapter | [`vlmeval/vlm/internvl/internvl_chat.py`](../vlmeval/vlm/internvl/internvl_chat.py) |
| InternVL multimodal model | [`visionpulse/modeling_internvl_chat.py`](../visionpulse/modeling_internvl_chat.py) |
| InternVL language attention | [`visionpulse/modeling_qwen3.py`](../visionpulse/modeling_qwen3.py) |
| Triton block-sparse backend | [`visionpulse/triton/`](../visionpulse/triton/) |

See [third-party notices](THIRD_PARTY_NOTICES.md) for upstream code attribution.
