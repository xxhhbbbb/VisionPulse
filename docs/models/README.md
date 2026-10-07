# Model configurations

Follow the [installation instructions](../../README.md#-quick-start) first.
Run all commands below from the repository root.

## Supported models

All entries below run in **Thinking mode**. The included InternVL versions are **4B and 8B**.

| Model alias (`--model`) | Visual retention range | τ | Anchor layer |
|---|---:|---:|---:|
| `Qwen3-VL-4B-Thinking-VisionPulse-5pct` | 1%–5% | 0.1 | 17 |
| `Qwen3-VL-4B-Thinking-VisionPulse-10pct` | 5%–10% | 0.4 | 17 |
| `Qwen3-VL-8B-Thinking-VisionPulse-5pct` | 1%–5% | 0.1 | 17 |
| `Qwen3-VL-8B-Thinking-VisionPulse-10pct` | 5%–10% | 0.4 | 17 |
| `InternVL3_5-4B-Thinking-VisionPulse-5pct` | 1%–5% | 0.1 | 17 |
| `InternVL3_5-8B-Thinking-VisionPulse-5pct` | 1%–5% | 0.1 | 17 |

Layer indices are zero-based. Retention ranges specify the configured dynamic
budgets, rather than a fixed number of tokens at every step. InternVL uses the
original `OpenGVLab/InternVL3_5-{4B,8B}` weights with the bundled Thinking prompt
configuration (`cot_prompt_version="r1"`). Use one sample per model invocation;
the included InternVL attention implementation explicitly requires batch size 1.

## Model weights and attention

VisionPulse uses the original model weights; no separate checkpoint or training
is required. Models are downloaded from Hugging Face by default. To use local
weights, set `model_path` in [`vlmeval/config.py`](../../vlmeval/config.py) to
the corresponding model directory, as in Quick Start.

The Qwen3-VL evaluation configurations use eager attention by default.
VisionPulse requires eager attention only in the language model; the vision
encoder backend can be configured independently. For the optional Triton
block-sparse backend and speed benchmark, see the
[efficiency guide](../../analysis/efficiency/README.md).

## Additional model examples

```bash
python run.py --model Qwen3-VL-8B-Thinking-VisionPulse-10pct --data RealWorldQA
python run.py --model Qwen3-VL-8B-Thinking-VisionPulse-5pct --data RealWorldQA
python run.py --model InternVL3_5-4B-Thinking-VisionPulse-5pct --data RealWorldQA
python run.py --model InternVL3_5-8B-Thinking-VisionPulse-5pct --data RealWorldQA
```

These examples use `RealWorldQA`. Change the dataset list with `--data`,
as shown in the [evaluation guide](../evaluation/README.md).

## VisionPulse parameters

Configure both Qwen3-VL and InternVL in
[`vlmeval/config.py`](../../vlmeval/config.py):

| Parameter | Meaning |
|---|---|
| `visionpulse_anchor_layer` | Zero-based layer at which visual tokens are selected |
| `visionpulse_visual_mass_tau` | Temperature used to calculate visual attention mass |
| `visionpulse_budget_min` | Lower bound on the visual retention ratio |
| `visionpulse_budget_max` | Upper bound on the visual retention ratio |

At each decoding step, the anchor layer selects visual tokens using the dynamic
budget; subsequent layers reuse that selection for the same step. Prefill remains
dense. The token count is rounded down, with at least one visual token retained
for a nonempty visual input.

See the [code entry points](../README.md#code-entry-points) to locate each model's
implementation.
