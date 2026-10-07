# Efficiency analysis

VisionPulse uses **block-level sparse attention with Triton** to reduce decoding
cost. This guide provides a speed benchmark and a Transformers integration example.

## Paper results

![Dense-versus-sparse latency comparison from the paper.](../../assets/latency_comparison.png)

The paper reports **1.20×–1.30×** speedups at 8k, 16k, and 32k visual context
lengths with batch size 8. The figure shows the paper's results; the benchmark
below measures the current implementation with dynamic **1%–5% visual budgets**.
Actual latency and speedup depend on hardware, software, inputs, and budget settings.

## Environment

The paper's inference experiments were conducted on **8 × NVIDIA A800 80GB GPUs**.
The benchmark below runs each dense/sparse comparison on **one GPU**, with the
entire model on that device. Batch size 8 refers to eight sequences on that GPU.

| Setting | Benchmark configuration |
|---|---|
| GPU | 1 × NVIDIA A800 80GB PCIe per run |
| Model | Qwen3-VL-4B-Thinking |
| Precision | BF16 |
| Validated software | PyTorch 2.4.0+cu121, Transformers 4.57.6, Triton 3.0.0 |

Follow the [analysis setup](../README.md#get-started). Use the Triton version
provided with your PyTorch installation. `CUDA_VISIBLE_DEVICES` selects the GPU.

## Run the benchmark

```bash
CUDA_VISIBLE_DEVICES=0 python -m analysis.efficiency.benchmark \
  --model Qwen/Qwen3-VL-4B-Thinking \
  --visual-lengths 8192 16384 32768 \
  --batch 8 --text-tokens 256 --decode-steps 1024 \
  --budget-min 0.01 --budget-max 0.05 --prefill-chunk-size 1024 \
  --warmup 1 --iters 2 \
  --output-dir outputs/analysis/efficiency
```

Use `--model /path/to/model` for local weights. To check the installation with
a smaller run, use `--visual-lengths 256 --batch 1 --decode-steps 2 --iters 1`.

The command saves **`outputs/analysis/efficiency/latency_comparison.png`** and
prints dense/sparse latency, per-run measurements, and actual visual-retention
statistics in the terminal.

### Reading the results

The benchmark measures **language-model prefill + decode latency**, including
KV-cache updates and excluding vision encoding. Synthetic inputs provide exact
visual lengths of 8192, 16384, and 32768 tokens, each with 256 text tokens and
1024 decode steps.

Dense and sparse runs use the same settings, a full warmup, and synchronized GPU
timing. Retention statistics are collected during the final untimed warmup.
Synthetic inputs can keep the dynamic budget at its upper bound. Because selection
uses **64-token blocks**, rounding and retained mixed boundary blocks can make
actual visual retention exceed the nominal budget.

## Transformers integration

To use the backend in your own inference code:

```python
import torch
from visionpulse.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from visionpulse.triton import enable_triton

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Thinking",
    dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="eager",
).eval()
model = enable_triton(model, budget=None, budget_min=0.01, budget_max=0.05)
# Continue with the processor and model.generate(...) workflow in Quick Start.
```

The outer `eager` setting routes language attention to the custom Triton kernels.
Prefill and the anchor output (layer 17) remain dense; later decode layers reuse
the blocks selected at the anchor. Non-visual and mixed boundary blocks are
always retained. Use `enable_triton(model, sparse=False)` for the dense Triton baseline.
