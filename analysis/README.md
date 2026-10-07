# Analysis tools

Explore visual attention during reasoning and benchmark VisionPulse's inference
speed. Each tool provides a runnable example and saves one PNG.

| Tool | What it shows |
|---|---|
| [Visual activation](activation/README.md) | How attention to image regions changes during reasoning. |
| [Attention mass correlation](attention_mass/README.md) | The relationship between visual attention mass and active visual token count. |
| [Layer-wise similarity](layer_similarity/README.md) | How similar visual attention distributions are across language-model layers. |
| [Efficiency](efficiency/README.md) | Dense versus block-sparse inference latency at different context lengths. |

## Get started

Complete the [project installation](../README.md#1-installation), then install
the plotting dependencies. Run all commands from the repository root on a CUDA GPU.

```bash
python -m pip install -r analysis/requirements.txt

python -m analysis.run activation --steps 0 11 22
python -m analysis.run attention_mass
python -m analysis.run layer_similarity
```

These examples use **Qwen3-VL-4B-Thinking** and the provided
[vent image](../assets/vent_example.png). Results are saved under
`outputs/analysis/<tool>/`. For speed measurements, follow the
[efficiency guide](efficiency/README.md).

## Use your own input

```bash
python -m analysis.run activation \
  --model /path/to/Qwen3-VL-4B-Thinking \
  --image /path/to/image.jpg \
  --question "Describe the image." \
  --output-dir outputs/analysis/my_image
```

Use `attention_mass` or `layer_similarity` in place of `activation` for the
other analyses. Activation and attention-mass analysis use layer 17 by default;
change it with `--layer`. Layer similarity compares all layers.
If generation reaches the default 2048-token limit, increase `--max-new-tokens`
to let the response finish before plotting.

## Analyze multiple samples

Collect attention from a dataset, then pass the saved traces to an analysis tool:

```bash
python -m analysis.collect \
  --model Qwen/Qwen3-VL-4B-Thinking \
  --dataset RealWorldQA --sample-size 50 --seed 42 --all-layers \
  --max-new-tokens 2048 --output-dir outputs/analysis/realworldqa

python -m analysis.attention_mass.correlate \
  --data outputs/analysis/realworldqa
python -m analysis.layer_similarity.similarity \
  --data outputs/analysis/realworldqa
```

Dataset preparation follows [VLMEvalKit's evaluation setup](../docs/evaluation/README.md).
Each sample must contain one image. Keep `--all-layers` for layer similarity;
omit it when collecting only activation or attention-mass data.
Use a new output directory for each collection.

<details>
<summary>Custom sample lists and reusing collected data</summary>

Use `--samples samples.json` instead of `--dataset` for your own image–question
pairs. Image paths are relative to the JSON file:

```json
[{"id": "vent", "image": "assets/vent_example.png", "question": "Where is the brown square vent relative to the door?"}]
```

To save attention for a single image:

```bash
python -m analysis.collect \
  --model Qwen/Qwen3-VL-4B-Thinking \
  --image assets/vent_example.png --all-layers --max-new-tokens 2048 \
  --output-dir outputs/analysis/traces

python -m analysis.activation.visualize \
  --data outputs/analysis/traces/sample_0000/trace.npz --steps 0 11 22
```

Each sample directory contains `trace.npz`, the image, generated response, and
run metadata. `summary.json` records successful and failed samples.
Saved traces can be plotted on CPU without running the model again.

</details>
