# Evaluation

Run commands from the repository root after completing
[Quick Start](../../README.md#-quick-start). The public registrations contain
VisionPulse variants; the comparison methods in the paper's table are not
included in this release.

## Datasets and scoring

The seven main-table benchmarks map to these bundled VLMEvalKit identifiers:

| Benchmark | `--data` identifier |
|---|---|
| CharXiv RQ | `CharXiv_reasoning_val` |
| InfoVQA | `InfoVQA_VAL` |
| ChartQA | `ChartQA_TEST` |
| MMStar | `MMStar` |
| RealWorldQA | `RealWorldQA` |
| MMVet | `MMVet` |
| MIA-Bench | `MIA-Bench` |

VLMEvalKit loads/downloads dataset files from the URLs registered in the bundled
dataset classes. The default cache is `~/LMUData`. To use another cache directory,
create it before setting `LMUData`:

```bash
mkdir -p /path/to/LMUData
export LMUData=/path/to/LMUData
```

Ensure dataset files and referenced images are accessible before a full run.

Evaluation follows the default scoring settings of the bundled VLMEvalKit.
Configure `OPENAI_API_KEY` for benchmarks that require an API judge:

```bash
export OPENAI_API_KEY=your_api_key
```

For a compatible custom endpoint, `OPENAI_API_BASE` accepts the full chat
completions URL, including `/v1/chat/completions`. The endpoint must support the
judge models required by the evaluators.

## Running and resuming

Use `run.py` with a model alias and dataset names, as shown in
[Quick Start](../../README.md#-quick-start). By default, predictions are saved as
TSV files with Thinking traces separated from final answers.

Run inference first, then score the saved predictions once the judge is configured:

```bash
python run.py --model Qwen3-VL-4B-Thinking-VisionPulse-10pct \
  --data MMVet --mode infer --work-dir outputs/mmvet_10pct
python run.py --model Qwen3-VL-4B-Thinking-VisionPulse-10pct \
  --data MMVet --mode eval --reuse --work-dir outputs/mmvet_10pct
```

For a resumed run, keep the model, datasets, and `--work-dir` unchanged and add
`--reuse`. The runner defaults to `--mode all` (inference followed by scoring).
For additional runner options, use `python run.py --help`.

## Reading results

Results are organized as:

```text
outputs/main_table/<5pct-or-10pct>/<model-alias>/T<date>_G<commit>/
```

- `<model-alias>_<dataset>.tsv`: predictions, with separate `thinking` and
  `prediction` columns when `SPLIT_THINK=1`.
- Files ending in `_acc.*` or `_score.*`: dataset-specific scoring outputs;
  judge names may appear in their filenames.

Inspect the benchmark's overall score for accuracy and the `thinking` column for
reasoning traces. Inference-only runs do not produce benchmark scores; check
scoring completion and failed samples before interpreting results.

For latency measurements with the Triton block-sparse backend, see the
[efficiency guide](../../analysis/efficiency/README.md).
