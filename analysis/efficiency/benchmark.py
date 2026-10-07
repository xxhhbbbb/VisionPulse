"""Measure Transformers language-model prefill + decode with Triton attention."""

import argparse
import json
import time
from pathlib import Path

import torch

from analysis.common import pyplot
from visionpulse.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from visionpulse.triton import enable_triton


@torch.inference_mode()
def measure_latency(model, args, visual_tokens: int, decode_steps: int) -> dict:
    """Time one prefill and fixed-length decode with a fresh KV cache."""
    device = next(model.parameters()).device
    context_length = visual_tokens + args.text_tokens
    past_key_values = None
    visual_mask = torch.zeros(
        args.batch, context_length + decode_steps, dtype=torch.bool, device=device
    )
    visual_mask[:, :visual_tokens] = True

    def forward_tokens(token_start, token_end):
        nonlocal past_key_values
        input_ids = torch.full(
            (args.batch, token_end - token_start), 42, dtype=torch.long, device=device
        )
        outputs = model.model.language_model(
            input_ids=input_ids,
            attention_mask=torch.ones(
                args.batch, token_end, dtype=torch.long, device=device
            ),
            cache_position=torch.arange(token_start, token_end, device=device),
            visual_pos_masks=(
                visual_mask[:, :token_end] if token_end - token_start == 1 else None
            ),
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        logits = model.lm_head(outputs.last_hidden_state[:, -1:, :])
        return logits

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    chunk_size = args.prefill_chunk_size or context_length
    for chunk_start in range(0, context_length, chunk_size):
        forward_tokens(chunk_start, min(chunk_start + chunk_size, context_length))
    torch.cuda.synchronize()
    prefill_end = time.perf_counter()
    for step in range(decode_steps):
        logits = forward_tokens(context_length + step, context_length + step + 1)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    if not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite model logits")
    return {
        "prefill_s": prefill_end - start_time,
        "decode_s": end_time - prefill_end,
        "total_s": end_time - start_time,
    }


class RetentionCollector:
    """Collect actual selected visual tokens during the final untimed warmup."""

    def __init__(self, model):
        self.config = model.config.text_config
        self.records = []
        anchor_attention = model.model.language_model.layers[
            self.config.sparse_vision_anchor_layer
        ].self_attn
        self.handle = anchor_attention.register_forward_hook(
            self.record, with_kwargs=True
        )

    def record(self, _module, _inputs, kwargs, _output):
        if not kwargs.get("is_pruned", False):
            return
        state = kwargs["visionpulse_step_state"].block_state
        visual_mask = kwargs["visual_mask"]
        block_size = self.config.sparse_vision_block_size
        padding = (-visual_mask.shape[-1]) % block_size
        visual_tokens_per_block = (
            torch.nn.functional.pad(visual_mask, (0, padding))
            .reshape(visual_mask.shape[0], -1, block_size)
            .sum(-1)
        )
        kept_indices = state["kept_block_indices"].long()
        valid_slots = (
            torch.arange(kept_indices.shape[-1], device=kept_indices.device)[None, :]
            < state["kept_block_counts"][:, None]
        )
        selected_visual_counts = visual_tokens_per_block.gather(
            1, kept_indices.clamp(0, visual_tokens_per_block.shape[-1] - 1)
        )
        actual_retention = (selected_visual_counts * valid_slots).sum(
            -1
        ) / visual_mask.sum(-1)
        visual_mass = state["visual_mass"].reshape(-1).float()
        fixed_budget = self.config.sparse_vision_budget
        budget = visual_mass.clamp(
            self.config.sparse_vision_budget_min, self.config.sparse_vision_budget_max
        )
        if fixed_budget is not None:
            budget = torch.full_like(visual_mass, fixed_budget)
        self.records.append(
            torch.stack((visual_mass, budget, actual_retention), dim=-1).detach()
        )

    def summary(self) -> dict:
        if not self.records:
            raise RuntimeError("No anchor selections were recorded")
        statistics = torch.cat(self.records).float().cpu()
        result = {"source": "final_untimed_warmup", "sample_steps": len(statistics)}
        for metric_index, name in enumerate(
            ("visual_mass", "budget_ratio", "actual_visual_retention")
        ):
            values = statistics[:, metric_index]
            result[name] = {
                "mean": values.mean().item(),
                "min": values.min().item(),
                "max": values.max().item(),
            }
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Thinking")
    parser.add_argument(
        "--visual-lengths", nargs="+", type=int, default=[8192, 16384, 32768]
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--text-tokens", type=int, default=256)
    parser.add_argument("--decode-steps", type=int, default=1024)
    parser.add_argument(
        "--retention",
        type=float,
        default=None,
        help="Optional fixed budget; default: dynamic",
    )
    parser.add_argument("--budget-min", type=float, default=0.01)
    parser.add_argument("--budget-max", type=float, default=0.05)
    parser.add_argument("--prefill-chunk-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/analysis/efficiency")
    )
    args = parser.parse_args()
    if (
        min(
            args.batch,
            args.text_tokens,
            args.decode_steps,
            args.iters,
            *args.visual_lengths,
        )
        < 1
    ):
        parser.error("Batch, lengths, decode steps and iterations must be positive")
    if args.warmup < 1 or args.prefill_chunk_size < 0:
        parser.error("Use at least one warmup and a non-negative chunk size")
    if not 0 <= args.budget_min <= args.budget_max <= 1:
        parser.error("Budget bounds must satisfy 0 <= min <= max <= 1")
    if args.retention is not None and not 0 <= args.retention <= 1:
        parser.error("Fixed retention must be in [0, 1]")
    return args


def benchmark_mode(model, args, visual_tokens: int, mode: str) -> dict:
    enable_triton(
        model,
        sparse=mode == "sparse",
        budget=args.retention,
        budget_min=args.budget_min,
        budget_max=args.budget_max,
    )
    print(f"{visual_tokens} visual tokens / {mode}: warmup", flush=True)
    retention = None
    for warmup_index in range(args.warmup):
        collector = (
            RetentionCollector(model)
            if mode == "sparse" and warmup_index == args.warmup - 1
            else None
        )
        try:
            measure_latency(model, args, visual_tokens, args.decode_steps)
        finally:
            if collector is not None:
                collector.handle.remove()
        if collector is not None:
            retention = collector.summary()
    # No hooks or retention-statistics operations remain in the timed runs.
    print(
        f"{visual_tokens} visual tokens / {mode}: measuring {args.iters} repetitions",
        flush=True,
    )
    samples = [
        measure_latency(model, args, visual_tokens, args.decode_steps)
        for _ in range(args.iters)
    ]
    result = {
        metric: sum(sample[metric] for sample in samples) / len(samples)
        for metric in samples[0]
    }
    result["samples"] = samples
    if retention is not None:
        result["retention"] = retention
    print(
        json.dumps({"visual_tokens": visual_tokens, "mode": mode, **result}), flush=True
    )
    return result


def plot_latency(rows, output_path: Path) -> None:
    plt = pyplot()
    figure, axis = plt.subplots(figsize=(9, 3.6))
    latency_percentages = [
        row["sparse"]["total_s"] / row["dense"]["total_s"] * 100 for row in rows
    ]
    for row_index, (row, latency_percent) in enumerate(zip(rows, latency_percentages)):
        axis.barh(
            row_index + 0.17,
            100,
            height=0.32,
            color="#76B7A7",
            label="Dense" if row_index == 0 else None,
        )
        axis.barh(
            row_index - 0.17,
            latency_percent,
            height=0.32,
            color="#F1A278",
            label="Sparse" if row_index == 0 else None,
        )
        axis.text(
            99,
            row_index + 0.17,
            f"{row['dense']['total_s']:.1f}s",
            ha="right",
            va="center",
        )
        axis.text(
            latency_percent - 1,
            row_index - 0.17,
            f"{row['sparse']['total_s']:.1f}s",
            ha="right",
            va="center",
        )
        axis.text(102, row_index, f"{row['speedup']:.2f}×", va="center")
    axis.set_yticks(range(len(rows)), [str(result["visual_tokens"]) for result in rows])
    axis.invert_yaxis()
    axis.set_xlim(0, max(115, max(latency_percentages) + 15))
    axis.set_xlabel("Latency (% of Dense, lower is better)")
    axis.set_ylabel("Visual tokens")
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2, frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    import transformers
    import triton

    print(
        json.dumps(
            {
                "config": {**vars(args), "output_dir": str(args.output_dir)},
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "triton": triton.__version__,
            }
        ),
        flush=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager",
    ).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for visual_tokens in args.visual_lengths:
        row = {"visual_tokens": visual_tokens}
        for mode in ("dense", "sparse"):
            row[mode] = benchmark_mode(model, args, visual_tokens, mode)
        row["speedup"] = row["dense"]["total_s"] / row["sparse"]["total_s"]
        rows.append(row)
    plot_latency(rows, args.output_dir / "latency_comparison.png")


if __name__ == "__main__":
    main()
