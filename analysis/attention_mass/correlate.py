"""Correlate visual attention mass with active token counts and plot the result."""

import argparse
from pathlib import Path

import numpy as np
import seaborn as sns

from analysis.common import load_trace, pearson, pyplot, trace_paths, visual_statistics

plt = pyplot()


OUTPUT_FILENAME = "mass_correlation.png"


def plot_correlation(visual_mass, counts_by_threshold, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(9, 6), dpi=300)

    visual_mass_values = np.array(visual_mass)
    colors = sns.color_palette("viridis", len(counts_by_threshold))

    for threshold_index, (threshold, token_counts) in enumerate(
        counts_by_threshold.items()
    ):
        active_counts = np.array(token_counts)

        correlation = pearson(visual_mass_values, active_counts)

        label = (
            f"Threshold $\\delta={threshold}$ ($r={correlation:.2f}$)"
            if correlation is not None
            else f"Threshold $\\delta={threshold}$ (r undefined)"
        )

        sns.regplot(
            x=visual_mass_values,
            y=active_counts,
            label=label,
            scatter_kws={"s": 15, "alpha": 0.6},
            line_kws={"linewidth": 1.5},
            color=colors[threshold_index],
            ci=None,
            ax=axis,
        )

    axis.set_xlabel(r"Visual Attention Mass", fontsize=28, fontweight="bold")
    axis.set_ylabel(r"Active Token Count", fontsize=28, fontweight="bold")
    axis.tick_params(axis="both", which="major", labelsize=20)
    axis.legend(fontsize=20)
    axis.grid(True, linestyle="--", alpha=0.3)
    figure.tight_layout()

    output_path = output_dir / OUTPUT_FILENAME
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True, help="A trace.npz or a collection directory"
    )
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.0005, 0.001, 0.0015, 0.002]
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/analysis/attention_mass")
    )
    args = parser.parse_args()
    paths = trace_paths(args.data)
    mass_batches, count_batches = [], []
    reference_identity = None
    for path in paths:
        trace = load_trace(path)
        identity = (
            trace["metadata"]["model"],
            trace["metadata"]["layer"],
            trace["metadata"]["attention"],
        )
        if reference_identity is not None and identity != reference_identity:
            raise ValueError(
                "Do not pool traces from different models, layers, or attention definitions."
            )
        reference_identity = identity
        visual_mass, active_counts = visual_statistics(
            trace["visual_scores"], args.thresholds
        )
        mass_batches.append(visual_mass)
        count_batches.append(active_counts)
    visual_mass_values, pooled_counts = np.concatenate(mass_batches), np.concatenate(
        count_batches
    )
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plot_correlation(
        visual_mass_values,
        {
            threshold: pooled_counts[:, threshold_index]
            for threshold_index, threshold in enumerate(args.thresholds)
        },
        args.output_dir,
    )
    print(
        f"Saved correlations for {len(paths)} sample(s), {len(visual_mass_values)} steps to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
