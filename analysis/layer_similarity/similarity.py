"""Aggregate and visualize layer-wise visual-attention cosine similarity."""

import argparse
from pathlib import Path

import numpy as np
import seaborn as sns

from analysis.common import load_trace, pyplot, trace_paths

plt = pyplot()


def set_plot_style(font_size: int = 20):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.titlesize": font_size + 2,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 2,
            "ytick.labelsize": font_size - 2,
            "legend.fontsize": font_size - 2,
        }
    )


def plot_similarity(
    similarity_matrix: np.ndarray,
    output_path: Path,
    title: str,
    font_size: int,
    cmap: str,
    vmin: float,
    vmax: float,
):
    similarity_matrix = np.asarray(similarity_matrix)
    if similarity_matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={similarity_matrix.shape}")

    set_plot_style(font_size)
    figure, axis = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix,
        ax=axis,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar=True,
        xticklabels=True,
        yticklabels=True,
    )
    axis.set_title(title, pad=14)
    axis.set_xlabel("Layer Index")
    axis.set_ylabel("Layer Index")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_path}")


def aggregate_similarity(paths):
    # Average within each sample before pooling so long responses do not dominate.
    sample_matrices, sample_metadata = [], []
    expected_layers, expected_model = None, None
    for path in paths:
        trace = load_trace(path)
        if "similarity_by_step" not in trace:
            raise ValueError(
                f"{path}: layer similarities missing; collect with --all-layers."
            )
        layer_indices, step_similarities = (
            trace["layer_indices"],
            trace["similarity_by_step"],
        )
        if (
            step_similarities.shape
            != (len(trace["token_ids"]), len(layer_indices), len(layer_indices))
            or not np.isfinite(step_similarities).all()
            or len(layer_indices) < 2
        ):
            raise ValueError(f"Invalid per-step layer similarity matrices in {path}")
        if expected_layers is not None and (
            not np.array_equal(layer_indices, expected_layers)
            or trace["metadata"]["model"] != expected_model
        ):
            raise ValueError("All traces must have the same model and layer indices.")
        expected_layers, expected_model = layer_indices, trace["metadata"]["model"]
        sample_matrices.append(step_similarities.mean(axis=0))
        sample_metadata.append(
            {
                "trace": str(path),
                "sample_id": trace["metadata"]["sample_id"],
                "num_steps": len(step_similarities),
            }
        )
    return np.mean(sample_matrices, axis=0), expected_layers, sample_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True, help="A trace.npz or collection directory"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/analysis/layer_similarity")
    )
    parser.add_argument("--vmin", type=float, default=0.5, help="Heatmap lower bound")
    parser.add_argument("--font-size", type=int, default=22)
    parser.add_argument("--title", default="Visual attention similarity across layers")
    args = parser.parse_args()
    if not 0 <= args.vmin < 1:
        parser.error("--vmin must be in [0, 1).")
    similarity_matrix, layer_indices, sample_metadata = aggregate_similarity(
        trace_paths(args.data)
    )
    plot_similarity(
        similarity_matrix,
        args.output_dir / "average_layer_similarity.png",
        args.title,
        args.font_size,
        "viridis",
        args.vmin,
        1.0,
    )
    print(
        f"Saved {len(layer_indices)}×{len(layer_indices)} similarity matrix from {len(sample_metadata)} sample(s) to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
