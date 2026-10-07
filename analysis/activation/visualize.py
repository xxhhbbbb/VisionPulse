"""Visualize attention mass and selected visual activations in one PNG."""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

from analysis.common import load_trace, pyplot

plt = pyplot()


CURVE_COLOR = "#1f77b4"
HEATMAP_COLORMAP = "autumn_r"
HEATMAP_ALPHA = 0.6
DEFAULT_PRUNE_LEVELS = (50, 90)


def create_activation_overlay(image, visual_scores, grid_shape, prune_percentile):
    """Overlay attention on the image, masking scores at or below the percentile."""
    image_width, image_height = image.size
    num_visual_tokens = len(visual_scores)

    if grid_shape is not None:
        grid_height, grid_width = int(grid_shape[-2]), int(grid_shape[-1])
        stride = (
            math.sqrt((grid_height * grid_width) / num_visual_tokens)
            if num_visual_tokens > 0
            else 0
        )
        if stride > 0:
            token_grid_height = int(round(grid_height / stride))
            token_grid_width = int(round(grid_width / stride))
        else:
            side_length = int(math.isqrt(num_visual_tokens))
            token_grid_height, token_grid_width = side_length, side_length
    else:

        side_length = int(math.isqrt(num_visual_tokens))
        token_grid_height, token_grid_width = side_length, side_length

    # Preserve the source image when token positions cannot be mapped to its grid.
    if token_grid_height * token_grid_width != num_visual_tokens:
        return image

    attention_grid = visual_scores.reshape(token_grid_height, token_grid_width)

    min_score, max_score = attention_grid.min(), attention_grid.max()
    if (max_score - min_score) > 1e-9:
        normalized_scores = (attention_grid - min_score) / (max_score - min_score)
    else:
        normalized_scores = np.zeros_like(attention_grid)

    background = image.convert("RGB")

    colormap = plt.get_cmap(HEATMAP_COLORMAP)
    heatmap_rgba = colormap(normalized_scores)
    heatmap_uint8 = (heatmap_rgba * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_uint8).resize(
        (image_width, image_height), resample=Image.NEAREST
    )

    # Exclude ties at the percentile, matching the original activation masks.
    if prune_percentile > 0:
        threshold = np.percentile(normalized_scores, prune_percentile)
    else:
        threshold = -1.0

    region_mask = np.where(normalized_scores <= threshold, 0, 255).astype(np.uint8)
    region_mask_pil = Image.fromarray(region_mask, mode="L").resize(
        (image_width, image_height), resample=Image.NEAREST
    )

    alpha_mask = normalized_scores.copy()
    alpha_mask[alpha_mask <= threshold] = 0
    alpha_mask = (alpha_mask * 255 * HEATMAP_ALPHA).astype(np.uint8)
    alpha_mask_pil = Image.fromarray(alpha_mask, mode="L").resize(
        (image_width, image_height), resample=Image.NEAREST
    )

    heatmap_pil.putalpha(alpha_mask_pil)

    overlay = Image.alpha_composite(background.convert("RGBA"), heatmap_pil)
    black_background = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 255))
    masked_overlay = Image.composite(overlay, black_background, region_mask_pil)

    return masked_overlay.convert("RGB")


def plot_activation(
    visual_scores, token_texts, image_grid, image_path, steps, prune_levels, output_path
):
    """Combine the attention-mass curve and selected activation maps into one PNG."""
    with Image.open(image_path) as source:
        raw_image = source.convert("RGB")
    plt.rcParams.update(
        {"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]}
    )
    figure = plt.figure(
        figsize=(14, 3.3 + 2.8 * len(prune_levels)), layout="constrained"
    )
    grid = figure.add_gridspec(
        1 + len(prune_levels), len(steps), height_ratios=[1.2] + [1] * len(prune_levels)
    )
    axis = figure.add_subplot(grid[0, :])
    visual_mass = visual_scores.sum(axis=-1)
    step_indices = np.arange(len(visual_mass))
    axis.plot(step_indices, visual_mass, color=CURVE_COLOR, linewidth=2)
    axis.fill_between(step_indices, visual_mass, color=CURVE_COLOR, alpha=0.1)
    axis.set_xlabel("Generated Step (zero-based)", fontsize=16)
    axis.set_ylabel("Visual Attention Mass", fontsize=16)
    axis.grid(True, linestyle="--", alpha=0.3)
    for step in steps:
        axis.axvline(step, color=CURVE_COLOR, linestyle="--", alpha=0.5)
    for row, percentile in enumerate(prune_levels, start=1):
        for column, step in enumerate(steps):
            axis = figure.add_subplot(grid[row, column])
            overlay = create_activation_overlay(
                raw_image, visual_scores[step], image_grid, percentile
            )
            axis.imshow(overlay)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            token = str(token_texts[step]).strip() or "[whitespace]"
            axis.set_title(f"Step {step}: {token!r}", fontsize=15)
            if column == 0:
                axis.set_ylabel(
                    f"Top {100 - percentile:g}% visual activation", fontsize=14
                )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        help="Zero-based steps; default: three evenly spaced steps",
    )
    parser.add_argument(
        "--prune-levels",
        type=float,
        nargs="+",
        default=DEFAULT_PRUNE_LEVELS,
        help="Percentiles to mask; 50 and 90 show the top 50%% and 10%%",
    )
    parser.add_argument("--image", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/analysis/activation")
    )
    args = parser.parse_args()
    trace = load_trace(args.data)
    visual_scores, token_texts = trace["visual_scores"], trace["token_texts"]
    if args.steps is not None and any(
        step < 0 or step >= len(visual_scores) for step in args.steps
    ):
        parser.error(f"Steps must be between 0 and {len(visual_scores) - 1}.")
    if any(not 0 <= percentile <= 100 for percentile in args.prune_levels):
        parser.error("Prune percentiles must be in [0, 100].")
    source = args.image or args.data.parent / "image.png"
    if not source.is_file():
        raise FileNotFoundError(source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    steps = (
        args.steps
        or np.unique(
            np.linspace(
                0, len(visual_scores) - 1, min(3, len(visual_scores)), dtype=int
            )
        ).tolist()
    )
    plot_activation(
        visual_scores,
        token_texts,
        trace["grid_hw"],
        source,
        steps,
        args.prune_levels,
        args.output_dir / "visual_activation.png",
    )


if __name__ == "__main__":
    main()
