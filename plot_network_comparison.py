"""Draw current DA-Forecast and original FuXi architectures side by side."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "input": "#E7EEF8",
    "era": "#8EC5E8",
    "obs": "#91D4A3",
    "fusion": "#C7A7E8",
    "swin": "#F6BF73",
    "decoder": "#F29B91",
    "output": "#F6D66F",
    "static": "#D8E2EA",
    "edge": "#34495E",
    "text": "#17202A",
}


def box(ax, x, y, w, h, text, color, fs=8.2, lw=1.25):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.025,rounding_size=0.10",
        facecolor=color, edgecolor=COLORS["edge"], linewidth=lw, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center",
        fontsize=fs, color=COLORS["text"], linespacing=1.2, zorder=4,
    )
    return patch


def arrow(ax, start, end, color=None, dashed=False, rad=0.0, lw=1.45):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=11,
        connectionstyle=f"arc3,rad={rad}", color=color or COLORS["edge"],
        linewidth=lw, linestyle="--" if dashed else "-", zorder=2,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, fs=7.2, color="#566573", **kwargs):
    ax.text(x, y, text, fontsize=fs, color=color, ha="center", va="center", **kwargs)


def draw_current(ax):
    x0 = 0.35
    ax.text(
        4.8, 11.45, "Current DA-Forecast (ERA5 + GNSS-ZTD)",
        ha="center", va="center", fontsize=15, fontweight="bold",
        color=COLORS["text"],
    )
    ax.text(
        4.8, 11.08, "Original FuXi path + mask-aware Obs encoder + two gated corrections",
        ha="center", va="center", fontsize=8.5, color="#566573",
    )

    box(ax, x0, 9.35, 2.15, 1.10,
        "ERA5 + constants\n2 frames x (70 + 5 channels)\n720 x 1440", COLORS["input"], 7.9)
    box(ax, x0, 6.90, 2.15, 1.38,
        "GNSS-ZTD sequence\nB x T x 1 x 720 x 1440\n+ valid mask\n+ 3 station static fields", "#E1F3E6", 7.6)

    box(ax, 2.90, 9.35, 2.35, 1.10,
        "Original FuXi CubeEmbed\nConv 4x4 / 4 + flatten + LN\n1536 x 180 x 360", COLORS["era"], 7.6)
    box(ax, 2.90, 6.70, 2.35, 1.75,
        "Mask-aware Obs Encoder\n3-D temporal convolution\nvalid-time aggregation\npatch embed + downsample", COLORS["obs"], 7.6)
    arrow(ax, (2.50, 9.90), (2.90, 9.90))
    arrow(ax, (2.50, 7.59), (2.90, 7.59))

    box(ax, 5.78, 9.35, 1.78, 1.10,
        "High gated correction\nERA high + gate x Obs delta\n1536 x 180 x 360", COLORS["fusion"], 7.15)
    arrow(ax, (5.25, 9.90), (5.78, 9.90))
    arrow(ax, (5.25, 7.90), (5.78, 9.55), rad=-0.12, color="#5B8E68")

    box(ax, 5.78, 7.78, 1.78, 0.98,
        "Original FuXi DownBlock / 2\ntime-conditioned\n1536 x 90 x 180", "#A7D0EA", 7.15)
    arrow(ax, (6.67, 9.35), (6.67, 8.76))

    box(ax, 5.78, 6.25, 1.78, 0.98,
        "Low gated correction\nFuXi low + gate x Obs delta\n1536 x 90 x 180", COLORS["fusion"], 7.15)
    arrow(ax, (6.67, 7.78), (6.67, 7.23))
    arrow(ax, (5.25, 7.16), (5.78, 6.74), color="#5B8E68")

    box(ax, 3.25, 4.40, 3.10, 1.25,
        "FuXi Swin Transformer @ 90 x 180\n4 stages x 12 blocks | window 9 x 9\n1536 dim | 24 heads | periodic longitude", COLORS["swin"], 8.0)
    arrow(ax, (6.35, 6.25), (5.45, 5.65), rad=0.08)
    box(ax, 3.25, 3.35, 3.10, 0.65,
        "FuXi FPN: LN(stage 1..4) -> concat -> Linear -> GELU",
        "#F9D9A7", 7.5)
    arrow(ax, (4.80, 4.40), (4.80, 4.00))

    box(ax, 2.95, 1.75, 3.70, 1.10,
        "FuXi Decoder\nconcat Transformer + low skip\n"
        "time-conditioned UpBlock (x2 spatial)", COLORS["decoder"], 7.7)
    arrow(ax, (4.80, 3.35), (4.80, 2.85))
    arrow(ax, (6.67, 6.25), (6.20, 2.85), dashed=True, rad=-0.20, color="#7D3C98")
    label(ax, 7.00, 4.10, "1/8 skip", color="#7D3C98")

    box(ax, 3.28, 0.78, 3.04, 0.58,
        "Linear(1536 -> 70 x 4 x 4) -> forecast delta",
        "#F5BDB7", 7.5)
    arrow(ax, (4.80, 1.75), (4.80, 1.36))
    box(ax, 3.43, -0.08, 2.74, 0.55,
        "ERA5(t) + delta -> ERA5(t+6h)", COLORS["output"], 7.7)
    arrow(ax, (4.80, 0.78), (4.80, 0.47))
    label(ax, 4.80, -0.48,
          "1.792B total | FuXi 1.557B frozen | Obs + gates 0.235B trainable", fs=7.5)


def draw_fuxi(ax):
    ax.text(
        14.6, 11.45, "Original FuXi Short-range Model",
        ha="center", va="center", fontsize=15, fontweight="bold",
        color=COLORS["text"],
    )
    ax.text(
        14.6, 11.08, "Single CubeEmbed path + FuXi U-Transformer",
        ha="center", va="center", fontsize=8.5, color="#566573",
    )

    box(ax, 12.80, 9.35, 3.60, 1.10,
        "ERA5 dynamic + constants\n2 frames x (70 ERA5 + 5 static) x 720 x 1440",
        COLORS["input"], 8.0)
    box(ax, 12.80, 7.85, 3.60, 0.90,
        "CubeEmbed: Conv 4x4 / 4 + flatten\nB x 64800 tokens x 1536",
        COLORS["era"], 8.0)
    arrow(ax, (14.60, 9.35), (14.60, 8.75))

    box(ax, 12.80, 6.45, 3.60, 0.90,
        "FuXi DownBlock / 2 (time-conditioned)\n1536 x 180 x 360 -> 1536 x 90 x 180",
        "#A7D0EA", 7.8)
    arrow(ax, (14.60, 7.85), (14.60, 7.35))

    box(ax, 12.80, 4.70, 3.60, 1.20,
        "FuXi Swin Transformer @ 90 x 180\n4 stages x 12 blocks | window 9 x 9\n1536 dim | 24 heads | periodic longitude", COLORS["swin"], 8.0)
    arrow(ax, (14.60, 6.45), (14.60, 5.90))
    box(ax, 12.80, 3.55, 3.60, 0.72,
        "FuXi FPN: LN(stage 1..4) -> concat -> Linear -> GELU",
        "#F9D9A7", 7.5)
    arrow(ax, (14.60, 4.70), (14.60, 4.27))

    box(ax, 12.55, 1.78, 4.10, 1.20,
        "FuXi Decoder\nconcat Transformer + DownBlock skip\n-> time-conditioned UpBlock x2 -> 1536 x 180 x 360",
        COLORS["decoder"], 7.8)
    arrow(ax, (14.60, 3.55), (14.60, 2.98))
    arrow(ax, (16.40, 6.90), (16.25, 2.78), dashed=True, rad=-0.20, color="#7D3C98")
    label(ax, 17.05, 4.68, "1/8 skip", color="#7D3C98")

    box(ax, 13.08, 0.72, 3.04, 0.66,
        "Linear(1536 -> 70 x 4 x 4) -> forecast delta",
        "#F5BDB7", 7.5)
    arrow(ax, (14.60, 1.78), (14.60, 1.38))
    box(ax, 13.23, -0.18, 2.74, 0.55,
        "ERA5(t) + delta -> ERA5(t+6h)", COLORS["output"], 7.7)
    arrow(ax, (14.60, 0.72), (14.60, 0.37))
    label(ax, 14.60, -0.52, "Increment prediction | 1.557B parameters", fs=7.8)


def draw(output: Path, dpi: int = 200):
    fig, ax = plt.subplots(figsize=(20, 13))
    ax.set_xlim(0, 19.4)
    ax.set_ylim(-1.35, 12.35)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.text(
        9.7, 12.08, "Global 6-hour Forecast Architecture Comparison",
        ha="center", va="center", fontsize=22, fontweight="bold",
        color=COLORS["text"],
    )
    ax.plot([9.7, 9.7], [-0.75, 11.65], color="#AAB7B8", linewidth=1.4)
    draw_current(ax)
    draw_fuxi(ax)

    box(ax, 0.65, -1.22, 18.10, 0.42,
        "Key difference: DA-Forecast keeps the complete pretrained FuXi path and adds only a mask-aware GNSS-ZTD encoder with gated corrections after CubeEmbed and DownBlock.",
        "#F4F6F7", 8.0, 1.0)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.16, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="network_architecture_comparison.png")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    output = Path(args.output)
    draw(output, args.dpi)
    print(output.resolve())


if __name__ == "__main__":
    main()
