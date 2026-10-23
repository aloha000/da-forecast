"""Draw the current DA-Forecast network architecture as a PNG."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


COLORS = {
    "input": "#DCEBFA",
    "era": "#88BDE6",
    "obs": "#8FD19E",
    "fusion": "#C6A5E8",
    "swin": "#F7BE72",
    "decoder": "#F39A8E",
    "output": "#F8D66D",
    "neutral": "#E9EDF2",
    "edge": "#34495E",
}


def add_box(ax, x, y, w, h, text, color, fontsize=9, linewidth=1.5):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.025,rounding_size=0.12",
        facecolor=color,
        edgecolor=COLORS["edge"],
        linewidth=linewidth,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        color="#17202A", linespacing=1.25, zorder=4,
    )
    return patch


def add_arrow(
    ax, start, end, label=None, rad=0.0, dashed=False,
    color=None, label_offset=(0.0, 0.0), linewidth=1.8,
):
    color = color or COLORS["edge"]
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=13,
        connectionstyle=f"arc3,rad={rad}",
        linewidth=linewidth,
        linestyle="--" if dashed else "-",
        color=color,
        zorder=2,
    )
    ax.add_patch(arrow)
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(
            mx, my, label, ha="center", va="center", fontsize=7.5,
            color=color, backgroundcolor="white", zorder=5,
        )
    return arrow


def draw(output: Path, dpi: int = 180):
    fig, ax = plt.subplots(figsize=(20, 11.25))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 11.25)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.text(
        10, 10.88, "ERA5 + GNSS-ZTD Regional 6-hour Forecast Network",
        ha="center", va="center", fontsize=21, fontweight="bold", color="#17202A",
    )
    ax.text(
        10, 10.48,
        "ERA5(t-6h, t) + ZTD(t-6h ... t, every 20 min; 19 frames)  ->  ERA5(t+6h)    |    Europe 80 x 120 @ 0.25 deg",
        ha="center", va="center", fontsize=11.5, color="#566573",
    )

    # Inputs.
    add_box(
        ax, 0.25, 8.25, 2.05, 1.25,
        "ERA5 dynamic\n[B, 2, 70, 80, 120]\nt-6h and t",
        COLORS["input"], 9,
    )
    add_box(
        ax, 0.25, 6.68, 2.05, 1.05,
        "ERA5 static\n[B, 5, 80, 120]",
        COLORS["input"], 9,
    )
    add_box(
        ax, 0.25, 3.15, 2.05, 2.55,
        "GNSS-ZTD\n[B, 19, 1, 80, 120]\n\nvalid mask\n[B, 19, 1, 80, 120]\n\nstation / lat / lon\n[B, 3, 80, 120]",
        "#DDF3E3", 8.5,
    )

    # Encoders.
    add_box(
        ax, 2.85, 7.15, 2.55, 2.35,
        "ERA5 Encoder\n\nDynamic: Conv 4x4 / 4\nStatic: Conv 4x4 / 4\nAdd + ResBlock\n\nDownsample Conv / 2\n+ ResBlock",
        COLORS["era"], 9,
    )
    add_box(
        ax, 2.85, 3.15, 2.55, 2.55,
        "Mask-aware Obs Encoder\n\nConv3D 2 -> 32 -> 64\nmasked temporal mean\nconcat 3 static channels\nConv 4x4 / 4 + ResBlock\n\nDownsample Conv / 2",
        COLORS["obs"], 8.8,
    )
    add_arrow(ax, (2.30, 8.88), (2.85, 8.68))
    add_arrow(ax, (2.30, 7.20), (2.85, 7.75))
    add_arrow(ax, (2.30, 4.42), (2.85, 4.42))

    # Two-scale features.
    add_box(
        ax, 5.85, 8.15, 1.7, 1.0,
        "ERA high\n[B,256,20,30]", COLORS["era"], 8.5,
    )
    add_box(
        ax, 5.85, 6.42, 1.7, 1.0,
        "ERA low\n[B,256,10,15]", COLORS["era"], 8.5,
    )
    add_box(
        ax, 5.85, 4.55, 1.7, 1.0,
        "Obs high + mask\n20 x 30", COLORS["obs"], 8.3,
    )
    add_box(
        ax, 5.85, 2.82, 1.7, 1.0,
        "Obs low + mask\n10 x 15", COLORS["obs"], 8.3,
    )
    add_arrow(ax, (5.40, 8.55), (5.85, 8.65))
    add_arrow(ax, (5.40, 7.65), (5.85, 6.92))
    add_arrow(ax, (5.40, 4.75), (5.85, 5.05))
    add_arrow(ax, (5.40, 3.72), (5.85, 3.32))

    # Gated fusion at high and low scales.
    add_box(
        ax, 8.05, 7.68, 2.1, 1.3,
        "High Gated Fusion\nconcat ERA / Obs / mask\nConv 513 -> 256 -> 256\nERA + sigmoid(g) x delta",
        COLORS["fusion"], 8.4,
    )
    add_box(
        ax, 8.05, 4.55, 2.1, 1.3,
        "Low Gated Fusion\nconcat ERA / Obs / mask\nConv 513 -> 256 -> 256\nERA + sigmoid(g) x delta",
        COLORS["fusion"], 8.4,
    )
    add_arrow(ax, (7.55, 8.65), (8.05, 8.45))
    add_arrow(ax, (7.55, 5.05), (8.05, 7.95), rad=-0.18)
    add_arrow(ax, (7.55, 6.92), (8.05, 5.35), rad=0.10)
    add_arrow(ax, (7.55, 3.32), (8.05, 4.95), rad=-0.10)

    # Time embedding and Swin backbone.
    add_box(
        ax, 8.05, 1.65, 2.1, 1.0,
        "Time embedding\n12 -> 256 -> 256",
        COLORS["neutral"], 8.7,
    )
    add_box(
        ax, 10.85, 4.40, 2.55, 1.75,
        "Regional Swin Backbone\n\n150 tokens x 256\n4 groups x 2 blocks = 8\nwindow 5 x 5, 8 heads\n2-D boundary mask (hw)",
        COLORS["swin"], 9,
    )
    add_arrow(ax, (10.15, 5.20), (10.85, 5.20), "[B,256,10,15]", label_offset=(0, 0.22))
    add_arrow(ax, (9.10, 2.65), (11.35, 4.40), "add to tokens", rad=-0.12, dashed=True, label_offset=(0.15, -0.05))

    # Low skip decoder stage.
    add_box(
        ax, 13.95, 4.40, 2.05, 1.75,
        "Low Skip Fuse\n\nconcat Swin + low skip\n[B,512,10,15]\nResBlock 512 -> 256",
        COLORS["decoder"], 8.7,
    )
    add_arrow(ax, (13.40, 5.20), (13.95, 5.20))
    add_arrow(
        ax, (9.10, 4.55), (14.10, 4.40), "low skip",
        rad=0.23, dashed=True, color="#7D3C98", label_offset=(0.0, -0.33),
    )

    # Upsample and high skip decoder stage.
    add_box(
        ax, 13.95, 7.50, 2.05, 1.15,
        "Deconv x2\n10x15 -> 20x30",
        COLORS["decoder"], 8.7,
    )
    add_arrow(ax, (14.98, 6.15), (14.98, 7.50), "[B,256,20,30]", label_offset=(0.8, 0))
    add_box(
        ax, 16.50, 7.30, 2.15, 1.55,
        "High Skip Fuse\n\nconcat upsample + high\n[B,512,20,30]\nResBlock 512 -> 256",
        COLORS["decoder"], 8.5,
    )
    add_arrow(ax, (16.00, 8.08), (16.50, 8.08))
    add_arrow(
        ax, (10.15, 8.33), (16.50, 8.42), "high skip [B,256,20,30]",
        rad=-0.24, dashed=True, color="#7D3C98", label_offset=(0, 0.78),
    )

    # Output head, residual addition and forecast.
    add_box(
        ax, 16.55, 4.35, 2.05, 1.25,
        "Output Head\nDeconv 4x4 / 4\nDelta: [B,70,80,120]",
        COLORS["decoder"], 8.6,
    )
    add_arrow(ax, (17.58, 7.30), (17.58, 5.60))
    add_arrow(ax, (18.60, 4.98), (19.08, 4.98))
    ax.plot(
        [2.05, 19.30, 19.30], [9.70, 9.70, 5.55],
        linestyle="--", linewidth=1.8, color="#2471A3", zorder=1,
    )
    ax.text(
        16.9, 9.88, "latest ERA5 X(t)", ha="center", va="center",
        fontsize=7.8, color="#2471A3", backgroundcolor="white", zorder=5,
    )
    add_arrow(
        ax, (19.30, 5.55), (19.30, 5.20),
        dashed=True, color="#2471A3", linewidth=1.8,
    )
    add_circle = Circle((19.30, 4.98), 0.22, facecolor=COLORS["output"],
                        edgecolor=COLORS["edge"], linewidth=1.5, zorder=4)
    ax.add_patch(add_circle)
    ax.text(19.30, 4.98, "+", ha="center", va="center", fontsize=15,
            fontweight="bold", zorder=5)
    add_box(
        ax, 18.38, 2.55, 1.55, 1.35,
        "Forecast\nX(t+6h)\n[B,70,80,120]",
        COLORS["output"], 9.2,
    )
    add_arrow(ax, (19.30, 4.76), (19.18, 3.90))

    # Notes and legend.
    add_box(
        ax, 10.75, 1.02, 7.20, 1.10,
        "Residual forecast:  X_hat(t+6h) = X(t) + Delta X    |    22,196,390 trainable parameters\n"
        "Normal regional convolutions (no periodic padding)    |    all-missing Obs -> exact ERA feature path",
        "#F7F9F9", 9.1, linewidth=1.2,
    )
    legend = [
        ("ERA5 path", COLORS["era"]),
        ("Observation path", COLORS["obs"]),
        ("Gated fusion", COLORS["fusion"]),
        ("Swin Transformer", COLORS["swin"]),
        ("Decoder", COLORS["decoder"]),
    ]
    lx, ly = 0.45, 0.70
    for label, color in legend:
        ax.add_patch(FancyBboxPatch(
            (lx, ly), 0.36, 0.25, boxstyle="round,pad=0.01,rounding_size=0.04",
            facecolor=color, edgecolor=COLORS["edge"], linewidth=0.9,
        ))
        ax.text(lx + 0.46, ly + 0.125, label, va="center", fontsize=8.5)
        lx += 1.85

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="network_architecture.png")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    output = Path(args.output)
    draw(output, args.dpi)
    print(output.resolve())


if __name__ == "__main__":
    main()
