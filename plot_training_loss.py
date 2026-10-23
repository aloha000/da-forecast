"""Plot batch and epoch losses from the DA forecast training log."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
BATCH_RE = re.compile(
    rf"epoch=(\d+) step=(\d+) lr=({NUMBER}) batch_loss=({NUMBER})"
)
EPOCH_RE = re.compile(
    rf"epoch=(\d+)/(\d+) step=(\d+) lr=({NUMBER}) "
    rf"train=({NUMBER}) val=({NUMBER}) "
    rf"val_global=({NUMBER}) val_halo=({NUMBER})"
)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smooth", type=int, default=25)
    args = parser.parse_args()

    text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    per_step: dict[int, list[float]] = defaultdict(list)
    for match in BATCH_RE.finditer(text):
        per_step[int(match.group(2))].append(float(match.group(4)))
    epochs = [
        {
            "epoch": int(m.group(1)), "train": float(m.group(5)),
            "val": float(m.group(6)), "global": float(m.group(7)),
            "halo": float(m.group(8)),
        }
        for m in EPOCH_RE.finditer(text)
    ]
    if not per_step or not epochs:
        raise RuntimeError("no batch or epoch loss records found")

    steps = np.asarray(sorted(per_step))
    batch = np.asarray([np.mean(per_step[step]) for step in steps])
    epoch_x = np.asarray([item["epoch"] for item in epochs])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(steps, batch, color="#9ecae1", linewidth=0.7, alpha=0.6,
                 label="Per-step mean (4 ranks)")
    axes[0].plot(steps, moving_average(batch, args.smooth), color="#08519c",
                 linewidth=1.8, label=f"Moving average ({args.smooth} steps)")
    axes[0].set(title="Training batch loss", xlabel="Optimizer step", ylabel="Normalized MAE")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epoch_x, [x["train"] for x in epochs], marker="o", label="Train")
    axes[1].plot(epoch_x, [x["val"] for x in epochs], marker="o", label="Validation/global")
    axes[1].plot(epoch_x, [x["halo"] for x in epochs], marker="o", label="Validation halo")
    best = min(epochs, key=lambda item: item["val"])
    axes[1].scatter([best["epoch"]], [best["val"]], color="red", zorder=5)
    axes[1].annotate(
        f"best {best['val']:.6f}\nepoch {best['epoch']}",
        (best["epoch"], best["val"]), xytext=(8, 10), textcoords="offset points",
    )
    axes[1].set(title="Epoch loss", xlabel="Epoch", ylabel="Normalized MAE")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(
        f"saved {output}; batch steps={len(steps)}, completed epochs={len(epochs)}, "
        f"best val={best['val']:.6f} at epoch={best['epoch']}"
    )


if __name__ == "__main__":
    main()
