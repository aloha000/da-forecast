"""Plot only the land-sea mask, deriving coordinates from conds.npy itself.

cos(latitude) does not encode hemisphere. Both possible row orientations are
shown explicitly; no external latitude/longitude arrays or map data are used.
"""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conds", type=Path, default=root / "model/conds.npy")
    parser.add_argument("--output", type=Path, default=root / "conds_land_mask.png")
    args = parser.parse_args()
    fields = np.load(args.conds)
    if fields.ndim != 3 or fields.shape[0] != 5:
        raise ValueError(f"expected [5,H,W], got {fields.shape}")
    mask = fields[0]
    if not np.isfinite(fields).all():
        raise ValueError("conds contains nonfinite values")
    if not np.allclose(fields[2], fields[2, :, :1], atol=1e-6):
        raise ValueError("latitude cosine is not constant along longitude")
    for index in (3, 4):
        if not np.allclose(fields[index], fields[index, :1, :], atol=1e-6):
            raise ValueError("longitude encoding is not constant along latitude")

    # These are coordinates inferred ONLY from channels 2, 3 and 4.
    abs_lat = np.rad2deg(np.arccos(np.clip(fields[2, :, 0].astype(float), -1, 1)))
    lon = np.rad2deg(np.arctan2(fields[4, 0].astype(float), fields[3, 0].astype(float)))
    lon = (lon + 180.0) % 360.0 - 180.0
    lon_order = np.argsort(lon)
    height = mask.shape[0]
    if height % 2:
        raise ValueError("this plot expects an even number of latitude rows")
    # Sign assignment is an assumption, not information recovered from cosine.
    north_first = abs_lat * np.where(np.arange(height) < height // 2, 1.0, -1.0)
    south_first = -north_first

    cmap = LinearSegmentedColormap.from_list("ocean_land", ["#e8f3fb", "#3c6849"])
    fig, axes = plt.subplots(2, 1, figsize=(15, 10.5), sharex=True)
    fig.subplots_adjust(left=0.07, right=0.88, top=0.90, bottom=0.12, hspace=0.25)
    fig.suptitle("FuXi conds.npy: land-sea mask (channel 0)", fontsize=19, y=0.97)
    for ax, lat, title in zip(axes, (north_first, south_first), (
        "A | Assume first row is NORTH: rows run 90 N to 90 S",
        "B | Assume first row is SOUTH: rows run 90 S to 90 N",
    )):
        lat_order = np.argsort(lat)
        mesh = ax.pcolormesh(lon[lon_order], lat[lat_order],
                             mask[lat_order][:, lon_order], shading="auto",
                             cmap=cmap, vmin=0, vmax=1, rasterized=True)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xticks(np.arange(-180, 181, 60))
        ax.set_yticks(np.arange(-90, 91, 30))
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda v, _: "0" if v == 0 else f"{abs(v):g} {'W' if v < 0 else 'E'}"))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _: "0" if v == 0 else f"{abs(v):g} {'S' if v < 0 else 'N'}"))
        ax.grid(alpha=0.28, linewidth=0.5)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.set_title(title, loc="left", fontsize=13, pad=10)
    axes[-1].set_xlabel("Longitude", fontsize=11)
    cax = fig.add_axes([0.905, 0.29, 0.016, 0.42])
    bar = fig.colorbar(mesh, cax=cax, ticks=[0, 0.5, 1])
    bar.ax.set_yticklabels(["0: ocean", "0.5", "1: land"])
    bar.set_label("Land fraction")
    fig.text(0.07, 0.048,
             "Coordinates from conds only: longitude = atan2(channel 4, channel 3); "
             "|latitude| = arccos(channel 2).\n"
             "Latitude cosine cannot distinguish north from south. "
             "Both sign assignments are shown; no external coastline is overlaid.",
             fontsize=10, color="#444444", linespacing=1.6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"saved: {args.output}")
    print(f"mask range: {mask.min():.6g} .. {mask.max():.6g}; shape: {mask.shape}")
    print("Coordinate files used: conds.npy only. Hemisphere signs are explicit assumptions.")


if __name__ == "__main__":
    main()
