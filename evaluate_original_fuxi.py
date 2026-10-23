"""Evaluate the original FuXi short-range checkpoint on normalized ERA5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr

from model.fuxi import UTransformer
from model.fuxi_pretrained import _PICKLE_COMPAT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="model/fuxi_short.pth")
    parser.add_argument("--era5", default="datasets/era5_global_0p25_6h.zarr")
    parser.add_argument("--conds", default="model/conds.npy")
    parser.add_argument("--start", default="2025-01-01T00:00:00")
    parser.add_argument("--end", default="2026-01-01T00:00:00")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", default="work_dirs/original_fuxi_short_mae.json")
    return parser.parse_args()


def decode_time(array) -> pd.DatetimeIndex:
    units = str(dict(array.attrs)["units"])
    unit, origin = [value.strip() for value in units.split("since", 1)]
    code = {"hours": "h", "hour": "h", "minutes": "min", "minute": "min"}[unit]
    return pd.DatetimeIndex(pd.Timestamp(origin) + pd.to_timedelta(array[:], unit=code))


def time_encoding(timestamp: pd.Timestamp, device) -> torch.Tensor:
    # Match code_inference/inference_forecast.py exactly.
    times = [timestamp + pd.Timedelta(hours=offset) for offset in (-6, 0, 6)]
    values = np.asarray(
        [(value.dayofyear / 366.0, value.hour / 24.0) for value in times],
        dtype=np.float32,
    )
    values = torch.from_numpy(values).to(device)
    return torch.cat((values.sin(), values.cos()), dim=-1).reshape(1, 12)


def load_model(path: str, device) -> UTransformer:
    model = UTransformer(
        in_chans=75, out_chans=70, in_frames=2, image_size=(720, 1440),
        window_size=9, patch_size=4, down_times=1, embed_dim=1536,
        num_heads=24, depths=[12, 12, 12, 12],
    ).to(device=device, dtype=torch.float16)
    checkpoint = torch.load(
        path, map_location="cpu", mmap=True, weights_only=False,
        pickle_module=_PICKLE_COMPAT,
    )
    state = {
        key.removeprefix("decoder."): value
        for key, value in checkpoint["model"].items()
        if key.startswith("decoder.")
    }
    model.load_state_dict(state, strict=True)
    return model.eval()


def main():
    args = parse_args()
    device = torch.device(args.device)
    group = zarr.open(args.era5, mode="r")
    data = group["z"]
    times = decode_time(group["time"])
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    indices = np.flatnonzero((times >= start) & (times + pd.Timedelta(hours=6) < end))
    indices = indices[(indices > 0) & (indices + 1 < len(times))]
    indices = indices[
        (times[indices] - times[indices - 1] == pd.Timedelta(hours=6))
        & (times[indices + 1] - times[indices] == pd.Timedelta(hours=6))
    ]
    if args.max_samples and len(indices) > args.max_samples:
        positions = np.linspace(0, len(indices) - 1, args.max_samples, dtype=int)
        indices = indices[positions]

    model = load_model(args.checkpoint, device)
    const = torch.from_numpy(np.load(args.conds)).to(device=device, dtype=torch.float16)
    lat = np.asarray(group["lat"][:], dtype=np.float64)
    weights = np.cos(np.deg2rad(lat)).clip(min=0)
    weights /= weights.mean()
    weights = torch.as_tensor(weights, device=device).view(1, 1, -1, 1)
    error_sum = persistence_sum = denominator = 0.0

    with torch.inference_mode():
        for number, index in enumerate(indices, 1):
            fields = np.asarray(data[index - 1:index + 2], dtype=np.float16)
            inputs = torch.from_numpy(fields[:2]).unsqueeze(0).to(device)
            target = torch.from_numpy(fields[2]).unsqueeze(0).to(device)
            residual = model(inputs, temb=time_encoding(times[index], device), const=const)
            prediction = residual + inputs[:, -1]
            valid = torch.isfinite(target)
            spatial_weights = weights * valid
            denominator += spatial_weights.sum(dtype=torch.float64).item()
            error_sum += ((prediction.float() - target.float()).abs() * spatial_weights).sum(
                dtype=torch.float64
            ).item()
            persistence_sum += ((inputs[:, -1].float() - target.float()).abs() * spatial_weights).sum(
                dtype=torch.float64
            ).item()
            if number == 1 or number % 20 == 0 or number == len(indices):
                print(f"samples={number}/{len(indices)} mae={error_sum / denominator:.6f}", flush=True)

    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "start": str(start), "end_exclusive": str(end),
        "samples": int(len(indices)),
        "overall_latitude_weighted_mae_normalized": error_sum / denominator,
        "persistence_latitude_weighted_mae_normalized": persistence_sum / denominator,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
