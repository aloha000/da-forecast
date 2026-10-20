"""Evaluate a checkpoint and make physical-unit weather animations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import DAForecastDataset
from model import DAForecastNet
from train import load_config, make_dataset, resolve_data_paths


CHANNELS = [
    "z50", "z100", "z150", "z200", "z250", "z300", "z400", "z500",
    "z600", "z700", "z850", "z925", "z1000",
    "t50", "t100", "t150", "t200", "t250", "t300", "t400", "t500",
    "t600", "t700", "t850", "t925", "t1000",
    "u50", "u100", "u150", "u200", "u250", "u300", "u400", "u500",
    "u600", "u700", "u850", "u925", "u1000",
    "v50", "v100", "v150", "v200", "v250", "v300", "v400", "v500",
    "v600", "v700", "v850", "v925", "v1000",
    "r50", "r100", "r150", "r200", "r250", "r300", "r400", "r500",
    "r600", "r700", "r850", "r925", "r1000",
    "t2m", "u10", "v10", "msl", "tp",
]

ANIMATION_VARIABLES = (
    ("z500", "500 hPa geopotential height", "gpm", "viridis"),
    ("t850", "850 hPa temperature", "degC", "coolwarm"),
    ("r850", "850 hPa relative humidity", "%", "YlGnBu"),
    ("t2m", "2 m temperature", "degC", "coolwarm"),
    ("wind10", "10 m wind speed", "m s-1", "plasma"),
    ("msl", "Mean sea-level pressure", "hPa", "viridis"),
    ("tp", "6 h total precipitation", "mm", "Blues"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_da_forecast.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="work_dirs/da_forecast_6h/test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--animation-start", default="2025-05-01T00:00:00")
    parser.add_argument("--animation-end", default="2025-05-05T23:59:59")
    parser.add_argument("--animation-fps", type=int, default=2)
    parser.add_argument("--skip-animations", action="store_true")
    return parser.parse_args()


def channel_unit(channel: str) -> str:
    if channel == "tp":
        return "mm"
    if channel.startswith("z"):
        return "gpm"
    if channel.startswith("t"):
        return "K"
    if channel.startswith("u") or channel.startswith("v"):
        return "m s-1"
    if channel.startswith("r"):
        return "%"
    if channel == "msl":
        return "hPa"
    return "native"


def inverse_all(tensor, mean, std):
    """Inverse ERA5 normalization into the units reported by channel_unit."""
    physical = tensor.float() * std + mean
    # Geopotential -> geopotential height.
    physical[:, :13] = physical[:, :13] / 9.80665
    # Mean sea-level pressure Pa -> hPa.
    physical[:, 68:69] = physical[:, 68:69] / 100.0
    # Precipitation was clip -> log1p -> z-score and represents millimetres.
    physical[:, 69:70] = torch.expm1(physical[:, 69:70]).clamp_min(0.0)
    return physical


def animation_field(normalized, channel, mean, std):
    """Convert one normalized field to display units."""
    index = CHANNELS.index(channel)
    field = normalized.astype(np.float64) * std[index] + mean[index]
    if channel.startswith("z"):
        field = field / 9.80665
    elif channel.startswith("t"):
        field = field - 273.15
    elif channel == "msl":
        field = field / 100.0
    elif channel == "tp":
        field = np.maximum(np.expm1(field), 0.0)
    return field.astype(np.float32)


def collect_animation_frame(storage, prediction, target, valid_time, mean, std):
    for name, _, _, _ in ANIMATION_VARIABLES:
        if name == "wind10":
            u_index, v_index = CHANNELS.index("u10"), CHANNELS.index("v10")
            pred_u = animation_field(prediction[u_index], "u10", mean, std)
            pred_v = animation_field(prediction[v_index], "v10", mean, std)
            true_u = animation_field(target[u_index], "u10", mean, std)
            true_v = animation_field(target[v_index], "v10", mean, std)
            pred_field = np.hypot(pred_u, pred_v)
            true_field = np.hypot(true_u, true_v)
        else:
            index = CHANNELS.index(name)
            pred_field = animation_field(prediction[index], name, mean, std)
            true_field = animation_field(target[index], name, mean, std)
        storage[name].append((valid_time, true_field, pred_field))


def robust_limits(arrays, lower=2.0, upper=98.0):
    values = np.concatenate([array.reshape(-1) for array in arrays])
    vmin, vmax = np.nanpercentile(values, [lower, upper])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def save_animation(frames, variable, title, unit, cmap, lat, lon, output, fps):
    truth = [frame[1] for frame in frames]
    prediction = [frame[2] for frame in frames]
    vmin, vmax = robust_limits(truth + prediction)
    errors = [pred - obs for pred, obs in zip(prediction, truth)]
    error_limit = float(np.nanpercentile(np.abs(np.stack(errors)), 98.0))
    error_limit = max(error_limit, 1e-6)
    lat_weight = np.cos(np.deg2rad(lat))[:, None]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    extent = [float(lon[0]), float(lon[-1]), float(lat[0]), float(lat[-1])]
    images = [
        axes[0].imshow(truth[0], origin="lower", extent=extent, cmap=cmap,
                       vmin=vmin, vmax=vmax, aspect="auto"),
        axes[1].imshow(prediction[0], origin="lower", extent=extent, cmap=cmap,
                       vmin=vmin, vmax=vmax, aspect="auto"),
        axes[2].imshow(errors[0], origin="lower", extent=extent, cmap="RdBu_r",
                       vmin=-error_limit, vmax=error_limit, aspect="auto"),
    ]
    axes[0].set_title("ERA5")
    axes[1].set_title("Forecast")
    axes[2].set_title("Forecast - ERA5")
    for axis in axes:
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
    fig.colorbar(images[0], ax=axes[:2], shrink=0.85, label=unit)
    fig.colorbar(images[2], ax=axes[2], shrink=0.85, label=unit)
    heading = fig.suptitle("")

    def update(frame_index):
        valid_time, observed, forecast = frames[frame_index]
        difference = forecast - observed
        denominator = np.broadcast_to(lat_weight, observed.shape).sum()
        mae = float((np.abs(difference) * lat_weight).sum() / denominator)
        images[0].set_data(observed)
        images[1].set_data(forecast)
        images[2].set_data(difference)
        heading.set_text(
            f"{title} | valid {valid_time:%Y-%m-%d %H:%M UTC} | "
            f"weighted MAE={mae:.3f} {unit}"
        )
        return (*images, heading)

    movie = animation.FuncAnimation(fig, update, frames=len(frames), interval=500, blit=False)
    movie.save(output, writer=animation.PillowWriter(fps=fps), dpi=110)
    plt.close(fig)


def write_metrics(output_dir, dataset, sample_count, norm_sum, physical_sum,
                  persistence_norm_sum, persistence_physical_sum, denominator):
    normalized = norm_sum / denominator
    physical = physical_sum / denominator
    persistence_normalized = persistence_norm_sum / denominator
    persistence_physical = persistence_physical_sum / denominator
    overall = float(norm_sum.sum() / denominator.sum())
    persistence_overall = float(persistence_norm_sum.sum() / denominator.sum())
    improvement = 100.0 * (persistence_overall - overall) / max(persistence_overall, 1e-12)

    rows = []
    for index, channel in enumerate(CHANNELS):
        base = float(persistence_physical[index])
        model = float(physical[index])
        rows.append({
            "channel": channel,
            "unit": channel_unit(channel),
            "weighted_mae_normalized": float(normalized[index]),
            "persistence_mae_normalized": float(persistence_normalized[index]),
            "weighted_mae_physical": model,
            "persistence_mae_physical": base,
            "improvement_vs_persistence_percent": 100.0 * (base - model) / max(base, 1e-12),
        })
    with open(output_dir / "weighted_mae_by_channel.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "test_split": [str(dataset.analysis_times[0]), str(dataset.analysis_times[-1] + pd.Timedelta(hours=6))],
        "samples": sample_count,
        "overall_latitude_weighted_mae_normalized": overall,
        "persistence_latitude_weighted_mae_normalized": persistence_overall,
        "improvement_vs_persistence_percent": improvement,
        "note": "Physical MAE is reported per channel; averaging different physical units is not meaningful.",
    }
    with open(output_dir / "weighted_mae_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def main():
    args = parse_args()
    config = load_config(args.config)
    data_config = resolve_data_paths(args.config, config["data"])
    dataset = make_dataset(data_config, "test")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("config", config).get("model", config["model"])
    model = DAForecastNet(**model_config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    stats_dir = Path(data_config["mean_std_dir"])
    mean_np = np.load(stats_dir / "mean_era5.npy").astype(np.float32).reshape(-1)
    std_np = np.load(stats_dir / "std_era5.npy").astype(np.float32).reshape(-1)
    if len(mean_np) != 70 or len(std_np) != 70:
        raise ValueError("ERA5 mean/std must contain exactly 70 channels")
    mean = torch.as_tensor(mean_np, device=device).view(1, -1, 1, 1)
    std = torch.as_tensor(std_np, device=device).view(1, -1, 1, 1)
    lat_weight = torch.cos(torch.deg2rad(torch.as_tensor(dataset.lat, device=device)))
    lat_weight = (lat_weight / lat_weight.mean()).view(1, 1, -1, 1)

    sums = {
        name: torch.zeros(70, dtype=torch.float64, device=device)
        for name in ("norm", "physical", "persistence_norm", "persistence_physical", "denominator")
    }
    animation_start = pd.Timestamp(args.animation_start)
    animation_end = pd.Timestamp(args.animation_end)
    frames = {spec[0]: [] for spec in ANIMATION_VARIABLES}
    sample_offset = 0
    amp_enabled = device.type == "cuda"

    with torch.inference_mode():
        for batch_number, batch in enumerate(loader):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                prediction = model(
                    batch["era5"], batch["era_static"], batch["time_features"],
                    batch["obs_ztd"], batch["obs_mask"], batch["obs_static"]
                )
            target = batch["target"]
            persistence = batch["era5"][:, -1]
            valid = batch["target_mask"].to(torch.float32)
            weights = valid * lat_weight
            reduction_dims = (0, 2, 3)
            sums["denominator"] += weights.sum(reduction_dims, dtype=torch.float64)
            sums["norm"] += ((prediction.float() - target).abs() * weights).sum(
                reduction_dims, dtype=torch.float64
            )
            sums["persistence_norm"] += ((persistence - target).abs() * weights).sum(
                reduction_dims, dtype=torch.float64
            )
            prediction_physical = inverse_all(prediction, mean, std)
            target_physical = inverse_all(target, mean, std)
            persistence_physical = inverse_all(persistence, mean, std)
            sums["physical"] += ((prediction_physical - target_physical).abs() * weights).sum(
                reduction_dims, dtype=torch.float64
            )
            sums["persistence_physical"] += (
                (persistence_physical - target_physical).abs() * weights
            ).sum(reduction_dims, dtype=torch.float64)

            if not args.skip_animations:
                prediction_cpu = prediction.float().cpu().numpy()
                target_cpu = target.cpu().numpy()
                for item in range(prediction.shape[0]):
                    valid_time = dataset.sample_time(sample_offset + item) + pd.Timedelta(hours=6)
                    if animation_start <= valid_time <= animation_end:
                        collect_animation_frame(
                            frames, prediction_cpu[item], target_cpu[item], valid_time,
                            mean_np, std_np
                        )
            sample_offset += prediction.shape[0]
            if (batch_number + 1) % 20 == 0:
                print(f"evaluated {sample_offset}/{len(dataset)} samples", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = write_metrics(
        output_dir, dataset, sample_offset,
        sums["norm"].cpu().numpy(), sums["physical"].cpu().numpy(),
        sums["persistence_norm"].cpu().numpy(),
        sums["persistence_physical"].cpu().numpy(), sums["denominator"].cpu().numpy(),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if not args.skip_animations:
        animation_dir = output_dir / "animations"
        animation_dir.mkdir(exist_ok=True)
        for name, title, unit, cmap in ANIMATION_VARIABLES:
            if not frames[name]:
                raise RuntimeError(f"no animation frames selected for {name}")
            output = animation_dir / f"{name}_20250501_20250505.gif"
            print(f"writing {output} ({len(frames[name])} frames)", flush=True)
            save_animation(
                frames[name], name, title, unit, cmap,
                dataset.lat, dataset.lon, output, args.animation_fps
            )


if __name__ == "__main__":
    main()
