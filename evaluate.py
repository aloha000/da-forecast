"""Evaluate a checkpoint and make physical-unit weather animations."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from data import DAForecastDataset
from model import DAForecastNet
from train import (
    autocast_settings, load_config, make_dataset, resolve_data_paths,
    setup_distributed, wrap_fsdp,
)


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
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--output-dir", default="work_dirs/da_forecast_6h_40epoch_halo_only/test"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="per-GPU batch size (global forecast evaluation is memory intensive)",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--animation-start", default="2025-05-01T00:00:00")
    parser.add_argument("--animation-end", default="2025-05-05T23:59:59")
    parser.add_argument("--animation-fps", type=int, default=2)
    parser.add_argument("--skip-animations", action="store_true")
    return parser.parse_args()


class DistributedEvalSampler(Sampler):
    """Shard evaluation indices evenly and mark padding as invalid."""

    def __init__(self, dataset_size: int, rank: int, world_size: int):
        self.dataset_size = int(dataset_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.num_samples = (self.dataset_size + self.world_size - 1) // self.world_size
        self.total_size = self.num_samples * self.world_size

    def __iter__(self):
        indices = [(index, True) for index in range(self.dataset_size)]
        if self.total_size > self.dataset_size:
            padding = self.total_size - self.dataset_size
            if not indices:
                return iter(())
            indices.extend((index % self.dataset_size, False) for index in range(padding))
        return iter(indices[self.rank:self.total_size:self.world_size])

    def __len__(self):
        return self.num_samples


class IndexedEvalDataset(Dataset):
    """Attach the original index and validity bit to each evaluation sample."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        index, is_valid = item if isinstance(item, tuple) else (item, True)
        sample = dict(self.dataset[index])
        sample["_eval_index"] = torch.tensor(index, dtype=torch.int64)
        sample["_eval_valid"] = torch.tensor(is_valid, dtype=torch.bool)
        return sample


def load_distributed_model(checkpoint, checkpoint_config, model_config, device, rank):
    """Strictly load on rank 0 BEFORE FSDP broadcasts and shards the model."""
    model = DAForecastNet(**model_config)
    load_error = [None]
    if rank == 0:
        try:
            model.load_state_dict(checkpoint["model"], strict=True)
        except Exception as error:
            load_error[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(load_error, src=0)
    if load_error[0] is not None:
        raise RuntimeError(f"checkpoint model load failed: {load_error[0]}")
    train_config = dict(checkpoint_config["train"])
    # This is required even if the saved training config disabled synchronization.
    train_config["fsdp"] = {
        **train_config.get("fsdp", {}), "sync_module_states": True,
    }
    model = wrap_fsdp(model, device, train_config)
    model.eval()
    return model


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


def save_animation(
    frames, variable, title, unit, cmap, lat, lon, halo_mask, output, fps
):
    truth = [frame[1] for frame in frames]
    prediction = [frame[2] for frame in frames]
    vmin, vmax = robust_limits(truth + prediction)
    errors = [pred - obs for pred, obs in zip(prediction, truth)]
    error_limit = float(np.nanpercentile(np.abs(np.stack(errors)), 98.0))
    error_limit = max(error_limit, 1e-6)
    lat_weight = np.cos(np.deg2rad(lat))[:, None] * halo_mask

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
            f"halo weighted MAE={mae:.3f} {unit}"
        )
        return (*images, heading)

    movie = animation.FuncAnimation(fig, update, frames=len(frames), interval=500, blit=False)
    movie.save(output, writer=animation.PillowWriter(fps=fps), dpi=110)
    plt.close(fig)


def write_metrics(
    output_dir, dataset, sample_count, sums, loss_mode, halo_weight, halo_radius,
    provenance=None,
):
    global_normalized = sums["global_norm"] / sums["global_denominator"]
    global_physical = sums["global_physical"] / sums["global_denominator"]
    global_persistence_normalized = (
        sums["global_persistence_norm"] / sums["global_denominator"]
    )
    global_persistence_physical = (
        sums["global_persistence_physical"] / sums["global_denominator"]
    )
    halo_normalized = sums["halo_norm"] / sums["halo_denominator"]
    halo_physical = sums["halo_physical"] / sums["halo_denominator"]
    halo_persistence_normalized = (
        sums["halo_persistence_norm"] / sums["halo_denominator"]
    )
    halo_persistence_physical = (
        sums["halo_persistence_physical"] / sums["halo_denominator"]
    )

    global_overall = float(
        sums["global_norm"].sum() / sums["global_denominator"].sum()
    )
    global_persistence = float(
        sums["global_persistence_norm"].sum()
        / sums["global_denominator"].sum()
    )
    halo_overall = float(
        sums["halo_norm"].sum() / sums["halo_denominator"].sum()
    )
    halo_persistence = float(
        sums["halo_persistence_norm"].sum() / sums["halo_denominator"].sum()
    )
    if loss_mode == "halo_only":
        selected, selected_persistence = halo_overall, halo_persistence
    elif loss_mode == "global_only":
        selected, selected_persistence = global_overall, global_persistence
    elif loss_mode == "global_plus_halo":
        selected = global_overall + float(halo_weight) * halo_overall
        selected_persistence = (
            global_persistence + float(halo_weight) * halo_persistence
        )
    else:
        raise ValueError(f"unsupported loss_mode={loss_mode!r}")
    selected_improvement = 100.0 * (selected_persistence - selected) / max(
        selected_persistence, 1e-12
    )

    rows = []
    for index, channel in enumerate(CHANNELS):
        global_base = float(global_persistence_physical[index])
        global_model = float(global_physical[index])
        halo_base = float(halo_persistence_physical[index])
        halo_model = float(halo_physical[index])
        rows.append({
            "channel": channel,
            "unit": channel_unit(channel),
            "global_weighted_mae_normalized": float(global_normalized[index]),
            "global_persistence_mae_normalized": float(
                global_persistence_normalized[index]
            ),
            "halo_weighted_mae_normalized": float(halo_normalized[index]),
            "halo_persistence_mae_normalized": float(
                halo_persistence_normalized[index]
            ),
            "global_weighted_mae_physical": global_model,
            "global_persistence_mae_physical": global_base,
            "global_improvement_percent": 100.0
            * (global_base - global_model)
            / max(global_base, 1e-12),
            "halo_weighted_mae_physical": halo_model,
            "halo_persistence_mae_physical": halo_base,
            "halo_improvement_percent": 100.0
            * (halo_base - halo_model)
            / max(halo_base, 1e-12),
        })
    with open(output_dir / "weighted_mae_by_channel.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    physical_fields = [
        "channel",
        "unit",
        "global_weighted_mae_physical",
        "global_persistence_mae_physical",
        "global_improvement_percent",
        "halo_weighted_mae_physical",
        "halo_persistence_mae_physical",
        "halo_improvement_percent",
    ]
    with open(
        output_dir / "physical_mae_by_channel.csv", "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=physical_fields)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in physical_fields} for row in rows
        )

    split_name = (provenance or {}).get("split", "test")
    summary = {
        f"{split_name}_split": [str(dataset.analysis_times[0]), str(dataset.analysis_times[-1] + pd.Timedelta(hours=6))],
        "samples": sample_count,
        "loss_mode": loss_mode,
        "halo_radius_cells": int(halo_radius),
        "latitude_weighted_mae_normalized": selected,
        "persistence_latitude_weighted_mae_normalized": selected_persistence,
        "improvement_vs_persistence_percent": selected_improvement,
        "global_latitude_weighted_mae_normalized": global_overall,
        "global_persistence_latitude_weighted_mae_normalized": global_persistence,
        "halo_latitude_weighted_mae_normalized": halo_overall,
        "halo_persistence_latitude_weighted_mae_normalized": halo_persistence,
        "note": "Physical MAE is reported per channel; averaging different physical units is not meaningful.",
    }
    if provenance is not None:
        summary["evaluation"] = provenance
    with open(output_dir / "weighted_mae_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def main():
    args = parse_args()
    config = load_config(args.config)
    data_config = resolve_data_paths(args.config, config["data"])
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        rank, _, world_size, device = setup_distributed()
        distributed = True
        object_group = (
            dist.new_group(backend="gloo") if not args.skip_animations else None
        )
    else:
        rank, world_size = 0, 1
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        distributed = False
        object_group = None

    # Rank 0 loads the CPU model before FSDP synchronizes and shards it.
    checkpoint = None
    checkpoint_metadata = [None]
    if rank == 0:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        saved_config = checkpoint.get("config", config)
        checkpoint_metadata[0] = {
            "model": saved_config.get("model", config["model"]),
            "train": saved_config.get("train", config["train"]),
            "epoch": checkpoint.get("epoch"),
            "step": checkpoint.get("step"),
            "best_val": checkpoint.get("best_val"),
        }
    if distributed:
        dist.broadcast_object_list(checkpoint_metadata, src=0)
    checkpoint_config = checkpoint_metadata[0]
    model_config = checkpoint_config["model"]
    checkpoint_train = checkpoint_config["train"]

    dataset = make_dataset(data_config, args.split)
    eval_dataset = IndexedEvalDataset(dataset)
    eval_sampler = DistributedEvalSampler(len(eval_dataset), rank, world_size)
    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        sampler=eval_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    loss_mode = checkpoint_train.get("loss_mode", "global_plus_halo")
    halo_weight = float(checkpoint_train.get("halo_weight", 0.5))
    halo_radius = int(checkpoint_train.get("halo_radius", 3))
    if distributed:
        model = load_distributed_model(
            checkpoint, checkpoint_config, model_config, device, rank
        )
        del checkpoint
    else:
        model = DAForecastNet(**model_config).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        del checkpoint

    provenance = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch_zero_based": checkpoint_config["epoch"],
        "checkpoint_step": checkpoint_config["step"],
        "checkpoint_best_val": checkpoint_config["best_val"],
        "split": args.split,
        "world_size": world_size,
        "weight_loading": "rank0_load_before_fsdp_sync" if distributed else "strict_full_state",
        "amp": checkpoint_train.get("amp", "bf16"),
    }
    if rank == 0:
        print("Checkpoint loaded: " + json.dumps(provenance), flush=True)

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
        for name in (
            "global_norm", "global_physical", "global_persistence_norm",
            "global_persistence_physical", "global_denominator",
            "halo_norm", "halo_physical", "halo_persistence_norm",
            "halo_persistence_physical", "halo_denominator",
        )
    }
    station_mask = torch.as_tensor(
        dataset.obs_static[:1], dtype=torch.float32, device=device
    ).unsqueeze(0)
    halo_mask = F.max_pool2d(
        station_mask,
        kernel_size=2 * halo_radius + 1,
        stride=1,
        padding=halo_radius,
    )
    animation_start = pd.Timestamp(args.animation_start)
    animation_end = pd.Timestamp(args.animation_end)
    frames = {spec[0]: [] for spec in ANIMATION_VARIABLES}
    local_sample_count = 0
    amp_enabled, amp_dtype = autocast_settings(device, provenance["amp"])

    with torch.inference_mode():
        for batch_number, batch in enumerate(loader):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                prediction = model(
                    batch["era5"], batch["era_static"], batch["time_features"],
                    batch["obs_ztd"], batch["obs_mask"], batch["obs_static"]
                )
            target = batch["target"]
            persistence = batch["era5"][:, -1]
            eval_valid = batch["_eval_valid"].to(torch.float32).view(-1, 1, 1, 1)
            valid = batch["target_mask"].to(torch.float32) * eval_valid
            global_weights = valid * lat_weight
            halo_weights = global_weights * halo_mask
            reduction_dims = (0, 2, 3)
            sums["global_denominator"] += global_weights.sum(
                reduction_dims, dtype=torch.float64
            )
            sums["halo_denominator"] += halo_weights.sum(
                reduction_dims, dtype=torch.float64
            )
            normalized_error = (prediction.float() - target).abs()
            persistence_normalized_error = (persistence - target).abs()
            sums["global_norm"] += (normalized_error * global_weights).sum(
                reduction_dims, dtype=torch.float64
            )
            sums["halo_norm"] += (normalized_error * halo_weights).sum(
                reduction_dims, dtype=torch.float64
            )
            sums["global_persistence_norm"] += (
                persistence_normalized_error * global_weights
            ).sum(reduction_dims, dtype=torch.float64)
            sums["halo_persistence_norm"] += (
                persistence_normalized_error * halo_weights
            ).sum(reduction_dims, dtype=torch.float64)
            prediction_physical = inverse_all(prediction, mean, std)
            target_physical = inverse_all(target, mean, std)
            persistence_physical = inverse_all(persistence, mean, std)
            physical_error = (prediction_physical - target_physical).abs()
            persistence_physical_error = (
                persistence_physical - target_physical
            ).abs()
            sums["global_physical"] += (physical_error * global_weights).sum(
                reduction_dims, dtype=torch.float64
            )
            sums["halo_physical"] += (physical_error * halo_weights).sum(
                reduction_dims, dtype=torch.float64
            )
            sums["global_persistence_physical"] += (
                persistence_physical_error * global_weights
            ).sum(reduction_dims, dtype=torch.float64)
            sums["halo_persistence_physical"] += (
                persistence_physical_error * halo_weights
            ).sum(reduction_dims, dtype=torch.float64)

            if not args.skip_animations:
                prediction_cpu = prediction.float().cpu().numpy()
                target_cpu = target.cpu().numpy()
                for item in range(prediction.shape[0]):
                    if not bool(batch["_eval_valid"][item].item()):
                        continue
                    sample_index = int(batch["_eval_index"][item].item())
                    valid_time = dataset.sample_time(sample_index) + pd.Timedelta(hours=6)
                    if animation_start <= valid_time <= animation_end:
                        collect_animation_frame(
                            frames, prediction_cpu[item], target_cpu[item], valid_time,
                            mean_np, std_np
                        )
            local_sample_count += int(batch["_eval_valid"].sum().item())
            if rank == 0:
                print(
                    f"rank 0 evaluated {local_sample_count}/{len(eval_sampler)} "
                    f"local samples; world_size={world_size}",
                    flush=True,
                )

    if distributed:
        for value in sums.values():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        sample_count_tensor = torch.tensor(
            local_sample_count, dtype=torch.int64, device=device
        )
        dist.all_reduce(sample_count_tensor, op=dist.ReduceOp.SUM)
        sample_count = int(sample_count_tensor.item())
    else:
        sample_count = local_sample_count

    if not args.skip_animations and distributed:
        gathered_frames = [None] * world_size if rank == 0 else None
        # Use a CPU/Gloo group: routing multi-day animation arrays through the
        # NCCL group would temporarily consume GPU memory for pickled frames.
        dist.gather_object(frames, gathered_frames, dst=0, group=object_group)
        if rank == 0:
            merged_frames = {spec[0]: [] for spec in ANIMATION_VARIABLES}
            for rank_frames in gathered_frames:
                for name in merged_frames:
                    merged_frames[name].extend(rank_frames[name])
            frames = merged_frames
            for name in frames:
                frames[name].sort(key=lambda frame: frame[0])

    if rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(args.output_dir)

    numpy_sums = {name: value.cpu().numpy() for name, value in sums.items()}
    if rank == 0:
        summary = write_metrics(
            output_dir, dataset, sample_count, numpy_sums,
            loss_mode, halo_weight, halo_radius, provenance=provenance
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    if rank == 0 and not args.skip_animations:
        animation_dir = output_dir / "animations"
        animation_dir.mkdir(exist_ok=True)
        for name, title, unit, cmap in ANIMATION_VARIABLES:
            if not frames[name]:
                raise RuntimeError(f"no animation frames selected for {name}")
            output = animation_dir / f"{name}_20250501_20250505.gif"
            print(f"writing {output} ({len(frames[name])} frames)", flush=True)
            save_animation(
                frames[name], name, title, unit, cmap,
                dataset.lat, dataset.lon, halo_mask[0, 0].cpu().numpy(),
                output, args.animation_fps
            )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
