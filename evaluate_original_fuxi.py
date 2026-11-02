"""Original FuXi baseline: independent +6h or two-hop +12h autoregression.

+12h uses exactly the rollout Zarr's split origins, never its DA predictions.
Weights remain unchanged. Writes small MAE CSV/JSON files, no forecast cache.
"""
from __future__ import annotations

import argparse
import csv
from datetime import timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Subset
import zarr

from data.da_dataset import _decode_time, build_static_fields, time_features
from data.rollout_fields import HOUR_NS, SCHEMA, SPLITS, open_store, times_ns
from data.rollout_identity import data_identity, file_identity
from evaluate import CHANNELS, channel_unit, inverse_all, DistributedEvalSampler, IndexedEvalDataset
from model.fuxi import UTransformer
from model.fuxi_pretrained import _PICKLE_COMPAT
from train import autocast_settings, load_config, resolve_data_paths, setup_distributed, wrap_fsdp


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/train_da_forecast_12h_fuxi.json")
    p.add_argument("--checkpoint", default="model/fuxi_short.pth")
    p.add_argument("--lead-hours", type=int, choices=(6, 12), default=None,
                   help="defaults to data.forecast_lead_hours in config")
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--era5", default=None, help="optional ERA5 path override")
    p.add_argument("--conds", default=None, help="optional static field path override")
    p.add_argument("--start", default=None, help="6h mode only")
    p.add_argument("--end", default=None, help="6h mode only, exclusive")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--batch-size", type=int, default=1, help="per GPU")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=0, help="first N origins; 0 means full split")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--output", default=None, help="optional extra legacy summary JSON")
    p.add_argument("--plan", action="store_true", help="metadata only; no model, GPU, or writes")
    args = p.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.max_samples < 0:
        p.error("invalid batch size, worker count, or sample limit")
    return args


def time_encoding(timestamp, device):
    return torch.from_numpy(time_features(pd.Timestamp(timestamp))).unsqueeze(0).to(device)


class OriginalInputs(Dataset):
    def __init__(self, data, lead_hours, split="test", start=None, end=None):
        self.era_path = data["era5_path"]
        self.lead_hours, self.split = lead_hours, split
        self._era = None
        era = zarr.open_group(self.era_path, mode="r")
        if era["z"].ndim != 4 or era["z"].shape[1] != 70:
            raise ValueError("expected normalized ERA5 z[time,70,lat,lon]")
        era_times = _decode_time(era, "time")
        if not era_times.is_unique or not era_times.is_monotonic_increasing:
            raise ValueError("ERA5 timestamps must be unique and increasing")
        lookup = {int(t.value): i for i, t in enumerate(era_times)}
        self.lat = np.asarray(era["lat"][:], dtype=np.float32)
        self.lon = np.asarray(era["lon"][:], dtype=np.float32)
        self.rollout_identity = None
        if lead_hours == 12:
            if start or end:
                raise ValueError("12h uses exact rollout split origins; use --max-samples for a smoke test")
            root = open_store(data["rollout_zarr"])
            if root.attrs.get("schema") != SCHEMA or not root.attrs.get("initialized"):
                raise ValueError("rollout Zarr metadata is not initialized")
            self.rollout_identity = dict(root.attrs["identity"])
            if self.rollout_identity["data"] != data_identity(data):
                raise ValueError("ERA5/static/normalization configuration differs from rollout Zarr")
            origins = np.asarray(root["init_time"][:], dtype=np.int64)
            codes = np.asarray(root["split_code"][:])
            if codes.shape != origins.shape or not np.isin(codes, [0, 1, 2]).all():
                raise ValueError("invalid split codes")
            if np.any(np.diff(origins) <= 0):
                raise ValueError("rollout origins must be ordered and unique")
            expected_hash = self.rollout_identity.get("origins_sha256")
            if expected_hash and hashlib.sha256(origins.astype("<i8").tobytes()).hexdigest() != expected_hash:
                raise ValueError("rollout origin fingerprint mismatch")
            self.analysis_times = pd.DatetimeIndex(pd.to_datetime(origins[codes == SPLITS.index(split)], unit="ns"))
            if not np.array_equal(root["lat"][:], self.lat) or not np.array_equal(root["lon"][:], self.lon):
                raise ValueError("rollout and original ERA5 coordinates differ")
            # Same aligned static fields used by DA. No OBS fields/forecast chunks read.
            self.const = np.asarray(root["era_static"][:], dtype=np.float32)
        else:
            begin, finish = map(pd.Timestamp, (start or data[split + "_split"][0],
                                              end or data[split + "_split"][1]))
            self.analysis_times = pd.DatetimeIndex([
                t for t in era_times if begin <= t and t + pd.Timedelta(hours=6) < finish
                and t.value - 6 * HOUR_NS in lookup and t.value + 6 * HOUR_NS in lookup
            ])
            self.const, _ = build_static_fields(
                data["conds_path"], data["global_lat_path"], data["global_lon_path"],
                self.lat, self.lon, np.zeros((len(self.lat), len(self.lon)), dtype=np.float32),
            )
        if not len(self.analysis_times):
            raise ValueError("no evaluation origins")
        if lead_hours == 12:
            begin, finish = map(pd.Timestamp, data[split + "_split"])
            if (self.analysis_times < begin).any() or (self.analysis_times + pd.Timedelta(hours=12) >= finish).any():
                raise ValueError("rollout origin/target crosses split boundary")
        if self.const.shape != (5, len(self.lat), len(self.lon)) or not np.isfinite(self.const).all():
            raise ValueError("expected five finite aligned static fields")
        self.rows = []
        for t in self.analysis_times:
            try:
                self.rows.append(tuple(lookup[t.value + hours * HOUR_NS] for hours in (-6, 0, lead_hours)))
            except KeyError as exc:
                raise ValueError(f"ERA5 missing required timestamp for {t}; refusing to drop a paired sample") from exc

    def __len__(self):
        return len(self.analysis_times)

    def __getstate__(self):
        return {**self.__dict__, "_era": None}

    def __getitem__(self, index):
        if self._era is None:
            self._era = zarr.open_group(self.era_path, mode="r")["z"]
        before, current, future = self.rows[index]
        # Explicit timestamps, no contiguous slice that might read ERA5(T+6).
        inputs = np.stack([self._era[before], self._era[current]]).astype(np.float32)
        target = np.asarray(self._era[future], dtype=np.float32)
        valid = np.isfinite(target)
        inputs = np.nan_to_num(inputs, nan=0., posinf=0., neginf=0.)
        t = self.analysis_times[index]
        return {
            "era5": torch.from_numpy(inputs),
            "target": torch.from_numpy(np.where(valid, target, 0).astype(np.float32)),
            "target_mask": torch.from_numpy(valid.astype(np.float32)),
            "time0": torch.from_numpy(time_features(t)),
            "time6": torch.from_numpy(time_features(t + pd.Timedelta(hours=6))),
        }


def load_model(path, device, rank=0, world_size=1, amp="bf16"):
    model = UTransformer(
        in_chans=75, out_chans=70, in_frames=2, image_size=(720, 1440),
        window_size=9, patch_size=4, down_times=1, embed_dim=1536,
        num_heads=24, depths=[12, 12, 12, 12],
    )
    error = [None]
    if rank == 0:
        try:
            checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False,
                                    pickle_module=_PICKLE_COMPAT)
            state = {key.removeprefix("decoder."): value for key, value in checkpoint["model"].items()
                     if key.startswith("decoder.")}
            model.load_state_dict(state, strict=True)
            del checkpoint
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    if world_size > 1:
        dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(f"original FuXi load failed: {error[0]}")
    if world_size > 1:
        model = wrap_fsdp(model, device, {
            "amp": amp, "fsdp": {"sync_module_states": True, "forward_prefetch": False},
        })
    else:
        model = model.to(device)
    return model.eval()


def forecast(model, batch, const, lead_hours, device, amp):
    enabled, dtype = autocast_settings(device, amp)
    inputs = batch["era5"].float()
    for hop in range(lead_hours // 6):
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
            residual = model(inputs, temb=batch["time0" if hop == 0 else "time6"], const=const)
        prediction = inputs[:, -1].float() + residual.float()
        if hop + 1 < lead_hours // 6:
            inputs = torch.stack((inputs[:, -1].float(), prediction), dim=1)
    return prediction


def metric_sums(prediction, batch, weights, mean, std):
    if not torch.isfinite(prediction).all():
        raise ValueError("nonfinite FuXi prediction")
    target, persistence = batch["target"], batch["era5"][:, -1]
    valid = batch["target_mask"] * batch["_eval_valid"].view(-1, 1, 1, 1)
    weight = valid * weights
    dims = (0, 2, 3)
    def weighted(error):
        error = torch.where(weight > 0, error, 0.)
        if not torch.isfinite(error).all():
            raise ValueError("nonfinite error on valid cells")
        return (error * weight).sum(dims, dtype=torch.float64)
    physical_target = inverse_all(target, mean, std)
    return torch.stack((
        weighted((prediction.float() - target).abs()),
        weighted((inverse_all(prediction, mean, std) - physical_target).abs()),
        weighted((persistence - target).abs()),
        weighted((inverse_all(persistence, mean, std) - physical_target).abs()),
        weight.sum(dims, dtype=torch.float64),
    ))


def write_results(output, sums, metadata, extra_output=None):
    if not np.isfinite(sums).all() or np.any(sums[4] <= 0):
        raise ValueError("invalid sums or a channel has no valid target cells")
    values = sums[:4] / sums[4]
    original = float(sums[0].sum() / sums[4].sum())
    persistence = float(sums[2].sum() / sums[4].sum())
    summary = {
        **metadata, "original_global_latitude_weighted_mae_normalized": original,
        "overall_latitude_weighted_mae_normalized": original,
        "persistence_latitude_weighted_mae_normalized": persistence,
        "note": "Unmodified original weights in both hops. Physical MAE is per-channel; TP is still 6h accumulation.",
    }
    rows = [dict(channel=c, unit=channel_unit(c),
                 original_mae_normalized=float(values[0, i]),
                 original_mae_physical=float(values[1, i]),
                 persistence_mae_normalized=float(values[2, i]),
                 persistence_mae_physical=float(values[3, i])) for i, c in enumerate(CHANNELS)]
    output = Path(output)
    csv_path = output / "channel_comparison.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    for path in (output / "summary.json", *([Path(extra_output)] if extra_output else [])):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    return summary


def sync_error(error, group, world_size):
    errors = [error]
    if world_size > 1:
        errors = [None] * world_size
        dist.all_gather_object(errors, error, group=group)
    if any(errors):
        raise RuntimeError("; ".join(f"rank{i}: {e}" for i, e in enumerate(errors) if e))


def main():
    args = parse_args()
    config = load_config(args.config)
    data = resolve_data_paths(args.config, config["data"])
    for option, key in ((args.era5, "era5_path"), (args.conds, "conds_path")):
        if option:
            data[key] = str(Path(option).resolve())
    lead = args.lead_hours or int(data.get("forecast_lead_hours", 6))
    if lead not in (6, 12):
        raise ValueError("only 6h and 12h are supported")
    dataset = OriginalInputs(data, lead, args.split, args.start, args.end)
    selected = min(args.max_samples, len(dataset)) if args.max_samples else len(dataset)
    origins = dataset.analysis_times[:selected]
    output = Path(args.output_dir or f"work_dirs/original_fuxi_{lead}h")
    metadata = {
        "checkpoint": file_identity(args.checkpoint), "forecast_lead_hours": lead,
        "samples": selected, "full_split_samples": len(dataset),
        "is_full_test": args.split == "test" and selected == len(dataset) and not (args.start or args.end),
        "split": args.split, "amp": args.amp,
        "init_time_first": str(origins[0]), "init_time_last": str(origins[-1]),
        "valid_time_first": str(origins[0] + pd.Timedelta(hours=lead)),
        "valid_time_last": str(origins[-1] + pd.Timedelta(hours=lead)),
        "test_origins_sha256": hashlib.sha256(times_ns(origins).astype("<i8").tobytes()).hexdigest(),
        "rollout_identity": dataset.rollout_identity,
        "era5_path": str(Path(data["era5_path"]).resolve()),
        "rollout_zarr": data.get("rollout_zarr") if lead == 12 else None,
        "observations": "none_in_either_hop", "weights": "original_unchanged",
    }
    if int(os.environ.get("RANK", 0)) == 0:
        print(f"Original FuXi +{lead}h: {selected}/{len(dataset)} {args.split} origins; "
              f"full_test={metadata['is_full_test']}; output={output}", flush=True)
    if args.plan:
        return

    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        rank, _, world_size, device = setup_distributed()
    else:
        rank, world_size = 0, 1
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        metadata["amp"] = "fp32"
    metadata["world_size"] = world_size
    control = dist.new_group(backend="gloo", timeout=timedelta(hours=2)) if world_size > 1 else None
    lock = None
    try:
        error = None
        if rank == 0:
            try:
                output.mkdir(parents=True, exist_ok=True)
                lock = (output / ".writer.lock").open("a+")
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if (output / "summary.json").exists():
                    raise ValueError("results already exist; choose a new output directory")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        sync_error(error, control, world_size)
        mean = torch.as_tensor(np.load(Path(data["mean_std_dir"]) / "mean_era5.npy"),
                               device=device, dtype=torch.float32).reshape(1, 70, 1, 1)
        std = torch.as_tensor(np.load(Path(data["mean_std_dir"]) / "std_era5.npy"),
                              device=device, dtype=torch.float32).reshape(1, 70, 1, 1)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError("normalization arrays must be finite with positive std")
        weights = torch.cos(torch.deg2rad(torch.as_tensor(dataset.lat, device=device))).clamp_min(0)
        weights = (weights / weights.mean()).reshape(1, 1, -1, 1)
        const = torch.as_tensor(dataset.const, device=device, dtype=torch.float32)
        model = load_model(args.checkpoint, device, rank, world_size, metadata["amp"])
        sampler = DistributedEvalSampler(selected, rank, world_size)
        loader = DataLoader(IndexedEvalDataset(Subset(dataset, range(selected))), sampler=sampler,
                            batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=device.type == "cuda")
        sums = torch.zeros((5, 70), device=device, dtype=torch.float64)
        count = torch.zeros((), device=device, dtype=torch.int64)
        iterator = iter(loader)
        with torch.inference_mode():
            for step in range(len(loader)):
                error = None
                try:
                    batch = {k: v.to(device, non_blocking=True) for k, v in next(iterator).items()}
                except Exception as exc:
                    error = f"input read failed: {exc}"
                sync_error(error, control, world_size)
                prediction = forecast(model, batch, const, lead, device, metadata["amp"])
                error = None
                try:
                    sums += metric_sums(prediction, batch, weights, mean, std)
                    count += batch["_eval_valid"].sum()
                except Exception as exc:
                    error = f"metrics failed: {exc}"
                sync_error(error, control, world_size)
                if rank == 0:
                    print(f"batch={step+1}/{len(loader)} rank0_samples={int(count)} "
                          f"local_mae={float(sums[0].sum()/sums[4].sum()):.6f}", flush=True)
        if world_size > 1:
            dist.all_reduce(sums)
            dist.all_reduce(count)
        if int(count) != selected:
            raise RuntimeError("distributed sample count mismatch")
        if rank == 0:
            summary = write_results(output, sums.cpu().numpy(), metadata, args.output)
            print(json.dumps({k: summary[k] for k in (
                "samples", "forecast_lead_hours", "is_full_test",
                "overall_latitude_weighted_mae_normalized", "persistence_latitude_weighted_mae_normalized",
            )}, indent=2), flush=True)
    finally:
        if lock is not None:
            lock.close()
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
