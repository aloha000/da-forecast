"""Build a standalone two-field Zarr for fixed-input +12h rollout training.

Source hop uses real or zero-ZTD OBS at T, explicitly selected in config.
No NPZ cache and no modification of source ERA5.
Run --plan first; --max-samples is a resumable smoke run, not a full dataset.
"""
import argparse
from datetime import timedelta
import fcntl
import hashlib
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
import zarr

from data.da_dataset import DAForecastDataset, time_features
from data.rollout_identity import data_identity, file_identity
from data.rollout_fields import (
    HOUR_NS, SPLITS, SCHEMA, axes_for_datasets, initialize_store, open_store, write_field, times_ns,
    source_observation_policy,
)
from evaluate import DistributedEvalSampler, IndexedEvalDataset, load_distributed_model
from model import DAForecastNet
from model.ztd_rollout import next_observations, check_ztd_config
from model.delay_components import COMPONENT_INPUTS, next_component_inputs
from data.delay_components import component_inputs, read_components
from train import autocast_settings, load_config, resolve_data_paths, setup_distributed


class SourceInputs(DAForecastDataset):
    """Select origins with T+12 labels, but read ONLY past inputs for the +6 hop."""
    def __init__(self, *, split, **kwargs):
        super().__init__(split=split, **kwargs)
        end = pd.Timestamp(split[1]).value
        self.analysis_times = pd.DatetimeIndex([
            t for t in self.analysis_times
            if t.value + 12 * HOUR_NS < end
            and t.value + 12 * HOUR_NS in self.label_lookup
        ])
        if not len(self.analysis_times):
            raise ValueError(f"no valid 12h origins in split {split}")

    def __getitem__(self, index):
        timestamp = self.sample_time(index)
        self._open()
        before = self.label_lookup[timestamp.value - 6 * HOUR_NS]
        current = self.label_lookup[timestamp.value]
        if current != before + 1:
            raise ValueError("source ERA5 history must be contiguous")
        era = np.asarray(self._era_group["z"][before:current + 1], dtype=np.float32)
        era = np.nan_to_num(era, nan=0.0, posinf=0.0, neginf=0.0)
        obs, mask = self._read_observations(timestamp)
        result = {
            "era5": torch.from_numpy(era),
            "era_static": torch.from_numpy(self.era_static),
            "time_features": torch.from_numpy(time_features(timestamp)),
            "obs_ztd": torch.from_numpy(obs[:, None]),
            "obs_mask": torch.from_numpy(mask[:, None].astype(np.float32)),
            "obs_static": torch.from_numpy(self.obs_static),
        }
        if self.predict_delay_components:
            result.update(component_inputs(self, timestamp))
        return result


def source_datasets(data):
    kwargs = {k: data[k] for k in (
        "era5_path", "ztd_path", "conds_path", "global_lat_path", "global_lon_path",
        "obs_history_hours", "obs_interval_minutes",
    )}
    if data.get("predict_delay_components", False):
        kwargs.update({k: data[k] for k in ("predict_ztd", "predict_delay_components", "component_loss_scales",
                      "use_delay_physics", "station_height_path", "station_height_datum", "mean_std_dir") if k in data})
    return {s: SourceInputs(split=data[s + "_split"], **kwargs) for s in SPLITS}


def source_metadata(saved, data, amp):
    if int(saved.get("data", {}).get("forecast_lead_hours", 6)) != 6:
        raise ValueError("source checkpoint must be +6h")
    model_config = dict(saved["model"])
    joint = data.get("rollout_observations", "branch_removed") in ("predicted_ztd_at_T", "predicted_components_at_T")
    if bool(model_config.get("predict_delay_components", False)) != bool(data.get("predict_delay_components", False)):
        raise ValueError("source checkpoint component mode mismatch")
    if data.get("predict_delay_components", False):
        check_ztd_config(data, model_config)
        for key in ("station_height_path", "station_height_datum", "component_loss_scales"):
            if saved.get("data", {}).get(key) != data.get(key) and key != "station_height_path":
                raise ValueError(f"source checkpoint {key} differs from generation config")
    if joint and (not model_config.get("predict_ztd", False) or model_config.get("zero_ztd", False)):
        raise ValueError("predicted-ZTD rollout requires a trained +6h predict_ztd checkpoint, zero_ztd=false")
    zero_ztd = source_observation_policy(data) == "zero_ztd_history_at_T"
    if (not model_config.get("use_obs", True)
            or bool(model_config.get("zero_ztd", False)) != zero_ztd):
        expected = "zero ZTD" if zero_ztd else "real OBS"
        raise ValueError(f"source checkpoint must use {expected}: use_obs=true, zero_ztd={zero_ztd}")
    for key in ("obs_history_hours", "obs_interval_minutes"):
        if key in saved.get("data", {}) and saved["data"][key] != data[key]:
            raise ValueError(f"source checkpoint {key} differs from generation config")
    train_config = {**saved.get("train", {}), "amp": amp}
    train_config["fsdp"] = {**train_config.get("fsdp", {}), "forward_prefetch": False}
    return {"model": model_config, "train": train_config}


def pending_rows(root, datasets, splits, max_samples):
    times, origins, _ = axes_for_datasets(datasets)
    selected = {}
    for s in splits:
        ds = datasets[s]
        indices = np.arange(min(len(ds), max_samples) if max_samples else len(ds))
        global_indices = np.searchsorted(origins, times_ns(ds.analysis_times)[indices])
        ready = np.asarray(root["forecast_complete"][:], dtype=bool) if root is not None else np.zeros(len(origins), bool)
        selected[s] = indices[~ready[global_indices]].tolist()
    # Copy truth even when forecast rows are already complete.
    wanted = np.concatenate([
        times_ns(datasets[s].analysis_times)[:max_samples or len(datasets[s])] for s in splits
    ])
    era_indices = np.searchsorted(times, np.unique(np.concatenate((wanted, wanted + 12 * HOUR_NS))))
    ready = np.asarray(root["era_complete"][:], dtype=bool) if root is not None else np.zeros(len(times), bool)
    return era_indices[~ready[era_indices]].tolist(), selected


def synchronize_error(message, group, world_size):
    errors = [message]
    if world_size > 1:
        errors = [None] * world_size
        dist.all_gather_object(errors, message, group=group)
    failures = [f"rank{i}: {e}" for i, e in enumerate(errors) if e]
    if failures:
        raise RuntimeError("; ".join(failures))


def copy_truth(root, source, indices, times, rank, world_size):
    # Do not initialize or read NGL here.
    array = zarr.open_group(source.era5_path, mode="r")["z"]
    local = indices[rank::world_size]
    for step, index in enumerate(local):
        value = array[source.label_lookup[int(times[index])]]
        write_field(root, "z", index, value)
        if rank == 0 and (step == 0 or (step + 1) % 10 == 0 or step + 1 == len(local)):
            print(f"ERA5 copy: {step + 1}/{len(local)} rows on rank0", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_da_forecast_12h_fuxi.json")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--max-samples", type=int, default=0, help="per split; 0=all")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--plan", action="store_true", help="read metadata only; no GPU or writes")
    args = parser.parse_args()
    if args.max_samples < 0 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("invalid sample/batch/worker count")
    args.splits = list(dict.fromkeys(args.splits))
    config = load_config(args.config)
    data = resolve_data_paths(args.config, config["data"])
    check_ztd_config(data, config["model"])
    joint = bool(config["model"].get("predict_ztd", False))
    dual = bool(config["model"].get("predict_delay_components", False))
    if int(data.get("forecast_lead_hours", 6)) != 12 or not data.get("rollout_zarr"):
        raise ValueError("requires forecast_lead_hours=12 and data.rollout_zarr")
    output = Path(data["rollout_zarr"]).resolve()
    for key in ("era5_path", "ztd_path"):
        source_path = Path(data[key]).resolve()
        if output == source_path or output in source_path.parents or source_path in output.parents:
            raise ValueError("output must be separate from source stores")
    datasets = source_datasets(data)
    times, origins, _ = axes_for_datasets(datasets)
    identity = {
        "schema": SCHEMA, "source_checkpoint": file_identity(data["rollout_source_checkpoint"]),
        "data": data_identity(data), "amp": args.amp,
        "origins_sha256": hashlib.sha256(origins.astype("<i8").tobytes()).hexdigest(),
        "source_observations": source_observation_policy(data),
        "rollout_observations": data.get("rollout_observations", "branch_removed"),
    }
    if joint:
        identity["ztd_schema"] = "18_future_20min_frames_v1"
        identity["ztd_normalization"] = datasets["train"].ztd_normalization
        if dual:
            identity["ztd_schema"] = "18_future_zhd_zwd_mm_v1"
            identity["zwd_normalization"] = datasets["train"].zwd_stats.tolist()
    existing = None
    if output.exists():
        existing = open_store(output)
        if (existing.attrs.get("schema") != SCHEMA or not existing.attrs.get("initialized")
                or existing.attrs.get("identity") != identity):
            raise ValueError("existing output identity mismatch/incomplete initialization; choose a new output")
    era_pending, forecasts_pending = pending_rows(existing, datasets, args.splits, args.max_samples)
    field_bytes = 70 * len(datasets["train"].lat) * len(datasets["train"].lon) * 4
    additional = (len(era_pending) + sum(map(len, forecasts_pending.values()))) * field_bytes
    if joint:
        additional += sum(map(len, forecasts_pending.values())) * int((datasets["train"].obs_static[0] > 0).sum()) * (19 + 2*37 if dual else 37) * 5
    if int(os.environ.get("RANK", 0)) == 0:
        for s in SPLITS:
            print(f"{s}: {len(datasets[s])} origins; selected missing forecasts={len(forecasts_pending.get(s, []))}", flush=True)
        print(f"output={output}\nz: ({len(times)},70,{len(datasets['train'].lat)},{len(datasets['train'].lon)})\n"
              f"z_fuxi6h: ({len(origins)},70,{len(datasets['train'].lat)},{len(datasets['train'].lon)})\n"
              f"full uncompressed GiB={(len(times) + len(origins))*field_bytes/2**30:.2f}; "
              f"selected pending GiB={additional/2**30:.2f}\n"
              f"source={identity['source_observations']}; rollout={identity['rollout_observations']}", flush=True)
    if args.plan:
        return

    rank, _, world_size, device = setup_distributed()
    # Large shared-filesystem copies must not sit in a 10-minute NCCL barrier.
    control = dist.new_group(backend="gloo", timeout=timedelta(hours=4)) if world_size > 1 else None
    lock = None
    checkpoint = None
    try:
        error = None
        if rank == 0:
            try:
                output.parent.mkdir(parents=True, exist_ok=True)
                lock = output.with_name(output.name + ".writer.lock").open("a+")
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock.seek(0)
                lock.truncate()
                lock.write(f"pid={os.getpid()}\n")
                lock.flush()
                if additional * 1.05 > shutil.disk_usage(output.parent).free:
                    raise RuntimeError("insufficient shared filesystem free space for estimated raw fields")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        synchronize_error(error, control, world_size)

        # Restart with complete rows needs no model/checkpoint load.
        if not era_pending and not any(forecasts_pending.values()):
            if rank == 0:
                print("Selected rows already complete; no inference needed", flush=True)
            return
        meta = [None]
        error = None
        if rank == 0:
            try:
                checkpoint = torch.load(data["rollout_source_checkpoint"], map_location="cpu",
                                        mmap=True, weights_only=False)
                meta[0] = source_metadata(checkpoint["config"], data, args.amp)
                if tuple(meta[0]["model"]["image_size"]) != (len(datasets["train"].lat), len(datasets["train"].lon)):
                    raise ValueError("source checkpoint grid differs from ERA5")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        synchronize_error(error, control, world_size)
        if world_size > 1:
            dist.broadcast_object_list(meta, src=0, group=control)
        error = None
        if rank == 0:
            try:
                initialize_store(output, identity, datasets, meta[0]["model"])
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        synchronize_error(error, control, world_size)
        root = open_store(output, "r+")
        # Recheck flags under the lock, including after a previous interrupted run.
        era_pending, forecasts_pending = pending_rows(root, datasets, args.splits, args.max_samples)
        error = None
        try:
            copy_truth(root, datasets["train"], era_pending, times, rank, world_size)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        synchronize_error(error, control, world_size)

        if any(forecasts_pending.values()):
            if rank == 0:
                print("ERA5 copy complete; loading source OBS model for +6h inference", flush=True)
            if world_size > 1:
                model = load_distributed_model(checkpoint, meta[0], meta[0]["model"], device, rank)
            else:
                model = DAForecastNet(**meta[0]["model"]).to(device)
                model.load_state_dict(checkpoint["model"], strict=True)
                model.eval()
            del checkpoint
            enabled, dtype = autocast_settings(device, args.amp)
            for s in args.splits:
                ds, pending = datasets[s], forecasts_pending[s]
                subset = Subset(ds, pending)
                sampler = DistributedEvalSampler(len(subset), rank, world_size)
                loader = DataLoader(IndexedEvalDataset(subset), sampler=sampler,
                                    batch_size=args.batch_size, num_workers=args.num_workers,
                                    pin_memory=device.type == "cuda")
                iterator = iter(loader)
                with torch.inference_mode():
                    for step in range(len(loader)):
                        error = None
                        try:
                            batch = next(iterator)
                        except Exception as exc:
                            error = f"source read: {type(exc).__name__}: {exc}"
                        synchronize_error(error, control, world_size)
                        tensors = [batch[k].to(device, non_blocking=True) for k in (
                            "era5", "era_static", "time_features", "obs_ztd", "obs_mask", "obs_static")]
                        extra = {k: batch[k].to(device, non_blocking=True) for k in COMPONENT_INPUTS if k in batch}
                        with torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled):
                            model_output = model(*tensors, **extra)
                        if joint:
                            next_components = None
                            if dual:
                                next_components = next_component_inputs({"obs_ztd": tensors[3], "obs_mask": tensors[4], **extra}, model_output)
                                obs_next, mask_next = next_components["obs_ztd"], next_components["obs_mask"]
                                next_components = {k: v.float().cpu().numpy() for k,v in next_components.items()}
                            else:
                                obs_next, mask_next = next_observations(tensors[3], tensors[4], model_output["ztd"])
                            obs_next, mask_next = obs_next.float().cpu().numpy(), mask_next.cpu().numpy()
                            prediction = model_output["weather"].float().cpu().numpy()
                        else:
                            prediction = (model_output["weather"] if isinstance(model_output, dict) else model_output).float().cpu().numpy()
                        error = None
                        try:
                            for j in range(len(prediction)):
                                if not bool(batch["_eval_valid"][j]):
                                    continue
                                local_index = pending[int(batch["_eval_index"][j])]
                                index = int(np.searchsorted(origins, ds.sample_time(local_index).value))
                                payload = None
                                if joint:
                                    # Future truth is read ONLY after inference and saved as labels.
                                    target_time = ds.sample_time(local_index) + pd.Timedelta(hours=6)
                                    payload = dict(ztd_input=obs_next[j], ztd_input_mask=mask_next[j])
                                    if dual:
                                        target, target_mask = read_components(ds, target_time, future=True)
                                        payload.update(component_input=next_components["obs_components"][j],
                                                       component_input_mask=next_components["obs_component_mask"][j],
                                                       component_target=target, component_target_mask=target_mask)
                                    else:
                                        target_ztd, target_mask = ds.read_future_ztd(target_time)
                                        payload.update(ztd_target=target_ztd, ztd_target_mask=target_mask)
                                write_field(root, "z_fuxi6h", index, prediction[j], ztd_payload=payload)
                        except Exception as exc:
                            error = f"forecast write: {type(exc).__name__}: {exc}"
                        synchronize_error(error, control, world_size)
                        if rank == 0:
                            print(f"{s}: forecast batch {step+1}/{len(loader)} per rank", flush=True)
        if rank == 0:
            for s in args.splits:
                remaining_era, remaining_forecast = pending_rows(root, datasets, [s], 0)
                print(f"{s}: {'COMPLETE' if not remaining_era and not remaining_forecast[s] else 'PARTIAL; rerun without --max-samples'}", flush=True)
    finally:
        if lock is not None:
            lock.close()  # OS also releases flock after a crash; leave the inode in place.
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
