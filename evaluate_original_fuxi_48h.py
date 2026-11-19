"""Unmodified fuxi_short weights, eight autoregressive hops, paired MAIN test set.

No GNSS values or DA predictions enter FuXi. GNSS metadata is used ONLY to
reproduce the reference origin selection and halo. Does not switch git branches.
"""
import argparse
from datetime import timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from evaluate import CHANNELS, IndexedEvalDataset, DistributedEvalSampler
from evaluate_original_fuxi import load_model, metric_sums, sync_error
from evaluate_main_rollout_spatial import source, load_module, REFERENCE, SOURCE_HASHES
from train import autocast_settings

ROOT = Path(__file__).resolve().parent
# Pin the data/static alignment used to obtain the supplied MAIN experiment.
_base = load_module("data._original48_main_base", source(REFERENCE, "data/da_dataset.py"))
_identity = load_module("_original48_identity", source(REFERENCE, "data/rollout_identity.py"))


class OriginalRolloutInputs(_base.DAForecastDataset):
    def __init__(self, data, split="test"):
        keys = ("era5_path", "ztd_path", "conds_path", "global_lat_path", "global_lon_path",
                "obs_history_hours", "obs_interval_minutes")
        super().__init__(**{k: data[k] for k in keys}, split=data[split + "_split"])
        end = pd.Timestamp(data[split + "_split"][1])
        self.analysis_times = pd.DatetimeIndex([t for t in self.analysis_times
            if t + pd.Timedelta(hours=48) < end
            and all((t + pd.Timedelta(hours=6*j)).value in self.label_lookup for j in range(1, 9))])
        if not len(self):
            raise ValueError("no common 48h origins")

    def __getitem__(self, index):
        # NEVER call the parent's observation reader / __getitem__.
        if self._era_group is None:
            import zarr
            self._era_group = zarr.open_group(self.era5_path, mode="r")
        origin = self.sample_time(index)
        indices = [self.label_lookup[(origin + pd.Timedelta(hours=6*j)).value] for j in range(-1, 9)]
        if not np.all(np.diff(indices) == 1):
            raise ValueError("ERA5 must be contiguous at 6h")
        values = np.asarray(self._era_group["z"][indices[0]:indices[-1]+1], np.float32)
        valid = np.isfinite(values[2:])
        return {"era5": torch.from_numpy(np.nan_to_num(values[:2], nan=0., posinf=0., neginf=0.)),
                "targets": torch.from_numpy(np.where(valid, values[2:], 0)),
                "target_masks": torch.from_numpy(valid),
                "times": torch.from_numpy(np.stack([
                    _base.time_features(origin + pd.Timedelta(hours=6*j)) for j in range(8)]))}


def forecasts(model, history, times, const, amp="bf16"):
    enabled, dtype = autocast_settings(history.device, amp)
    for j in range(times.shape[1]):
        with torch.autocast(history.device.type, dtype=dtype, enabled=enabled):
            residual = model(history, temb=times[:, j], const=const)
        # Original model predicts an increment, not the next absolute field.
        prediction = history[:, -1].float() + residual.float()
        yield prediction
        if j + 1 < times.shape[1]:
            history = torch.stack((history[:, -1].float(), prediction), dim=1)


def paired_reference(experiment):
    config = json.loads((experiment / "resolved_config.json").read_text())
    report = json.loads((experiment / "test/rollout_mae_summary.json").read_text())
    if report["evaluation"]["split"] != "test" or report["evaluation"]["evaluated_steps"] != 8:
        raise ValueError("reference must be an eight-step test evaluation")
    if config["curriculum"]["observation_policy"] != "origin_only" or config["model"].get("obs_channels") != 2:
        raise ValueError("expected MAIN-branch dual-input reference")
    data = dict(config["data"])
    for key in ("era5_path", "ztd_path", "conds_path", "global_lat_path", "global_lon_path", "mean_std_dir"):
        p = Path(data[key]); data[key] = str(p if p.is_absolute() else ROOT / p)
    if _identity.data_identity(data) != config["rollout_data_identity"]:
        raise ValueError("reference data/static/normalization fingerprint changed")
    if (ROOT / "model/fuxi.py").read_text() != source(REFERENCE, "model/fuxi.py"):
        raise ValueError("FuXi architecture differs from pinned reference; review before evaluating")
    ds = OriginalRolloutInputs(data)
    ev = report["evaluation"]
    if (len(ds) != report["samples"] or str(ds.sample_time(0)) != ev["origin_first"]
            or str(ds.sample_time(len(ds)-1)) != ev["origin_last"]):
        raise ValueError("reconstructed test origins differ from reference")
    return config, data, report, ds


def atomic_npz(path, **arrays):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as f:
        np.savez(f, **arrays)
    os.replace(temporary, path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-dir", type=Path, default=ROOT / "work_dirs/da_forecast_rollout48h_zero_zhd_zwd")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "model/fuxi_short.pth")
    p.add_argument("--output-dir", type=Path, default=ROOT / "work_dirs/original_fuxi_48h_vs_zero_zhd_zwd")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=0, help="smoke only; comparison disabled for subsets")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--plan", action="store_true", help="check metadata/origins only; no inference or writes")
    p.add_argument("--plot-only", action="store_true", help="regenerate comparison from completed sums")
    args = p.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.max_samples < 0 or args.save_every < 1:
        p.error("invalid batch size / workers / sample limit / save interval")
    return args


def main():
    args = parse_args()
    if args.plot_only:
        from plot_original_fuxi_48h import make_comparison
        make_comparison(args.output_dir)
        return
    config, data, reference, ds = paired_reference(args.reference_dir)
    selected = min(args.max_samples, len(ds)) if args.max_samples else len(ds)
    world = int(os.environ.get("WORLD_SIZE", 1)); rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    metadata = {
        "checkpoint": _identity.file_identity(args.checkpoint), "reference_dir": str(args.reference_dir.resolve()),
        "reference_csv": _identity.file_identity(args.reference_dir / "test/rollout_mae_by_channel.csv"),
        "reference_summary": _identity.file_identity(args.reference_dir / "test/rollout_mae_summary.json"),
        "reference_config": _identity.file_identity(args.reference_dir / "resolved_config.json"),
        "samples": selected, "full_split_samples": len(ds), "full_test": selected == len(ds),
        "origin_first": str(ds.sample_time(0)), "origin_last": str(ds.sample_time(selected-1)),
        "origins_sha256": hashlib.sha256(ds.analysis_times[:selected].asi8.astype("<i8").tobytes()).hexdigest(),
        "data_identity": config["rollout_data_identity"], "source_revision": REFERENCE,
        "source_sha256": dict(SOURCE_HASHES), "amp": reference["evaluation"]["amp"],
        "world_size": world, "batch_size": args.batch_size, "channels": CHANNELS,
        "halo_radius": config["train"].get("halo_radius", 3),
        "observations": "none; metadata only for matched origins and halo",
        "weights": "original fuxi_short, unchanged at all eight hops",
        "tp": "each lead represents its last 6h accumulation, not cumulative 48h precipitation",
    }
    for path in ("evaluate_original_fuxi_48h.py", "evaluate_original_fuxi.py", "evaluate.py", "train.py"):
        metadata["source_sha256"][path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    if rank == 0:
        print(json.dumps(metadata, indent=2), flush=True)
    if args.plan:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("full-size original FuXi requires CUDA; use --plan for CPU metadata checks")
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    control = dist.new_group(backend="gloo", timeout=timedelta(minutes=60)) if world > 1 else None
    output = args.output_dir
    lock = None
    try:
        problem = None
        if rank == 0:
            try:
                output.mkdir(parents=True, exist_ok=True)
                lock = (output / ".writer.lock").open("a+")
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if (output / "sums.npz").exists():
                    raise ValueError("completed sums already exist; use --plot-only or a new output directory")
                if not args.resume and (output / "progress.npz").exists():
                    raise ValueError("partial results exist; add --resume")
            except Exception as exc:
                problem = repr(exc)
        sync_error(problem, control, world)
        identity = json.dumps(metadata, sort_keys=True)
        sums = torch.zeros((8, 7, 70), dtype=torch.float64, device=device)
        count = torch.zeros((), dtype=torch.int64, device=device)
        cursor = [0]; problem = None
        if rank == 0:
            try:
                if args.resume:
                    with np.load(output / "progress.npz") as saved:
                        if str(saved["identity"]) != identity:
                            raise ValueError("resume identity differs (data/code/checkpoint/world/batch)")
                        cursor[0] = int(saved["batches"])
                        sums.copy_(torch.as_tensor(saved["sums"], device=device))
                        count.fill_(int(saved["samples"]))
            except Exception as exc:
                problem = repr(exc)
        sync_error(problem, control, world)
        if world > 1:
            dist.broadcast_object_list(cursor, src=0)
        # Saved global totals live ONLY on rank 0; other ranks add new local sums.
        mean = torch.as_tensor(np.load(Path(data["mean_std_dir"]) / "mean_era5.npy"),
                               device=device, dtype=torch.float32).reshape(1, 70, 1, 1)
        std = torch.as_tensor(np.load(Path(data["mean_std_dir"]) / "std_era5.npy"),
                              device=device, dtype=torch.float32).reshape(1, 70, 1, 1)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError("invalid normalization")
        lat = torch.cos(torch.deg2rad(torch.as_tensor(ds.lat, device=device))).clamp_min(0)
        weights = (lat / lat.mean()).view(1, 1, -1, 1)
        radius = metadata["halo_radius"]
        halo = F.max_pool2d(torch.as_tensor(ds.obs_static[:1], device=device)[None],
                            2*radius+1, stride=1, padding=radius)
        const = torch.as_tensor(ds.era_static, device=device)
        sampler = list(DistributedEvalSampler(selected, rank, world))
        total_batches = (len(sampler) + args.batch_size - 1) // args.batch_size
        if not 0 <= cursor[0] <= total_batches:
            raise ValueError("invalid resume cursor")
        options = dict(num_workers=args.num_workers, pin_memory=False)
        if args.num_workers:
            options["prefetch_factor"] = 1
        loader = DataLoader(IndexedEvalDataset(Subset(ds, range(selected))), batch_size=args.batch_size,
                            sampler=sampler[cursor[0]*args.batch_size:], **options)
        if rank == 0:
            print(f"Loading original FuXi; batches/rank={total_batches}, resume={cursor[0]}", flush=True)
        model = load_model(args.checkpoint, device, rank, world, metadata["amp"])
        iterator = iter(loader); started = time.monotonic()
        with torch.inference_mode():
            for step in range(cursor[0], total_batches):
                problem = None
                try:
                    batch = next(iterator)
                    history = batch["era5"].to(device)
                    times = batch["times"].to(device)
                    flags = batch["_eval_valid"].to(device)
                except Exception as exc:
                    problem = f"data read failed: {exc}"
                sync_error(problem, control, world)
                for j, prediction in enumerate(forecasts(model, history, times, const, metadata["amp"])):
                    problem = None
                    try:
                        # Keep the other seven targets on CPU, limiting GPU peak memory.
                        target = batch["targets"][:, j].to(device)
                        mask = batch["target_masks"][:, j].to(device).float()
                        item = {"era5": history, "target": target, "target_mask": mask, "_eval_valid": flags}
                        sums[j, :5] += metric_sums(prediction, item, weights, mean, std)
                        hw = mask * flags.view(-1, 1, 1, 1) * weights * halo
                        sums[j, 5] += ((prediction.float()-target).abs()*hw).sum((0, 2, 3), dtype=torch.float64)
                        sums[j, 6] += hw.sum((0, 2, 3), dtype=torch.float64)
                    except Exception as exc:
                        problem = f"lead={6*(j+1)}h metrics failed: {exc}"
                    sync_error(problem, control, world)
                    del target, mask, item, hw
                count += flags.sum()
                done = step + 1
                if rank == 0 and (done == cursor[0]+1 or done % 10 == 0 or done == total_batches):
                    seconds = (time.monotonic()-started)/(done-cursor[0])
                    print(f"batch={done}/{total_batches} seconds/batch={seconds:.1f} "
                          f"ETA_min={seconds*(total_batches-done)/60:.1f}", flush=True)
                if done % args.save_every == 0 or done == total_batches:
                    total_sums, total_count = sums.clone(), count.clone()
                    if world > 1:
                        dist.all_reduce(total_sums); dist.all_reduce(total_count)
                    problem = None
                    if rank == 0:
                        try:
                            arrays = dict(identity=np.asarray(identity), sums=total_sums.cpu().numpy(),
                                          samples=total_count.cpu().numpy(), batches=np.asarray(done))
                            atomic_npz(output / "progress.npz", **arrays)
                            if done == total_batches:
                                if int(total_count) != selected:
                                    raise ValueError("distributed sample count mismatch")
                                atomic_npz(output / "sums.npz", **arrays)
                        except Exception as exc:
                            problem = f"save failed: {exc}"
                    sync_error(problem, control, world)
                del batch, history, times, flags, prediction
        # If interrupted after final progress write but before final publication.
        if rank == 0 and cursor[0] == total_batches:
            if int(count) != selected:
                raise ValueError("completed progress sample count mismatch")
            atomic_npz(output / "sums.npz", identity=np.asarray(identity), sums=sums.cpu().numpy(),
                       samples=count.cpu().numpy(), batches=np.asarray(cursor[0]))
    finally:
        if lock is not None:
            lock.close()
        if world > 1:
            dist.destroy_process_group()
    if rank == 0:
        from plot_original_fuxi_48h import make_comparison
        make_comparison(output)


if __name__ == "__main__":
    main()
