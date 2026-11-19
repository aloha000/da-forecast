"""CPU-only paired MAIN checkpoint spatial evaluation; never allocates CUDA.

CPU BF16 is a separate backend evaluation, not bit-identical to CUDA BF16.
Always preserve backend provenance and compare available GPU smoke references.
"""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

import evaluate_main_rollout_spatial as spatial


def forward_cpu(network, batch, progress=None):
    # Match FSDP's root input casting and BF16 parameter storage. FP32 labels
    # must NOT be cast: they were not inputs to the original FSDP model either.
    cast = lambda x: x.to(torch.bfloat16) if x.is_floating_point() else x
    history = cast(batch["era5"])
    static = cast(batch["era_static"])
    times = cast(batch["time"])
    obs, mask, station = (cast(batch[k]) for k in ("obs_ztd", "obs_mask", "obs_static"))
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        for j in range(8):
            started = time.monotonic()
            pred = network(history, static, times[:, j], obs if j == 0 else None,
                           mask if j == 0 else None, station if j == 0 else None,
                           obs_enabled=j == 0)
            if not torch.isfinite(pred).all():
                raise FloatingPointError(f"nonfinite CPU prediction at {6*(j+1)}h")
            if progress:
                progress(j+1, time.monotonic()-started)
            if j < 7:
                history = torch.stack((history[:, -1], pred), dim=1)
    return pred[:, spatial.INDICES].float()


def metadata_and_dataset(variant, rt):
    Dataset, Network, identity, _ = rt
    name = "zero_zhd_zwd" if variant == "zero" else "zhd_zwd"
    exp = spatial.ROOT / "work_dirs" / ("da_forecast_rollout48h_"+name)
    path = exp / "stage_48h/best.pt"
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    saved = ck["config"]
    if saved["model"].get("obs_channels") != 2 or saved["model"].get("predict_delay_components"):
        raise ValueError("requires main dual-input weather-only checkpoint")
    if saved["curriculum"]["observation_policy"] != "origin_only" or saved["train"]["amp"] != "bf16":
        raise ValueError("reference policy/precision mismatch")
    cfg = saved["data"].copy()
    for key in ("era5_path", "ztd_path", "conds_path", "global_lat_path", "global_lon_path", "mean_std_dir"):
        p = Path(cfg[key]); cfg[key] = str(p if p.is_absolute() else spatial.ROOT / p)
    if identity.data_identity(cfg) != saved["rollout_data_identity"]:
        raise ValueError("checkpoint data identity mismatch")
    if cfg.get("zero_delay_inputs", False) != (variant == "zero"):
        raise ValueError("observation policy mismatch")
    keys = ("era5_path", "ztd_path", "conds_path", "global_lat_path", "global_lon_path",
            "obs_history_hours", "obs_interval_minutes")
    ds = Dataset(**{k:cfg[k] for k in keys}, split=cfg["test_split"],
                 zero_delay_inputs=cfg.get("zero_delay_inputs", False), max_steps=8, horizon=8)
    if ds.observation_normalization != saved["observation_normalization"]:
        raise ValueError("normalization mismatch")
    report = json.loads((exp / "test/rollout_mae_summary.json").read_text())
    if (len(ds) != report["samples"] or str(ds.sample_time(0)) != report["evaluation"]["origin_first"]
            or str(ds.sample_time(len(ds)-1)) != report["evaluation"]["origin_last"]):
        raise ValueError("CPU test origins differ from saved evaluation")
    return ds, Network, identity, saved, cfg, ck, path, exp


def compare_gpu_smoke(out, variant, en, ep, counts, sample_count):
    smoke = spatial.ROOT / "work_dirs/comparison_rollout48h_zhd_zwd_vs_zero_test/spatial_smoke_4"
    if sample_count not in (1, 4) or not smoke.exists():
        return
    if sample_count == 1:
        with np.load(smoke / f"partial_{variant}_rank0.npz") as gpu:
            if int(gpu["num_samples"]) != 1 or int(gpu["done"]) != 1:
                return
            ident = json.loads(str(gpu["identity"]))
            if ident["world_size"] != 4 or ident["sample_count"] != 4:
                return
            gn, gp, gcnt = gpu["error_normalized"], gpu["error_physical"], gpu["count"]
    else:
        with np.load(smoke / f"mae48_{variant}.npz") as gpu:
            gcnt = gpu["count"]; gn = gpu["mae_normalized"]*gcnt; gp = gpu["mae_physical"]*gcnt
    if not np.array_equal(gcnt, counts):
        raise ValueError("GPU smoke count mismatch")
    rows = []
    for mode, cpu, gpu in (("normalized", en, gn), ("physical", ep, gp)):
        for i,c in enumerate(spatial.CHANNELS):
            valid = counts[i] > 0
            diff = np.abs(cpu[i][valid]-gpu[i][valid])/counts[i][valid]
            scale = np.abs(gpu[i][valid]).sum()
            rows.append({"mode":mode,"channel":c,
                         "mean_abs_cell_mae_backend_difference":float(diff.mean()),
                         "p99_abs_cell_mae_backend_difference":float(np.percentile(diff,99)),
                         "aggregate_mae_relative_difference_pct":float(100*(cpu[i][valid].sum()-gpu[i][valid].sum())/scale)
                             if scale > 0 else None})
    (out / f"cpu_vs_gpu_smoke_{variant}.json").write_text(json.dumps(rows, indent=2)+"\n")
    print(f"CPU/GPU smoke diagnostic saved for {variant}: {rows[:7]}", flush=True)


def evaluate(args, variant, rt):
    started = time.monotonic()
    ds, Network, identity, saved, cfg, ck, path, exp = metadata_and_dataset(variant, rt)
    size = min(len(ds),args.max_samples) if args.max_samples else len(ds)
    ident = {"variant":variant,"checkpoint":identity.file_identity(path),"completed_updates":ck["step"],
             "source_revision":spatial.REFERENCE,"source_sha256":dict(spatial.SOURCE_HASHES),
             "sample_count":size,"full_test":size==len(ds),"channels":spatial.CHANNELS,
             "origin_first":str(ds.sample_time(0)),"origin_last":str(ds.sample_time(size-1)),
             "origins_sha256":hashlib.sha256(ds.analysis_times[:size].asi8.tobytes()).hexdigest(),
             "amp":"bf16","device":"cpu","world_size":1,"batch_size":1,
             "cpu_threads":args.threads,"torch_version":str(torch.__version__),
             "data_identity":saved["rollout_data_identity"],
             "backend_note":"CPU BF16 params/root inputs; kernels may differ from CUDA BF16. Do not mix backends."}
    ident["source_sha256"]["cpu_runner"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ident_text = json.dumps(ident,sort_keys=True)
    out = args.output_dir
    final, partial = out/f"mae48_{variant}.npz", out/f"partial_cpu_{variant}.npz"
    if final.exists():
        with np.load(final) as old:
            if str(old["identity"]) != ident_text:
                raise ValueError("completed result identity differs")
        print(f"{variant}: completed result already exists",flush=True)
        return
    shape = (7,len(ds.lat),len(ds.lon))
    en = np.zeros(shape,np.float64); ep = np.zeros_like(en); counts = np.zeros(shape,np.int64); done = 0
    if partial.exists():
        with np.load(partial) as old:
            if str(old["identity"]) != ident_text:
                raise ValueError("partial CPU identity differs")
            en[:]=old["error_normalized"]; ep[:]=old["error_physical"]; counts[:]=old["count"]; done=int(old["done"])
    if not 0 <= done <= size:
        raise ValueError("invalid resume cursor")
    print(f"{variant}: loading full model on CPU; {done}/{size} samples, threads={args.threads}",flush=True)
    # Instantiating on meta avoids random initialization and a second full FP32 copy.
    with torch.device("meta"):
        network = Network(**saved["model"])
    network.load_state_dict(ck["model"], strict=True, assign=True)
    network.bfloat16().eval()
    del ck
    gc.collect()
    print(f"{variant}: CPU model ready, initialization seconds={time.monotonic()-started:.1f}",flush=True)
    mean = torch.from_numpy(np.load(Path(cfg["mean_std_dir"])/"mean_era5.npy").reshape(-1)[spatial.INDICES].astype(np.float32)).view(1,-1,1,1)
    std = torch.from_numpy(np.load(Path(cfg["mean_std_dir"])/"std_era5.npy").reshape(-1)[spatial.INDICES].astype(np.float32)).view(1,-1,1,1)
    run_start = time.monotonic(); start_done = done
    for index in range(done,size):
        tick = time.monotonic()
        batch = {k:v[None] for k,v in ds[index].items()}
        print(f"{variant}: sample={index+1}/{size}, read_seconds={time.monotonic()-tick:.1f}",flush=True)
        def log_step(step, seconds):
            print(f"{variant}: sample={index+1} lead={6*step}h cpu_seconds={seconds:.1f}",flush=True)
        pred = forward_cpu(network,batch,log_step)
        truth, valid = batch["target"].float(), batch["target_mask"].bool()
        pp, tp = spatial.physical(pred,mean,std), spatial.physical(truth,mean,std)
        if not torch.isfinite(pp).all():
            raise FloatingPointError("nonfinite CPU physical forecast")
        en += torch.where(valid,(pred-truth).abs(),0).sum(0,dtype=torch.float64).numpy()
        ep += torch.where(valid,(pp-tp).abs(),0).sum(0,dtype=torch.float64).numpy()
        counts += valid.sum(0).numpy(); done=index+1
        elapsed = time.monotonic()-run_start
        per_sample = elapsed/(done-start_done)
        print(f"{variant}: completed={done}/{size}, seconds/sample={per_sample:.1f}, "
              f"remaining_hours_this_variant={per_sample*(size-done)/3600:.2f}, "
              f"estimated_full_two_models_hours={per_sample*len(ds)*2/3600:.2f}",flush=True)
        if done%args.save_every==0 or done==size:
            spatial.atomic_npz(partial,identity=np.asarray(ident_text),done=np.asarray(done),
                               error_normalized=en,error_physical=ep,count=counts)
        del batch,pred,truth,valid,pp,tp
    mn = np.divide(en,counts,out=np.full_like(en,np.nan),where=counts>0)
    mp = np.divide(ep,counts,out=np.full_like(ep,np.nan),where=counts>0)
    compare_gpu_smoke(out,variant,en,ep,counts,size)
    # Report, do not silently claim exact agreement across CPU/GPU backends.
    reference = spatial.reference_rows(exp); checks=[]
    weights = np.maximum(np.cos(np.deg2rad(ds.lat)),0).astype(np.float64)[:,None]
    for i,c in enumerate(spatial.CHANNELS):
        denominator = (counts[i]*weights).sum()
        row={"channel":c,"cpu_mae_normalized":float((en[i]*weights).sum()/denominator),
             "gpu_reference_mae_normalized":float(reference[c]["global_mae_normalized"]),
             "cpu_mae_physical":float((ep[i]*weights).sum()/denominator),
             "gpu_reference_mae_physical":float(reference[c]["global_mae_physical"]),
             "comparable_sample_set":size==len(ds)}
        checks.append(row)
    (out/f"reference_check_{variant}.json").write_text(json.dumps(checks,indent=2)+"\n")
    spatial.atomic_npz(final,identity=np.asarray(ident_text),mae_normalized=mn,mae_physical=mp,
                       count=counts,lat=ds.lat,lon=ds.lon,land_mask=ds.era_static[0],
                       channels=np.array(spatial.CHANNELS),units=np.array(spatial.UNITS))
    print(f"{variant}: CPU result saved {final}",flush=True)
    del network,ds
    gc.collect()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,default=spatial.ROOT/"work_dirs/comparison_rollout48h_zhd_zwd_vs_zero_test/spatial_48h_cpu")
    p.add_argument("--threads",type=int,default=16)
    p.add_argument("--max-samples",type=int,default=0)
    p.add_argument("--save-every",type=int,default=1)
    p.add_argument("--variant",choices=("zero","obs","both"),default="both")
    p.add_argument("--plot-only",action="store_true")
    args=p.parse_args()
    if args.threads<1 or args.max_samples<0 or args.save_every<1:
        p.error("invalid limits")
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    if args.plot_only:
        spatial.plot_maps(args.output_dir);return
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/".cpu_writer.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rt=spatial.runtime(spatial.REFERENCE)
        for variant in (("zero","obs") if args.variant=="both" else (args.variant,)):
            evaluate(args,variant,rt)
        if all((args.output_dir/f"mae48_{v}.npz").exists() for v in ("zero","obs")):
            spatial.plot_maps(args.output_dir)


if __name__=="__main__":
    main()
