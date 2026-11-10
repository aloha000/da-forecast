"""Full-test 48h spatial MAE comparison for the two MAIN-branch checkpoints.

Loads the pinned main architecture/dataset in isolated in-memory modules; never
checks out a branch. One FSDP model at a time. No training or checkpoint updates.
"""
import argparse
import csv
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parent
CHANNELS=["t2m","r1000","u1000","v1000","tp","msl","z1000"]
INDICES=[65,64,38,51,69,68,12]
UNITS=["K","RH percentage points","m/s","m/s","mm","hPa","gpm"]
REFERENCE="dcbb37ce5429acb7adf3542adae68c87788036b7"
SOURCE_HASHES={}


def source(revision,path):
    text=subprocess.check_output(["git","show",revision+":"+path],cwd=ROOT,text=True)
    SOURCE_HASHES[path]=hashlib.sha256(text.encode()).hexdigest()
    return text


def load_module(name,text):
    module=types.ModuleType(name);module.__package__=name.rpartition(".")[0]
    sys.modules[name]=module
    exec(compile(text,name,"exec"),module.__dict__)
    return module


def runtime(revision):
    import data, model
    assert (ROOT/"model/fuxi.py").read_text()==source(revision,"model/fuxi.py")
    load_module("data._spatial_main_base",source(revision,"data/da_dataset.py"))
    ds_source=source(revision,"data/rollout_sequence.py").replace(
        "from .da_dataset import","from ._spatial_main_base import")
    ds=load_module("data._spatial_main_sequence",ds_source)
    net=load_module("model._spatial_main_net",source(revision,"model/da_forecast_model.py"))
    identity=load_module("_spatial_main_identity",source(revision,"data/rollout_identity.py"))
    training=load_module("_spatial_main_train",source(revision,"train.py"))
    class FinalLeadDataset(ds.RolloutSequenceDataset):
        def __getitem__(self,index):
            t=self.sample_time(index);self._open()
            a=self.label_lookup[(t-pd.Timedelta(hours=6)).value]
            b=self.label_lookup[t.value]
            assert b==a+1
            era=np.asarray(self._era_group["z"][a:b+1],np.float32)
            end=self.label_lookup[(t+pd.Timedelta(hours=48)).value]
            target=np.asarray(self._era_group["z"].oindex[end,INDICES,:,:],np.float32)
            mask=np.isfinite(target)
            obs,obs_mask=self._read_components(t)
            if self.zero_delay_inputs: obs.fill(0.)
            return {"era5":torch.from_numpy(np.nan_to_num(era,nan=0.,posinf=0.,neginf=0.)),
                    "era_static":torch.from_numpy(self.era_static),
                    "obs_ztd":torch.from_numpy(obs),"obs_mask":torch.from_numpy(obs_mask),
                    "obs_static":torch.from_numpy(self.obs_static),
                    "target":torch.from_numpy(np.where(mask,target,0)),
                    "target_mask":torch.from_numpy(mask),
                    "time":torch.from_numpy(np.stack([
                        ds.time_features(t+pd.Timedelta(hours=6*j)) for j in range(8)]))}
    return FinalLeadDataset,net.DAForecastNet,identity,training


def physical(x,mean,std):
    result=x.float()*std+mean
    result[:,4]=torch.expm1(result[:,4]).clamp_min(0.)
    result[:,5]=result[:,5]/100.
    result[:,6]=result[:,6]/9.80665
    return result


def atomic_npz(path,**arrays):
    temporary=path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle,**arrays)
    os.replace(temporary,path)


def finite(tensor,device,message):
    flag=torch.isfinite(tensor).all().to(torch.int32)
    if dist.is_initialized(): dist.all_reduce(flag,op=dist.ReduceOp.MIN)
    if not bool(flag): raise FloatingPointError(message)


def reference_rows(experiment):
    with (experiment/"test/rollout_mae_by_channel.csv").open() as f:
        return {r["channel"]:r for r in csv.DictReader(f) if int(r["lead_hours"])==48}


@torch.no_grad()
def evaluate_variant(args,variant,rank,world,device,rt):
    Dataset,Network,identity,training=rt
    name="zero_zhd_zwd" if variant=="zero" else "zhd_zwd"
    exp=ROOT/"work_dirs"/("da_forecast_rollout48h_"+name)
    path=exp/"stage_48h/best.pt"
    ck=None;payload=[None];error=[None]
    if rank==0:
        try:
            ck=torch.load(path,map_location="cpu",weights_only=False,mmap=True)
            payload[0]={"config":ck["config"],"step":ck["step"],
                        "checkpoint":identity.file_identity(path)}
        except Exception as exc: error[0]=repr(exc)
    dist.broadcast_object_list(error,src=0)
    if error[0]: raise RuntimeError(error[0])
    dist.broadcast_object_list(payload,src=0)
    saved=payload[0]["config"]
    if saved["model"].get("obs_channels")!=2 or saved["model"].get("predict_delay_components"):
        raise ValueError("requires main dual-input weather-only checkpoint")
    if saved["curriculum"]["observation_policy"]!="origin_only" or saved["train"]["amp"]!="bf16":
        raise ValueError("reference policy/precision mismatch")
    cfg=saved["data"].copy()
    for key in ("era5_path","ztd_path","conds_path","global_lat_path","global_lon_path","mean_std_dir"):
        p=Path(cfg[key]);cfg[key]=str(p if p.is_absolute() else ROOT/p)
    if identity.data_identity(cfg)!=saved["rollout_data_identity"]:
        raise ValueError("checkpoint data identity mismatch")
    if cfg.get("zero_delay_inputs",False)!=(variant=="zero"):
        raise ValueError("observation policy mismatch")
    keys=("era5_path","ztd_path","conds_path","global_lat_path","global_lon_path",
          "obs_history_hours","obs_interval_minutes")
    ds=Dataset(**{k:cfg[k] for k in keys},split=cfg["test_split"],
               zero_delay_inputs=cfg.get("zero_delay_inputs",False),max_steps=8,horizon=8)
    if ds.observation_normalization!=saved["observation_normalization"]:
        raise ValueError("normalization mismatch")
    existing=json.loads((exp/"test/rollout_mae_summary.json").read_text())
    assert len(ds)==existing["samples"]
    assert str(ds.sample_time(0))==existing["evaluation"]["origin_first"]
    assert str(ds.sample_time(len(ds)-1))==existing["evaluation"]["origin_last"]
    size=min(len(ds),args.max_samples) if args.max_samples else len(ds)
    assert size>0
    origins=ds.analysis_times[:size].asi8
    ident={"variant":variant,"checkpoint":payload[0]["checkpoint"],"completed_updates":payload[0]["step"],
           "source_revision":args.source_revision,"source_sha256":SOURCE_HASHES,
           "sample_count":size,"full_test":size==len(ds),"channels":CHANNELS,
           "origin_first":str(ds.sample_time(0)),"origin_last":str(ds.sample_time(size-1)),
           "origins_sha256":hashlib.sha256(origins.tobytes()).hexdigest(),
           "amp":"bf16","world_size":world,"batch_size":1,
           "data_identity":saved["rollout_data_identity"]}
    identity_json=json.dumps(ident,sort_keys=True)
    final=args.output_dir/("mae48_"+variant+".npz")
    if final.exists():
        with np.load(final) as old:
            if str(old["identity"])!=identity_json: raise ValueError("existing spatial output identity differs")
        if rank==0: print(f"{variant}: existing complete result verified; skip",flush=True)
        return
    per_rank=(size+world-1)//world
    shard=[i if i<size else 0 for i in range(rank,per_rank*world,world)]
    shard_valid=[i<size for i in range(rank,per_rank*world,world)]
    shape=(len(CHANNELS),len(ds.lat),len(ds.lon))
    errors_n=torch.zeros(shape,dtype=torch.float64,device=device)
    errors_p=torch.zeros_like(errors_n);counts=torch.zeros(shape,dtype=torch.int64,device=device)
    done=0;num_samples=0
    partial=args.output_dir/f"partial_{variant}_rank{rank}.npz"
    if partial.exists():
        with np.load(partial) as old:
            if str(old["identity"])!=identity_json: raise ValueError("partial identity differs")
            done=int(old["done"]);num_samples=int(old["num_samples"])
            errors_n.copy_(torch.as_tensor(old["error_normalized"],device=device))
            errors_p.copy_(torch.as_tensor(old["error_physical"],device=device))
            counts.copy_(torch.as_tensor(old["count"],device=device))
    cursors=[None]*world
    dist.all_gather_object(cursors,done)
    if len(set(cursors))!=1: raise ValueError(f"partial rank cursors differ: {cursors}; use a new output directory")
    if rank==0:
        print(f"{variant}: samples={size}, batches/rank={per_rank}, resume={done}; loading {path}",flush=True)
    network=Network(**saved["model"])
    error=[None]
    if rank==0:
        try: network.load_state_dict(ck["model"],strict=True)
        except Exception as exc: error[0]=repr(exc)
    dist.broadcast_object_list(error,src=0)
    if error[0]: raise RuntimeError(error[0])
    del ck
    config=dict(saved["train"]);config["fsdp"]={**config["fsdp"],"sync_module_states":True,"forward_prefetch":False}
    network=training.wrap_fsdp(network,device,config);network.eval()
    mean=torch.as_tensor(np.load(Path(cfg["mean_std_dir"])/"mean_era5.npy").reshape(-1)[INDICES],
                         device=device,dtype=torch.float32).view(1,-1,1,1)
    std=torch.as_tensor(np.load(Path(cfg["mean_std_dir"])/"std_era5.npy").reshape(-1)[INDICES],
                        device=device,dtype=torch.float32).view(1,-1,1,1)
    options={"num_workers":args.num_workers,"pin_memory":False}
    if args.num_workers: options["prefetch_factor"]=1
    loader=DataLoader(ds,batch_size=1,sampler=shard[done:],**options)
    started=time.monotonic();start_done=done
    selection=torch.tensor(INDICES,device=device)
    for batch in loader:
        batch={k:v.to(device) for k,v in batch.items()}
        history=batch["era5"]
        with torch.autocast("cuda",dtype=torch.bfloat16):
            for step in range(8):
                pred=network(history,batch["era_static"],batch["time"][:,step],
                             batch["obs_ztd"] if step==0 else None,
                             batch["obs_mask"] if step==0 else None,
                             batch["obs_static"] if step==0 else None,
                             obs_enabled=step==0)
                if step<7: history=torch.stack((history[:,-1],pred),dim=1)
        pred=pred.index_select(1,selection).float()
        truth=batch["target"].float()
        valid=batch["target_mask"] & shard_valid[done]
        pred_p=physical(pred,mean,std);truth_p=physical(truth,mean,std)
        finite(pred,device,"nonfinite normalized predictions")
        finite(pred_p,device,"nonfinite physical predictions")
        errors_n+=torch.where(valid,(pred-truth).abs(),0).sum(0,dtype=torch.float64)
        errors_p+=torch.where(valid,(pred_p-truth_p).abs(),0).sum(0,dtype=torch.float64)
        counts+=valid.sum(0)
        num_samples+=int(shard_valid[done]);done+=1
        del batch,history,pred,truth,valid,pred_p,truth_p
        if rank==0 and (done==start_done+1 or done%10==0 or done==per_rank):
            elapsed=time.monotonic()-started
            eta=elapsed/max(done-start_done,1)*(per_rank-done)
            print(f"{variant}: {done}/{per_rank} batches/rank | {elapsed/max(done-start_done,1):.1f}s/batch | ETA {eta/60:.1f}min",flush=True)
        if done%args.save_every==0 or done==per_rank:
            atomic_npz(partial,identity=np.asarray(identity_json),done=np.asarray(done),
                       num_samples=np.asarray(num_samples),error_normalized=errors_n.cpu().numpy(),
                       error_physical=errors_p.cpu().numpy(),count=counts.cpu().numpy())
            dist.barrier()
    for values in (errors_n,errors_p,counts): dist.all_reduce(values)
    sample_tensor=torch.tensor(num_samples,dtype=torch.int64,device=device);dist.all_reduce(sample_tensor)
    if int(sample_tensor)!=size: raise ValueError("sample count mismatch")
    if rank==0:
        en,ep,n=errors_n.cpu().numpy(),errors_p.cpu().numpy(),counts.cpu().numpy()
        mn=np.divide(en,n,out=np.full(shape,np.nan),where=n>0)
        mp=np.divide(ep,n,out=np.full(shape,np.nan),where=n>0)
        # Re-aggregate with latitude weights to check against the published CSV.
        weights=np.maximum(np.cos(np.deg2rad(ds.lat)),0).astype(np.float64)[None,:,None]
        checks=[];reference=reference_rows(exp)
        for j,c in enumerate(CHANNELS):
            norm=float((en[j]*weights[0]).sum()/(n[j]*weights[0]).sum())
            phys=float((ep[j]*weights[0]).sum()/(n[j]*weights[0]).sum())
            row={"channel":c,"recomputed_normalized":norm,"reference_normalized":float(reference[c]["global_mae_normalized"]),
                 "recomputed_physical":phys,"reference_physical":float(reference[c]["global_mae_physical"])}
            if size==len(ds):
                if not np.isclose(norm,row["reference_normalized"],rtol=2e-5,atol=2e-7):
                    raise ValueError(f"reference normalized MAE mismatch: {row}")
                if not np.isclose(phys,row["reference_physical"],rtol=2e-5,atol=2e-7):
                    raise ValueError(f"reference physical MAE mismatch: {row}")
            checks.append(row)
        atomic_npz(final,identity=np.asarray(identity_json),mae_normalized=mn,mae_physical=mp,
                   count=n,lat=ds.lat,lon=ds.lon,land_mask=ds.era_static[0],
                   channels=np.array(CHANNELS),units=np.array(UNITS))
        (args.output_dir/f"reference_check_{variant}.json").write_text(json.dumps(checks,indent=2)+"\n")
        print(f"{variant}: saved {final}; reference MAE checks passed (full_test={size==len(ds)})",flush=True)
    dist.barrier()
    del network,errors_n,errors_p,counts,loader,ds
    gc.collect();torch.cuda.empty_cache()
    dist.barrier()


def plot_maps(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FuncFormatter
    z=np.load(out/"mae48_zero.npz");o=np.load(out/"mae48_obs.npz")
    zi=json.loads(str(z["identity"]));oi=json.loads(str(o["identity"]))
    for key in ("sample_count","full_test","origins_sha256","channels","amp"):
        assert zi[key]==oi[key],key
    assert np.array_equal(z["count"],o["count"])
    assert np.array_equal(z["lat"],o["lat"]) and np.array_equal(z["lon"],o["lon"])
    lon=(z["lon"]+180)%360-180;xx=np.argsort(lon)
    lat=z["lat"];yy=np.argsort(lat)
    land=z["land_mask"][yy][:,xx]
    lat,lon=lat[yy],lon[xx]
    cmap=LinearSegmentedColormap.from_list("red_white_green",["#a50f15","#f7f7f7","#006d2c"])
    cmap.set_bad("#dddddd")
    notes=[];matrices={}
    def panel(ax,values,j,limit,fig):
        field=values[j][yy][:,xx]
        mesh=ax.pcolormesh(lon,lat,field,shading="auto",cmap=cmap,vmin=-limit,vmax=limit,rasterized=True)
        ax.contour(lon,lat,land,levels=[.5],colors="#333333",linewidths=.35)
        ax.set(xlim=(-180,180),ylim=(-90,90),xticks=np.arange(-180,181,60),yticks=np.arange(-90,91,30))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f"{abs(v):g}"+("W" if v<0 else "E" if v>0 else "")))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f"{abs(v):g}"+("S" if v<0 else "N" if v>0 else "")))
        ax.grid(alpha=.15,lw=.4)
        ax.set_title(f"{CHANNELS[j]} | range +/-{limit:.3g}% (99th percentile)",fontsize=12,loc="left")
        cb=fig.colorbar(mesh,ax=ax,orientation="horizontal",pad=.08,fraction=.045,extend="both")
        cb.set_label("MAE relative improvement (%)",fontsize=9)
    for mode in ("physical","normalized"):
        zero=z["mae_"+mode];obs=o["mae_"+mode]
        valid=(z["count"]>0)&np.isfinite(zero)&np.isfinite(obs)&(zero>1e-12)
        values=np.divide(100*(zero-obs),zero,out=np.full_like(zero,np.nan),where=valid)
        matrices["improvement_"+mode+"_pct"]=values
        limits=[max(float(np.nanpercentile(np.abs(v),99)),1e-6) for v in values]
        fig,axes=plt.subplots(4,2,figsize=(20,18))
        fig.subplots_adjust(left=.055,right=.97,top=.92,bottom=.055,hspace=.40,wspace=.14)
        for j in range(7): panel(axes.flat[j],values,j,limits[j],fig)
        axes.flat[-1].axis("off")
        axes.flat[-1].text(.02,.85,
            "Green: real ZHD/ZWD inputs improve MAE\nRed: zero inputs perform better\n\n"
            "100 x (MAE_zero - MAE_real) / MAE_zero\n"
            "MAE is averaged over test origins at EACH grid cell.\n"
            "Only the +48h forecast is scored.\n\n"
            "r/u/v/z1000 refer to the 1000 hPa level, not 1000 m.\n"
            "Grey: no valid samples or near-zero baseline MAE.\n"
            "Color limits clip tails; saved arrays are NOT clipped.\n"
            "Coast outline: aligned FuXi land-mask 0.5 contour.",
            transform=axes.flat[-1].transAxes,va="top",fontsize=11,linespacing=1.5)
        label="FULL TEST" if zi["full_test"] else "SMOKE SUBSET - NOT FULL TEST"
        fig.suptitle(f"48h spatial MAE improvement | {mode} MAE | {label}\n"
                     f"Real ZHD/ZWD vs zero inputs; {zi['sample_count']} identical forecast origins",
                     fontsize=18,fontweight="bold")
        fig.savefig(out/f"improvement_48h_{mode}.png",dpi=180,facecolor="white");plt.close(fig)
        for j,c in enumerate(CHANNELS):
            if mode=="physical":
                one,ax=plt.subplots(figsize=(13,6.5))
                one.subplots_adjust(left=.065,right=.96,top=.83,bottom=.14)
                panel(ax,values,j,limits[j],one)
                one.suptitle(f"{c}: +48h physical-MAE relative improvement\n{label}; "
                             f"{zi['sample_count']} origins; green = real-input model better",fontsize=15)
                one.savefig(out/f"improvement_48h_{c}.png",dpi=180,facecolor="white");plt.close(one)
            area=np.maximum(np.cos(np.deg2rad(z["lat"])),0)[:,None]*np.ones_like(values[j])
            use=np.isfinite(values[j])
            notes.append({"mode":mode,"channel":c,"color_limit_pct":limits[j],
                          "valid_cells":int(use.sum()),"area_fraction_improved":float(area[use&(values[j]>0)].sum()/area[use].sum()),
                          "median_cell_improvement_pct":float(np.nanmedian(values[j]))})
    atomic_npz(out/"spatial_improvement_48h.npz",**matrices,lat=z["lat"],lon=z["lon"],count=z["count"],channels=z["channels"])
    (out/"spatial_summary.json").write_text(json.dumps({"zero":zi,"obs":oi,"maps":notes,
        "formula":"100*(time_mean_abs_error_zero-time_mean_abs_error_obs)/time_mean_abs_error_zero",
        "note":"Maps are pointwise time means, not a latitude-weighted spatial reduction. Physical tp uses expm1 inverse normalization."},indent=2)+"\n")
    z.close();o.close()
    print(f"All 48h spatial maps saved under {out}",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,default=ROOT/"work_dirs/comparison_rollout48h_zhd_zwd_vs_zero_test/spatial_48h")
    p.add_argument("--source-revision",default=REFERENCE)
    p.add_argument("--num-workers",type=int,default=1)
    p.add_argument("--save-every",type=int,default=25)
    p.add_argument("--max-samples",type=int,default=0)
    p.add_argument("--plot-only",action="store_true")
    args=p.parse_args()
    if args.plot_only:
        plot_maps(args.output_dir);return
    if args.num_workers<0 or args.save_every<1 or args.max_samples<0: p.error("invalid limits")
    world=int(os.environ.get("WORLD_SIZE","1"));rank=int(os.environ.get("RANK","0"))
    local=int(os.environ.get("LOCAL_RANK","0"))
    if world!=4: p.error("use 4 GPU processes to match the existing evaluation")
    torch.cuda.set_device(local);device=torch.device("cuda",local)
    dist.init_process_group("nccl",timeout=timedelta(minutes=60))
    try:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        rt=runtime(args.source_revision)
        for variant in ("zero","obs"):
            evaluate_variant(args,variant,rank,world,device,rt)
        dist.barrier()
    finally:
        dist.destroy_process_group()
    if rank==0: plot_maps(args.output_dir)


if __name__=="__main__":
    main()
