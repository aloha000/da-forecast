"""Utilities shared by the 48h curriculum trainer and evaluator."""
from datetime import timedelta
import csv
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, BatchSampler, DistributedSampler
from evaluate import DistributedEvalSampler, IndexedEvalDataset, CHANNELS, channel_unit, inverse_all
from model.delay_rollout import rollout_predictions, lead_labels
from model.delay_components import component_error_sums, latest_components
from train import autocast_settings, move_batch

def setup_rollout_distributed():
    world=int(os.environ.get("WORLD_SIZE","1"));rank=int(os.environ.get("RANK","0"))
    local=int(os.environ.get("LOCAL_RANK","0"))
    if world>1:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed rollout requires CUDA")
        torch.cuda.set_device(local)
        dist.init_process_group("nccl",timeout=timedelta(minutes=60))
    device=torch.device("cuda",local) if torch.cuda.is_available() else torch.device("cpu")
    return rank,world,device

def stage_for_update(completed,steps_per_stage,max_steps):
    return min(completed//steps_per_stage+1,max_steps)

class OffsetBatchSampler(BatchSampler):
    def __init__(self,sampler,batch_size,offset):
        super().__init__(sampler,batch_size,drop_last=True)
        self.offset=offset
    def __iter__(self):
        for i,batch in enumerate(super().__iter__()):
            if i>=self.offset:
                yield batch
    def __len__(self):
        return max(0,super().__len__()-self.offset)

class BatchStream:
    """Resume by sampler offset, without reading skipped weather samples."""
    def __init__(self,dataset,config,rank,world,completed_microbatches):
        self.dataset,self.config,self.rank,self.world=dataset,config,rank,world
        self.position=completed_microbatches
        self.sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=True,
                                       seed=config["seed"],drop_last=True)
        self.batches_per_epoch=len(self.sampler)//config["batch_size_per_gpu"]
        if not self.batches_per_epoch:
            raise ValueError("not enough samples for a complete distributed batch")
        self.iterator=None
    def __next__(self):
        if self.iterator is None:
            epoch,offset=divmod(self.position,self.batches_per_epoch)
            self.sampler.set_epoch(epoch)
            generator=torch.Generator().manual_seed(self.config["seed"]+epoch*1000+self.rank)
            kwargs=loader_options(self.config["num_workers"])
            self.loader=DataLoader(self.dataset,batch_sampler=OffsetBatchSampler(self.sampler,
                    self.config["batch_size_per_gpu"],offset),generator=generator,**kwargs)
            self.iterator=iter(self.loader)
        result=next(self.iterator)
        self.position+=1
        if self.position%self.batches_per_epoch==0:
            self.iterator=None
        return result

def loader_options(workers):
    # A 48h sample contains up to 8 global labels; bound worker prefetch memory.
    result=dict(num_workers=workers,pin_memory=False,persistent_workers=False)
    if workers:
        result["prefetch_factor"]=1
    return result

def eval_loader(dataset,batch_size,workers,rank,world):
    return DataLoader(IndexedEvalDataset(dataset),batch_size=batch_size,
                      sampler=DistributedEvalSampler(len(dataset),rank,world),
                      **loader_options(workers))

def require_finite(value,world,message):
    flag=torch.isfinite(value.detach()).all().to(torch.int32)
    if world>1:
        dist.all_reduce(flag,op=dist.ReduceOp.MIN)
    if not bool(flag):
        raise FloatingPointError(message)

def rng_snapshot(world,device):
    state={"python":random.getstate(),"numpy":np.random.get_state(),
           "torch":torch.get_rng_state(),
           "cuda":torch.cuda.get_rng_state(device) if device.type=="cuda" else None}
    if world==1:
        return [state]
    states=[None]*world
    dist.all_gather_object(states,state)
    return states

def restore_rng(states,rank,device):
    state=states[rank]
    random.setstate(state["python"]);np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type=="cuda":
        torch.cuda.set_rng_state(state["cuda"],device)

@torch.no_grad()
def measure_rollout(model,loader,horizon,lat,device,amp,world,halo_radius=3,mean=None,std=None,log=False):
    model.eval()
    fields=("global_error","global_count","halo_error","halo_count","persistence_error",
            "physical_error","physical_persistence_error")
    sums={k:torch.zeros(horizon,70,dtype=torch.float64,device=device) for k in fields}
    for key in ("component_error", "component_count", "component_persistence_error"):
        sums[key] = torch.zeros(horizon,2,18,dtype=torch.float64,device=device)
    sample_count=torch.zeros((),dtype=torch.int64,device=device)
    latitude=torch.cos(torch.deg2rad(torch.as_tensor(lat,device=device))).clamp_min(0)
    latitude=(latitude/latitude.mean()).view(1,1,-1,1)
    enabled,dtype=autocast_settings(device,amp)
    for i,batch in enumerate(loader):
        batch=move_batch(batch,device)
        sample_count+=batch["_eval_valid"].sum()
        flag=batch["_eval_valid"].view(-1,1,1,1)
        station=batch["obs_static"][:,:1].float()
        halo=F.max_pool2d(station,2*halo_radius+1,stride=1,padding=halo_radius)
        persistence=batch["era5"][:,-1].float()  # Always original ERA(T), not P(T+6).
        component_base, _ = latest_components(batch["obs_components"], batch["obs_component_mask"])
        with torch.autocast(device_type=device.type,dtype=dtype,enabled=enabled):
            for j,output in enumerate(rollout_predictions(model,batch,horizon)):
                prediction = output["weather"]
                require_finite(output["components"],world,"nonfinite predicted delay components")
                labels = lead_labels(batch,j)
                error_mm, count = component_error_sums(output,labels,batch["_eval_valid"],scaled=False)
                base_error, _ = component_error_sums(
                    {**output,"components":component_base.expand_as(output["components"])},
                    labels,batch["_eval_valid"],scaled=False)
                sums["component_error"][j] += error_mm.double()
                sums["component_count"][j] += count.double()
                sums["component_persistence_error"][j] += base_error.double()
                require_finite(prediction,world,"nonfinite rollout prediction during evaluation")
                target=batch["targets"][:,j].float()
                weights=batch["target_masks"][:,j].float()*flag*latitude
                error=(prediction.float()-target).abs()
                base=(persistence-target).abs()
                for key,value in (("global_error",error*weights),("global_count",weights),
                                  ("halo_error",error*weights*halo),("halo_count",weights*halo),
                                  ("persistence_error",base*weights)):
                    sums[key][j]+=value.sum((0,2,3),dtype=torch.float64)
                if mean is not None:
                    physical=inverse_all(prediction,mean,std)
                    truth=inverse_all(target,mean,std)
                    baseline=inverse_all(persistence,mean,std)
                    require_finite(physical,world,"nonfinite inverse-normalized weather")
                    sums["physical_error"][j]+=((physical-truth).abs()*weights).sum((0,2,3),dtype=torch.float64)
                    sums["physical_persistence_error"][j]+=((baseline-truth).abs()*weights).sum((0,2,3),dtype=torch.float64)
        if log and (i==0 or (i+1)%10==0 or i+1==len(loader)):
            print(f"evaluate rollout: batch={i+1}/{len(loader)}, horizon={6*horizon}h",flush=True)
    if world>1:
        for value in sums.values():
            dist.all_reduce(value)
        dist.all_reduce(sample_count)
    if (sums["global_count"].sum(1)<=0).any():
        raise ValueError("no valid weather targets at a forecast lead")
    sums["samples"]=int(sample_count.item())
    return sums

def write_rollout_metrics(output,sums,provenance):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    arrays={k:v.cpu().numpy() for k,v in sums.items() if k!="samples"}
    rows=[];leads=[];delay_rows=[]
    def ratio(a,b):
        return float(a/b) if b>0 else None
    for j in range(arrays["global_error"].shape[0]):
        count=arrays["global_count"][j]
        lead={"lead_hours":6*(j+1),"global_latitude_weighted_mae_normalized":ratio(arrays["global_error"][j].sum(),count.sum()),
              "halo_latitude_weighted_mae_normalized":ratio(arrays["halo_error"][j].sum(),arrays["halo_count"][j].sum()),
              "persistence_latitude_weighted_mae_normalized":ratio(arrays["persistence_error"][j].sum(),count.sum())}
        for c,name in enumerate(("zhd","zwd")):
            errors=arrays["component_error"][j,c]
            counts=arrays["component_count"][j,c]
            baselines=arrays["component_persistence_error"][j,c]
            lead[name+"_mae_mm"]=ratio(errors.sum(),counts.sum())
            lead[name+"_valid_count"]=int(counts.sum())
            lead[name+"_persistence_mae_mm"]=ratio(baselines.sum(),counts.sum())
            # 18 future frames per weather hop: T+20min,...,T+48h.
            for f in range(18):
                delay_rows.append({"lead_minutes":j*360+(f+1)*20,
                    "weather_lead_hours":6*(j+1),"component":name,
                    "mae_mm":ratio(errors[f],counts[f]),
                    "persistence_mae_mm":ratio(baselines[f],counts[f]),
                    "valid_count":int(counts[f])})
        leads.append(lead)
        for c,name in enumerate(CHANNELS):
            rows.append({"lead_hours":6*(j+1),"channel":name,"unit":channel_unit(name),
                "global_mae_normalized":ratio(arrays["global_error"][j,c],count[c]),
                "global_mae_physical":ratio(arrays["physical_error"][j,c],count[c]),
                "persistence_mae_normalized":ratio(arrays["persistence_error"][j,c],count[c]),
                "persistence_mae_physical":ratio(arrays["physical_persistence_error"][j,c],count[c]),
                "halo_mae_normalized":ratio(arrays["halo_error"][j,c],arrays["halo_count"][j,c])})
    report={"evaluation":provenance,"samples":sums["samples"],"leads":leads,
            "observation_policy":"Real history at origin; subsequent OBS inputs use predicted components and derived ZTD, never future measurements",
            "note":"Common origins require T+48 inside split by default; sample count differs from legacy 6h evaluation."}
    with (output/"rollout_mae_summary.json").open("w") as handle:
        json.dump(report,handle,indent=2,allow_nan=False)
    for name,values in (("rollout_mae_by_lead.csv",leads),("rollout_mae_by_channel.csv",rows),("rollout_delay_mae_by_frame.csv",delay_rows)):
        with (output/name).open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    return report
