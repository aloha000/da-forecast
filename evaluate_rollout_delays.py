#!/usr/bin/env python3
"""Evaluate a single shared model at 6,12,...,48h with predicted weather and delay feedback."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from data.delay_sequence import make_rollout_dataset
from data.rollout_identity import data_identity
from model import DAForecastNet
from evaluate import load_distributed_model
from train import load_config,resolve_data_paths
from train_rollout_delays import validate_config
from delay_rollout_utils import setup_rollout_distributed,eval_loader,measure_rollout,write_rollout_metrics

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/train_rollout_48h_predict_delays.json")
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--split",choices=("val","test"),default="test")
    p.add_argument("--output-dir",required=True)
    p.add_argument("--steps",type=int,default=8)
    p.add_argument("--batch-size",type=int,default=1)
    p.add_argument("--num-workers",type=int,default=1)
    args=p.parse_args()
    if args.batch_size<1 or args.num_workers<0:
        p.error("invalid batch size/workers")
    config=load_config(args.config)
    data=resolve_data_paths(args.config,config["data"])
    rank,world,device=setup_rollout_distributed()
    ck=None;metadata=[None];problem=[None]
    if rank==0:
        try:
            ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False,mmap=True)
            metadata[0]={"config":ck["config"],"step":ck["step"]}
        except Exception as exc:
            problem[0]=str(exc)
    if world>1:
        dist.broadcast_object_list(problem,src=0);dist.broadcast_object_list(metadata,src=0)
    if problem[0]: raise ValueError(problem[0])
    saved=metadata[0]["config"];validate_config(saved)
    max_steps=int(saved["curriculum"]["max_steps"])
    if not 1<=args.steps<=max_steps:
        raise ValueError("requested steps exceed the checkpoint curriculum")
    if saved["rollout_data_identity"]!=data_identity(data):
        raise ValueError("evaluation data/static/normalization/split identity differs from training")
    ds=make_rollout_dataset(data,args.split,max_steps,args.steps)
    if saved["observation_normalization"]!=ds.observation_normalization:
        raise ValueError("observation normalization differs from checkpoint")
    if world>1:
        model=load_distributed_model(ck,saved,saved["model"],device,rank)
    else:
        model=DAForecastNet(**saved["model"]).to(device)
        model.load_state_dict(ck["model"],strict=True)
    del ck
    loader=eval_loader(ds,args.batch_size,args.num_workers,rank,world)
    mean=np.load(Path(data["mean_std_dir"])/"mean_era5.npy").astype(np.float32).reshape(1,70,1,1)
    std=np.load(Path(data["mean_std_dir"])/"std_era5.npy").astype(np.float32).reshape(1,70,1,1)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
        raise ValueError("invalid ERA normalization")
    provenance={"checkpoint":str(Path(args.checkpoint).resolve()),"completed_updates":metadata[0]["step"],
                "trained_stage":saved["curriculum_state"]["stage"],"evaluated_steps":args.steps,
                "amp":saved["train"]["amp"],"world_size":world,"split":args.split,
                "origin_first":str(ds.sample_time(0)),"origin_last":str(ds.sample_time(len(ds)-1)),
                "observation_normalization":ds.observation_normalization}
    provenance["observation_policy"] = saved["curriculum"]["observation_policy"]
    provenance["zero_delay_inputs"] = bool(saved["model"].get("zero_delay_inputs", False))
    if rank==0:
        print(json.dumps(provenance,indent=2),flush=True)
        if args.steps>saved["curriculum_state"]["stage"]:
            print("WARNING: evaluating beyond this checkpoint's trained rollout horizon",flush=True)
    sums=measure_rollout(model,loader,args.steps,ds.lat,device,saved["train"]["amp"],world,
                         saved["train"].get("halo_radius",3),
                         torch.as_tensor(mean,device=device),torch.as_tensor(std,device=device),log=rank==0)
    if rank==0:
        report=write_rollout_metrics(args.output_dir,sums,provenance)
        print(json.dumps(report,indent=2),flush=True)
    if world>1:
        dist.destroy_process_group()

if __name__=="__main__":
    main()
