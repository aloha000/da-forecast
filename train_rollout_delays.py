#!/usr/bin/env python3
"""One shared model: 1..8 six-hour rollout steps, 500 optimizer updates per stage."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import Subset
from data.delay_sequence import make_rollout_dataset
from data.rollout_identity import data_identity, file_identity
from model import DAForecastNet
from model.delay_rollout import rollout_loss
from delay_rollout_utils import (setup_rollout_distributed,stage_for_update,BatchStream,eval_loader,
                          require_finite,rng_snapshot,restore_rng,measure_rollout)
from train import (load_config,resolve_data_paths,seed_everything,initialize_model_weights,
                   apply_freeze_policy,wrap_fsdp,autocast_settings,move_batch,save_checkpoints,load_checkpoint)

def validate_config(config):
    m,t,c=config["model"],config["train"],config["curriculum"]
    from model.ztd_rollout import check_ztd_config
    check_ztd_config(config["data"],m)
    if not m.get("predict_delay_components"):
        raise ValueError("joint rollout requires weather + ZHD/ZWD heads")
    if config["data"].get("zero_delay_inputs"):
        raise ValueError("use model.zero_delay_inputs, not data.zero_delay_inputs; retain true labels")
    if c.get("loss_aggregation")!="mean_all_active_leads":
        raise ValueError("loss_aggregation must be mean_all_active_leads")
    if t.get("checkpoint_metric")!="weather_global":
        raise ValueError("stage checkpoints selected by final-lead weather_global")
    for key in ("zhd_loss_weight","zwd_loss_weight"):
        if not math.isfinite(float(t[key])) or float(t[key])<0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if "ztd_loss_weight" in t:
        raise ValueError("no independent ZTD loss in component rollout")
    if not 1<=int(c["max_steps"])<=8 or int(c["updates_per_stage"])<1:
        raise ValueError("invalid curriculum limits")
    expected_policy = "zero_all_steps" if m.get("zero_delay_inputs", False) else "predicted_components"
    if c.get("observation_policy") != expected_policy:
        raise ValueError(f"observation_policy must be {expected_policy}")
    if t.get("loss_mode")!="global_only":
        raise ValueError("this curriculum uses global_only weather supervision")
    if any(t.get(k,False) for k in ("freeze_fuxi","freeze_obs","freeze_delay_heads","freeze_component_adapter")):
        raise ValueError("joint curriculum requires both branches unfrozen")
    if t["amp"] not in ("bf16","fp32"):
        raise ValueError("use bf16 or fp32; update counting does not include skipped FP16 scaler steps")
    for key in ("batch_size_per_gpu","gradient_accumulation","validate_every","save_every"):
        if int(t[key])<1:
            raise ValueError(f"{key} must be positive")
    if not 0<float(t["min_learning_rate"])<=float(t["learning_rate"]):
        raise ValueError("invalid learning rate range")

def lr_factor(step,updates_per_stage,warmup,min_ratio):
    local=step%updates_per_stage
    warmup=min(int(warmup),max(0,updates_per_stage-1))
    if warmup and local<warmup:
        return (local+1)/warmup
    progress=(local-warmup)/max(1,updates_per_stage-warmup-1)
    return min_ratio+.5*(1-min_ratio)*(1+math.cos(math.pi*progress))

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/train_rollout_48h_predict_delays.json")
    p.add_argument("--init-checkpoint")
    p.add_argument("--resume")
    p.add_argument("--check-data",action="store_true")
    p.add_argument("--output-dir")
    p.add_argument("--overfit-samples",type=int,default=0)
    p.add_argument("--updates-per-stage",type=int,help="override for isolated smoke experiments; production default is 500")
    p.add_argument("--gradient-accumulation",type=int,help="override for isolated smoke experiments")
    p.add_argument("--stop-after-updates",type=int,default=0,help="clean early stop for smoke/resume; does not alter curriculum schedule")
    return p.parse_args()

def main():
    args=parse_args()
    if args.resume and args.init_checkpoint:
        raise ValueError("choose --resume OR --init-checkpoint")
    config=load_config(args.config)
    if args.updates_per_stage is not None:
        config["curriculum"]["updates_per_stage"]=args.updates_per_stage
    if args.gradient_accumulation is not None:
        config["train"]["gradient_accumulation"]=args.gradient_accumulation
    validate_config(config)
    data=resolve_data_paths(args.config,config["data"])
    tc,cc=config["train"],config["curriculum"]
    if args.output_dir:
        tc["output_dir"]=args.output_dir
    if args.overfit_samples<0 or args.stop_after_updates<0:
        raise ValueError("negative smoke limits")
    tc["overfit_samples"]=args.overfit_samples
    max_steps,per_stage=int(cc["max_steps"]),int(cc["updates_per_stage"])
    total=max_steps*per_stage
    dataset=make_rollout_dataset(data,"train",max_steps,1)
    val_dataset=make_rollout_dataset(data,"val",max_steps,1)
    config["observation_normalization"]=dataset.observation_normalization
    config["rollout_data_identity"]=data_identity(data)
    if int(os.environ.get("RANK", "0")) == 0:
        policy = ("ZTD(normalized)=0, ZHD/ZWD(mm)=0 at EVERY hop, including residual base; "
                  "retain masks/static fields and true supervision"
                  if config["model"].get("zero_delay_inputs", False) else
                  "real history at T; predicted ZHD/ZWD and derived ZTD at later hops")
        print(f"OBS policy: {policy}; shared model, all branches/heads trainable", flush=True)
    if args.check_data:
        # Inspect one horizon only; all origins already checked through max_steps.
        sample=dataset[0]
        print(f"train={len(dataset)} val={len(val_dataset)}; common origins support {6*max_steps}h")
        print(json.dumps(dataset.observation_normalization,indent=2))
        for key,value in sample.items():
            print(key,tuple(value.shape),value.dtype,bool(torch.isfinite(value).all()))
        print("Future observations are supervision ONLY. Shapes above are raw dataset tensors; "
              "model.zero_delay_inputs, when enabled, zeros histories inside every model forward.")
        return
    rank,world,device=setup_rollout_distributed()
    fsdp=bool(tc.get("fsdp",{}).get("enabled",True))
    if (world>1 and not fsdp) or (fsdp and world<2):
        raise ValueError("use >=2 GPU processes with FSDP, or explicitly disable FSDP for CPU/single-device tests")
    seed_everything(tc["seed"],rank)
    output=Path(tc["output_dir"])
    if not args.resume and (output/"resolved_config.json").exists():
        raise FileExistsError("output already contains an experiment; use --resume or a new --output-dir")
    if args.overfit_samples:
        indices=list(range(min(args.overfit_samples,len(dataset))))
        train_data=Subset(dataset,indices);val_data=Subset(dataset,indices)
    else:
        train_data,val_data=dataset,val_dataset
    model=DAForecastNet(**config["model"]).to(device)
    initialize=args.init_checkpoint or config.get("initial_checkpoint")
    if initialize and not args.resume:
        path=Path(initialize)
        if not path.is_absolute():
            path=Path(args.config).resolve().parent.parent/path
        problem=[None]
        if rank==0:
            try:
                ck=torch.load(path,map_location="cpu",weights_only=False,mmap=True)
                report=initialize_model_weights(model,ck,config["model"])
                print(f"initialization={path}: {report}",flush=True)
                del ck
            except Exception as exc:
                problem[0]=str(exc)
        if world>1:
            dist.broadcast_object_list(problem,src=0)
        if problem[0]:
            raise ValueError(problem[0])
        config["initialization"]=file_identity(path)
    elif not args.resume:
        raise ValueError("provide initial_checkpoint in config or --init-checkpoint")
    apply_freeze_policy(model,tc)
    if rank==0:
        print(f"parameters={sum(p.numel() for p in model.parameters())}, "
              f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)}",flush=True)
    if fsdp:
        if not tc["fsdp"].get("sync_module_states",True):
            raise ValueError("FSDP initialization requires sync_module_states=true")
        model=wrap_fsdp(model,device,tc)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                               lr=tc["learning_rate"],weight_decay=tc["weight_decay"])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:lr_factor(step,per_stage,
                                tc["warmup_steps"],tc["min_learning_rate"]/tc["learning_rate"]))
    done=0;active_stage=None;best=float("inf");rng=None
    accumulation=int(tc["gradient_accumulation"])
    if args.resume:
        saved=torch.load(args.resume,map_location="cpu",weights_only=False,mmap=True)
        old=saved["config"]
        for key in ("model","curriculum","rollout_data_identity","observation_normalization"):
            if old.get(key)!=config.get(key):
                raise ValueError(f"resume {key} mismatch")
        for key in ("amp","seed","batch_size_per_gpu","gradient_accumulation","learning_rate",
                    "min_learning_rate","warmup_steps","weight_decay","gradient_clip","rollout_checkpoint","overfit_samples",
                    "zhd_loss_weight","zwd_loss_weight","checkpoint_metric","loss_mode","halo_radius"):
            if old["train"].get(key)!=tc.get(key):
                raise ValueError(f"resume train.{key} mismatch")
        state=old["curriculum_state"]
        if state["world_size"]!=world:
            raise ValueError("resume requires the same GPU process count")
        done=int(saved["step"]);active_stage=state["stage"];best=saved["best_val"]
        rng=old["rollout_rng_states"]
        config["initialization"]=old.get("initialization")
        del saved
        load_checkpoint(args.resume,model,optimizer,scheduler,rank)
    if done>total:
        raise ValueError("checkpoint exceeds configured curriculum")
    if rank==0:
        output.mkdir(parents=True,exist_ok=True)
        with (output/"resolved_config.json").open("w") as handle:
            json.dump(config,handle,indent=2)
        print(f"curriculum: {max_steps} stages x {per_stage} optimizer updates = {total}; resume={done}",flush=True)
        print(f"world_size={world}, effective_batch={world*tc['batch_size_per_gpu']*accumulation}, full BPTT; weather + scaled ZHD/ZWD losses averaged over active leads",flush=True)
    if rng:
        restore_rng(rng,rank,device)
    lat=torch.as_tensor(dataset.lat,device=device)
    amp_enabled,amp_dtype=autocast_settings(device,tc["amp"])
    stream=None
    stop=min(total,args.stop_after_updates) if args.stop_after_updates else total
    while done<stop:
        stage=stage_for_update(done,per_stage,max_steps)
        if stream is None or active_stage!=stage:
            if active_stage!=stage:
                best=float("inf")
            active_stage=stage
            dataset.horizon=stage;val_dataset.horizon=stage
            stream=BatchStream(train_data,tc,rank,world,done*accumulation)
            if rank==0:
                print(f"stage={stage}/{max_steps}, horizon={stage*6}h, stage_updates={done%per_stage}/{per_stage}",flush=True)
        model.train();optimizer.zero_grad(set_to_none=True)
        started=time.monotonic();loss_sum=torch.zeros(stage,4,device=device)
        used_lr=optimizer.param_groups[0]["lr"]
        for _ in range(accumulation):
            batch=move_batch(next(stream),device)
            with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=amp_enabled):
                loss,per_lead=rollout_loss(model,batch,stage,lat,tc)
            require_finite(loss,world,"nonfinite rollout loss; no optimizer update performed")
            # Synchronize every microbatch to avoid unsharded FSDP no_sync gradients.
            (loss/accumulation).backward()
            loss_sum+=per_lead/accumulation
            del batch,loss,per_lead
        norm=model.clip_grad_norm_(tc["gradient_clip"]) if fsdp else torch.nn.utils.clip_grad_norm_(model.parameters(),tc["gradient_clip"])
        require_finite(norm,world,"nonfinite gradients; no optimizer update performed")
        optimizer.step();scheduler.step();done+=1
        if world>1:
            dist.all_reduce(loss_sum);loss_sum/=world
        if rank==0 and (done==1 or done%tc.get("log_interval",10)==0 or done%per_stage==0):
            print(f"stage={stage} lead={stage*6}h stage_update={(done-1)%per_stage+1}/{per_stage} "
                  f"update={done}/{total} lr={used_lr:.3e} loss={loss_sum[:,0].mean().item():.6f} "
                  f"per_lead_columns=[total,weather_normalized,zhd_mm,zwd_mm] "
                  f"per_lead={loss_sum.cpu().tolist()} seconds={time.monotonic()-started:.1f}",flush=True)
        end_stage=done%per_stage==0
        validation=done%tc["validate_every"]==0 or end_stage
        improved=False
        if validation:
            loader=eval_loader(val_data,tc.get("val_batch_size",1),tc["num_workers"],rank,world)
            measured=measure_rollout(model,loader,stage,dataset.lat,device,tc["amp"],world,tc.get("halo_radius",3))
            metrics=measured["global_error"].sum(1)/measured["global_count"].sum(1)
            score=metrics[-1].item() # Select within-stage best at its final forecast lead.
            improved=score<best
            if improved: best=score
            if rank==0:
                delay_errors=measured["component_error"].sum(2)
                delay_counts=measured["component_count"].sum(2)
                delay_mae=[[float(e/n) if n>0 else None for e,n in zip(es,ns)]
                           for es,ns in zip(delay_errors.cpu(),delay_counts.cpu())]
                record={"val_delay_mae_mm_by_lead":delay_mae,
                        "val_delay_valid_count_by_lead":delay_counts.cpu().tolist(),"update":done,"stage":stage,"lead_hours":stage*6,"val_global_by_lead":metrics.cpu().tolist(),"samples":measured["samples"]}
                print("validation: "+json.dumps(record),flush=True)
                with (output/"validation.jsonl").open("a") as handle:
                    handle.write(json.dumps(record)+"\n")
            del loader,measured
        if done%tc["save_every"]==0 or validation or done==stop:
            config["curriculum_state"]={"stage":stage,"completed_updates":done,"world_size":world,
                                        "completed_microbatches":done*accumulation}
            config["rollout_rng_states"]=rng_snapshot(world,device)
            paths=[output/"last.pt"]
            if improved:
                best_path=output/f"stage_{stage*6:02d}h"/"best.pt"
                if rank==0: best_path.parent.mkdir(parents=True,exist_ok=True)
                paths.append(best_path)
            save_checkpoints(paths,model,optimizer,scheduler,stage-1,done,best,config,rank)
            if rank==0:
                print(f"checkpoint saved: update={done}",flush=True)
        if end_stage:
            stream=None
    if rank==0:
        print(f"stopped cleanly at {done}/{total} optimizer updates",flush=True)
    if world>1:
        dist.destroy_process_group()

if __name__=="__main__":
    main()
