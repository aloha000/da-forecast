"""Two-rank CPU/Gloo smoke for multi-forward FSDP + outer checkpoint + full BPTT.

Run: python -m torch.distributed.run --standalone --nproc_per_node=2 tests/fsdp_rollout_smoke.py
No CUDA allocations. Requires a PyTorch version supporting explicit FSDP device_id=cpu.
"""
from pathlib import Path
import os
import sys
import torch
import torch.distributed as dist
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import DAForecastNet
from model.weather_rollout import rollout_loss
from train import wrap_fsdp

def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    rank=dist.get_rank()
    torch.manual_seed(6)
    model=DAForecastNet(image_size=(40,40),era5_channels=4,era_static_channels=2,
                      out_channels=4,embed_dim=32,patch_size=4,window_size=5,
                      depths=(1,),num_heads=4,obs_channels=2)
    model=wrap_fsdp(model,torch.device("cpu"),{"amp":"fp32","fsdp":{"sync_module_states":False,
                         "forward_prefetch":False,"limit_all_gathers":True,"use_orig_params":True}})
    torch.manual_seed(10+rank)
    batch={"era5":torch.randn(1,2,4,40,40),"era_static":torch.randn(1,2,40,40),
           "obs_ztd":torch.randn(1,3,2,40,40),"obs_mask":torch.ones(1,3,2,40,40),
           "obs_static":torch.ones(1,3,40,40),"rollout_time_features":torch.randn(1,8,12),
           "targets":torch.randn(1,8,4,40,40),"target_masks":torch.ones(1,8,4,40,40)}
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
    for update in range(2):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(2):
            loss,_=rollout_loss(model,batch,8,torch.linspace(50,40,40),
                               {"rollout_checkpoint":True,"loss_mode":"global_only"})
            (loss/2).backward()
        norm=model.clip_grad_norm_(1.)
        if not torch.isfinite(norm):
            raise RuntimeError("nonfinite gradient norm")
        optimizer.step()
        print(f"rank={rank} update={update+1} eight-hop FSDP loss={loss.item():.6f} norm={norm.item():.6f}",flush=True)
    dist.destroy_process_group()

if __name__=="__main__":
    main()
