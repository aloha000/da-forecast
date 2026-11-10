"""Two-process CPU FSDP smoke: real small joint model, eight differentiable hops."""
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from train import wrap_fsdp
from model.delay_rollout import rollout_predictions
from tests.test_delay_components import tiny_model, component_batch
from tests.test_da_forecast_model import make_inputs

def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    torch.manual_seed(33)
    model=wrap_fsdp(tiny_model(),torch.device("cpu"),{
        "amp":"fp32","fsdp":{"sync_module_states":False,"forward_prefetch":False}})
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
    batch=make_inputs();batch.update(component_batch(40,40))
    batch["rollout_time_features"]=torch.randn(1,8,12)
    for update in range(2):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(2):
            loss=0.
            for output in rollout_predictions(model,batch,8,checkpoint_steps=True):
                loss=loss+(output["weather"].square().mean()+output["components"].square().mean()/1e6)/8
            (loss/2).backward()
        norm=model.clip_grad_norm_(1.)
        assert torch.isfinite(norm) and norm>0
        optimizer.step()
        print(f"rank={dist.get_rank()} update={update+1} loss={float(loss.detach()):.6f} norm={float(norm):.6f}",flush=True)
    dist.destroy_process_group()

if __name__=="__main__":
    main()
