"""Two-GPU smoke test for the FSDP training/checkpoint path.

Run with::

    torchrun --standalone --nproc_per_node=2 tests/fsdp_smoke.py
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from model import DAForecastNet
from train import load_checkpoint, save_checkpoints, wrap_fsdp


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(1234 + rank)

    def build():
        raw = DAForecastNet(
            image_size=(40, 40), era5_channels=4, era5_frames=2,
            era_static_channels=2, obs_static_channels=3, out_channels=4,
            embed_dim=32, patch_size=4, window_size=5, depths=(1,),
            num_heads=4, use_obs=False,
        ).to(device)
        return wrap_fsdp(
            raw, device,
            {"amp": "bf16", "fsdp": {
                "sync_module_states": True, "forward_prefetch": True,
                "limit_all_gathers": True, "use_orig_params": True,
            }},
        )

    model = build()
    assert sum(isinstance(module, FSDP) for module in model.modules()) == 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    inputs = (
        torch.randn(1, 2, 4, 40, 40, device=device),
        torch.randn(1, 2, 40, 40, device=device),
        torch.randn(1, 12, device=device),
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(*inputs).float().square().mean()
    loss.backward()
    model.clip_grad_norm_(1.0)
    optimizer.step()
    scheduler.step()

    holder = [tempfile.mkdtemp(prefix="da_fsdp_smoke_") if rank == 0 else None]
    dist.broadcast_object_list(holder, src=0)
    path = Path(holder[0]) / "checkpoint.pt"
    save_checkpoints(
        [path], model, optimizer, scheduler, 0, 1, float(loss.detach()),
        {"train": {"loss_mode": "global_only"}}, rank,
    )
    dist.barrier()
    if rank == 0:
        assert path.is_file() and path.stat().st_size > 0

    restored = build()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer, lambda _: 1.0
    )
    metadata = load_checkpoint(
        path, restored, restored_optimizer, restored_scheduler, rank
    )
    assert metadata["epoch"] == 0 and metadata["step"] == 1
    dist.barrier()
    if rank == 0:
        shutil.rmtree(holder[0])
        print("FSDP smoke test: OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
