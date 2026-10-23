"""Small multi-GPU regression for evaluation and resume checkpoint loading.

Run from the project root (two or four GPUs)::

    python -m torch.distributed.run --standalone --nproc_per_node=4 \
        tests/fsdp_checkpoint_regression.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType,
)

from evaluate import load_distributed_model
from model import DAForecastNet
from train import load_checkpoint, save_checkpoints, wrap_fsdp


def full_state(model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
        ):
            return {key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()}


def assert_state(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, msg=key)


def run_case(rank, device, freeze_core):
    model_config = dict(
        image_size=(40, 40), era5_channels=4, era5_frames=2,
        era_static_channels=2, obs_static_channels=3, out_channels=4,
        embed_dim=32, patch_size=4, window_size=5, depths=(1,),
        num_heads=4, use_obs=True,
    )
    train_config = dict(amp="bf16", loss_mode="global_only", fsdp=dict(
        sync_module_states=True, forward_prefetch=True, use_orig_params=True,
    ))
    config = {"model": model_config, "train": train_config}

    def build(seed):
        torch.manual_seed(seed + rank)
        raw = DAForecastNet(**model_config).to(device)
        if freeze_core:
            for module in (raw.era_encoder, raw.time_embed, raw.backbone, raw.decoder):
                module.requires_grad_(False)
        model = wrap_fsdp(raw, device, train_config)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        return model, optimizer, scheduler

    model, optimizer, scheduler = build(123)
    torch.manual_seed(321 + rank)
    inputs = (
        torch.randn(1, 2, 4, 40, 40, device=device),
        torch.randn(1, 2, 40, 40, device=device),
        torch.randn(1, 12, device=device),
        torch.randn(1, 3, 1, 40, 40, device=device),
        torch.ones(1, 3, 1, 40, 40, device=device),
        torch.randn(1, 3, 40, 40, device=device),
    )

    def step(net, opt, sched):
        net.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = net(*inputs).float().square().mean()
        loss.backward()
        opt.step()
        sched.step()
        return float(loss.detach())

    def predict(net):
        net.eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            return net(*inputs).clone()

    loss = step(model, optimizer, scheduler)
    temporary = tempfile.TemporaryDirectory(prefix="fsdp_checkpoint_regression_") if rank == 0 else None
    paths = [str(Path(temporary.name) / "checkpoint.pt") if rank == 0 else None]
    dist.broadcast_object_list(paths, src=0)
    path = paths[0]
    save_checkpoints([path], model, optimizer, scheduler, 0, 1, loss, config, rank)
    dist.barrier()
    expected_state = full_state(model)
    expected_prediction = predict(model)

    restored, restored_optimizer, restored_scheduler = build(456)
    metadata = load_checkpoint(path, restored, restored_optimizer, restored_scheduler, rank)
    assert metadata["epoch"] == 0 and metadata["step"] == 1
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert_state(full_state(restored), expected_state)
    torch.testing.assert_close(predict(restored), expected_prediction, rtol=0, atol=0)
    original_opt, loaded_opt = optimizer.state_dict(), restored_optimizer.state_dict()
    assert original_opt["param_groups"] == loaded_opt["param_groups"]
    for param_id, values in original_opt["state"].items():
        assert_state(loaded_opt["state"][param_id], values)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False) if rank == 0 else None
    # Even a saved config with synchronization disabled must evaluate correctly.
    eval_config = {"train": {**train_config, "fsdp": {"sync_module_states": False}}}
    evaluated = load_distributed_model(checkpoint, eval_config, model_config, device, rank)
    assert_state(full_state(evaluated), expected_state)
    torch.testing.assert_close(predict(evaluated), expected_prediction, rtol=0, atol=0)

    # Verify optimizer restoration by matching the next update, not just metadata.
    step(model, optimizer, scheduler)
    step(restored, restored_optimizer, restored_scheduler)
    assert_state(full_state(restored), full_state(model))

    bad_checkpoint = {"model": {}} if rank == 0 else None
    try:
        load_distributed_model(bad_checkpoint, eval_config, model_config, device, rank)
    except RuntimeError as error:
        assert "checkpoint model load failed" in str(error)
    else:
        raise AssertionError("missing checkpoint keys must fail on all ranks")

    dist.barrier()
    if rank == 0:
        temporary.cleanup()
        print(f"PASS freeze_core={freeze_core}: full state/output/optimizer/next update "
              "match exactly; invalid checkpoint rejected", flush=True)


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.backends.cudnn.deterministic = True
    try:
        for freeze_core in (False, True):
            run_case(dist.get_rank(), device, freeze_core)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
