"""FSDP training for the global DA forecast model."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time
import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import DataLoader, DistributedSampler, Subset

from data import DAForecastDataset
from model import DAForecastNet
from model.fuxi import SwinBlock
from model.fuxi_pretrained import load_fuxi_pretrained


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_da_forecast.json")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--pretrained-fuxi", default=None,
                        help="initialize FuXi forecast core before FSDP wrapping")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def seed_everything(seed: int, rank: int):
    seed = int(seed) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        if world_size > 1:
            raise RuntimeError("distributed training requires CUDA/NCCL")
        device = torch.device("cpu")
    return rank, local_rank, world_size, device


def resolve_data_paths(config_path: str, data_config: dict):
    root = Path(config_path).resolve().parent.parent
    result = dict(data_config)
    for key in ("era5_path", "ztd_path", "conds_path", "global_lat_path", "global_lon_path"):
        value = Path(result[key])
        result[key] = str(value if value.is_absolute() else root / value)
    return result


def make_dataset(data_config: dict, split_name: str):
    return DAForecastDataset(
        era5_path=data_config["era5_path"],
        ztd_path=data_config["ztd_path"],
        split=data_config[f"{split_name}_split"],
        conds_path=data_config["conds_path"],
        global_lat_path=data_config["global_lat_path"],
        global_lon_path=data_config["global_lon_path"],
        obs_history_hours=data_config["obs_history_hours"],
        obs_interval_minutes=data_config["obs_interval_minutes"],
    )


def print_data_check(train_dataset, val_dataset):
    sample = train_dataset[0]
    print(f"train samples: {len(train_dataset)}, {train_dataset.sample_time(0)} -> "
          f"{train_dataset.sample_time(len(train_dataset)-1)}")
    print(f"val samples:   {len(val_dataset)}, {val_dataset.sample_time(0)} -> "
          f"{val_dataset.sample_time(len(val_dataset)-1)}")
    for name, value in sample.items():
        finite = bool(torch.isfinite(value).all())
        print(f"{name:14s} shape={tuple(value.shape)!s:22s} finite={finite}")
    valid = sample["obs_mask"].sum().item()
    total = sample["obs_mask"].numel()
    print(f"observed ZTD cells: {int(valid)}/{total} ({100.0 * valid / total:.2f}%)")
    print(
        "alignment: ERA5(t-6h,t) + "
        f"ZTD(t-{train_dataset.obs_history_hours}h..t every "
        f"{train_dataset.obs_interval_minutes}min; "
        f"{train_dataset.obs_frames} frames) -> ERA5(t+6h)"
    )


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def weighted_losses(
    prediction,
    target,
    target_mask,
    obs_static,
    lat,
    halo_weight,
    radius,
    loss_mode="global_only",
):
    error = (prediction - target).abs()
    lat_weight = torch.cos(torch.deg2rad(lat)).clamp_min(0.0)
    lat_weight = lat_weight / lat_weight.mean()
    lat_weight = lat_weight.view(1, 1, -1, 1)
    valid = target_mask.to(error.dtype)
    global_denominator = (valid * lat_weight).sum().clamp_min(1.0)
    global_loss = (error * valid * lat_weight).sum() / global_denominator

    station = obs_static[:, :1].to(error.dtype)
    kernel = 2 * int(radius) + 1
    halo = F.max_pool2d(station, kernel_size=kernel, stride=1, padding=radius)
    halo_valid = valid * halo
    halo_denominator = (halo_valid * lat_weight).sum().clamp_min(1.0)
    halo_loss = (error * halo_valid * lat_weight).sum() / halo_denominator
    if loss_mode == "halo_only":
        total_loss = halo_loss
    elif loss_mode == "global_only":
        total_loss = global_loss
    elif loss_mode == "global_plus_halo":
        total_loss = global_loss + float(halo_weight) * halo_loss
    else:
        raise ValueError(
            f"unsupported loss_mode={loss_mode!r}; expected halo_only, "
            "global_only, or global_plus_halo"
        )
    return total_loss, global_loss, halo_loss


def autocast_settings(device, mode: str):
    if device.type != "cuda" or mode == "fp32":
        return False, torch.float32
    if mode == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("this GPU does not support bf16; set train.amp to fp16")
        return True, torch.bfloat16
    if mode == "fp16":
        return True, torch.float16
    raise ValueError(f"unsupported amp mode {mode!r}")


def fsdp_mixed_precision(device, mode: str):
    """Build FSDP's parameter and reduction precision policy."""
    if device.type != "cuda" or mode == "fp32":
        return None
    if mode == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("this GPU does not support bf16; set train.amp to fp16")
        dtype = torch.bfloat16
    elif mode == "fp16":
        dtype = torch.float16
    else:
        raise ValueError(f"unsupported amp mode {mode!r}")
    return MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
        cast_forward_inputs=True,
    )


def wrap_fsdp(model, device, train_config):
    """FULL_SHARD the root model and wrap each FuXi Swin block separately."""
    fsdp_config = train_config.get("fsdp", {})
    return FSDP(
        model,
        auto_wrap_policy=ModuleWrapPolicy({SwinBlock}),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        mixed_precision=fsdp_mixed_precision(device, train_config["amp"]),
        device_id=device,
        sync_module_states=bool(fsdp_config.get("sync_module_states", True)),
        forward_prefetch=bool(fsdp_config.get("forward_prefetch", True)),
        limit_all_gathers=bool(fsdp_config.get("limit_all_gathers", True)),
        use_orig_params=bool(fsdp_config.get("use_orig_params", True)),
    )


def forward_model(model, batch):
    return model(
        batch["era5"],
        batch["era_static"],
        batch["time_features"],
        batch["obs_ztd"],
        batch["obs_mask"],
        batch["obs_static"],
    )


@torch.no_grad()
def validate(model, loader, device, lat, train_config, amp_enabled, amp_dtype, world_size):
    model.eval()
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            prediction = forward_model(model, batch)
            loss, global_loss, halo_loss = weighted_losses(
                prediction,
                batch["target"],
                batch["target_mask"],
                batch["obs_static"],
                lat,
                train_config["halo_weight"],
                train_config["halo_radius"],
                train_config.get("loss_mode", "global_plus_halo"),
            )
        totals += torch.tensor(
            [loss.item(), global_loss.item(), halo_loss.item(), 1.0],
            device=device,
            dtype=torch.float64,
        )
    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return tuple((totals[:3] / totals[3].clamp_min(1.0)).tolist())


def save_checkpoints(
    paths, model, optimizer, scheduler, epoch, step, best_val, config, rank
):
    """Collect one full CPU checkpoint and write it on rank 0.

    All ranks must enter this function because FSDP state gathering is a
    collective. The expensive gather is done once even when both last and best
    checkpoints are written.
    """
    if isinstance(model, FSDP):
        state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"FSDP\.state_dict_type\(\) and FSDP\.set_state_dict_type\(\) are being deprecated\.",
                category=FutureWarning,
                module=r"torch\.distributed\.fsdp\.fully_sharded_data_parallel",
            )
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, state_config, optim_config
            ):
                model_state = model.state_dict()
                optimizer_state = FSDP.optim_state_dict(model, optimizer)
    else:
        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict()
    if rank != 0:
        return
    state = {
        "model": model_state,
        "optimizer": optimizer_state,
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val": best_val,
        "config": config,
        "distributed": "fsdp_full_shard" if isinstance(model, FSDP) else "none",
    }
    for value in paths:
        path = Path(value)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)


def load_checkpoint(path, model, optimizer, scheduler, rank):
    """Load a full checkpoint and distribute sharded model/optimizer state."""
    if not isinstance(model, FSDP):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        return checkpoint

    # FULL_STATE_DICT loading requires the full model state on EVERY rank.
    # mmap avoids eagerly duplicating the multi-GB CPU checkpoint in each process;
    # the local workers share the OS file cache. Optimizer state is still scattered.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    metadata = [
        None if checkpoint is None else {
            key: checkpoint[key]
            for key in ("epoch", "step", "best_val", "config", "scheduler")
        }
    ]
    dist.broadcast_object_list(metadata, src=0)
    metadata = metadata[0]
    state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"FSDP\.state_dict_type\(\) and FSDP\.set_state_dict_type\(\) are being deprecated\.",
            category=FutureWarning,
            module=r"torch\.distributed\.fsdp\.fully_sharded_data_parallel",
        )
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_config):
            # rank0_only controls exporting a state dict, NOT load-time broadcast.
            model.load_state_dict(checkpoint["model"], strict=True)
    sharded_optimizer = FSDP.scatter_full_optim_state_dict(
        checkpoint["optimizer"] if rank == 0 else None,
        model,
        optim=optimizer,
    )
    optimizer.load_state_dict(sharded_optimizer)
    scheduler.load_state_dict(metadata["scheduler"])
    return metadata


def main():
    args = parse_args()
    if args.resume and args.pretrained_fuxi:
        raise ValueError("--pretrained-fuxi initializes a new run; do not combine it with --resume")
    config = load_config(args.config)
    data_config = resolve_data_paths(args.config, config["data"])
    train_config = config["train"]
    if args.output_dir:
        train_config["output_dir"] = args.output_dir
    if args.epochs is not None:
        train_config["epochs"] = args.epochs

    # Data check deliberately runs without initializing distributed training.
    if args.check_data:
        train_dataset = make_dataset(data_config, "train")
        val_dataset = make_dataset(data_config, "val")
        print_data_check(train_dataset, val_dataset)
        return

    rank, _, world_size, device = setup_distributed()
    fsdp_enabled = bool(train_config.get("fsdp", {}).get("enabled", True))
    if fsdp_enabled and world_size < 2:
        raise RuntimeError(
            "FSDP training requires at least 2 GPUs; launch with torchrun "
            "--nproc_per_node=4. Use --check-data for a single-process data check."
        )
    seed_everything(train_config["seed"], rank)
    train_dataset = make_dataset(data_config, "train")
    val_dataset = make_dataset(data_config, "val")
    if args.overfit_samples:
        count = min(args.overfit_samples, len(train_dataset))
        train_dataset = Subset(train_dataset, list(range(count)))
        val_dataset = train_dataset

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
        seed=train_config["seed"], drop_last=True
    ) if world_size > 1 else None
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    ) if world_size > 1 else None
    loader_kwargs = {
        "batch_size": train_config["batch_size_per_gpu"],
        "num_workers": train_config["num_workers"],
        "pin_memory": device.type == "cuda",
        "persistent_workers": train_config["num_workers"] > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, sampler=val_sampler, shuffle=False, drop_last=False, **loader_kwargs
    )

    model = DAForecastNet(**config["model"]).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if args.pretrained_fuxi:
        sync_states = bool(train_config.get("fsdp", {}).get("sync_module_states", True))
        if fsdp_enabled and not sync_states:
            raise ValueError("--pretrained-fuxi with FSDP requires sync_module_states=true")
        # Only rank 0 reads the multi-GB checkpoint. FSDP broadcasts its
        # initialized module state to other ranks during wrapping.
        if rank == 0 or not fsdp_enabled:
            loaded = load_fuxi_pretrained(model, args.pretrained_fuxi)
            if rank == 0:
                print(
                    f"FuXi pretrained: {loaded['state_tensors']} state tensors, "
                    f"{loaded['parameters']:,}/{loaded['total_parameters']:,} "
                    f"parameters ({loaded['parameters'] / loaded['total_parameters']:.1%}); "
                    "only OBS encoder and gated fusion remain newly initialized",
                    flush=True,
                )
    if train_config.get("freeze_fuxi", False):
        for module in (
            model.era_encoder, model.time_embed, model.backbone, model.decoder
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise ValueError(
            "no trainable parameters: enable model.use_obs or set train.freeze_fuxi=false"
        )
    if fsdp_enabled:
        model = wrap_fsdp(model, device, train_config)

    opt_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        opt_params,
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )
    accumulation = int(train_config["gradient_accumulation"])
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = max(1, steps_per_epoch * int(train_config["epochs"]))
    requested_warmup = int(train_config["warmup_steps"])
    if args.overfit_samples:
        requested_warmup = min(requested_warmup, max(1, total_steps // 20))
    warmup_steps = min(requested_warmup, max(0, total_steps - 1))
    min_ratio = train_config["min_learning_rate"] / train_config["learning_rate"]

    def lr_factor(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    amp_enabled, amp_dtype = autocast_settings(device, train_config["amp"])
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled and amp_dtype == torch.float16
    )

    start_epoch, optimizer_step, best_val = 0, 0, float("inf")
    resume_path = args.resume
    if resume_path:
        checkpoint = load_checkpoint(
            resume_path, model, optimizer, scheduler, rank
        )
        saved_mode = checkpoint.get("config", {}).get("train", {}).get(
            "loss_mode", "global_plus_halo"
        )
        current_mode = train_config.get("loss_mode", "global_plus_halo")
        if saved_mode != current_mode:
            raise ValueError(
                f"checkpoint loss_mode={saved_mode!r}, but current config uses "
                f"{current_mode!r}; start a new run instead of resuming"
            )
        start_epoch = int(checkpoint["epoch"]) + 1
        optimizer_step = int(checkpoint["step"])
        best_val = float(checkpoint["best_val"])

    output_dir = Path(train_config["output_dir"])
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "resolved_config.json", "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        effective_batch = (
            train_config["batch_size_per_gpu"] * world_size * accumulation
        )
        print(f"parameters={total_parameters:,}, train={len(train_dataset)}, val={len(val_dataset)}")
        print(
            f"distributed={'FSDP FULL_SHARD' if fsdp_enabled else 'none'}, "
            f"world_size={world_size}, effective_batch={effective_batch}, device={device}"
        )

    lat = torch.as_tensor(
        train_dataset.dataset.lat if isinstance(train_dataset, Subset) else train_dataset.lat,
        device=device,
    )
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, int(train_config["epochs"])):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running = torch.zeros(2, device=device, dtype=torch.float64)
        epoch_start = time.time()
        for batch_index, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            synchronize = (
                (batch_index + 1) % accumulation == 0
                or batch_index + 1 == len(train_loader)
            )
            sync_context = (
                nullcontext() if synchronize or not isinstance(model, FSDP) else model.no_sync()
            )
            with sync_context:
                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
                ):
                    prediction = forward_model(model, batch)
                    loss, _, _ = weighted_losses(
                        prediction,
                        batch["target"],
                        batch["target_mask"],
                        batch["obs_static"],
                        lat,
                        train_config["halo_weight"],
                        train_config["halo_radius"],
                        train_config.get("loss_mode", "global_plus_halo"),
                    )
                    scaled_loss = loss / accumulation
                scaler.scale(scaled_loss).backward()

            if synchronize:
                scaler.unscale_(optimizer)
                if isinstance(model, FSDP):
                    model.clip_grad_norm_(train_config["gradient_clip"])
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), train_config["gradient_clip"]
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_step += 1

                print(
                    f"epoch={epoch + 1} step={optimizer_step} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e} "
                    f"batch_loss={loss.item():.6f}",
                    flush=True,
                )
            running[0] += loss.detach().double()
            running[1] += 1

        if world_size > 1:
            dist.all_reduce(running, op=dist.ReduceOp.SUM)
        train_loss = (running[0] / running[1].clamp_min(1.0)).item()
        val_loss, val_global, val_halo = validate(
            model, val_loader, device, lat, train_config, amp_enabled, amp_dtype, world_size
        )
        improved = val_loss < best_val
        if improved:
            best_val = val_loss
        if rank == 0:
            elapsed = time.time() - epoch_start
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"epoch={epoch + 1}/{train_config['epochs']} step={optimizer_step} "
                f"lr={lr:.3e} train={train_loss:.6f} val={val_loss:.6f} "
                f"val_global={val_global:.6f} val_halo={val_halo:.6f} "
                f"seconds={elapsed:.1f}",
                flush=True,
            )
        checkpoint_paths = [output_dir / "last.pt"]
        if improved:
            checkpoint_paths.append(output_dir / "best.pt")
        save_checkpoints(
            checkpoint_paths, model, optimizer, scheduler,
            epoch, optimizer_step, best_val, config, rank,
        )
        if world_size > 1:
            dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
