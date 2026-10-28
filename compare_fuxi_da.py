"""Paired original-FuXi vs DA evaluation on identical samples and precision.

Saves small, atomic per-sample files so an interrupted run can be resumed.
No full forecast fields are retained. Existing experiment outputs are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import pandas as pd
import torch
import torch.distributed as dist
import zarr

from evaluate import CHANNELS, inverse_all, load_distributed_model
from model.fuxi import UTransformer
from model.fuxi_pretrained import _PICKLE_COMPAT
from train import load_config, resolve_data_paths, make_dataset, wrap_fsdp, setup_distributed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/train_da_forecast.json')
    p.add_argument('--da-checkpoint', default='work_dirs/da_forecast_global_6h_fsdp_train_obs/best.pt')
    p.add_argument('--fuxi-checkpoint', default='model/fuxi_short.pth')
    p.add_argument('--conds', default=None)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--seed', type=int, default=20261026)
    p.add_argument('--days', type=int, default=7)
    p.add_argument('--stations', type=int, default=5)
    p.add_argument('--max-samples', type=int, default=0, help='0=entire test set; otherwise smoke-test subset')
    p.add_argument('--plot-only', action='store_true')
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def file_identity(path):
    p = Path(path).resolve()
    st = p.stat()
    return dict(path=str(p), bytes=st.st_size, mtime_ns=st.st_mtime_ns)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def select_station_window(dataset, data, seed, days, station_count):
    rng = np.random.default_rng(seed)
    steps = days * 4
    starts = [i for i in range(len(dataset) - steps + 1)
              if (dataset.sample_time(i) + pd.Timedelta(hours=6)).hour == 0]
    if not starts:
        raise ValueError('test period is too short for the requested window')
    first = int(rng.choice(starts))
    window = list(range(first, first + steps))
    required = np.unique(np.concatenate([dataset._obs_indices(dataset.sample_time(i)) for i in window]))
    obs = zarr.open(data['ztd_path'], mode='r')
    names = np.asarray(obs['station'][:])
    station_lat = np.asarray(obs['station_lat'][:])
    station_lon = np.asarray(obs['station_lon'][:])
    selected = []
    used_cells = set()
    stride = int(required[1] - required[0])
    if not np.all(np.diff(required) == stride):
        raise ValueError('selected observation times are not regular')
    for candidate in rng.permutation(len(dataset.obs_target_flat)):
        cell = int(dataset.obs_target_flat[candidate])
        if cell in used_cells:
            continue
        i, j = int(dataset.obs_source_lat[candidate]), int(dataset.obs_source_lon[candidate])
        series = np.asarray(obs['ztd'][slice(int(required[0]), int(required[-1]) + 1, stride), i, j])
        coverage = float(np.isfinite(series).mean())
        if coverage < 0.8:
            continue
        row, col = divmod(cell, len(dataset.lon))
        selected.append(dict(id=str(names[i, j]), latitude=float(station_lat[i, j]),
                             longitude=float(station_lon[i, j]), grid_row=row, grid_col=col,
                             grid_latitude=float(dataset.lat[row]), grid_longitude=float(dataset.lon[col]),
                             ztd_coverage=coverage))
        used_cells.add(cell)
        if len(selected) == station_count:
            break
    if len(selected) != station_count:
        raise ValueError('not enough distinct station grid cells with >=80% ZTD coverage')
    return dict(indices=window,
                valid_start=str(dataset.sample_time(window[0]) + pd.Timedelta(hours=6)),
                valid_end=str(dataset.sample_time(window[-1]) + pd.Timedelta(hours=6)),
                stations=selected,
                method='uniform random contiguous window; randomly permuted occupied global stations, '
                       'accept >=80% finite ZTD at input sampling times and distinct target grid cells')


def load_original(path, device, rank):
    model = UTransformer(in_chans=75, out_chans=70, in_frames=2,
                         image_size=(720, 1440), window_size=9, patch_size=4,
                         down_times=1, embed_dim=1536, num_heads=24, depths=[12] * 4)
    error = [None]
    if rank == 0:
        try:
            checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False,
                                    pickle_module=_PICKLE_COMPAT)
            state = {k.removeprefix('decoder.'): v for k, v in checkpoint['model'].items()
                     if k.startswith('decoder.')}
            model.load_state_dict(state, strict=True)
        except Exception as exc:
            error[0] = f'{type(exc).__name__}: {exc}'
    dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])
    return wrap_fsdp(model, device, {'amp': 'bf16', 'fsdp': {'sync_module_states': True}}).eval()


def weighted_sums(prediction, target, weights, mean, std, target_physical):
    if not torch.isfinite(prediction).all():
        raise RuntimeError('model prediction contains NaN/Inf; refusing to hide invalid predictions')
    norm = ((prediction.float() - target).abs() * weights).sum((0, 2, 3), dtype=torch.float64)
    physical = ((inverse_all(prediction, mean, std) - target_physical).abs() * weights).sum(
        (0, 2, 3), dtype=torch.float64)
    return norm.cpu().numpy(), physical.cpu().numpy()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    if args.plot_only:
        from plot_fuxi_da_comparison import make_report
        make_report(out)
        return
    rank, _, world_size, device = setup_distributed()
    if world_size < 2:
        raise ValueError('launch with at least two ranks; four GPUs recommended')
    config = load_config(args.config)
    data = resolve_data_paths(args.config, config['data'])
    if args.conds:
        data['conds_path'] = str(Path(args.conds).resolve())
    for key in ('era5_path', 'ztd_path', 'conds_path', 'mean_std_dir'):
        data[key] = str(Path(data[key]).resolve())
    dataset = make_dataset(data, 'test')
    indices = list(range(len(dataset)))
    if args.max_samples:
        if args.max_samples < 1:
            raise ValueError('max-samples must be nonnegative')
        indices = np.unique(np.linspace(0, len(dataset)-1, min(args.max_samples, len(dataset)), dtype=int)).tolist()

    holder = [None]
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'samples').mkdir(exist_ok=True)
        identity = dict(version=1, data=data, da_checkpoint=file_identity(args.da_checkpoint),
                        fuxi_checkpoint=file_identity(args.fuxi_checkpoint),
                        conds_sha256=sha256(data['conds_path']),
                        mean_sha256=sha256(Path(data['mean_std_dir'])/'mean_era5.npy'),
                        std_sha256=sha256(Path(data['mean_std_dir'])/'std_era5.npy'),
                        amp='bf16', seed=args.seed, days=args.days, station_count=args.stations,
                        evaluation_indices=indices, full_test_samples=len(dataset))
        metadata_path = out / 'metadata.json'
        if metadata_path.exists():
            saved = json.loads(metadata_path.read_text())
            if saved['identity'] != identity:
                raise ValueError('output directory belongs to different inputs/configuration; use a new directory')
            holder[0] = saved
        else:
            selection = select_station_window(dataset, data, args.seed, args.days, args.stations)
            holder[0] = dict(identity=identity, selection=selection, channels=CHANNELS,
                             note='Paired global +6h forecasts, common BF16 and static fields. '
                                  'DA checkpoint was trained before the static latitude correction. '
                                  'Station curves are nearest ERA5-grid samples, not observed station meteorology; '
                                  'each point is an independent +6h forecast, not an autoregressive rollout.')
            write_json(metadata_path, holder[0])
        print('SELECTION ' + json.dumps(holder[0]['selection']), flush=True)
    dist.broadcast_object_list(holder, src=0)
    metadata = holder[0]
    checkpoint = torch.load(args.da_checkpoint, map_location='cpu', mmap=True, weights_only=False) if rank == 0 else None
    saved_cfg = [checkpoint['config'] if rank == 0 else None]
    dist.broadcast_object_list(saved_cfg, src=0)
    model_cfg = saved_cfg[0]['model']
    if model_cfg['image_size'] != [720, 1440] or model_cfg['era5_channels'] != 70:
        raise ValueError('this comparison expects the global 70-channel model')
    train_cfg = {**saved_cfg[0]['train'], 'amp': 'bf16'}
    da = load_distributed_model(checkpoint, {'train': train_cfg}, model_cfg, device, rank)
    if rank == 0:
        metadata['da_epoch_zero_based'] = checkpoint['epoch']
        metadata['da_step'] = checkpoint['step']
        metadata['da_best_val'] = checkpoint['best_val']
        write_json(out/'metadata.json', metadata)
        print('DA checkpoint strictly loaded and synchronized', flush=True)
    del checkpoint
    original = load_original(args.fuxi_checkpoint, device, rank)
    if rank == 0:
        print('Original FuXi strictly loaded and synchronized; starting paired evaluation', flush=True)

    mean = torch.as_tensor(np.load(Path(data['mean_std_dir'])/'mean_era5.npy'),device=device).float().reshape(1,70,1,1)
    std = torch.as_tensor(np.load(Path(data['mean_std_dir'])/'std_era5.npy'),device=device).float().reshape(1,70,1,1)
    lat = torch.as_tensor(dataset.lat,device=device)
    lat_weights = torch.cos(torch.deg2rad(lat)).clamp_min(0)
    lat_weights = (lat_weights/lat_weights.mean()).reshape(1,1,-1,1)
    rows = torch.tensor([s['grid_row'] for s in metadata['selection']['stations']],device=device)
    cols = torch.tensor([s['grid_col'] for s in metadata['selection']['stations']],device=device)
    window = set(metadata['selection']['indices'])
    blocks = (len(indices)+world_size-1)//world_size
    started = time.monotonic()
    timings = np.zeros(3)
    for block in range(blocks):
        position = block*world_size+rank
        real = position < len(indices)
        index = indices[position] if real else indices[0]
        path = out/'samples'/f'{index:06d}.npz'
        cached = torch.tensor(int(not real or path.exists()),device=device)
        dist.all_reduce(cached, op=dist.ReduceOp.MIN)
        if not bool(cached.item()):
            t = time.monotonic()
            cpu_batch = dataset[index]
            batch = {k:v.unsqueeze(0).to(device) for k,v in cpu_batch.items()}
            del cpu_batch
            timings[0] += time.monotonic()-t
            t = time.monotonic()
            with torch.inference_mode():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    residual = original(batch['era5'],temb=batch['time_features'],const=batch['era_static'][0])
                    # Match DA's root-FSDP BF16 residual addition exactly.
                    pred_original = residual + batch['era5'][:,-1].to(residual.dtype)
                    del residual
                    pred_da = da(batch['era5'],batch['era_static'],batch['time_features'],
                                 batch['obs_ztd'],batch['obs_mask'],batch['obs_static'])
                torch.cuda.synchronize(device)
                timings[1] += time.monotonic()-t
                t = time.monotonic()
                target = batch['target']
                weights = batch['target_mask']*lat_weights
                target_physical = inverse_all(target,mean,std)
                norm_orig,phys_orig = weighted_sums(pred_original,target,weights,mean,std,target_physical)
                norm_da,phys_da = weighted_sums(pred_da,target,weights,mean,std,target_physical)
                norm_base,phys_base = weighted_sums(batch['era5'][:,-1],target,weights,mean,std,target_physical)
                result = dict(index=np.int64(index),valid_time=np.array(str(dataset.sample_time(index)+pd.Timedelta(hours=6))),
                              denominator=weights.sum((0,2,3),dtype=torch.float64).cpu().numpy(),
                              original_normalized=norm_orig,da_normalized=norm_da,persistence_normalized=norm_base,
                              original_physical=phys_orig,da_physical=phys_da,persistence_physical=phys_base)
                if index in window:
                    result['station_values'] = torch.stack((batch['target'][0,:,rows,cols],
                        pred_original[0,:,rows,cols].float(),pred_da[0,:,rows,cols].float())).cpu().numpy()
                    result['station_target_valid'] = batch['target_mask'][0,:,rows,cols].bool().cpu().numpy()
                if real:
                    temporary = path.with_suffix('.npz.tmp')
                    with temporary.open('wb') as f:
                        np.savez_compressed(f,**result)
                    os.replace(temporary,path)
                timings[2] += time.monotonic()-t
                del batch,pred_original,pred_da,target,weights,target_physical,result
        if rank == 0:
            done=min((block+1)*world_size,len(indices))
            elapsed=time.monotonic()-started
            message=dict(completed=done,total=len(indices),elapsed_seconds=round(elapsed,1),
                         rank0_data_seconds=round(timings[0],1),rank0_forward_seconds=round(timings[1],1),
                         rank0_metrics_seconds=round(timings[2],1))
            write_json(out/'progress.json',message)
            print('PROGRESS '+json.dumps(message),flush=True)
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        from plot_fuxi_da_comparison import make_report
        make_report(out)


if __name__ == '__main__':
    main()
