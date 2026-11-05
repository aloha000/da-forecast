"""Paired GNSS components in physical millimetres, not differences of z-scores."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def normalization(group, name):
    if name not in group or group[name].attrs.get("raw_units") != "mm" or str(group[name].attrs.get("units")) != "1":
        raise ValueError(f"{name} must be standardized data with raw_units=mm, units=1")
    values = np.array([group.attrs.get(name+"_train_mean", np.nan),
                       group.attrs.get(name+"_train_std", np.nan)], dtype=np.float64)
    if not np.isfinite(values).all() or values[1] <= 0:
        raise ValueError(f"missing/invalid {name} normalization")
    return values

def paired_components(ztd, zwd, ztd_stats, zwd_stats):
    """Subtract physical values BEFORE cell averaging, using identical valid pairs."""
    valid = np.isfinite(ztd) & np.isfinite(zwd)
    total = ztd.astype(np.float64)*ztd_stats[1] + ztd_stats[0]
    wet = zwd.astype(np.float64)*zwd_stats[1] + zwd_stats[0]
    return np.where(valid, total-wet, np.nan).astype(np.float32), np.where(valid, wet, np.nan).astype(np.float32)

def setup_components(ds, group, scales, use_physics, height_path, height_datum, mean_std_dir):
    ds.component_loss_scales = np.asarray(scales, np.float32)
    if ds.component_loss_scales.shape != (2,) or not np.isfinite(ds.component_loss_scales).all() or np.any(ds.component_loss_scales <= 0):
        raise ValueError("component_loss_scales must be two positive finite mm scales")
    ds.ztd_stats = normalization(group, "ztd")
    ds.zwd_stats = normalization(group, "zwd")
    ds.component_context = {"ztd_stats": ds.ztd_stats.astype(np.float32)}
    if use_physics:
        if not height_path or height_datum != "orthometric":
            raise ValueError("physics needs station_height_path on the ERA grid, in metres, station_height_datum=orthometric")
        height = np.asarray(np.load(height_path), np.float32)
        if height.shape != ds.obs_static.shape[-2:]:
            raise ValueError("station heights must align EXACTLY with ERA lat/lon order")
        occupied = ds.obs_static[0] > 0
        if not np.isfinite(height[occupied]).all() or np.any((height[occupied] < -500) | (height[occupied] > 9000)):
            raise ValueError("missing or implausible station heights; do not substitute 0")
        latitude = np.broadcast_to(ds.lat[:, None], height.shape).copy()
        mean = np.load(Path(mean_std_dir)/"mean_era5.npy").reshape(-1).astype(np.float32)
        std = np.load(Path(mean_std_dir)/"std_era5.npy").reshape(-1).astype(np.float32)
        if mean.shape != (70,) or std.shape != (70,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("physics requires finite 70-channel ERA normalization and positive std")
        ds.component_context.update(station_height=np.where(occupied, height, 0).astype(np.float32),
                                    station_latitude=latitude, weather_mean=mean, weather_std=std)

def read_components(ds, timestamp, future=False):
    from .da_dataset import project_station_values
    ds._open()
    offsets = pd.to_timedelta(np.arange(1, 19)*20, unit="min") if future else ds.obs_offsets
    indices = ds.obs_times.get_indexer(pd.DatetimeIndex(pd.Timestamp(timestamp)+offsets))
    frames = len(indices)
    values = np.zeros((frames, 2, len(ds.lat), len(ds.lon)), np.float32)
    masks = np.zeros_like(values)
    present = np.flatnonzero(indices >= 0)
    if len(present):
        idx = indices[present]
        def read(name):
            if len(idx) > 1 and np.all(np.diff(idx) == np.diff(idx)[0]):
                return np.asarray(ds._ztd_group[name][int(idx[0]):int(idx[-1])+1:int(idx[1]-idx[0])], np.float32)
            return np.stack([np.asarray(ds._ztd_group[name][int(i)], np.float32) for i in idx])
        h, w = paired_components(read("ztd"), read("zwd"), ds.ztd_stats, ds.zwd_stats)
        for channel, raw in enumerate((h, w)):
            projected, valid = project_station_values(raw, ds.obs_source_lat, ds.obs_source_lon,
                                                       ds.obs_target_flat, (len(ds.lat), len(ds.lon)))
            values[present, channel], masks[present, channel] = projected, valid
    return values, masks

def component_inputs(ds, timestamp):
    values, mask = read_components(ds, timestamp)
    return {"obs_components": torch.from_numpy(values), "obs_component_mask": torch.from_numpy(mask),
            **{k: torch.from_numpy(v.copy()) for k,v in ds.component_context.items()}}
