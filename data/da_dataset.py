"""Dataset for 6-hour global forecasts from ERA5 and GNSS-ZTD."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import zarr


def _decode_time(group, name: str) -> pd.DatetimeIndex:
    array = group[name]
    units = str(dict(array.attrs)["units"])
    unit, origin = [part.strip() for part in units.split("since", maxsplit=1)]
    unit_code = {
        "hour": "h",
        "hours": "h",
        "minute": "min",
        "minutes": "min",
        "second": "s",
        "seconds": "s",
    }[unit.lower()]
    return pd.DatetimeIndex(
        pd.Timestamp(origin) + pd.to_timedelta(np.asarray(array[:]), unit=unit_code)
    )


def _nearest_indices(source: np.ndarray, target: np.ndarray, cyclic=False) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    result = []
    for value in target:
        if cyclic:
            error = np.abs((source - value + 180.0) % 360.0 - 180.0)
        else:
            error = np.abs(source - value)
        result.append(int(np.argmin(error)))
    return np.asarray(result, dtype=np.int64)


def build_static_fields(
    conds_path: str,
    global_lat_path: str,
    global_lon_path: str,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    station_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align FuXi's five static fields and build three station-side fields."""
    conds = np.load(conds_path).astype(np.float32)
    source_lat = np.load(global_lat_path)
    source_lon = np.load(global_lon_path)
    source_lat = source_lat[:, 0] if source_lat.ndim == 2 else source_lat
    source_lon = source_lon[0] if source_lon.ndim == 2 else source_lon

    if conds.shape[-2:] != (source_lat.size, source_lon.size):
        raise ValueError("FuXi static fields and static coordinate shapes do not match")
    if (np.allclose(source_lat, target_lat, atol=1e-4) and
            np.allclose(source_lon, target_lon, atol=1e-4)):
        era_static = conds
    else:
        lat_index = _nearest_indices(source_lat, target_lat)
        lon_index = _nearest_indices(source_lon, target_lon, cyclic=True)
        lat_error = np.max(np.abs(source_lat[lat_index] - target_lat))
        lon_error = np.max(
            np.abs((source_lon[lon_index] - target_lon + 180.0) % 360.0 - 180.0)
        )
        if lat_error > 0.13 or lon_error > 0.13:
            raise ValueError(
                f"static grid does not match target: lat error={lat_error}, "
                f"lon error={lon_error}"
            )
        era_static = conds[:, lat_index, :][:, :, lon_index]

    lat_scaled = 2.0 * (target_lat - target_lat.min()) / np.ptp(target_lat) - 1.0
    lon_scaled = 2.0 * (target_lon - target_lon.min()) / np.ptp(target_lon) - 1.0
    lat_grid = np.broadcast_to(lat_scaled[:, None], station_mask.shape)
    lon_grid = np.broadcast_to(lon_scaled[None, :], station_mask.shape)
    obs_static = np.stack((station_mask, lat_grid, lon_grid)).astype(np.float32)
    return np.ascontiguousarray(era_static), np.ascontiguousarray(obs_static)


def station_grid_mapping(ztd_group, era_lat, era_lon):
    """Map each occupied NGL cell onto its nearest ERA5 cell by station location."""
    occupied = ~np.asarray(ztd_group["mask"][:], dtype=bool)
    source_lat, source_lon = np.nonzero(occupied)
    station_lat = np.asarray(ztd_group["station_lat"][:])[occupied]
    station_lon = np.asarray(ztd_group["station_lon"][:])[occupied]
    if not (np.isfinite(station_lat).all() and np.isfinite(station_lon).all()):
        raise ValueError("occupied NGL cells require finite station coordinates")
    target_lat = _nearest_indices(era_lat, station_lat)
    target_lon = _nearest_indices(era_lon, station_lon, cyclic=True)
    target_flat = target_lat * len(era_lon) + target_lon
    target_mask = np.zeros((len(era_lat), len(era_lon)), dtype=np.float32)
    target_mask.flat[np.unique(target_flat)] = 1.0
    return source_lat, source_lon, target_flat, target_mask


def project_station_values(values, source_lat, source_lon, target_flat, shape):
    """Place NGL station series on ERA5; average stations sharing an ERA cell."""
    selected = values[:, source_lat, source_lon]
    finite = np.isfinite(selected)
    n_time = values.shape[0]
    n_cells = shape[0] * shape[1]
    output = np.zeros((n_time, n_cells), dtype=np.float32)
    count = np.zeros((n_time, n_cells), dtype=np.uint16)
    for frame in range(n_time):
        valid = finite[frame]
        np.add.at(output[frame], target_flat[valid], selected[frame, valid])
        np.add.at(count[frame], target_flat[valid], 1)
    np.divide(output, count, out=output, where=count > 0)
    return output.reshape((n_time, *shape)), (count > 0).reshape((n_time, *shape))


def time_features(analysis_time: pd.Timestamp) -> np.ndarray:
    """Match FuXi's original 12-value time encoding exactly."""
    values = np.asarray([
        (timestamp.dayofyear / 366.0, timestamp.hour / 24.0)
        for timestamp in (
        analysis_time - pd.Timedelta(hours=6),
        analysis_time,
        analysis_time + pd.Timedelta(hours=6),
        )
    ], dtype=np.float32)
    return np.concatenate((np.sin(values), np.cos(values)), axis=-1).reshape(-1)


def observation_offsets(
    history_hours: int = 6, interval_minutes: int = 20
) -> pd.TimedeltaIndex:
    """Return inclusive observation offsets from ``T-history`` through ``T``."""
    history_minutes = int(history_hours) * 60
    interval_minutes = int(interval_minutes)
    if history_minutes <= 0:
        raise ValueError("obs_history_hours must be positive")
    if interval_minutes <= 0:
        raise ValueError("obs_interval_minutes must be positive")
    if history_minutes % interval_minutes != 0:
        raise ValueError(
            "obs_history_hours * 60 must be divisible by obs_interval_minutes "
            "so both T-history and T are included"
        )
    minutes = np.arange(-history_minutes, 1, interval_minutes, dtype=np.int64)
    return pd.to_timedelta(minutes, unit="min")


class DAForecastDataset(Dataset):
    """Return ERA5(t-6,t), ZTD(t-6h:t every 20 min), and ERA5(t+6h)."""

    def __init__(
        self,
        era5_path: str,
        ztd_path: str,
        split: Sequence[str],
        conds_path: str,
        global_lat_path: str,
        global_lon_path: str,
        obs_history_hours: int = 6,
        obs_interval_minutes: int = 20,
    ) -> None:
        super().__init__()
        self.era5_path = str(Path(era5_path))
        self.ztd_path = str(Path(ztd_path))
        self.obs_offsets = observation_offsets(
            obs_history_hours, obs_interval_minutes
        )
        self.obs_history_hours = int(obs_history_hours)
        self.obs_interval_minutes = int(obs_interval_minutes)
        self.obs_frames = len(self.obs_offsets)

        era_group = zarr.open(self.era5_path, mode="r")
        ztd_group = zarr.open(self.ztd_path, mode="r")
        if "z" not in era_group:
            raise ValueError("ERA5 store must contain the global 70-channel 'z' array")
        era_array = era_group["z"]
        if era_array.ndim != 4 or era_array.shape[1] != 70:
            raise ValueError(f"expected ERA5 z[time,70,lat,lon], got {era_array.shape}")
        self.label_times = _decode_time(era_group, "time")
        self.obs_times = _decode_time(ztd_group, "time")
        self.label_lookup = {timestamp.value: i for i, timestamp in enumerate(self.label_times)}

        target_lat = np.asarray(era_group["lat"][:], dtype=np.float64)
        target_lon = np.asarray(era_group["lon"][:], dtype=np.float64)
        ztd_lat = np.asarray(ztd_group["lat"][:], dtype=np.float64)
        ztd_lon = np.asarray(ztd_group["lon"][:], dtype=np.float64)
        if era_array.shape[-2:] != (len(target_lat), len(target_lon)):
            raise ValueError("ERA5 array and coordinate dimensions do not match")
        if ztd_group["ztd"].shape[-2:] != (len(ztd_lat), len(ztd_lon)):
            raise ValueError("NGL array and coordinate dimensions do not match")

        (self.obs_source_lat, self.obs_source_lon,
         self.obs_target_flat, station_mask) = station_grid_mapping(
             ztd_group, target_lat, target_lon
         )
        self.era_static, self.obs_static = build_static_fields(
            conds_path,
            global_lat_path,
            global_lon_path,
            target_lat,
            target_lon,
            station_mask.astype(np.float32),
        )
        self.lat = target_lat.astype(np.float32)
        self.lon = target_lon.astype(np.float32)

        start, end = pd.Timestamp(split[0]), pd.Timestamp(split[1])
        lead = pd.Timedelta(hours=6)
        candidates = []
        for timestamp in self.label_times:
            # Target remains inside the half-open split. Historical input may
            # precede the validation/test boundary, which is valid at inference.
            if timestamp < start or timestamp + lead >= end:
                continue
            if (timestamp - lead).value not in self.label_lookup:
                continue
            if (timestamp + lead).value not in self.label_lookup:
                continue
            if self._obs_indices(timestamp) is None:
                continue
            candidates.append(timestamp)
        self.analysis_times = pd.DatetimeIndex(candidates)
        if len(self.analysis_times) == 0:
            raise ValueError(f"no valid samples in split {tuple(split)}")

        self._era_group = None
        self._ztd_group = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_era_group"] = None
        state["_ztd_group"] = None
        return state
    def _open(self) -> None:
        if self._era_group is None:
            self._era_group = zarr.open(self.era5_path, mode="r")
        if self._ztd_group is None:
            self._ztd_group = zarr.open(self.ztd_path, mode="r")

    def __len__(self) -> int:
        return len(self.analysis_times)

    def sample_time(self, index: int) -> pd.Timestamp:
        return self.analysis_times[index]

    def _obs_indices(self, timestamp: pd.Timestamp):
        wanted = pd.DatetimeIndex(timestamp + self.obs_offsets)
        indices = np.asarray(self.obs_times.searchsorted(wanted), dtype=np.int64)
        if np.any(indices >= len(self.obs_times)):
            return None
        if not np.array_equal(self.obs_times[indices].asi8, wanted.asi8):
            return None
        return indices

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        self._open()
        timestamp = self.analysis_times[index]
        before = self.label_lookup[(timestamp - pd.Timedelta(hours=6)).value]
        current = self.label_lookup[timestamp.value]
        target_index = self.label_lookup[(timestamp + pd.Timedelta(hours=6)).value]

        era_array = self._era_group["z"]
        if (current, target_index) != (before + 1, before + 2):
            raise RuntimeError("ERA5 time axis is not contiguous at 6-hour intervals")
        triplet = np.asarray(era_array[before:target_index + 1], dtype=np.float32)
        era_before, era_current, target_raw = triplet
        era5 = np.stack((era_before, era_current))
        target_valid = np.isfinite(target_raw)
        era5 = np.nan_to_num(era5, nan=0.0, posinf=0.0, neginf=0.0)
        target = np.nan_to_num(target_raw, nan=0.0, posinf=0.0, neginf=0.0)

        obs_indices = self._obs_indices(timestamp)
        if obs_indices is None:
            raise RuntimeError(f"missing configured ZTD history for {timestamp}")
        step = int(obs_indices[1] - obs_indices[0])
        if not np.all(np.diff(obs_indices) == step):
            raise RuntimeError("ZTD sample indices are not regularly spaced")
        obs_slice = slice(int(obs_indices[0]), int(obs_indices[-1]) + 1, step)
        obs_raw = np.asarray(self._ztd_group["ztd"][obs_slice], dtype=np.float32)
        if obs_raw.shape[0] != self.obs_frames:
            raise RuntimeError(
                f"expected {self.obs_frames} ZTD frames, got {obs_raw.shape[0]}"
            )
        obs, obs_valid = project_station_values(
            obs_raw, self.obs_source_lat, self.obs_source_lon,
            self.obs_target_flat, (len(self.lat), len(self.lon)),
        )

        return {
            "era5": torch.from_numpy(era5),
            "era_static": torch.from_numpy(self.era_static),
            "time_features": torch.from_numpy(time_features(timestamp)),
            "obs_ztd": torch.from_numpy(obs[:, None]),
            "obs_mask": torch.from_numpy(obs_valid[:, None].astype(np.float32)),
            "obs_static": torch.from_numpy(self.obs_static),
            "target": torch.from_numpy(target),
            "target_mask": torch.from_numpy(target_valid.astype(np.float32)),
        }


__all__ = [
    "DAForecastDataset",
    "build_static_fields",
    "observation_offsets",
    "project_station_values",
    "station_grid_mapping",
    "time_features",
]
