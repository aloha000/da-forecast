"""Dataset for 6-hour regional forecasts from ERA5 and GNSS-ZTD."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import zarr


# label[69] is IMERG precipitation.  This task uses ERA5 as both input and
# supervision, so channel 70 (era5_tp) is selected instead.
ERA5_CHANNEL_INDEX = np.asarray(list(range(69)) + [70], dtype=np.int64)


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
    """Crop FuXi's five static fields and build three station-side fields."""
    conds = np.load(conds_path).astype(np.float32)
    source_lat = np.load(global_lat_path)
    source_lon = np.load(global_lon_path)
    source_lat = source_lat[:, 0] if source_lat.ndim == 2 else source_lat
    source_lon = source_lon[0] if source_lon.ndim == 2 else source_lon

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


def time_features(analysis_time: pd.Timestamp) -> np.ndarray:
    """Periodic encoding of t-6h, t and t+6h, producing 12 values."""
    values = []
    for timestamp in (
        analysis_time - pd.Timedelta(hours=6),
        analysis_time,
        analysis_time + pd.Timedelta(hours=6),
    ):
        days = 366.0 if timestamp.is_leap_year else 365.0
        values.extend(
            [2.0 * np.pi * (timestamp.dayofyear - 1) / days,
             2.0 * np.pi * timestamp.hour / 24.0]
        )
    values = np.asarray(values, dtype=np.float32)
    return np.concatenate((np.sin(values), np.cos(values))).astype(np.float32)


class DAForecastDataset(Dataset):
    """Return ERA5(t-6,t), ZTD(t-45min..t), and ERA5(t+6h)."""

    def __init__(
        self,
        era5_path: str,
        ztd_path: str,
        split: Sequence[str],
        conds_path: str,
        global_lat_path: str,
        global_lon_path: str,
        obs_frames: int = 10,
    ) -> None:
        super().__init__()
        self.era5_path = str(Path(era5_path))
        self.ztd_path = str(Path(ztd_path))
        self.obs_frames = int(obs_frames)
        if self.obs_frames <= 0:
            raise ValueError("obs_frames must be positive")

        label_group = zarr.open(self.era5_path, mode="r")
        ztd_group = zarr.open(self.ztd_path, mode="r")
        self.label_times = _decode_time(label_group, "time")
        self.obs_times = _decode_time(ztd_group, "time")
        self.label_lookup = {timestamp.value: i for i, timestamp in enumerate(self.label_times)}

        target_lat = np.asarray(label_group["lat"][:], dtype=np.float64)
        target_lon = np.asarray(label_group["lon"][:], dtype=np.float64)
        ztd_lat = np.asarray(ztd_group["lat"][:], dtype=np.float64)
        ztd_lon = np.asarray(ztd_group["lon"][:], dtype=np.float64)
        if not np.allclose(target_lat, ztd_lat) or not np.allclose(target_lon, ztd_lon):
            raise ValueError("ERA5 and ZTD grids do not match")

        # In this store mask=True means no station, hence the inversion.
        station_mask = ~np.asarray(ztd_group["mask"][:], dtype=bool)
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
            obs_end = int(self.obs_times.searchsorted(timestamp))
            if obs_end >= len(self.obs_times) or self.obs_times[obs_end] != timestamp:
                continue
            obs_start = obs_end - self.obs_frames + 1
            if obs_start < 0:
                continue
            expected_start = timestamp - pd.Timedelta(minutes=5 * (self.obs_frames - 1))
            if self.obs_times[obs_start] != expected_start:
                continue
            candidates.append(timestamp)
        self.analysis_times = pd.DatetimeIndex(candidates)
        if len(self.analysis_times) == 0:
            raise ValueError(f"no valid samples in split {tuple(split)}")

        self._label_group = None
        self._ztd_group = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_label_group"] = None
        state["_ztd_group"] = None
        return state

    def _open(self) -> None:
        if self._label_group is None:
            self._label_group = zarr.open(self.era5_path, mode="r")
        if self._ztd_group is None:
            self._ztd_group = zarr.open(self.ztd_path, mode="r")

    def __len__(self) -> int:
        return len(self.analysis_times)

    def sample_time(self, index: int) -> pd.Timestamp:
        return self.analysis_times[index]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        self._open()
        timestamp = self.analysis_times[index]
        before = self.label_lookup[(timestamp - pd.Timedelta(hours=6)).value]
        current = self.label_lookup[timestamp.value]
        target_index = self.label_lookup[(timestamp + pd.Timedelta(hours=6)).value]

        label_array = self._label_group["label"]
        if (current, target_index) != (before + 1, before + 2):
            raise RuntimeError("ERA5 time axis is not contiguous at 6-hour intervals")
        triplet = np.asarray(label_array[before:target_index + 1], dtype=np.float32)
        triplet = triplet[:, ERA5_CHANNEL_INDEX]
        era_before, era_current, target_raw = triplet
        era5 = np.stack((era_before, era_current))
        target_valid = np.isfinite(target_raw)
        era5 = np.nan_to_num(era5, nan=0.0, posinf=0.0, neginf=0.0)
        target = np.nan_to_num(target_raw, nan=0.0, posinf=0.0, neginf=0.0)

        obs_end = int(self.obs_times.searchsorted(timestamp))
        obs_start = obs_end - self.obs_frames + 1
        obs_raw = np.asarray(
            self._ztd_group["ztd"][obs_start:obs_end + 1], dtype=np.float32
        )
        obs_valid = np.isfinite(obs_raw)
        obs = np.nan_to_num(obs_raw, nan=0.0, posinf=0.0, neginf=0.0)

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
    "ERA5_CHANNEL_INDEX",
    "build_static_fields",
    "time_features",
]
