"""Two 4-D fields: ERA5 truth by valid time, +6h forecast by original time T."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import zarr
from numcodecs import Blosc

from .da_dataset import time_features

SCHEMA = "da_forecast_12h_two_fields_v1"
SPLITS = ("train", "val", "test")
HOUR_NS = 3600 * 10**9


def source_observation_policy(data):
    policy = data.get("rollout_source_observations", "real_history_at_T")
    if policy not in ("real_history_at_T", "zero_ztd_history_at_T"):
        raise ValueError(f"unknown rollout source observation policy: {policy}")
    return policy


def times_ns(index):
    # pandas may hold datetime64[us]/[s]; .asi8 is NOT always nanoseconds.
    return np.asarray(index, dtype="datetime64[ns]").astype(np.int64)


def open_store(path, mode="r"):
    key = "zarr_format" if int(zarr.__version__.split(".")[0]) >= 3 else "zarr_version"
    return zarr.open_group(str(path), mode=mode, **{key: 2})


def create_array(group, name, shape, dtype, chunks, dims, fill_value=0):
    create = group.create_array if hasattr(group, "create_array") else group.create_dataset
    arr = create(name, shape=shape, dtype=dtype, chunks=chunks,
                 compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
                 fill_value=fill_value)
    arr.attrs["_ARRAY_DIMENSIONS"] = list(dims)
    return arr


def axes_for_datasets(datasets):
    origins = np.concatenate([times_ns(datasets[s].analysis_times) for s in SPLITS])
    codes = np.concatenate([np.full(len(datasets[s]), i, dtype=np.int8)
                            for i, s in enumerate(SPLITS)])
    if not len(origins) or np.any(np.diff(origins) <= 0):
        raise ValueError("split origins must be unique, ordered, and non-overlapping")
    times = np.unique(np.concatenate((origins, origins + 12 * HOUR_NS)))
    return times, origins, codes


def initialize_store(path, identity, datasets, source_model_config):
    path = Path(path)
    if path.exists():
        root = open_store(path, "r+")
        if (root.attrs.get("schema") != SCHEMA or not root.attrs.get("initialized")
                or root.attrs.get("identity") != identity):
            raise ValueError("Zarr identity differs or initialization incomplete; use a new output path")
        return root
    times, origins, codes = axes_for_datasets(datasets)
    first = datasets["train"]
    h, w = len(first.lat), len(first.lon)
    channels = int(source_model_config["era5_channels"])
    for ds in datasets.values():
        if not np.array_equal(ds.lat, first.lat) or not np.array_equal(ds.lon, first.lon):
            raise ValueError("split coordinate mismatch")
    root = open_store(path, "w-")
    root.attrs.update({"schema": SCHEMA, "identity": identity, "initialized": False,
                       "source_model_config": source_model_config,
                       "forecast_lead_hours": 12, "source_lead_hours": 6,
                       "source_observations": identity.get("source_observations", "real_history_at_T"),
                       "rollout_observations": identity.get("rollout_observations", "branch_removed")})
    for name, value, dims in (
        ("time", times, ("time",)), ("init_time", origins, ("init_time",)),
        ("split_code", codes, ("init_time",)),
        ("lat", first.lat, ("lat",)), ("lon", first.lon, ("lon",)),
        ("channel", np.arange(channels, dtype=np.int32), ("channel",)),
        ("era_static", first.era_static, ("static_channel", "lat", "lon")),
        # Used only for metrics, never for the no-OBS network.
        ("metric_station_mask", first.obs_static[0], ("lat", "lon")),
    ):
        value = np.asarray(value)
        arr = create_array(root, name, value.shape, value.dtype, value.shape, dims)
        arr[:] = value
    for name in ("time", "init_time"):
        root[name].attrs["units"] = "nanoseconds since 1970-01-01 00:00:00"
        root[name].attrs["calendar"] = "proleptic_gregorian"
    for name, length, dim in (("z", len(times), "time"),
                              ("z_fuxi6h", len(origins), "init_time")):
        create_array(root, name, (length, channels, h, w), "f4",
                     (1, min(10, channels), h, w),
                     (dim, "channel", "lat", "lon"), np.nan)
        root[name].attrs["normalization"] = "same_as_source_era5"
    root["z_fuxi6h"].attrs["lead_hours"] = 6
    root["z_fuxi6h"].attrs["valid_time"] = "init_time + 6 hours"
    root["split_code"].attrs["codes"] = dict(enumerate(SPLITS))
    # One row per chunk: ranks never share a mutable chunk, including flags.
    create_array(root, "era_complete", (len(times),), "bool", (1,), ("time",))
    create_array(root, "forecast_complete", (len(origins),), "bool", (1,), ("init_time",))
    if identity.get("rollout_observations") in ("predicted_ztd_at_T", "predicted_components_at_T"):
        from .ztd_fields import initialize_ztd_fields
        if not source_model_config.get("predict_ztd", False):
            raise ValueError("source needs a trained ZTD head")
        initialize_ztd_fields(root, first, len(origins))
    root.attrs["initialized"] = True
    return root


def write_field(root, name, index, value, ztd_payload=None):
    if name not in ("z", "z_fuxi6h"):
        raise ValueError("unknown field")
    if not 0 <= index < root[name].shape[0]:
        raise IndexError(index)
    value = np.asarray(value, dtype=np.float32)
    if value.shape != root[name].shape[1:]:
        raise ValueError("field shape mismatch")
    if name == "z_fuxi6h" and not np.isfinite(value).all():
        raise ValueError("nonfinite source prediction")
    flag = root["era_complete" if name == "z" else "forecast_complete"]
    flag[index] = False
    if name == "z_fuxi6h" and root.attrs.get("rollout_observations") in ("predicted_ztd_at_T", "predicted_components_at_T"):
        from .ztd_fields import write_ztd_payload
        if ztd_payload is None:
            raise ValueError("joint forecast requires ZTD input and labels before completion")
        write_ztd_payload(root, index, ztd_payload)
    root[name][index] = value
    flag[index] = True


class RolloutFieldsDataset(Dataset):
    """Self-contained training data: never opens source ERA5 or GNSS Zarr."""
    def __init__(self, path, split_name, split_bounds, source_identity):
        self.path = str(Path(path).resolve())
        self.split_name = split_name
        self._root = None
        root = open_store(path)
        if root.attrs.get("schema") != SCHEMA or not root.attrs.get("initialized"):
            raise ValueError("not a fully initialized two-field 12h Zarr")
        if (root.attrs.get("source_observations") not in ("real_history_at_T", "zero_ztd_history_at_T")
                or root.attrs.get("rollout_observations") not in ("branch_removed", "predicted_ztd_at_T", "predicted_components_at_T")):
            raise ValueError("Zarr observation policy mismatch")
        self.rollout_identity = dict(root.attrs["identity"])
        self.rollout_observations = root.attrs["rollout_observations"]
        if self.rollout_observations != self.rollout_identity.get("rollout_observations", "branch_removed"):
            raise ValueError("Zarr rollout observation policy differs from identity")
        self.predict_delay_components = self.rollout_observations == "predicted_components_at_T"
        if self.predict_delay_components != bool(root.attrs.get("predict_delay_components", False)):
            raise ValueError("rollout component schema mismatch")
        self.predict_ztd = self.rollout_observations in ("predicted_ztd_at_T", "predicted_components_at_T")
        if self.predict_delay_components:
            self.component_loss_scales = np.asarray(root.attrs["component_loss_scales"], np.float32)
        self.ztd_normalization = dict(root.attrs.get("ztd_normalization", {}))
        self.source_observations = root.attrs["source_observations"]
        if self.source_observations != self.rollout_identity.get("source_observations", "real_history_at_T"):
            raise ValueError("Zarr observation policy differs from identity")
        if self.rollout_identity["source_checkpoint"] != source_identity:
            raise ValueError("Zarr source checkpoint differs from config")
        bounds = self.rollout_identity["data"][split_name + "_split"]
        if list(map(pd.Timestamp, bounds)) != list(map(pd.Timestamp, split_bounds)):
            raise ValueError("Zarr split bounds differ from config")
        times = np.asarray(root["time"][:], dtype=np.int64)
        origins = np.asarray(root["init_time"][:], dtype=np.int64)
        codes = np.asarray(root["split_code"][:])
        if (np.any(np.diff(times) <= 0) or np.any(np.diff(origins) <= 0)
                or codes.shape != origins.shape or not np.isin(codes, [0, 1, 2]).all()):
            raise ValueError("invalid time axes or split codes")
        self.forecast_indices = np.flatnonzero(codes == SPLITS.index(split_name))
        selected = origins[self.forecast_indices]
        self.analysis_times = pd.DatetimeIndex(pd.to_datetime(selected, unit="ns"))
        if (not len(selected) or (self.analysis_times < pd.Timestamp(split_bounds[0])).any()
                or (self.analysis_times + pd.Timedelta(hours=12) >= pd.Timestamp(split_bounds[1])).any()):
            raise ValueError("empty split or target crosses split boundary")
        lookup = {int(t): i for i, t in enumerate(times)}
        try:
            self.current_indices = np.array([lookup[int(t)] for t in selected])
            self.target_indices = np.array([lookup[int(t + 12 * HOUR_NS)] for t in selected])
        except KeyError as exc:
            raise ValueError("missing ERA5 current/target timestamp") from exc
        self.lat = np.asarray(root["lat"][:], dtype=np.float32)
        self.lon = np.asarray(root["lon"][:], dtype=np.float32)
        self.era_static = np.asarray(root["era_static"][:], dtype=np.float32)
        station = np.asarray(root["metric_station_mask"][:], dtype=np.float32)
        self.obs_static = np.stack((station, np.zeros_like(station), np.zeros_like(station)))
        self.lead_hours, self.obs_frames = 12, 1
        self.obs_history_hours, self.obs_interval_minutes = 0, 20
        if self.predict_ztd:
            from .ztd_fields import validate_ztd_fields
            validate_ztd_fields(root, len(origins), len(self.lat), len(self.lon))
            self.obs_static = np.asarray(root["obs_static"][:], np.float32)
            self.obs_frames, self.obs_history_hours = 19, 6
        self.channels = int(root["z"].shape[1])
        shape = (self.channels, len(self.lat), len(self.lon))
        if (root["z"].shape != (len(times), *shape)
                or root["z_fuxi6h"].shape != (len(origins), *shape)
                or root["era_complete"].shape != times.shape
                or root["forecast_complete"].shape != origins.shape):
            raise ValueError("field/completion shapes do not match coordinates")
        ready_era = np.asarray(root["era_complete"][:], dtype=bool)
        ready_forecast = np.asarray(root["forecast_complete"][:], dtype=bool)
        needed = np.unique(np.concatenate((self.current_indices, self.target_indices)))
        if not ready_era[needed].all() or not ready_forecast[self.forecast_indices].all():
            raise ValueError(f"{split_name}: incomplete Zarr; finish generation first")

    def __len__(self):
        return len(self.analysis_times)

    def sample_time(self, index):
        return self.analysis_times[index]

    def __getstate__(self):
        return {**self.__dict__, "_root": None}

    def __getitem__(self, index):
        if self._root is None:
            self._root = open_store(self.path)
        current = np.asarray(self._root["z"][int(self.current_indices[index])], dtype=np.float32)
        prediction = np.asarray(self._root["z_fuxi6h"][int(self.forecast_indices[index])], dtype=np.float32)
        target = np.asarray(self._root["z"][int(self.target_indices[index])], dtype=np.float32)
        if not np.isfinite(prediction).all():
            raise ValueError(f"nonfinite forecast at origin {self.sample_time(index)}")
        current = np.where(np.isfinite(current), current, 0).astype(np.float32)
        valid = np.isfinite(target)
        zero = torch.zeros((1, 1, len(self.lat), len(self.lon)), dtype=torch.float32)
        result = {
            "era5": torch.from_numpy(np.stack((current, prediction))),
            "persistence": torch.from_numpy(current.copy()),
            "target": torch.from_numpy(np.where(valid, target, 0).astype(np.float32)),
            "target_mask": torch.from_numpy(valid.astype(np.float32)),
            "time_features": torch.from_numpy(time_features(self.sample_time(index) + pd.Timedelta(hours=6))),
            "era_static": torch.from_numpy(self.era_static),
            "obs_ztd": zero, "obs_mask": zero.clone(),
            "obs_static": torch.from_numpy(self.obs_static),
        }
        if self.predict_ztd:
            from .ztd_fields import read_ztd_payload
            result.update(read_ztd_payload(self._root, int(self.forecast_indices[index])))
        return result
