"""Sparse station-cell storage for self-contained predicted-ZTD rollout data."""
import numpy as np
import torch


def initialize_ztd_fields(root, dataset, origins):
    from .rollout_fields import create_array
    flat = np.flatnonzero(dataset.obs_static[0].reshape(-1) > 0).astype(np.int64)
    if not len(flat):
        raise ValueError("predicted-ZTD rollout requires station cells")
    for name, value, dims in (
        ("station_flat", flat, ("station_cell",)),
        ("obs_static", dataset.obs_static, ("obs_static_channel", "lat", "lon")),
    ):
        value = np.asarray(value)
        create_array(root, name, value.shape, value.dtype, value.shape, dims)[:] = value
    for name, frames, dtype in (("ztd_input", 19, "f4"), ("ztd_input_mask", 19, "bool"),
                                 ("ztd_target", 18, "f4"), ("ztd_target_mask", 18, "bool")):
        create_array(root, name, (origins, frames, len(flat)), dtype,
                     (1, frames, len(flat)), ("init_time", name + "_frame", "station_cell"))
    root["ztd_input"].attrs.update({"offset_minutes_from_T": list(range(0, 361, 20)),
                                   "description": "observed T endpoint + 18 predicted frames; not future truth"})
    root["ztd_target"].attrs["offset_minutes_from_T"] = list(range(380, 721, 20))
    root.attrs["ztd_normalization"] = dataset.ztd_normalization
    root.attrs["ztd_input_mask_policy"] = "T endpoint validity; future eligibility = any valid input in T-6h..T"


def write_ztd_payload(root, index, payload):
    """Called after clearing forecast_complete; caller sets it only after ALL writes."""
    h, w = len(root["lat"]), len(root["lon"])
    flat = np.asarray(root["station_flat"][:])
    for name, frames in (("ztd_input", 19), ("ztd_input_mask", 19),
                         ("ztd_target", 18), ("ztd_target_mask", 18)):
        value = np.asarray(payload[name])
        if value.shape != (frames, 1, h, w) or not np.isfinite(value).all():
            raise ValueError(f"invalid {name} shape/value")
        root[name][index] = value.reshape(frames, -1)[:, flat]


def validate_ztd_fields(root, origins, h, w):
    flat = np.asarray(root["station_flat"][:], np.int64)
    if (not len(flat) or np.any(np.diff(flat) <= 0) or flat.min() < 0 or flat.max() >= h*w
            or root["obs_static"].shape != (3, h, w)):
        raise ValueError("invalid ZTD station mapping/static")
    for name, frames in (("ztd_input", 19), ("ztd_input_mask", 19),
                         ("ztd_target", 18), ("ztd_target_mask", 18)):
        if root[name].shape != (origins, frames, len(flat)):
            raise ValueError(f"invalid {name} dimensions")


def read_ztd_payload(root, index):
    h, w = len(root["lat"]), len(root["lon"])
    flat = np.asarray(root["station_flat"][:])
    result = {}
    for name, output, frames in (("ztd_input", "obs_ztd", 19), ("ztd_input_mask", "obs_mask", 19),
                                ("ztd_target", "ztd_target", 18), ("ztd_target_mask", "ztd_target_mask", 18)):
        sparse = np.asarray(root[name][index], np.float32)
        if not np.isfinite(sparse).all():
            raise ValueError(f"nonfinite {name}")
        dense = np.zeros((frames, h*w), np.float32)
        dense[:, flat] = sparse
        result[output] = torch.from_numpy(dense.reshape(frames, 1, h, w))
    return result
