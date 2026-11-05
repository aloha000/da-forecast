"""Sparse station-cell storage; component mode contains no independent ZTD label."""
import numpy as np
import torch

def field_specs(dual):
    fields = [("ztd_input", "obs_ztd", 19, 1), ("ztd_input_mask", "obs_mask", 19, 1)]
    if dual:
        fields += [("component_input", "obs_components", 19, 2),
                   ("component_input_mask", "obs_component_mask", 19, 2),
                   ("component_target", "component_target", 18, 2),
                   ("component_target_mask", "component_target_mask", 18, 2)]
    else:
        fields += [("ztd_target", "ztd_target", 18, 1), ("ztd_target_mask", "ztd_target_mask", 18, 1)]
    return fields

def initialize_ztd_fields(root, dataset, origins):
    from .rollout_fields import create_array
    flat = np.flatnonzero(dataset.obs_static[0].reshape(-1) > 0).astype(np.int64)
    if not len(flat):
        raise ValueError("predicted-ZTD rollout requires station cells")
    dual = bool(getattr(dataset, "predict_delay_components", False))
    if dual:
        root.attrs["predict_delay_components"] = True
        root.attrs["component_units"] = "mm"
        root.attrs["component_order"] = ["zhd", "zwd"]
        root.attrs["component_loss_scales"] = dataset.component_loss_scales.tolist()
        root.attrs["zwd_normalization"] = dataset.zwd_stats.tolist()
        root.attrs["component_context_keys"] = list(dataset.component_context)
        for name, value in dataset.component_context.items():
            value = np.asarray(value)
            create_array(root, "context_"+name, value.shape, value.dtype, value.shape,
                         tuple(name+"_"+str(i) for i in range(value.ndim)))[:] = value
    for name, value, dims in (
        ("station_flat", flat, ("station_cell",)),
        ("obs_static", dataset.obs_static, ("obs_static_channel", "lat", "lon")),
    ):
        value = np.asarray(value)
        create_array(root, name, value.shape, value.dtype, value.shape, dims)[:] = value
    for name, _, frames, channels in field_specs(dual):
        tail = (frames, channels, len(flat)) if channels == 2 else (frames, len(flat))
        dims = ("init_time", name+"_frame") + (("delay_component",) if channels == 2 else ()) + ("station_cell",)
        create_array(root, name, (origins, *tail), "bool" if name.endswith("_mask") else "f4",
                     (1, *tail), dims)
        root[name].attrs["offset_minutes_from_T"] = list(range(380, 721, 20)) if "target" in name else list(range(0, 361, 20))
    root["ztd_input"].attrs["description"] = "T endpoint + 18 predicted frames; NOT future truth"
    root.attrs["ztd_normalization"] = dataset.ztd_normalization
    root.attrs["ztd_input_mask_policy"] = "T endpoint validity; future eligibility from past history only"

def write_ztd_payload(root, index, payload):
    """Caller sets forecast_complete only AFTER every component/weather write."""
    h, w = len(root["lat"]), len(root["lon"])
    flat = np.asarray(root["station_flat"][:])
    for name, _, frames, channels in field_specs(root.attrs.get("predict_delay_components", False)):
        value = np.asarray(payload[name])
        if value.shape != (frames, channels, h, w) or not np.isfinite(value).all():
            raise ValueError(f"invalid {name} shape/value")
        sparse = value.reshape(frames, channels, -1)[:, :, flat]
        root[name][index] = sparse if channels == 2 else sparse[:, 0]

def validate_ztd_fields(root, origins, h, w):
    flat = np.asarray(root["station_flat"][:], np.int64)
    if (not len(flat) or np.any(np.diff(flat) <= 0) or flat.min() < 0 or flat.max() >= h*w
            or root["obs_static"].shape != (3, h, w)):
        raise ValueError("invalid ZTD station mapping/static")
    dual = root.attrs.get("predict_delay_components", False)
    if dual and (root.attrs.get("component_units") != "mm" or root.attrs.get("component_order") != ["zhd", "zwd"]):
        raise ValueError("component units/order mismatch")
    for name, _, frames, channels in field_specs(dual):
        tail = (frames, channels, len(flat)) if channels == 2 else (frames, len(flat))
        if root[name].shape != (origins, *tail):
            raise ValueError(f"invalid {name} dimensions")
    if dual:
        for key in root.attrs["component_context_keys"]:
            if not np.isfinite(root["context_"+key][:]).all():
                raise ValueError(f"nonfinite component context {key}")

def read_ztd_payload(root, index):
    h, w = len(root["lat"]), len(root["lon"])
    flat = np.asarray(root["station_flat"][:])
    result = {}
    dual = root.attrs.get("predict_delay_components", False)
    for name, output, frames, channels in field_specs(dual):
        sparse = np.asarray(root[name][index], np.float32).reshape(frames, channels, -1)
        if not np.isfinite(sparse).all():
            raise ValueError(f"nonfinite {name}")
        dense = np.zeros((frames, channels, h*w), np.float32)
        dense[:, :, flat] = sparse
        result[output] = torch.from_numpy(dense.reshape(frames, channels, h, w))
    if dual:
        result.update({key: torch.from_numpy(np.asarray(root["context_"+key][:], np.float32))
                       for key in root.attrs["component_context_keys"]})
        result["component_loss_scales"] = torch.tensor(root.attrs["component_loss_scales"], dtype=torch.float32)
    return result
