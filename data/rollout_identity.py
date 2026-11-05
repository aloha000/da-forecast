"""Provenance fingerprints for immutable rollout inputs (not whole-data hashes)."""
from pathlib import Path
import hashlib


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def data_identity(config):
    result = {key: config[key] for key in (
        "era5_path", "ztd_path", "conds_path", "global_lat_path", "global_lon_path",
        "mean_std_dir", "obs_history_hours", "obs_interval_minutes",
        "train_split", "val_split", "test_split",
    )}
    if config.get("predict_delay_components", False):
        result["component_mode"] = {k: config.get(k) for k in
            ("predict_delay_components", "component_loss_scales", "use_delay_physics",
             "station_height_path", "station_height_datum")}
    hashes = {}
    paths = [config[k] for k in ("conds_path", "global_lat_path", "global_lon_path")]
    paths += [str(Path(config["mean_std_dir"]) / f"{name}_era5.npy") for name in ("mean", "std")]
    if config.get("predict_delay_components") and config.get("station_height_path"):
        paths.append(config["station_height_path"])
    for value in paths:
        digest = hashlib.sha256()
        with open(value, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        hashes[str(Path(value).resolve())] = digest.hexdigest()
    result["static_and_normalization_sha256"] = hashes
    return result
