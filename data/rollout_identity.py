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
    hashes = {}
    zero_delay_inputs = config.get("zero_delay_inputs", False)
    if not isinstance(zero_delay_inputs, bool):
        raise ValueError("zero_delay_inputs must be a boolean")
    # Omit the default to keep existing non-ablation checkpoints compatible.
    if zero_delay_inputs:
        result["zero_delay_inputs"] = True
    paths = [config[k] for k in ("conds_path", "global_lat_path", "global_lon_path")]
    paths += [str(Path(config["mean_std_dir"]) / f"{name}_era5.npy") for name in ("mean", "std")]
    for value in paths:
        digest = hashlib.sha256()
        with open(value, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        hashes[str(Path(value).resolve())] = digest.hexdigest()
    result["static_and_normalization_sha256"] = hashes
    return result
