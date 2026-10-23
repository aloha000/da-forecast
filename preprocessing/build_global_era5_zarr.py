#!/usr/bin/env python3
"""Prepare the complete global ERA5-only 6-hourly Zarr for DA-Forecast.

The source ``ERA5_2017_2025`` is already a standardized, global, 70-channel
Zarr.  No IMERG data are read or written.  The default ``link`` mode validates
the source and creates a zero-copy dataset entry under this repository.  Use
``--mode copy`` only when an independent physical 1.2-TB copy is required.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(
    "/inspire/ssd/project/sais-weather/public/fuxi-obs/ERA5_2017_2025"
)
DEFAULT_OUTPUT = ROOT / "datasets" / "era5_global_0p25_6h.zarr"
DEFAULT_STATS = Path(
    "/inspire/ssd/project/sais-weather/public/fuxi-obs/obs-grid-qc_new/mean_std"
)

CHANNELS = [
    "z50", "z100", "z150", "z200", "z250", "z300", "z400", "z500",
    "z600", "z700", "z850", "z925", "z1000",
    "t50", "t100", "t150", "t200", "t250", "t300", "t400", "t500",
    "t600", "t700", "t850", "t925", "t1000",
    "u50", "u100", "u150", "u200", "u250", "u300", "u400", "u500",
    "u600", "u700", "u850", "u925", "u1000",
    "v50", "v100", "v150", "v200", "v250", "v300", "v400", "v500",
    "v600", "v700", "v850", "v925", "v1000",
    "r50", "r100", "r150", "r200", "r250", "r300", "r400", "r500",
    "r600", "r700", "r850", "r925", "r1000",
    "t2m", "u10", "v10", "msl", "tp",
]


def decode_time(group) -> pd.DatetimeIndex:
    array = group["time"]
    units = str(dict(array.attrs).get("units", ""))
    unit, separator, origin = units.partition(" since ")
    if not separator:
        raise ValueError(f"unsupported time units: {units!r}")
    codes = {
        "hour": "h", "hours": "h",
        "minute": "min", "minutes": "min",
        "day": "D", "days": "D",
    }
    code = codes.get(unit.strip().lower())
    if code is None:
        raise ValueError(f"unsupported time unit: {unit!r}")
    return pd.DatetimeIndex(
        pd.Timestamp(origin.strip())
        + pd.to_timedelta(np.asarray(array[:]), unit=code)
    )


def validate_source(path: Path, deep_check: bool = True) -> dict:
    if not path.is_dir():
        raise FileNotFoundError(f"ERA5 source does not exist: {path}")
    group = zarr.open(str(path), mode="r")
    required = {"z", "time", "channel", "lat", "lon"}
    missing = required.difference(group.array_keys())
    if missing:
        raise ValueError(f"source is missing arrays: {sorted(missing)}")

    z = group["z"]
    channels = [str(value) for value in group["channel"][:]]
    lat = np.asarray(group["lat"][:])
    lon = np.asarray(group["lon"][:])
    times = decode_time(group)
    expected_shape = (len(times), 70, 720, 1440)
    if tuple(z.shape) != expected_shape:
        raise ValueError(f"expected z{expected_shape}, got {tuple(z.shape)}")
    if channels != CHANNELS:
        raise ValueError("source channel order does not match the FuXi 70-channel order")
    if lat.shape != (720,) or lon.shape != (1440,):
        raise ValueError(f"unexpected grid: lat={lat.shape}, lon={lon.shape}")
    if not np.all(np.diff(lat) < 0):
        raise ValueError("latitude must be strictly descending north-to-south")
    if not np.all(np.diff(lon) > 0):
        raise ValueError("longitude must be strictly ascending")
    # Do not assume pandas stores DatetimeIndex internally at nanosecond
    # resolution (pandas 3 may preserve seconds).  Convert explicitly.
    steps = np.diff(times.to_numpy()).astype("timedelta64[h]").astype(np.int64)
    if len(steps) and not np.all(steps == 6):
        bad = np.flatnonzero(steps != 6)[:10]
        raise ValueError(f"time axis is not continuous at 6 hours; bad indices={bad.tolist()}")
    if z.dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError(f"expected floating ERA5 data, got {z.dtype}")

    samples = []
    if deep_check:
        for index in sorted({0, len(times) // 2, len(times) - 1}):
            field = np.asarray(z[index])
            finite_fraction = float(np.isfinite(field).mean())
            if finite_fraction != 1.0:
                raise ValueError(
                    f"ERA5 sample {index} contains non-finite data: "
                    f"finite_fraction={finite_fraction}"
                )
            samples.append({
                "index": index,
                "time": str(times[index]),
                "minimum": float(np.min(field)),
                "maximum": float(np.max(field)),
                "mean_float32": float(np.mean(field, dtype=np.float32)),
            })

    return {
        "source": str(path.resolve()),
        "array": "z",
        "shape": list(z.shape),
        "chunks": list(z.chunks),
        "dtype": str(z.dtype),
        "channels": channels,
        "n_times": len(times),
        "time_start": str(times[0]),
        "time_end": str(times[-1]),
        "time_frequency_hours": 6,
        "latitude_start": float(lat[0]),
        "latitude_end": float(lat[-1]),
        "longitude_start": float(lon[0]),
        "longitude_end": float(lon[-1]),
        "grid": "global 0.25 degree, 720 x 1440",
        "contains_imerg": False,
        "normalization": "source is already standardized per ERA5 channel",
        "sample_checks": samples,
    }


def safe_remove_output(output: Path) -> None:
    resolved_parent = output.parent.resolve()
    expected_parent = (ROOT / "datasets").resolve()
    if resolved_parent != expected_parent or output.suffix != ".zarr":
        raise ValueError(f"refusing to remove unsafe output path: {output}")
    if output.is_symlink():
        output.unlink()
    elif output.exists():
        shutil.rmtree(output)


def make_link(source: Path, output: Path, force: bool) -> None:
    if output.is_symlink():
        current = output.resolve()
        if current == source.resolve():
            print(f"[link] already correct: {output} -> {current}")
            return
        if not force:
            raise FileExistsError(f"{output} points to {current}; use --force to replace")
        safe_remove_output(output)
    elif output.exists():
        if not force:
            raise FileExistsError(f"{output} exists; use --force to replace")
        safe_remove_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.symlink_to(source.resolve(), target_is_directory=True)
    print(f"[link] {output} -> {source.resolve()}")


def create_array(group, name: str, data: np.ndarray, chunks, compressor):
    array = group.create_dataset(
        name,
        shape=data.shape,
        chunks=chunks,
        dtype=data.dtype,
        compressor=compressor,
    )
    array[:] = data
    return array


def copy_store(source: Path, output: Path, force: bool, resume: bool) -> None:
    if output.is_symlink():
        if not force:
            raise FileExistsError(f"{output} is a symlink; use --force to replace")
        safe_remove_output(output)
    if output.exists() and not resume:
        if not force:
            raise FileExistsError(f"{output} exists; use --resume or --force")
        safe_remove_output(output)

    source_group = zarr.open(str(source), mode="r")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    if output.exists():
        target = zarr.open_group(str(output), mode="r+")
        written = target["_written"]
        print(f"[copy] resuming; {int(np.asarray(written[:]).sum())}/{len(written)} complete")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        target = zarr.open_group(str(output), mode="w", zarr_format=2)
        for name in ("time", "channel", "lat", "lon"):
            source_array = source_group[name]
            data = np.asarray(source_array[:])
            array = create_array(target, name, data, source_array.chunks, compressor)
            array.attrs.update(dict(source_array.attrs))
        source_z = source_group["z"]
        target_z = target.create_dataset(
            "z",
            shape=source_z.shape,
            chunks=source_z.chunks,
            dtype=source_z.dtype,
            fill_value=np.nan,
            compressor=compressor,
        )
        target_z.attrs.update(dict(source_z.attrs))
        written = target.create_dataset(
            "_written", shape=(source_z.shape[0],), chunks=(4096,),
            dtype="bool", fill_value=False, compressor=compressor,
        )
        target.attrs.update({
            "title": "Global 6-hourly ERA5-only analysis for DA-Forecast",
            "source": str(source.resolve()),
            "contains_imerg": False,
            "normalization": "copied without transformation; already standardized",
        })

    target_z = target["z"]
    total = target_z.shape[0]
    start_time = time.time()
    for index in range(total):
        if bool(written[index]):
            continue
        target_z[index] = source_group["z"][index]
        written[index] = True
        done = index + 1
        if done % 10 == 0 or done == total:
            elapsed = time.time() - start_time
            print(f"[copy] {done}/{total} frames, elapsed={elapsed / 3600:.2f} h", flush=True)
    del target["_written"]
    zarr.consolidate_metadata(str(output))
    print(f"[copy] completed: {output}")


def write_manifest(output: Path, metadata: dict, mode: str, stats_dir: Path) -> Path:
    manifest = output.parent / f"{output.name.removesuffix('.zarr')}.manifest.json"
    content = dict(metadata)
    content.update({
        "dataset_path": str(output.absolute()),
        "storage_mode": mode,
        "mean_std_dir": str(stats_dir),
        "note": "ERA5 only; no IMERG channels or replacement precipitation",
    })
    with manifest.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, ensure_ascii=False)
    print(f"[manifest] {manifest}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--mode", choices=("link", "copy"), default="link")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true", help="resume an interrupted physical copy")
    parser.add_argument("--quick-check", action="store_true", help="skip reading three full sample fields")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if not args.stats_dir.joinpath("mean_era5.npy").is_file():
        raise FileNotFoundError(f"missing ERA5 normalization stats: {args.stats_dir}")
    metadata = validate_source(args.source, deep_check=not args.quick_check)
    print(
        f"[validated] z={tuple(metadata['shape'])} dtype={metadata['dtype']} "
        f"time={metadata['time_start']} .. {metadata['time_end']}"
    )
    if args.validate_only:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return
    if args.mode == "link":
        if args.resume:
            raise ValueError("--resume is only valid with --mode copy")
        make_link(args.source, args.output, args.force)
    else:
        copy_store(args.source, args.output, args.force, args.resume)
    write_manifest(args.output, metadata, args.mode, args.stats_dir)


if __name__ == "__main__":
    main()
