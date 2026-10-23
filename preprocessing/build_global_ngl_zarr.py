#!/usr/bin/env python3
"""Build a global 0.25-degree, native 5-minute NGL ZTD/ZWD Zarr.

The input layout is expected to be::

    global_ngl/<STATION>/<STATION>.<YEAR>.trop.zip

Station coordinates are read from ``+TROP/STA_COORDINATES`` in the SINEX-TRO
files and mapped to the ERA5 720x1440 grid.  If several stations map to the
same cell, the station with the most available yearly archives (then largest
total archive size) is selected.

All native 5-minute records from 2023 through 2025 are retained.  The script
does not allocate a dense [time, 720, 1440] array in RAM: selected station
series are parsed to a resumable cache and then written one spatial Zarr tile
at a time.  Missing grid cells/times remain NaN and compress efficiently.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import shutil
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "datasets" / "global_ngl"
DEFAULT_OUTPUT = ROOT / "datasets" / "global_ngl_0p25_5min_2023_2025.zarr"

START = pd.Timestamp("2023-01-01T00:00:00")
END = pd.Timestamp("2026-01-01T00:00:00")
EPOCH = "2023-01-01 00:00:00"
FREQUENCY_MINUTES = 5
YEARS = (2023, 2024, 2025)

RESOLUTION = 0.25
N_LAT = 720
N_LON = 1440
LAT = 90.0 - RESOLUTION * np.arange(N_LAT, dtype=np.float64)
LON = RESOLUTION * np.arange(N_LON, dtype=np.float64)


def parse_epoch(epoch: str) -> np.datetime64:
    year, day, seconds = (int(value) for value in epoch.split(":"))
    year += 2000 if year < 80 else 1900
    base = np.datetime64(f"{year:04d}-01-01T00:00:00", "s")
    return base + np.timedelta64(day - 1, "D") + np.timedelta64(seconds, "s")


def ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float]:
    """Convert IGS/WGS84 ECEF metres to geodetic latitude/longitude degrees."""
    a = 6378137.0
    inverse_flattening = 298.257223563
    flattening = 1.0 / inverse_flattening
    eccentricity_sq = flattening * (2.0 - flattening)
    longitude = math.atan2(y, x)
    radius = math.hypot(x, y)
    latitude = math.atan2(z, radius * (1.0 - eccentricity_sq))
    for _ in range(8):
        sin_lat = math.sin(latitude)
        prime_vertical = a / math.sqrt(1.0 - eccentricity_sq * sin_lat * sin_lat)
        height = radius / max(math.cos(latitude), 1e-12) - prime_vertical
        updated = math.atan2(
            z,
            radius * (1.0 - eccentricity_sq * prime_vertical / (prime_vertical + height)),
        )
        if abs(updated - latitude) < 1e-13:
            latitude = updated
            break
        latitude = updated
    return math.degrees(latitude), math.degrees(longitude) % 360.0


def station_archives(source: Path, station: str) -> list[Path]:
    """Return available 2023--2025 archives, also accepting the old layout."""
    archives = []
    for year in YEARS:
        name = f"{station}.{year}.trop.zip"
        candidates = (source / station / name, source / str(year) / name)
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is not None:
            archives.append(path)
    return archives


def read_coordinates(task):
    """Read one station coordinate and cheap archive-coverage metadata."""
    station, archive_names = task
    archives = [Path(name) for name in archive_names]
    for archive in archives:
        try:
            with zipfile.ZipFile(archive) as outer:
                for member in outer.namelist():
                    try:
                        payload = outer.read(member)
                        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as inner:
                            in_coordinates = False
                            for raw in inner:
                                line = raw.decode("ascii", errors="replace").strip()
                                if line == "+TROP/STA_COORDINATES":
                                    in_coordinates = True
                                    continue
                                if line == "-TROP/STA_COORDINATES":
                                    break
                                if not in_coordinates or not line or line.startswith("*"):
                                    continue
                                fields = line.split()
                                if len(fields) >= 7:
                                    lat, lon = ecef_to_geodetic(
                                        float(fields[4]), float(fields[5]), float(fields[6])
                                    )
                                    return {
                                        "station": station,
                                        "lat": lat,
                                        "lon": lon,
                                        "archives": len(archives),
                                        "archive_bytes": sum(p.stat().st_size for p in archives),
                                        "archive_names": [str(p) for p in archives],
                                    }
                    except (OSError, EOFError, gzip.BadGzipFile, ValueError):
                        continue
        except (OSError, zipfile.BadZipFile):
            continue
    return None


def grid_cell(latitude: float, longitude: float) -> tuple[int, int]:
    lat_index = int(np.rint((90.0 - latitude) / RESOLUTION))
    lon_index = int(np.rint((longitude % 360.0) / RESOLUTION)) % N_LON
    return int(np.clip(lat_index, 0, N_LAT - 1)), lon_index


def discover_stations(source: Path, workers: int) -> tuple[list[dict], dict]:
    station_names = sorted(
        path.name for path in source.iterdir() if path.is_dir()
    )
    tasks = []
    for station in station_names:
        archives = station_archives(source, station)
        if archives:
            tasks.append((station, [str(path) for path in archives]))
    print(f"[scan] {len(station_names)} directories, {len(tasks)} with 2023--2025 archives")

    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, result in enumerate(pool.map(read_coordinates, tasks, chunksize=8), 1):
            if result is not None:
                result["lat_index"], result["lon_index"] = grid_cell(
                    result["lat"], result["lon"]
                )
                records.append(result)
            if done % 1000 == 0 or done == len(tasks):
                print(f"[scan] coordinates {done}/{len(tasks)}; valid={len(records)}", flush=True)

    by_cell = defaultdict(list)
    for record in records:
        by_cell[(record["lat_index"], record["lon_index"])].append(record)
    selected = []
    for candidates in by_cell.values():
        selected.append(
            max(
                candidates,
                key=lambda item: (
                    item["archives"], item["archive_bytes"], item["station"]
                ),
            )
        )
    selected.sort(key=lambda item: (item["lat_index"], item["lon_index"]))
    summary = {
        "station_directories": len(station_names),
        "stations_with_archives": len(tasks),
        "stations_with_coordinates": len(records),
        "occupied_grid_cells": len(selected),
        "colliding_stations_removed": len(records) - len(selected),
    }
    print(f"[grid] {summary}")
    return selected, summary


def parse_station_to_cache(task):
    """Parse one selected station and atomically write raw float32 cache."""
    (
        station,
        archive_names,
        cache_name,
        start_text,
        end_text,
        n_steps,
        with_wet,
        stats_start_index,
        stats_end_index,
    ) = task
    cache_path = Path(cache_name)
    if cache_path.is_file():
        try:
            cached = np.load(cache_path)
            ztd = np.asarray(cached["ztd"], dtype=np.float32)
            zwd = np.asarray(cached["zwd"], dtype=np.float32) if with_wet else None
            if ztd.shape == (n_steps,) and (not with_wet or zwd.shape == (n_steps,)):
                return station, _series_statistics(
                    ztd, zwd, stats_start_index, stats_end_index
                ), True
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            pass

    start = np.datetime64(start_text, "s")
    end = np.datetime64(end_text, "s")
    ztd = np.full(n_steps, np.nan, dtype=np.float32)
    zwd = np.full(n_steps, np.nan, dtype=np.float32) if with_wet else None
    records = 0
    for archive_name in archive_names:
        try:
            with zipfile.ZipFile(archive_name) as outer:
                for member in outer.namelist():
                    try:
                        payload = outer.read(member)
                        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as inner:
                            in_solution = False
                            for raw in inner:
                                line = raw.decode("ascii", errors="replace").strip()
                                if line == "+TROP/SOLUTION":
                                    in_solution = True
                                    continue
                                if line == "-TROP/SOLUTION":
                                    break
                                if not in_solution or not line or line.startswith("*"):
                                    continue
                                fields = line.split()
                                if len(fields) < 5:
                                    continue
                                seconds = int(fields[1].rsplit(":", 1)[1])
                                if seconds % (FREQUENCY_MINUTES * 60) != 0:
                                    continue
                                timestamp = parse_epoch(fields[1])
                                if timestamp < start or timestamp >= end:
                                    continue
                                index = int(
                                    
                                    (timestamp - start)
                                    / np.timedelta64(FREQUENCY_MINUTES, "m")
                                )
                                if 0 <= index < n_steps:
                                    ztd[index] = float(fields[2])
                                    if with_wet:
                                        try:
                                            zwd[index] = float(fields[4])
                                        except ValueError:
                                            pass
                                    records += 1
                    except (OSError, EOFError, gzip.BadGzipFile, ValueError):
                        continue
        except (OSError, zipfile.BadZipFile):
            continue

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        if with_wet:
            np.savez_compressed(handle, ztd=ztd, zwd=zwd)
        else:
            np.savez_compressed(handle, ztd=ztd)
    os.replace(temporary, cache_path)
    stats = _series_statistics(ztd, zwd, stats_start_index, stats_end_index)
    stats["records"] = records
    return station, stats, False


def _field_statistics(values: np.ndarray) -> tuple[int, float, float]:
    finite = values[np.isfinite(values)].astype(np.float64)
    return len(finite), float(finite.sum()), float(np.square(finite).sum())


def _series_statistics(ztd, zwd, first, last):
    ztd_count, ztd_sum, ztd_sumsq = _field_statistics(ztd[first:last])
    result = {
        "ztd_count": ztd_count,
        "ztd_sum": ztd_sum,
        "ztd_sumsq": ztd_sumsq,
        "records": int(np.isfinite(ztd).sum()),
    }
    if zwd is not None:
        count, total, sumsq = _field_statistics(zwd[first:last])
        result.update(zwd_count=count, zwd_sum=total, zwd_sumsq=sumsq)
    return result


def mean_std(total: dict, name: str) -> tuple[float, float]:
    count = int(total[f"{name}_count"])
    if count == 0:
        raise RuntimeError(f"no finite {name} values in the statistics period")
    mean = total[f"{name}_sum"] / count
    variance = max(total[f"{name}_sumsq"] / count - mean * mean, 0.0)
    std = math.sqrt(variance)
    if not np.isfinite(std) or std <= 0:
        raise RuntimeError(f"invalid {name} standard deviation: {std}")
    return float(mean), float(std)


def create_array(group, name, data, chunks, compressor, fill_value=None):
    array = group.create_dataset(
        name,
        shape=data.shape,
        chunks=chunks,
        dtype=data.dtype,
        fill_value=fill_value,
        compressor=compressor,
    )
    array[:] = data
    return array


def create_store(output, times, selected, with_wet, time_chunk, tile_size):
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    # Zarr 3 uses ``zarr_format`` while older Zarr 2 releases either use
    # ``zarr_version`` or always create v2 stores.  Keep the generated store
    # at format 2 because numcodecs.Blosc is accepted consistently there.
    try:
        group = zarr.open_group(str(output), mode="w", zarr_format=2)
    except TypeError:
        try:
            group = zarr.open_group(str(output), mode="w", zarr_version=2)
        except TypeError:
            group = zarr.open_group(str(output), mode="w")
    minute_values = (
        (times - pd.Timestamp(EPOCH)).total_seconds().to_numpy().astype(np.int64) // 60
    )
    create_array(group, "time", minute_values, (4096,), compressor, None)
    create_array(group, "lat", LAT, (N_LAT,), compressor, None)
    create_array(group, "lon", LON, (N_LON,), compressor, None)

    max_length = max((len(item["station"]) for item in selected), default=1)
    stations = np.full((N_LAT, N_LON), "", dtype=f"<U{max_length}")
    station_lat = np.full((N_LAT, N_LON), np.nan, dtype=np.float32)
    station_lon = np.full((N_LAT, N_LON), np.nan, dtype=np.float32)
    for item in selected:
        i, j = item["lat_index"], item["lon_index"]
        stations[i, j] = item["station"]
        station_lat[i, j] = item["lat"]
        station_lon[i, j] = item["lon"]
    create_array(group, "station", stations, (90, 180), compressor, None)
    create_array(group, "mask", stations == "", (90, 180), compressor, None)
    create_array(group, "station_lat", station_lat, (90, 180), compressor, np.nan)
    create_array(group, "station_lon", station_lon, (90, 180), compressor, np.nan)

    chunks = (time_chunk, tile_size, tile_size)
    group.create_dataset(
        "ztd", shape=(len(times), N_LAT, N_LON), chunks=chunks,
        dtype="f4", fill_value=np.nan, compressor=compressor,
    )
    if with_wet:
        group.create_dataset(
            "zwd", shape=(len(times), N_LAT, N_LON), chunks=chunks,
            dtype="f4", fill_value=np.nan, compressor=compressor,
        )
    return group, stations


def write_tiles(group, selected, cache_dir, stats, time_chunk, tile_size, with_wet):
    by_tile = defaultdict(list)
    for item in selected:
        key = (item["lat_index"] // tile_size, item["lon_index"] // tile_size)
        by_tile[key].append(item)
    print(f"[write] {len(by_tile)} occupied spatial tiles")

    for tile_number, ((tile_i, tile_j), items) in enumerate(sorted(by_tile.items()), 1):
        lat0, lon0 = tile_i * tile_size, tile_j * tile_size
        lat1, lon1 = min(lat0 + tile_size, N_LAT), min(lon0 + tile_size, N_LON)
        cache = {}
        for item in items:
            loaded = np.load(cache_dir / f"{item['station']}.npz")
            cache[item["station"]] = {
                "ztd": np.asarray(loaded["ztd"], dtype=np.float32),
                **({"zwd": np.asarray(loaded["zwd"], dtype=np.float32)} if with_wet else {}),
            }
        n_time = group["ztd"].shape[0]
        for start in range(0, n_time, time_chunk):
            stop = min(start + time_chunk, n_time)
            for field in ("ztd", "zwd") if with_wet else ("ztd",):
                block = np.full(
                    (stop - start, lat1 - lat0, lon1 - lon0),
                    np.nan, dtype=np.float32,
                )
                mean, std = stats[field]
                for item in items:
                    values = cache[item["station"]][field][start:stop]
                    block[:, item["lat_index"] - lat0, item["lon_index"] - lon0] = (
                        values - mean
                    ) / std
                group[field][start:stop, lat0:lat1, lon0:lon1] = block
        if tile_number % 20 == 0 or tile_number == len(by_tile):
            print(f"[write] tiles {tile_number}/{len(by_tile)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--time-chunk", type=int, default=288,
                        help="5-minute frames per chunk; 288 is one day")
    parser.add_argument("--tile-size", type=int, default=45)
    parser.add_argument("--stats-start", default="2023-01-01T00:00:00")
    parser.add_argument("--stats-end", default="2024-05-01T00:00:00")
    parser.add_argument("--with-wet", dest="with_wet", action="store_true", default=True)
    parser.add_argument("--no-wet", dest="with_wet", action="store_false")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args()

    if not args.source.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {args.source}")
    if args.time_chunk <= 0 or args.tile_size <= 0:
        raise ValueError("time-chunk and tile-size must be positive")
    times = pd.date_range(START, END, freq="5min", inclusive="left")
    expected = int((END - START) / pd.Timedelta(minutes=5))
    if len(times) != expected:
        raise AssertionError((len(times), expected))
    stats_start, stats_end = pd.Timestamp(args.stats_start), pd.Timestamp(args.stats_end)
    if not (START <= stats_start < stats_end <= END):
        raise ValueError("statistics period must be inside 2023-01-01..2026-01-01")
    stats_first = int((stats_start - START) / pd.Timedelta(minutes=5))
    stats_last = int((stats_end - START) / pd.Timedelta(minutes=5))
    cache_dir = args.cache_dir or args.output.with_suffix(".station_cache")

    selected, scan_summary = discover_stations(args.source, args.workers)
    if args.scan_only:
        print(json.dumps(scan_summary, indent=2))
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            item["station"], item["archive_names"],
            str(cache_dir / f"{item['station']}.npz"), str(START), str(END),
            len(times), args.with_wet, stats_first, stats_last,
        )
        for item in selected
    ]
    totals = defaultdict(float)
    cache_hits = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, (_, values, cached) in enumerate(
            pool.map(parse_station_to_cache, tasks, chunksize=1), 1
        ):
            for key, value in values.items():
                totals[key] += value
            cache_hits += int(cached)
            if done % 200 == 0 or done == len(tasks):
                print(
                    f"[parse] {done}/{len(tasks)} stations; cache_hits={cache_hits}",
                    flush=True,
                )
    stats = {"ztd": mean_std(totals, "ztd")}
    if args.with_wet:
        stats["zwd"] = mean_std(totals, "zwd")
    print(f"[stats] period={stats_start}..{stats_end}; {stats}")

    if args.output.exists():
        if not args.force:
            raise FileExistsError(f"{args.output} exists; use --force to replace it")
        if args.output.is_symlink() or not args.output.is_dir():
            raise ValueError(f"refusing to recursively remove unsafe output: {args.output}")
        shutil.rmtree(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    group, stations = create_store(
        args.output, times, selected, args.with_wet, args.time_chunk, args.tile_size
    )
    write_tiles(
        group, selected, cache_dir, stats, args.time_chunk, args.tile_size, args.with_wet
    )

    group["time"].attrs.update({
        "_ARRAY_DIMENSIONS": ["time"], "standard_name": "time",
        "timezone": "UTC", "units": f"minutes since {EPOCH}",
    })
    group["lat"].attrs.update({"_ARRAY_DIMENSIONS": ["lat"], "units": "degrees_north"})
    group["lon"].attrs.update({"_ARRAY_DIMENSIONS": ["lon"], "units": "degrees_east"})
    for name in ("station", "mask", "station_lat", "station_lon"):
        group[name].attrs["_ARRAY_DIMENSIONS"] = ["lat", "lon"]
    for field, (mean, std) in stats.items():
        group[field].attrs.update({
            "_ARRAY_DIMENSIONS": ["time", "lat", "lon"],
            "_FillValue": np.nan, "raw_units": "mm", "units": "1",
            "standardisation": f"({field}_mm - {field}_train_mean) / {field}_train_std",
        })
        group.attrs[f"{field}_train_mean"] = mean
        group.attrs[f"{field}_train_std"] = std
    group["mask"].attrs["description"] = "True where no selected GNSS station is present"
    group.attrs.update({
        "title": "Global NGL native 5-minute zenith delays on the ERA5 0.25-degree grid",
        "source": str(args.source.resolve()),
        "grid": "global 720x1440 at 0.25 degree; latitude north-to-south; longitude 0..359.75",
        "time_sampling": "5 minutes; all native exact UTC 5-minute records retained",
        "start_utc": str(times[0]), "end_utc": str(times[-1]),
        "n_times": len(times), "n_stations": len(selected),
        "masked_cells": int((stations == "").sum()),
        "statistics_start": str(stats_start), "statistics_end": str(stats_end),
        "station_selection": "one station per 0.25-degree cell; max archive years then bytes",
        **scan_summary,
    })
    zarr.consolidate_metadata(str(args.output))
    manifest = args.output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({
        "output": str(args.output.absolute()), "cache_dir": str(cache_dir.absolute()),
        "shape": [len(times), N_LAT, N_LON], "chunks": list(group["ztd"].chunks),
        "time_start": str(times[0]), "time_end": str(times[-1]),
        "frequency_minutes": FREQUENCY_MINUTES, "statistics": stats,
        "with_wet": args.with_wet, **scan_summary,
    }, indent=2), encoding="utf-8")
    print(f"[output] {args.output}")
    print(f"[manifest] {manifest}")


if __name__ == "__main__":
    main()
