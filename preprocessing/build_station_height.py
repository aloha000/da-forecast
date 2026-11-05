#!/usr/bin/env python3
"""Extract existing NGL-Zarr stations' ECEF heights and apply EGM96, without changing Zarr.

Defaults require datasets/geoid/us_nga_egm96_15.tif (PROJ CDN, public domain).
Outputs are aligned with the ERA grid using the SAME mapping as DAForecastDataset.
No zero-height fallback, no station reselection, and no overwrite.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile
import numpy as np
from PIL import Image
import zarr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from preprocessing.build_global_ngl_zarr import ecef_to_geodetic, station_archives
from data.da_dataset import station_grid_mapping

def ecef_height(x, y, z):
    latitude, longitude = ecef_to_geodetic(x, y, z)
    a, f = 6378137., 1 / 298.257223563
    e2 = f * (2-f)
    lat = math.radians(latitude)
    # Stable normal projection, including near-polar stations.
    height = math.hypot(x,y)*math.cos(lat) + z*math.sin(lat) - a*math.sqrt(1-e2*math.sin(lat)**2)
    return latitude, longitude, height

def read_station(task):
    source, station = task
    errors = []
    for path in station_archives(Path(source), station):
        try:
            with zipfile.ZipFile(path) as archive:
                # Match original Zarr builder's first-valid archive/member policy.
                for member in archive.namelist():
                    try:
                        with archive.open(member) as compressed, gzip.GzipFile(fileobj=compressed) as stream:
                            section = None
                            approximate_height = None
                            for raw in stream:
                                line = raw.decode("ascii", errors="replace").strip()
                                if line.startswith("+"):
                                    section = line
                                elif line.startswith("-"):
                                    if section == "+TROP/STA_COORDINATES":
                                        break
                                    section = None
                                elif line and not line.startswith("*"):
                                    fields = line.split()
                                    if section == "+SITE/ID" and fields[0] == station:
                                        approximate_height = float(fields[-1])
                                    if section == "+TROP/STA_COORDINATES" and len(fields) >= 8 and fields[0] == station:
                                        xyz = tuple(map(float, fields[4:7]))
                                        if not np.isfinite(xyz).all():
                                            raise ValueError("nonfinite ECEF coordinates")
                                        lat, lon, height = ecef_height(*xyz)
                                        if not -1000 < height < 10000:
                                            raise ValueError("implausible station height")
                                        return dict(station=station, latitude=lat, longitude=lon,
                                                    ellipsoidal_height_m=height, approximate_height_m=approximate_height,
                                                    ecef_x_m=xyz[0], ecef_y_m=xyz[1], ecef_z_m=xyz[2],
                                                    reference_frame=fields[7], archive=str(path), member=member)
                    except (OSError, EOFError, ValueError, zipfile.BadZipFile) as exc:
                        errors.append(str(exc))
        except (OSError, zipfile.BadZipFile) as exc:
            errors.append(str(exc))
    raise ValueError(f"{station}: no valid matching station coordinates; {errors[:3]}")

def load_geoid(path):
    with Image.open(path) as im:
        tags = dict(im.tag_v2)
        description = str(tags.get(270, ""))
        metadata = str(tags.get(42112, ""))
        if ("EGM96" not in description or "geoid_undulation" not in metadata
                or "metre" not in metadata):
            raise ValueError("expected PROJ EGM96 geoid-undulation grid in metres")
        if tuple(tags.get(33550, ())) != (.25,.25,0.) or tuple(tags.get(33922, ())) != (0.,0.,0.,-180.,90.,0.):
            raise ValueError("unexpected EGM96 grid coordinates")
        keys = tags.get(34735, ())
        entries = [tuple(keys[i:i+4]) for i in range(4,len(keys),4)]
        if (1025,0,1,2) not in entries:
            raise ValueError("geoid must use pixel-is-point coordinates")
        grid = np.asarray(im, dtype=np.float64)
    if grid.shape != (721,1440) or not np.isfinite(grid).all():
        raise ValueError("invalid global EGM96 grid")
    return grid

def geoid_at(grid, latitude, longitude):
    """Bilinear point interpolation, longitude periodic, latitude including poles."""
    lat, lon = np.broadcast_arrays(np.asarray(latitude,float), np.asarray(longitude,float))
    if not np.isfinite(lat).all() or not np.isfinite(lon).all() or np.any(np.abs(lat)>90):
        raise ValueError("invalid geodetic coordinates")
    y = (90-lat)/.25
    x = ((lon+180)%360)/.25
    y0 = np.minimum(np.floor(y).astype(int),719)
    x0 = np.floor(x).astype(int)%1440
    fy, fx = y-y0, x-x0
    return ((1-fy)*((1-fx)*grid[y0,x0]+fx*grid[y0,(x0+1)%1440])
            +fy*((1-fx)*grid[y0+1,x0]+fx*grid[y0+1,(x0+1)%1440]))

def sha256(path):
    digest = hashlib.sha256()
    with open(path,"rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"datasets/global_ngl")
    parser.add_argument("--ngl-zarr", type=Path, default=ROOT/"datasets/global_ngl_0p25_5min_2023_2025.zarr")
    parser.add_argument("--era-zarr", type=Path, default=ROOT/"datasets/era5_global_0p25_6h.zarr")
    parser.add_argument("--geoid", type=Path, default=ROOT/"datasets/geoid/us_nga_egm96_15.tif")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"datasets")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    names = ["station_height_orthometric_era5.npy", "station_height_ellipsoidal_era5.npy",
             "station_geoid_undulation_era5.npy", "station_height_count_era5.npy",
             "station_height_latitude.npy", "station_height_longitude.npy",
             "station_height_stations.csv", "station_height_manifest.json"]
    paths = {name: args.output_dir/name for name in names}
    if any(p.exists() for p in paths.values()):
        raise FileExistsError("height output already exists; use a NEW --output-dir")
    geoid = load_geoid(args.geoid)
    ngl = zarr.open_group(str(args.ngl_zarr),mode="r")
    era = zarr.open_group(str(args.era_zarr),mode="r")
    lat, lon = np.asarray(era["lat"][:]), np.asarray(era["lon"][:])
    sy, sx, target, mask = station_grid_mapping(ngl, lat, lon)
    stations = np.asarray(ngl["station"][:])[sy,sx]
    stored_lat = np.asarray(ngl["station_lat"][:])[sy,sx]
    stored_lon = np.asarray(ngl["station_lon"][:])[sy,sx]
    if not len(stations) or any(not str(s) for s in stations) or len(np.unique(stations)) != len(stations):
        raise ValueError("existing Zarr station IDs must be nonempty and unique")
    records = []
    print(f"[height] extracting {len(stations)} EXISTING Zarr stations; workers={args.workers}",flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(read_station, ((str(args.source),str(s)) for s in stations))):
            error_lat = abs(row["latitude"]-float(stored_lat[i]))
            error_lon = abs((row["longitude"]-float(stored_lon[i])+180)%360-180)
            if max(error_lat,error_lon)>3e-5:
                raise ValueError(f"{row['station']}: raw coordinates differ from selected Zarr station")
            n = float(geoid_at(geoid,row["latitude"],row["longitude"]))
            row.update(geoid_undulation_m=n, orthometric_height_m=row["ellipsoidal_height_m"]-n,
                       source_lat_index=int(sy[i]),source_lon_index=int(sx[i]),
                       era_lat_index=int(target[i]//len(lon)),era_lon_index=int(target[i]%len(lon)),
                       coordinate_error_deg=max(error_lat,error_lon))
            records.append(row)
            if (i+1)%1000==0 or i+1==len(stations):
                print(f"[height] {i+1}/{len(stations)}",flush=True)
    shape = (len(lat),len(lon))
    count = np.bincount(target,minlength=mask.size).reshape(shape)
    arrays = {}
    for filename,field in zip(names[:3],("orthometric_height_m","ellipsoidal_height_m","geoid_undulation_m")):
        sums = np.bincount(target,weights=[r[field] for r in records],minlength=mask.size).reshape(shape)
        arrays[filename] = np.divide(sums,count,out=np.full(shape,np.nan),where=count>0).astype(np.float32)
    height = arrays[names[0]]
    valid = mask>0
    if not np.array_equal(np.isfinite(height),valid) or np.any((height[valid]<-500)|(height[valid]>9000)):
        raise ValueError("height coverage/range incompatible with physics loader")
    arrays.update({names[3]:count.astype(np.int32),names[4]:lat,names[5]:lon})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,value in arrays.items():
        with paths[name].open("xb") as handle:
            np.save(handle,value)
    with paths[names[6]].open("x",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    heights=np.array([r["orthometric_height_m"] for r in records])
    manifest = dict(schema="ngl_station_height_v1", units="m",height_datum="orthometric",
        vertical_crs="EGM96 height EPSG:5773",ellipsoidal_reference="IGS20 ECEF approximated on WGS84 ellipsoid",
        formula="orthometric_height = ellipsoidal_height - EGM96_geoid_undulation",
        source=str(args.source.resolve()),ngl_zarr=str(args.ngl_zarr.resolve()),era_zarr=str(args.era_zarr.resolve()),
        geoid_source="https://cdn.proj.org/us_nga_egm96_15.tif",geoid_sha256=sha256(args.geoid),
        station_selection="EXACT IDs from existing NGL Zarr; first valid coordinate using original builder archive/member order",
        station_count=len(records),occupied_era_cells=int(valid.sum()),collision_cells=int((count>1).sum()),
        aggregation="mean of selected stations mapped to each ERA cell; not a terrain-height grid",
        shape=list(shape),nonstation_fill="NaN",height_min_m=float(heights.min()),height_max_m=float(heights.max()),
        max_coordinate_error_deg=max(r["coordinate_error_deg"] for r in records),
        files={name:dict(path=str(path.resolve()),sha256=sha256(path)) for name,path in paths.items() if name!=names[-1]},
        limitations="One representative coordinate epoch per station. IGS20/WGS84 realization differences and EGM96 errors remain; static mean height at colliding station cells.")
    with paths[names[-1]].open("x") as handle:
        json.dump(manifest,handle,indent=2,allow_nan=False)
    print(json.dumps({k:manifest[k] for k in ("station_count","occupied_era_cells","collision_cells","shape","height_min_m","height_max_m","max_coordinate_error_deg")},indent=2),flush=True)
    print(f"station_height_path={paths[names[0]].resolve()}",flush=True)

if __name__=="__main__":
    main()
