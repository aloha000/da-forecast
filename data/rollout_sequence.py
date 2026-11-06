"""On-the-fly weather rollout labels; real ZHD/ZWD only at the original origin."""
import numpy as np
import pandas as pd
import torch
import zarr
from .da_dataset import DAForecastDataset, project_station_values, time_features

def delay_stats(group, field):
    attrs = dict(group[field].attrs)
    stats = np.array([group.attrs.get(field+"_train_mean",np.nan),
                      group.attrs.get(field+"_train_std",np.nan)],dtype=np.float64)
    if attrs.get("raw_units") != "mm" or str(attrs.get("units")) != "1":
        raise ValueError(f"{field}: expected standardized source data, raw_units=mm and units=1")
    if not np.isfinite(stats).all() or stats[1]<=0:
        raise ValueError(f"invalid {field} train normalization")
    return stats

def split_delays(ztd, zwd, total_stats, wet_stats):
    """Shared total std preserves old ZTD encoder by a strict kernel migration."""
    valid = np.isfinite(ztd) & np.isfinite(zwd)
    # Compute in float64 before casting: independent source normalizations.
    total = ztd.astype(np.float64)*total_stats[1]+total_stats[0]
    wet = zwd.astype(np.float64)*wet_stats[1]+wet_stats[0]
    h = (total-wet-(total_stats[0]-wet_stats[0]))/total_stats[1]
    w = (wet-wet_stats[0])/total_stats[1]
    return np.where(valid,h,np.nan).astype(np.float32), np.where(valid,w,np.nan).astype(np.float32)

class RolloutSequenceDataset(DAForecastDataset):
    def __init__(self, *, max_steps=8, horizon=1, **kwargs):
        super().__init__(**kwargs)
        self.max_steps,self.horizon=int(max_steps),int(horizon)
        if not 1<=self.horizon<=self.max_steps<=8:
            raise ValueError("require 1 <= horizon <= max_steps <= 8")
        self.split_bounds=list(kwargs["split"])
        end=pd.Timestamp(self.split_bounds[1])
        # Fixed population for all curriculum stages. No T+48 label crosses split.
        self.analysis_times=pd.DatetimeIndex([t for t in self.analysis_times
            if t+pd.Timedelta(hours=6*self.max_steps)<end
            and all((t+pd.Timedelta(hours=6*j)).value in self.label_lookup for j in range(1,self.max_steps+1))])
        if not len(self):
            raise ValueError("no origins with full rollout labels inside split")
        group=zarr.open_group(self.ztd_path,mode="r")
        self.total_stats=delay_stats(group,"ztd")
        self.wet_stats=delay_stats(group,"zwd")
        self.observation_normalization={"order":["zhd","zwd"],
            "mean_mm":[float(self.total_stats[0]-self.wet_stats[0]),float(self.wet_stats[0])],
            "scale_mm":[float(self.total_stats[1])]*2,"source_ztd":self.total_stats.tolist(),
            "source_zwd":self.wet_stats.tolist(),"policy":"paired_valid_shared_ztd_std_v1"}

    def _read_components(self, timestamp):
        self._open()
        indices=self._obs_indices(timestamp)
        if indices is None:
            raise ValueError("missing origin observation history")
        stride=int(indices[1]-indices[0])
        if not np.all(np.diff(indices)==stride):
            raise ValueError("irregular observation history")
        section=slice(int(indices[0]),int(indices[-1])+1,stride)
        ztd=np.asarray(self._ztd_group["ztd"][section],np.float32)
        zwd=np.asarray(self._ztd_group["zwd"][section],np.float32)
        components=split_delays(ztd,zwd,self.total_stats,self.wet_stats)
        values,masks=[],[]
        for component in components:
            value,mask=project_station_values(component,self.obs_source_lat,self.obs_source_lon,
                                              self.obs_target_flat,(len(self.lat),len(self.lon)))
            values.append(value);masks.append(mask)
        return np.stack(values,axis=1),np.stack(masks,axis=1).astype(np.float32)

    def __getitem__(self,index):
        t=self.sample_time(index)
        self._open()
        indices=[self.label_lookup[(t+pd.Timedelta(hours=6*j)).value] for j in range(-1,self.horizon+1)]
        if not np.all(np.diff(indices)==1):
            raise ValueError("ERA sequence must be contiguous at 6h")
        sequence=np.asarray(self._era_group["z"][indices[0]:indices[-1]+1],np.float32)
        targets=sequence[2:]
        valid=np.isfinite(targets)
        era=np.nan_to_num(sequence[:2],nan=0.,posinf=0.,neginf=0.)
        targets=np.where(valid,targets,0).astype(np.float32)
        obs,mask=self._read_components(t) # NO observations after T are read.
        return {"era5":torch.from_numpy(era),"targets":torch.from_numpy(targets),
                "target_masks":torch.from_numpy(valid),
                "era_static":torch.from_numpy(self.era_static),"obs_ztd":torch.from_numpy(obs),
                "obs_mask":torch.from_numpy(mask),"obs_static":torch.from_numpy(self.obs_static),
                "rollout_time_features":torch.from_numpy(np.stack([
                    time_features(t+pd.Timedelta(hours=6*j)) for j in range(self.horizon)]))}

def make_rollout_dataset(data,split,max_steps=8,horizon=1):
    if data.get("rollout_zarr"):
        raise ValueError("online curriculum must read original ERA, not offline rollout_zarr")
    keys=("era5_path","ztd_path","conds_path","global_lat_path","global_lon_path",
          "obs_history_hours","obs_interval_minutes")
    return RolloutSequenceDataset(**{k:data[k] for k in keys},split=data[split+"_split"],
                                  max_steps=max_steps,horizon=horizon)
