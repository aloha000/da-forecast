"""Online joint weather/delay labels; only the origin history is a model input."""
import numpy as np
import pandas as pd
import torch
from .da_dataset import DAForecastDataset, time_features
from .delay_components import component_inputs, read_components


class DelaySequenceDataset(DAForecastDataset):
    def __init__(self, *, max_steps=8, horizon=1, **kwargs):
        super().__init__(**kwargs)
        if not self.predict_delay_components:
            raise ValueError("joint rollout requires component mode")
        self.max_steps, self.horizon = int(max_steps), int(horizon)
        if not 1 <= self.horizon <= self.max_steps <= 8:
            raise ValueError("require 1 <= horizon <= max_steps <= 8")
        end = pd.Timestamp(kwargs["split"][1])
        self.analysis_times = pd.DatetimeIndex([
            t for t in self.analysis_times
            if t + pd.Timedelta(hours=6*self.max_steps) < end
            and all((t + pd.Timedelta(hours=6*j)).value in self.label_lookup
                    for j in range(1, self.max_steps+1))
        ])
        if not len(self):
            raise ValueError("no origins with all weather targets inside split")
        self.observation_normalization = {
            "ztd_stats": self.ztd_stats.tolist(), "zwd_stats": self.zwd_stats.tolist(),
            "components": ["zhd", "zwd"], "component_units": "mm",
            "component_loss_scales": self.component_loss_scales.tolist(),
            "policy": "original_ztd_encoder_plus_physical_component_adapter_v1",
        }

    def __getitem__(self, index):
        t = self.sample_time(index)
        self._open()
        indices = [self.label_lookup[(t+pd.Timedelta(hours=6*j)).value]
                   for j in range(-1, self.horizon+1)]
        if not np.all(np.diff(indices) == 1):
            raise ValueError("ERA sequence must be contiguous at 6h")
        sequence = np.asarray(self._era_group["z"][indices[0]:indices[-1]+1], np.float32)
        targets = sequence[2:]
        valid = np.isfinite(targets)
        era = np.nan_to_num(sequence[:2], nan=0., posinf=0., neginf=0.)
        obs, mask = self._read_observations(t)
        result = {
            "era5": torch.from_numpy(era),
            "targets": torch.from_numpy(np.where(valid, targets, 0)),
            "target_masks": torch.from_numpy(valid),
            "era_static": torch.from_numpy(self.era_static),
            "obs_ztd": torch.from_numpy(obs[:, None]),
            "obs_mask": torch.from_numpy(mask[:, None].astype(bool)),
            "obs_static": torch.from_numpy(self.obs_static),
            "time_features": torch.from_numpy(time_features(t)),
            "rollout_time_features": torch.from_numpy(np.stack([
                time_features(t+pd.Timedelta(hours=6*j)) for j in range(self.horizon)])),
            "component_loss_scales": torch.from_numpy(self.component_loss_scales.copy()),
        }
        result.update(component_inputs(self, t))  # ONLY history T-6..T.
        result["obs_component_mask"] = result["obs_component_mask"].bool()
        # Future GNSS values are LABELS ONLY. Their availability never changes
        # origin selection, input values, or recurrent pseudo-observation masks.
        shape = (self.horizon, 18, 2, len(self.lat), len(self.lon))
        values = np.empty(shape, np.float32)
        masks = np.empty(shape, bool)
        for j in range(self.horizon):
            target, target_mask = read_components(self, t+pd.Timedelta(hours=6*j), future=True)
            values[j], masks[j] = target, target_mask > 0
        result["component_targets"] = torch.from_numpy(values)
        result["component_target_masks"] = torch.from_numpy(masks)
        return result


def make_rollout_dataset(data, split, max_steps=8, horizon=1):
    if data.get("rollout_zarr") or int(data.get("forecast_lead_hours", 6)) != 6:
        raise ValueError("online joint rollout reads original ERA at a 6h base step")
    if data.get("zero_delay_inputs"):
        raise ValueError("zero-input ablation is not enabled for predicted-component rollout")
    keys = ("era5_path", "ztd_path", "conds_path", "global_lat_path", "global_lon_path",
            "obs_history_hours", "obs_interval_minutes", "predict_ztd", "predict_delay_components",
            "component_loss_scales", "use_delay_physics", "station_height_path",
            "station_height_datum", "mean_std_dir")
    return DelaySequenceDataset(**{k:data[k] for k in keys if k in data},
                                split=data[split+"_split"], max_steps=max_steps, horizon=horizon)
