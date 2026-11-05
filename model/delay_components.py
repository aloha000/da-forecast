"""ZHD/ZWD in millimetres, ordered [ZHD, ZWD]. No independent ZTD loss."""
import torch
from torch import nn
from .ztd_rollout import next_observations, ztd_error_sums

COMPONENT_INPUTS = ("obs_components", "obs_component_mask", "ztd_stats",
                    "station_height", "station_latitude", "weather_mean", "weather_std")


def latest_components(values, mask):
    if values.ndim != 5 or values.shape[1:3] != (19, 2) or mask.shape != values.shape:
        raise ValueError("components must be [B,19,2,H,W] in mm")
    valid = (mask > 0) & torch.isfinite(values)
    indices = torch.arange(19, device=values.device).view(1, 19, 1, 1, 1)
    last = torch.where(valid, indices, -1).amax(1, keepdim=True)
    available = last >= 0
    selected = torch.gather(torch.where(valid, values, 0), 1, last.clamp_min(0))
    return torch.where(available, selected, 0), available


class ComponentAdapter(nn.Module):
    """Zero-initialized correction; retains all original ZTD-encoder weights."""
    def __init__(self, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(6, embed_dim, patch_size, stride=patch_size)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, values, mask):
        last, available = latest_components(values, mask)
        valid = (mask > 0) & torch.isfinite(values)
        mean = torch.where(valid, values, 0).sum(1) / valid.sum(1).clamp_min(1)
        features = torch.cat((last[:, 0] / 1000., mean / 1000., available[:, 0].to(values.dtype)), 1)
        gate = torch.nn.functional.max_pool2d(available[:, 0].any(1, keepdim=True).float(),
                                              self.proj.kernel_size, self.proj.stride)
        return self.proj(features) * gate


def component_prediction(delta, batch_values, batch_mask, scales_mm, ztd_stats, physics_change=None):
    if batch_values.dtype not in (torch.float32, torch.float64) or ztd_stats.dtype not in (torch.float32, torch.float64):
        raise ValueError("physical component inputs/ZTD statistics require FP32 (disable FSDP root input casting)")
    if ztd_stats.ndim != 2 or ztd_stats.shape != (delta.shape[0], 2) or not torch.isfinite(ztd_stats).all() or not (ztd_stats[:, 1] > 0).all():
        raise ValueError("ztd_stats must contain finite [mean_mm, positive_std_mm] per sample")
    base, available = latest_components(batch_values.float(), batch_mask)
    scales = delta.new_tensor(scales_mm).view(1, 1, 2, 1, 1).float()
    components = base + delta.float() * scales
    if physics_change is not None:
        alpha = torch.arange(1, 19, device=delta.device, dtype=torch.float32).view(1, 18, 1, 1, 1) / 18
        components = components + alpha * physics_change[:, None]
    forecast_mask = available.expand_as(components)
    components = torch.where(forecast_mask, components, 0)
    # Sum physical mm first, then use the ORIGINAL ZTD normalization for its encoder.
    mean = ztd_stats[:, 0].float().view(-1, 1, 1, 1, 1)
    std = ztd_stats[:, 1].float().view(-1, 1, 1, 1, 1)
    ztd = (components.sum(2, keepdim=True) - mean) / std
    return {"components": components, "component_mask": forecast_mask, "ztd": ztd}


def component_error_sums(output, batch, sample_valid=None, scaled=True):
    prediction = output["components"].float()
    target = batch["component_target"].float()
    mask = (batch["component_target_mask"] > 0) & (output["component_mask"] > 0)
    if scaled:
        scales = batch["component_loss_scales"].float().view(-1, 1, 2, 1, 1)
        prediction, target = prediction / scales, target / scales
    results = [ztd_error_sums(prediction[:, :, j:j+1], target[:, :, j:j+1],
                              mask[:, :, j:j+1], sample_valid) for j in range(2)]
    return torch.stack([r[0] for r in results]), torch.stack([r[1] for r in results])


def component_maes(output, batch):
    sums, counts = component_error_sums(output, batch)
    return sums.sum(1) / counts.sum(1).clamp_min(1)


def component_aux_loss(output, batch, config):
    h, w = component_maes(output, batch)
    return float(config.get("zhd_loss_weight", .1))*h + float(config.get("zwd_loss_weight", .1))*w


def next_component_inputs(batch, output):
    pairs = [next_observations(batch["obs_components"][:, :, j:j+1],
                               batch["obs_component_mask"][:, :, j:j+1],
                               output["components"][:, :, j:j+1]) for j in range(2)]
    values = torch.cat([p[0] for p in pairs], 2)
    masks = torch.cat([p[1] for p in pairs], 2)
    ztd, mask = next_observations(batch["obs_ztd"], batch["obs_mask"], output["ztd"])
    # Predicted total is only eligible when BOTH components are available.
    mask[:, 1:] = masks[:, 1:].bool().all(2, keepdim=True).to(mask.dtype)
    ztd = torch.where(mask.bool(), ztd, 0)
    result = {"obs_ztd": ztd, "obs_mask": mask, "obs_components": values, "obs_component_mask": masks}
    result.update({k: batch[k] for k in COMPONENT_INPUTS[2:] if k in batch})
    return result
