"""Shared weights, predicted weather AND 18 delay frames fed back without detach."""
import torch
from torch.utils.checkpoint import checkpoint
from .delay_components import COMPONENT_INPUTS, component_maes
from .ztd_rollout import advance_rollout_batch

INPUT_KEYS = ("era5", "era_static", "time_features", "obs_ztd", "obs_mask", "obs_static")


def rollout_predictions(model, batch, horizon, checkpoint_steps=False):
    if not 1 <= horizon <= batch["rollout_time_features"].shape[1]:
        raise ValueError("missing calendar features")
    state = {k:batch[k] for k in INPUT_KEYS + COMPONENT_INPUTS if k in batch}
    keys = tuple(state)  # Stable keys; tensor values explicitly passed on every hop.
    def run(*values):
        return model(**dict(zip(keys, values)))
    for j in range(horizon):
        state["time_features"] = batch["rollout_time_features"][:, j]
        if checkpoint_steps and torch.is_grad_enabled():
            output = checkpoint(run, *(state[k] for k in keys), use_reentrant=False)
        else:
            output = run(*(state[k] for k in keys))
        if not isinstance(output, dict) or not all(k in output for k in
                ("weather", "components", "component_mask", "ztd")):
            raise ValueError("rollout needs joint weather/ZHD/ZWD outputs")
        yield output
        if j+1 < horizon:
            state = advance_rollout_batch(state, output, batch["rollout_time_features"][:, j+1])


def lead_labels(batch, j):
    return {"component_target": batch["component_targets"][:, j],
            "component_target_mask": batch["component_target_masks"][:, j],
            "component_loss_scales": batch["component_loss_scales"]}


def rollout_loss(model, batch, horizon, lat, config):
    from train import weighted_losses
    total = 0.
    stats = []
    for j, output in enumerate(rollout_predictions(model, batch, horizon,
                                  config.get("rollout_checkpoint", True))):
        weather, _, _ = weighted_losses(output["weather"].float(), batch["targets"][:, j],
                batch["target_masks"][:, j], batch["obs_static"], lat,
                config.get("halo_weight", .5), config.get("halo_radius", 3),
                config.get("loss_mode", "global_only"))
        h, w = component_maes(output, lead_labels(batch, j))
        loss = weather + config.get("zhd_loss_weight", .1)*h + config.get("zwd_loss_weight", .1)*w
        total = total + loss / horizon
        scales = batch["component_loss_scales"][0]
        stats.append(torch.stack((loss, weather, h*scales[0], w*scales[1])).detach())
    # Columns: joint loss, weather normalized MAE, ZHD mm MAE, ZWD mm MAE.
    return total, torch.stack(stats)
