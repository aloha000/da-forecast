"""18 future 20-minute frames; no future observations enter recurrent inputs."""
import torch


def next_observations(obs_ztd, obs_mask, predicted_ztd):
    """[last input frame, 18 forecasts]; eligibility uses ONLY current history.

    The same helper applies at every 6h hop. Predictions are pseudo-observations,
    not new measurements. No detach here: callers choose offline/online gradients.
    """
    if obs_ztd.ndim != 5 or obs_ztd.shape[1:3] != (19, 1) or obs_mask.shape != obs_ztd.shape:
        raise ValueError("rollout needs 19 input frames [B,19,1,H,W]")
    if predicted_ztd.shape != (obs_ztd.shape[0], 18, 1, *obs_ztd.shape[-2:]):
        raise ValueError("ZTD prediction must be [B,18,1,H,W]")
    valid = (obs_mask > 0) & torch.isfinite(obs_ztd)
    available = valid.any(dim=1, keepdim=True)
    future_mask = available.expand_as(predicted_ztd)
    # Non-finite predictions are model errors, not missing observations to hide.
    if not torch.isfinite(predicted_ztd).all():
        raise ValueError("nonfinite predicted ZTD")
    last = torch.where(valid[:, -1:], obs_ztd[:, -1:], 0)
    future = torch.where(future_mask, predicted_ztd, 0)
    return (torch.cat((last, future), dim=1),
            torch.cat((valid[:, -1:], future_mask), dim=1).to(obs_mask.dtype))


def ztd_error_sums(prediction, target, mask, sample_valid=None):
    """Per-lead absolute-error sums and counts (not latitude weighted)."""
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("ZTD prediction/target/mask shape mismatch")
    valid = (mask > 0) & torch.isfinite(target)
    if sample_valid is not None:
        valid = valid & sample_valid.bool().view(-1, 1, 1, 1, 1)
    error = (prediction.float() - torch.where(valid, target, 0).float()).abs()
    error = torch.where(valid, error, 0)
    return error.sum((0, 2, 3, 4)), valid.sum((0, 2, 3, 4))


def advance_rollout_batch(batch, output, next_time_features):
    """Create only model inputs for the next hop; never carry truth/label fields."""
    obs, mask = next_observations(batch["obs_ztd"], batch["obs_mask"], output["ztd"])
    result = {"era5": torch.stack((batch["era5"][:, -1], output["weather"]), dim=1),
            "era_static": batch["era_static"], "time_features": next_time_features,
            "obs_ztd": obs, "obs_mask": mask, "obs_static": batch["obs_static"]}
    if "components" in output:
        from .delay_components import next_component_inputs
        result.update(next_component_inputs(batch, output))
    return result


def forecast_rollout(model, batch, time_features_by_step):
    """Yield any number of 6h joint forecasts with predicted weather/ZTD feedback.

    Caller supplies calendar-only features at T,T+6,... and controls inference_mode,
    autocast and whether graphs are retained. No labels are used by this iterator.
    """
    state = {k: batch[k] for k in ("era5", "era_static", "time_features", "obs_ztd", "obs_mask", "obs_static")}
    from .delay_components import COMPONENT_INPUTS
    state.update({k: batch[k] for k in COMPONENT_INPUTS if k in batch})
    previous = None
    for features in time_features_by_step:
        if previous is not None:
            state = advance_rollout_batch(state, previous, features)
        else:
            state["time_features"] = features
        previous = model(**state)
        if not isinstance(previous, dict) or "ztd" not in previous:
            raise ValueError("rollout requires a predict_ztd model")
        yield previous


def ztd_mae(prediction, batch):
    sums, counts = ztd_error_sums(prediction, batch["ztd_target"], batch["ztd_target_mask"])
    return sums.sum() / counts.sum().clamp_min(1)


def check_ztd_config(data, model):
    enabled = bool(model.get("predict_ztd", False))
    if enabled != bool(data.get("predict_ztd", False)):
        raise ValueError("data.predict_ztd must match model.predict_ztd")
    dual = bool(model.get("predict_delay_components", False))
    if dual != bool(data.get("predict_delay_components", False)):
        raise ValueError("data/model.predict_delay_components mismatch")
    if dual:
        if not enabled:
            raise ValueError("component mode requires predict_ztd=true for derived total")
        if list(model.get("component_loss_scales", [100., 100.])) != list(data.get("component_loss_scales", [100., 100.])):
            raise ValueError("data/model component loss scales mismatch")
        if bool(model.get("use_delay_physics", False)) != bool(data.get("use_delay_physics", False)):
            raise ValueError("data/model use_delay_physics mismatch")
    if enabled:
        if not model.get("use_obs", True) or model.get("zero_ztd", False):
            raise ValueError("predict_ztd requires active nonzero OBS branch")
        if data["obs_history_hours"] != 6 or data["obs_interval_minutes"] != 20:
            raise ValueError("predict_ztd requires 6h history and 20min spacing")
    if data.get("rollout_zarr"):
        policy = data.get("rollout_observations", "branch_removed")
        expected = "predicted_components_at_T" if dual else ("predicted_ztd_at_T" if enabled else "branch_removed")
        if policy != expected:
            raise ValueError("rollout_observations incompatible with predict_ztd")
        if not enabled and model.get("use_obs", True):
            raise ValueError("branch_removed rollout requires use_obs=false")
