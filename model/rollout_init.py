"""Strict extraction of the FuXi core from a DA checkpoint for no-OBS rollout."""
OBS_PREFIXES = ("obs_encoder.", "high_fusion.", "low_fusion.")


def load_fuxi_core_from_da(model, state):
    if model.use_obs:
        raise ValueError("FuXi-core-only initialization requires model.use_obs=false")
    # Only these known branch keys may be dropped. All core keys/shapes stay strict.
    core = {name: value for name, value in state.items() if not name.startswith(OBS_PREFIXES)}
    model.load_state_dict(core, strict=True)
    return {"loaded_tensors": len(core), "discarded_obs_tensors": len(state) - len(core)}
