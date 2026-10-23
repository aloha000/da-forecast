"""Load the forecast core from a trusted original FuXi checkpoint.

The upstream checkpoint also pickles OmegaConf trainer metadata.  Only its
``model`` state is used here; an inert placeholder lets us read the file in
environments without OmegaConf.  As with any pickle checkpoint, only load a
file from a trusted source.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from types import ModuleType

import torch
from torch import nn


class _IgnoredOmegaConf:
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __setstate__(self, state):
        pass


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("omegaconf."):
            return _IgnoredOmegaConf
        return super().find_class(module, name)


_PICKLE_COMPAT = ModuleType("fuxi_pickle_compat")
_PICKLE_COMPAT.Unpickler = _CheckpointUnpickler


def rename_fuxi_key(key: str) -> str | None:
    """Map every original FuXi forecast parameter into the DA model."""
    for source, target in (
        ("decoder.patch_embed.", "era_encoder.patch_embed."),
        ("decoder.down_blocks.0.", "era_encoder.down_block."),
        ("decoder.layers.", "backbone.layers."),
        ("decoder.fpn.", "backbone.fpn."),
        ("decoder.up_blocks.0.", "decoder.up_block."),
        ("decoder.time_embed.", "time_embed."),
    ):
        if key.startswith(source):
            return target + key[len(source):]
    for index in range(4):
        source = f"decoder.norm{index}."
        if key.startswith(source):
            return f"backbone.norms.{index}." + key[len(source):]
    if key.startswith("decoder.head."):
        return key
    return None


def load_fuxi_pretrained(model: nn.Module, path: str | Path) -> dict[str, int]:
    """Initialize the complete FuXi path, leaving only OBS and gates random.

    All target core tensors must be present and shape-compatible.  The loader
    is deliberately strict about the transferable subnetwork, while ignoring
    incompatible FuXi patch/downsampling layers and checkpoint metadata.
    ``mmap`` avoids materializing the entire multi-gigabyte checkpoint in RAM.
    """
    path = Path(path)
    checkpoint = torch.load(
        path, map_location="cpu", mmap=True, weights_only=False,
        pickle_module=_PICKLE_COMPAT,
    )
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError(f"{path} is not a FuXi checkpoint with a model state")
    source_state = checkpoint["model"]
    target_state = model.state_dict()
    new_prefixes = ("obs_encoder.", "high_fusion.", "low_fusion.")
    expected = {key for key in target_state if not key.startswith(new_prefixes)}
    renamed = {}
    for source_key, tensor in source_state.items():
        target_key = rename_fuxi_key(source_key)
        if target_key is None:
            continue
        if target_key not in expected:
            raise ValueError(f"mapped FuXi key {source_key!r} is absent from model")
        if target_key in renamed:
            raise ValueError(f"duplicate FuXi mapping to {target_key!r}")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"FuXi state {source_key!r} is not a tensor")
        if tensor.shape != target_state[target_key].shape:
            raise ValueError(
                f"FuXi {source_key!r} -> {target_key!r}: shape {tuple(tensor.shape)} "
                f"does not match {tuple(target_state[target_key].shape)}"
            )
        renamed[target_key] = tensor
    missing = sorted(expected - renamed.keys())
    if missing:
        raise ValueError(
            f"FuXi checkpoint lacks {len(missing)} forecast-core tensors; "
            f"first missing keys: {missing[:8]}"
        )
    result = model.load_state_dict(renamed, strict=False)
    if result.unexpected_keys or set(result.missing_keys) != set(target_state) - expected:
        raise RuntimeError(f"unexpected model load result: {result}")
    parameters = dict(model.named_parameters())
    loaded_parameters = sum(parameters[key].numel() for key in renamed if key in parameters)
    total_parameters = sum(parameter.numel() for parameter in parameters.values())
    return {
        "state_tensors": len(renamed),
        "parameters": loaded_parameters,
        "total_parameters": total_parameters,
    }


__all__ = ["load_fuxi_pretrained", "rename_fuxi_key"]
