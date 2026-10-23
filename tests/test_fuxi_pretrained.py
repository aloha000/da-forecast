import tempfile
import unittest
from pathlib import Path

import torch

from model.da_forecast_model import DAForecastNet
from model.fuxi_pretrained import load_fuxi_pretrained, rename_fuxi_key


def make_model():
    return DAForecastNet(
        image_size=(40, 40), era5_channels=4, era5_frames=2,
        era_static_channels=2, obs_static_channels=3, out_channels=4,
        embed_dim=32, patch_size=4, window_size=5, depths=(1, 1, 1, 1),
        num_heads=4,
    )


def original_key(target):
    if target.startswith("era_encoder.patch_embed."):
        return "decoder.patch_embed." + target[len("era_encoder.patch_embed."):]
    if target.startswith("era_encoder.down_block."):
        return "decoder.down_blocks.0." + target[len("era_encoder.down_block."):]
    if target.startswith("time_embed."):
        return "decoder.time_embed." + target[len("time_embed."):]
    if target.startswith("backbone.layers."):
        return "decoder.layers." + target[len("backbone.layers."):]
    if target.startswith("backbone.fpn."):
        return "decoder.fpn." + target[len("backbone.fpn."):]
    if target.startswith("backbone.norms."):
        tail = target[len("backbone.norms."):]
        return "decoder.norm" + tail[0] + tail[1:]
    if target.startswith("decoder.up_block."):
        return "decoder.up_blocks.0." + target[len("decoder.up_block."):]
    return target


class FuXiPretrainedTest(unittest.TestCase):
    def test_mapping_and_loading_preserve_new_branches(self):
        model = make_model()
        before = model.obs_encoder.temporal[0].weight.detach().clone()
        target = model.state_dict()
        core = {key: value for key, value in target.items()
                if not key.startswith(("obs_encoder.", "high_fusion.", "low_fusion."))}
        source = {original_key(key): torch.ones_like(value)
                  for key, value in core.items()}
        self.assertEqual(
            rename_fuxi_key("decoder.norm2.weight"), "backbone.norms.2.weight"
        )
        self.assertEqual(
            rename_fuxi_key("decoder.patch_embed.proj.weight"),
            "era_encoder.patch_embed.proj.weight",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fuxi.pth"
            torch.save({"model": source, "trainer": {}}, path)
            result = load_fuxi_pretrained(model, path)
            self.assertEqual(result["state_tensors"], len(core))
            for key, value in core.items():
                torch.testing.assert_close(model.state_dict()[key], torch.ones_like(value))
            torch.testing.assert_close(model.obs_encoder.temporal[0].weight, before)

    def test_missing_and_mismatched_core_are_rejected(self):
        model = make_model()
        core = {original_key(key): torch.ones_like(value)
                for key, value in model.state_dict().items()
                if not key.startswith(("obs_encoder.", "high_fusion.", "low_fusion."))}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fuxi.pth"
            removed = core.pop("decoder.head.weight")
            torch.save({"model": core}, path)
            with self.assertRaisesRegex(ValueError, "lacks"):
                load_fuxi_pretrained(model, path)
            core["decoder.head.weight"] = removed[:1]
            torch.save({"model": core}, path)
            with self.assertRaisesRegex(ValueError, "shape"):
                load_fuxi_pretrained(model, path)


if __name__ == "__main__":
    unittest.main()
