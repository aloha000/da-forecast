import io
import unittest

import torch

from model.da_forecast_model import DAForecastNet, ERA5Encoder, GatedFusion
from model.fuxi import PeriodicConv2d


def make_model(use_obs=True):
    return DAForecastNet(
        image_size=(40, 40),
        era5_channels=4,
        era5_frames=2,
        era_static_channels=2,
        obs_static_channels=3,
        out_channels=4,
        embed_dim=32,
        patch_size=4,
        window_size=5,
        depths=(1,),
        num_heads=4,
        use_obs=use_obs,
    )


def make_inputs(batch=1):
    return {
        "era5": torch.randn(batch, 2, 4, 40, 40),
        "era_static": torch.randn(batch, 2, 40, 40),
        "time_features": torch.randn(batch, 12),
        "obs_ztd": torch.randn(batch, 3, 1, 40, 40),
        "obs_mask": torch.ones(batch, 3, 1, 40, 40),
        "obs_static": torch.randn(batch, 3, 40, 40),
    }


class DAForecastModelTest(unittest.TestCase):
    def test_forward_shape_finite_and_gradients(self):
        torch.manual_seed(0)
        model = make_model(use_obs=True)
        output = model(**make_inputs())
        self.assertEqual(output.shape, (1, 4, 40, 40))
        self.assertTrue(torch.isfinite(output).all())

        output.square().mean().backward()
        groups = {
            "era_encoder": model.era_encoder,
            "time_embed": model.time_embed,
            "obs_encoder": model.obs_encoder,
            "high_fusion": model.high_fusion,
            "backbone": model.backbone,
            "decoder": model.decoder,
        }
        for name, module in groups.items():
            gradients = [p.grad for p in module.parameters() if p.requires_grad]
            self.assertTrue(any(g is not None for g in gradients), name)
            self.assertTrue(
                all(torch.isfinite(g).all() for g in gradients if g is not None), name
            )
            self.assertTrue(
                any(g.abs().sum() > 0 for g in gradients if g is not None), name
            )

    def test_era5_only_forward_and_backward(self):
        model = make_model(use_obs=False)
        inputs = make_inputs()
        output = model(
            inputs["era5"], inputs["era_static"], inputs["time_features"]
        )
        self.assertEqual(output.shape, (1, 4, 40, 40))
        output.mean().backward()
        self.assertIsNone(model.obs_encoder)
        self.assertIsNotNone(model.era_encoder.patch_embed.proj.weight.grad)

    def test_all_missing_observations_are_finite_and_neutral(self):
        torch.manual_seed(1)
        model = make_model(use_obs=True).eval()
        inputs = make_inputs()
        inputs["obs_mask"].zero_()
        inputs["obs_ztd"][..., 0, 0] = float("nan")
        first = model(**inputs)
        inputs["obs_ztd"].normal_(mean=1000.0, std=100.0)
        second = model(**inputs)
        self.assertTrue(torch.isfinite(first).all())
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_gate_starts_near_two_percent(self):
        fusion = GatedFusion(embed_dim=32)
        expected = torch.tensor(0.01798621)
        torch.testing.assert_close(
            fusion.gate.mean(), expected, rtol=1e-5, atol=1e-7
        )

    def test_global_decoder_uses_periodic_convolution(self):
        model = make_model(use_obs=True)
        self.assertTrue(
            any(isinstance(module, PeriodicConv2d) for module in model.modules())
        )

        # A west-edge impulse must not wrap through the patch convolution.
        # (The later GroupNorm intentionally uses whole-domain statistics, so
        # testing the fully normalized feature map would not be a locality test.)
        encoder = ERA5Encoder(1, 1, 1, embed_dim=32, patch_size=4).eval()
        fields = torch.zeros(1, 2, 40, 40)
        baseline = encoder.patch_embed.proj(fields)
        fields[..., 20, 0] = 1.0
        impulse = encoder.patch_embed.proj(fields)
        torch.testing.assert_close(impulse[..., -1], baseline[..., -1])

    def test_fuxi_backbone_and_decoder_structure(self):
        model = make_model(use_obs=True)
        self.assertEqual(len(model.backbone.layers), 1)
        self.assertIsInstance(model.backbone.fpn[0], torch.nn.Linear)
        self.assertIsInstance(model.era_encoder.patch_embed, torch.nn.Module)
        self.assertIsInstance(model.era_encoder.down_block, torch.nn.Module)
        self.assertIsInstance(model.decoder.up_block, torch.nn.Module)
        self.assertIsInstance(model.decoder.head, torch.nn.Linear)
        self.assertEqual(model.backbone.layers[0].blocks[0].mask_type, "h")

    def test_invalid_window_fails_early(self):
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            DAForecastNet(
                image_size=(40, 40),
                era5_channels=4,
                era_static_channels=2,
                obs_static_channels=3,
                out_channels=4,
                embed_dim=32,
                window_size=3,
                depths=(1,),
                num_heads=4,
            )

    def test_state_dict_round_trip_is_exact(self):
        torch.manual_seed(2)
        model = make_model(use_obs=True).eval()
        inputs = make_inputs()
        expected = model(**inputs)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)

        restored = make_model(use_obs=True).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        actual = restored(**inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
