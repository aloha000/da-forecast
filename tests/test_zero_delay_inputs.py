"""All-hop value ablation: unchanged modules, masks and supervision."""
import json
import unittest
from pathlib import Path

import torch
from model import DAForecastNet
from model.delay_rollout import rollout_predictions, rollout_loss
from tests.test_delay_components import tiny_model, component_batch
from tests.test_da_forecast_model import make_inputs
from train import initialize_model_weights
from train_rollout_delays import validate_config


class ZeroDelayInputsTest(unittest.TestCase):
    def test_config_only_changes_ablation_policy_and_output(self):
        root = Path(__file__).resolve().parents[1] / "configs"
        original = json.loads((root / "train_rollout_48h_predict_delays.json").read_text())
        zero = json.loads((root / "train_rollout_48h_predict_delays_zero_inputs.json").read_text())
        validate_config(original)
        validate_config(zero)
        self.assertTrue(zero["model"].pop("zero_delay_inputs"))
        zero["train"]["output_dir"] = original["train"]["output_dir"]
        zero["curriculum"]["observation_policy"] = "predicted_components"
        self.assertEqual(original, zero)

    def test_all_eight_hops_zero_values_preserve_masks_and_no_input_leak(self):
        model = tiny_model().eval()
        before = {k: v.shape for k, v in model.state_dict().items()}
        count = sum(p.numel() for p in model.parameters())
        model.zero_delay_inputs = True
        self.assertEqual(before, {k: v.shape for k, v in model.state_dict().items()})
        self.assertEqual(count, sum(p.numel() for p in model.parameters()))
        # Existing nonzero-input weights are compatible with the new experiment.
        initialize_model_weights(model, {"model": model.state_dict(), "config": {"model": {
            "predict_ztd": True, "predict_delay_components": True}}}, {
            "predict_ztd": True, "predict_delay_components": True, "zero_delay_inputs": True})
        with torch.no_grad():
            model.decoder.zhd_head.bias.fill_(.25)
            model.decoder.zwd_head.bias.fill_(.25)
        batch = make_inputs()
        batch.update(component_batch(40, 40))
        batch["rollout_time_features"] = torch.zeros(1, 8, 12)
        batch["obs_mask"][..., :4, :4] = 0
        batch["obs_component_mask"][..., :4, :4] = 0
        original = {k: v.clone() for k, v in batch.items()}
        seen = [0, 0]
        def check_obs(module, args):
            self.assertEqual(torch.count_nonzero(args[0]), 0)
            torch.testing.assert_close(args[1], original["obs_mask"])
            torch.testing.assert_close(args[2], original["obs_static"])
            seen[0] += 1
        def check_components(module, args):
            self.assertEqual(torch.count_nonzero(args[0]), 0)
            torch.testing.assert_close(args[1], original["obs_component_mask"])
            seen[1] += 1
        hooks = [model.obs_encoder.register_forward_pre_hook(check_obs),
                 model.component_adapter.register_forward_pre_hook(check_components)]
        with torch.no_grad():
            outputs = list(rollout_predictions(model, batch, 8))
            # Nonzero forecasts must be re-zeroed at the next hop. Prediction
            # base stays ZERO rather than real history or last predicted 25 mm.
            for output in outputs:
                torch.testing.assert_close(output["components"],
                    output["component_mask"].to(torch.float32) * 25.)
                self.assertTrue(output["component_mask"].any())
            for key in ("obs_ztd", "obs_components"):
                batch[key] = batch[key] + 543.21
            changed = list(rollout_predictions(model, batch, 8))
        for hook in hooks:
            hook.remove()
        self.assertEqual(seen, [16, 16])
        for a, b in zip(outputs, changed):
            for key in a:
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        for key in original:
            if key not in ("obs_ztd", "obs_components"):
                torch.testing.assert_close(batch[key], original[key], rtol=0, atol=0)

    def test_checkpointed_backward_keeps_true_component_supervision(self):
        model = tiny_model()
        model.zero_delay_inputs = True
        batch = make_inputs()
        batch.update(component_batch(40, 40))
        batch.update(rollout_time_features=torch.zeros(1, 2, 12),
                     targets=torch.ones(1, 2, 4, 40, 40),
                     target_masks=torch.ones(1, 2, 4, 40, 40),
                     component_targets=torch.full((1, 2, 18, 2, 40, 40), 100.),
                     component_target_masks=torch.ones(1, 2, 18, 2, 40, 40),
                     component_loss_scales=torch.tensor([[100., 100.]]))
        labels = batch["component_targets"].clone()
        loss, stats = rollout_loss(model, batch, 2, torch.linspace(-80, 80, 40),
                                   {"rollout_checkpoint": True})
        torch.testing.assert_close(stats[:, 2:], torch.full((2, 2), 100.))
        loss.backward()
        for module in (model.decoder.zhd_head, model.decoder.zwd_head,
                       model.era_encoder.patch_embed.proj):
            self.assertGreater(module.weight.grad.abs().sum(), 0)
        torch.testing.assert_close(batch["component_targets"], labels, rtol=0, atol=0)

    def test_flag_requires_component_model_and_policy_match(self):
        with self.assertRaisesRegex(ValueError, "zero_delay_inputs"):
            DAForecastNet(zero_delay_inputs=True)
        root = Path(__file__).resolve().parents[1] / "configs"
        cfg = json.loads((root / "train_rollout_48h_predict_delays.json").read_text())
        cfg["model"]["zero_delay_inputs"] = True
        with self.assertRaisesRegex(ValueError, "zero_all_steps"):
            validate_config(cfg)


if __name__ == "__main__":
    unittest.main()
