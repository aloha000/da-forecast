"""No measured origin values; fixed means, masks and differentiable feedback."""
import json
from pathlib import Path
import unittest
import numpy as np
import torch
import zarr
from data.delay_sequence import make_rollout_dataset
from data.rollout_identity import data_identity
from model.delay_components import COMPONENT_INPUTS, component_prediction
from model.delay_rollout import INPUT_KEYS, rollout_predictions
from train_rollout_delays import validate_config
from tests import test_delay_curriculum as dc


class MeanOriginConfigTest(unittest.TestCase):
    def test_same_model_initialization_hyperparameters_and_conflict_guard(self):
        root = Path(__file__).resolve().parents[1] / "configs"
        original = json.loads((root / "train_rollout_48h_predict_delays.json").read_text())
        mean = json.loads((root / "train_rollout_48h_predict_delays_mean_origin.json").read_text())
        validate_config(mean)
        mean["model"]["zero_delay_inputs"] = True
        with self.assertRaisesRegex(ValueError, "retain predicted feedback"):
            validate_config(mean)
        del mean["model"]["zero_delay_inputs"]
        self.assertTrue(mean["data"].pop("mean_origin_inputs"))
        mean["curriculum"]["observation_policy"] = "predicted_components"
        mean["train"]["output_dir"] = original["train"]["output_dir"]
        self.assertEqual(mean, original)


class MeanOriginPipelineTest(dc.DelayCurriculumPipelineTest):
    # Inherit the end-to-end training, bit-exact resume, evaluation batch-size,
    # future-truth leakage and distributed-padding tests under the NEW policy.
    def setUp(self):
        super().setUp()
        self.data["mean_origin_inputs"] = True
        self.config["curriculum"]["observation_policy"] = "mean_origin_predicted_components"
        self.path.write_text(json.dumps(self.config))

    def test_means_masks_labels_all_splits_and_real_values_cannot_leak(self):
        real_data = {**self.data, "mean_origin_inputs": False}
        self.assertNotEqual(data_identity(real_data), data_identity(self.data))
        for split in ("train", "val", "test"):
            ds = make_rollout_dataset(self.data, split, 8, 8)
            real = make_rollout_dataset(real_data, split, 8, 8)
            self.assertTrue(ds.analysis_times.equals(real.analysis_times))
            a, b = ds[0], real[0]
            self.assertEqual(torch.count_nonzero(a["obs_ztd"]), 0)
            means = torch.tensor([1900., 100.]).view(1, 2, 1, 1)
            torch.testing.assert_close(a["obs_components"],
                                      torch.where(a["obs_component_mask"], means, 0.))
            for key in a:
                if key not in ("obs_ztd", "obs_components"):
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        ds = make_rollout_dataset(self.data, "train", 8, 8)
        before = ds[0]
        group = zarr.open_group(self.data["ztd_path"], mode="r+")
        # Keep missingness and training statistics identical, alter values only.
        for name in ("ztd", "zwd"):
            group[name][:] = np.asarray(group[name][:]) + 5.
        after = ds[0]
        for key in INPUT_KEYS + COMPONENT_INPUTS + ("rollout_time_features",):
            if key in before:
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        self.assertFalse(torch.equal(before["component_targets"], after["component_targets"]))
        self.assertTrue(after["component_target_masks"].any())
        # Identical input policy is enforced by resume AND evaluation identities.
        self.assertNotEqual(data_identity(self.data), data_identity(real_data))

    def test_eight_hop_mean_baseline_prediction_feedback_and_gradients(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.tensor(.1))
                self.d = torch.nn.Parameter(torch.tensor(.2))
            def forward(self, era5, obs_components, obs_component_mask, ztd_stats, **kwargs):
                b, _, _, h, w = obs_components.shape
                return {"weather": era5[:, -1] + self.w + obs_components[:, -1, 1:2] * .01,
                        **component_prediction(self.d.expand(b, 18, 2, h, w),
                            obs_components, obs_component_mask, [1, 1], ztd_stats)}
        sample = make_rollout_dataset(self.data, "train", 8, 8)[0]
        batch = {k: v[None] for k, v in sample.items()}
        values, grads = [], []
        # Include only the fixture's occupied (0,0) station for exact derivatives.
        for checkpointed in (False, True):
            model = Toy()
            outputs = list(rollout_predictions(model, batch, 8, checkpointed))
            for j, output in enumerate(outputs):
                self.assertAlmostEqual(output["components"][0, -1, 1, 0, 0].item(),
                                       100. + (j + 1) * .2, places=3)
                self.assertGreater(output["ztd"][0, -1, 0, 0, 0].item(), 0.)
            outputs[-1]["weather"][0, :, 0, 0].mean().backward()
            self.assertAlmostEqual(model.w.grad.item(), 8., places=4)
            self.assertAlmostEqual(model.d.grad.item(), .28, places=4)
            values.append(outputs[-1]["weather"].detach())
            grads.append(torch.stack((model.w.grad, model.d.grad)))
        torch.testing.assert_close(*values, rtol=0, atol=0)
        torch.testing.assert_close(*grads, rtol=0, atol=0)

    def test_cannot_resume_mean_checkpoint_with_real_origin(self):
        self.run_train("--stop-after-updates", "1")
        self.data["mean_origin_inputs"] = False
        self.config["curriculum"]["observation_policy"] = "predicted_components"
        self.path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.run_train("--resume", str(self.root / "run/last.pt"))


if __name__ == "__main__":
    unittest.main()
