"""Joint ZTD-head, label timing, leakage, offline rollout and CLI regression tests."""
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import zarr

import build_rollout_zarr as builder
import evaluate
import train
from data.rollout_fields import open_store, write_field
from model import DAForecastNet
from model.ztd_rollout import next_observations, ztd_error_sums, ztd_mae, check_ztd_config, forecast_rollout
from tests import test_rollout_fields as fixtures
from tests import test_da_forecast_model as model_fixtures


def small_joint_model():
    return DAForecastNet(image_size=(40, 40), era5_channels=4, era_static_channels=2,
                         out_channels=4, embed_dim=32, patch_size=4, window_size=5,
                         depths=(1,), num_heads=4, predict_ztd=True)


class JointModelTest(unittest.TestCase):
    def test_old_weights_weather_parity_and_new_head_gradients(self):
        torch.manual_seed(1)
        old = model_fixtures.make_model().eval()
        new = small_joint_model().eval()
        report = train.initialize_model_weights(new, {
            "model": old.state_dict(), "config": {"model": {"use_obs": True}}},
            {"use_obs": True, "predict_ztd": True})
        self.assertEqual(report["discarded_obs_tensors"], 0)
        inputs = model_fixtures.make_inputs()
        inputs["obs_ztd"] = torch.randn(1, 19, 1, 40, 40)
        inputs["obs_mask"] = torch.ones_like(inputs["obs_ztd"])
        output = new(**inputs)
        torch.testing.assert_close(output["weather"], old(**inputs), rtol=0, atol=0)
        self.assertEqual(output["ztd"].shape, (1, 18, 1, 40, 40))
        train.apply_freeze_policy(new, {"freeze_fuxi": True, "freeze_obs": True})
        output = new(**inputs)
        output["ztd"].square().mean().backward()
        for name, parameter in new.named_parameters():
            self.assertEqual(parameter.requires_grad, name.startswith("decoder.ztd_head."))
        self.assertGreater(new.decoder.ztd_head.weight.grad.abs().sum().item(), 0)
        broken = dict(old.state_dict())
        del broken["decoder.head.weight"]
        with self.assertRaisesRegex(ValueError, "ONLY"):
            train.initialize_model_weights(new, {"model": broken, "config": {"model": {}}},
                                           {"predict_ztd": True})

    def test_recurrent_window_multiple_steps_and_gradients(self):
        obs = torch.arange(19.).view(1, 19, 1, 1, 1).expand(1, 19, 1, 2, 2).clone()
        mask = torch.zeros_like(obs)
        mask[:, :, :, 0, 0] = 1
        mask[:, -1, :, 0, 0] = 0  # endpoint missing, earlier history still eligible
        pred = torch.arange(100., 118.).view(1, 18, 1, 1, 1).expand(1, 18, 1, 2, 2).clone().requires_grad_()
        next_obs, next_mask = next_observations(obs, mask, pred)
        self.assertEqual(next_obs[0, 0, 0, 0, 0], 0)
        self.assertEqual(next_mask[0, 0, 0, 0, 0], 0)
        torch.testing.assert_close(next_obs[0, 1:, 0, 0, 0], torch.arange(100., 118.))
        self.assertTrue(next_mask[0, 1:, 0, 0, 0].all())
        self.assertFalse(next_mask[..., 1, 1].any())
        for hop in range(3):  # same window rule at 12/18/24h
            newer, newer_mask = next_observations(next_obs, next_mask, pred + hop + 1)
            torch.testing.assert_close(newer[:, :1], next_obs[:, -1:])
            next_obs, next_mask = newer, newer_mask
        next_obs.sum().backward()
        self.assertGreater(pred.grad[..., 0, 0].abs().sum(), 0)
        self.assertEqual(pred.grad[..., 1, 1].abs().sum(), 0)
        bad = pred.detach().clone()
        bad[0, 0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            next_observations(obs, mask, bad)

    def test_masked_mae_empty_labels_padding_and_units(self):
        pred = torch.ones(2, 18, 1, 2, 2, requires_grad=True)
        target = torch.full_like(pred, float("nan"))
        mask = torch.zeros_like(pred)
        target[0, :, :, 0, 0] = 3
        mask[0, :, :, 0, 0] = 1
        target[1] = 1000
        mask[1] = 1
        error, count = ztd_error_sums(pred, target, mask, torch.tensor([True, False]))
        torch.testing.assert_close(error, torch.full((18,), 2.))
        self.assertTrue(count.eq(1).all())
        empty = ztd_mae(pred, {"ztd_target": target, "ztd_target_mask": torch.zeros_like(mask)})
        self.assertEqual(empty.item(), 0)
        empty.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_invalid_config(self):
        with self.assertRaisesRegex(ValueError, "match"):
            check_ztd_config({}, {"predict_ztd": True})
        with self.assertRaisesRegex(ValueError, "nonzero"):
            check_ztd_config({"predict_ztd": True}, {"predict_ztd": True, "zero_ztd": True})

    def test_full_rollout_updates_weather_ztd_and_calendar_only(self):
        batch = {"era5": torch.zeros(1, 2, 1, 2, 2), "era_static": torch.zeros(1, 5, 2, 2),
                 "time_features": torch.zeros(1, 12), "obs_ztd": torch.ones(1, 19, 1, 2, 2),
                 "obs_mask": torch.ones(1, 19, 1, 2, 2), "obs_static": torch.zeros(1, 3, 2, 2),
                 "target": torch.full((1, 1, 2, 2), 999.), "ztd_target": torch.full((1, 18, 1, 2, 2), 999.)}
        seen = []
        def model(**inputs):
            self.assertNotIn("target", inputs)
            self.assertNotIn("ztd_target", inputs)
            seen.append(inputs)
            return {"weather": inputs["era5"][:, -1] + 2,
                    "ztd": inputs["obs_ztd"][:, -1:].expand(-1, 18, -1, -1, -1) + 3}
        features = [torch.full((1, 12), float(i)) for i in range(4)]
        out = list(forecast_rollout(model, batch, features))
        for i in range(4):
            self.assertTrue(out[i]["weather"].eq((i+1)*2).all())
            self.assertTrue(out[i]["ztd"].eq(1+(i+1)*3).all())
            torch.testing.assert_close(seen[i]["time_features"], features[i])


class TinyJoint(fixtures.TinyForecastCore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoder.ztd_head = torch.nn.Linear(1, 18)
        torch.nn.init.constant_(self.decoder.ztd_head.weight, .01)
        torch.nn.init.constant_(self.decoder.ztd_head.bias, 2.)

    def forward(self, era, static, features, obs, mask, obs_static):
        assert self.use_obs and obs.shape[1] == 19
        core = sum(getattr(self, name).weight.sum() for name in
                   ("era_encoder", "time_embed", "backbone", "decoder"))
        # Head predictions distinctly differ from truth7; future inputs detectable.
        v = self.decoder.ztd_head(obs[:, -1, 0, 0, 0, None]) + core
        ztd = v[:, :, None, None, None].expand(-1, -1, 1, *era.shape[-2:])
        return {"weather": era[:, -1] + core, "ztd": ztd}


class JointDataTest(unittest.TestCase):
    setUp = fixtures.TwoFieldTest.setUp

    def config(self):
        self.data.update(predict_ztd=True, rollout_observations="predicted_ztd_at_T")
        return {"data": self.data,
                "model": {**self.model_config, "predict_ztd": True},
                "train": {"seed": 4, "epochs": 1, "batch_size_per_gpu": 2,
                          "gradient_accumulation": 1, "num_workers": 0,
                          "learning_rate": .001, "min_learning_rate": .0001,
                          "weight_decay": 0., "warmup_steps": 0, "gradient_clip": 1.,
                          "amp": "fp32", "freeze_fuxi": False, "freeze_obs": True,
                          "loss_mode": "global_only", "halo_weight": .5, "halo_radius": 0,
                          "ztd_loss_weight": .1, "checkpoint_metric": "weather_global",
                          "validate_before_training": True, "fsdp": {"enabled": False},
                          "output_dir": str(self.root / "joint_train")}}

    def build_joint(self):
        c = self.config()
        torch.save({"model": TinyJoint(**c["model"]).state_dict(),
                    "config": {"model": c["model"], "data": {"forecast_lead_hours": 6}}},
                   self.data["rollout_source_checkpoint"])
        path = self.root / "joint.json"
        path.write_text(json.dumps(c))
        argv = ["build_rollout_zarr.py", "--config", str(path), "--amp", "fp32", "--num-workers", "0"]
        with patch.object(builder, "DAForecastNet", TinyJoint), patch.object(
                builder, "setup_distributed", return_value=(0, 0, 1, torch.device("cpu"))), patch("sys.argv", argv), redirect_stdout(io.StringIO()):
            builder.main()
        return c, path

    def test_future_labels_exact_offsets_and_missing_times(self):
        ds = self.sources["train"]
        group = zarr.open_group(self.data["ztd_path"], mode="r+")
        values = np.full(group["ztd"].shape, np.nan, np.float32)
        values[:, 0, 0] = np.arange(len(values))
        group["ztd"][:] = values
        timestamp = ds.sample_time(0)
        target, mask = ds.read_future_ztd(timestamp)
        expected = ds.obs_times.get_indexer(timestamp + pd.to_timedelta(np.arange(1, 19)*20, unit="min"))
        np.testing.assert_array_equal(target[:, 0, 0, 0], expected)
        self.assertTrue(mask[:, 0, 0, 0].all())
        self.assertFalse(mask[..., 1, 1].any())
        target, mask = ds.read_future_ztd(ds.obs_times[-1])
        self.assertFalse(mask.any())
        self.assertFalse(target.any())
        # Weather sample selection doesn't depend on future ZTD completeness.
        data6 = {k: v for k, v in self.data.items() if k != "rollout_zarr"}
        data6.update(forecast_lead_hours=6, predict_ztd=True)
        joint = train.make_dataset(data6, "train")
        plain = train.make_dataset({**data6, "predict_ztd": False}, "train")
        self.assertEqual(len(joint), len(plain))
        self.assertIn("ztd_target", joint[0])

    def test_generator_sparse_roundtrip_and_transaction(self):
        c, path = self.build_joint()
        with patch("train.DAForecastDataset", side_effect=AssertionError("source data opened")):
            ds = train.make_dataset(self.data, "train")
            sample = ds[0]
        self.assertEqual(sample["obs_ztd"].shape, (19, 1, 2, 2))
        self.assertEqual(sample["obs_ztd"][0, 0, 0, 0], 7)
        self.assertTrue(torch.allclose(sample["obs_ztd"][1:, 0, 0, 0], torch.full((18,), 2.47)))
        self.assertTrue(sample["ztd_target"][:, 0, 0, 0].eq(7).all())
        self.assertFalse(sample["obs_mask"][..., 1, 1].any())
        root = open_store(self.data["rollout_zarr"], "r+")
        self.assertEqual(root["ztd_input"].shape, (18, 19, 1))
        np.testing.assert_array_equal(ds.obs_static, self.sources["train"].obs_static)
        # Corrupt/incomplete extras must not leave a row marked ready.
        with self.assertRaisesRegex(ValueError, "requires ZTD"):
            write_field(root, "z_fuxi6h", 0, np.zeros((70, 2, 2), np.float32))
        self.assertFalse(root["forecast_complete"][0])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            train.make_dataset(self.data, "train")
        # Resume fills the joint row again.
        argv = ["build_rollout_zarr.py", "--config", str(path), "--amp", "fp32", "--num-workers", "0"]
        with patch.object(builder, "DAForecastNet", TinyJoint), patch.object(
                builder, "setup_distributed", return_value=(0, 0, 1, torch.device("cpu"))), patch("sys.argv", argv), redirect_stdout(io.StringIO()):
            builder.main()
        self.assertTrue(root["forecast_complete"][0])

    def test_future_truth_cannot_change_rollout_inputs(self):
        c = self.config()
        ds = self.sources["train"]
        before = ds[0]
        model = TinyJoint(**c["model"])
        def predict(batch):
            b = {k: v.unsqueeze(0) for k, v in batch.items()}
            out = train.forward_model(model, b)
            return next_observations(b["obs_ztd"], b["obs_mask"], out["ztd"])
        original = predict(before)
        group = zarr.open_group(self.data["ztd_path"], mode="r+")
        cut = int(ds.obs_times.searchsorted(ds.sample_time(0) + pd.Timedelta(minutes=5)))
        group["ztd"][cut:] = np.nan
        changed = predict(ds[0])
        torch.testing.assert_close(original[0], changed[0], rtol=0, atol=0)
        torch.testing.assert_close(original[1], changed[1], rtol=0, atol=0)
        self.assertFalse(ds.read_future_ztd(ds.sample_time(0))[1].any())

    def test_train_evaluate_and_resume_joint_cli(self):
        c, path = self.build_joint()
        source_state = torch.load(self.data["rollout_source_checkpoint"], weights_only=False)["model"]
        argv = ["train.py", "--config", str(path), "--init-checkpoint", self.data["rollout_source_checkpoint"]]
        with patch.object(train, "DAForecastNet", TinyJoint), patch.object(
                train, "setup_distributed", return_value=(0, 0, 1, torch.device("cpu"))), patch("sys.argv", argv), redirect_stdout(io.StringIO()):
            train.main()
        out = Path(c["train"]["output_dir"])
        saved = torch.load(out / "best.pt", weights_only=False)
        self.assertFalse(saved["config"]["initialization"]["fuxi_only"])
        for key, value in source_state.items():
            if key.startswith(("obs_encoder.", "high_fusion.", "low_fusion.")):
                torch.testing.assert_close(saved["model"][key], value, rtol=0, atol=0)
        self.assertFalse(torch.equal(saved["model"]["decoder.ztd_head.weight"], source_state["decoder.ztd_head.weight"]))
        reports = []
        for bs in (1, 3):
            dest = out / f"test{bs}"
            argv = ["evaluate.py", "--config", str(path), "--checkpoint", str(out / "best.pt"),
                    "--output-dir", str(dest), "--batch-size", str(bs), "--num-workers", "0",
                    "--device", "cpu", "--skip-animations", "--skip-mae-plots"]
            with patch.object(evaluate, "DAForecastNet", TinyJoint), patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                evaluate.main()
            reports.append(json.loads((dest / "ztd_mae_summary.json").read_text()))
            weather = json.loads((dest / "weighted_mae_summary.json").read_text())
            self.assertEqual(weather["samples"], 6)
        self.assertEqual(reports[0]["valid_count"], 6*18)
        self.assertEqual(reports[0]["lead_minutes_from_original_T"], list(range(380, 721, 20)))
        self.assertAlmostEqual(reports[0]["mae_normalized"], reports[1]["mae_normalized"], places=5)
        c["train"]["epochs"] = 2
        path.write_text(json.dumps(c))
        argv = ["train.py", "--config", str(path), "--resume", str(out / "last.pt")]
        with patch.object(train, "DAForecastNet", TinyJoint), patch.object(
                train, "setup_distributed", return_value=(0, 0, 1, torch.device("cpu"))), patch("sys.argv", argv), redirect_stdout(io.StringIO()):
            train.main()
        self.assertEqual(torch.load(out / "last.pt", weights_only=False)["epoch"], 1)

    def test_ztd_physical_metrics(self):
        class DS:
            ztd_normalization = {"ztd_train_std": 12., "ztd_train_mean": 2000., "raw_units": "mm"}
        error, count = np.full(18, 3.), np.full(18, 2.)
        report = evaluate.write_ztd_metrics(self.root, {"ztd_absolute_error": error, "ztd_valid_count": count}, DS(), 6, {})
        self.assertEqual(report["mae_normalized"], 1.5)
        self.assertEqual(report["mae_mm"], 18.)
        self.assertEqual(report["lead_minutes_from_original_T"], list(range(20, 361, 20)))


if __name__ == "__main__":
    unittest.main()
