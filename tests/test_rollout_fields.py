"""Regression tests for real-OBS +6h generation and no-OBS +12h training."""
import json
import csv
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import zarr

import build_rollout_zarr as builder
import train
import evaluate
from data.da_dataset import time_features
from data.rollout_identity import data_identity, file_identity
from data.rollout_fields import (
    HOUR_NS, axes_for_datasets, initialize_store, open_store, times_ns, write_field,
)
from model import DAForecastNet
from model.rollout_init import load_fuxi_core_from_da
from train import apply_freeze_policy, make_dataset


def array(group, name, value):
    create = group.create_array if hasattr(group, "create_array") else group.create_dataset
    return create(name, data=np.asarray(value))


class TinyObservedSource(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        assert kwargs["use_obs"] and not kwargs["zero_ztd"]
        self.delta = torch.nn.Parameter(torch.tensor(2.))

    def forward(self, era, static, time, obs, mask, obs_static):
        assert obs[:, :, :, 0, 0].eq(7).all()
        assert mask[:, :, :, 0, 0].eq(1).all()
        # Predictions must depend on observations, not just presence of the branch.
        return era[:, -1] + self.delta + obs[:, 0, 0, 0, 0, None, None, None]


class TinyForecastCore(torch.nn.Module):
    """Exercise actual train/evaluate entry points without allocating FuXi."""
    def __init__(self, use_obs=False, **kwargs):
        super().__init__()
        self.use_obs = use_obs
        for name in ("era_encoder", "time_embed", "backbone", "decoder"):
            layer = torch.nn.Linear(1, 1, bias=False)
            torch.nn.init.constant_(layer.weight, .1)
            setattr(self, name, layer)
        for name in ("obs_encoder", "high_fusion", "low_fusion"):
            setattr(self, name, torch.nn.Linear(1, 1) if use_obs else None)

    def forward(self, era, static, features, obs, mask, obs_static):
        assert not self.use_obs and not obs.any() and not mask.any()
        correction = sum(getattr(self, name).weight.sum()
                         for name in ("era_encoder", "time_embed", "backbone", "decoder"))
        return era[:, -1] + correction


class TinyZeroZtdSource(TinyObservedSource):
    def __init__(self, **kwargs):
        assert kwargs["use_obs"] and kwargs["zero_ztd"]
        super().__init__(**{**kwargs, "zero_ztd": False})

    def forward(self, era, static, time, obs, mask, obs_static):
        # Like the real model, zero values INSIDE forward, retaining availability.
        mask = (mask > 0) & torch.isfinite(obs)
        obs = torch.zeros_like(obs)
        assert mask[:, :, :, 0, 0].all() and not obs.any()
        return era[:, -1] + self.delta + mask[:, 0, 0, 0, 0, None, None, None]


class TwoFieldTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        era = zarr.open_group(str(self.root / "era.zarr"), mode="w")
        obs = zarr.open_group(str(self.root / "obs.zarr"), mode="w")
        lat, lon = np.array([45., 44.75]), np.array([10., 10.25])
        for group in (era, obs):
            array(group, "lat", lat)
            array(group, "lon", lon)
        t = array(era, "time", np.arange(30) * 6)
        t.attrs["units"] = "hours since 2024-12-31 18:00:00"
        self.truth = np.broadcast_to(np.arange(30, dtype=np.float32)[:, None, None, None],
                                     (30, 70, 2, 2)).copy()
        self.truth[3, 0, 0, 1] = np.nan
        array(era, "z", self.truth)
        t = array(obs, "time", np.arange(2305) * 5)
        t.attrs["units"] = "minutes since 2024-12-31 12:00:00"
        array(obs, "mask", np.array([[False, True], [True, True]]))
        array(obs, "station_lat", np.full((2, 2), lat[0]))
        array(obs, "station_lon", np.full((2, 2), lon[0]))
        ztd = np.full((2305, 2, 2), np.nan, np.float32)
        ztd[:, 0, 0] = 7
        array(obs, "ztd", ztd)
        for name, value in (
            ("lat", lat), ("lon", lon), ("conds", np.zeros((5, 2, 2), np.float32)),
            ("mean_era5", np.zeros(70)), ("std_era5", np.ones(70)),
        ):
            np.save(self.root / (name + ".npy"), value)
        self.data = {
            "era5_path": str(self.root / "era.zarr"), "ztd_path": str(self.root / "obs.zarr"),
            "conds_path": str(self.root / "conds.npy"), "global_lat_path": str(self.root / "lat.npy"),
            "global_lon_path": str(self.root / "lon.npy"), "mean_std_dir": str(self.root),
            "obs_history_hours": 6, "obs_interval_minutes": 20, "forecast_lead_hours": 12,
            "rollout_zarr": str(self.root / "rollout.zarr"),
            "rollout_source_checkpoint": str(self.root / "source.pt"),
            "train_split": ["2025-01-01", "2025-01-03"],
            "val_split": ["2025-01-03", "2025-01-05"],
            "test_split": ["2025-01-05", "2025-01-07"],
        }
        self.model_config = {"image_size": [2, 2], "era5_channels": 70,
                             "use_obs": True, "zero_ztd": False}
        torch.save({"model": TinyObservedSource(**self.model_config).state_dict(),
                    "config": {"model": self.model_config, "data": {"forecast_lead_hours": 6},
                               "train": {"amp": "fp32"}}}, self.data["rollout_source_checkpoint"])
        self.sources = builder.source_datasets(self.data)
        self.identity = {"source_checkpoint": file_identity(self.data["rollout_source_checkpoint"]),
                         "data": data_identity(self.data), "amp": "fp32"}

    def initialize(self):
        return initialize_store(self.data["rollout_zarr"], self.identity, self.sources, self.model_config)

    def fill(self, group):
        times, origins, _ = axes_for_datasets(self.sources)
        for i, t in enumerate(times):
            write_field(group, "z", i, self.truth[self.sources["train"].label_lookup[int(t)]])
        for i in range(len(origins)):
            write_field(group, "z_fuxi6h", i, np.full((70, 2, 2), 9, np.float32))

    def test_ns_precision_and_deduplication(self):
        times, origins, _ = axes_for_datasets(self.sources)
        self.assertEqual(origins[0], pd.Timestamp("2025-01-01").value)
        self.assertEqual(len(origins), 18)
        self.assertEqual(len(times), 24)  # overlapping current/target rows stored once
        for ds in self.sources.values():
            ds.analysis_times = pd.DatetimeIndex(ds.analysis_times.to_numpy(dtype="datetime64[us]"))
        np.testing.assert_array_equal(axes_for_datasets(self.sources)[0], times)
        np.testing.assert_array_equal(times_ns(self.sources["train"].analysis_times)[0], origins[0])

    def test_four_dimensions_time_target_mask_and_no_source_reads(self):
        group = self.initialize()
        self.fill(group)
        self.assertEqual(group["z"].ndim, 4)
        self.assertEqual(group["z_fuxi6h"].ndim, 4)
        self.assertNotIn("input", group)
        # make_dataset must dispatch without touching source ERA5/ZTD.
        with patch("train.DAForecastDataset", side_effect=AssertionError("source opened")):
            ds = make_dataset(self.data, "train")
            sample = ds[0]
        self.assertTrue(torch.all(sample["era5"][0] == 1))
        self.assertTrue(torch.all(sample["era5"][1] == 9))
        self.assertEqual(sample["target"][0, 0, 0], 3)
        self.assertEqual(sample["target"][0, 0, 1], 0)
        self.assertEqual(sample["target_mask"][0, 0, 1], 0)
        self.assertTrue(torch.equal(sample["persistence"], sample["era5"][0]))
        self.assertFalse(sample["obs_ztd"].any())
        self.assertFalse(sample["obs_mask"].any())
        np.testing.assert_array_equal(sample["time_features"], time_features(pd.Timestamp("2025-01-01 06:00")))
        self.assertTrue(np.isnan(group["z"][2, 0, 0, 1]))  # original missing truth preserved

    def test_incomplete_identity_and_split_guards(self):
        group = self.initialize()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            make_dataset(self.data, "train")
        self.fill(group)
        with self.assertRaisesRegex(ValueError, "identity"):
            initialize_store(self.data["rollout_zarr"], {**self.identity, "amp": "bf16"},
                             self.sources, self.model_config)
        wrong = {**self.data, "train_split": ["2025-01-01", "2025-01-04"]}
        with self.assertRaisesRegex(ValueError, "split bounds"):
            make_dataset(wrong, "train")
        group["era_complete"][0] = False
        with self.assertRaisesRegex(ValueError, "incomplete"):
            make_dataset(self.data, "train")

    def test_bad_forecast_and_overlapping_origins_rejected(self):
        group = self.initialize()
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            write_field(group, "z_fuxi6h", 0, np.full((70, 2, 2), np.nan))
        self.assertFalse(group["forecast_complete"][0])
        self.sources["val"].analysis_times = self.sources["train"].analysis_times
        with self.assertRaisesRegex(ValueError, "non-overlapping"):
            axes_for_datasets(self.sources)

    def test_normalization_change_rejected(self):
        group = self.initialize()
        self.fill(group)
        np.save(self.root / "std_era5.npy", np.full(70, 2.))
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            make_dataset(self.data, "train")

    def test_source_only_reads_past_fields_and_real_obs(self):
        ds = self.sources["train"]
        with patch.object(ds, "_read_era", side_effect=AssertionError("generic target read")):
            sample = ds[0]
        self.assertNotIn("target", sample)
        self.assertTrue(sample["obs_ztd"].any())
        self.assertEqual(sample["obs_ztd"].shape[0], 19)
        self.assertEqual(sample["era5"][0, 0, 0, 0], 0)
        self.assertEqual(sample["era5"][1, 0, 0, 0], 1)
        np.testing.assert_array_equal(sample["time_features"], time_features(pd.Timestamp("2025-01-01")))
        for bad in ({"use_obs": False}, {"zero_ztd": True}):
            with self.assertRaisesRegex(ValueError, "real OBS"):
                builder.source_metadata({"model": {**self.model_config, **bad}}, self.data, "fp32")
        data6 = {k: v for k, v in self.data.items() if k != "rollout_zarr"}
        data6["forecast_lead_hours"] = 6
        regular = make_dataset(data6, "train")
        self.assertEqual(len(regular), len(ds) + 1)
        self.assertEqual(regular[0]["target"][0, 0, 0], 2)

    def test_generator_partial_resume_complete_and_plan(self):
        config = self.root / "config.json"
        config.write_text(json.dumps({"data": self.data, "model": {"use_obs": False}}))
        argv = ["build_rollout_zarr.py", "--config", str(config), "--amp", "fp32", "--num-workers", "0"]
        with patch("sys.argv", argv + ["--plan"]):
            builder.main()
        self.assertFalse(Path(self.data["rollout_zarr"]).exists())
        with patch.object(builder, "DAForecastNet", TinyObservedSource), patch.object(
                builder, "setup_distributed", return_value=(0, 0, 1, torch.device("cpu"))):
            with patch("sys.argv", argv + ["--max-samples", "2"]):
                builder.main()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                make_dataset(self.data, "train")
            with patch("sys.argv", argv):
                builder.main()
            with patch.object(builder, "DAForecastNet", side_effect=AssertionError("already complete")):
                with patch("sys.argv", argv):
                    builder.main()
        ds = make_dataset(self.data, "train")
        self.assertTrue(torch.all(ds[0]["era5"][1] == 10))  # ERA(T)=1 + delta2 + OBS7
        self.assertTrue(open_store(self.data["rollout_zarr"])["forecast_complete"][:].all())

    def test_zero_ztd_source_policy_is_explicit(self):
        data = {**self.data, "rollout_source_observations": "zero_ztd_history_at_T"}
        saved = {"model": {**self.model_config, "zero_ztd": True}}
        self.assertTrue(builder.source_metadata(saved, data, "fp32")["model"]["zero_ztd"])
        with self.assertRaisesRegex(ValueError, "real OBS"):
            builder.source_metadata(saved, self.data, "fp32")
        with self.assertRaisesRegex(ValueError, "zero ZTD"):
            builder.source_metadata({"model": self.model_config}, data, "fp32")
        with self.assertRaisesRegex(ValueError, "unknown"):
            builder.source_metadata(saved, {**data, "rollout_source_observations": "typo"}, "fp32")

    def test_zero_ztd_generation_and_no_obs_second_hop(self):
        self.data["rollout_source_observations"] = "zero_ztd_history_at_T"
        model_config = {**self.model_config, "zero_ztd": True}
        torch.save({"model": TinyZeroZtdSource(**model_config).state_dict(),
                    "config": {"model": model_config, "data": {"forecast_lead_hours": 6}}},
                   self.data["rollout_source_checkpoint"])
        config = self.root / "zero_config.json"
        config.write_text(json.dumps({"data": self.data, "model": {"use_obs": False}}))
        argv = ["build_rollout_zarr.py", "--config", str(config), "--amp", "fp32", "--num-workers", "0"]
        with patch.object(builder, "DAForecastNet", TinyZeroZtdSource), patch.object(
                builder, "setup_distributed", return_value=(0, 0, 1, torch.device("cpu"))), patch("sys.argv", argv):
            builder.main()
        with patch("train.DAForecastDataset", side_effect=AssertionError("source opened")):
            dataset = make_dataset(self.data, "train")
            sample = dataset[0]
        self.assertEqual(dataset.source_observations, "zero_ztd_history_at_T")
        self.assertTrue(sample["era5"][1].eq(4).all())  # ERA(T)=1 + delta2 + mask1, NOT real ZTD7
        self.assertFalse(sample["obs_ztd"].any())
        self.assertFalse(sample["obs_mask"].any())
        wrong = {k: v for k, v in self.data.items() if k != "rollout_source_observations"}
        with self.assertRaisesRegex(ValueError, "observation policy"):
            make_dataset(wrong, "train")

    def test_train_save_evaluate_and_resume_entrypoints(self):
        source_state = TinyForecastCore(use_obs=True).state_dict()
        torch.save({
            "model": source_state, "epoch": 99, "step": 999,
            "optimizer": {"must_not_be_restored": True},
            "config": {"model": self.model_config, "data": {"forecast_lead_hours": 6}},
        }, self.data["rollout_source_checkpoint"])
        self.identity["source_checkpoint"] = file_identity(self.data["rollout_source_checkpoint"])
        group = self.initialize()
        self.fill(group)
        output = self.root / "training"
        config = {
            "data": self.data, "model": {"use_obs": False},
            "train": {
                "seed": 4, "epochs": 1, "batch_size_per_gpu": 2, "gradient_accumulation": 1,
                "num_workers": 0, "learning_rate": .001, "min_learning_rate": .0001,
                "weight_decay": 0., "warmup_steps": 0, "gradient_clip": 1.,
                "amp": "fp32", "freeze_fuxi": False, "loss_mode": "global_only",
                "halo_weight": .5, "halo_radius": 0, "validate_before_training": True,
                "fsdp": {"enabled": False}, "output_dir": str(output),
            },
        }
        path = self.root / "config.json"
        path.write_text(json.dumps(config))
        launch = ["train.py", "--config", str(path), "--init-checkpoint",
                  self.data["rollout_source_checkpoint"], "--init-fuxi-only"]
        with patch.object(train, "DAForecastNet", TinyForecastCore), patch.object(
                train, "setup_distributed", return_value=(0, 0, 1, torch.device("cpu"))):
            with patch("sys.argv", launch), redirect_stdout(io.StringIO()):
                train.main()
        saved = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
        self.assertEqual(saved["epoch"], 0)
        self.assertEqual(saved["step"], 3)
        self.assertTrue(saved["config"]["initialization"]["fuxi_only"])
        self.assertEqual(saved["config"]["rollout_identity"], self.identity)
        self.assertFalse(any(k.startswith(("obs_encoder.", "high_fusion.", "low_fusion."))
                             for k in saved["model"]))
        self.assertFalse(torch.equal(saved["model"]["era_encoder.weight"], source_state["era_encoder.weight"]))
        self.assertTrue((output / "initial_validation.json").exists())

        summaries = []
        for batch_size in (1, 3):
            destination = output / f"test_bs{batch_size}"
            argv = ["evaluate.py", "--config", str(path), "--checkpoint", str(output / "best.pt"),
                    "--output-dir", str(destination), "--batch-size", str(batch_size),
                    "--num-workers", "0", "--device", "cpu", "--skip-animations"]
            with patch.object(evaluate, "DAForecastNet", TinyForecastCore), patch(
                    "sys.argv", argv), redirect_stdout(io.StringIO()):
                evaluate.main()
            summary = json.loads((destination / "weighted_mae_summary.json").read_text())
            summaries.append(summary)
            self.assertEqual(summary["samples"], 6)
            self.assertEqual(summary["forecast_lead_hours"], 12)
            self.assertEqual(pd.Timestamp(summary["valid_time_first"]), pd.Timestamp("2025-01-05 12:00"))
            self.assertAlmostEqual(summary["global_persistence_latitude_weighted_mae_normalized"], 2.)
            with (destination / "weighted_mae_by_channel.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 70)
        self.assertAlmostEqual(summaries[0]["latitude_weighted_mae_normalized"],
                               summaries[1]["latitude_weighted_mae_normalized"], places=6)
        # Resume the SAME no-OBS stage; source-6h counters must never leak in.
        with patch.object(train, "DAForecastNet", TinyForecastCore), patch.object(
                train, "setup_distributed", return_value=(0, 0, 1, torch.device("cpu"))):
            with patch("sys.argv", ["train.py", "--config", str(path), "--resume",
                                   str(output / "last.pt"), "--epochs", "2"]), redirect_stdout(io.StringIO()):
                train.main()
        resumed = torch.load(output / "last.pt", weights_only=False, map_location="cpu")
        self.assertEqual((resumed["epoch"], resumed["step"]), (1, 6))

    def test_6h_reference_rejected_before_checkpoint_load(self):
        config = self.root / "config.json"
        config.write_text(json.dumps({"data": self.data}))
        (self.root / "summary.json").write_text(json.dumps({"forecast_lead_hours": 6}))
        argv = ["evaluate.py", "--config", str(config), "--checkpoint", "unused.pt",
                "--fuxi-reference", str(self.root / "channel_comparison.csv")]
        with patch("sys.argv", argv), patch.object(evaluate.torch, "load", side_effect=AssertionError("too late")):
            with self.assertRaisesRegex(ValueError, "lead does not match"):
                evaluate.main()


class CoreInitializationTest(unittest.TestCase):
    def model(self, use_obs):
        return DAForecastNet(image_size=(40, 40), era5_channels=4, out_channels=4,
                             era_static_channels=2, embed_dim=32, window_size=5,
                             depths=(1,), num_heads=4, use_obs=use_obs)

    def test_strict_core_transfer_and_masked_branch_equivalence(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)
        source, core = self.model(True).eval(), self.model(False).eval()
        report = load_fuxi_core_from_da(core, source.state_dict())
        self.assertGreater(report["discarded_obs_tensors"], 0)
        self.assertIsNone(core.obs_encoder)
        inputs = (torch.randn(1, 2, 4, 40, 40), torch.randn(1, 2, 40, 40),
                  torch.randn(1, 12), torch.zeros(1, 3, 1, 40, 40),
                  torch.zeros(1, 3, 1, 40, 40), torch.randn(1, 3, 40, 40))
        with torch.no_grad():
            torch.testing.assert_close(source(*inputs), core(*inputs))
        state = dict(source.state_dict())
        state.pop(next(k for k in state if k.startswith("era_encoder.")))
        with self.assertRaises(RuntimeError):
            load_fuxi_core_from_da(core, state)
        with self.assertRaises(ValueError):
            load_fuxi_core_from_da(source, source.state_dict())

    def test_only_core_trainable_and_updates(self):
        model = self.model(False)
        apply_freeze_policy(model, {"freeze_fuxi": False, "freeze_obs": True})
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertFalse(any(n.startswith(("obs_encoder.", "high_fusion.", "low_fusion."))
                             for n, p in model.named_parameters()))
        args = (torch.randn(1, 2, 4, 40, 40), torch.randn(1, 2, 40, 40),
                torch.randn(1, 12), None, None, None)
        loss = model(*args).square().mean()
        loss.backward()
        for name in ("era_encoder", "time_embed", "backbone", "decoder"):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in getattr(model, name).parameters()), name)
        parameter = next(model.era_encoder.parameters())
        before = parameter.detach().clone()
        torch.optim.AdamW(model.parameters(), lr=1e-4).step()
        self.assertFalse(torch.equal(before, parameter))


if __name__ == "__main__":
    unittest.main()
