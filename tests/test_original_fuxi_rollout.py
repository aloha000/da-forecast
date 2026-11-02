"""CPU checks for original-only two-hop forecasting and baseline CSV compatibility."""
import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import DataLoader

import evaluate_original_fuxi as original
from data.da_dataset import time_features
from evaluate import (
    DistributedEvalSampler, IndexedEvalDataset, read_fuxi_comparison,
    plot_channel_mae_comparison,
)
from tests import test_rollout_fields as fixtures


class ResidualTwo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, inputs, temb=None, const=None):
        self.seen.append((inputs.clone(), temb.clone()))
        return torch.full_like(inputs[:, -1], 2.)


class OriginalFuXiTest(unittest.TestCase):
    setUp = fixtures.TwoFieldTest.setUp
    initialize = fixtures.TwoFieldTest.initialize

    def test_reads_only_original_history_and_12h_label(self):
        self.initialize()  # No DA predictions or ERA5 copies are completed.
        ds = original.OriginalInputs(self.data, 12, "train")
        reads = []
        truth = self.truth
        class Track:
            def __getitem__(self, key):
                reads.append(key)
                return truth[key]
        ds._era = Track()
        sample = ds[0]
        self.assertEqual(reads, [0, 1, 3])  # Never read ERA5(T+6), index=2.
        self.assertEqual(sample["target_mask"][0, 0, 1], 0)
        self.assertEqual(sample["target"][0, 0, 1], 0)
        self.assertEqual(sample["era5"].dtype, torch.float32)
        self.assertEqual(sample["target"].dtype, torch.float32)
        self.assertNotIn("obs_ztd", sample)

    def test_two_residual_additions_predicted_feedback_and_time(self):
        self.initialize()
        ds = original.OriginalInputs(self.data, 12, "train")
        batch = next(iter(DataLoader(IndexedEvalDataset(ds), batch_size=1)))
        model = ResidualTwo()
        prediction = original.forecast(model, batch, torch.tensor(ds.const), 12,
                                       torch.device("cpu"), "fp32")
        self.assertTrue(torch.all(prediction == 5))  # ERA(T)=1, then +2, then +2.
        self.assertEqual(len(model.seen), 2)
        self.assertTrue(torch.all(model.seen[1][0][:, 0] == 1))
        self.assertTrue(torch.all(model.seen[1][0][:, 1] == 3))  # Not ERA5(T+6)=2.
        np.testing.assert_array_equal(model.seen[0][1][0], time_features(pd.Timestamp("2025-01-01")))
        np.testing.assert_array_equal(model.seen[1][1][0], time_features(pd.Timestamp("2025-01-01 06:00")))
        single = original.forecast(ResidualTwo(), batch, torch.tensor(ds.const), 6,
                                   torch.device("cpu"), "fp32")
        self.assertTrue(torch.all(single == 3))

    def test_padding_and_missing_target_do_not_enter_metrics(self):
        self.initialize()
        ds = original.OriginalInputs(self.data, 12, "train")
        batch = next(iter(DataLoader(IndexedEvalDataset(ds), batch_size=2)))
        batch["_eval_valid"][1] = False
        prediction = batch["target"] + 2
        weights = torch.ones(1, 1, 2, 1)
        mean, std = torch.zeros(1, 70, 1, 1), torch.ones(1, 70, 1, 1)
        sums = original.metric_sums(prediction, batch, weights, mean, std)
        self.assertEqual(sums[4].sum(), 279)  # 70*2*2 - one NaN cell; no padding.
        self.assertTrue(torch.allclose(sums[0], 2*sums[4]))
        self.assertTrue(torch.isfinite(sums).all())
        prediction[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            original.metric_sums(prediction, batch, weights, mean, std)
        selected = [i for rank in range(4) for i, valid in DistributedEvalSampler(6, rank, 4) if valid]
        self.assertEqual(sorted(selected), list(range(6)))

    def run_baseline(self, name, extra=()):
        config = self.root / "config.json"
        config.write_text(json.dumps({"data": self.data}))
        out = self.root / name
        argv = ["evaluate_original_fuxi.py", "--config", str(config),
                "--checkpoint", self.data["rollout_source_checkpoint"],
                "--device", "cpu", "--amp", "fp32", "--num-workers", "0",
                "--output-dir", str(out), *extra]
        with patch("sys.argv", argv), patch.object(original, "load_model", return_value=ResidualTwo()), redirect_stdout(io.StringIO()):
            original.main()
        return out

    def test_full_baseline_batchsize_and_plot_contract(self):
        self.initialize()
        output = self.run_baseline("baseline", ["--batch-size", "1"])
        other = self.run_baseline("baseline_bs3", ["--batch-size", "3"])
        summary = json.loads((output / "summary.json").read_text())
        summary3 = json.loads((other / "summary.json").read_text())
        self.assertEqual(summary["samples"], 6)
        self.assertTrue(summary["is_full_test"])
        self.assertEqual(summary["forecast_lead_hours"], 12)
        self.assertEqual(pd.Timestamp(summary["valid_time_first"]), pd.Timestamp("2025-01-05 12:00"))
        self.assertAlmostEqual(summary["overall_latitude_weighted_mae_normalized"], 2.)
        self.assertAlmostEqual(summary["persistence_latitude_weighted_mae_normalized"], 2.)
        self.assertAlmostEqual(summary["overall_latitude_weighted_mae_normalized"],
                               summary3["overall_latitude_weighted_mae_normalized"], places=6)
        with (output / "channel_comparison.csv").open() as handle:
            reference = list(csv.DictReader(handle))
        self.assertEqual(len(reference), 70)
        da = self.root / "da_test"
        da.mkdir()
        rows = []
        for ref in reference:
            row = {"channel": ref["channel"], "unit": ref["unit"]}
            for region in ("global", "halo"):
                for scale in ("normalized", "physical"):
                    row[f"{region}_weighted_mae_{scale}"] = float(ref[f"original_mae_{scale}"])*.5
                    row[f"{region}_persistence_mae_{scale}"] = float(ref[f"persistence_mae_{scale}"])
            rows.append(row)
        with (da / "weighted_mae_by_channel.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        model_summary = {
            "samples": 6, "forecast_lead_hours": 12, "test_split": self.data["test_split"],
            "valid_time_first": summary["valid_time_first"], "valid_time_last": summary["valid_time_last"],
            "evaluation": {"split": "test", "amp": "fp32", "rollout": {"identity": self.identity}},
            "global_latitude_weighted_mae_normalized": 1.,
        }
        (da / "weighted_mae_summary.json").write_text(json.dumps(model_summary))
        paired, _, _ = read_fuxi_comparison(da, output / "channel_comparison.csv")
        self.assertEqual(len(paired), 70)
        self.assertAlmostEqual(paired[0]["da_improvement_percent_normalized"], 50.)
        with redirect_stdout(io.StringIO()):
            plots = plot_channel_mae_comparison(da, output / "channel_comparison.csv")
        self.assertTrue(all(p.exists() and p.stat().st_size > 0 for p in plots))
        model_summary["evaluation"]["rollout"]["identity"] = {"wrong": True}
        (da / "weighted_mae_summary.json").write_text(json.dumps(model_summary))
        with self.assertRaisesRegex(ValueError, "identity differs"):
            read_fuxi_comparison(da, output / "channel_comparison.csv")

    def test_smoke_and_plan_are_not_full_test(self):
        self.initialize()
        output = self.run_baseline("smoke", ["--max-samples", "2"])
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["samples"], 2)
        self.assertFalse(summary["is_full_test"])
        with self.assertRaisesRegex(RuntimeError, "results already exist"):
            self.run_baseline("smoke", ["--max-samples", "2"])
        output = self.run_baseline("plan", ["--plan"])
        self.assertFalse(output.exists())

    def test_missing_history_and_wrong_config_rejected(self):
        self.initialize()
        wrong = {**self.data, "obs_interval_minutes": 5}
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            original.OriginalInputs(wrong, 12)
        with self.assertRaisesRegex(ValueError, "exact rollout"):
            original.OriginalInputs(self.data, 12, start="2025-01-01")
        ds = original.OriginalInputs(self.data, 6, "train")
        self.assertEqual(len(ds), 7)
        self.assertEqual(ds[0]["target"][0, 0, 0], 2)
        era = zarr.open_group(self.data["era5_path"], mode="r+")
        era["time"][0] = -6  # Remove T-6 for the first training origin.
        with self.assertRaisesRegex(ValueError, "missing required timestamp"):
            original.OriginalInputs(self.data, 12, "train")

    def test_checkpoint_prefix_loading_stays_strict(self):
        checkpoint = self.root / "fuxi.pt"
        torch.save({"model": {"decoder.weight": torch.tensor([[3.]])}}, checkpoint)
        with patch.object(original, "UTransformer", side_effect=lambda **kwargs: torch.nn.Linear(1, 1, bias=False)):
            model = original.load_model(str(checkpoint), torch.device("cpu"), amp="fp32")
            self.assertEqual(float(model.weight[0, 0].detach()), 3.)
            torch.save({"model": {"wrong.weight": torch.tensor([[3.]])}}, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                original.load_model(str(checkpoint), torch.device("cpu"), amp="fp32")


if __name__ == "__main__":
    unittest.main()
