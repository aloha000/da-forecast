import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from evaluate import (
    CHANNELS, channel_unit, parse_args, plot_channel_mae_comparison,
    read_channel_mae_rows, read_fuxi_comparison,
)


class ChannelMAEPlotTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.rows = []
        for index, channel in enumerate(CHANNELS):
            row = {"channel": channel, "unit": channel_unit(channel)}
            for region in ("global", "halo"):
                for scale in ("normalized", "physical"):
                    row[f"{region}_weighted_mae_{scale}"] = .01 * (index + 1)
                    row[f"{region}_persistence_mae_{scale}"] = .02 * (index + 1)
            self.rows.append(row)
        self.reference_dir = self.output / "reference"
        self.reference_dir.mkdir()
        self.reference = self.reference_dir / "channel_comparison.csv"
        refs = []
        for row in self.rows:
            ref = {"channel": row["channel"], "unit": row["unit"]}
            for scale in ("normalized", "physical"):
                ref[f"original_mae_{scale}"] = row[f"global_persistence_mae_{scale}"]
                ref[f"persistence_mae_{scale}"] = row[f"global_persistence_mae_{scale}"]
                ref[f"da_mae_{scale}"] = 999  # Must NOT be used as the baseline.
            refs.append(ref)
        with self.reference.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=refs[0].keys())
            writer.writeheader()
            writer.writerows(refs[::-1])
        self.model_summary = {
            "samples": 2, "test_split": ["2025-01-01", "2025-01-01 12:00:00"],
            "evaluation": {"split": "test", "amp": "bf16"},
            "global_latitude_weighted_mae_normalized": .1,
        }
        self.ref_summary = {
            "samples": 2, "is_full_test": True, "amp": "bf16",
            "valid_time_first": "2025-01-01 06:00:00",
            "valid_time_last": "2025-01-01 12:00:00",
            "original_global_latitude_weighted_mae_normalized": .2,
        }
        (self.output / "weighted_mae_summary.json").write_text(json.dumps(self.model_summary))
        self.write_ref_summary()

    def write_ref_summary(self):
        (self.reference_dir / "summary.json").write_text(json.dumps(self.ref_summary))

    def write(self, rows):
        with (self.output / "weighted_mae_by_channel.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def test_channel_order_and_values(self):
        self.write(self.rows[::-1])
        rows = read_channel_mae_rows(self.output)
        self.assertEqual([r["channel"] for r in rows], CHANNELS)
        self.assertAlmostEqual(rows[69]["global_weighted_mae_physical"], .7)

    def test_missing_or_duplicate_channel_rejected(self):
        for rows in (self.rows[:-1], self.rows + [self.rows[0]]):
            self.write(rows)
            with self.assertRaisesRegex(ValueError, "exactly once"):
                read_channel_mae_rows(self.output)

    def test_negative_mae_rejected(self):
        self.rows[0]["global_weighted_mae_physical"] = -1
        self.write(self.rows)
        with self.assertRaisesRegex(ValueError, "invalid"):
            read_channel_mae_rows(self.output)

    def test_plot_only_needs_no_checkpoint(self):
        with patch("sys.argv", ["evaluate.py", "--plot-only", "--output-dir", str(self.output),
                                "--fuxi-reference", str(self.reference)]):
            args = parse_args()
        self.assertTrue(args.plot_only)
        self.assertIsNone(args.checkpoint)

    def test_original_column_and_channel_alignment(self):
        self.write(self.rows[::-1])
        rows, _, _ = read_fuxi_comparison(self.output, self.reference)
        self.assertEqual([r["channel"] for r in rows], CHANNELS)
        self.assertEqual(rows[0]["original_mae_normalized"], .02)
        self.assertEqual(rows[0]["da_improvement_percent_normalized"], 50.)

    def test_incompatible_reference_rejected(self):
        self.write(self.rows)
        original = dict(self.ref_summary)
        for changes in ({"is_full_test": False}, {"samples": 4},
                        {"amp": "fp32"}, {"valid_time_first": "2024-01-01"}):
            self.ref_summary = {**original, **changes}
            self.write_ref_summary()
            with self.assertRaises(ValueError):
                read_fuxi_comparison(self.output, self.reference)

    def test_different_persistence_rejected(self):
        self.rows[0]["global_persistence_mae_normalized"] = 10
        self.write(self.rows)
        with self.assertRaisesRegex(ValueError, "persistence consistency"):
            read_fuxi_comparison(self.output, self.reference)

    def test_three_fuxi_plots_and_close_figures(self):
        self.rows[0]["halo_weighted_mae_normalized"] = float("nan")
        self.write(self.rows)
        before = plt.get_fignums()
        original_savefig = Figure.savefig

        def checked_savefig(fig, path, *args, **kwargs):
            if Path(path).name == "mae_vs_fuxi_normalized.png":
                for ax in fig.axes:
                    self.assertGreaterEqual(ax.get_ylim()[1], 1.4)
                    for bar in ax.patches:
                        self.assertLessEqual(bar.get_height(), ax.get_ylim()[1])
            return original_savefig(fig, path, *args, **kwargs)

        with patch.object(Figure, "savefig", checked_savefig):
            paths = plot_channel_mae_comparison(self.output, self.reference)
        self.assertEqual(len(paths), 3)
        for path in paths:
            self.assertGreater(path.stat().st_size, 1000)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(plt.get_fignums(), before)


if __name__ == "__main__":
    unittest.main()
