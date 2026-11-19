"""CPU regression for original eight-hop inference and paired comparisons."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader

import evaluate_original_fuxi_48h as runner
import plot_original_fuxi_48h as plotting
from evaluate import IndexedEvalDataset, DistributedEvalSampler
from evaluate_original_fuxi import metric_sums
from tests import test_delay_curriculum as fixtures


class Original48DatasetTest(unittest.TestCase):
    setUp = fixtures.DelayCurriculumPipelineTest.setUp

    def test_same_common_origins_no_gnss_values_and_eight_step_feedback(self):
        ds = runner.OriginalRolloutInputs(self.data, "train")
        paired = fixtures.make_rollout_dataset(self.data, "train", 8, 8)
        self.assertTrue(ds.analysis_times.equals(paired.analysis_times))
        with patch.object(ds, "_read_observations", side_effect=AssertionError("no GNSS reads")):
            sample = ds[0]
        self.assertIsNone(ds._ztd_group)
        self.assertNotIn("obs_ztd", sample)
        self.assertEqual(sample["targets"].shape, (8, 70, 2, 2))
        expected = paired[0]
        for key in ("era5", "targets", "target_masks"):
            torch.testing.assert_close(sample[key], expected[key])
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.seen = []
            def forward(self, x, temb=None, const=None):
                self.seen.append((x.clone(), temb.clone()))
                return torch.full_like(x[:, -1], .1)
        batch = {k:v[None] for k,v in sample.items()}
        model = Toy()
        outputs = list(runner.forecasts(model, batch["era5"], batch["times"], torch.tensor(ds.era_static), "fp32"))
        self.assertEqual(len(outputs), 8)
        for j, out in enumerate(outputs):
            torch.testing.assert_close(out, batch["era5"][:, -1]+.1*(j+1))
            torch.testing.assert_close(model.seen[j][1], batch["times"][:, j])
            if j:
                torch.testing.assert_close(model.seen[j][0][:, -1], outputs[j-1])
        # Future truth is never supplied to the forecasting function.
        batch["targets"].fill_(9999.)
        again = list(runner.forecasts(Toy(), batch["era5"], batch["times"], torch.tensor(ds.era_static), "fp32"))
        for a,b in zip(outputs, again):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_padded_rank_aggregation_matches_single_process(self):
        ds = runner.OriginalRolloutInputs(self.data, "train")
        weights = torch.ones(1,1,2,1); mean = torch.zeros(1,70,1,1); std = torch.ones_like(mean)
        def run(rank, world, batch_size):
            total = torch.zeros(8,5,70,dtype=torch.float64)
            loader = DataLoader(IndexedEvalDataset(ds), batch_size=batch_size,
                                sampler=DistributedEvalSampler(len(ds),rank,world))
            for batch in loader:
                for j in range(8):
                    item = {"era5":batch["era5"],"target":batch["targets"][:,j],
                            "target_mask":batch["target_masks"][:,j].float(),"_eval_valid":batch["_eval_valid"]}
                    total[j] += metric_sums(item["target"]+.125,item,weights,mean,std)
            return total
        expected = run(0,1,1)
        torch.testing.assert_close(sum(run(r,3,2) for r in range(3)), expected)


class Original48OutputTest(unittest.TestCase):
    def setUp(self):
        self.sums = np.ones((8,7,70),np.float64)
        self.sums[:,0] = 20; self.sums[:,1] = 30; self.sums[:,2] = 40
        self.sums[:,3] = 50; self.sums[:,4] = 10; self.sums[:,5] = 10; self.sums[:,6] = 5

    def test_pairing_checks_and_improvement_sign(self):
        original, leads = plotting.baseline_tables(self.sums)
        reference = copy.deepcopy(original)
        for r in reference:
            for k in ("global_mae_normalized","global_mae_physical","halo_mae_normalized"):
                r[k] *= .8
        ref_leads = copy.deepcopy(leads)
        for r in ref_leads:
            r["global_latitude_weighted_mae_normalized"] *= .8
            r["halo_latitude_weighted_mae_normalized"] *= .8
        rows, summary = plotting.comparison_tables(original, reference, leads, {"leads":ref_leads})
        self.assertEqual(len(rows),560)
        self.assertAlmostEqual(rows[-1]["global_mae_normalized_improvement_pct"],20.)
        self.assertEqual(summary[-1]["global_improved_channels"],70)
        reference[0]["persistence_mae_normalized"] += 1.
        with self.assertRaisesRegex(ValueError,"paired-data"):
            plotting.comparison_tables(original,reference,leads,{"leads":ref_leads})

    def test_subset_never_compared_and_full_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); exp = root/"reference"; (exp/"test").mkdir(parents=True)
            original, leads = plotting.baseline_tables(self.sums)
            plotting.write_csv(exp/"test/rollout_mae_by_channel.csv",original)
            (exp/"test/rollout_mae_summary.json").write_text(json.dumps({"samples":5,"leads":leads}))
            (exp/"resolved_config.json").write_text("{}")
            meta = {"samples":5,"full_test":False,"reference_dir":str(exp),
                    "origin_first":"2025-01-01","origin_last":"2025-01-02"}
            for name,path in (("reference_csv",exp/"test/rollout_mae_by_channel.csv"),
                              ("reference_summary",exp/"test/rollout_mae_summary.json"),
                              ("reference_config",exp/"resolved_config.json")):
                meta[name] = plotting.file_identity(path)
            runner.atomic_npz(root/"sums.npz",identity=np.asarray(json.dumps(meta)),sums=self.sums,samples=np.asarray(5))
            plotting.make_comparison(root)
            self.assertFalse((root/"channel_comparison_all_leads.csv").exists())
            meta["full_test"] = True
            runner.atomic_npz(root/"sums.npz",identity=np.asarray(json.dumps(meta)),sums=self.sums,samples=np.asarray(5))
            plotting.make_comparison(root)
            for name in ("channel_comparison_all_leads.csv","comparison_report.md","channel_mae_48h.png",
                         "channel_improvement_all_leads.png","channel_mae_physical_48h.png"):
                self.assertTrue((root/name).is_file())


if __name__ == "__main__":
    unittest.main()
