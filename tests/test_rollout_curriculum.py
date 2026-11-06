"""Origin-only component input, full BPTT, 8-stage curriculum and exact resume tests."""
import io
import json
from pathlib import Path
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import zarr
import train
import train_rollout as trainer
import evaluate_rollout as evaluator
from data.rollout_sequence import split_delays,make_rollout_dataset
from model import DAForecastNet
from model.weather_rollout import rollout_predictions,rollout_loss
from rollout_utils import BatchStream,stage_for_update,eval_loader,measure_rollout
from tests import test_da_forecast_model as mf
from tests import test_rollout_fields as fixtures

def dual_model():
    return DAForecastNet(image_size=(40,40),era5_channels=4,era_static_channels=2,
                        out_channels=4,embed_dim=32,patch_size=4,window_size=5,
                        depths=(1,),num_heads=4,obs_channels=2)

class RolloutUnitTest(unittest.TestCase):
    def test_component_units_pair_mask_and_preserved_sum(self):
        total=np.array([1.,2.,np.nan],np.float32)
        wet=np.array([3.,np.nan,1.],np.float32)
        h,w=split_delays(total,wet,[2000,100],[100,10])
        self.assertAlmostEqual(h[0],.7,places=6)
        self.assertAlmostEqual(w[0],.3,places=6)
        self.assertAlmostEqual(h[0]+w[0],total[0],places=6)
        self.assertTrue(np.isnan(h[1:]).all())
        self.assertTrue(np.isnan(w[1:]).all())

    def test_old_kernel_migration_and_direct_obs_bypass(self):
        torch.manual_seed(8)
        old=mf.make_model().eval();new=dual_model().eval()
        train.initialize_model_weights(new,{"model":old.state_dict(),"config":{"model":{}}},
                                       {"obs_channels":2})
        batch=mf.make_inputs()
        dual=torch.cat((batch["obs_ztd"]*.3,batch["obs_ztd"]*.7),2)
        new_batch={**batch,"obs_ztd":dual,"obs_mask":torch.ones_like(dual)}
        torch.testing.assert_close(new(**new_batch),old(**batch),rtol=1e-4,atol=3e-5)
        with patch.object(new.obs_encoder,"forward",side_effect=AssertionError("OBS must not run")):
            out=new(batch["era5"],batch["era_static"],batch["time_features"],obs_enabled=False)
        no_obs=mf.make_model(use_obs=False).eval()
        no_obs.load_state_dict({k:v for k,v in old.state_dict().items() if k in no_obs.state_dict()},strict=True)
        torch.testing.assert_close(out,no_obs(**batch),rtol=0,atol=0)

    def test_eight_hops_checkpoint_bptt_and_no_teacher_forcing(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w=torch.nn.Parameter(torch.tensor(.1))
                self.obs=torch.nn.Parameter(torch.tensor(.5))
            def forward(self,era,static,time,ztd=None,mask=None,obs_static=None,obs_enabled=True):
                if not obs_enabled:
                    assert ztd is None and mask is None and obs_static is None
                return era[:,-1]+self.w+(self.obs if obs_enabled else 0)
        batch={"era5":torch.zeros(1,2,1,2,2),"era_static":torch.zeros(1,5,2,2),
               "rollout_time_features":torch.zeros(1,8,12),"obs_ztd":torch.ones(1,19,2,2,2),
               "obs_mask":torch.ones(1,19,2,2,2),"obs_static":torch.ones(1,3,2,2),
               "targets":torch.full((1,8,1,2,2),999.)}
        gradients=[]
        for checkpointed in (False,True):
            model=Toy()
            outputs=list(rollout_predictions(model,batch,8,checkpointed))
            self.assertEqual(len(outputs),8)
            self.assertAlmostEqual(outputs[-1].mean().item(),1.3,places=5)
            outputs[-1].mean().backward()
            self.assertAlmostEqual(model.obs.grad.item(),1.)
            self.assertAlmostEqual(model.w.grad.item(),8.)
            gradients.append((model.w.grad,model.obs.grad))
        for a,b in zip(*gradients):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_real_network_two_step_recompute_gradients(self):
        model=dual_model()
        batch=mf.make_inputs()
        batch.update(obs_ztd=torch.randn(1,3,2,40,40),obs_mask=torch.ones(1,3,2,40,40),
                     rollout_time_features=torch.randn(1,2,12))
        outputs=list(rollout_predictions(model,batch,2,True))
        outputs[-1].square().mean().backward()
        self.assertGreater(model.obs_encoder.temporal[0].weight.grad.abs().sum(),0)
        self.assertGreater(model.era_encoder.patch_embed.proj.weight.grad.abs().sum(),0)

    def test_stage_boundaries_and_lr(self):
        self.assertEqual([stage_for_update(i,500,8) for i in (0,499,500,999,1000,3499,3500,3999)],
                         [1,1,2,2,3,7,8,8])
        for j in range(8):
            self.assertAlmostEqual(trainer.lr_factor(j*500,500,25,.1),1/25)
            self.assertAlmostEqual(trainer.lr_factor(j*500+499,500,25,.1),.1)

class TinyWeather(torch.nn.Module):
    def __init__(self,obs_channels=2,**kwargs):
        super().__init__()
        self.obs_channels=obs_channels
        for name in ("era_encoder","time_embed","backbone","decoder","obs_encoder","high_fusion","low_fusion"):
            module=torch.nn.Linear(1,1)
            torch.nn.init.constant_(module.weight,.001)
            torch.nn.init.zeros_(module.bias)
            setattr(self,name,module)
    def forward(self,era,static,time,ztd=None,mask=None,obs_static=None,obs_enabled=True):
        x=era[:,-1,:1]
        delta=sum(getattr(self,n).weight.sum()+getattr(self,n).bias.sum()
                  for n in ("era_encoder","time_embed","backbone","decoder"))
        if obs_enabled:
            assert ztd.shape[2]==2
            delta=delta+sum(getattr(self,n).weight.sum()+getattr(self,n).bias.sum()
                           for n in ("obs_encoder","high_fusion","low_fusion"))
        else:
            assert ztd is None and mask is None
        # Randomness exercises checkpoint/RNG preservation on CPU.
        noise=torch.rand((),device=era.device)*.0001 if self.training else 0.
        return era[:,-1]+delta+noise

class CurriculumPipelineTest(unittest.TestCase):
    def setUp(self):
        fixtures.TwoFieldTest.setUp(self)
        obs=zarr.open_group(self.data["ztd_path"],mode="r+")
        obs.attrs.update(ztd_train_mean=2000.,ztd_train_std=100.,zwd_train_mean=100.,zwd_train_std=10.)
        obs["ztd"].attrs.update(units="1",raw_units="mm")
        wet=np.full(obs["ztd"].shape,np.nan,np.float32);wet[:,0,0]=2.
        array=fixtures.array(obs,"zwd",wet);array.attrs.update(units="1",raw_units="mm")
        era=zarr.open_group(self.data["era5_path"],mode="r+")
        era["z"].resize((50,70,2,2))
        era["z"][:]=np.broadcast_to(np.arange(50,dtype=np.float32)[:,None,None,None]*.01,(50,70,2,2))
        era["time"].resize((50,));era["time"][:]=np.arange(50)*6
        self.data={k:v for k,v in self.data.items() if not k.startswith("rollout_")}
        self.data.update(forecast_lead_hours=6,train_split=["2025-01-01","2025-01-04"],
                         val_split=["2025-01-04","2025-01-07"],test_split=["2025-01-07","2025-01-10"])
        model_config={"image_size":[2,2],"era5_channels":70,"use_obs":True,"zero_ztd":False,"obs_channels":2}
        initial=self.root/"init.pt"
        torch.save({"model":TinyWeather().state_dict(),"config":{"model":model_config}},initial)
        self.config={"data":self.data,"model":model_config,"initial_checkpoint":str(initial),
            "curriculum":{"max_steps":8,"updates_per_stage":2,"observation_policy":"origin_only","loss_aggregation":"mean_all_active_leads"},
            "train":{"seed":22,"amp":"fp32","batch_size_per_gpu":1,"gradient_accumulation":2,"num_workers":0,
                "learning_rate":.001,"min_learning_rate":.0001,"warmup_steps":1,"weight_decay":0.,
                "gradient_clip":1.,"freeze_fuxi":False,"freeze_obs":False,"rollout_checkpoint":True,
                "loss_mode":"global_only","halo_weight":.5,"halo_radius":0,"save_every":2,"validate_every":2,
                "log_interval":1,"fsdp":{"enabled":False},"output_dir":str(self.root/"run")}}
        self.path=self.root/"cfg.json";self.path.write_text(json.dumps(self.config))

    def run_train(self,*args):
        with patch.object(trainer,"DAForecastNet",TinyWeather),patch.object(trainer,"setup_rollout_distributed",return_value=(0,1,torch.device("cpu"))),patch("sys.argv",["train_rollout.py","--config",str(self.path),*args]),redirect_stdout(io.StringIO()):
            trainer.main()

    def test_eight_stage_schedule_and_bit_exact_resume_and_eval(self):
        self.run_train("--output-dir",str(self.root/"full"))
        self.run_train("--stop-after-updates","3")
        self.run_train("--resume",str(self.root/"run/last.pt"))
        full=torch.load(self.root/"full/last.pt",weights_only=False)
        resumed=torch.load(self.root/"run/last.pt",weights_only=False)
        self.assertEqual(full["step"],16)
        self.assertEqual(resumed["step"],16)
        self.assertEqual(resumed["config"]["curriculum_state"]["stage"],8)
        for key in full["model"]:
            torch.testing.assert_close(full["model"][key],resumed["model"][key],rtol=0,atol=0)
        records=[json.loads(s) for s in (self.root/"run/validation.jsonl").read_text().splitlines()]
        self.assertEqual([r["stage"] for r in records],list(range(1,9)))
        self.assertEqual([r["update"] for r in records],list(range(2,17,2)))
        for lead in range(6,49,6):
            self.assertTrue((self.root/f"run/stage_{lead:02d}h/best.pt").exists())
        reports=[]
        for batch_size in (1,3):
            out=self.root/f"eval{batch_size}"
            args=["evaluate_rollout.py","--config",str(self.path),"--checkpoint",str(self.root/"run/last.pt"),
                  "--output-dir",str(out),"--batch-size",str(batch_size),"--num-workers","0"]
            with patch.object(evaluator,"DAForecastNet",TinyWeather),patch.object(evaluator,"setup_rollout_distributed",return_value=(0,1,torch.device("cpu"))),patch("sys.argv",args),redirect_stdout(io.StringIO()):
                evaluator.main()
            report=json.loads((out/"rollout_mae_summary.json").read_text());reports.append(report)
            self.assertEqual([v["lead_hours"] for v in report["leads"]],list(range(6,49,6)))
        for a,b in zip(reports[0]["leads"],reports[1]["leads"]):
            self.assertAlmostEqual(a["global_latitude_weighted_mae_normalized"],b["global_latitude_weighted_mae_normalized"],places=7)

    def test_labels_alignment_and_no_future_observation_input(self):
        ds=make_rollout_dataset(self.data,"train",8,8)
        sample=ds[0];t=ds.sample_time(0)
        self.assertEqual(sample["targets"].shape,(8,70,2,2))
        self.assertTrue(t+pd.Timedelta(hours=48)<pd.Timestamp(self.data["train_split"][1]))
        torch.testing.assert_close(sample["targets"][:,0,0,0],torch.arange(2,10)*.01)
        total=sample["obs_ztd"].sum(1)
        torch.testing.assert_close(total[:,0,0],torch.full((19,),7.))
        group=zarr.open_group(self.data["ztd_path"],mode="r+")
        cut=int(ds.obs_times.searchsorted(t+pd.Timedelta(minutes=5)))
        group["ztd"][cut:]=np.nan;group["zwd"][cut:]=np.nan
        changed=ds[0]
        for key in ("era5","obs_ztd","obs_mask","rollout_time_features"):
            torch.testing.assert_close(sample[key],changed[key],rtol=0,atol=0)

    def test_eval_padding_excluded_and_later_origins_same(self):
        ds=make_rollout_dataset(self.data,"train",8,1)
        before=ds.analysis_times.copy();ds.horizon=8
        self.assertTrue(before.equals(ds.analysis_times))
        loader=eval_loader(ds,1,0,0,1)
        model=TinyWeather()
        sums=measure_rollout(model,loader,8,ds.lat,torch.device("cpu"),"fp32",1,0)
        self.assertEqual(sums["samples"],len(ds))
        # Simulate two padded shards independently without distributed collectives.
        shards=[]
        for rank in range(3):
            shards.append(measure_rollout(model,eval_loader(ds,1,0,rank,3),8,ds.lat,torch.device("cpu"),"fp32",1,0))
        torch.testing.assert_close(sum(s["global_error"] for s in shards),sums["global_error"])
        self.assertEqual(sum(s["samples"] for s in shards),len(ds))

    def test_resume_mismatch_rejected(self):
        self.run_train("--stop-after-updates","1")
        self.config["train"]["gradient_accumulation"]=3;self.path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError,"gradient_accumulation"):
            self.run_train("--resume",str(self.root/"run/last.pt"))

if __name__=="__main__":
    unittest.main()
