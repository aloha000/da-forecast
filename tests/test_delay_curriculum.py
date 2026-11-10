"""Online joint 48h recurrence, real data interfaces, resume and eval regression."""
import io
import json
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import zarr
import train
import train_rollout_delays as trainer
import evaluate_rollout_delays as evaluator
from data.delay_sequence import make_rollout_dataset
from model.delay_rollout import rollout_predictions, rollout_loss, INPUT_KEYS
from model.delay_components import component_prediction, COMPONENT_INPUTS
from delay_rollout_utils import stage_for_update, eval_loader, measure_rollout
from tests import test_delay_components as dc
from tests import test_da_forecast_model as mf
from tests import test_rollout_fields as fixtures


class TinyJoint(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.predict_delay_components = True
        for name in ("era_encoder","time_embed","backbone","decoder","obs_encoder",
                     "high_fusion","low_fusion","component_adapter"):
            setattr(self,name,torch.nn.Linear(1,1))
        self.decoder.zhd_head = torch.nn.Linear(1,18)
        self.decoder.zwd_head = torch.nn.Linear(1,18)
        for p in self.parameters():
            torch.nn.init.constant_(p,.0001)

    def forward(self,era5,era_static,time_features,obs_ztd,obs_mask,obs_static,
                obs_components,obs_component_mask,ztd_stats,**kwargs):
        # Exercise weather -> delay and delay -> weather gradients.
        gain=sum(p.sum() for name in ("era_encoder","time_embed","backbone",
                 "obs_encoder","high_fusion","low_fusion","component_adapter")
                 for p in getattr(self,name).parameters())
        gain=gain+self.decoder.weight.sum()+self.decoder.bias.sum()
        x=era5[:,-1,0].mean((1,2))[:,None]*.01+1.
        delta=torch.stack([getattr(self.decoder,n)(x) for n in ("zhd_head","zwd_head")],2)
        delta=delta[:,:,:,None,None].expand(-1,-1,-1,*era5.shape[-2:])
        noise=torch.rand((),device=era5.device)*1e-5 if self.training else 0.
        return {"weather":era5[:,-1]+gain+obs_components[:,-1,1:2]*1e-5+noise,
                **component_prediction(delta,obs_components,obs_component_mask,
                                       [100,100],ztd_stats)}


def tiny_batch():
    batch=dc.component_batch()
    batch.update(era5=torch.zeros(1,2,1,2,2),era_static=torch.zeros(1,5,2,2),
                 obs_static=torch.ones(1,3,2,2),time_features=torch.zeros(1,12),
                 rollout_time_features=torch.zeros(1,8,12))
    return batch


class DelayRolloutUnitTest(unittest.TestCase):
    def test_eight_hop_checkpoint_equivalence_and_feedback_gradients(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w=torch.nn.Parameter(torch.tensor(.1))
                self.d=torch.nn.Parameter(torch.tensor(.2))
            def forward(self,era5,era_static,time_features,obs_ztd,obs_mask,obs_static,
                        obs_components,obs_component_mask,ztd_stats):
                return {"weather":era5[:,-1]+self.w+obs_components[:,-1,1:2]*.01,
                        **component_prediction(self.d.expand(1,18,2,2,2),obs_components,
                                               obs_component_mask,[1,1],ztd_stats)}
        values=[]; gradients=[]
        for ck in (False,True):
            model=Toy(); batch=tiny_batch()
            # Ensure no accidental truth fields are passed to the model.
            batch["component_targets"]=torch.full((1,8,18,2,2,2),99999.)
            outs=list(rollout_predictions(model,batch,8,ck))
            self.assertEqual(len(outs),8)
            self.assertAlmostEqual(outs[-1]["components"][0,-1,1,0,0].item(),
                                   100.125+8*.2,places=3)
            outs[-1]["weather"].mean().backward()
            self.assertAlmostEqual(model.w.grad.item(),8.,places=4)
            self.assertAlmostEqual(model.d.grad.item(),.28,places=4)
            values.append(outs[-1]["weather"].detach())
            gradients.append(torch.stack((model.w.grad,model.d.grad)))
        torch.testing.assert_close(*values,rtol=0,atol=0)
        torch.testing.assert_close(*gradients,rtol=0,atol=0)

    def test_real_network_recompute_and_mm_precision(self):
        model=dc.tiny_model()
        batch=mf.make_inputs();batch.update(dc.component_batch(40,40))
        batch["rollout_time_features"]=torch.randn(1,2,12)
        with patch.object(model.obs_encoder,"forward",wraps=model.obs_encoder.forward) as obs:
            outputs=list(rollout_predictions(model,batch,2,True))
            # Second hop still executes OBS, unlike the main-branch experiment.
            self.assertEqual(obs.call_count,2)
        self.assertEqual(outputs[-1]["components"].dtype,torch.float32)
        loss=outputs[-1]["weather"].square().mean()+outputs[-1]["components"].mean()
        loss.backward()
        self.assertGreater(model.decoder.zhd_head.weight.grad.abs().sum(),0)
        self.assertGreater(model.decoder.zwd_head.weight.grad.abs().sum(),0)
        self.assertGreater(model.era_encoder.patch_embed.proj.weight.grad.abs().sum(),0)

    def test_curriculum_boundaries_and_loss_average_no_ztd_loss(self):
        self.assertEqual([stage_for_update(i,500,8) for i in (0,499,500,3499,3500,3999)],
                         [1,1,2,7,8,8])
        model=TinyJoint().eval();batch=tiny_batch()
        batch.update(targets=torch.zeros(1,2,1,2,2),target_masks=torch.ones(1,2,1,2,2),
                     component_targets=torch.zeros(1,2,18,2,2,2),
                     component_target_masks=torch.ones(1,2,18,2,2,2),
                     component_loss_scales=torch.tensor([[100.,100.]]))
        loss,stats=rollout_loss(model,batch,2,torch.tensor([45.,46.]),
                               {"rollout_checkpoint":False})
        torch.testing.assert_close(loss,stats[:,0].mean())
        torch.testing.assert_close(stats[:,0],stats[:,1]+.1*stats[:,2]/100+.1*stats[:,3]/100)
        batch["ztd_target"]=torch.full((1,18,1,2,2),float("nan"))
        again,_=rollout_loss(model,batch,2,torch.tensor([45.,46.]),{"rollout_checkpoint":False})
        torch.testing.assert_close(loss,again)


class DelayCurriculumPipelineTest(unittest.TestCase):
    def setUp(self):
        fixtures.TwoFieldTest.setUp(self)
        obs=zarr.open_group(self.data["ztd_path"],mode="r+")
        obs.attrs.update(ztd_train_mean=2000.,ztd_train_std=100.,zwd_train_mean=100.,zwd_train_std=10.)
        obs["ztd"].attrs.update(units="1",raw_units="mm")
        wet=np.full(obs["ztd"].shape,np.nan,np.float32);wet[:,0,0]=2.
        fixtures.array(obs,"zwd",wet).attrs.update(units="1",raw_units="mm")
        era=zarr.open_group(self.data["era5_path"],mode="r+")
        era["z"].resize((50,70,2,2))
        era["z"][:]=np.broadcast_to(np.arange(50,dtype=np.float32)[:,None,None,None]*.01,(50,70,2,2))
        era["time"].resize((50,));era["time"][:]=np.arange(50)*6
        self.data={k:v for k,v in self.data.items() if not k.startswith("rollout_")}
        self.data.update(forecast_lead_hours=6,train_split=["2025-01-01","2025-01-04"],
                         val_split=["2025-01-04","2025-01-07"],test_split=["2025-01-07","2025-01-10"],
                         predict_ztd=True,predict_delay_components=True,component_loss_scales=[100,100],
                         use_delay_physics=False)
        cfg=json.loads((Path(__file__).resolve().parents[1]/"configs/train_rollout_48h_predict_delays.json").read_text())
        cfg["data"]=self.data
        cfg["model"]={"use_obs":True,"predict_ztd":True,"predict_delay_components":True,
                      "component_loss_scales":[100,100],"use_delay_physics":False}
        cfg["train"].update(amp="fp32",batch_size_per_gpu=1,gradient_accumulation=2,num_workers=0,
            seed=22,learning_rate=.001,min_learning_rate=.0001,warmup_steps=1,weight_decay=0.,
            log_interval=1,save_every=2,validate_every=2,fsdp={"enabled":False},
            output_dir=str(self.root/"run"))
        cfg["curriculum"]["updates_per_stage"]=2
        cfg["initial_checkpoint"]=str(self.root/"init.pt")
        torch.save({"model":TinyJoint().state_dict(),"config":{"model":cfg["model"]}},cfg["initial_checkpoint"])
        self.config=cfg
        self.path=self.root/"cfg.json";self.path.write_text(json.dumps(cfg))

    def run_train(self,*args):
        with patch.object(trainer,"DAForecastNet",TinyJoint),patch.object(trainer,"setup_rollout_distributed",return_value=(0,1,torch.device("cpu"))),patch("sys.argv",["train_rollout_delays.py","--config",str(self.path),*args]),redirect_stdout(io.StringIO()):
            trainer.main()

    def test_future_truth_never_changes_recurrent_inputs(self):
        ds=make_rollout_dataset(self.data,"train",8,8)
        a=ds[0];t=ds.sample_time(0)
        self.assertEqual(a["component_targets"].shape,(8,18,2,2,2))
        self.assertTrue(t+pd.Timedelta(hours=48)<pd.Timestamp(self.data["train_split"][1]))
        torch.testing.assert_close(a["targets"][:,0,0,0],torch.arange(2,10)*.01)
        self.assertNotIn("ztd_target",a)
        group=zarr.open_group(self.data["ztd_path"],mode="r+")
        cut=int(ds.obs_times.searchsorted(t+pd.Timedelta(minutes=5)))
        group["ztd"][cut:]=np.nan;group["zwd"][cut:]=np.nan
        b=ds[0]
        self.assertTrue(a["component_target_masks"].any())
        self.assertFalse(b["component_target_masks"].any())
        for key in INPUT_KEYS+COMPONENT_INPUTS+("rollout_time_features",):
            if key in a:
                torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        model=TinyJoint().eval()
        def predict(sample):
            batch={k:v[None] for k,v in sample.items()}
            return list(rollout_predictions(model,batch,8))[-1]
        pa,pb=predict(a),predict(b)
        for k in pa:
            torch.testing.assert_close(pa[k],pb[k],rtol=0,atol=0)

    def test_all_stages_bit_exact_resume_and_evaluation_batch_invariance(self):
        self.run_train("--output-dir",str(self.root/"full"))
        self.run_train("--stop-after-updates","3")
        self.run_train("--resume",str(self.root/"run/last.pt"))
        full=torch.load(self.root/"full/last.pt",weights_only=False)
        resumed=torch.load(self.root/"run/last.pt",weights_only=False)
        self.assertEqual(full["step"],16)
        for key in full["model"]:
            torch.testing.assert_close(full["model"][key],resumed["model"][key],rtol=0,atol=0)
        for lead in range(6,49,6):
            self.assertTrue((self.root/f"run/stage_{lead:02d}h/best.pt").exists())
        results=[]
        for bs in (1,3):
            out=self.root/f"eval{bs}"
            args=["evaluate_rollout_delays.py","--config",str(self.path),"--checkpoint",
                  str(self.root/"run/last.pt"),"--output-dir",str(out),
                  "--batch-size",str(bs),"--num-workers","0"]
            with patch.object(evaluator,"DAForecastNet",TinyJoint),patch.object(evaluator,"setup_rollout_distributed",return_value=(0,1,torch.device("cpu"))),patch("sys.argv",args),redirect_stdout(io.StringIO()):
                evaluator.main()
            report=json.loads((out/"rollout_mae_summary.json").read_text())
            self.assertEqual(len(report["leads"]),8)
            self.assertTrue((out/"rollout_delay_mae_by_frame.csv").exists())
            results.append(report)
        for a,b in zip(results[0]["leads"],results[1]["leads"]):
            for key in ("global_latitude_weighted_mae_normalized","zhd_mae_mm","zwd_mae_mm"):
                self.assertAlmostEqual(a[key],b[key],places=5)

    def test_eval_padding_and_loss_weight_resume_guard(self):
        ds=make_rollout_dataset(self.data,"train",8,8);model=TinyJoint()
        total=measure_rollout(model,eval_loader(ds,1,0,0,1),8,ds.lat,torch.device("cpu"),"fp32",1,0)
        shards=[measure_rollout(model,eval_loader(ds,1,0,r,3),8,ds.lat,torch.device("cpu"),"fp32",1,0) for r in range(3)]
        for key in ("global_error","component_error","component_count"):
            torch.testing.assert_close(sum(s[key] for s in shards),total[key])
        self.assertEqual(sum(s["samples"] for s in shards),len(ds))
        self.run_train("--stop-after-updates","1")
        self.config["train"]["zwd_loss_weight"]=.2
        self.path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError,"zwd_loss_weight"):
            self.run_train("--resume",str(self.root/"run/last.pt"))


if __name__=="__main__":
    unittest.main()
