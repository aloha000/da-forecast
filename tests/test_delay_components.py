"""Component-only loss, residual initialization, physics, and CLI rollout tests."""
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
import train
import evaluate
import build_rollout_zarr as builder
from model import DAForecastNet
from model.delay_components import (component_prediction, component_error_sums, component_aux_loss,
                                    next_component_inputs, latest_components)
from model.delay_physics import profile_delays_mm, weather_delays_mm
from model.ztd_rollout import advance_rollout_batch, forecast_rollout
from data.delay_components import paired_components, read_components
from data.rollout_fields import open_store, write_field
from tests import test_da_forecast_model as mf
from tests import test_rollout_fields as rf

def tiny_model():
    return DAForecastNet(image_size=(40,40), era5_channels=4, era_static_channels=2,
                        out_channels=4, embed_dim=32, patch_size=4, window_size=5,
                        depths=(1,), num_heads=4, predict_ztd=True, predict_delay_components=True)

def component_batch(h=2,w=2):
    values = torch.zeros(1,19,2,h,w)
    values[:,:,0] = 2300.25
    values[:,:,1] = 100.125
    mask = torch.ones_like(values)
    return {"obs_components":values, "obs_component_mask":mask,
            "ztd_stats":torch.tensor([[2320.239,169.43678]]),
            "obs_ztd":torch.zeros(1,19,1,h,w), "obs_mask":torch.ones(1,19,1,h,w)}

class ComponentUnitTest(unittest.TestCase):
    def test_physical_subtraction_before_projection(self):
        total = np.array([[[1., 2., np.nan]]],np.float32)
        wet = np.array([[[3., np.nan, 4.]]],np.float32)
        h,w = paired_components(total,wet,[2000,100],[100,10])
        self.assertEqual(h[0,0,0],1970)
        self.assertEqual(w[0,0,0],130)
        self.assertTrue(np.isnan(h[0,0,1:]).all())
        self.assertTrue(np.isnan(w[0,0,1:]).all())

    def test_zero_heads_persistence_sum_interpolation_and_no_total_loss(self):
        batch = component_batch()
        delta = torch.zeros(1,18,2,2,2,requires_grad=True)
        change = torch.zeros(1,2,2,2)
        change[:,0]=18.
        change[:,1]=36.
        out = component_prediction(delta,batch["obs_components"],batch["obs_component_mask"],
                                   [100,100],batch["ztd_stats"],change)
        torch.testing.assert_close(out["components"][0,:,0,0,0],
                                   2300.25+torch.arange(1,19))
        torch.testing.assert_close(out["components"][0,:,1,0,0],
                                   100.125+torch.arange(1,19)*2)
        total = out["ztd"]*batch["ztd_stats"][:,1,None,None,None,None]+batch["ztd_stats"][:,0,None,None,None,None]
        torch.testing.assert_close(total,out["components"].sum(2,keepdim=True))
        batch.update(component_target=out["components"].detach()+10,
                     component_target_mask=torch.ones_like(out["components"]),
                     component_loss_scales=torch.tensor([[100.,100.]]))
        loss = component_aux_loss(out,batch,{})
        torch.testing.assert_close(loss,torch.tensor(.02))
        altered={**out,"ztd":torch.full_like(out["ztd"],float("nan"))}
        torch.testing.assert_close(component_aux_loss(altered,batch,{}),loss)
        loss.backward()
        self.assertGreater(delta.grad.abs().sum(),0)

    def test_missing_history_and_padded_evaluation(self):
        batch=component_batch()
        batch["obs_component_mask"][...,1,1]=0
        batch["obs_component_mask"][:,-1,:,0,0]=0
        batch["obs_components"][:,-1,:,0,0]=float("nan")
        out=component_prediction(torch.zeros(1,18,2,2,2),batch["obs_components"],
                                 batch["obs_component_mask"],[100,100],batch["ztd_stats"])
        self.assertFalse(out["component_mask"][...,1,1].any())
        self.assertTrue(out["component_mask"][...,0,0].all())
        batch.update(component_target=torch.ones_like(out["components"]),
                     component_target_mask=torch.ones_like(out["components"]),
                     component_loss_scales=torch.tensor([[100.,100.]]))
        err,count=component_error_sums(out,batch,torch.tensor([False]))
        self.assertEqual(err.sum(),0)
        self.assertEqual(count.sum(),0)
        nxt=next_component_inputs(batch,out)
        self.assertFalse(nxt["obs_mask"][...,1,1][:,1:].any())
        self.assertFalse(nxt["obs_component_mask"][:,0,:,0,0].any())
        self.assertTrue(nxt["obs_component_mask"][:,1:,:,0,0].all())

    def test_checkpoint_parity_heads_freezing_bf16(self):
        old=mf.make_model().eval()
        model=tiny_model().eval()
        report=train.initialize_model_weights(model,{"model":old.state_dict(),"config":{"model":{}}},
                       {"predict_ztd":True,"predict_delay_components":True})
        self.assertEqual(len(report["new_component_tensors"]),6)
        batch=mf.make_inputs()
        batch.update(component_batch(40,40))
        output=model(**batch)
        old_inputs={k:v for k,v in batch.items() if k not in ("obs_components","obs_component_mask","ztd_stats")}
        torch.testing.assert_close(output["weather"],old(**old_inputs),rtol=0,atol=0)
        base,_=latest_components(batch["obs_components"],batch["obs_component_mask"])
        torch.testing.assert_close(output["components"],base.expand_as(output["components"]),rtol=0,atol=0)
        train.apply_freeze_policy(model,{"freeze_fuxi":True,"freeze_obs":True,"freeze_component_adapter":True})
        for name,p in model.named_parameters():
            self.assertEqual(p.requires_grad,name.startswith(("decoder.zhd_head.","decoder.zwd_head.")))
        output=model(**batch)  # Freeze before building the checkpointed autograd graph.
        output["components"].mean().backward()
        self.assertGreater(model.decoder.zhd_head.weight.grad.abs().sum(),0)
        self.assertGreater(model.decoder.zwd_head.weight.grad.abs().sum(),0)
        with torch.autocast("cpu",dtype=torch.bfloat16):
            mixed=model(**batch)
        torch.testing.assert_close(mixed["components"],base.expand_as(output["components"]),rtol=0,atol=0)
        broken=dict(old.state_dict())
        del broken["decoder.head.weight"]
        with self.assertRaisesRegex(ValueError,"ONLY"):
            train.initialize_model_weights(tiny_model(),{"model":broken,"config":{"model":{}}},
                          {"predict_ztd":True,"predict_delay_components":True})

    def test_fsdp_preserves_physical_input_precision(self):
        model=tiny_model()
        with patch.object(train, "FSDP") as wrapper, patch.object(train.torch.cuda, "is_bf16_supported", return_value=True):
            train.wrap_fsdp(model, torch.device("cuda",0), {"amp":"bf16"})
            precision=wrapper.call_args.kwargs["mixed_precision"]
            self.assertFalse(precision.cast_forward_inputs)
            self.assertFalse(precision.cast_root_forward_inputs)
            self.assertEqual(precision.param_dtype,torch.bfloat16)
        batch=component_batch()
        with self.assertRaisesRegex(ValueError,"FP32"):
            component_prediction(torch.zeros(1,18,2,2,2),batch["obs_components"].bfloat16(),
                                 batch["obs_component_mask"],[100,100],batch["ztd_stats"])

    def test_multihop_uses_predictions_only(self):
        batch=component_batch()
        batch.update(era5=torch.zeros(1,2,1,2,2),era_static=torch.zeros(1,5,2,2),
                     obs_static=torch.ones(1,3,2,2),time_features=torch.zeros(1,12),
                     component_target=torch.full((1,18,2,2,2),99999.))
        def model(**inputs):
            self.assertNotIn("component_target",inputs)
            return {"weather":inputs["era5"][:,-1]+1,
                    **component_prediction(torch.ones(1,18,2,2,2),inputs["obs_components"],
                                           inputs["obs_component_mask"],[1,2],inputs["ztd_stats"])}
        outputs=list(forecast_rollout(model,batch,[torch.zeros(1,12)]*4))
        for k,out in enumerate(outputs):
            self.assertEqual(out["components"][0,-1,0,0,0],2300.25+k+1)
            self.assertEqual(out["components"][0,-1,1,0,0],100.125+2*(k+1))

class PhysicsTest(unittest.TestCase):
    def test_hydrostatic_zero_humidity_gradient(self):
        t=torch.full((2,13),280.,requires_grad=True)
        rh=torch.zeros(2,13)
        ts=torch.tensor([280.,280.])
        p=torch.tensor([100000.,100000.],requires_grad=True)
        out=profile_delays_mm(t,rh,ts,p,torch.tensor([0.,1000.]),torch.tensor([45.,45.]))
        self.assertAlmostEqual(out[0,0].item(),2276.8,places=2)
        self.assertTrue(out[:,1].eq(0).all())
        self.assertLess(out[1,0],out[0,0])
        out.sum().backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(p.grad.abs().sum(),0)

    def test_wet_belowground_levels_do_not_contribute(self):
        t=torch.full((1,13),280.,requires_grad=True)
        rh=torch.full((1,13),50.)
        args=(torch.tensor([280.]),torch.tensor([100000.]),torch.tensor([2000.]),torch.tensor([45.]))
        out=profile_delays_mm(t,rh,*args)
        self.assertGreater(out[0,1],0)
        changed=t.detach().clone()
        changed[:,-3:]=330. # 850,925,1000 hPa all below surface ~783 hPa
        torch.testing.assert_close(profile_delays_mm(changed,rh,*args),out)
        out.sum().backward()
        self.assertTrue(torch.isfinite(t.grad).all())
        self.assertGreater(t.grad.abs().sum(),0)

    def test_model_physics_feedback_can_be_detached(self):
        model=DAForecastNet(image_size=(40,40),embed_dim=32,patch_size=4,window_size=5,
                            depths=(1,),num_heads=4,predict_ztd=True,predict_delay_components=True,
                            use_delay_physics=True)
        batch=component_batch(40,40)
        mean=torch.zeros(1,70); mean[:,13:26]=280;mean[:,52:65]=50;mean[:,65]=280;mean[:,68]=100000
        batch.update(era5=torch.randn(1,2,70,40,40)*.01,era_static=torch.zeros(1,5,40,40),
                     time_features=torch.zeros(1,12),obs_static=torch.ones(1,3,40,40),
                     station_height=torch.zeros(1,40,40),station_latitude=torch.full((1,40,40),45.),
                     weather_mean=mean,weather_std=torch.ones(1,70))
        model(**batch)["components"].sum().backward()
        self.assertIsNone(model.decoder.head.weight.grad)
        model.zero_grad(set_to_none=True)
        model.delay_physics_detach=False
        result=model(**batch)
        result["components"].sum().backward()
        self.assertTrue(torch.isfinite(model.decoder.head.weight.grad).all())
        self.assertGreater(model.decoder.head.weight.grad.abs().sum(),0)

    def test_sparse_operator(self):
        x=torch.zeros(1,70,2,2)
        x[:,13:26]=280
        x[:,52:65]=50
        x[:,65]=280
        x[:,68]=100000
        x.requires_grad_()
        static=torch.zeros(1,3,2,2)
        static[:,0,0,0]=1
        h=torch.zeros(1,2,2);lat=torch.full_like(h,45.)
        out=weather_delays_mm(x,static,h,lat,torch.zeros(1,70),torch.ones(1,70))
        direct=profile_delays_mm(x[0,13:26,0,0],x[0,52:65,0,0],x[0,65,0,0],x[0,68,0,0],h[0,0,0],lat[0,0,0])
        torch.testing.assert_close(out[0,:,0,0],direct)
        self.assertFalse(out[...,1,1].any())
        out.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        h[0,0,0]=float("nan")
        with self.assertRaisesRegex(ValueError,"heights"):
            weather_delays_mm(x,static,h,lat,torch.zeros(1,70),torch.ones(1,70))

class TinyComponents(rf.TinyForecastCore):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.predict_delay_components=True
        self.component_adapter=torch.nn.Linear(1,1)
        torch.nn.init.zeros_(self.component_adapter.weight)
        torch.nn.init.zeros_(self.component_adapter.bias)
        for name in ("zhd_head","zwd_head"):
            layer=torch.nn.Linear(1,18)
            torch.nn.init.zeros_(layer.weight);torch.nn.init.zeros_(layer.bias)
            setattr(self.decoder,name,layer)
    def forward(self,era,static,time,obs,mask,obs_static,**extra):
        assert self.use_obs
        core=sum(getattr(self,name).weight.sum() for name in ("era_encoder","time_embed","backbone","decoder"))
        x=era[:,0,0,0,0,None]*.001+1
        delta=torch.stack([getattr(self.decoder,name)(x) for name in ("zhd_head","zwd_head")],2)
        delta=delta[:,:,:,None,None].expand(-1,-1,-1,*era.shape[-2:])
        return {"weather":era[:,-1]+core,
                **component_prediction(delta,extra["obs_components"],extra["obs_component_mask"],
                                       [100,100],extra["ztd_stats"])}

class ComponentPipelineTest(unittest.TestCase):
    def setUp(self):
        rf.TwoFieldTest.setUp(self)
        group=zarr.open_group(self.data["ztd_path"],mode="r+")
        group.attrs.update(ztd_train_mean=2000.,ztd_train_std=100.,zwd_train_mean=100.,zwd_train_std=10.)
        group["ztd"].attrs.update(raw_units="mm",units="1")
        wet=np.full(group["ztd"].shape,np.nan,np.float32)
        wet[:,0,0]=np.arange(len(wet))/1000.
        arr=rf.array(group,"zwd",wet);arr.attrs.update(raw_units="mm",units="1")
        self.data.update(predict_ztd=True,predict_delay_components=True,
                         component_loss_scales=[100,100],use_delay_physics=False,
                         rollout_observations="predicted_components_at_T")
        self.model_config.update(predict_ztd=True,predict_delay_components=True,
                                 component_loss_scales=[100,100],use_delay_physics=False)
        self.cfg={"data":self.data,"model":self.model_config,
                  "train":{"seed":4,"epochs":1,"batch_size_per_gpu":2,"gradient_accumulation":1,
                           "num_workers":0,"learning_rate":.001,"min_learning_rate":.0001,
                           "weight_decay":0.,"warmup_steps":0,"gradient_clip":1.,"amp":"fp32",
                           "freeze_fuxi":True,"freeze_obs":True,"freeze_component_adapter":True,
                           "loss_mode":"global_only","halo_weight":.5,"halo_radius":0,
                           "zhd_loss_weight":.1,"zwd_loss_weight":.1,"checkpoint_metric":"components",
                           "validate_before_training":True,"fsdp":{"enabled":False},
                           "output_dir":str(self.root/"dual_train")}}

    def test_labels_and_no_future_leakage(self):
        ds=builder.source_datasets(self.data)["train"]
        first=ds[0]
        self.assertNotIn("component_target",first)
        target,mask=read_components(ds,ds.sample_time(0),future=True)
        self.assertTrue(mask[:,:,0,0].all())
        self.assertFalse(mask[...,1,1].any())
        np.testing.assert_allclose(target.sum(1)[:,0,0],2700,atol=.001)
        obs=zarr.open_group(self.data["ztd_path"],mode="r+")
        cut=ds.obs_times.searchsorted(ds.sample_time(0)+pd.Timedelta(minutes=5))
        obs["ztd"][int(cut):]=np.nan;obs["zwd"][int(cut):]=np.nan
        changed=ds[0]
        for k in first:
            torch.testing.assert_close(first[k],changed[k],rtol=0,atol=0)
        self.assertFalse(read_components(ds,ds.sample_time(0),future=True)[1].any())

    def test_train6_build12_train12_eval_resume(self):
        # Real CLI orchestration with tiny CPU model; actual dataset/loss/Zarr/metrics.
        cfg6=json.loads(json.dumps(self.cfg))
        cfg6["data"]={k:v for k,v in self.data.items() if not k.startswith("rollout_")}
        cfg6["data"]["forecast_lead_hours"]=6
        cfg6["train"]["output_dir"]=str(self.root/"six_train")
        path6=self.root/"six.json";path6.write_text(json.dumps(cfg6))
        with patch.object(train,"DAForecastNet",TinyComponents), patch.object(train,"setup_distributed",return_value=(0,0,1,torch.device("cpu"))), patch("sys.argv",["train.py","--config",str(path6)]), redirect_stdout(io.StringIO()):
            train.main()
        self.data["rollout_source_checkpoint"]=str(self.root/"six_train"/"best.pt")
        path=self.root/"dual.json";path.write_text(json.dumps(self.cfg))
        with patch.object(builder,"DAForecastNet",TinyComponents), patch.object(builder,"setup_distributed",return_value=(0,0,1,torch.device("cpu"))), patch("sys.argv",["build_rollout_zarr.py","--config",str(path),"--amp","fp32","--num-workers","0"]), redirect_stdout(io.StringIO()):
            builder.main()
        ds=train.make_dataset(self.data,"train")
        sample=ds[0]
        self.assertNotIn("ztd_target",sample)
        self.assertEqual(sample["obs_components"].shape,(19,2,2,2))
        total=sample["obs_ztd"]*100+2000
        torch.testing.assert_close(total[1:,0,0,0],sample["obs_components"][1:,:,0,0].sum(1))
        root=open_store(self.data["rollout_zarr"])
        self.assertNotIn("ztd_target",root)
        self.assertEqual(root["component_input"].shape,(18,19,2,1))
        with patch.object(train,"DAForecastNet",TinyComponents), patch.object(train,"setup_distributed",return_value=(0,0,1,torch.device("cpu"))), patch("sys.argv",["train.py","--config",str(path),"--init-checkpoint",self.data["rollout_source_checkpoint"]]), redirect_stdout(io.StringIO()):
            train.main()
        out=Path(self.cfg["train"]["output_dir"])
        reports=[]
        for bs in (1,3):
            dest=out/f"test{bs}"
            argv=["evaluate.py","--config",str(path),"--checkpoint",str(out/"best.pt"),"--output-dir",str(dest),
                  "--batch-size",str(bs),"--num-workers","0","--device","cpu","--skip-animations","--skip-mae-plots"]
            with patch.object(evaluate,"DAForecastNet",TinyComponents),patch("sys.argv",argv),redirect_stdout(io.StringIO()):
                evaluate.main()
            reports.append(json.loads((dest/"delay_mae_summary.json").read_text()))
            self.assertFalse((dest/"ztd_mae_summary.json").exists())
        for name in ("zhd","zwd"):
            self.assertEqual(reports[0]["components"][name]["valid_count"],6*18)
            self.assertAlmostEqual(reports[0]["components"][name]["mae_mm"],reports[1]["components"][name]["mae_mm"],places=4)
        self.cfg["train"]["epochs"]=2;path.write_text(json.dumps(self.cfg))
        with patch.object(train,"DAForecastNet",TinyComponents),patch.object(train,"setup_distributed",return_value=(0,0,1,torch.device("cpu"))),patch("sys.argv",["train.py","--config",str(path),"--resume",str(out/"last.pt")]),redirect_stdout(io.StringIO()):
            train.main()
        self.assertEqual(torch.load(out/"last.pt",weights_only=False)["epoch"],1)

    def test_colliding_cells_use_identical_valid_pairs(self):
        group=zarr.open_group(self.data["ztd_path"],mode="r+")
        group["mask"][:]=np.array([[False,False],[True,True]])
        group["ztd"][:,0,1]=1000.  # second station mapped to same ERA cell; no valid ZWD
        ds=builder.source_datasets(self.data)["train"]
        sample=ds[0]
        summed=sample["obs_components"][:,:,0,0].sum(1)
        torch.testing.assert_close(summed,torch.full((19,),2700.),rtol=0,atol=.001)

    def test_physics_requires_height_not_zero_fallback(self):
        data6={k:v for k,v in self.data.items() if not k.startswith("rollout_")}
        data6.update(forecast_lead_hours=6,use_delay_physics=True)
        with self.assertRaisesRegex(ValueError,"station_height"):
            train.make_dataset(data6,"train")

if __name__=="__main__":
    unittest.main()
