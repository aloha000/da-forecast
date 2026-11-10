"""Draw current predict_ztd vs main weather-only rollout from their actual configs.

Uses Matplotlib, meta-device models (no weights/GPU), and git show main:...
Leaves the old FuXi-vs-DA diagram untouched.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import torch

ROOT = Path(__file__).resolve().parent
C = {"core":"#e5eefb", "obs":"#dff1e9", "new":"#ffebd2", "input":"#eff2f6",
     "out":"#eae3f6", "text":"#183047", "edge":"#4d637b",
     "green":"#2d8169", "orange":"#c67a29", "purple":"#8060a9"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.unicode_minus":False})


def git(path):
    return subprocess.check_output(["git","show","main:"+path],cwd=ROOT,text=True)


def inspect():
    from model.da_forecast_model import DAForecastNet
    current_source = (ROOT/"model/da_forecast_model.py").read_text()
    main_source = git("model/da_forecast_model.py")
    current = json.loads((ROOT/"configs/train_rollout_48h_predict_delays.json").read_text())
    main = json.loads(git("configs/train_rollout_48h_zhd_zwd.json"))
    for cfg in (current,main):
        m=cfg["model"]
        for key,value in {"image_size":[720,1440],"embed_dim":1536,"patch_size":4,
                          "depths":[12]*4,"num_heads":24,"window_size":9,
                          "era5_frames":2,"era5_channels":70}.items():
            assert m[key]==value, f"Review drawing for changed {key}"
        assert not cfg["train"]["freeze_fuxi"] and not cfg["train"]["freeze_obs"]
    assert current["model"]["predict_delay_components"]
    assert not current["model"]["use_delay_physics"]
    assert main["model"]["obs_channels"]==2
    # Verify the shared blocks really are identical, rather than just similarly named.
    def classes(source):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body
                if isinstance(n,ast.ClassDef)}
    a,b=classes(current_source),classes(main_source)
    for name in ("ERA5Encoder","GatedFusion","ForecastBackbone"):
        assert a[name]==b[name], f"{name} changed: review shared-block annotation"
    assert (ROOT/"model/fuxi.py").read_text()==git("model/fuxi.py"), "FuXi core changed"
    ns=types.ModuleType("model._main_diagram_snapshot")
    ns.__package__="model"
    exec(compile(main_source,"main:model/da_forecast_model.py","exec"),ns.__dict__)
    with torch.device("meta"):
        models=(ns.DAForecastNet(**main["model"]),DAForecastNet(**current["model"]))
    totals=[sum(p.numel() for p in m.parameters()) for m in models]
    extras={name:sum(p.numel() for p in models[1].get_submodule(name).parameters())
            for name in ("component_adapter","decoder.zhd_head","decoder.zwd_head")}
    first_conv=[m.obs_encoder.temporal[0].weight.numel() for m in models]
    return {"main_commit":subprocess.check_output(["git","rev-parse","main"],cwd=ROOT,text=True).strip(),
        "current_branch":subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),
        "current_model_sha256":hashlib.sha256(current_source.encode()).hexdigest(),
        "main_parameters":totals[0],"current_parameters":totals[1],
        "net_added_parameters":totals[1]-totals[0],"current_added_modules":extras,
        "obs_first_conv_parameter_difference":first_conv[1]-first_conv[0],
        "configs":["main:configs/train_rollout_48h_zhd_zwd.json",
                   "configs/train_rollout_48h_predict_delays.json"],
        "note":"Counts are nn.Parameter only, not buffers. Both configs unfreeze all active modules."}


def text(ax,x,y,s,size=10,weight="normal",color="text",ha="center",**kwargs):
    return ax.text(x,y,s,fontsize=size,fontweight=weight,color=C.get(color,color),
                   ha=ha,va="center",linespacing=1.32,zorder=5,**kwargs)


def box(ax,x,y,w,h,title,body="",color="core",size=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.025,rounding_size=.13",
                               facecolor=C[color],edgecolor="#a7b6c5",lw=1.1,zorder=3))
    if body:
        text(ax,x+w/2,y+h*.76,title,size+1,"bold")
        text(ax,x+w/2,y+h*.35,body,size)
    else:
        text(ax,x+w/2,y+h/2,title,size,"bold")


def arrow(ax,pts,color="edge",dash=False):
    if len(pts)>2:
        xx,yy=zip(*pts[:-1]);ax.plot(xx,yy,color=C[color],lw=1.4,
                                   ls="--" if dash else "-",zorder=1)
    ax.add_patch(FancyArrowPatch(pts[-2],pts[-1],arrowstyle="-|>",mutation_scale=11,
                                lw=1.5,color=C[color],linestyle="--" if dash else "-",zorder=2))


def network(ax,current,meta):
    ax.set(xlim=(0,17.3),ylim=(0,36));ax.axis("off")
    title="predict_ztd  |  Weather + delay prediction" if current else "main  |  Weather-only prediction"
    count=meta["current_parameters" if current else "main_parameters"]
    text(ax,8.5,35.45,title,17,"bold")
    text(ax,8.5,34.7,f"{count:,} parameters  |  same shared model at every lead  |  all unfrozen",10)
    x,w,cx=6.1,6.1,9.15
    box(ax,x,31.85,w,1.7,"Weather + FuXi static fields",
        "2 weather frames + 5 static channels per frame\n[B,2,75,720,1440] -> [B,150,720,1440]","input")
    box(ax,x,29.45,w,1.65,"ERA encoder: CubeEmbed",
        "Conv2d k4/s4 + LayerNorm (time folded into channels)\nhigh = [B,1536,180,360]")
    box(ax,x,27.15,w,1.55,"high_fusion  |  gated feature correction",
        "high + available * sigmoid(gate_logit) * correction","obs",9.5)
    box(ax,x,23.85,w,1.65,"ERA encoder: DownBlock2D",
        "time-conditioned residual downsampling /2\nlow = [B,1536,90,180]")
    box(ax,x,21.55,w,1.55,"low_fusion  |  gated feature correction",
        "low + available * sigmoid(gate_logit) * correction","obs",9.5)
    box(ax,x,18.85,w,2.0,"Swin Transformer  |  backbone.layers",
        "4 stages x 12 blocks = 48 blocks; 24 attention heads\n1536 channels; 9 x 9 windows; periodic longitude\nOnly the low grid (90 x 180) enters Transformer")
    box(ax,x,16.6,w,1.5,"Multi-stage FPN",
        "LayerNorm output of each of the 4 stages\nconcat 6144 -> Linear 1536 -> GELU")
    box(ax,x,14.3,w,1.6,"Shared decoder: UpBlock2D",
        "concat(FPN, fused-low skip) -> time-conditioned x2\n[B,1536,180,360]")
    box(ax,x,11.8,w,1.65,"Weather head  |  decoder.head",
        "Linear 1536 -> 70 x 4 x 4 -> patch reconstruction\nweather increment [B,70,720,1440]")
    box(ax,x,9.35,w,1.65,"Weather forecast P(t+6h)",
        "last weather input + predicted increment\n[B,70,720,1440]","out")
    for top,bottom in ((31.85,31.10),(29.45,28.70),(23.85,23.10),
                       (21.55,20.85),(18.85,18.1),(16.6,15.9),
                       (14.3,13.45),(11.8,11.0)):
        arrow(ax,[(cx,top),(cx,bottom)])

    obs_title="Normalized ZTD + validity mask" if current else "Normalized ZHD / ZWD + paired mask"
    obs_shape="[B,19,1,720,1440]" if current else "[B,19,2,720,1440]"
    box(ax,.25,31.85,5.1,1.7,obs_title,
        f"T-6h ... T, every 20 min\n{obs_shape}","input",9.3)
    first="2 -> 32 -> 64" if current else "3 -> 32 -> 64"
    box(ax,.25,21.55,5.1,9.55,"ObsEncoder  |  obs_encoder",
        "Conv3d + SiLU, twice\n"
        f"input channels: {first}\n"
        "(values + one validity channel)\n\n"
        "Mask-weighted temporal pooling -> 64 channels\n"
        "+ 3 station static fields: station mask, lat, lon\n\n"
        "Conv2d k4/s4: 67 -> 1536 + ResBlock\n"
        "obs_high: [B,1536,180,360]\n"
        "high patch-mask applied\n\n"
        "Conv2d s2 + ResBlock\n"
        "obs_low: [B,1536,90,180]\n"
        "low patch-mask applied","obs",9.8)
    arrow(ax,[(2.8,31.85),(2.8,31.1)],"green")
    arrow(ax,[(5.35,27.9),(6.1,27.9)],"green")
    arrow(ax,[(5.35,22.3),(6.1,22.3)],"green")

    # Time conditioning and ONLY low-scale decoder skip.
    box(ax,.25,17.85,5.1,2.0,"Calendar conditioning",
        "12 calendar features at each step\ntime_embed: Linear 12 -> 1536 -> 1536\nFeeds DownBlock2D and UpBlock2D","input")
    arrow(ax,[(5.35,19.1),(5.65,19.1),(5.65,24.65),(6.1,24.65)],dash=True)
    arrow(ax,[(5.35,18.3),(5.65,18.3),(5.65,15.1),(6.1,15.1)],dash=True)
    arrow(ax,[(cx,21.55),(cx,21.25),(12.55,21.25),(12.55,15.1),(12.2,15.1)],"purple",True)
    text(ax,12.74,18.0,"fused-low skip (1/8)",8.8,color="purple",rotation=90)

    if current:
        box(ax,13.25,31.85,3.55,1.7,"ZHD / ZWD in mm",
            "[B,19,2,720,1440]\n+ component validity masks","input",9.2)
        box(ax,13.25,26.75,3.55,4.1,"Component adapter",
            "Last valid (H,W) / 1000\nHistory mean (H,W) / 1000\nAvailability (H,W)\n= 6 spatial channels\n\nConv2d k4/s4: 6 -> 1536\nApply patch-availability mask","new",9.3)
        arrow(ax,[(15.025,31.85),(15.025,30.85)],"orange")
        ax.add_patch(Circle((cx,26.1),.23,color=C["new"],ec=C["orange"],zorder=4))
        text(ax,cx,26.1,"+",15,"bold")
        arrow(ax,[(cx,27.15),(cx,26.33)])
        arrow(ax,[(cx,25.87),(cx,25.5)])
        arrow(ax,[(15.025,26.75),(15.025,26.1),(cx+.23,26.1)],"orange")
        box(ax,13.25,11.8,3.55,1.65,"ZHD head + ZWD head",
            "Two Linear: 1536 -> 18 x 4 x 4\nReconstruct 18 residual frames","new",9.0)
        arrow(ax,[(cx,14.05),(15.025,14.05),(15.025,13.45)],"orange")
        box(ax,13.25,9.35,3.55,1.65,"18-frame delay forecasts",
            "last valid H/W + 100 mm x residual\n[B,18,2,720,1440]  (FP32)","out",8.7)
        arrow(ax,[(15.025,11.8),(15.025,11)],"orange")
        arrow(ax,[(16.8,32.7),(17.08,32.7),(17.08,10.15),(16.8,10.15)],"orange",True)
        text(ax,16.99,19.2,"last valid physical H/W: residual baseline",8.5,
             color="orange",rotation=90,ha="right")
        box(ax,.25,11.0,5.1,5.8,"What changes vs main",
            "Original single-ZTD encoder is retained.\n"
            "H/W adapter is added AFTER high_fusion,\n"
            "BEFORE the ERA down block.\n\n"
            "All 3 heads share the same upsampled features.\n"
            "ZTD = predicted ZHD + predicted ZWD,\n"
            "then original ZTD normalization is applied.\n\n"
            "No independent ZTD head or ZTD loss.\n"
            "Optional physics operator: OFF in this config.","new",9.4)
    else:
        arrow(ax,[(cx,27.15),(cx,25.5)])
        box(ax,13.25,25.1,3.55,5.4,"No delay prediction",
            "ZHD/ZWD directly enter\nthe 2-value ObsEncoder.\n\n"
            "No component adapter.\nNo ZHD/ZWD heads.\nNo residual delay baseline.\n\n"
            "Only the weather field\nis predicted.","input",9.3)
        box(ax,.25,11.0,5.1,5.8,"Main-branch details",
            "Both H/W inputs use a shared ZTD scale.\n"
            "First Conv3d sees [H,W,paired mask].\n\n"
            "Only the origin step executes OBS/fusions.\n"
            "Later steps bypass BOTH fusion modules\n"
            "and the ObsEncoder.\n\n"
            "Input-zero ablation has the same architecture:\n"
            "values set to 0; masks/static fields retained.","input",9.4)

    # The autoregressive execution differs even when weather-core shapes agree.
    text(ax,8.5,8.25,"ROLLOUT EXECUTION  |  one set of weights, repeatedly applied",12,"bold")
    xs=[.25,4.6,8.95,13.3];wid=3.45
    if current:
        names=["6h: P6 + D1","12h: P12 + D2","... 42h: P42 + D7","48h: P48 + D8"]
        notes=["real origin history","predicted history","predicted history","predicted history"]
    else:
        names=["6h: P6","12h: P12","... 42h: P42","48h: P48"]
        notes=["OBS + gates ON","OBS + gates bypassed","OBS + gates bypassed","OBS + gates bypassed"]
    for j,(xx,note,name) in enumerate(zip(xs,notes,names)):
        box(ax,xx,6.15,wid,1.35,name,note,"out" if current else "core",9.2)
        if j<3: arrow(ax,[(xx+wid,6.82),(xs[j+1],6.82)],"orange" if current else "edge")
    detail=("Weather history: shift + P6/P12/...; delay history: last frame + 18 predicted frames.\n"
            "Dk = 18 x (ZHD,ZWD); derive normalized ZTD for the next step. No future truth as input."
            if current else
            "Weather history: [ERA(T-6),ERA(T)] -> [ERA(T),P6] -> [P6,P12] -> ...\n"
            "No observed or predicted delay inputs after the first step.")
    text(ax,8.5,5.15,detail,10)
    loss=("Per hop: weather MAE + 0.1 x ZHD_MAE_mm/100 + 0.1 x ZWD_MAE_mm/100"
          if current else "Per hop: weather global latitude-weighted MAE only")
    box(ax,.25,2.15,16.5,1.95,"TRAINING: average loss over all active hops; full BPTT (no detach)",
        loss+"\n6h -> 12h -> ... -> 48h curriculum; 500 optimizer updates per stage","input",10)
    text(ax,8.5,1.18,"Blue = shared FuXi weather path   |   Green = OBS/gated fusion   |   Orange = added component path",9)
    text(ax,8.5,.58,"Solid arrows: forward features / predictions. Dashed: time, low skip, or residual baseline. No direct high-to-decoder skip.",8.6)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=ROOT/"network_predict_delays_vs_main.png")
    p.add_argument("--dpi",type=int,default=190)
    args=p.parse_args()
    info=inspect()
    fig,axes=plt.subplots(1,2,figsize=(28,27),facecolor="white")
    network(axes[0],False,info);network(axes[1],True,info)
    fig.suptitle("DA-Forecast: main vs predict_ztd  |  Global 48h autoregressive forecasting",
                 fontsize=23,fontweight="bold",color=C["text"],y=.985)
    fig.text(.5,.96,
        f"Current minus main: +{info['net_added_parameters']:,} parameters "
        f"(adapter + two heads - one ObsEncoder input channel). "
        "These colors do not denote frozen modules.",
        ha="center",fontsize=12,color=C["text"])
    fig.subplots_adjust(left=.025,right=.975,bottom=.015,top=.94,wspace=.045)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=args.dpi,facecolor="white")
    # Small preview for legibility/layout review; keep high-res PNG as deliverable.
    fig.savefig(args.output.with_name(args.output.stem+"_preview.png"),dpi=80,facecolor="white")
    plt.close(fig)
    args.output.with_suffix(".json").write_text(json.dumps(info,indent=2)+"\n")
    print(json.dumps(info,indent=2))
    print("saved",args.output)

if __name__=="__main__":
    main()
