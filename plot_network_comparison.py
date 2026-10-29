"""Code-aligned architecture and conditional freeze map; counts use meta models.

No checkpoint loading, GPU use, or multi-GB parameter allocation is needed.
The drawing shows freeze_fuxi=True, with the actual config flag separately.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent
COLORS = dict(frozen="#DCEBFA", new="#DCF1DF", input="#EEF0F5",
              output="#FFF0CC", note="#F5F6F8", text="#172B42", edge="#52677F",
              obs_edge="#358650", time="#758296", skip="#8861A8", warning="#FFF0D9")
GROUPS = (
    ("era_encoder.patch_embed", "decoder.patch_embed.*"),
    ("era_encoder.down_block", "decoder.down_blocks.0.*"),
    ("time_embed", "decoder.time_embed.*"),
    ("backbone.layers", "decoder.layers.*"),
    ("backbone.norms", "decoder.norm0..3.*"),
    ("backbone.fpn", "decoder.fpn.*"),
    ("decoder.up_block", "decoder.up_blocks.0.*"),
    ("decoder.head", "decoder.head.*"),
    ("obs_encoder", "Not in original FuXi"),
    ("high_fusion", "Not in original FuXi"),
    ("low_fusion", "Not in original FuXi"),
)
PARAMETER_SCOPE = {
    "era_encoder.patch_embed": "Conv2d W,b + LayerNorm affine",
    "era_encoder.down_block": "Residual/downsample/time/norm params",
    "time_embed": "Both Linear weights and biases",
    "backbone.layers": "Attention/MLP/norm + logit_scale",
    "backbone.norms": "Four LayerNorm weights and biases",
    "backbone.fpn": "Linear weight and bias (GELU: none)",
    "decoder.up_block": "Residual/upsample/time/norm params",
    "decoder.head": "Linear weight and bias",
    "obs_encoder": "All conv and residual/norm parameters",
    "high_fusion": "Both convs W,b + 1,536 gate logits",
    "low_fusion": "Both convs W,b + 1,536 gate logits",
}


def freeze_roots_from_training():
    """Extract the module tuple actually frozen by train.py."""
    matches = []
    for node in ast.walk(ast.parse((ROOT / "train.py").read_text())):
        if not isinstance(node, ast.For) or not isinstance(node.iter, ast.Tuple):
            continue
        elements = node.iter.elts
        if not elements or not all(isinstance(e, ast.Attribute)
                and isinstance(e.value, ast.Name) and e.value.id == "model" for e in elements):
            continue
        if any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
               and c.func.attr == "requires_grad_" and c.args
               and isinstance(c.args[0], ast.Constant) and c.args[0].value is False
               for c in ast.walk(node)):
            matches.append(tuple(e.attr for e in elements))
    if len(matches) != 1:
        raise ValueError("train.py freezing logic changed; review the figure")
    return matches[0]


def inspect_models(config_path):
    import torch
    from model.da_forecast_model import DAForecastNet
    from model.fuxi import UTransformer
    from model.fuxi_pretrained import rename_fuxi_key

    config = json.loads(Path(config_path).read_text())
    cfg = config["model"]
    required = dict(image_size=[720,1440], era5_channels=70, era5_frames=2,
                    era_static_channels=5, obs_static_channels=3, out_channels=70,
                    embed_dim=1536, patch_size=4, window_size=9,
                    depths=[12,12,12,12], num_heads=24, use_obs=True)
    for key, value in required.items():
        if cfg.get(key) != value:
            raise ValueError(f"Layout expects {key}={value!r}; review layout for {cfg.get(key)!r}")
    with torch.device("meta"):
        da = DAForecastNet(**cfg)
        original = UTransformer(in_chans=75, out_chans=70, in_frames=2,
            image_size=(720,1440), window_size=9, patch_size=4, down_times=1,
            embed_dim=1536, num_heads=24, depths=[12]*4, mlp_ratio=cfg.get("mlp_ratio",4))
    roots = freeze_roots_from_training()
    target = dict(da.named_parameters())
    covered, groups = set(), []
    for name, source in GROUPS:
        names = {key for key in target if key.startswith(name + ".")}
        if not names or covered.intersection(names):
            raise ValueError(f"Empty/overlapping parameter group: {name}")
        covered.update(names)
        groups.append(dict(name=name, source=source,
            parameters=sum(target[key].numel() for key in names), frozen=name.split(".")[0] in roots))
    if covered != set(target):
        raise ValueError(f"Unplotted parameters: {set(target)-covered}")
    mapped = set()
    for key, parameter in original.named_parameters():
        name = rename_fuxi_key("decoder." + key)
        if name not in target or target[name].shape != parameter.shape:
            raise ValueError(f"Original FuXi mapping mismatch: {key} -> {name}")
        mapped.add(name)
    if mapped != {key for key in target if key.split(".")[0] in roots}:
        raise ValueError("Pretrained mapping and freeze scope differ; review diagram")
    for root in roots:
        da.get_submodule(root).requires_grad_(False)
    total = sum(p.numel() for p in da.parameters())
    trainable = sum(p.numel() for p in da.parameters() if p.requires_grad)
    minutes = int(config["data"]["obs_history_hours"]) * 60
    interval = int(config["data"]["obs_interval_minutes"])
    if minutes <= 0 or interval <= 0 or minutes % interval:
        raise ValueError("Invalid observation interval")
    return dict(groups=groups, frozen_roots=roots, total=total, frozen=total-trainable,
        trainable=trainable, config_freeze=bool(config["train"].get("freeze_fuxi",False)),
        config_path=str(Path(config_path).resolve()), obs_hours=minutes//60,
        obs_interval=interval, obs_frames=minutes//interval+1)


def box(ax,x,y,w,h,text,color="note",fs=9):
    patch=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.025,rounding_size=0.10",
        facecolor=COLORS.get(color,color),edgecolor=COLORS["edge"],linewidth=1.1,zorder=3)
    ax.add_patch(patch)
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,
            color=COLORS["text"],linespacing=1.3,zorder=4)
    return patch


def label(ax,x,y,text,fs=8.5,color="text",**kwargs):
    return ax.text(x,y,text,ha="center",va="center",fontsize=fs,
                   color=COLORS.get(color,color),**kwargs)


def arrow(ax,start,end,color="edge",dashed=False):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=11,
        color=COLORS[color],linewidth=1.35,linestyle="--" if dashed else "-",zorder=2))


def route(ax,points,color="edge",dashed=False):
    if len(points)>2:
        xs,ys=zip(*points[:-1])
        ax.plot(xs,ys,color=COLORS[color],lw=1.25,linestyle="--" if dashed else "-",zorder=1)
    arrow(ax,points[-2],points[-1],color,dashed)


def module_box(ax,a,x,y,w,h,names,title,details,source=False,fs=9):
    names=(names,) if isinstance(names,str) else names
    selected=[g for g in a["groups"] if g["name"] in names]
    count=sum(g["parameters"] for g in selected)
    frozen=all(g["frozen"] for g in selected)
    status="PRETRAINED SOURCE" if source else ("FROZEN*" if frozen else "TRAINABLE*")
    return box(ax,x,y,w,h,f"{title}\n{details}\n{status} | {count/1e6:.3f}M parameters",
               "frozen" if frozen else "new",fs)


def draw_current(ax,a):
    ax.set(xlim=(0,12.3),ylim=(0,23.3)); ax.axis("off")
    label(ax,6.1,22.95,"DA-Forecast: implemented model",16,fontweight="bold")
    label(ax,6.1,22.42,"Blue: pretrained + frozen*    Green: newly initialized + trainable*",9.5)
    x,w,cx=4.75,4.05,6.775
    box(ax,x,20.9,w,1.05,"ERA5(T-6h,T) + 5 FuXi static fields\n[B,2,75,720,1440] -> [B,150,720,1440]","input")
    stages=[
        (19.,1.3,"era_encoder.patch_embed","era_encoder.patch_embed = CubeEmbed",
         "Conv2d k4/s4 -> flatten -> LayerNorm -> reshape\nhigh: [B,1536,180,360]"),
        (17.1,1.3,"high_fusion","high_fusion",
         "concat(ERA high, obs_high, high_mask) -> 2 convs\nERA high + available * sigmoid(gate_logit) * delta"),
        (15.3,1.2,"era_encoder.down_block","era_encoder.down_block",
         "time-conditioned residual downsampling /2\nlow: [B,1536,90,180]"),
        (13.5,1.2,"low_fusion","low_fusion",
         "concat(ERA low, obs_low, low_mask) -> 2 convs\nERA low + available * sigmoid(gate_logit) * delta"),
        (11.1,1.75,"backbone.layers","backbone.layers: Swin Transformer",
         "4 stages x 12 blocks = 48 blocks\n1536 channels; 24 heads; windows 9 x 9\n90 x 180 grid; periodic longitude"),
        (9.15,1.3,("backbone.norms","backbone.fpn"),"backbone.norms + backbone.fpn",
         "Collect LN outputs of ALL 4 stages\nconcat 6144 -> Linear 1536 -> GELU"),
        (7.15,1.35,"decoder.up_block","decoder.up_block",
         "concat(FPN output, fused-low skip): 3072 channels\ntime-conditioned UpBlock x2 -> [B,1536,180,360]"),
        (5.3,1.2,"decoder.head","decoder.head",
         "Linear 1536 -> 1120 (=70 x 4 x 4)\nRearrange patches -> delta [B,70,720,1440]"),
    ]
    previous=20.9
    for y,h,names,title,detail in stages:
        module_box(ax,a,x,y,w,h,names,title,detail,fs=8.6)
        arrow(ax,(cx,previous),(cx,y+h));previous=y
    box(ax,x,3.7,w,.95,"DAForecastNet.forward returns\nERA5(T) + delta = forecast at T+6h","output")
    arrow(ax,(cx,5.3),(cx,4.65))
    box(ax,.2,19.2,3.9,2.1,
        f"GNSS-ZTD: T-{a['obs_hours']}h to T, every {a['obs_interval']} min\n"
        f"[B,{a['obs_frames']},1,720,1440] + validity mask\n"
        "3 station static fields (not FuXi constants):\nstation-grid mask, latitude, longitude","input",8.8)
    module_box(ax,a,.2,12.5,3.9,6.,"obs_encoder","obs_encoder",
        "Conv3d([ZTD, valid mask]): 2 -> 32 -> 64\nkernel 3, padding 1, with SiLU\n\n"
        "Mask-weighted aggregation over time\nTHEN concat masked station static (64+3)\n\n"
        "Conv2d 67 -> 1536, k4/s4\nResBlock x high_mask -> obs_high\n\n"
        "Conv2d k3/s2 -> ResBlock x low_mask\n-> obs_low",fs=8.9)
    arrow(ax,(2.15,19.2),(2.15,18.5),"obs_edge")
    arrow(ax,(4.1,17.75),(x,17.75),"obs_edge")
    arrow(ax,(4.1,14.1),(x,14.1),"obs_edge")
    label(ax,4.43,18.,"high",7.5,"obs_edge")
    label(ax,4.43,14.35,"low",7.5,"obs_edge")
    box(ax,.2,9.55,3.9,2.2,
        "Fusion is NOT simply gate x raw observations.\n"
        "delta = Conv-SiLU-Conv([ERA, OBS, mask])\n"
        "available = spatial max of patch_mask\n"
        "All-missing observations -> zero correction.\n"
        "Both fusion convolutions AND gates train.","note",8.7)
    box(ax,.2,6.05,3.9,2.75,
        "Code-alignment notes\n\nCubeEmbed uses Conv2d, not Conv3d.\n"
        "Time frames are folded into input channels.\n"
        "No direct high-to-decoder skip.\n"
        "Only the low grid enters Transformer.\n"
        "No ERA-only Adapter has been added.","note",8.9)
    box(ax,9.45,20.3,2.55,.95,"12 calendar/time features\n(no learned input parameters)","input",8.2)
    module_box(ax,a,9.45,17.8,2.55,1.8,"time_embed","time_embed",
               "Linear 12 -> 1536\nSiLU -> Linear 1536",fs=8.1)
    arrow(ax,(10.725,20.3),(10.725,19.6))
    route(ax,[(10.725,17.8),(10.725,15.9),(8.8,15.9)],"time",True)
    route(ax,[(10.725,15.9),(10.725,7.55),(8.8,7.55)],"time",True)
    label(ax,10.9,11.2,"time conditioning",8.5,"time",rotation=90)
    route(ax,[(cx,13.5),(cx,13.17),(9.15,13.17),(9.15,8.),(8.8,8.)],"skip",True)
    label(ax,9.40,10.9,"1/8 skip AFTER low fusion",8.3,"skip",rotation=90)
    box(ax,.2,.65,11.8,2.,
        "*Freeze map shown for train.freeze_fuxi=True; the actual config flag is reported above.\n"
        "All FuXi parameters freeze, including attention logit_scale and norm affine parameters.\n"
        "OBS encoder and both complete fusion modules remain trainable.\n"
        "Frozen parameters still allow gradients through the core to the OBS/fusion modules.","note",10)


def draw_fuxi(ax,a):
    ax.set(xlim=(0,12.3),ylim=(0,23.3));ax.axis("off")
    label(ax,6.1,22.95,"Original FuXi: source of the blue modules",16,fontweight="bold")
    label(ax,6.1,22.42,"UTransformer in model/fuxi.py; checkpoint keys carry a decoder. prefix",9.5)
    x,w,cx=4.,4.7,6.35
    box(ax,x,20.9,w,1.05,"ERA5(T-6h,T); resize_input to 720 x 1440 if needed\nconcat 5 static fields; fold time -> [B,150,720,1440]","input",8.7)
    stages=[
        (19.,1.3,"era_encoder.patch_embed","patch_embed = CubeEmbed","Conv2d k4/s4 -> flatten -> LayerNorm\n64800 tokens x 1536 -> 180 x 360 feature grid"),
        (16.9,1.3,"era_encoder.down_block","down_blocks.0","Time-conditioned residual downsampling /2\n[B,1536,90,180]"),
        (13.8,2.05,"backbone.layers","layers: Swin Transformer","4 stages x 12 blocks = 48 blocks\n1536 channels; 24 heads; windows 9 x 9\n90 x 180 grid; periodic longitude"),
        (11.55,1.45,("backbone.norms","backbone.fpn"),"norm0..3 + fpn","Collect LN outputs of ALL 4 stages\nconcat 6144 -> Linear 1536 -> GELU"),
        (8.75,1.5,"decoder.up_block","up_blocks.0","concat(FPN output, DownBlock skip): 3072 channels\nTime-conditioned UpBlock x2 -> [B,1536,180,360]"),
        (6.5,1.45,"decoder.head","head","Linear 1536 -> 1120 (=70 x 4 x 4)\nRearrange -> delta [B,70,720,1440]"),
    ]
    previous=20.9
    for y,h,names,title,detail in stages:
        module_box(ax,a,x,y,w,h,names,title,detail,source=True,fs=9)
        arrow(ax,(cx,previous),(cx,y+h));previous=y
    box(ax,x,4.75,w,1.05,"UTransformer.forward returns delta only\nThe residual addition is OUTSIDE this class.","output",9.3)
    arrow(ax,(cx,6.5),(cx,5.8))
    box(ax,x,3.1,w,1.05,"Forecast wrapper: ERA5(T) + delta\n-> forecast at T+6h (no extra parameters)","output",9.3)
    arrow(ax,(cx,4.75),(cx,4.15))
    box(ax,.25,20.25,2.95,1.05,"12 calendar/time features\nSame encoding as DA-Forecast","input",8.6)
    module_box(ax,a,.25,17.4,2.95,2.05,"time_embed","time_embed",
               "Linear 12 -> 1536\nSiLU -> Linear 1536",source=True,fs=8.5)
    arrow(ax,(1.725,20.25),(1.725,19.45))
    route(ax,[(3.2,18.),(3.6,18.),(4.,18.)],"time",True)
    route(ax,[(1.725,17.4),(1.725,9.5),(4.,9.5)],"time",True)
    label(ax,1.47,13.5,"time conditioning",8.8,"time",rotation=90)
    route(ax,[(cx,16.9),(cx,16.55),(9.5,16.55),(9.5,9.5),(8.7,9.5)],"skip",True)
    label(ax,9.76,13.1,"1/8 DownBlock skip (no OBS fusion)",8.8,"skip",rotation=90)
    box(ax,.25,.65,11.7,1.65,
        "The checkpoint name decoder.* denotes the whole original UTransformer,\n"
        "not only the upsampling decoder in DA-Forecast.\n"
        "The loader remaps it across era_encoder, time_embed, backbone and decoder.","note",10)


def draw_parameter_table(ax,a):
    ax.axis("off")
    ax.text(.5,.98,"Exact parameter inventory and checkpoint-name mapping",ha="center",va="top",fontsize=15,fontweight="bold",color=COLORS["text"])
    rows=[]
    for g in a["groups"]:
        scope=PARAMETER_SCOPE[g["name"]]
        rows.append([g["name"],g["source"],f"{g['parameters']:,}","FROZEN" if g["frozen"] else "TRAINABLE",scope])
    table=ax.table(cellText=rows,colLabels=["DA module / parameter group","Original checkpoint prefix","Parameters (not buffers)","If freeze_fuxi=True","Scope"],
        cellLoc="left",colLoc="left",colWidths=[.23,.24,.16,.13,.24],bbox=[.01,.15,.98,.76])
    table.auto_set_font_size(False);table.set_fontsize(10)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor("#C4CED8");cell.set_linewidth(.6)
        if row==0:
            cell.set_facecolor("#294762");cell.get_text().set_color("white");cell.get_text().set_weight("bold")
        else:cell.set_facecolor(COLORS["frozen"] if a["groups"][row-1]["frozen"] else COLORS["new"])
        if col==2:cell.get_text().set_ha("right")
    ax.text(.5,.095,f"Total {a['total']:,}  =  frozen FuXi {a['frozen']:,} ({100*a['frozen']/a['total']:.1f}%)  +  trainable OBS/fusion {a['trainable']:,} ({100*a['trainable']/a['total']:.1f}%)",ha="center",fontsize=12,fontweight="bold",color=COLORS["text"])
    ax.text(.5,.035,"Counts: live model classes on meta device. Mapping: fuxi_pretrained.py. Freeze scope: train.py. No checkpoint or GPU is loaded to draw this figure.",ha="center",fontsize=9.3,color=COLORS["edge"])


def draw(output: Path,dpi: int=200,config_path: Path | None=None):
    a=inspect_models(config_path or ROOT/"configs/train_da_forecast.json")
    fig=plt.figure(figsize=(26,22),facecolor="white")
    grid=fig.add_gridspec(3,2,height_ratios=[1.6,15.3,5.1],hspace=.055,wspace=.035)
    header=fig.add_subplot(grid[0,:]);header.axis("off")
    header.text(.5,.82,"Global +6h Forecast | Architecture, Pretrained Weights & Freeze Map",ha="center",va="center",fontsize=23,fontweight="bold",color=COLORS["text"])
    current=str(a["config_freeze"]).lower()
    status="core frozen; only OBS/fusion trainable" if a["config_freeze"] else "ALL model parameters trainable on a new launch"
    header.text(.5,.42,f"Illustration: freeze_fuxi=true scenario    |    Config snapshot: freeze_fuxi={current} -> {status}",ha="center",va="center",fontsize=12.5,color="#7A4B0D",bbox=dict(boxstyle="round,pad=.5",facecolor=COLORS["warning"],edgecolor="#D5B178"))
    header.text(.5,.10,"Pretrained initialization requires --pretrained-fuxi (or suitable --resume). A config edit does not change an already-running process.",ha="center",va="center",fontsize=10,color=COLORS["edge"])
    draw_current(fig.add_subplot(grid[1,0]),a)
    draw_fuxi(fig.add_subplot(grid[1,1]),a)
    draw_parameter_table(fig.add_subplot(grid[2,:]),a)
    fig.subplots_adjust(left=.025,right=.985,top=.985,bottom=.025)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=dpi,bbox_inches="tight",pad_inches=.18,facecolor="white")
    plt.close(fig)
    return a


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/train_da_forecast.json")
    parser.add_argument("--output",type=Path,default=ROOT/"network_architecture_comparison.png")
    parser.add_argument("--dpi",type=int,default=200)
    parser.add_argument("--audit-only",action="store_true",help="verify mapping/counts without plotting")
    args=parser.parse_args()
    if args.audit_only:print(json.dumps(inspect_models(args.config),indent=2))
    else:
        a=draw(args.output,args.dpi,args.config)
        print(args.output.resolve())
        print(f"freeze=true: {a['frozen']:,} frozen / {a['trainable']:,} trainable; current config freeze={a['config_freeze']}")


if __name__=="__main__":
    main()
