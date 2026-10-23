# 当前全球 ERA5 + GNSS-ZTD 6小时预报网络

当前模型完整保留原 FuXi 的 ERA5 预报路径，只增加 GNSS-ZTD 编码器和
双尺度门控修正。实现位置：

- `model/da_forecast_model.py`
- `model/fuxi.py`
- `configs/train_da_forecast.json`

## 1. 任务与输入输出

```text
ERA5(t-6h, t) + GNSS-ZTD(t-6h到t，每20分钟一帧，共19帧) -> ERA5(t+6h)
```

全球网格为 `720×1440`，分辨率 `0.25°`。动态 ERA5 和输出均为 70 通道。

| 输入 | shape |
|---|---:|
| ERA5 | `[B,2,70,720,1440]` |
| ERA5静态场 | `[B,5,720,1440]` |
| 时间编码 | `[B,12]` |
| ZTD | `[B,T,1,720,1440]` |
| ZTD有效mask | `[B,T,1,720,1440]` |
| 站点静态信息 | `[B,3,720,1440]` |
| 输出 | `[B,70,720,1440]` |

输出与原 FuXi 一样采用增量预报：`ERA5(t) + delta`。

## 2. 总体数据流

```text
ERA5 + ERA5静态场                         ZTD + mask + 站点静态场
        │                                           │
        ▼                                           ▼
  原FuXi PatchEmbed                            Obs Encoder
        │                                           │
        ├── 1/4: [B,1536,180,360] ◄──门控融合───────┤
        │                                           │
        └── 1/8: [B,1536, 90,180] ◄──门控融合───────┘
                             │
                             ▼
              FuXi Swin Transformer（90×180）
                 4 stages × 12 blocks, window=9
                             │
                             ▼
                 FuXi FPN：拼接4个stage输出
                             │
                             ▼
        FuXi UpBlock（1/8→1/4，low skip，12维时间调制）
                             │
                             ▼
        Linear(1536, 70×4×4) + patch rearrange
                             │
                             ▼
                    delta [B,70,720,1440]
                             │
                       + ERA5(t)
```

## 3. 原 FuXi ERA5 路径与 OBS 分支

每个 ERA5 时次与相同的5个静态场拼接，两帧合并为150通道，然后经过原
FuXi 的 `CubeEmbed` 和 `DownBlock2D`：

```text
high_era = [B,1536,180,360]
low_era  = [B,1536, 90,180]
```

ZTD 和有效 mask 进入两层 3D 卷积，按有效时次做 mask-aware 聚合，再和站点
静态信息一起编码到相同的两个尺度：

```text
high_obs = [B,1536,180,360]
low_obs  = [B,1536, 90,180]
```

高、低尺度各自使用一个 `GatedFusion`。初始 gate 约为
`sigmoid(-4)=0.018`。完全没有有效观测时，融合结果严格退化为 ERA5-only
路径。

## 4. 恢复的 FuXi Transformer

融合后的低分辨率特征展开为16200个 token：

```text
[B,1536,90,180] -> [B,16200,1536]
```

Transformer 参数恢复为原 FuXi 配置：

```text
embed_dim = 1536
num_heads = 24
window_size = 9
depths = [12,12,12,12]
mlp_ratio = 4
```

shifted-window mask 只阻断南北边界，经度方向保持周期连续。每个 stage 的输出
分别 LayerNorm，四路拼接后经过 `Linear(4×1536,1536) -> GELU`。

## 5. 恢复的 FuXi Decoder

Transformer 输出与融合后的 low feature 拼接，通过原 FuXi 的 UpBlock：

```text
[B,3072,90,180] -> time-conditioned ResBlock
                  -> ConvTranspose2d(stride=2)
                  -> [B,1536,180,360]
```

卷积在经度方向使用周期 padding。UpBlock 输出直接进入原 FuXi patch head，
不再额外叠加1/4尺度 high feature。

最后使用原 FuXi 的 linear patch head：

```text
Linear(1536,70×4×4) -> rearrange -> delta
forecast = ERA5(t) + delta
```

## 6. 当前状态

OBS 分支仍在模型中，并且 `use_obs=true`。预训练加载器覆盖原 FuXi 路径的
全部886个状态张量；`freeze_fuxi=true` 时只训练 OBS encoder 和两个 gate。
训练入口采用4卡FSDP `FULL_SHARD`，每个SwinBlock单独包装，
默认每卡batch size 1、梯度累积4次、BF16混合精度。

小尺寸单元测试覆盖输出尺寸、有限值、所有分支梯度、全缺测观测退化、全球
周期卷积、配置错误和权重保存/恢复。
