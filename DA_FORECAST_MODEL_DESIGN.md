# 欧洲区域 ERA5 + GNSS-ZTD 预报模型设计建议

## 1. 任务定义

目标是在 FuXi-short 思路的基础上构建欧洲区域气象预报模型：

- 分析场输入：ERA5；
- 观测输入：GNSS ZTD；
- 监督标签：未来时刻 ERA5；
- 区域：欧洲 0.25° 网格，当前尺寸为 `80 × 120`；
- 第一阶段任务：

  ```text
  ERA5(t-6h, t) + GNSS-ZTD(<=t) -> ERA5(t+6h)
  ```

第一版应先验证 6h 单步预报中 GNSS-ZTD 是否带来稳定增益，再扩展到 12h、18h、24h。GNSS 观测窗口必须截止到分析时刻 `t`，不能使用 `t` 之后的观测，避免未来信息泄漏。

## 2. 为什么需要简化原始 FuXi

当前 FuXi-short 的主要配置为：

| 配置 | 原 FuXi-short |
|---|---:|
| 输入网格 | 720 × 1440 |
| Embed dimension | 1536 |
| Attention heads | 24 |
| Transformer blocks | 48 |
| Window size | 9 × 9 |
| 可训练参数 | 1,556,718,304 |

该模型面向全球网格和大规模数据。欧洲区域数据仅约几千个时次，直接使用 15.57 亿参数会造成：

- 训练和显存成本过高；
- 参数量远大于有效数据量，容易过拟合；
- 原全球经度周期边界不适合欧洲裁剪区域；
- 原 `9 × 9` 窗口与欧洲下采样后的 `10 × 15` 特征图不匹配。

建议保留 FuXi 的核心思想——双时次输入、增量预报、Swin 主干、U 形解码和多深度特征融合——但将模型压缩到约 1000 万至 2000 万参数。

## 3. 推荐的总体结构

```text
ERA5(t-6h, t)                         GNSS ZTD(t-6h ... t, every 20 min)
[B,2,70,80,120]                       [B,19,1,80,120]
       |                                       |
ERA5 Encoder                              Obs Encoder
       |                                       |
[B,256,20,30]                         [B,128/256,20,30]
       +---------- 门控融合（尺度1）-----------+
                            |
                     [B,256,20,30]
                            |
                         Downsample
                            v
                     [B,256,10,15]
                            |
                   门控融合（尺度2）
                            |
       4组轻量 Swin，每组2个Block，window=5
                            |
                     [B,256,10,15]
                            |
              Upsample + 20×30 跳跃连接
                            |
                 Patch Head 恢复到80×120
                            v
                   70通道预报增量
                            |
                       + ERA5(t)
                            v
                    ERA5(t+6h)预报
```

推荐的第一版超参数：

```python
image_size = (80, 120)

era_chans = 70
era_frames = 2
obs_chans = 1
obs_history_hours = 6
obs_interval_minutes = 20
obs_frames = 19  # includes both t-6h and t
out_chans = 70

embed_dim = 256
num_heads = 8
patch_size = 4
window_size = 5
down_times = 1
depths = [2, 2, 2, 2]
```

`80 × 120` 经 `patch_size=4` 后为 `20 × 30`，再下采样一次为 `10 × 15`。两个尺度都可以被 `window_size=5` 整除。

## 4. ERA5 Encoder

建议输入两个连续 ERA5 分析场：

```text
ERA5(t-6h), ERA5(t)
```

而不是只输入当前时刻。两个时次能够提供温度、风和水汽的演变趋势。

输入张量：

```python
era5.shape == [B, 2, 70, 80, 120]
```

将时间与通道合并后：

```text
[B, 140, 80, 120]
```

使用 Patch Embedding：

```python
Conv2d(
    in_channels=140 + static_chans,
    out_channels=256,
    kernel_size=4,
    stride=4,
)
```

输出：

```text
[B, 256, 20, 30]
```

静态输入可以包括经纬度、地形和陆海掩膜等。当前全球 `conds.npy` 的形状为 `5 × 720 × 1440`，需要按照欧洲区域的同一经纬度范围裁剪或重新生成，不能直接缩放后使用。

## 5. GNSS-ZTD Encoder

### 5.1 输入组成

不能只将缺测格点填成零后输入网络。标准化空间中的零通常表示均值，如果没有 mask，网络无法区分“缺测”和“观测等于均值”。至少需要：

```text
ZTD值 + 有效性mask
```

推荐输入：

```python
obs.shape      == [B, 19, 1, 80, 120]
obs_mask.shape == [B, 19, 1, 80, 120]
```

还可以加入：

- 站点高度；
- 纬度、经度；
- 每帧相对分析时刻的时间偏移；
- patch 内是否存在有效站点的 mask。

如果已有 ERA5 计算得到的背景 ZTD，建议额外输入创新量：

```text
GNSS ZTD - H(ERA5)
```

最终观测分支可以包含：

```text
GNSS ZTD、ZTD innovation、valid mask、站点静态信息
```

创新量能够更直接地告诉网络 ERA5 水汽状态应该向哪个方向修正。

### 5.2 编码结构

推荐使用浅层时空卷积：

```python
Conv3d(2, 32, kernel_size=3, padding=1)
SiLU()
Conv3d(32, 64, kernel_size=3, padding=1)
SiLU()
TemporalPooling()
Conv2d(64, 256, kernel_size=4, stride=4)
```

输出：

```text
[B, 256, 20, 30]
```

`TemporalPooling` 应结合有效性 mask，避免缺测帧参与普通平均。第一版可以使用 mask 加权平均，之后再尝试可学习时间注意力。

## 6. ERA5 与观测融合

欧洲数据较少，第一版不建议直接加入复杂的全局 Cross-Attention。优先使用门控残差融合：

```python
delta = fusion(torch.cat([
    era_feature,
    obs_feature,
    obs_patch_mask,
], dim=1))

fused = era_feature + torch.sigmoid(gate) * delta
```

门控参数建议初始化为负值：

```python
self.gate = nn.Parameter(torch.tensor(-4.0))
```

此时：

```text
sigmoid(-4)约等于0.018
```

训练初期模型行为接近 ERA5-only；只有当观测确实能够降低预报误差时，网络才逐渐增大观测贡献。这能降低稀疏观测和小数据条件下的训练不稳定性。

融合模块可以使用：

```python
Conv2d(513, 256, kernel_size=1)
GroupNorm(32, 256)
SiLU()
Conv2d(256, 256, kernel_size=3, padding=1)
```

建议在两个尺度融合：

1. `20 × 30`：保留站点附近的局地信息；
2. `10 × 15`：在 Transformer 主干前融合较大尺度信息。

如果门控卷积融合已经表现出稳定观测收益，再将局部或窗口 Cross-Attention 作为后续对照实验。

## 7. Swin Transformer 主干

保留当前的 Swin V2 Block，但缩小为：

```python
embed_dim = 256
num_heads = 8
window_size = 5
depths = [2, 2, 2, 2]
```

共 8 个 Transformer Block。四组 Swin 输出仍可以像原 FuXi 一样保留并拼接：

```text
4 × 256 -> Linear(1024, 256) -> GELU
```

这样可以保留多深度特征融合，但不会引入原模型的巨大计算量。

## 8. 欧洲区域边界修改

### 8.1 取消全球经度周期卷积

原始 `PeriodicConv2d` 在经度方向执行 circular padding，适用于全球网格，但不适用于欧洲裁剪区域：

```text
欧洲最西侧不与欧洲最东侧相邻
```

否则网络会在两个区域边界之间制造错误连接。建议替换为普通区域卷积，例如：

```python
class RegionalConv2d(nn.Conv2d):
    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        return F.conv2d(
            x, self.weight, self.bias,
            self.stride, 0, self.dilation, self.groups,
        )
```

也可以直接使用：

```python
nn.Conv2d(..., padding=1)
```

### 8.2 修改移动窗口 mask

原 Swin Block 默认使用 `mask_type='h'`，偏向全球经度周期设计。欧洲区域应同时限制两个空间方向，使用完整二维移动窗口 mask，例如：

```python
mask_type = 'hw'
```

代码中应确保 shift 后的纬度和经度边界都不会被错误连接。

## 9. Decoder 与预报目标

建议保留 FuXi 的增量预报形式：

```python
delta = decoder(fused_features)
forecast = era5[:, -1] + delta
```

监督目标为：

```python
label = ERA5(t + 6h)
```

6h 状态变化量通常比完整大气状态更容易拟合，也能保留当前 ERA5 分析场的大尺度结构。

Decoder 建议包括：

1. 将 `10 × 15` 特征反卷积上采样到 `20 × 30`；
2. 与融合后的浅层 `20 × 30` 特征做跳跃连接；
3. 使用 ResBlock 融合；
4. Linear Head 将每个 token 映射为 `70 × 4 × 4`；
5. 重排得到 `70 × 80 × 120`；
6. 加到当前 ERA5 分析场上。

输出70个通道：

```text
13层 × (z,t,u,v,r) = 65
t2m,u10,v10,msl,tp = 5
总计70
```

## 10. Loss 设计

任务是预报整个欧洲，不建议只使用 station-halo loss。只监督站点周围会使区域其他位置得不到充分优化。

建议：

```python
loss = loss_global + 0.5 * loss_halo
```

其中：

- `loss_global`：整个欧洲、cos(latitude) 加权的70通道MAE；
- `loss_halo`：GNSS站点及周围区域的MAE；
- 各变量在标准化空间计算，避免量纲不同导致某些变量主导训练。

降水如果使用 `log1p` 标准化，应继续在相同空间训练；物理单位评估时必须逐点执行逆变换，再计算毫米 MAE。

## 11. 建议的代码拆分

不要继续让一个 `UTransformer` 类承担所有功能。建议拆为：

```python
class ERA5Encoder(nn.Module):
    ...

class ObsEncoder(nn.Module):
    ...

class GatedFusion(nn.Module):
    ...

class ForecastBackbone(nn.Module):
    ...

class ForecastDecoder(nn.Module):
    ...

class DAForecastNet(nn.Module):
    def forward(self, era5, obs, obs_mask, temb=None, const=None):
        era_features = self.era_encoder(era5, const)
        obs_features = self.obs_encoder(obs, obs_mask)

        fused_features = self.fusion(
            era_features,
            obs_features,
            obs_mask,
        )

        hidden = self.backbone(fused_features, temb)
        delta = self.decoder(hidden, fused_features)

        return era5[:, -1] + delta
```

建议保留旧 `UTransformer`，新增 `DAForecastNet`，不要直接覆盖原模型。这样便于：

- 加载和检查旧 FuXi 权重；
- ERA5-only 与 ERA5+Obs 使用同一代码库；
- 做消融实验；
- 排查结构修改导致的问题。

## 12. 推荐训练顺序

### 阶段1：ERA5-only预训练

关闭观测分支，使用欧洲 ERA5 训练 6h 预报网络：

```text
ERA5(t-6h,t) -> ERA5(t+6h)
```

得到稳定的区域预报基线。

### 阶段2：训练观测分支

加载 ERA5-only 权重，先冻结：

- ERA5 Encoder；
- Swin主干；
- 大部分Decoder。

只训练：

- Obs Encoder；
- Fusion；
- Fusion gate；
- Decoder最后一部分。

### 阶段3：联合微调

解冻全部网络，使用不同学习率：

```text
ERA5 Encoder / Swin Backbone：1e-5
Obs Encoder / Fusion：1e-4
```

具体数值需要根据 batch size 和 loss 曲线调整，但观测分支学习率应高于已预训练主干。

## 13. 必须完成的消融实验

至少比较以下实验，并保证网络主干、训练样本、随机种子和评估日期一致：

| 实验 | 输入 |
|---|---|
| ERA5-only | ERA5，无观测分支 |
| Mask-only | ERA5 + 站点/缺测mask |
| Raw ZTD | ERA5 + GNSS ZTD |
| ZTD innovation | ERA5 + GNSS ZTD + `GNSS-H(ERA5)` |
| Shuffled ZTD | ERA5 + 时空打乱的ZTD |

`Shuffled ZTD` 很重要。如果真实ZTD和打乱ZTD的表现相近，说明网络没有真正利用观测值，提升可能来自站点位置、参数量变化或训练随机性。

建议至少使用3个随机种子报告均值和标准差，因为此前6h实验中有Obs相对无Obs的加权loss提升仅约 `0.0835%`，单次训练波动可能与该增益同量级。

## 14. 评估建议

需要同时报告：

- 全区域纬度加权70通道loss；
- station-halo loss；
- 各通道标准化MAE；
- 反归一化后的物理单位MAE；
- 有Obs相对无Obs的提升；
- 多个随机种子的均值和标准差。

重点关注与ZTD物理联系较强的变量：

- 中低层相对湿度；
- 温度；
- 位势高度；
- 海平面气压；
- 降水。

验证集与测试集必须使用不同日期。不能再将同一个时间段同时作为验证集和测试集，否则最优checkpoint选择已经间接使用了测试期信息。

## 15. 多时效预报的后续扩展

纯预报和循环同化必须分开定义。

### 纯预报

只使用分析时刻 `t` 及之前的GNSS观测：

```text
ERA5(t-6h,t) + GNSS(<=t)
    -> ERA5(t+6h,t+12h,t+18h,t+24h)
```

可以采用直接多时效输出，或自回归预报；自回归时不能在未来步骤输入真实GNSS观测。

### 循环同化预报

每6h获得新的GNSS观测并重新分析：

```text
t时刻分析与预报
-> 获得t+6h观测
-> t+6h重新分析与预报
```

这是另一个任务，不应和固定起报时刻的纯预报混合评估。

建议扩展顺序为：

1. 6h单步；
2. 12h双步；
3. 24h直接多时效或rollout训练；
4. 最后再研究循环同化。

## 16. 预期与风险

ERA5本身已经同化大量常规和卫星观测，也包含水汽相关信息。使用GNSS ZTD预测未来ERA5时，额外信息很可能主要体现在：

- 中低层湿度；
- 降水；
- 近地面气压；
- 较短预报时效。

因此不应通过扩大主干掩盖观测贡献。第一版最重要的目标是：

1. 建立可靠的ERA5-only区域预报基线；
2. 确认观测分支确实使用了ZTD数值；
3. 用多随机种子证明增益超过训练波动；
4. 再逐步增加融合复杂度和预报时效。

## 17. FuXi Encoder 与 Assimilation Encoder 对比

两份代码中的“encoder”不是同一层级的概念：

- FuXi encoder 的主要目标是把连续气象场变成 Transformer token，用于未来预报；
- `assimilation.py` 的 encoder 是多分支卷积编码器，目标是保持背景和观测各自特征，并在多个尺度反复融合，用于当前时刻分析订正。

### 17.1 核心差异

| 项目 | FuXi encoder | Assimilation encoder |
|---|---|---|
| 主要任务 | 未来气象预报 | 当前时刻数据同化/订正 |
| 输入分支 | 单分支，所有输入早期拼接 | 背景、Obs、Side 三个独立分支 |
| 时间处理 | 多时次合并为通道，并有显式时间编码 | 多帧合并为通道，无显式时间编码 |
| 第一层降采样 | `Conv2d(kernel=4,stride=4)` | `Conv2d(kernel=2,stride=2)` |
| 主干 | Swin Transformer V2 | 纯卷积、多层小型U-Net |
| 融合方式 | 输入层直接拼接 | 每个尺度分别编码后融合 |
| Obs处理 | 原模型没有独立Obs Encoder | 有独立Obs Encoder |
| 归一化 | Patch后LayerNorm；ResBlock用GroupNorm | 每个EncoderBlock用LayerNorm |
| 时间条件 | 12维时间编码调制ResBlock | 无显式时间条件 |
| 边界 | 全球经度周期卷积 | 普通卷积；尺寸不足时replicate补齐 |
| 输出语义 | 未来状态增量 | 背景分析订正结果 |

### 17.2 FuXi 的编码过程

FuXi 最前面的编码器是 `CubeEmbed`。它先将两个时次、气象变量和静态场全部拼接：

```text
2个时次 × (70个气象通道 + 5个静态通道)
= 150通道
```

然后用一次卷积完成空间分块和通道投影：

```python
Conv2d(
    in_channels=150,
    out_channels=1536,
    kernel_size=4,
    stride=4,
)
```

全球模型中的形状变化为：

```text
[B,150,720,1440]
-> [B,1536,180,360]
-> [B,64800,1536]
```

欧洲简化版建议改为：

```text
[B,140(+静态场),80,120]
-> [B,256,20,30]
```

FuXi编码器具有以下特点：

- 所有输入在进入encoder前已经混合；
- 使用一个非重叠的 `4 × 4` 卷积将每个patch压缩成一个token；
- 空间尺寸在第一层一次缩小4倍；
- token直接进入Transformer；
- 原模型没有独立观测分支。

随后 `DownBlock2D` 再下采样一次：

```text
180×360 -> 90×180
```

该模块由步长2卷积和残差块组成，时间编码通过scale/shift调制残差特征。

严格来说，FuXi完整的编码侧包括：

```text
CubeEmbed -> DownBlock2D -> Swin Transformer
```

### 17.3 Assimilation 的编码过程

`assimilation.py` 的基础编码单元是 `EncoderBlock`：

```python
Conv2d(in_chans, embed_dim, kernel_size=2, stride=2)
LayerNorm
SiLU
Conv2d(embed_dim, out_chans, kernel_size=3, stride=1)
LayerNorm
SiLU
```

每经过一个EncoderBlock：

```text
H×W -> H/2×W/2
```

它不是patch-token encoder，输出仍是普通二维特征图：

```text
[B,C,H,W]
```

Assimilation为不同输入分别建立encoder：

```python
bg = bg_encoder(bg)
obs = obs_encoder(obs)
side_info = side_encoder(side_info)
```

三个分支不会在最开始直接混合，而是先各自编码，再进入融合模块。

### 17.4 Assimilation 对观测时间维的处理

假设观测输入为：

```text
[B,10,4,80,120]
```

当前实现直接将时间和通道展平：

```text
[B,10×4,80,120]
= [B,40,80,120]
```

然后把40个通道一起送入二维Obs Encoder。不同观测时次相当于固定输入通道：

```text
第1帧ZTD -> 一组固定通道
第2帧ZTD -> 另一组固定通道
...
```

这种处理实现简单，但存在以下限制：

- 没有显式建模时间演变；
- 无法自然支持变化的观测窗口长度；
- 没有时间间隔编码；
- 对缺测帧和不规则采样的处理较弱。

FuXi同样将两个ERA5时次合并到通道维，但另外使用12维显式时间编码。Assimilation没有对应时间条件。这也是新模型的Obs Encoder更适合采用 `Conv3d + masked temporal pooling`，而不是原样复制Assimilation Obs Encoder的原因。

### 17.5 两种融合方式的区别

#### FuXi：早期融合

FuXi在encoder之前执行：

```python
x = torch.cat([era5, const], dim=channel)
```

所有变量经过同一个 `CubeEmbed`。如果直接把稀疏ZTD拼到FuXi输入：

```text
ERA5 + ZTD + mask -> 一个CubeEmbed
```

少量ZTD通道很容易被70个密集ERA5通道淹没，而且缺测mask和时间结构无法得到专门处理。

#### Assimilation：分支编码后融合

Assimilation先分别编码：

```text
Background Encoder
Observation Encoder
Side Encoder
```

然后在 `FussionNetv2` 中拼接：

```python
h = torch.concat([bg, obs, side_info], dim=1)
```

融合模块本身又是一个小U-Net：

```text
concat
-> 下采样
-> 再下采样
-> 上采样
-> 跳跃连接
-> 分别生成bg、obs、side更新量
```

背景和观测通过残差方式更新：

```python
bg = bg + out_bg
obs = obs + out_obs
```

因此Assimilation的融合比FuXi复杂，并且在多个分辨率上重复发生。

### 17.6 Assimilation 是嵌套U-Net

Assimilation整体主路径近似为：

```text
80×128
  ↓ bg/obs/side各自编码
40×64
  ↓
20×32
  ↑
40×64
  ↑
80×128
```

但每个尺度内部的 `FussionNetv2` 又包含两次下采样和两次上采样。因此它不是简单的：

```text
Encoder -> Fusion -> Decoder
```

而更接近：

```text
多分支外层U-Net
    └── 每层内部再嵌套一个融合U-Net
```

这种结构适合在背景与稀疏观测之间反复交换局地信息，但参数量较大，而且纯卷积的远距离信息传播能力弱于Swin Transformer。

### 17.7 残差和输出语义的区别

FuXi包含两类残差：

1. ResBlock内部的特征残差；
2. 推理时将预报增量加到最新ERA5状态：

   ```python
   forecast = model(input) + input[:, -1]
   ```

Assimilation包含更多层次的残差：

```python
bg = bg + out_bg
obs = obs + out_obs
out = out + bg_cp
out = enhance(out) + out
```

因此两者的目标不同：

```text
Assimilation：在背景场基础上做当前时刻的小幅订正
FuXi：从当前和前一时刻状态预测未来变化量
```

### 17.8 新模型应如何组合两者

不建议直接选择其中任意一个完整架构，而应组合各自优点：

| 新模型部分 | 建议参考 |
|---|---|
| ERA5 Encoder | FuXi的CubeEmbed思想 |
| Obs Encoder | Assimilation的独立分支思想，但重新设计时间处理 |
| 时间建模 | FuXi时间编码 + Obs时序卷积 |
| 融合逻辑 | Assimilation的“先编码、后融合”思想 |
| 融合实现 | 简化门控残差融合，不使用嵌套FussionNetv2 |
| 全局主干 | FuXi的Swin Transformer |
| Decoder | FuXi增量预报 + U形跳跃连接 |
| 区域边界 | 非周期区域边界 |

推荐的数据流为：

```text
ERA5
  -> CubeEmbed / ERA5Encoder
  -> era_feature ----------------┐
                                 ├-> GatedFusion -> Swin -> Decoder
GNSS ZTD                         │
  -> 时空ObsEncoder              │
  -> obs_feature ----------------┘
```

### 17.9 不建议直接复用 Assimilation EncoderBlock

不建议把Assimilation的 `EncoderBlock` 原样用作新模型的全部encoder，原因包括：

- 每层都缩小2倍，空间压缩较重；
- Obs时间帧只是展平为通道；
- 没有显式时间编码；
- 输出不是Transformer token；
- 配套的 `FussionNetv2` 结构复杂，容易在小数据上过拟合。

但是，可以保留它的两个关键思想：

1. ERA5与观测使用独立encoder；
2. 分别提取特征后再融合。

### 17.10 不建议 ERA5 与 ZTD 共用一个 CubeEmbed

也不建议将ERA5和ZTD直接拼接后共同进入一个FuXi CubeEmbed，原因包括：

- 稀疏ZTD容易被70通道密集ERA5淹没；
- 无法专门处理ZTD缺测mask；
- 无法专门建模高频观测时间序列；
- 无Obs消融会同时改变第一层输入通道和参数结构。

最终推荐方案为：

```text
FuXi式ERA5 Encoder
+ 改进的独立Obs Encoder
+ Assimilation式“先编码后融合”
+ 简化的门控残差Fusion
+ FuXi式Swin预报主干和增量Decoder
```

