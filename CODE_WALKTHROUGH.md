# da-forecast 代码导读：从一个样本到一次训练更新

核对日期：2026-10-23。本文解释当前全球 ERA5 + GNSS-ZTD 版本的实际实现，
不是未来改造方案。修改代码后，应重新核对这里的配置数值和行为。

本文只配套说明代码，没有修改训练、评估或数据处理逻辑。
更细的网络层级说明见 [CURRENT_NETWORK_ARCHITECTURE.md](CURRENT_NETWORK_ARCHITECTURE.md)。

## 1. 先建立整体认识

任务是：已知起报时刻 T 的 ERA5 分析场与此前观测，预测未来 6 小时的 ERA5。

```text
磁盘上的 ERA5 / ZTD
        ↓ Dataset：选择时刻、对齐网格、处理缺测
一个样本的字典
        ↓ DataLoader：组成 batch；分布式采样器分配样本
        ↓ move_batch：搬到当前 GPU
DAForecastNet：ERA5 编码 + OBS 编码 → 双尺度融合 → FuXi 主体 → 预报
        ↓ weighted_losses：与未来 ERA5 比较
训练：backward → 梯度累积/同步 → optimizer.step
验证：只前向 → 汇总 loss → 选择 best.pt
评估：恢复 checkpoint → 前向 → 汇总指标 → CSV / JSON / 动图
```

三个概念不要混淆：

- **Dataset** 决定一个样本里面是什么。
- **Model** 决定如何把输入变成预测。
- **训练循环** 决定如何利用预测误差更新参数。

## 2. 推荐阅读顺序

第一遍只读以下五处，不必从头钻进注意力公式：

1. [配置文件](configs/train_da_forecast.json)：知道训练什么、数据在哪里。
2. [DAForecastDataset.__getitem__](data/da_dataset.py)：知道输入和标签如何生成。
3. [DAForecastNet.forward](model/da_forecast_model.py)：知道网络的数据流。
4. [weighted_losses 和 main](train.py)：知道 loss 与更新过程。
5. [evaluate.main](evaluate.py)：知道 checkpoint 如何变成测试指标。

第二遍再看 ObsEncoder、GatedFusion、FSDP 包装和预训练权重映射。
最后再看 [model/fuxi.py](model/fuxi.py) 中的 SwinBlock / WindowAttentionV2。

可以用以下只读命令快速定位函数：

```bash
rg -n '^class |^def |^    def ' train.py evaluate.py data/da_dataset.py model/da_forecast_model.py
```

## 3. 项目中各文件负责什么

| 文件 | 主要职责 |
|---|---|
| [train.py](train.py) | 训练入口、数据加载、FSDP、loss、优化器、验证、checkpoint |
| [evaluate.py](evaluate.py) | 评估入口、分布式测试、持久性基线、反归一化与动图 |
| [configs/train_da_forecast.json](configs/train_da_forecast.json) | 数据、模型、训练超参数 |
| [data/da_dataset.py](data/da_dataset.py) | ERA5/ZTD 时空对齐、静态场、样本字典 |
| [model/da_forecast_model.py](model/da_forecast_model.py) | 当前实际训练的 DAForecastNet，以及 OBS 与融合分支 |
| [model/fuxi.py](model/fuxi.py) | FuXi 基础模块及原始 UTransformer；当前模型复用其基础模块 |
| [model/fuxi_pretrained.py](model/fuxi_pretrained.py) | 原始 FuXi checkpoint 的名字映射、完整性检查和加载 |
| [preprocessing/build_global_era5_zarr.py](preprocessing/build_global_era5_zarr.py) | 校验并链接/复制现有已标准化 ERA5；不加入 IMERG |
| [preprocessing/build_global_ngl_zarr.py](preprocessing/build_global_ngl_zarr.py) | 原始站点文件转全球 5min Zarr，按训练时段统计均值方差 |
| [evaluate_original_fuxi.py](evaluate_original_fuxi.py) | 原始 FuXi 权重的单独基线评估，不是主训练入口 |
| tests/ | 小网络、数据处理、权重映射、FSDP 恢复的测试 |
| work_dirs/ | 日志、配置快照、checkpoint 和评估结果，不是网络源码 |

注意：当前 `train.py` 创建的是 `DAForecastNet`，不是直接创建 `UTransformer`。

## 4. 先读配置：哪些参数控制什么

### 4.1 data：决定样本

关键参数是 `era5_path`、`ztd_path`、`train_split`、`val_split`、`test_split`、
`obs_history_hours` 和 `obs_interval_minutes`。

当前时段均为左闭右开：

| 用途 | 区间 | 有效样本数 |
|---|---|---:|
| 训练 | 2023-01-01 ～ 2024-05-01 | 1942 |
| 验证 | 2024-05-01 ～ 2025-01-01 | 979 |
| 测试 | 2025-01-01 ～ 2026-01-01 | 1459 |

`resolve_data_paths()` 将主要数据路径相对于“配置文件所在目录的上一级”解析。
因此移动配置文件到别的目录，可能改变相对路径含义。

`mean_std_dir` 供评估反归一化使用；训练 Dataset 直接读取已标准化的 ERA5/ZTD，
不是每取一个样本就重新计算均值方差。

预报提前量目前在 Dataset、时间编码及部分评估代码中写为 6h。
**仅修改观测历史长度，不会把预报提前量改成其他时长。**

### 4.2 model：决定结构

主要参数：全球形状 `720×1440`、ERA5 70 通道/2 帧、特征宽度 1536、
patch size 4、窗口大小 9、4 个 stage 且每个 12 个 SwinBlock、24 个注意力头。

`use_obs=true` 会创建 OBS 编码器和两套融合模块；`false` 时这三个模块不创建。
但当前 Dataset 不会因为 `use_obs=false` 就自动停止读取 ZTD。

### 4.3 train：决定怎么优化

核对本文时，当前配置为：

```text
batch_size_per_gpu = 4
gradient_accumulation = 4
learning_rate = 2e-4
warmup_steps = 100
epochs = 40
loss_mode = global_only
freeze_fuxi = true
amp = bf16
```

完整累积组的有效 batch = 每卡 batch × GPU 数 × 累积次数。
四卡下，当前配置为 `4×4×4=64`。

**历史 best 实验不同：每卡 batch=1，有效 batch=16。**
`work_dirs/da_forecast_global_6h_fsdp_train_obs/best.pt` 对应的是旧实验，
不能用后来修改的 JSON 推断它当时的训练参数；应看 checkpoint 内的 `config`
或该实验的 `resolved_config.json`。

当前默认输出目录使用 `...fsdp_train-obs`，历史目录使用 `...fsdp_train_obs`，
连字符与下划线不是同一个目录。训练的 `--output-dir` 可以覆盖配置。
重定向日志到某目录，并不会自动改变 checkpoint 的输出目录。

## 5. 跟踪一个实际时间样本

假设起报时刻是 `2024-04-01 12:00`：

```text
ERA5 输入：2024-04-01 06:00 和 12:00
ZTD 输入：06:00、06:20、……、11:40、12:00，共 19 帧
监督标签：2024-04-01 18:00 的 ERA5
```

模型没有把每个站点当成独立训练样本。一个样本包含整个全球网格以及这些时刻的观测。
也不是将 19 帧 ZTD 当作 19 个独立样本。

为什么训练样本是 1942？训练区间共 486 天，理论上 `486×4=1944` 个起报时刻。
第一个起报时刻缺少前一天的 ZTD；最后一个起报时刻的标签越过训练边界，
各排除一个，得到 `1944−2=1942`。

ERA5 文件包含 2017～2025 年，不代表当前训练使用了全部年份。
当前配置选择 2023～2024 年训练，并要求对应的历史 ZTD 时间存在。

## 6. Dataset 内部发生了什么

对应 [data/da_dataset.py](data/da_dataset.py)。

### 6.1 初始化：先列出能使用的时刻

`DAForecastDataset.__init__()` 主要完成：

1. 以只读方式打开 Zarr，检查 ERA5 `z` 的 70 通道及坐标维度。
2. 用 `_decode_time()` 解码时间，用 `label_lookup` 建立时间到数组下标的映射。
3. 将站点实际经纬度映射到 ERA5 网格，构建静态场。
4. 遍历候选 T，要求 T 在当前 split、T+6h 不越界、T−6h/T+6h 的 ERA5 存在，
   且所需的 19 个观测时间戳都存在。
5. 将合法 T 存入 `analysis_times`；`__len__()` 就是其长度。

这里要求的是观测时间轴存在，不是要求每个站点每帧都有效；站点缺测由 mask 处理。
验证/测试可以读取边界之前的历史输入，但其标签仍处于自身 split，不属于未来信息泄漏。

### 6.2 取样：`__getitem__(index)`

ERA5 路径：读取连续三帧 `[T−6h,T,T+6h]`，前两帧作输入，后一帧作标签。
标签非有限值先记录进 `target_mask`，随后填零；输入中的非有限值也填零。
**填零不等于监督它为零：loss 中还会使用 target_mask 排除无效标签。**

ZTD 路径：`_obs_indices(T)` 查出 19 个时次，用步长切片读取，再通过
`project_station_values()` 投影到 ERA5 网格；同一个 ERA5 格点中的有效值取平均。

不要混淆两种 mask：

- Zarr 中的 `mask=True` 表示该格点没有选定站点。
- 模型输入 `obs_mask=1` 表示该时刻、该格点观测有效。

缺测值填零，同时提供 mask，因为“标准化值等于零”本身也可能是一条有效观测。
当前网络读取 `ztd`，即使 Zarr 里还有 `zwd`，也不会自动把它输入网络。

### 6.3 DataLoader 加上 batch 维后的字典

下面 B 指单卡 batch，H=720，W=1440；Dataset 单样本不带 B。

| 字段 | 形状 | 含义 |
|---|---|---|
| `era5` | `[B,2,70,H,W]` | 两帧标准化分析场 |
| `era_static` | `[B,5,H,W]` | 对齐后的 FuXi 静态场 |
| `time_features` | `[B,12]` | 三个时刻的日期/小时编码 |
| `obs_ztd` | `[B,19,1,H,W]` | 标准化 ZTD，缺测填零 |
| `obs_mask` | `[B,19,1,H,W]` | 动态观测有效性 |
| `obs_static` | `[B,3,H,W]` | 站点分布、归一化纬度、归一化经度 |
| `target` | `[B,70,H,W]` | T+6h 的标准化 ERA5 |
| `target_mask` | `[B,70,H,W]` | 标签有效性 |

当前这些返回张量都是 float32，包括 mask。bool mask、静态场 GPU 缓存等优化尚未实现。
70 个 ERA5 通道为 z/t/u/v/r 各 13 个气压层，再加 t2m、u10、v10、msl、tp。
tp 在当前任务中也是 ERA5 通道，不是 IMERG。

### 6.4 两类静态场和时间编码

`era_static` 来自 `model/conds.npy`，通过坐标对齐后送入 ERA5 encoder。
当前加载代码没有逐通道命名这些静态场；不能仅凭“五个通道”猜测各自物理含义，
需要追溯 conds.npy 的生成定义。

`obs_static` 则由代码明确构造：站点分布 mask、缩放到约 −1～1 的纬度、经度。
它的第一通道是固定站点分布，不等于每帧实际有效观测 mask。

`time_features()` 对 T−6h、T、T+6h 分别使用 `dayofyear/366` 和 `hour/24`，
做 sin/cos 后拼成 12 维。当前没有额外乘 `2π`，这是对齐原始 FuXi 的实现。
T+6h 的日历信息是已知量，不是读取未来气象观测。

## 7. 从 forward 看懂整个网络

对应 [DAForecastNet.forward](model/da_forecast_model.py)。以下是等价阅读伪代码，
不是另一个需要运行的脚本：

```python
temb = time_embed(time_features)
high = era_encoder.encode_high(era5, era_static)
if use_obs:
    obs_high, obs_low, high_mask, low_mask = obs_encoder(obs_ztd, obs_mask, obs_static)
    high = high_fusion(high, obs_high, high_mask)
low = era_encoder.encode_low(high, temb)
if use_obs:
    low = low_fusion(low, obs_low, low_mask)
bottleneck = backbone(low)
residual = decoder(bottleneck, low, temb)
prediction = era5[:, -1] + residual
```

最后一行最重要：decoder 输出的是增量，整个网络返回的才是未来分析场。
loss 的 target 是未来 ERA5 本身，不要在训练或评估外部再额外加一次 ERA5(T)。

## 8. 分模块理解网络和形状

### 8.1 ERA5Encoder：两帧天气场如何变成特征

每帧 70 个动态通道与相同的 5 个静态通道拼接，两帧展平成 150 通道：

```text
[B,2,70,720,1440] + 静态场
→ [B,150,720,1440]
→ CubeEmbed：kernel=4，stride=4 的 Conv2d + LayerNorm
→ high：[B,1536,180,360]
→ 高分辨率 OBS 融合
→ DownBlock2D（带时间调制，空间再缩小 2 倍）
→ low：[B,1536,90,180]
→ 低分辨率 OBS 融合
```

当前 CubeEmbed 名称虽然含 Cube，但实现是先合并时间/通道，再做 Conv2d。
1/4 和 1/8 指高、宽分别缩小到原来的 1/4 和 1/8，不是通道数缩小。

### 8.2 ObsEncoder：19 帧观测如何融合

先把 ZTD 和 mask 拼在通道维，再变为 Conv3d 需要的 `[B,2,19,H,W]`。
两层 3D 卷积产生 32、64 通道特征；之后沿时间维按有效 mask 聚合。
所以时间信息先经过卷积，并不是对原始 19 帧直接求平均。

聚合结果与站点静态场拼接，通过 patch 卷积与残差模块得到：

```text
obs_high：[B,1536,180,360]，high_mask：[B,1,180,360]
obs_low ：[B,1536, 90,180]，low_mask ：[B,1, 90,180]
```

缺测位置的特征会被 mask 屏蔽。OBS encoder 本身没有 Transformer，
观测通过融合进入低分辨率特征后，才共同经过 FuXi Transformer。

### 8.3 GatedFusion：门控到底控制什么

两套融合模块分别作用于 high 和 low，核心计算是：

```text
delta = 两层卷积([ERA 特征, OBS 特征, patch_mask])
gate = sigmoid(gate_logit)
output = ERA 特征 + available × gate × delta
```

`gate_logit` 是 `[1,1536,1,1]` 的可训练参数，初始化 −4，sigmoid 后约 0.018。
这是每特征通道一个、空间共享的参数门控，不是每个站点或像素一个门控。
delta 随输入变化，但 gate 本身不是由当前观测即时计算的。

`available` 是当前样本 patch_mask 的全空间最大值。因此：

- 整个样本都没有有效观测时，两次融合都退化为原 ERA5 路径。
- 只要某处有观测，修正就可能作用于全域，不只作用于站点附近。
- 0.018 是 delta 的缩放系数，不代表最终预报误差或输出改变量只有 1.8%。

### 8.4 ForecastBackbone：Transformer 与 FPN

low 展平为 `[B,16200,1536]`，因为 `90×180=16200`。
经过 4 个 stage，每个含 12 个 SwinBlock；窗口大小 9×9，24 个注意力头。
注意力主要在局部窗口内计算，移位窗口实现跨窗口交换，不是所有全球 token 的稠密全连接注意力。

当前四个 stage 的空间分辨率相同，都在 90×180。
FPN 收集四个 stage 经 LayerNorm 的输出，沿特征维拼接成 6144 通道，
再经 Linear + GELU 压回 1536；这里不是四个不同空间尺度的金字塔。

`torch.utils.checkpoint.checkpoint` 表示激活重计算：训练时少存部分中间结果，
反向时重新计算，以时间换显存。它与保存 best.pt 的 checkpoint 不是一回事。

### 8.5 ForecastDecoder：如何恢复全球网格

```text
Transformer 输出 + low skip（通道拼接）
→ [B,3072,90,180]
→ UpBlock2D，恢复到 [B,1536,180,360]
→ Linear(1536,70×4×4)
→ 把每个位置的 4×4 patch 重排回 [B,70,720,1440]
→ 加 ERA5(T)，得到 prediction

```

当前 skip 是经过融合的 low，不是 high 直接绕过 Transformer 后连接 decoder。
也不要把最后的 patch 重排理解为双线性插值。

FuXi 主体使用经度周期卷积；OBS 与融合模块仍使用普通 padding，边界处理并不完全一致。

## 9. 预训练、冻结和 FSDP 的关系

### 9.1 为什么需要权重名字映射

原 FuXi 把很多模块放在 `decoder` 下面，当前网络按职责拆成多个模块，名字变了。
[rename_fuxi_key()](model/fuxi_pretrained.py) 做的就是对应关系，例如：

| 原始 checkpoint 名字前缀 | 当前模型前缀 |
|---|---|
| `decoder.patch_embed.` | `era_encoder.patch_embed.` |
| `decoder.down_blocks.0.` | `era_encoder.down_block.` |
| `decoder.layers.` | `backbone.layers.` |
| `decoder.norm0.` | `backbone.norms.0.` |
| `decoder.fpn.` | `backbone.fpn.` |
| `decoder.time_embed.` | `time_embed.` |
| `decoder.up_blocks.0.` | `decoder.up_block.` |
| `decoder.head.` | `decoder.head.` |

加载器检查主体键是否齐全、形状是否一致，只有 OBS 与两套融合模块允许保持新初始化。
不是看到 `strict=False` 就意味着缺失的主体权重会被静默忽略；外围代码另有完整性校验。

### 9.2 freeze_fuxi 冻结哪些东西

冻结 `era_encoder`、`time_embed`、`backbone`、`decoder` 的参数，
只将 OBS encoder、high/low fusion 的可训练参数交给 AdamW。

**冻结参数不等于阻断梯度。** loss 仍需通过主体向前面的 OBS 分支传播梯度。
如果把主体整个包在 `torch.no_grad()` 中，反而可能切断 OBS 学习路径。

当前完整模型约 17.92 亿参数，其中 FuXi 主体约 15.57 亿，新增分支约 2.35 亿。
因此“只训练 OBS”仍然不是一个很小的网络。

启动时必须区分：

- `--pretrained-fuxi model/fuxi_short.pth`：新实验，从原始 FuXi 初始化。
- `--resume .../last.pt`：继续当前 DA 模型的训练，并恢复优化器、学习率状态。
- 两者不能同时使用。
- `freeze_fuxi=true` 却两者都不传，会冻结随机主体；当前代码没有禁止这个组合。

### 9.3 FSDP 并不是把全球地图分给四张卡

`wrap_fsdp()` 使用 FULL_SHARD，并单独包装每个 SwinBlock。
每个 rank 仍处理完整全球样本；切分的是参数、梯度和优化器状态，不是把地球切成四块。
前向需要某模块时，FSDP 收集相应参数；反向时再同步、分片梯度。

训练首次初始化顺序：创建模型 → rank 0 加载原始权重 → 冻结指定参数 →
FSDP 用 `sync_module_states=True` 同步参数并分片。

多 GPU 需要分布式 launcher。直接设置四张可见 GPU 后运行 `python train.py`
不会自动得到四个训练进程；FSDP 模式下当前代码会检查 world_size。

## 10. 一次训练更新到底经过哪些步骤

对应 [train.main](train.py)。

1. 读取配置、初始化 rank/device、设随机种子。
2. 构建 Dataset、DistributedSampler、DataLoader。
3. 创建并初始化模型、冻结参数、包装 FSDP。
4. 创建 AdamW、warmup/cosine scheduler；必要时恢复 checkpoint。
5. 每个 epoch 对训练 sampler 调用 `set_epoch()`，改变 shuffle 顺序。
6. 读取一个 micro-batch，经 `move_batch()` 搬到 GPU，再前向计算 loss。
7. loss 除以累积次数，反向传播；未到更新时使用 FSDP `no_sync()` 延后梯度同步。
8. 累积组结束：同步、梯度裁剪、optimizer.step、清梯度、scheduler.step。
9. 遍历完训练数据后验证，比较 best_val，保存 last/best。

术语对照：

- sample：一个起报时刻对应的全球样本。
- micro-batch / batch_index：一次 DataLoader 返回、一次前向反向。
- optimizer step：累积完成后真正执行一次更新；日志中的 `step` 指这个。
- epoch：当前训练 DataLoader 被完整遍历一遍，不等于一个 step。

四卡、1942 个样本、当前 drop_last 规则下：

| 设置 | 每卡 loader 批数 | 每 epoch 更新次数 | 每 epoch 实际使用样本 |
|---|---:|---:|---:|
| 历史每卡 B=1，累积 4 | 485 | 122 | 1940 |
| 当前每卡 B=4，累积 4 | 121 | 31 | 1936 |

先由 sampler 每卡分到 485 个样本，再由 DataLoader 按 batch size 丢弃尾部不足一批的数据。
有效 batch 16/64 只适用于完整累积组，epoch 最后一组可能更小。

学习率按 optimizer step 变化，不是按 micro-batch 或 epoch 变化。
同样 `warmup_steps=100`，改变 batch 后对应的 epoch 数也会改变。
BF16 是混合精度计算，不代表磁盘数据、CPU 张量、模型主参数全部变成 BF16。

## 11. loss 如何计算

对应 [weighted_losses()](train.py)。令 p 为预测，y 为标签，m 为标签有效 mask，
纬度权重为 `w=cos(latitude)`，再除以纬度权重均值：

```text
global_loss = sum(|p-y| × m × w) / sum(m × w)
```

求和覆盖 batch、70 个通道和全球空间。它是标准化空间的 MAE，不是 MSE，
不是把不同气象变量的物理单位直接相加；当前也没有额外的变量/气压层权重。

站点 halo 来自 `obs_static[:, :1]`，经 max_pool2d 膨胀。
radius=3 对应 7×7 格点邻域，是网格半径，不是全球统一公里数。
halo_loss 使用同样的误差公式，但权重额外乘 halo mask。

| loss_mode | 用来反向传播、挑选 best 的指标 |
|---|---|
| `global_only` | global_loss |
| `halo_only` | halo_loss |
| `global_plus_halo` | global_loss + halo_weight × halo_loss |

当前是 `global_only`，所以 halo_weight 不改变训练目标，val_halo 只是附加监控指标。
halo 基于固定站点分布，不是每个样本实际有效站点的动态邻域。

## 12. 怎么读日志和 checkpoint

iteration 日志中的 batch_loss 是当前 rank 最后一个 micro-batch 的 loss，
不是所有 GPU、全部累积样本的平均；目前四个 rank 都打印，所以同一步可能有四条。
打印 lr 时 scheduler 已经推进，所以显示的是下一次更新的学习率。

epoch 行中的 train 是 batch loss 汇总；val/val_global/val_halo 来自验证汇总。
当前验证是先计算每个 batch 的 loss，再跨 batch/rank 平均。
`seconds` 在保存 checkpoint 前统计，不含后续写 checkpoint 的耗时。

[save_checkpoints()](train.py) 收集完整 CPU 状态后，由 rank 0 写临时文件再替换正式文件。
`last.pt` 每个完成的 epoch 更新，`best.pt` 仅在 val 改善时更新。

文件主要包含 model、optimizer、scheduler、config、epoch、step、best_val 等。
checkpoint 的 epoch 从 0 开始，日志从 1 开始。
不要只凭文件名判断模型结构或实验设置，应优先看它保存的 config。

## 13. evaluate 与训练验证有什么区别

对应 [evaluate.py](evaluate.py)。

### 13.1 配置来源并不是完全相同

- 模型结构、loss_mode、halo 参数、精度：优先读取 checkpoint 保存的配置。
- 数据路径和时间划分：读取命令行 `--config` 指向的配置。
- 评估 batch/worker：读取 `--batch-size`、`--num-workers`，当前默认分别为 1、4。
- `--split val` 选验证集；默认 `--split test` 选测试集。

所以修改训练 JSON 的 batch，不会自动改变 evaluate 的 batch；
而修改数据 split 会影响下一次 evaluate。比较实验时要同时保存 checkpoint 和数据配置。

### 13.2 多卡权重恢复为什么必须特别处理

修复后的 evaluate：rank 0 在未包装模型上严格加载完整权重，随后 FSDP 强制同步并分片。
修复后的训练 resume：每个 rank 内存映射读取 checkpoint，严格加载完整模型状态；
优化器状态仍采用分发恢复。

旧写法“先包装，然后 rank 0 加载、其他 rank 传空字典”不会自动广播模型权重。
`rank0_only=True` 控制状态导出，不等于加载时广播。旧分布式测试结果不能当作真实模型表现。

### 13.3 指标怎样跨卡汇总

`DistributedEvalSampler` 保证各 rank 迭代长度相同；补齐条目标记 `_eval_valid=False`。
`IndexedEvalDataset` 附上样本索引，保证补齐样本不计入指标、动画能按正确时间归并。

每个通道分别累计 `sum(error × weight)` 和 `sum(weight)`，使用 float64 累加，
各 rank all_reduce 后统一相除。整体指标是所有通道的误差和除以所有通道的权重和。
当前训练验证仍会重复计算 sampler 补齐样本，因此两者不要求逐位相等。

持久性基线：直接用 ERA5(T) 当作 T+6h 的预测，再与同一标签比较。
提升比例为 `(基线 MAE − 模型 MAE) / 基线 MAE × 100%`；正值表示模型更好。

### 13.4 反归一化和输出

`inverse_all()` 先执行 `x×std+mean`，再处理变量单位：z 除以 9.80665，
msl 除以 100，tp 按既有 log1p 预处理约定做 expm1 并截到非负。
tp 是非线性变换，因此物理 MAE 不能简单由标准化 MAE 乘 std 得到。
不同变量的物理单位不同，物理 MAE 按通道报告，不合成一个跨单位平均值。

结果包括 `weighted_mae_summary.json`、`weighted_mae_by_channel.csv`、
`physical_mae_by_channel.csv` 与可选的 animations/。
JSON 的 evaluation 字段记录 checkpoint 路径、epoch、step、best_val、split 和加载方式。
默认动画日期为 2025-05-01～05-05；复核验证集时应加 `--skip-animations`，
因为该日期不在当前验证区间内。

日志中的 `Checkpoint loaded` 只说明模型已恢复，不说明第一个 batch 已完成。
之后目前每 20 个本地 batch 打印一次进度；最终结果文件才在汇总后写出。

## 14. 已知问题与尚未实现的优化

不要把以下内容误认为已经修复：

| 当前行为 | 阅读/运行时注意 |
|---|---|
| 尾部累积组仍固定除以 accumulation | 最后一组不足时梯度被额外缩小 |
| 验证 DistributedSampler 自动补齐 | 979 个样本四卡会按 980 次计入训练验证 |
| log_interval 没有用于控制打印 | 四个 rank 每次更新都可能打印 |
| FSDP 关闭但仍多进程启动 | 没有 DDP 替代，各卡独立更新，不是正确的联合训练 |
| FP16 使用普通 GradScaler，且未保存 scaler 状态 | 不应把已验证的 BF16 路径直接等同于 FP16 完整可靠 |
| resume 只显式检查 loss_mode | 改 epochs/batch/累积/数据区间可能改变续训语义与调度 |
| Dataset 总是读取 ZTD | 关闭网络 OBS 不是读取层面的消融优化 |
| mask 使用 float32，静态场重复返回 | CPU 内存和传输存在压缩空间 |
| ZTD 按稠密全球网格、一天时间块存储 | 只取 19 帧也可能产生较大的解压开销 |

DataLoader 当前没有显式设置 spawn、prefetch_factor=1 或关闭 pin_memory；
这些只是此前讨论过的优化方向。station-major ZTD 存储也尚未实现。

B=1 时返回张量总量约 1.26 GiB。四 rank × 四 worker × 默认预取两批，
理论上仅预取样本就可能约 40 GiB；B=4 时还会放大，不包括临时解压数组。
FSDP 分片模型参数，不会自动分片这些 CPU 数据。

近期遇到的日志停滞同时有内核内存压力证据；不能仅凭 CPU 100% 就认定在做有效计算，
也不能保证改 worker 数就解除节点级阻塞。需要区分数据处理、磁盘、CPU 内存与 GPU 显存。

## 15. 如何安全地边看边验证

以下均从项目根目录、正确 Python 环境执行；本文没有替你启动它们。

先跑小网络单元测试，不需要完整全球数据：

```bash
python -m unittest discover -s tests -v
```

看 tests 中的 `make_model()`、`make_inputs()`，先用 `[1,2,4,40,40]` 的小输入
理解形状，再对应回全球版本。小测试通过不等于全球 batch=4 一定装得进显存。

数据检查会真实读取一个全球样本，节点阻塞期间不要反复启动：

```bash
python train.py --config configs/train_da_forecast.json --check-data
```

FSDP 恢复回归测试使用小模型，但需要空闲 GPU；检查参数、预测、优化器和下一次更新一致性：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  tests/fsdp_checkpoint_regression.py
```

`python -m torch.distributed.run` 是创建多个进程的 launcher，实际运行的项目入口
仍是其后面的脚本；不要写成 `python test_da_forecast_model.py -m unittest`。

`--overfit-samples` 会把 train/val 都改成同一小子集，仅用于调试，不能用于正式泛化评估。
子集必须足够组成各卡 batch；例如四个样本、两卡、每卡 batch=4 时，drop_last
可能让训练 loader 为空。不要直接照搬旧 batch=1 时的小样本命令。

## 16. 想改某件事，先去哪里看

| 想理解/调整的内容 | 首先看 |
|---|---|
| 为什么样本这么少、某时刻为什么被排除 | Dataset.__init__、analysis_times |
| 观测取多长时间、多久一帧 | data 配置、observation_offsets、_obs_indices |
| 观测如何匹配 ERA5 网格 | station_grid_mapping、project_station_values |
| 新增/去掉某个输入字段 | Dataset.__getitem__ → forward_model → DAForecastNet.forward 三处一起追踪 |
| 只看 ERA5、不使用 OBS | use_obs 与 freeze_fuxi 的组合，以及 Dataset 是否仍读 ZTD |
| 融合发生在哪里、gate 的意义 | GatedFusion、DAForecastNet.forward |
| 学习率为什么这样变 | lr_factor、scheduler.step、optimizer_step |
| loss 是全球还是站点附近 | weighted_losses、loss_mode、obs_static 第一通道 |
| 预训练哪些权重被继承 | rename_fuxi_key、load_fuxi_pretrained |
| 测试指标为什么不等于日志 val | 数据 split、权重恢复、精度、补齐与聚合方式 |
| 日志为何不更新 | main 当前阶段、DataLoader、每 20 batch 打印的位置、系统压力 |

如果只记住一句话：先搞清楚一个样本的每个字段，再沿 `forward_model →
DAForecastNet.forward → weighted_losses → backward/step` 追踪它，就能把项目串起来。
