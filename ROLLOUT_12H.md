# 12h 自回归：真实观测生成第一跳，第二跳仅微调 FuXi

## 数据和时间定义

起报时刻为 T，源模型固定使用：

```text
work_dirs/da_forecast_stage2_finetune_fuxi/best.pt
```

1. 第一跳输入 `[ERA5(T-6), ERA5(T)]` 和真实 `ZTD(T-6..T)`，每 20min 一帧，共 19 帧。源模型保持 `use_obs=true, zero_ztd=false`，生成预测 T+6。时间编码为 T。
2. 第二跳输入 `[ERA5(T), 预测(T+6)]`，标签为 ERA5(T+12)。时间编码为 T+6，网络预测最后一个 6h 步长。
3. 第二跳 `use_obs=false`：不构建 OBS encoder 或 high/low fusion，不读取任何时刻的 ZTD 数据。不再进行“先训练零观测分支”的阶段。
4. 第一跳预测固定，不随第二跳训练更新，也不会反向传播到源模型。这是固定上游模型的离线 rollout 训练，不是端到端跨两步反传。
5. split 为半开区间，T 和 T+12 都在对应 split 内。测试不会读取 T 之后的观测；第一跳也不读取未来 ERA5 作为输入。

## 独立新 Zarr

输出：

```text
datasets/era5_rollout12h_stage2_realobs.zarr
```

不替换、不修改原始 ERA5/ZTD。保存两个四维数组，不使用五维 sample-input 或逐样本 NPZ。

| 数组 | 维度 | 内容 |
|---|---|---|
| z | time, channel, lat, lon | 原始标准化 ERA5 真值；同一时刻只存一次 |
| z_fuxi6h | init_time, channel, lat, lon | 在 init_time=T 起报、T+6 有效的第一跳预测 |
| time / init_time | 各自的一维时间轴 | 显式 int64 纳秒，附 units/calendar |
| era_complete / forecast_complete | 各自的一维标记 | 完整写完一行所有 chunks 后置 true |
| era_static | static_channel, lat, lon | 与生成时一致的五个静态场 |
| metric_station_mask | lat, lon | 只供 halo 指标统计，不进入 OBS 分支 |

`z` 保留原来的字段名、四维布局、通道顺序、坐标及标准化值。预测也保存 FP32，使用 Zstd 无损压缩；BF16 推理本身的精度不等于 FP32 推理。

**不是把预测混入 z 的普通时间序列。** 同一时刻的真值和预测不同，因此读取器需要一个很小的分支：
`z[T]`、`z_fuxi6h[T]` 作为输入，`z[T+12]` 作为标签。原有 6h 读取路径不变。
`train.py` 的优化循环无需为 rollout 重写；仅新增数据分派及 FuXi-only 初始化。

## 已核对的规模

当前配置（2026-10-30 元数据检查）：

| split | 起报样本数 |
|---|---:|
| train | 1941 |
| val | 978 |
| test | 1458 |
| 合计 | 4377 |

`z.shape=(4383,70,720,1440)`，`z_fuxi6h.shape=(4377,70,720,1440)`。
去重后的真值和预测合计约 **2368.41 GiB / 2.31 TiB，未压缩**；实际磁盘占用取决于压缩率。
这包含独立 ERA5 副本，不能再按只保存预测的容量估计。

共享文件系统空闲空间不等于个人配额。脚本检查预计新增未压缩容量并留 5% 余量，个人配额需自行确认。
只生成 train/val 后可先训练，再单独补 test。

## 文件

- `build_rollout_zarr.py`：真实观测第一跳推理、ERA5 去重复制、断点续写、空间预估。
- `data/rollout_fields.py`：双四维 Zarr 写入、完整性验证和读取。
- `data/rollout_identity.py`：源文件身份及静态场/归一化指纹。
- `model/rollout_init.py`：严格提取完整 DA checkpoint 的 FuXi 主体权重。
- `configs/train_da_forecast_12h_fuxi.json`：真实观测第一跳的 12h 训练配置。
- 零 ZTD 第一跳、无 OBS 第二跳的独立 baseline：见 [BASELINE_12H_ZERO_ZTD.md](BASELINE_12H_ZERO_ZTD.md)。
- `tests/test_rollout_fields.py`：时间精度、输入/标签、续写、无观测等价性和主体梯度测试。

以下命令在项目根目录、已激活的 gnss 环境执行。不要与已有占满 GPU 的任务同时运行。

## 1. 只检查规模

```bash
python build_rollout_zarr.py --config configs/train_da_forecast_12h_fuxi.json --plan
```

此命令不写 Zarr，不加载大模型，不使用 GPU。配置中的数据源文件必须保持不变。

## 2. 四个样本的真实四卡试运行

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc_per_node=4 build_rollout_zarr.py \
  --config configs/train_da_forecast_12h_fuxi.json \
  --splits train --max-samples 4 --batch-size 1 --num-workers 2
```

先复制所需 ERA5，再加载有 OBS 的源模型生成 T+6。
4 个连续起报样本需要 6 个去重 ERA5 场和 4 个预测场，约 2.70 GiB 未压缩。
默认使用 BF16；不要在同一个输出目录中混用 BF16/FP32。

只有部分样本完成时，完整训练读取会报 `incomplete Zarr`，这是保护机制。全量命令可接着补齐。

## 3. 后台生成全部 split

```bash
mkdir -p work_dirs/rollout_12h
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
  --standalone --nproc_per_node=4 build_rollout_zarr.py \
  --config configs/train_da_forecast_12h_fuxi.json \
  --splits train val test --batch-size 1 --num-workers 2 \
  > work_dirs/rollout_12h/build_zarr.log 2>&1 < /dev/null &
```

相同命令重跑会跳过完成的行，重写未完成的行。源 checkpoint/精度/配置不匹配时会拒绝续写，不能自动覆盖。

四卡使用 FSDP 推理，padding 样本不落盘；各 rank 写不同时间行，完成标记也按单行分块。
耗时的磁盘阶段使用独立 Gloo 控制组同步，避免在默认 10 分钟 NCCL barrier 中等待长时间复制。
日志会区分 ERA5 copy 与 forecast batch；复制期间 GPU 空闲是正常的。

同一输出只有一个任务能拿到 `.writer.lock` 的文件锁。进程退出/崩溃后 OS 会释放锁，
旁边留下的锁文件无害，**不要手动删锁来绕过仍在运行的任务**。

记录了源 checkpoint 路径/大小/mtime，静态场和归一化文件 SHA256，以及配置、精度、起报时间轴。
没有对多 TB ERA5/ZTD 或 checkpoint 全内容计算哈希：这些源文件须保持不可变。
本次实现不会自动修改既有实验 best.pt。

## 4. 检查生成结果

train 和 val 均完成后：

```bash
python train.py --config configs/train_da_forecast_12h_fuxi.json --check-data
```

预期：`era5=(2,70,720,1440)`、`target=(70,720,1440)`；
dummy `obs_ztd` 和 `obs_mask` 都全零；`target_mask` 保留 ERA5 原始缺测状态。
站点静态 mask 仅用于计算 halo 指标；训练 loss 为全球纬度加权 MAE。

## 5. 从 6h checkpoint 初始化，仅微调 FuXi

```bash
mkdir -p work_dirs/da_forecast_12h_fuxi_only
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
  --standalone --nproc_per_node=4 train.py \
  --config configs/train_da_forecast_12h_fuxi.json \
  --init-checkpoint work_dirs/da_forecast_stage2_finetune_fuxi/best.pt \
  --init-fuxi-only \
  > work_dirs/da_forecast_12h_fuxi_only/train.log 2>&1 < /dev/null &
```

`--init-fuxi-only` 仅允许丢弃：
`obs_encoder.*`、`high_fusion.*`、`low_fusion.*`。
其余主体权重仍严格校验名称/形状，缺键或多键直接报错，不能用任意 `strict=False` 静默跳过。

训练：ERA encoder、time embedding、backbone/FPN、decoder 全部解冻。
`use_obs=false` 比“ZTD 数值为零但保留非零 mask”更彻底；后者仍可能产生观测分支修正。
当前模型在全零 mask 下，移除 OBS 分支与保留分支的输出等价，已有小模型测试验证。

默认学习率 1e-5，最低 1e-6，warmup=100 更新步，10 epochs。
四卡、每卡 batch=1、累积 8 次，有效 batch=32。建议先检查初始验证值，再观察前 1–2 epochs。
所有主体训练的显存开销高于冻结主体训练；真实四卡显存/吞吐尚需上面的试运行验证。

`--init-checkpoint` 不继承优化器、调度器或 epoch。训练开始前会保存 `initial_validation.json`。
同一次 12h 训练中断后用 `--resume .../last.pt`，不要再带两个初始化参数。

## 6. 测试

确认 test 预测已生成：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc_per_node=4 evaluate.py \
  --config configs/train_da_forecast_12h_fuxi.json \
  --checkpoint work_dirs/da_forecast_12h_fuxi_only/best.pt \
  --output-dir work_dirs/da_forecast_12h_fuxi_only/test \
  --batch-size 1 --num-workers 2 --skip-animations
```

训练与测试共用 `forward_model`，评估按 checkpoint 中保存的结构构建模型，
并校验时效及 rollout 来源；不会因当前 6h 配置而恢复 OBS 分支。
测试 batchsize 可以与训练不同，当前小数据入口测试覆盖 batchsize=1 和 3。

不指定 `--output-dir` 时，输出到 checkpoint 所在目录下的 `test/` 或 `val/`，
不再默认写入某个历史实验目录。

去掉 `--skip-animations` 可生成指定有效时间范围的动图。结果有效时间为 T+12，
persistence 为 ERA5(T)，不是预测(T+6)。同时保留全球/halo、标准化及物理单位指标。

12h 结果不能复用原始 FuXi 的 **6h** 对照 CSV。必须另外生成原始 FuXi 在相同起报样本上的两步自回归基线。
评估不再默认加载历史 FuXi 对照。未提供 `--fuxi-reference` 时照常输出 MAE CSV/JSON，
只跳过 FuXi 对比柱状图；有匹配的 12h 对照后，通过 `--fuxi-reference /path/to/channel_comparison.csv`
显式指定。6h 实验的对照绘图也需要显式传入此参数。
降水通道仍是源数据定义的 6h 累积降水，不能称为 12h 累积量。

## 7. 原始 FuXi 两步 12h 基线

`evaluate_original_fuxi.py` 已支持同一个 `model/fuxi_short.pth` 连续预测两步。
第一步输入 ERA5(T−6)、ERA5(T)，第二步输入 ERA5(T)、原始 FuXi 自己的预测(T+6)。
时间编码分别为 T、T+6，每一步都加回最新输入帧。两步都不使用 OBS，不做权重更新。

起报时间严格采用新 Zarr 的 test `init_time`，已核对当前为 **1458** 个样本。
只从新 Zarr 读取时间/坐标/静态场元数据，不读取 `z_fuxi6h`。
输入及 ERA5(T+12) 标签来自原始 ERA5 文件，所以不需要等待 DA 预测字段全部生成。
原始 ERA5、静态场和归一化文件必须保持与生成 Zarr 时一致。

先检查元数据，不用 GPU：

```bash
python evaluate_original_fuxi.py \
  --config configs/train_da_forecast_12h_fuxi.json --plan
```

先测试 4 个样本，结果与正式基线放在不同目录：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc_per_node=4 evaluate_original_fuxi.py \
  --config configs/train_da_forecast_12h_fuxi.json \
  --checkpoint model/fuxi_short.pth --amp bf16 \
  --max-samples 4 --batch-size 1 --num-workers 2 \
  --output-dir work_dirs/original_fuxi_12h_smoke
```

小样本标记 `is_full_test=false`，不能充当正式绘图基线。
确认四卡空闲、试运行正常后，后台统计完整测试集：

```bash
mkdir -p work_dirs/original_fuxi_12h
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
  --standalone --nproc_per_node=4 evaluate_original_fuxi.py \
  --config configs/train_da_forecast_12h_fuxi.json \
  --checkpoint model/fuxi_short.pth --amp bf16 \
  --batch-size 1 --num-workers 2 \
  --output-dir work_dirs/original_fuxi_12h \
  > work_dirs/original_fuxi_12h/evaluate.log 2>&1 < /dev/null &
```

输出 `summary.json` 和 70 行的 `channel_comparison.csv`：
全球纬度加权总体/逐通道标准化 MAE、逐通道物理 MAE、ERA5(T) persistence 基线，
以及时效、样本数、有效时间、精度、模型文件身份和 rollout 来源。
不保存全球预测场，不另生成 TB 级缓存。缺测标签先置零再加 mask，补齐的分布式样本不计入统计。

已存在 `summary.json` 时拒绝覆盖，请选新输出目录。文件锁防止同目录并发写入；
退出后锁会自动释放，锁文件可以保留。
当前没有逐样本指标断点续算：中断后重跑会从头计算。CPU 模式主要用于小模型测试；
真实全球模型建议四卡运行。当前默认配置是 12h，旧 6h 用法请显式加 `--lead-hours 6`；
`--start/--end` 仅适用于 6h，12h 不允许另外筛日期导致对照样本错位。

你的 12h 模型已完成 `evaluate.py` 后，无需再次推理，直接画图：

```bash
python evaluate.py --plot-only \
  --output-dir work_dirs/da_forecast_12h_fuxi_only/test \
  --fuxi-reference work_dirs/original_fuxi_12h/channel_comparison.csv
```

输出 `mae_vs_fuxi_normalized.png`、`mae_vs_fuxi_physical.png`、
`mae_vs_fuxi_improvement.png` 和逐通道对比 CSV/汇总 JSON。
绘图前检查完整样本数、12h 时效、有效时间、BF16 精度、rollout 来源及 persistence 一致性。
这比较的是整条 DA 模型链与原始 FuXi 模型链；不是单独证明 12h 微调本身带来的收益。

## 验证范围和后续

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

已提供合成 Zarr 生成→中断→补齐→读取的 CPU 测试，以及 no-OBS 主体加载、mask 等价性、
梯度与原有 6h 功能回归测试。另外覆盖真实 train/evaluate 入口的小模型训练、checkpoint 保存、
不同 batchsize 测试及续训，确认 6h checkpoint 的旧 optimizer/epoch 不会被当作续训恢复。
尚未自动运行真实全球四卡生成或训练。
原始 FuXi 基线另有 CPU 测试覆盖两次残差加回、预测反馈、时间编码、无未来真值输入、
缺测/补齐样本排除、不同 batchsize 一致性，以及输出 CSV 与对比图接口。
只执行过真实测试起报时间的元数据核对，尚未自动启动真实原始 FuXi 全量 12h 推理。

已清理旧的 `build_12h_inputs.py` 转发入口、`--allow-zero-ztd-change` 覆盖开关、
数据加载器内旧 `current_only` / 可变时效 / ZTD 置零分支，以及 12h 配置中无 OBS 时无效的冻结 OBS 设置。
现在只使用 `build_rollout_zarr.py` 生成输入；6h 数据读取仍由原数据集负责，
12h 样本筛选仅在生成脚本内完成，训练/评估读取 `RolloutFieldsDataset`。

18h/24h 需要继续保存前序模型预测并维护原始起报时间 T；当前实现仅支持这条固定两步 12h 模型链。
不能只把 `forecast_lead_hours` 改成 18/24 就直接运行。
