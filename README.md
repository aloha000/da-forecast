# da-forecast

第一次阅读项目，建议先看中文
[代码导读：从一个样本到一次训练更新](CODE_WALKTHROUGH.md)。
其中解释数据、网络、FSDP、loss、checkpoint、评估和已知注意事项，
并区分当前配置与历史实验参数。下文部分启动示例保留历史目录，请核对实际配置后使用。

当前模型的完整逐层结构见
[CURRENT_NETWORK_ARCHITECTURE.md](CURRENT_NETWORK_ARCHITECTURE.md)。

## 构建全球ERA5-only数据集

源 `ERA5_2017_2025` 已经是标准化后的全球70通道、720×1440、6小时Zarr，
约1.2 TB。默认命令会完整校验源数据，并在本项目 `datasets/` 下建立零拷贝
数据集入口；不会读取或写入IMERG：

```bash
python preprocessing/build_global_era5_zarr.py
```

输出：

```text
datasets/era5_global_0p25_6h.zarr
datasets/era5_global_0p25_6h.manifest.json
```

如果明确需要额外占用约1.2 TB的独立物理副本，可以使用可断点续传的copy模式：

```bash
python preprocessing/build_global_era5_zarr.py --mode copy
```

复制中断后继续：

```bash
python preprocessing/build_global_era5_zarr.py --mode copy --resume
```

## 构建全球NGL 5分钟Zarr

全球NGL源文件位于 `datasets/global_ngl/<STATION>/<STATION>.<YEAR>.trop.zip`。
下面的脚本保留2023—2025年所有原生5分钟记录，不会预先降采样到20分钟；
模型的数据加载器再按需每隔4帧抽样：

```bash
python preprocessing/build_global_ngl_zarr.py \
  --workers 32
```

默认输出：

```text
datasets/global_ngl_0p25_5min_2023_2025.zarr
datasets/global_ngl_0p25_5min_2023_2025.manifest.json
datasets/global_ngl_0p25_5min_2023_2025.station_cache/
```

中断后执行相同命令会复用已经完成的站点缓存。只扫描站点坐标和全球网格映射：

```bash
python preprocessing/build_global_ngl_zarr.py --scan-only --workers 32
```

## 训练 ERA5 + GNSS-ZTD 全球预报模型

进入项目目录并激活环境：

```bash
cd /inspire/ssd/project/sais-weather/public/linan/da-forecast
conda activate gnss
```

### 1. 检查数据和时间对齐

```bash
python train.py \
  --config configs/train_da_forecast.json \
  --check-data
```

数据对应关系为：

```text
ERA5(t-6h, t) + ZTD(t-6h ... t，每20分钟一帧，共19帧) -> ERA5(t+6h)
```

训练配置读取全球ERA5的 `z[time,70,720,1440]`（70通道均来自ERA5），
以及全球NGL的 `ztd[time,720,1440]`。两套纬度坐标略有差异；加载时
依据站点实际经纬度将观测映射到最近的ERA5格点，同格多站取有效值平均。
训练/验证/测试时段分别为2023-01-01～2024-05-01、2024-05-01～
2025-01-01、2025-01-01～2026-01-01，右端均不包含。

### 2. 两卡FSDP小样本过拟合测试

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  train.py \
  --config configs/train_da_forecast.json \
  --overfit-samples 4 \
  --epochs 20 \
  --output-dir work_dirs/fsdp_overfit4
```

确认训练loss能够明显下降后，再启动正式训练。

### 3. 四卡正式训练

当前配置使用FSDP `FULL_SHARD`，每个FuXi SwinBlock独立包装，BF16前向和
梯度通信。当前 `loss_mode="global_only"`，优化全球纬度加权MAE。

若从FuXi短时预训练模型初始化，首次训练使用独立输出目录：

```bash
mkdir -p work_dirs/da_forecast_global_6h_fuxi_init
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc_per_node=4 train.py \
  --config configs/train_da_forecast.json \
  --pretrained-fuxi model/fuxi_short.pth \
  --output-dir work_dirs/da_forecast_global_6h_fuxi_init \
  > work_dirs/da_forecast_global_6h_fuxi_init/train.log 2>&1 &
```

加载器继承FuXi的完整ERA5路径，包括Patch Embedding、DownBlock、时间嵌入、
Transformer/FPN、UpBlock和输出头；只重新初始化OBS编码器与门控融合。
`train.freeze_fuxi=true` 时只训练OBS编码器和门控融合。日志会报告实际加载的
张量和参数数量。
`--pretrained-fuxi` 只能用于新训练，不能与 `--resume` 同时使用；后续续训
使用新目录中的 `last.pt`。FuXi `.pth` 使用pickle格式，仅加载可信来源的文件。

```bash
mkdir -p work_dirs/da_forecast_global_6h_fsdp

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  train.py \
  --config configs/train_da_forecast.json \
  --pretrained-fuxi model/fuxi_short.pth  \
  > work_dirs/da_forecast_global_6h_fsdp/train.log 2>&1 &
```

默认配置为每卡batch size 1、梯度累积4次、有效全局batch size 16、BF16、
学习率 `2e-4`、warmup 100步和40 epochs。

训练结果保存在：

```text
work_dirs/da_forecast_global_6h_fsdp/best.pt
work_dirs/da_forecast_global_6h_fsdp/last.pt
work_dirs/da_forecast_global_6h_fsdp/train.log
```

### 4. 断点续训

```bash
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  train.py \
  --config configs/train_da_forecast.json \
  --resume work_dirs/da_forecast_global_6h_fsdp/last.pt \
  >> work_dirs/da_forecast_global_6h_fsdp/train.log 2>&1 &
```

查看训练日志：

```bash
tail -f work_dirs/da_forecast_global_6h_fsdp/train.log
```

## 测试集评估与天气要素动图

测试集范围由配置文件 `data.test_split` 决定（右端不包含）。评估脚本会读取checkpoint中的 `loss_mode`，输出对应主指标，并同时报告全球、站点halo区域的MAE及持久性基线。以下命令使用四卡 FSDP 分片评估；每卡的 `--batch-size` 是单卡 batch size。命令会完成统计并生成有效时刻
`2025-05-01` 至 `2025-05-05` 的重要天气要素动图：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  evaluate.py \
  --config configs/train_da_forecast.json \
  --checkpoint work_dirs/da_forecast_6h_40epoch_halo_only/best.pt \
  --output-dir work_dirs/da_forecast_6h_40epoch_halo_only/test \
  --batch-size 1 \
  --num-workers 4
```

每个进程只评估测试集的一部分，FSDP 在四卡间切分模型参数；最终指标和动图由 rank 0 写入。若只需指标、不需动图，可额外传入 `--skip-animations`。也可以直接单卡运行 `python evaluate.py`，但该模式不会分片模型。

### FSDP 权重恢复修复后的复核

评估现在在 rank 0 严格加载完整权重，再通过 FSDP 的
`sync_module_states=True` 同步并分片。旧的“包装后仅 rank 0 加载、其他 rank
传空字典”不会广播模型权重，旧分布式评估结果需要重算。训练 `--resume`
也已改为每个 rank 通过内存映射读取 checkpoint 并严格恢复模型，优化器仍采用分发恢复。
日志和结果 JSON 的 `evaluation` 字段会记录 checkpoint、epoch（从 0 开始）、
step、保存的最佳验证 loss、评估 split 和加载方式。

先复核当前全球模型的验证集（与训练日志中的 best val 对照）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc_per_node=4 evaluate.py \
  --config configs/train_da_forecast.json \
  --checkpoint work_dirs/da_forecast_global_6h_fsdp_train_obs/best.pt \
  --split val --skip-animations \
  --output-dir work_dirs/da_forecast_global_6h_fsdp_train_obs/val_fixed \
  --batch-size 1 --num-workers 4
```

测试集改用 `--split test` 和输出目录 `test_fixed`；需要动图时去掉
`--skip-animations`。使用新目录可保留旧结果供排查。验证与评估的样本聚合方式、
分布式补齐处理略有不同，因此验证指标允许有微小差异，不要求逐位相等。

四卡小模型权重恢复回归测试（同时覆盖全量训练和冻结 FuXi 主体）：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  tests/fsdp_checkpoint_regression.py
```

输出文件：

```text
work_dirs/da_forecast_6h_40epoch_halo_only/test/weighted_mae_summary.json
work_dirs/da_forecast_6h_40epoch_halo_only/test/weighted_mae_by_channel.csv
work_dirs/da_forecast_6h_40epoch_halo_only/test/animations/z500_20250501_20250505.gif
work_dirs/da_forecast_6h_40epoch_halo_only/test/animations/t850_20250501_20250505.gif
work_dirs/da_forecast_6h_40epoch_halo_only/test/animations/r850_20250501_20250505.gif
work_dirs/da_forecast_6h_40epoch_halo_only/test/animations/t2m_20250501_20250505.gif
work_dirs/da_forecast_6h_40epoch_halo_only/test/animations/wind10_20250501_20250505.gif
work_dirs/da_forecast_6h_40epoch_halo_only/test/animations/msl_20250501_20250505.gif
work_dirs/da_forecast_6h_40epoch_halo_only/test/animations/tp_20250501_20250505.gif
```
