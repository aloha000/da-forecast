# Baseline 第二步：零 ZTD 第一跳 → 无 OBS 的 12h 微调

本实验与 `da_forecast_12h_fuxi_only` 的第二跳结构、训练超参数一致。
使用独立配置 `configs/train_baseline_12h_zero_ztd.json`，不覆盖已有实验。

## 两步定义

1. 固定第一步模型：`work_dirs/da_forecast_global_6h_fsdp_train_obs-bs2_zero-ztd/best.pt`。
   输入 ERA5(T−6)、ERA5(T)。保持该 checkpoint 的 OBS 分支及 `zero_ztd=true`：
   模型内将 ZTD 值置零，但保留 T−6..T 的实际有效 mask、站点静态场和已学习的 gate。
   不是将 mask 也置零，不使用 T 之后的观测。生成固定的 P6。
2. 第二步输入 ERA5(T)、P6，时间编码 T+6，监督 ERA5(T+12)。
   `model.use_obs=false`，OBS encoder/high_fusion/low_fusion 都不构建。
   用同一个第一步 checkpoint 的 FuXi 权重初始化，丢弃上述三个 OBS 模块的权重。
   `train.freeze_fuxi=false`，训练 era_encoder、time_embed、backbone、decoder。
   新建优化器和学习率调度器，不恢复第一步的 epoch/step。
3. 第一跳不会随第二步训练更新，不进行跨两步端到端反传。
   第二步 train/evaluate 不读取源 ZTD 数据，也不需保存第一步 mask。
4. 训练保持 10 epochs、每卡 batch=1、梯度累积=8、四卡有效 batch=32、
   LR=1e-5、最低 LR=1e-6、warmup=100、BF16、FSDP、global_only loss，
   时间划分沿用原 12h 实验。

新 Zarr：`datasets/era5_rollout12h_baseline_zero_ztd.zarr`。
必须重新生成，不能复用真实 ZTD 模型生成的 P6。
格式不变：四维 `z` 真值和 `z_fuxi6h` 预报，支持断点续写。
全量真值+预报约 2.31 TiB 未压缩，检查个人配额及剩余空间。

## 命令

在 da-forecast 根目录，激活 gnss 环境后运行；不要同时启动生成和训练。

### 1. 检查规模（不写数据、不使用 GPU）

```bash
python build_rollout_zarr.py --config configs/train_baseline_12h_zero_ztd.json --plan
```

### 2. 可选：四个样本试运行

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc_per_node=4 build_rollout_zarr.py \
  --config configs/train_baseline_12h_zero_ztd.json \
  --splits train --max-samples 4 --batch-size 1 --num-workers 2
```

### 3. 后台生成全部数据（可接着试运行补齐）

```bash
mkdir -p work_dirs/da_forecast_12h_baseline_zero_ztd
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
  --standalone --nproc_per_node=4 build_rollout_zarr.py \
  --config configs/train_baseline_12h_zero_ztd.json \
  --splits train val test --batch-size 1 --num-workers 2 \
  > work_dirs/da_forecast_12h_baseline_zero_ztd/build_zarr.log 2>&1 < /dev/null &
```

### 4. 数据完成后检查并训练

```bash
python train.py --config configs/train_baseline_12h_zero_ztd.json --check-data

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
  --standalone --nproc_per_node=4 train.py \
  --config configs/train_baseline_12h_zero_ztd.json \
  --init-checkpoint work_dirs/da_forecast_global_6h_fsdp_train_obs-bs2_zero-ztd/best.pt \
  --init-fuxi-only \
  > work_dirs/da_forecast_12h_baseline_zero_ztd/train.log 2>&1 < /dev/null &
```

首次启动必须用 `--init-checkpoint ... --init-fuxi-only`；
不要用 `--pretrained-fuxi` 或 `--resume` 代替新阶段初始化。
以后断点续训时再用该 12h 实验自身的 checkpoint 配合 `--resume`，
不要同时给初始化参数。后台命令的日志重定向会覆盖同名日志，重跑时注意备份。

### 5. 测试

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc_per_node=4 evaluate.py \
  --config configs/train_baseline_12h_zero_ztd.json \
  --checkpoint work_dirs/da_forecast_12h_baseline_zero_ztd/best.pt \
  --output-dir work_dirs/da_forecast_12h_baseline_zero_ztd/test \
  --batch-size 1 --num-workers 2
```

测试会校验 checkpoint 与 Zarr 的第一步模型、数据和归一化指纹一致。
如需与原始 FuXi 12h 对比，用这份新配置运行 `evaluate_original_fuxi.py`：
它仍使用原始 FuXi 自身两步预报，不读取本实验 P6 作为第一步结果。
