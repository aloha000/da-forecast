# da-forecast

当前模型的完整逐层结构见
[CURRENT_NETWORK_ARCHITECTURE.md](CURRENT_NETWORK_ARCHITECTURE.md)。

## 训练 ERA5 + GNSS-ZTD 区域预报模型

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
ERA5(t-6h, t) + ZTD(t-45min ... t) -> ERA5(t+6h)
```

### 2. 单卡16样本过拟合测试

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/train_da_forecast.json \
  --overfit-samples 16 \
  --epochs 50 \
  --output-dir work_dirs/overfit16
```

确认训练loss能够明显下降后，再启动正式训练。

### 3. 四卡正式训练

```bash
mkdir -p work_dirs/da_forecast_6h

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  train.py \
  --config configs/train_da_forecast.json \
  > work_dirs/da_forecast_6h/train.log 2>&1 &
```

默认配置为每卡batch size 8、全局batch size 32、BF16、学习率
`2e-4`、warmup 100步和20 epochs。

训练结果保存在：

```text
work_dirs/da_forecast_6h/best.pt
work_dirs/da_forecast_6h/last.pt
work_dirs/da_forecast_6h/train.log
```

### 4. 断点续训

```bash
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  train.py \
  --config configs/train_da_forecast.json \
  --resume work_dirs/da_forecast_6h/last.pt \
  >> work_dirs/da_forecast_6h/train.log 2>&1 &
```

查看训练日志：

```bash
tail -f work_dirs/da_forecast_6h/train.log
```

## 测试集评估与天气要素动图

测试集为验证集结束后的 `2025-01-01 00:00` 至 `2025-10-01 00:00`
（右端不包含）。以下命令统计整个测试集的纬度加权MAE，并生成有效时刻
`2025-05-01` 至 `2025-05-05` 的重要天气要素动图：

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --config configs/train_da_forecast.json \
  --checkpoint work_dirs/da_forecast_6h/best.pt \
  --output-dir work_dirs/da_forecast_6h/test
```

输出文件：

```text
work_dirs/da_forecast_6h/test/weighted_mae_summary.json
work_dirs/da_forecast_6h/test/weighted_mae_by_channel.csv
work_dirs/da_forecast_6h/test/animations/z500_20250501_20250505.gif
work_dirs/da_forecast_6h/test/animations/t850_20250501_20250505.gif
work_dirs/da_forecast_6h/test/animations/r850_20250501_20250505.gif
work_dirs/da_forecast_6h/test/animations/t2m_20250501_20250505.gif
work_dirs/da_forecast_6h/test/animations/wind10_20250501_20250505.gif
work_dirs/da_forecast_6h/test/animations/msl_20250501_20250505.gif
work_dirs/da_forecast_6h/test/animations/tp_20250501_20250505.gif
```
