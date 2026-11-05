# 预测 ZTD 并反馈到下一轮（predict_ztd 分支）

> 本文描述旧版直接预测 ZTD 的实验。新的“只监督 ZHD/ZWD、ZTD 相加派生”方案及命令见 [PREDICT_DELAYS.md](PREDICT_DELAYS.md)。

## 已实现的范围

- 6h 气象 + 18 帧 ZTD 联合训练、独立新头预热、旧 checkpoint 初始化。
- 现有离线 12h rollout：固定第一跳模型，同时保存天气预测和 ZTD 预测输入。
- train.py / evaluate.py 支持两种输出；旧配置不打开 predict_ztd 时完全走旧路径。
- 通用 forecast_rollout 迭代器支持同一模型连续多步反馈（6/12/18/24h 等）。
  当前离线 Zarr 生成器和按阶段训练配置仍针对两跳 12h；它们尚不能仅改 lead_hours 就生成 18h/24h 数据。
- 不会自动启动 GPU 训练或生成生产数据。

## 时间关系与 mask

以第一跳起报时刻 T 为例：

```text
ERA5(T-6), ERA5(T) + 19帧真实 ZTD(T-6..T)
                        ↓
        气象预测 P(T+6)
        ZTD预测 Z(T+20min..T+6h)，共18帧
                        ↓
[ERA5(T), P(T+6)] + [真实 ZTD(T), 18帧预测 ZTD]
                        ↓
        气象预测 P(T+12)
        ZTD预测 Z(T+6h20min..T+12h)，共18帧
```

第二跳的时间编码是 T+6，标签为 ERA5(T+12)，ZTD 标签是 T+6h20min..T+12h。
接着推第三跳时，天气输入必须是 [P(T+6), P(T+12)]，不能重新读取未来 ERA5 真值。

每次的下一轮 OBS 窗口都由“当前末帧 + 18帧预测”组成，窗口保持 [B,19,1,H,W]。
模型产生的 ZTD 是伪观测/预测状态，不是新增实测信息：

- 边界末帧保持原来的值和有效 mask；若缺测则值0、mask0，不补真实未来值。
- 18帧预测的输入资格由当前19帧中“是否至少有一次有效观测/预测”决定。
- 完全无有效历史的格点不参与预测反馈。
- 未来标签 mask 只参与损失，不决定输入 mask，也不改变气象训练样本集合。
- 预测出现 NaN/Inf 时生成/评估会报错，不能把失败预测悄悄当作缺测。
- 站点位置延用固定静态场，禁止用未来观测可用性更新站点集合。

## 网络、权重和冻结

ForecastDecoder.up_block 的 1/4 分辨率特征同时送往两个独立线性 patch 重建头：

- 原 decoder.head：70通道天气增量；天气结果仍是“最新输入天气场 + 增量”。
- 新 decoder.ztd_head：18帧直接 ZTD 值，输出 [B,18,1,H,W]，空间重建后只在站点位置监督/反馈。
  标准 embed_dim=1536、patch=4 时仅新增 442656 个参数。
  不将原70通道头改成88通道，保留原气象参数名称与形状。

predict_ztd=true 时 forward 返回：
```python
{"weather": weather_prediction, "ztd": future_ztd}
```
默认 predict_ztd=false 时仍返回原来的天气 Tensor。

必须同时配置：
```text
model.predict_ztd=true
data.predict_ztd=true
model.use_obs=true
model.zero_ztd=false
data.obs_history_hours=6
data.obs_interval_minutes=20
```

新头预热可以加载旧的有 OBS、非零 ZTD 6h checkpoint；
加载器只允许缺少 decoder.ztd_head.weight / bias 两个新参数，其他参数和形状严格检查。
加载完整联合 checkpoint 时所有参数严格匹配。
不能用 zero_ztd checkpoint 冒充非零观测模型；不能使用 --init-fuxi-only 丢掉 OBS。
同样不要用 --resume 开始一个冻结策略不同的新阶段。

freeze_fuxi 控制 ERA encoder/time_embed/Transformer/FPN/原 decoder；
freeze_obs 控制 OBS encoder 和两级 fusion/gate；
freeze_ztd_head 单独控制新头，默认 false，即使 FuXi 冻结也可以训练新头。
冻结其他模块不会截断新头对共享特征的梯度。

## 损失与评价

训练目标：
```text
total_loss = weather_loss + ztd_loss_weight * ztd_loss
```

weather_loss 保留原纬度加权 global_only MAE。
ZTD loss 为有效站点网格位置与时间上的平均绝对误差，不做纬度权重；
多个站点落在同一 ERA5 格点时沿用已有读取器的平均规则。
无站点区域的填充零不计入损失；未来所有 ZTD 标签缺测时该项为0，天气样本仍保留。
训练按各卡各 microbatch 的 masked mean 优化；验证 ZTD 项汇总误差和有效计数。
既有训练验证的 DistributedSampler 可能为齐整长度重复少量样本；
evaluate 使用 _eval_valid 排除填充样本，并按全测试集有效计数统计，跨batch结果一致。

输入、标签、预测反馈始终使用源 NGL Zarr 的标准化尺度，不套用 ERA5 标准化。
当前 NGL store 元数据提供 ztd_train_mean/std 和原始单位 mm，物理 MAE = 标准化 MAE × std，
均值在差值中抵消。缺少可信标准差/单位时不伪造毫米结果。

日志包含 batch_loss、weather_loss、ztd_loss、val_ztd。
evaluate 保留原气象 MAE 文件，另外输出：

- ztd_mae_summary.json：整体标准化/毫米 MAE、有效标签数、归一化与递推来源。
- ztd_mae_by_lead.csv：18个时间点的标准化/毫米 MAE、有效标签数。
- 对6h模型的时间偏移为20..360min；对12h第二跳为380..720min。
  12h文件统计的是第二段ZTD预测，不自动合并第一段指标。

新头预热按 checkpoint_metric=ztd 选择 best；
6h/12h联合微调按 checkpoint_metric=weather_global 选择 best，避免仅因ZTD辅助项改善就称天气变好。
best_val 的含义由 checkpoint 内的 checkpoint_metric 决定，val 日志仍是联合目标。
loss_weight=0.1 是起点，需要结合验证天气与 ZTD 指标再调整。

## 离线 Zarr

新增独立路径：
`datasets/era5_rollout12h_predict_ztd.zarr`。

气象 z / z_fuxi6h 仍为原四维格式；附加字段：

| 字段 | 维度/含义 |
|---|---|
| station_flat | 排序后的有效站点网格扁平索引 |
| obs_static | 完整3通道站点静态信息 |
| ztd_input | [init_time,19,station_cell]，真实T端点 + 18帧预测 |
| ztd_input_mask | 对应的历史可用性/伪观测输入mask |
| ztd_target | [init_time,18,station_cell]，T+6h20min..T+12h真值 |
| ztd_target_mask | 仅供监督的有效性mask |

稀疏存储避免为无站点区域写大数组，读取时还原成网格。
预测天气、ZTD输入和标签全部写完后才将 forecast_complete 标记为 true，支持中断续写。
第二跳训练/评估不读取原始 NGL，只读取这个独立 Zarr。
第一跳使用真实历史 ZTD；未来真值在推理完成后单独读取并存为标签。
identity 记录预测策略、源checkpoint、归一化等，不能混用旧无OBS Zarr。
全量天气字段本身仍约2.31 TiB未压缩，另外加少量稀疏站点字段；注意磁盘及个人配额。

## 运行步骤

在项目根目录，先激活 gnss 环境。各阶段依次完成，不要同时占用同一组GPU。
以下都使用新输出目录。再次执行 > 重定向会覆盖同名日志，请先备份。
第一步建议从已训练的有真实观测模型初始化：
`work_dirs/da_forecast_stage2_finetune_fuxi/best.pt`。
它不是 zero_ztd_stage2 模型。

### 0. CPU测试和真实数据检查

```bash
python -m unittest discover -s tests -p 'test_*.py' -q
python train.py --config configs/train_6h_ztd_head.json --check-data
```

### 1. 新 ZTD 头预热

配置 train_6h_ztd_head.json：3 epochs，只有 ZTD 头可训练，LR=1e-4。
四卡每卡batch=1、累积8。先观察初始验证和首个epoch的 ZTD MAE，再决定是否延长预热。

```bash
mkdir -p work_dirs/da_forecast_6h_ztd_head
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
  --standalone --nproc_per_node=4 train.py \
  --config configs/train_6h_ztd_head.json \
  --init-checkpoint work_dirs/da_forecast_stage2_finetune_fuxi/best.pt \
  > work_dirs/da_forecast_6h_ztd_head/train.log 2>&1 < /dev/null &
```

### 2. 6h 联合微调

配置 train_6h_predict_ztd.json：解冻 FuXi、OBS/gate、新头，10 epochs，LR=1e-5。
完整初始化而非 resume，因为冻结策略和优化器参数集合变了。

```bash
mkdir -p work_dirs/da_forecast_6h_predict_ztd
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
  --standalone --nproc_per_node=4 train.py \
  --config configs/train_6h_predict_ztd.json \
  --init-checkpoint work_dirs/da_forecast_6h_ztd_head/best.pt \
  > work_dirs/da_forecast_6h_predict_ztd/train.log 2>&1 < /dev/null &
```

6h训练结束后先测试，确认ZTD头不是无效/发散输出：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc_per_node=4 evaluate.py \
  --config configs/train_6h_predict_ztd.json \
  --checkpoint work_dirs/da_forecast_6h_predict_ztd/best.pt \
  --output-dir work_dirs/da_forecast_6h_predict_ztd/test \
  --batch-size 1 --num-workers 2 --skip-animations
```

### 3. 生成12h联合输入

固定已经训练结束的6h checkpoint，生成期间不能覆盖该文件。
随机新增头未经训练的checkpoint不能直接拿来生成可信rollout。
先用四个样本测试真实四卡/FSDP路径：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc_per_node=4 build_rollout_zarr.py \
  --config configs/train_12h_predict_ztd.json \
  --splits train --max-samples 4 --batch-size 1 --num-workers 2
```

试运行无误后补齐所有数据：

```bash
mkdir -p work_dirs/da_forecast_12h_predict_ztd
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
  --standalone --nproc_per_node=4 build_rollout_zarr.py \
  --config configs/train_12h_predict_ztd.json \
  --splits train val test --batch-size 1 --num-workers 2 \
  > work_dirs/da_forecast_12h_predict_ztd/build_zarr.log 2>&1 < /dev/null &
```

### 4. 12h训练

使用生成输入的同一个6h联合模型完整初始化，不能添加 --init-fuxi-only。
这一版保持OBS/gate可训练，以适应预测ZTD与实测ZTD的分布差异。
不是第二跳继续输入真实观测；实际输入来自新Zarr中的预测序列。

```bash
python train.py --config configs/train_12h_predict_ztd.json --check-data

nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
  --standalone --nproc_per_node=4 train.py \
  --config configs/train_12h_predict_ztd.json \
  --init-checkpoint work_dirs/da_forecast_6h_predict_ztd/best.pt \
  > work_dirs/da_forecast_12h_predict_ztd/train.log 2>&1 < /dev/null &
```

### 5. 12h测试

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc_per_node=4 evaluate.py \
  --config configs/train_12h_predict_ztd.json \
  --checkpoint work_dirs/da_forecast_12h_predict_ztd/best.pt \
  --output-dir work_dirs/da_forecast_12h_predict_ztd/test \
  --batch-size 1 --num-workers 2 --skip-animations
```

## 通用多步推理接口

`model/ztd_rollout.py:forecast_rollout` 接收：
- 一个 predict_ztd 模型；
- 第一跳的模型输入batch；
- 仅由日历计算出的 T、T+6、T+12... 的 time_features 序列。

它每轮只用上轮的天气和ZTD预测更新输入，不携带 batch 中的 target / ztd_target。
调用者使用 torch.inference_mode 和 autocast 控制推理；如用于端到端训练，默认不会 detach，
因此多步反传会显著增加显存。现在 train.py 使用离线固定第一跳，不跨第一跳反传。

## 解释限制

预测ZTD不是新增信息，效果必须由消融验证：
无ZTD反馈、仅ZTD辅助监督、ZTD辅助监督+预测反馈。
新实现不保证一定提高天气MAE，也不等同于每6h同化一次真实GNSS观测。
