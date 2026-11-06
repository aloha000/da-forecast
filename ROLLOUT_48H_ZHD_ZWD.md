# ZHD/ZWD 输入、起报后关闭观测、48h 课程训练

本实验在 main 分支实现，不预测 ZHD/ZWD，没有任何延迟监督头或延迟 loss。
保留一个共享的 6h 天气模型，逐阶段增加连续 rollout 长度。

## 时间与训练关系

| 阶段 | 连续预报步数 | 最远时效 | 优化器更新范围 |
|---|---:|---:|---:|
| 1 | 1 | 6h | 1–500 |
| 2 | 2 | 12h | 501–1000 |
| 3 | 3 | 18h | 1001–1500 |
| 4 | 4 | 24h | 1501–2000 |
| 5 | 5 | 30h | 2001–2500 |
| 6 | 6 | 36h | 2501–3000 |
| 7 | 7 | 42h | 3001–3500 |
| 8 | 8 | 48h | 3501–4000 |

~~~text
第 1 步：[ERA(T-6), ERA(T)] + 真实 ZHD/ZWD(T-6..T) -> P(T+6)
第 2 步：[ERA(T), P(T+6)]，关闭 OBS -> P(T+12)
第 3 步：[P(T+6), P(T+12)]，关闭 OBS -> P(T+18)
...
第 8 步：[P(T+36), P(T+42)]，关闭 OBS -> P(T+48)
~~~

- 第一跳仍取过去 6h、每 20min 一帧，共 19 帧观测。第二跳以后不读、不输入未来观测。
- 后续跳直接跳过 OBS encoder、高/低分辨率 gate，不是只把观测值置零后仍运行 gate。
- 所有阶段、所有跳共享同一组模型参数。不是八个独立模型，也不是离线固定上游预测。
- 每个阶段训练 500 次 optimizer.step；不是 500 个 epoch，也不是 500 个微批。
- 当前默认 4 卡 × 每卡 batch 1 × 梯度累积 8，等效 batch 32 个起报样本。
- AdamW 状态跨阶段继续，不从头初始化；每阶段 LR 重新 warmup 25 次更新，再余弦降至最小 LR。
- 阶段切换继续使用当前模型权重，不回退到前阶段 best。
- 天气、OBS encoder、gate 均解冻。阶段末保存的 checkpoint 只是同一模型在不同训练阶段的快照。

## Loss 和梯度

阶段 N 中，计算 T+6、T+12、…、T+6N 的纬度加权天气 MAE，取算术平均。
例如 24h 阶段：loss = (loss_6h + loss_12h + loss_18h + loss_24h) / 4。

预测不会 detach，未来 ERA5 仅用于监督，不替换自回归输入；48h 损失可以穿过全部预测步反传到第一步 OBS。
没有 ZHD loss、ZWD loss 或 ZTD loss，不需要 station_height_path，也不使用气象延迟算子。

这里是观测仅在起报时使用的一次 48h 预报，不是每 6h 吸收新观测的循环同化实验。

## 输入和旧权重兼容

数据直接读取现有 ztd、zwd，先分别反归一化成 mm，计算 ZHD = ZTD - ZWD。必须两项都有效才使用该源站点/时刻；先求分量后做站点格点合并。
OBS 数据变成 [B,19,2,720,1440]，顺序 [ZHD,ZWD]。为兼容调用接口，批次键仍名为 obs_ztd，但里面已经是两个分量。

当前采用中心化加共享尺度：

~~~text
H_input = (ZHD_mm - 2210.3834168753006) / 169.4367826272024
W_input = (ZWD_mm - 109.85565038275689) / 169.4367826272024
~~~

169.4368 mm 是原 ZTD 标准差，作为两个分量的共享输入尺度，不是各分量自身的标准差。
这样的 H_input + W_input 恰好等于同一有效数据对的原 ZTD 标准分数。

第一层 Conv3d 从旧 [ZTD,mask] 的 2 通道输入变成 [ZHD,ZWD,paired_mask] 的 3 通道输入。
初始化时将旧 ZTD 卷积核复制到 H/W 两个通道，原 mask 卷积核放入第三通道，其余层严格加载不变。参数仅增加 864 个。
这能在有效样本集合一致且计算精度一致时保持初始响应近似一致；ZWD 缺测导致有效集合缩小，或 BF16 舍入，会产生差异。

默认初始化文件：
work_dirs/da_forecast_stage2_finetune_fuxi/best.pt

请使用这个原天气模型（真实 OBS，zero_ztd=false），不要传入之前带延迟预测头的 checkpoint；不支持静默丢弃其他网络层。
新的初始权重来源和观测归一化记录会写入 resolved_config.json/checkpoint。

## 数据读取与显存

不生成新的 Zarr，不修改原 ERA/NGL。每个阶段仅读取该阶段需要的天气标签。

所有阶段使用相同起报样本集合：必须保证 T+48h 仍在各自 split 内。实际检查得到 train=1935、val=972，少于原 6h 实验是因为剔除了边界起报点，并非漏读数据。
测试也采用共同的 48h 起报集合，因此与旧单步测试直接横比前需对齐样本集合。

默认开启整步 non-reentrant checkpoint 重计算，同时保留原模型内部 checkpoint；不 detach 自回归路径。
FSDP 每个微批都同步梯度，再累积 8 次更新，避免 no_sync 保留完整非分片梯度的显存代价。

每个 48h 样本包含 8 帧全球标签，仅 FP32 标签约 2.16 GiB，还有历史输入、掩膜、静态场和重计算缓存。
默认每卡 batch 1、num_workers=1、prefetch_factor=1、pin_memory=false，避免大量预取占满主机内存。
完整八步的显存和速度仍需实际 GPU smoke 验证；不要直接把单步可以用的 batch size 照搬到 48h。

checkpoint 含完整模型、优化器、调度器、阶段进度和各 rank RNG 状态。约每 100 次更新覆盖 last.pt；每阶段验证后保存 stage_XXh/best.pt。
本模型很大，保留 8 个完整阶段快照加 last/temp 文件可能需要约 200 GiB 以上空间，请预留充足配额。
best 按同一阶段最远时效的验证 MAE 选择；默认每 500 次更新验证一次（阶段末），不是跨时效比较 best_val。

## 先检查

在 gnss 环境、项目根目录运行：

~~~bash
cd /inspire/ssd/project/sais-weather/public/linan/da-forecast
python train_rollout.py --config configs/train_rollout_48h_zhd_zwd.json --check-data
python -m unittest -v tests.test_rollout_curriculum
~~~

原来的 train.py/evaluate.py 会明确拒绝这个双通道课程实验，避免误用成旧单步数据读取流程。
新入口为 train_rollout.py 和 evaluate_rollout.py；底层模型、FSDP、旧权重初始化仍复用本项目代码。

## 建议先做四卡完整八步 smoke

每阶段仅更新 1 次、累积 1 个微批，使用独立输出目录；只用于显存和流程验证，不代表训练效果。
总共跑 8 个更新，能实际走到 48h 阶段：

~~~bash
mkdir -p work_dirs/da_forecast_rollout48h_zhd_zwd_smoke
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
  --standalone --nproc_per_node=4 train_rollout.py \
  --config configs/train_rollout_48h_zhd_zwd.json \
  --updates-per-stage 1 --gradient-accumulation 1 --overfit-samples 8 \
  --output-dir work_dirs/da_forecast_rollout48h_zhd_zwd_smoke \
  > work_dirs/da_forecast_rollout48h_zhd_zwd_smoke/train.log 2>&1 < /dev/null &
~~~

smoke 也会保存大型 checkpoint。如果只想先测单步，可改用 --stop-after-updates 2 配合独立目录，但这不能验证八步显存。
正式训练不用 smoke 的输出作初始化，而是从默认原 6h 模型重新开始。

## 正式后台训练：每阶段 500 次更新

~~~bash
mkdir -p work_dirs/da_forecast_rollout48h_zhd_zwd
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  LD_PRELOAD=/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/2660-linan/conda-envs/gnss/lib/libstdc++.so.6 \
  python -u -m torch.distributed.run \
  --standalone --nproc_per_node=4 train_rollout.py \
  --config configs/train_rollout_48h_zhd_zwd.json \
  --init-checkpoint work_dirs/da_forecast_stage2_finetune_fuxi/best.pt \
  > work_dirs/da_forecast_rollout48h_zhd_zwd/train.log 2>&1 < /dev/null &
~~~

~~~bash
tail -f work_dirs/da_forecast_rollout48h_zhd_zwd/train.log
~~~

启动后查看 stage、lead、stage_update、update。最终应为 stage=8、lead=48h、stage_update=500/500、update=4000/4000。
不设置 --epochs。num_workers 建议先保持 1，后期确认主机内存和 IO 后再调整。

## 中断后恢复

不要同时运行原进程和恢复进程。必须使用相同 GPU 数、数据、归一化、batch、累积次数和阶段设置：

~~~bash
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
  --standalone --nproc_per_node=4 train_rollout.py \
  --config configs/train_rollout_48h_zhd_zwd.json \
  --resume work_dirs/da_forecast_rollout48h_zhd_zwd/last.pt \
  >> work_dirs/da_forecast_rollout48h_zhd_zwd/train.log 2>&1 < /dev/null &
~~~

更新计数、优化器、scheduler、随机数状态和 sampler 游标恢复；不会重新从第一阶段计数。
--stop-after-updates 用于在指定总更新数干净停止并保存，不改变总训练计划。
若恢复 smoke，需再次传入原来的 --updates-per-stage、--gradient-accumulation、--overfit-samples 和输出目录。

## 一次测试全部八个时效

~~~bash
mkdir -p work_dirs/da_forecast_rollout48h_zhd_zwd/test
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
  --standalone --nproc_per_node=4 evaluate_rollout.py \
  --config configs/train_rollout_48h_zhd_zwd.json \
  --checkpoint work_dirs/da_forecast_rollout48h_zhd_zwd/stage_48h/best.pt \
  --split test --steps 8 --batch-size 1 --num-workers 1 \
  --output-dir work_dirs/da_forecast_rollout48h_zhd_zwd/test \
  > work_dirs/da_forecast_rollout48h_zhd_zwd/test/evaluate.log 2>&1 < /dev/null &
~~~

输出：
- rollout_mae_summary.json：6/12/18/24/30/36/42/48h 的全球和 halo 纬度加权 MAE。
- rollout_mae_by_lead.csv：按时效汇总，并提供 ERA(T) persistence 对照。
- rollout_mae_by_channel.csv：各时效 70 通道的标准化 MAE、反归一化 MAE 和 persistence。
- 物理单位沿用 evaluate.py：z 换为位势高度 m、msl 为 hPa、tp 逆 log1p；不把不同变量的物理单位直接求平均。
- 不输出 ZHD/ZWD MAE，因为该实验不预测观测；不保存巨大的全球预测场或动画。

使用阶段较早的 checkpoint 测试更长时效时会警告“超出该 checkpoint 已训练的 rollout 长度”。
评估不会吸收未来观测、不会 teacher forcing，分布式补齐样本不计入指标。

## 已完成的验证范围

- 原 main 分支 42 项回归测试通过。
- 新增 CPU 测试：双通道物理单位和掩膜、旧核转换、OBS 真正绕过、8 步梯度不断链、真实小网络重计算、500 次更新边界、八阶段微型训练/测试、按位一致的 CPU 中断恢复、评估 padding 排除。
- 两进程 CPU/Gloo FSDP 小模型实际跑过 8 步全反传、整步重计算、梯度累积和两次优化器更新。
- 真实数据的双通道读取及 48h 样本边界检查通过。
- 未启动全球模型的 GPU 训练，也未验证全球八步在当前 GPU 占用下的峰值显存；请先执行上面的独立 smoke。
