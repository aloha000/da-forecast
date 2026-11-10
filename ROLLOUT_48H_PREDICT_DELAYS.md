# 48h 气象场 + ZHD/ZWD 联合 rollout（predict_ztd 分支）

## 1. 流程与 main 分支的区别

模型结构保持当前分支原样：原 ZTD encoder、ZHD/ZWD component adapter、融合 gate、FuXi 主体、weather head、ZHD head、ZWD head。没有新增独立的 12h/18h 网络，参数量不会随 rollout 长度增长。保留现有归一化/物理量约定，不能把 main 分支的双通道卷积权重直接当成本分支权重加载。

- 第一轮输入：ERA(T−6h)、ERA(T)，以及 T−6h…T 每 20min 的真实观测，共 19 帧。
- 第一轮输出：P6 气象场；T+20min…T+6h 的 18 帧 ZHD/ZWD。
- 第二轮气象输入：ERA(T)、P6。观测历史变成「T 时刻最后一帧 + 第一轮预测的 18 帧」。
- 第二轮输出：P12；T+6h20min…T+12h 的 18 帧 ZHD/ZWD。
- 第三轮输入：P6、P12；上轮的最后历史帧与新预测的 18 帧。以此类推，最后输出 P48。
- ZTD 始终由预测的 ZHD+ZWD（先在 mm 中相加）再按原 ZTD 统计量标准化得到，供原 ZTD encoder 使用。
- **后续步仍执行 OBS encoder、adapter、gate**。这一点和 main 分支“后续完全跳过 OBS”不同。
- 未来真实 GNSS 只用作监督标签，不作为下一步输入，也不用于生成下一步输入 mask。预测站点的可用性从起报历史继承。
- 没有 teacher forcing，没有跨步 detach；梯度可经过气象反馈和延迟反馈两条路径。
- 真实分量、预测分量及标准化常数保留 FP32；神经网络卷积/Transformer 使用 BF16 autocast/FSDP。不要打开 FSDP 根输入的 BF16 自动强制转换。

## 2. 训练策略与损失

继续使用共享参数的 curriculum：6h、12h、18h、24h、30h、36h、42h、48h，共 8 阶段，每阶段 500 次 optimizer.step，总计 4000 次。进入新阶段时沿用上一阶段最后一套参数及 AdamW 状态，不重新加载上一阶段 best。

默认全部参数可训练，包括 FuXi、OBS、adapter 和两预测头。每卡 batch=1、4 卡、梯度累积 8，有效 batch=32。每个 microbatch 同步 FSDP 梯度，以免 no_sync 保留完整梯度占满显存。启用逐步重计算，支持完整跨步反向传播。

每个六小时段的损失（普通文字公式）：

    该步损失 = 气象场全球纬度加权 MAE
             + 0.1 × (ZHD 的 MAE_mm / 100)
             + 0.1 × (ZWD 的 MAE_mm / 100)

    当前阶段总损失 = 当前所有步骤损失的算术平均

没有独立 ZTD loss。ZHD/ZWD 监督该段全部 18 帧、有效站点网格，不做纬度加权；需要真实标签有效且该站点从历史可预测。100 mm 是损失尺度，不是输入标准差。训练先算各 microbatch 的有效点 MAE 再平均；验证/测试按全体有效点累计绝对误差和计数再相除。

默认 lr=1e-5，每阶段 warmup 25 步后 cosine 降至 1e-6。与 main 的 500 步阶段一致。物理算子保持现有 6h 配置的 use_delay_physics=false，不擅自启用。

每个阶段以「该阶段最后时效的 weather 全球纬度加权 MAE」选 best，不用延迟损失选择。日志同时列出所有时效的 weather、ZHD/ZWD MAE 和有效计数；没有有效延迟标签时验证 MAE 为 null，不伪装成 0。

## 3. 数据与初始化

不生成缓存，不生成新 Zarr，不修改源数据。使用原始 ERA5 和 NGL。按完整 T+48h 过滤起报时间，8 阶段始终使用同一批起报时刻，所有标签都落在 split 的半开区间内。实际检查 train=1935、val=972，与 main 的全时效样本筛选一致。

每阶段只读取其需要的未来天气和分量标签。48h 阶段单样本含 8 张 70 通道天气场、144 帧双分量标签，CPU 内存和显存显著高于单步；num_workers=1、prefetch_factor=1、pin_memory=false。共享内存紧张时可将 num_workers 改为 0。

默认初始化：

    work_dirs/da_forecast_6h_predict_delays/best.pt

已核对该文件配置含 ZHD/ZWD 双头且物理算子关闭。只加载权重，curriculum 的优化器和更新计数从头开始。可通过 --init-checkpoint 指定兼容权重。本次并非重新开始纯 FuXi 预训练；与 main 实验做严格因果对照时还需控制初始化历史和所有其他训练条件。

默认仍为真实观测起报、预测观测反馈。全零输入消融现已实现，见第 9 节；不要套用 main 的 data.zero_delay_inputs，也不要单独开启旧 model.zero_ztd。

## 4. 启动前检查

在 gnss 环境和项目目录下运行；以下 LD_PRELOAD 用于解决本机 Matplotlib 的 CXXABI 运行库问题。

```bash
cd /inspire/ssd/project/sais-weather/public/linan/da-forecast
export LD_PRELOAD=/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/2660-linan/conda-envs/gnss/lib/libstdc++.so.6
python train_rollout_delays.py --config configs/train_rollout_48h_predict_delays.json --check-data
```

check-data 只实际读取一个 6h 样本，同时验证起报时间具备完整 48h 的天气标签，不代表全球八步 GPU 前后向已经验证。

建议先完成全部 8 阶段的短测试（单独目录，每阶段 1 次更新）：

```bash
mkdir -p work_dirs/da_forecast_rollout48h_predict_delays_smoke
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
  --standalone --nproc_per_node=4 train_rollout_delays.py \
  --config configs/train_rollout_48h_predict_delays.json \
  --updates-per-stage 1 --gradient-accumulation 1 --overfit-samples 8 \
  --output-dir work_dirs/da_forecast_rollout48h_predict_delays_smoke \
  >> work_dirs/da_forecast_rollout48h_predict_delays_smoke/train.log 2>&1 < /dev/null &
```

这里仍然会保存完整模型和优化器，不能把它当成小存储量测试。每个全参数 checkpoint 约 21 GiB，8 个 stage best + last 和临时文件需预留约 200 GiB 以上空间。只要检查 6h 起步，可用生产设置加 --stop-after-updates 1 和独立目录，但这不能验证 48h 的显存。

## 5. 正式训练

先执行上面的 cd 和 export，然后：

```bash
mkdir -p work_dirs/da_forecast_rollout48h_predict_delays
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
  --standalone --nproc_per_node=4 train_rollout_delays.py \
  --config configs/train_rollout_48h_predict_delays.json \
  >> work_dirs/da_forecast_rollout48h_predict_delays/train.log 2>&1 < /dev/null &
```

不要用 train.py 执行这个配置；旧入口会明确报错，防止误跑成单步。

输出含 resolved_config.json、validation.jsonl、last.pt，以及 stage_06h/best.pt…stage_48h/best.pt。已有 resolved_config.json 的目录不能直接从头运行：用 --resume，或使用新的 --output-dir。初始化、数据读取较慢时，先看 OBS policy 和 initialization 日志。

## 6. 中断恢复

保持配置及 GPU 数不变，不同时传 --init-checkpoint：

```bash
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
  --standalone --nproc_per_node=4 train_rollout_delays.py \
  --config configs/train_rollout_48h_predict_delays.json \
  --resume work_dirs/da_forecast_rollout48h_predict_delays/last.pt \
  >> work_dirs/da_forecast_rollout48h_predict_delays/train.log 2>&1 < /dev/null &
```

恢复包括阶段、优化器、调度器、随机状态和样本游标。损失权重、数据标识、物理/归一化设置等不一致会被拒绝。短测试恢复时需要重复同样的 --updates-per-stage、--gradient-accumulation、--overfit-samples 参数。

## 7. 测试

先执行第 4 节的 cd 和 export：

```bash
mkdir -p work_dirs/da_forecast_rollout48h_predict_delays/test
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
  --standalone --nproc_per_node=4 evaluate_rollout_delays.py \
  --config configs/train_rollout_48h_predict_delays.json \
  --checkpoint work_dirs/da_forecast_rollout48h_predict_delays/stage_48h/best.pt \
  --split test --steps 8 --batch-size 1 --num-workers 1 \
  --output-dir work_dirs/da_forecast_rollout48h_predict_delays/test \
  >> work_dirs/da_forecast_rollout48h_predict_delays/test/evaluate.log 2>&1 < /dev/null &
```

结果：
- rollout_mae_summary.json：8 个时效 weather 全球/halo MAE，各六小时段 ZHD/ZWD MAE_mm 和有效计数。
- rollout_mae_by_lead.csv：同上按时效汇总。
- rollout_mae_by_channel.csv：每个时效的 70 通道归一化 MAE、物理单位 MAE。
- rollout_delay_mae_by_frame.csv：T+20min 至 T+48h，144 个时刻 × 2 个分量的 MAE_mm。

包含 persistence：天气一直保持 ERA(T)，分量一直保持起报历史最后一个有效值。延迟段 MAE 是该段 18 帧的有效点汇总，不等于只取最后一帧。分布式补齐样本排除在统计之外。没有独立 ZTD loss/MAE，也没有自动生成动画或缓存预报场。

## 8. 验证情况

- 75 项 CPU 单元测试通过（含新增 6 项多步测试）。
- 小型真实网络 8 步、两进程 CPU FSDP、每次更新累积 2 个 microbatch，运行 2 次更新成功；跨步重计算与反向传播正常。
- 真实数据起报时段筛选和一个 6h 样本形状/有限值检查通过。
- 未启动正式训练，未在全球尺寸上验证 8 步 GPU 显存或收敛。请先运行短测试。

## 9. ZTD、ZHD、ZWD 全零输入消融

配置：`configs/train_rollout_48h_predict_delays_zero_inputs.json`。

- `model.zero_delay_inputs=true`，`curriculum.observation_policy="zero_all_steps"`。
- 保持 `use_obs=true`、`zero_ztd=false`、两个预测头开启，保留全部网络模块和参数。
- 每次 model.forward（第一步及所有后续步）都把 ZTD 编码器输入置为标准化空间的 0，把 ZHD/ZWD 历史置为物理空间的 0 mm；不是先在原始 ZTD 上填 0 再标准化。
- Adapter 和分量预测残差基线共同使用清零后的历史，防止真实分量从残差基线进入预测。
- mask、站点静态场、ERA5、日期信息和真实监督标签不变。并非去掉站点位置/可用性信息的消融。
- 分量仍然预测并接受真实标签监督，ZTD 仍由二者相加得到；输出不强制置零。但下一步送入网络时再次清零，所以延迟数值反馈的跨步梯度被切断，天气反馈仍完整反传。
- 物理算子仍关闭。分量预测变成 `0 + 100 mm × 预测残差`，因此初期 ZHD MAE/总 loss 可能显著增大。该实验同时移除了观测残差基线；它不是仅测试 Encoder 融合贡献的对照。
- 与原实验使用同一 6h 初始权重、数据划分、随机种子、损失、学习率、4 卡 FSDP、每卡 batch=1、累积 8。共 8 阶段，每阶段 500 次更新。
- 不从已完成的 48h 实验继续训练；输出到新目录，原实验不覆盖。初始 6h 权重曾使用真实观测，此对照检验共同初始化下的零输入训练，不代表全训练历史从未接触观测。
- 开关写入 checkpoint 配置；`evaluate_rollout_delays.py` 从 checkpoint 构建模型，所以测试自动执行同样的全零策略。评估中的真实观测 persistence 仅为独立参考，不进入模型。

在 gnss 环境中正式启动：

```bash
cd /inspire/ssd/project/sais-weather/public/linan/da-forecast
export LD_PRELOAD=/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/2660-linan/conda-envs/gnss/lib/libstdc++.so.6
mkdir -p work_dirs/da_forecast_rollout48h_predict_delays_zero_inputs
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
  --standalone --nproc_per_node=4 train_rollout_delays.py \
  --config configs/train_rollout_48h_predict_delays_zero_inputs.json \
  >> work_dirs/da_forecast_rollout48h_predict_delays_zero_inputs/train.log 2>&1 < /dev/null &
```

建议先用相同命令添加以下参数测试 8 个阶段，输出必须用独立目录（并相应修改日志重定向目录）：

```text
--updates-per-stage 1 --gradient-accumulation 1 --overfit-samples 8
--output-dir work_dirs/da_forecast_rollout48h_predict_delays_zero_inputs_smoke
```

测试集评估（先执行上述 cd 和 export）：

```bash
mkdir -p work_dirs/da_forecast_rollout48h_predict_delays_zero_inputs/test
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
  --standalone --nproc_per_node=4 evaluate_rollout_delays.py \
  --config configs/train_rollout_48h_predict_delays_zero_inputs.json \
  --checkpoint work_dirs/da_forecast_rollout48h_predict_delays_zero_inputs/stage_48h/best.pt \
  --split test --steps 8 --batch-size 1 --num-workers 1 \
  --output-dir work_dirs/da_forecast_rollout48h_predict_delays_zero_inputs/test \
  >> work_dirs/da_forecast_rollout48h_predict_delays_zero_inputs/test/evaluate.log 2>&1 < /dev/null &
```
