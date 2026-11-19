# 原始 FuXi 48h 与 main 分支零 ZHD/ZWD 实验逐通道对比

## 对比口径

- 原始权重：`model/fuxi_short.pth`。调用 `model/fuxi.py` 中的原始 `UTransformer`，严格加载 checkpoint 的 `decoder.*` 权重；不是调用带 OBS/adapter/gate 的 DA 网络。
- 连续 8 个 6h 步骤均使用原始固定权重、不微调。输入从 ERA(T−6)、ERA(T) 开始，每步将网络输出增量加到当前场，并将两帧窗口更新为上一帧及预测场。未来 ERA5 只用于评估，绝不替代预测反馈。
- 原始 FuXi 不读取 ZTD/ZHD/ZWD 数值。读取 NGL 时间轴、站点网格等元数据仅用于精确复现对照的样本筛选和 halo 区域，不传入网络。
- 对照结果直接读取 `work_dirs/da_forecast_rollout48h_zero_zhd_zwd/test/rollout_mae_by_channel.csv`，不重新推理该模型。
- 在当前 predict_ztd 分支也可运行，不必 checkout main。使用固定 main 源版本 `dcbb37ce5429acb7adf3542adae68c87788036b7` 的样本筛选/静态场代码，并校验当前原始 FuXi 结构与该版本相同。
- 共同起报样本数 1452；2025-01-01 00:00 至 2025-12-29 18:00；所有时效使用相同起报集合，T+48h 标签仍在测试区间内。
- BF16、全球 cos(latitude) 加权 MAE、70 通道，输出 6、12、18、24、30、36、42、48h。各通道同时输出标准化及物理 MAE，halo 为原配置半径。
- TP 始终为该时效末尾 6h 的降水量，不是累计到该时效的总降水。
- 正式对比前逐通道校验两次评估的 persistence MAE（标准化与物理两种口径）。不一致就报错，不产生误配的对比图。提升率为 `100 × (原始 FuXi MAE − 零输入微调 MAE) / 原始 FuXi MAE`，正数表示零输入微调模型更好。

## 运行

请使用空闲 GPU 节点，或等待训练结束。不要与当前四卡均值起报训练并发，避免争抢显存。该脚本不自动停止或等待其他任务。

在 gnss 环境和项目目录下，可先做仅元数据检查（不使用 GPU、不生成结果）：

```bash
cd /inspire/ssd/project/sais-weather/public/linan/da-forecast
env LD_PRELOAD=/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/2660-linan/conda-envs/gnss/lib/libstdc++.so.6 \
  python evaluate_original_fuxi_48h.py --plan
```

正式四卡后台推理并自动绘图：

```bash
mkdir -p work_dirs/original_fuxi_48h_vs_zero_zhd_zwd
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  LD_PRELOAD=/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/2660-linan/conda-envs/gnss/lib/libstdc++.so.6 \
  MPLCONFIGDIR=/tmp/da_forecast_mpl_cache \
  python -u -m torch.distributed.run \
  --standalone --nproc_per_node=4 evaluate_original_fuxi_48h.py \
  --checkpoint model/fuxi_short.pth \
  --reference-dir work_dirs/da_forecast_rollout48h_zero_zhd_zwd \
  --output-dir work_dirs/original_fuxi_48h_vs_zero_zhd_zwd \
  --batch-size 1 --num-workers 1 \
  >> work_dirs/original_fuxi_48h_vs_zero_zhd_zwd/evaluate.log 2>&1 < /dev/null &
```

建议先 `--max-samples 4 --output-dir work_dirs/original_fuxi_48h_smoke` 做小样本检查，日志也用独立路径；小样本不会和完整测试集对照生成比较结论。模型仍是完整尺寸，所需显存不会因为样本少而消失。

每 10 个 batch/rank 保存聚合误差和计数到 `progress.npz`，不会保存预测场缓存。中断后在相同命令后添加 `--resume`；恢复要求相同源代码、数据/权重身份、GPU 进程数、batch size 和样本设置。不要在计算过程中切换分支或修改依赖文件。

## 输出

默认目录：`work_dirs/original_fuxi_48h_vs_zero_zhd_zwd`。

- `original_summary.json`、`original_mae_by_lead.csv`、`original_mae_by_channel.csv`：原始 FuXi 的评估结果及数据来源。
- `channel_comparison_all_leads.csv`：560 行，逐通道/时效的双方 MAE、绝对减少量、相对提升率（全球标准化、全球物理、halo 标准化）。
- `summary_by_lead.csv`：各时效的整体 MAE 对比及改善通道数。
- `comparison_report.md`：中文说明、8 时效汇总和 48h 全部 70 通道物理 MAE 表。
- `channel_mae_06h.png` … `channel_mae_48h.png`：每个时效的逐通道标准化 MAE 双柱状图。
- `channel_mae_physical_48h.png`：48h 反归一化 MAE，按变量分图以避免混用单位。
- `channel_improvement_all_leads.png`：8 时效 × 70 通道的相对提升热图，不是空间经纬度热图。
- `sums.npz`：完整评估的误差和计数，可重画图而不重新推理。

仅重画已完成结果：

```bash
env LD_PRELOAD=/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/2660-linan/conda-envs/gnss/lib/libstdc++.so.6 \
  python evaluate_original_fuxi_48h.py --plot-only \
  --output-dir work_dirs/original_fuxi_48h_vs_zero_zhd_zwd
```

## 验证状态

元数据检查成功，1452 个共同起报样本与对照一致。11 项 CPU 测试通过，覆盖原始 6/12h 回归、8 步预测反馈与日历更新、无观测数值读取、分布式补齐剔除、比率方向、口径校验、小样本拒绝与完整测试混比、CSV/图像输出。当前 GPU 正在训练，尚未运行原始 FuXi 全球八步 GPU smoke 或完整推理，暂没有新的真实 MAE 比较数值。
