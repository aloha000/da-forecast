# ZHD/ZWD 输入值置零消融

使用 configs/train_rollout_48h_zero_zhd_zwd.json。与原实验相比只增加 data.zero_delay_inputs=true 并更换输出目录；模型结构、参数量、初始化权重、种子、学习率、训练划分及 8×500 optimizer updates 保持不变。

- 输入仍是 [B,19,2,720,1440]，通道顺序 ZHD、ZWD。
- 在反标准化拆分、重新标准化、投影和有效性 mask 生成后，将两个输入通道全部置为数值 0。这里不是物理量 0 mm，而是各通道训练均值对应的标准化值。
- 原始 paired-valid mask 和站点静态场保持不变；OBS encoder、high/low fusion gate 保留且训练策略不变。即消融观测数值，不是移除所有站点信息；gate 不强制为 0。
- 第一步仍执行观测分支，后续 12–48h 步仍完全绕过观测分支，与原实验一致。
- 不要改变 model.zero_ztd=false、use_obs=true、obs_channels=2。
- 初始化继续使用原实验同一个 6h checkpoint，不使用训练完成的真实观测 rollout checkpoint，以便配对比较。该初始化曾使用真实观测，本实验结论是当前 rollout 训练/推理中观测数值的贡献，不是从未使用观测的纯 FuXi 对照。
- train/val/test 共用开关，恢复及测试时数据标识检查拒绝真实观测/置零配置混用。旧的非消融 checkpoint 保持兼容。

## 启动

在 da-forecast 项目目录和 gnss 环境下运行：

```bash
mkdir -p work_dirs/da_forecast_rollout48h_zero_zhd_zwd
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 \
  LD_PRELOAD=/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/2660-linan/conda-envs/gnss/lib/libstdc++.so.6 \
  python -u -m torch.distributed.run --standalone --nproc_per_node=4 train_rollout.py \
  --config configs/train_rollout_48h_zero_zhd_zwd.json \
  >> work_dirs/da_forecast_rollout48h_zero_zhd_zwd/train.log 2>&1 < /dev/null &
```

初始化路径已在配置中。目录若已有 resolved_config.json，使用新的 --output-dir；若已有 last.pt 可用 --resume 恢复，不删除旧实验。

评估使用 evaluate_rollout.py，--config 同上，--checkpoint 指向该消融实验的 stage_48h/best.pt，--steps 8；不要使用真实观测配置。结果 provenance 中 zero_delay_inputs=true。
