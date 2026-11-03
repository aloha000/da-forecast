# 6h 零 ZTD 对照实验：第二阶段

配置：`configs/train_6h_zero_ztd_stage2.json`。

固定第一阶段训练好的 OBS encoder、high_fusion、low_fusion（包括 gate），
微调 era_encoder、time_embed、backbone（Transformer/FPN）、decoder。
保留 `use_obs=true, zero_ztd=true`：ZTD 数值置零，但有效 mask 和站点静态信息保留。
不要将 OBS 分支删除或将 mask 全置零。

完整初始化自：
`work_dirs/da_forecast_global_6h_fsdp_train_obs-bs2_zero-ztd/best.pt`。
只加载模型参数，优化器、调度器、epoch/step 重新开始。
冻结通过 requires_grad 实现，不截断 fusion 对 FuXi 输入特征的梯度。

这是原始 6h 任务的阶段切换，不是 12h rollout，不需要生成新 Zarr。
保持当前 6h 配置的数据划分和其余训练设置：
40 epochs、每卡 batch=2、梯度累积=4、4卡有效 batch=32，
LR=1e-5、最低 LR=1e-6、warmup=100、BF16/FSDP、global_only loss。
训练前进行一次验证。与有观测第二阶段比较时还应核对双方训练预算一致。

## 运行

在 da-forecast 根目录，激活 gnss 环境后执行。
先确认 GPU 空闲，不要与其他占用 GPU 的训练同时启动。

数据检查（不训练）：

```bash
python train.py --config configs/train_6h_zero_ztd_stage2.json --check-data
```

后台训练：

```bash
mkdir -p work_dirs/da_forecast_6h_zero_ztd_stage2
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
  --standalone --nproc_per_node=4 train.py \
  --config configs/train_6h_zero_ztd_stage2.json \
  --init-checkpoint work_dirs/da_forecast_global_6h_fsdp_train_obs-bs2_zero-ztd/best.pt \
  > work_dirs/da_forecast_6h_zero_ztd_stage2/train.log 2>&1 < /dev/null &
```

不要加 `--init-fuxi-only`，否则不能完整继承 OBS/gate。
不要用 `--pretrained-fuxi` 或 `--resume` 代替新阶段初始化。
只有恢复本次第二阶段的中断训练时，才改用其自身 checkpoint 的 `--resume`，
同时去掉 `--init-checkpoint`。
重新运行时上述重定向会覆盖 train.log，请先备份旧日志，并保留旧实验的输出目录。

日志应显示：

```text
discarded_obs_tensors: 0
trainable obs_encoder: 0
trainable high_fusion: 0
trainable low_fusion: 0
```

初始化报告的实际显示是 Python 字典格式；另外四个 FuXi 模块的 trainable 应大于零。
初始化后、训练前的验证可用于检查第一阶段权重是否正确继承；
比较时需保持相同样本集合、损失定义和精度设置。

## 测试

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc_per_node=4 evaluate.py \
  --config configs/train_6h_zero_ztd_stage2.json \
  --checkpoint work_dirs/da_forecast_6h_zero_ztd_stage2/best.pt \
  --output-dir work_dirs/da_forecast_6h_zero_ztd_stage2/test \
  --batch-size 1 --num-workers 2
```

必须使用这次第二阶段的 best.pt，不能误用第一阶段或 12h checkpoint。
