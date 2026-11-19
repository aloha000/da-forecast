# 48h 逐格点 MAE：与训练共存的 GPU 评估

2026-11-11 共存小样本测试：两套 main 模型分别用独立的四进程 FSDP 评估，均完成 4 个样本的 8 步推理。每卡峰值 allocated 约 9.63 GiB、reserved 约 13.19 GiB。实际数据为全局 720×1440，统计 t2m、r1000、u1000、v1000、tp、msl、z1000 的 +48h 时效。

运行限制和注意事项：

- `--gpu-memory-limit-gib 14` 限制评估进程的 PyTorch allocator；CUDA context/NCCL 等额外占用不完全受此上限控制。
- 初始化要求空闲显存至少达到上限 + 8 GiB。每个 batch 前再次检查可用余量（含评估自身可重用缓存）；不足时各 rank 保存部分统计后报错退出。
- 这降低风险，但不是训练显存的硬件隔离，也不能保证训练瞬时峰值永远安全。并发任务会争用计算、通信和共享文件系统带宽，训练可能变慢。
- 两模型必须用 `--variant zero` 和 `--variant obs` 分别启动进程。首次测试在同一进程连续加载第二个模型时额外占用了约 2.5 GiB，触发 14 GiB 上限。独立进程均成功，不需要提高上限。
- `num_workers=0` 减少与训练数据加载的 CPU/内存争用；每 10 个 batch/rank 保存一次中间结果。
- 当前完整评估目录：`work_dirs/comparison_rollout48h_zhd_zwd_vs_zero_test/spatial_48h`。日志为其下的 `evaluate.log`。不覆盖 `spatial_smoke_4`。

重启/续算（gnss 环境、项目目录下；先确认没有同目录的评估进程）：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export LD_PRELOAD=/inspire/ssd/tenant_predefaa-9a1b-4522-bb10-8850f313be13/global_user/2660-linan/conda-envs/gnss/lib/libstdc++.so.6

python -u -m torch.distributed.run --standalone --nproc_per_node=4 \
  evaluate_main_rollout_spatial.py --variant zero \
  --num-workers 0 --save-every 10 --gpu-memory-limit-gib 14 \
  --output-dir work_dirs/comparison_rollout48h_zhd_zwd_vs_zero_test/spatial_48h

# 仅在上一个命令成功退出后运行：
python -u -m torch.distributed.run --standalone --nproc_per_node=4 \
  evaluate_main_rollout_spatial.py --variant obs \
  --num-workers 0 --save-every 10 --gpu-memory-limit-gib 14 \
  --output-dir work_dirs/comparison_rollout48h_zhd_zwd_vs_zero_test/spatial_48h
```

同目录同配置会自动恢复 partial，完整的单模型结果则自动跳过。必须保持 4 个 rank；不允许把 CPU 或不同样本集合的 partial 混合进来。

## 数值校验

两次独立的限显存有观测模型小样本评估，逐格点标准化和物理 MAE 完全一致。但相同 checkpoint 在旧 `spatial_smoke_4` 中的结果与新共存结果存在差异：部分通道的 4 样本全球 MAE 相差约 0.01%～0.04%，原因尚未确定，不能直接归因于 GPU 争用。新旧 checkpoint/数据静态身份记录相同。

由于原实验的改善幅度也很小，完整评估仍保留原有严格校验：把逐格点误差汇总回全球纬度加权 MAE，核对保存的测试 CSV。若不通过，进程会报错，不会自动发布最终热图；全量 partial 已保留，后续可排查而不丢失累计结果。不要直接放宽阈值后宣称复现成功。

两个模型均完成并通过校验后，脚本自动保存 7 个气象要素的全球提升热图、合并 PNG、原始 MAE/有效计数 NPZ 和说明 JSON。`full_test=true` 才是 1452 个样本的完整结果。
