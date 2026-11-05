# ZHD/ZWD 预测与 rollout（predict_ztd 分支）

## 本次实现

新模式只监督 **ZHD、ZWD** 两个分量；ZTD 仅由 ZHD + ZWD 得到，没有独立 ZTD 预测头，也没有 ZTD loss。气象场的原有 ERA5 监督保留。

为兼容旧实验，使用新配置：
- configs/train_6h_delay_head.json：冻结已有气象/OBS 网络和新分量适配器，只训练两个残差头。
- configs/train_6h_predict_delays.json：解冻气象、OBS、分量适配器和两个头，小学习率联合微调。
- configs/train_12h_predict_delays.json：用训练好的 6h 分量模型生成新 rollout 数据后，训练下一跳。
- 旧 predict_ztd 配置仍是旧版直接预测 ZTD 的实验，不能误用来启动本实验。

兼容字段 predict_ztd=true 在新模式下表示“返回可供 rollout 使用的派生 ZTD”，不是启用 ZTD loss。data 和 model 中还必须同时设置 predict_delay_components=true。

输出 components 的形状为 [B,18,2,H,W]，分量顺序固定 [ZHD,ZWD]，单位 mm；weather 仍为原来的 70 通道标准化气象场。
内部 ztd 输出使用原 ZTD 归一化，方便复用 OBS encoder；先在 mm 中相加，再归一化，不是相加两个标准分数。

## 观测与标签

当前全球 NGL Zarr 已有 ztd、zwd：
- ZTD mean = 2320.2390672580573 mm，std = 169.4367826272024 mm。
- ZWD mean = 109.85565038275689 mm，std = 77.26589863040513 mm。
- 分别反归一化后计算 ZHD_mm = ZTD_mm - ZWD_mm，不能直接减两个标准化字段。
- 只使用同一时刻、同一源站点两项都有效的数据对。多个源格点落入同一 ERA 网格时，先求分量，再对相同有效站点集合平均。
- 输入为 T-6h 到 T 的 19 帧，每 20min 一帧；标签为 T+20min 到 T+6h 的 18 帧。
- ZHD 标签是从已有产品推导的，不是额外独立实测 ZHD；如果 NGL 产品本身的 ZWD 分离使用了气象模型，其误差也会继承到标签。
- 只有历史有有效分量、且该未来标签有效的位置参与分量损失；未来可用性不参与输入构建，也不因此删除天气样本。
- 目前按“站点格点”监督；碰撞合并后不是每个原始 GNSS 站点等权。

ZTD 的原 encoder 和 gate 不改参数形状，继续接收 ZTD 历史。新增 component_adapter 接收 ZHD/ZWD 的最近有效值、历史均值、可用掩膜（共 6 通道），经 patch 卷积加到高分辨率特征，初始化为 0，保持旧模型加载后的天气输出不变。
这是“额外分量特征适配器”，不是把原 Conv3d 的输入通道扩成三个；分量的完整 19 帧目前用作基线和历史均值，尚未新增双分量的独立时间卷积。

decoder 新增 zhd_head、zwd_head，各输出未来 18 帧残差，权重和偏置初始为 0。没有 decoder.ztd_head。
从旧天气模型初始化时严格检查其余键和形状，只允许新增这两个头及 component_adapter；从旧 ZTD-head 模型初始化时明确丢弃旧绝对 ZTD 头，不把它当残差头。

## 18 个时刻的 ZHD 怎样计算（普通文字版）

先说明：ERA5/天气模型只有 6h 间隔，不能凭空得到 18 份真实气象场。本实现使用“两个端点的算子变化量 + 可学习的逐时刻修正”。

1. 找到站点截至 T 的最近有效观测 ZHD，作为基准值。
2. 用 T 的气象场计算一个算子 ZHD。
3. 用模型预测的 T+6h 气象场，再计算一个算子 ZHD。
4. 第 3 步减第 2 步，得到算子预测的“未来 6 小时总变化量”。
5. 将这个变化量按时间比例分配到 18 帧，再加上每一帧自己学习的残差。

以下都是 mm，k 从 1 到 18：

~~~text
六小时变化量 = 算子ZHD(T+6h的预报气象场) - 算子ZHD(T的输入气象场)

第k帧ZHD预报 = 最近有效观测ZHD
             + (k / 18) * 六小时变化量 * delay_physics_weight
             + 第k帧学习的残差

第k帧ZTD预报 = 第k帧ZHD预报 + 第k帧ZWD预报
~~~

例如算子认为未来 6h 的 ZHD 一共增加 18 mm，观测基准是 2300 mm，暂不计学习残差：
- T+20min：2300 + 1 = 2301 mm。
- T+40min：2300 + 2 = 2302 mm。
- T+3h：2300 + 9 = 2309 mm。
- T+6h：2300 + 18 = 2318 mm。

18 帧时刻为：20、40、60、80、100、120、140、160、180、200、220、240、260、280、300、320、340、360 分钟。
ZWD 同样使用“最近有效观测 + 算子端点变化的时间插值 + 每帧残差”。并不是对 ZHD/ZWD 绝对值做从零开始的插值。

若 T 时刻缺测，基准使用前 6h 内最近有效观测，而不是用未来真值填补；这时基线可能偏旧，应在评估中关注缺测/陈旧观测的影响。完全没有历史分量的格点不伪造观测。

## 气象算子及其限制

代码：model/delay_physics.py。只在站点格点采样气象场后计算，避免创建全球 13 层物理中间数组；计算保持 FP32。

ZHD：先由海平面气压 msl、t2m、站点高度近似估算站点气压，再用 Saastamoinen 形式计算：

~~~text
P_station_hPa = (msl_Pa / 100) * exp(-9.80665 * height_m / (287.05 * t2m_K))

latitude_factor = cos_degrees(2 * latitude_deg)
denominator = 1 - 0.00266 * latitude_factor - 0.00028 * (height_m / 1000)
ZHD_mm = 2.2768 * P_station_hPa / denominator
~~~

其中 cos_degrees 表示以“度”为角度单位的余弦；代码用 deg2rad 转成弧度。气压用 hPa，高度输入用 m，输出为 mm。
站点压力的公式假设等温气柱，是近似，不是 ERA5 原生站点地面气压。

ZWD：由 13 个气压层的温度和相对湿度估算水汽压，计算湿折射率，再沿气压方向积分到站点地面。地下气压层不参加积分；地面湿度暂用最低地上层近似；50 hPa 以上湿延迟忽略。
使用原 FuXi 70 通道次序、t 为 K、r 为 %、msl 为 Pa；不支持任意通道排列。
当前高度接口对应 ERA 站点格点，纬度使用该格点纬度，属于网格近似；多个站点合并、山地地形、MSL 到站压换算、冰相饱和水汽压等都会产生误差，不能把算子结果视作 GNSS 真值。

**新配置默认 use_delay_physics=false。**
原始 NGL 的 ECEF 坐标现已提取，并通过 EGM96 转为正高：datasets/station_height_orthometric_era5.npy。
文件、生成脚本、检查结果和配置方式见 [STATION_HEIGHT.md](STATION_HEIGHT.md)。没有自动开启算子或改变已有训练配置；仍建议先验证算子误差。
算子关闭时，“六小时变化量”项完全去掉：预测 = 最近有效观测 + 学习残差，流程仍可训练、评估和 rollout。

启用前需要：
1. 将 data.station_height_path 设置为 datasets/station_height_orthometric_era5.npy，形状 [720,1440]、严格按 ERA 的 lat/lon 顺序，每个站点格点都有有效的代表站高，单位 m。非站点格点为 NaN。
2. 确认是正高（orthometric），配置 station_height_datum="orthometric"。ECEF 转换产生的椭球高不能不加转换直接冒充正高。
3. data.use_delay_physics 和 model.use_delay_physics 同时设 true；在 data 中填上述路径和高度基准；mean_std_dir 必须对应输入 ERA 的 70 通道归一化。
4. 先比较 ERA5 经过算子的 ZHD/ZWD 与观测的偏差、随时间的变化误差，尤其分海拔检查，再决定是否启用、使用多大权重。
5. model.delay_physics_weight 初始可取 1；model.delay_physics_detach=true 先阻止分量损失通过算子反传影响天气，确认可靠后另开阶段设为 false。关闭 detach 才有这条物理反馈梯度。
6. 算子/高度/尺度变化要启动新阶段（--init-checkpoint），不要作为原阶段 --resume。

不会替你填一个零高度文件或启动全量算子数据生成。

## 损失及精度

新配置的默认损失：

~~~text
total_loss = weather_loss
           + 0.1 * (ZHD_MAE_mm / 100)
           + 0.1 * (ZWD_MAE_mm / 100)
~~~

weather_loss 沿用纬度加权天气 MAE（默认 global_only）。
分量 MAE 按有效站点格点与时刻计数平均，不再额外做纬度加权。
100 mm 是显式的误差缩放尺度，**不是** ZHD/ZWD 训练标准差；两个尺度可通过 data/model.component_loss_scales 配置，需保持一致。若调整尺度/权重，原来的 total loss 数值就不可直接横比。
新模式不接受非零 ztd_loss_weight，防止以为它生效。

日志输出 zhd_mae_mm、zwd_mae_mm；验证输出 val_zhd_scaled、val_zwd_scaled，默认乘 100 即 mm。
新头热身以 components（两个加权、缩放分量损失之和）选 best；联合微调默认仍以 weather_global 选 best，需同时监控分量和天气误差。

BF16 的神经网络计算仍保留，但 FSDP 不把根模块物理输入/归一化统计转成 BF16。约 2300 mm 的数在 BF16 下相邻可表示值相差 16 mm，必须避免先舍入再转回 FP32。气象算子和分量基准叠加始终 FP32。

评估另输出：
- delay_mae_summary.json：两项分别的 mm MAE、缩放 MAE、有效数量、同一掩膜下的 persistence MAE。
- delay_mae_by_lead.csv：每个分量、18 个预报时效的 mm MAE。
- 原天气 weighted_mae_summary.json 等保持不变。
- 不生成独立 ZTD loss/MAE 文件；没有有效样本的评估项写 null，不写伪造的 0 MAE。

## 启动顺序

从项目根目录，在已安装 PyTorch/Zarr 等依赖的 gnss 环境运行。以下只是命令，代码修改过程中没有启动生产训练或生成全量数据。

先运行测试：

~~~bash
python -m unittest -v tests.test_delay_components
python train.py --config configs/train_6h_delay_head.json --check-data
~~~

新头热身（使用原来有真实 OBS 的 6h 天气模型，不是 zero-ztd 对照模型）：

~~~bash
mkdir -p work_dirs/da_forecast_6h_delay_head
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
  --standalone --nproc_per_node=4 train.py \
  --config configs/train_6h_delay_head.json \
  --init-checkpoint work_dirs/da_forecast_stage2_finetune_fuxi/best.pt \
  > work_dirs/da_forecast_6h_delay_head/train.log 2>&1 < /dev/null &
~~~

建议先用独立输出目录做小样本 smoke：在训练命令中加
--overfit-samples 16 --epochs 1 --output-dir work_dirs/da_forecast_6h_delay_smoke，并使用独立日志路径。
只有这个 GPU/FSDP smoke 也通过后，才开始全量训练。

热身完成后联合微调：

~~~bash
mkdir -p work_dirs/da_forecast_6h_predict_delays
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
  --standalone --nproc_per_node=4 train.py \
  --config configs/train_6h_predict_delays.json \
  --init-checkpoint work_dirs/da_forecast_6h_delay_head/best.pt \
  > work_dirs/da_forecast_6h_predict_delays/train.log 2>&1 < /dev/null &
~~~

6h 测试（省略动画，只统计指标）：

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
  --standalone --nproc_per_node=4 evaluate.py \
  --config configs/train_6h_predict_delays.json \
  --checkpoint work_dirs/da_forecast_6h_predict_delays/best.pt \
  --output-dir work_dirs/da_forecast_6h_predict_delays/test \
  --batch-size 1 --num-workers 2 --skip-animations --skip-mae-plots
~~~

有合格的 6h 新模型后生成独立的 12h 数据，绝不覆盖旧 Zarr：

~~~bash
python build_rollout_zarr.py --config configs/train_12h_predict_delays.json --plan

mkdir -p work_dirs/rollout_12h_predict_delays
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
  --standalone --nproc_per_node=4 build_rollout_zarr.py \
  --config configs/train_12h_predict_delays.json \
  --splits train val test --batch-size 1 --num-workers 2 \
  > work_dirs/rollout_12h_predict_delays/build.log 2>&1 < /dev/null &
~~~

先加 --max-samples 4 可验证少量生成，然后去掉该参数继续补齐；不完整的 split 不允许直接开始训练。--plan 会报告存储容量需求，全球气象场仍很大。

新文件 datasets/era5_rollout12h_predict_delays.zarr 保持原天气字段 z、z_fuxi6h 形式，新增稀疏的 component_input、component_input_mask、component_target、component_target_mask 和必要上下文；保留派生 ztd_input，不写 ztd_target。
第 2 跳输入为 [ERA(T), forecast(T+6)]；分量历史为 [T 的观测端点, 18 个预报分量]。第 2 跳监督时刻是 T+6h20min 到 T+12h。标签在第一跳推理完成后才读取，不参与输入/掩膜生成。

然后训练 12h：

~~~bash
mkdir -p work_dirs/da_forecast_12h_predict_delays
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
  --standalone --nproc_per_node=4 train.py \
  --config configs/train_12h_predict_delays.json \
  --init-checkpoint work_dirs/da_forecast_6h_predict_delays/best.pt \
  > work_dirs/da_forecast_12h_predict_delays/train.log 2>&1 < /dev/null &
~~~

12h 评估复用上面的 evaluate.py 命令，把 config、checkpoint、output-dir 换成对应 12h 路径即可。
model.ztd_rollout.forecast_rollout 支持继续反馈分量的 18h/24h 在线推理；当前离线 build_rollout_zarr.py / 训练配置仍只生成和训练第 2 跳，不声称已实现第 3/4 跳离线数据生成。

## 代码位置

- data/delay_components.py：配对反归一化、分量读取、物理上下文检查。
- data/da_dataset.py：输入 19 帧和监督 18 帧。
- model/delay_components.py：基线、输入适配器、分量残差叠加、损失、下一跳输入。
- model/delay_physics.py：可微站点气象算子。
- model/da_forecast_model.py：两个预测头、特征融合、可选算子。
- train.py：冻结策略、兼容加载、训练/验证、FSDP 物理输入精度。
- evaluate.py：两项 mm MAE 和 persistence 对照。
- build_rollout_zarr.py、data/ztd_fields.py、data/rollout_fields.py：独立双分量 rollout 存储与读取。
- tests/test_delay_components.py：新模型与微型数据的单元/端到端测试。

自动测试使用 CPU 和微型模型/数据，不等同于全球模型的四卡显存或收敛验证。
