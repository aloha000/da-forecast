# NGL 站点高度文件

已从 datasets/global_ngl 原始 SINEX-TRO 文件提取站点 ECEF 坐标，生成：

- datasets/station_height_orthometric_era5.npy：用于 station_height_path，EGM96 正高，单位 m。
- datasets/station_height_ellipsoidal_era5.npy：由 IGS20 ECEF 在 WGS84 椭球上换算的椭球高，供核对。
- datasets/station_geoid_undulation_era5.npy：EGM96 大地水准面起伏。
- datasets/station_height_count_era5.npy：每个 ERA 格点合并的源站点数量。
- datasets/station_height_latitude.npy、station_height_longitude.npy：网格坐标。
- datasets/station_height_stations.csv：逐站点 ID、原始 XYZ、椭球高、正高、改正量、原始压缩包/成员文件、映射位置。
- datasets/station_height_manifest.json：单位、基准、来源、处理方法、SHA256 校验和及统计。

高度网格形状 [720,1440]，按实际 ERA5 Zarr 坐标排列；非站点位置 NaN，不用 0 冒充站点高度。
当前数据加载器会对原始 NGL 站点坐标重新映射；本脚本复用了同一个 station_grid_mapping 函数，而不是直接照抄原 NGL 格点索引。

## 生成方法

1. 读取现有 NGL Zarr 的 station 字段，只处理现有数据实际选中的 11,183 个站点，不重新筛选原始目录。
2. 按原始构建脚本的年份与 ZIP 成员顺序，读取第一个有效的同名站点 TROP/STA_COORDINATES 记录。
3. 检查原始 XYZ 换算的经纬度与 Zarr 的 station_lat/station_lon 一致。最大差约 0.0000153 度，符合保存为 float32 带来的舍入。
4. ECEF 坐标换算出椭球高 h。SITE/ID 中 APP_H 仅记录作核对，不把它当正高。
5. 在实际站点经纬度处双线性插值 EGM96 起伏 N，得到正高 H = h - N。
6. 按实际加载器映射到 ERA 格点；合并多个站点时对各自转换后的正高取算术平均。

官方 EGM96 网格来自 https://cdn.proj.org/us_nga_egm96_15.tif ，约 2.6 MiB，保存于 datasets/geoid/。GeoTIFF 元数据为 WGS84 三维坐标到 EGM96 height (EPSG:5773)，geoid_undulation，单位 metre，Public Domain。
文件为 15 arc-minute 网格；插值按 pixel-is-point 坐标处理，经度周期衔接，含南北极。

结果：11,183 个源站点，10,518 个有效 ERA 格点，其中 665 个格点合并多个站点。源站点正高约 -393.90～5031.87 m。所有目标站点格点均有有效高度。
这些是观测站高度，不是全球地形高程，不能把非站点 NaN 区域用作地形场。

## 配置

在 data 中增加：

~~~json
"station_height_path": "datasets/station_height_orthometric_era5.npy",
"station_height_datum": "orthometric"
~~~

需要启用气象算子时，另外将 data.use_delay_physics 和 model.use_delay_physics 同时设为 true。
本次只生成文件、脚本和文档，没有修改运行中的训练配置，也没有自动启用算子。

建议以 --init-checkpoint 开启带算子的独立阶段/输出目录；不要更改原实验配置后直接 --resume。
先用 --check-data 检查完整配置，再小样本验证算子变化量和观测变化量。该高度转换不能消除天气算子的 MSL→站压近似、地形代表性和水汽廓线误差。

## 复现

~~~bash
python preprocessing/build_station_height.py --workers 8
python -m unittest -v tests.test_station_height
~~~

已有输出时脚本拒绝覆盖。需要重新生成比较时使用独立目录：

~~~bash
python preprocessing/build_station_height.py --workers 8 --output-dir datasets/station_height_rebuild
~~~

脚本只读原始压缩包和现有 ERA/NGL Zarr，不修改其内容。
3 项单元测试覆盖 ECEF 高度换算、双线性插值/跨经度边界/极点、同名站点提取及 h-N 的方向。

## 注意事项

- IGS20 与 WGS84 的具体参考框架实现并非完全相同；此处采用 WGS84 椭球近似。
- EGM96 是全球模型正高，不等于每个国家当地高程基准；局地误差仍存在。
- 每个站点使用与既有 Zarr 位置对应的一次代表坐标，没有估计多年站点位移。
- 同一 ERA 格点可能有不同高度的站点。目前均值高度只是代表值；观测缺测改变有效站点集合时，实际平均高度也可能变化。
