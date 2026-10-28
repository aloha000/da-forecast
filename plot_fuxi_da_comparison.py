"""Create channel statistics, figures and a Chinese report from paired caches."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from evaluate import CHANNELS, channel_unit

IMPORTANT = ['r850','t2m','msl','t850','z500','u10','v10']
BLUE, ORANGE = '#2878b5', '#e77c32'


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentage(original, da):
    return float(100*(original-da)/original) if original>1e-12 else float('nan')


def make_report(out):
    out=Path(out)
    metadata=json.loads((out/'metadata.json').read_text())
    identity=metadata['identity']
    requested=identity['evaluation_indices']
    missing=[i for i in requested if not (out/'samples'/f'{i:06d}.npz').exists()]
    if missing:
        raise RuntimeError(f'{len(missing)} requested samples missing: {missing[:10]}; refuse incomplete report')
    keys=['denominator']+[f'{m}_{u}' for m in ('original','da','persistence') for u in ('normalized','physical')]
    sums={k:np.zeros(70,dtype=np.float64) for k in keys}
    point_values=[]; point_valid=[]; point_times=[]
    valid_times=[]
    for index in requested:
        with np.load(out/'samples'/f'{index:06d}.npz') as record:
            assert int(record['index'])==index
            for key in keys:
                sums[key]+=record[key]
            valid_times.append(str(record['valid_time']))
            if 'station_values' in record:
                point_values.append(record['station_values'])
                point_valid.append(record['station_target_valid'])
                point_times.append(str(record['valid_time']))
    if not np.all(sums['denominator']>0):
        raise ValueError('a channel has no valid target weights')
    rows=[]
    for i,name in enumerate(CHANNELS):
        row=dict(channel=name,unit=channel_unit(name))
        for unit in ('normalized','physical'):
            original=float(sums[f'original_{unit}'][i]/sums['denominator'][i])
            da=float(sums[f'da_{unit}'][i]/sums['denominator'][i])
            row.update({f'original_mae_{unit}':original,f'da_mae_{unit}':da,
                        f'da_minus_original_{unit}':da-original,
                        f'da_improvement_percent_{unit}':percentage(original,da),
                        f'persistence_mae_{unit}':float(sums[f'persistence_{unit}'][i]/sums['denominator'][i])})
        rows.append(row)
    write_csv(out/'channel_comparison.csv',rows)
    original=float(sums['original_normalized'].sum()/sums['denominator'].sum())
    da=float(sums['da_normalized'].sum()/sums['denominator'].sum())
    summary=dict(samples=len(requested),full_test_samples=identity['full_test_samples'],
                 is_full_test=len(requested)==identity['full_test_samples'],
                 valid_time_first=min(valid_times),valid_time_last=max(valid_times),
                 original_global_latitude_weighted_mae_normalized=original,
                 da_global_latitude_weighted_mae_normalized=da,
                 improvement_percent=percentage(original,da),
                 improved_channels=sum(r['da_mae_normalized']<r['original_mae_normalized'] for r in rows),
                 worse_channels=sum(r['da_mae_normalized']>r['original_mae_normalized'] for r in rows),
                 amp=identity['amp'],conds_path=identity['data']['conds_path'],
                 conds_sha256=identity['conds_sha256'],selection=metadata['selection'],
                 note=metadata['note'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')

    def bar_groups(groups,unit,filename,title):
        fig,axes=plt.subplots(len(groups),1,figsize=(15,3*len(groups)),squeeze=False)
        for ax,(label,names) in zip(axes[:,0],groups):
            selected=[rows[CHANNELS.index(n)] for n in names]
            x=np.arange(len(names)); width=.38
            ax.bar(x-width/2,[r[f'original_mae_{unit}'] for r in selected],width,label='Original FuXi',color=BLUE)
            ax.bar(x+width/2,[r[f'da_mae_{unit}'] for r in selected],width,label='DA best',color=ORANGE)
            ax.set_xticks(x,names,rotation=30 if len(names)>10 else 0)
            ax.set_ylabel(label);ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True)
        axes[0,0].legend(ncol=2)
        fig.suptitle(title,fontsize=16)
        fig.tight_layout(rect=(0,0,1,.98))
        fig.savefig(out/filename,dpi=160);plt.close(fig)

    groups=[(f'{kind}: normalized MAE',CHANNELS[i*13:(i+1)*13]) for i,kind in enumerate(('Z','T','U','V','RH'))]
    groups.append(('Surface: normalized MAE',CHANNELS[65:]))
    bar_groups(groups,'normalized','mae_70_channels_normalized.png',
               f'Global latitude-weighted MAE | {len(requested)} paired +6h forecasts | BF16')
    physical_groups=[('Geopotential height (gpm)',CHANNELS[:13]),
                     ('Temperature (K)',CHANNELS[13:26]+['t2m']),
                     ('U wind (m/s)',CHANNELS[26:39]+['u10']),
                     ('V wind (m/s)',CHANNELS[39:52]+['v10']),
                     ('Relative humidity (percentage points)',CHANNELS[52:65]),
                     ('MSL pressure (hPa)',['msl']),('Precipitation (mm)',['tp'])]
    bar_groups(physical_groups,'physical','mae_70_channels_physical.png',
               'Physical-unit MAE comparison | separate axes for different units')
    fig,ax=plt.subplots(figsize=(13,18))
    improvements=np.array([r['da_improvement_percent_normalized'] for r in rows])
    ax.barh(np.arange(70),improvements,color=np.where(improvements>=0,'#33945a','#d65353'))
    ax.set_yticks(np.arange(70),CHANNELS,fontsize=8);ax.invert_yaxis();ax.axvline(0,color='black',lw=.8)
    ax.set_xlabel('(Original MAE - DA MAE) / Original MAE (%)  |  positive = DA better')
    ax.set_title('DA improvement relative to original FuXi: 70 channels')
    ax.grid(axis='x',alpha=.25);ax.set_axisbelow(True);fig.tight_layout()
    fig.savefig(out/'mae_70_channels_improvement.png',dpi=160);plt.close(fig)

    station_rows=[]
    series_rows=[]
    if point_values:
        values=np.stack(point_values).astype(np.float64) # time, source, channel, station
        valid=np.stack(point_valid)
        mean=np.load(Path(identity['data']['mean_std_dir'])/'mean_era5.npy').reshape(1,1,70,1)
        std=np.load(Path(identity['data']['mean_std_dir'])/'std_era5.npy').reshape(1,1,70,1)
        values=values*std+mean
        values[:,:,:13]/=9.80665
        values[:,:,68]/=100
        values[:,:,69]=np.maximum(np.expm1(values[:,:,69]),0)
        values=np.where(valid[:,None],values,np.nan)
        order=np.argsort(np.array(point_times))
        values=values[order]
        times=pd.to_datetime(np.array(point_times)[order])
        plot_dir=out/'station_timeseries';plot_dir.mkdir(exist_ok=True)
        for station_idx,station in enumerate(metadata['selection']['stations']):
            fig,axes=plt.subplots(4,2,figsize=(17,13),sharex=True)
            for ax,name in zip(axes.ravel(),IMPORTANT):
                ch=CHANNELS.index(name)
                observations=values[:,:,ch,station_idx]
                orig_mae=float(np.nanmean(np.abs(observations[:,1]-observations[:,0])))
                da_mae=float(np.nanmean(np.abs(observations[:,2]-observations[:,0])))
                station_rows.append(dict(station=station['id'],channel=name,unit=channel_unit(name),
                                         samples=int(np.isfinite(observations[:,0]).sum()),
                                         original_mae=orig_mae,da_mae=da_mae,
                                         da_improvement_percent=percentage(orig_mae,da_mae)))
                for t,v in zip(times,observations):
                    series_rows.append(dict(valid_time=str(t),station=station['id'],channel=name,
                                            unit=channel_unit(name),era5=v[0],original_fuxi=v[1],da=v[2]))
                displayed=observations-273.15 if name.startswith('t') and name!='tp' else observations
                for source,label,color in [(0,'ERA5','black'),(1,'Original FuXi',BLUE),(2,'DA best',ORANGE)]:
                    ax.plot(times,displayed[:,source],label=label,color=color,lw=1.5,marker='.',ms=3)
                unit='degC' if name.startswith('t') and name!='tp' else channel_unit(name)
                ax.set_title(f'{name} | MAE: FuXi {orig_mae:.3g}, DA {da_mae:.3g}')
                ax.set_ylabel(unit);ax.grid(alpha=.25)
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                ax.tick_params(axis='x',labelbottom=True)
            axes.ravel()[-1].axis('off');axes.ravel()[0].legend(fontsize=9)
            fig.suptitle(f"GNSS station {station['id']} ({station['latitude']:.2f}, {station['longitude']:.2f})\n"
                         f"Nearest ERA5 grid ({station['grid_latitude']:.2f}, {station['grid_longitude']:.2f}); "
                         f"independent +6h forecasts at valid times {times[0]:%Y-%m-%d} to {times[-1]:%Y-%m-%d}",fontsize=13)
            fig.tight_layout(rect=(0,0,1,.94))
            fig.savefig(plot_dir/f"{station['id']}_important_channels.png",dpi=160);plt.close(fig)
    write_csv(out/'station_window_mae.csv',station_rows)
    write_csv(out/'station_timeseries.csv',series_rows)
    write_csv(out/'selected_stations.csv',metadata['selection']['stations'])

    label='完整测试集' if summary['is_full_test'] else '小样本试运行，不代表完整测试集'
    lines=['# 原始 FuXi 与 DA 模型测试集对比','',f'范围：{label}；{len(requested)} 个独立 +6h 预报样本。',
           f"有效时刻：{summary['valid_time_first']} ～ {summary['valid_time_last']}。",'',
           '## 评估口径','',
           '- 两个模型使用相同 ERA5 输入、标签、静态场和 BF16/FSDP 精度；原始模型不输入 ZTD。',
           '- 统计区域为全球所有有效 ERA5 格点，按 cos(latitude) 加权；不是只统计站点或 halo。',
           '- 标准化 MAE 按 70 通道及有效空间权重统一汇总；物理 MAE 按通道报告，不跨单位平均。',
           '- 单通道 MAE = Σ[有效掩膜 × cos(纬度) × |预报−ERA5|] / Σ[有效掩膜 × cos(纬度)]，求和覆盖全部测试时间和全球格点。',
           '- 物理 MAE 先将预报和标签分别反归一化再求绝对误差；降水包含 expm1 逆变换，不能直接用标准化 MAE 乘标准差。',
           '- 湿度误差的 % 表示相对湿度的百分点；“提升比例 %”才是误差的相对变化。温度 MAE 的 K 与摄氏度差值相同。',
           f"- 静态场：`{identity['data']['conds_path']}`；SHA256：`{identity['conds_sha256']}`。",
           '- 当前使用修正后的静态场；DA 现有权重在修正前训练。因此结果是当前输入约定下的对比，不是旧日志的精确复现。',
           '- r800 不在 70 通道内，重点湿度通道使用 r850（850 hPa）。',
           '- 相同输入和精度减少比较误差，但单个模型对比不能把所有增益严格归因于 ZTD 的因果作用。','',
           f"模型权重：原始 `{identity['fuxi_checkpoint']['path']}`；DA `{identity['da_checkpoint']['path']}`。",
           f"DA checkpoint：第 {metadata.get('da_epoch_zero_based', -1)+1} 个 epoch，step={metadata.get('da_step', 'unknown')}。",
           '本报告不是 FuXi 官方论文评测，原始模型的数值取决于当前数据、静态场及推理精度。','',
           '## 整体结果','',
           '| 模型 | 全球标准化纬度加权 MAE |','|---|---:|',
           f'| 原始 FuXi | {original:.8f} |',f'| DA best | {da:.8f} |','',
           f"DA 相对提升 **{summary['improvement_percent']:.3f}%**；"
           f"70 通道中 {summary['improved_channels']} 个改善、{summary['worse_channels']} 个变差。",
           '提升比例=(FuXi MAE−DA MAE)/FuXi MAE×100%；正数表示 DA 更好。','',
           '## 重点通道（物理单位）','',
           '| 通道 | 单位 | FuXi MAE | DA MAE | DA−FuXi | 提升比例 |',
           '|---|---|---:|---:|---:|---:|']
    for name in IMPORTANT:
        r=rows[CHANNELS.index(name)]
        lines.append(f"| {name} | {r['unit']} | {r['original_mae_physical']:.6g} | {r['da_mae_physical']:.6g} | {r['da_minus_original_physical']:+.6g} | {r['da_improvement_percent_physical']:+.3f}% |")
    ranked=sorted(rows,key=lambda r:r['da_improvement_percent_normalized'],reverse=True)
    lines+=['','按标准化 MAE 相对提升排序：','',
            '- 排名前五：'+ '；'.join(f"{r['channel']} {r['da_improvement_percent_normalized']:+.2f}%" for r in ranked[:5]),
            '- 排名后五：'+ '；'.join(f"{r['channel']} {r['da_improvement_percent_normalized']:+.2f}%" for r in ranked[-5:]),
            '', '## 随机时段与站点','',
            f"随机种子 {identity['seed']}；有效时刻 {metadata['selection']['valid_start']} ～ {metadata['selection']['valid_end']}。",
            '从全球站点随机顺序中选择 5 个不同 ERA5 格点、所需观测时间覆盖率至少 80% 的站点。',
            '曲线是站点最近 ERA5 格点的气象量，不是 GNSS 实测气象量；ERA5 是对照真值。',
            '每个时序点分别使用当时 ERA5 和历史观测预报 +6h，不是连续滚动积分。','',
            '| 站点 | 纬度 | 经度 | ZTD 覆盖率 |','|---|---:|---:|---:|']
    for s in metadata['selection']['stations']:
        lines.append(f"| {s['id']} | {s['latitude']:.4f} | {s['longitude']:.4f} | {100*s['ztd_coverage']:.1f}% |")
    if station_rows:
        better=sum(r['da_mae'] < r['original_mae'] for r in station_rows)
        worse=sum(r['da_mae'] > r['original_mae'] for r in station_rows)
        lines += ['', f'这个随机周的 {len(station_rows)} 组“站点×变量”中，DA 改善 {better} 组、变差 {worse} 组。',
                  '这是局部短时段结果，不代表全年全球统计；全球平均改善也不保证每个站点、变量和时段都改善。',
                  '重点站点 MAE：r850 单位为相对湿度百分点，t2m 单位为 K。', '',
                  '| 站点 | r850 FuXi | r850 DA | r850 提升 | t2m FuXi | t2m DA | t2m 提升 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        lookup={(r['station'],r['channel']):r for r in station_rows}
        for s in metadata['selection']['stations']:
            rh=lookup[(s['id'],'r850')];temp=lookup[(s['id'],'t2m')]
            lines.append(f"| {s['id']} | {rh['original_mae']:.4f} | {rh['da_mae']:.4f} | {rh['da_improvement_percent']:+.2f}% | "
                         f"{temp['original_mae']:.4f} | {temp['da_mae']:.4f} | {temp['da_improvement_percent']:+.2f}% |")
    lines+=['','## 图件','',
            '![70 通道标准化 MAE](mae_70_channels_normalized.png)','',
            '![70 通道物理 MAE](mae_70_channels_physical.png)','',
            '![相对提升](mae_70_channels_improvement.png)','']
    if point_values:
        for s in metadata['selection']['stations']:
            lines += [f"![{s['id']} 时序](station_timeseries/{s['id']}_important_channels.png)",'']
    else:
        lines += ['本次抽样没有覆盖选定时段；试运行不生成站点时序。','']
    lines+=['## 全部 70 通道','',
            '| 通道 | FuXi 标准化 MAE | DA 标准化 MAE | 标准化提升 | 物理单位 | FuXi 物理 MAE | DA 物理 MAE |',
            '|---|---:|---:|---:|---|---:|---:|']
    for r in rows:
        lines.append(f"| {r['channel']} | {r['original_mae_normalized']:.7f} | {r['da_mae_normalized']:.7f} | {r['da_improvement_percent_normalized']:+.3f}% | {r['unit']} | {r['original_mae_physical']:.6g} | {r['da_mae_physical']:.6g} |")
    lines += ['', '完整数值：channel_comparison.csv；站点时序：station_timeseries.csv；',
              '站点时段 MAE：station_window_mae.csv；模型、输入和选样信息：metadata.json。','']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('REPORT_COMPLETE '+str(out/'REPORT.md'),flush=True)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('selection','note')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output_dir');args=p.parse_args()
    make_report(Path(args.output_dir))
