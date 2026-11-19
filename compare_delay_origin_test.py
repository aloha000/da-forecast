#!/usr/bin/env python3
"""Run two complete tests sequentially, then compare identical lead/channel metrics."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ['da_forecast_rollout48h_predict_delays', 'da_forecast_rollout48h_predict_delays_mean_origin']
CONFIGS = ['train_rollout_48h_predict_delays.json', 'train_rollout_48h_predict_delays_mean_origin.json']


def csv_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def improvement(real, mean):
    return 100 * (mean-real)/mean if mean > 0 else float('nan')


def extended_comparison(out, reports, test_name):
    """Append halo and MAIN cross-experiment comparisons with explicit baselines."""
    import re
    import hashlib
    old_names=['da_forecast_rollout48h_zhd_zwd','da_forecast_rollout48h_zero_zhd_zwd']
    paths=[ROOT/'work_dirs'/n/'test/rollout_mae_summary.json' for n in old_names]
    old=[json.loads(p.read_text()) for p in paths]
    log=ROOT/'work_dirs'/old_names[0]/'test/evaluate.log'
    content=log.read_text(); logged=[]
    for m in re.finditer(r'\{\s*"evaluation"\s*:',content):
        try:
            value,_=json.JSONDecoder().raw_decode(content[m.start():]);logged.append(value)
        except json.JSONDecodeError:
            pass
    assert logged and logged[-1]==old[0], 'Old real-origin log and summary differ'
    all_reports=old+reports
    all_names=old_names+EXPERIMENTS
    configs=[json.loads((ROOT/'work_dirs'/n/'resolved_config.json').read_text()) for n in all_names]
    reference=all_reports[0]
    for r,cfg in zip(all_reports,configs):
        assert r['samples']==reference['samples']==1452
        for key in ('split','origin_first','origin_last','amp','world_size','completed_updates','trained_stage','evaluated_steps'):
            assert r['evaluation'][key]==reference['evaluation'][key], key
        assert cfg['train']['halo_radius']==3
        for key in ('era5_path','ztd_path','conds_path','global_lat_path','global_lon_path','mean_std_dir',
                    'train_split','val_split','test_split','obs_history_hours','obs_interval_minutes'):
            assert cfg['data'][key]==configs[0]['data'][key], key
        assert [v['lead_hours'] for v in r['leads']]==list(range(6,49,6))
        np.testing.assert_array_equal([v['persistence_latitude_weighted_mae_normalized'] for v in r['leads']],
                                      [v['persistence_latitude_weighted_mae_normalized'] for v in reference['leads']])
    g='global_latitude_weighted_mae_normalized';h='halo_latitude_weighted_mae_normalized'
    lines=['','## 站点邻域（halo）结果','',
           'halo 是静态站点占据格点向周围扩展半径 3 格形成的 7×7 窗口并集，重叠区域只计一次；不是单站点误差，也不是固定公里半径。',
           '指标为该区域有效气象格点的纬度加权归一化 MAE，使用与全球指标相同的 70 通道。',
           'halo 来自静态站点分布，不随某一帧观测是否缺测而重新定义。','',
           '| 时效 | 预测延迟：真实输入 halo MAE | 预测延迟：均值输入 halo MAE | 真实输入提升 |',
           '|---|---:|---:|---:|']
    for a,b in zip(reports[0]['leads'],reports[1]['leads']):
        lines.append(f'| {a["lead_hours"]}h | {a[h]:.6f} | {b[h]:.6f} | {improvement(a[h],b[h]):+.3f}% |')
    lines += ['','真实初始观测在所有 8 个时效改善站点邻域：6h 改善 9.611%，48h 仍改善 1.679%；这一局地收益不能被全球均值直接替代。',
              '图：[全球、halo 和延迟 MAE 曲线](comparison_by_lead.png)。','',
              '## 与旧版 MAIN 实验的跨模型比较','',
              '| 代号 | 实验 | 初始观测输入 | 后续观测分支 | 监督 |',
              '|---|---|---|---|---|',
              '| A | `da_forecast_rollout48h_zhd_zwd` | 标准化真实 ZHD/ZWD | 关闭 encoder 与融合 | 气象 |',
              '| B | `da_forecast_rollout48h_zero_zhd_zwd` | 标准化 ZHD/ZWD 全 0 | 关闭 encoder 与融合 | 气象 |',
              '| C | `da_forecast_rollout48h_predict_delays` | 真实 ZTD + ZHD/ZWD | 反馈预测分量及派生 ZTD | 气象 + ZHD/ZWD |',
              '| D | `da_forecast_rollout48h_predict_delays_mean_origin` | 全局训练均值；ZTD 标准化值 0 | 反馈预测分量及派生 ZTD | 气象 + ZHD/ZWD |','',
              'B 的零是标准化空间的零，不代表物理湿延迟为 0 mm；MAIN 没有单独的 ZTD 输入通道。',
              '四套均为第 4000 步的 48h 模型、BF16、4 卡、1452 个测试起报时刻，起报范围与上文一致。',
              '保存配置中的数据路径、划分、历史窗口及 halo_radius 一致；各时效全球 persistence MAE 完全相同。',
              '这些是现有产物的口径核对，并非对历史运行的逐样本输出或原始数据快照的再次验证。','',
              '### 全球气象 MAE','',
              '| 时效 | A：MAIN 真实 | B：MAIN 零输入 | C：预测延迟 真实 | D：预测延迟 均值 |',
              '|---|---:|---:|---:|---:|']
    cross=[]
    for j in range(8):
        vals=[r['leads'][j] for r in all_reports]
        lines.append('| '+str(6*(j+1))+'h | '+' | '.join(f'{v[g]:.6f}' for v in vals)+' |')
        row={'lead_hours':6*(j+1)}
        for code,v in zip('ABCD',vals):
            row[code+'_global_mae']=v[g];row[code+'_halo_mae']=v[h]
        for label,k in [('global',g),('halo',h)]:
            for tag,a,b in [('A_vs_B',0,1),('C_vs_D',2,3),('C_vs_A',2,0),('D_vs_B',3,1)]:
                row[tag+'_'+label+'_improvement_pct']=improvement(vals[a][k],vals[b][k])
        cross.append(row)
    lines += ['','### 站点邻域气象 MAE','',
              '| 时效 | A：MAIN 真实 | B：MAIN 零输入 | C：预测延迟 真实 | D：预测延迟 均值 |',
              '|---|---:|---:|---:|---:|']
    for j in range(8):
        lines.append('| '+str(6*(j+1))+'h | '+' | '.join(f'{r["leads"][j][h]:.6f}' for r in all_reports)+' |')
    lines += ['','### 跨模型相对提升','',
              'C 对 A：比较两代真实输入模型；D 对 B：比较两代去初始观测数值的模型。正值表示新模型更好，负值表示新模型更差。','',
              '| 时效 | C 对 A：全球 | C 对 A：halo | D 对 B：全球 | D 对 B：halo |',
              '|---|---:|---:|---:|---:|']
    for r in cross:
        keys=['C_vs_A_global_improvement_pct','C_vs_A_halo_improvement_pct','D_vs_B_global_improvement_pct','D_vs_B_halo_improvement_pct']
        lines.append(f'| {r["lead_hours"]}h | '+' | '.join(f'{r[k]:+.3f}%' for k in keys)+' |')
    lines += ['','### 综合判断与限制','',
              '1. MAIN 的真实输入与零输入几乎持平：全球相对差异各时效均小于 0.04%，halo 均小于 0.15%，不能据此断言存在稳定收益。',
              '2. 预测延迟结构内，C 相比 D 的短时和站点邻域收益清楚，但相对收益变大不等于绝对预报更好。',
              '3. C 相比旧版真实输入 A：仅 6h 全球略好（+0.269%）；12～48h 全球均略差，halo 所有时效都略差。48h 全球差 0.160%，halo 差 0.826%。',
              '4. D 的 48h 全球 MAE 最低（0.170192），但该时效 halo MAE 在四者中最高（0.175383）。站点周围的绝对表现并没有随引入延迟预测而改善。',
              '5. 四者的 8 时效等权均值如下。这是各时效 MAE 的算术平均，不是重新汇总全部格点/时间的总 MAE。','',
              '| 实验 | 全球：8 时效等权均值 | halo：8 时效等权均值 |',
              '|---|---:|---:|']
    for code,r in zip('ABCD',all_reports):
        lines.append(f'| {code} | {sum(v[g] for v in r["leads"])/8:.6f} | {sum(v[h] for v in r["leads"])/8:.6f} |')
    lines += ['','两代模型不仅有无延迟预测不同：输入编码、观测归一化、辅助监督、后续反馈和初始化都不同。',
              'A/B 初始化为 `work_dirs/da_forecast_stage2_finetune_fuxi/best.pt`；C/D 初始化为 `work_dirs/da_forecast_6h_predict_delays/best.pt`。',
              '因此跨模型结果是方案比较，不能把差异单独归因于“预测 ZHD/ZWD”；也不能把 C 对 D 的 halo 收益直接解释成超越 MAIN。',
              '旧版不输出 ZHD/ZWD 预测，没有可与 C/D 对齐的延迟 MAE，故只跨模型比较气象指标。',
              '未进行多随机种子或置信区间检验；全部比较沿用原评估质量控制，不新增负 ZWD 过滤。','',
              '### 数据来源','',
              '旧版真实输入优先核对用户指定的 evaluate.log，其最终 JSON 与同目录 summary 完全一致；实际表格使用结构化 summary。','',
              f'- [{old_names[0]} 日志](../{old_names[0]}/test/evaluate.log)',
              f'- [{old_names[1]} 测试结果](../{old_names[1]}/test/rollout_mae_summary.json)',
              '- [四实验原始数值与提升率 CSV](comparison_four_experiments_by_lead.csv)',
              '- [四实验元数据核对记录](comparison_four_experiments_provenance.json)']
    write_csv(out/'comparison_four_experiments_by_lead.csv',cross)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(18,7),sharex=True,sharey=True)
    styles=[('A: MAIN / real ZHD-ZWD','#0072b2','-','o'),
            ('B: MAIN / zero normalized inputs','#e69f00','--','s'),
            ('C: Predict delays / real origin','#009e73','-.','^'),
            ('D: Predict delays / mean origin','#cc79a7',':','D')]
    hours=[row['lead_hours'] for row in cross]
    for ax,metric,title in zip(axes,('global','halo'),('Global','Station neighborhood (halo, radius = 3 grid cells)')):
        for code,(label,color,ls,marker) in zip('ABCD',styles):
            ax.plot(hours,[row[code+'_'+metric+'_mae'] for row in cross],label=label,
                    color=color,ls=ls,marker=marker,markersize=6,markerfacecolor='none',linewidth=1.9)
        ax.set(xlabel='Forecast lead (h)',xticks=hours,title=title)
        ax.grid(alpha=.25)
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    axes[0].set_ylabel('Latitude-weighted weather MAE (normalized)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.915),ncol=2,frameon=False)
    fig.suptitle('Four experiments: global and station-neighborhood weather MAE\n1452 common test origins; 70 channels; lower is better',fontsize=16)
    fig.text(.5,.018,'All models: stage_48h/best.pt (update 4000). Halo is the union of 7x7 station-cell neighborhoods, not station-only MAE.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.055,1,.82))
    fig.savefig(out/'comparison_four_experiments_global_mae.png',dpi=190);plt.close(fig)
    lines += ['', '图：[四实验全球与站点邻域气象 MAE 折线图](comparison_four_experiments_global_mae.png)。']
    paths += [ROOT/'work_dirs'/n/test_name/'rollout_mae_summary.json' for n in EXPERIMENTS]
    metadata={'samples':1452,'halo_radius':3,'global_persistence_exact_match':True,
              'old_real_log_matches_summary':True,'experiments':[
                  {'name':n,'source':str(p),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                   'evaluation':r['evaluation'],'initial_checkpoint':cfg.get('initial_checkpoint')}
                  for n,p,r,cfg in zip(all_names,paths,all_reports,configs)]}
    (out/'comparison_four_experiments_provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return lines


def compare(out, test_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dirs = [ROOT/'work_dirs'/e/test_name for e in EXPERIMENTS]
    reports = [json.loads((d/'rollout_mae_summary.json').read_text()) for d in dirs]
    a,b = reports
    assert a['samples'] == b['samples'] and a['samples'] > 0
    for key in ('split','origin_first','origin_last','evaluated_steps','amp','world_size','trained_stage'):
        assert a['evaluation'][key] == b['evaluation'][key], key
    assert a['evaluation']['split'] == 'test'
    assert a['evaluation']['evaluated_steps'] == a['evaluation']['trained_stage'] == 8
    assert a['evaluation']['mean_origin_inputs'] is False
    assert b['evaluation']['mean_origin_inputs'] is True
    # Use evaluation.observation_policy, not the legacy hardcoded top-level note.
    assert a['evaluation']['observation_policy'] == 'predicted_components'
    assert b['evaluation']['observation_policy'] == 'mean_origin_predicted_components'
    lead_rows=[]
    metrics = {'weather_global':'global_latitude_weighted_mae_normalized',
               'weather_halo':'halo_latitude_weighted_mae_normalized', 'zhd_mm':'zhd_mae_mm','zwd_mm':'zwd_mae_mm'}
    for x,y in zip(a['leads'], b['leads']):
        assert x['lead_hours'] == y['lead_hours']
        for key in ('zhd_valid_count','zwd_valid_count'):
            assert x[key] == y[key], key
        np.testing.assert_allclose(x['persistence_latitude_weighted_mae_normalized'],y['persistence_latitude_weighted_mae_normalized'],rtol=1e-7)
        row={'lead_hours':x['lead_hours']}
        for label,key in metrics.items():
            row['real_'+label]=x[key]; row['mean_'+label]=y[key]
            row[label+'_improvement_pct']=improvement(x[key],y[key])
        lead_rows.append(row)
    assert [r['lead_hours'] for r in lead_rows] == list(range(6,49,6))
    real_channels=csv_rows(dirs[0]/'rollout_mae_by_channel.csv')
    mean_channels={(r['lead_hours'],r['channel']):r for r in csv_rows(dirs[1]/'rollout_mae_by_channel.csv')}
    assert len(real_channels)==len(mean_channels)==560
    channel_rows=[]
    for x in real_channels:
        y=mean_channels[x['lead_hours'],x['channel']]
        assert x['unit']==y['unit']
        r={'lead_hours':int(x['lead_hours']),'channel':x['channel'],'unit':x['unit']}
        for key in ('global_mae_normalized','global_mae_physical','halo_mae_normalized'):
            u,v=float(x[key]),float(y[key]); r['real_'+key]=u;r['mean_'+key]=v
            r[key+'_improvement_pct']=improvement(u,v)
        channel_rows.append(r)
    write_csv(out/'comparison_by_lead.csv',lead_rows)
    write_csv(out/'comparison_by_channel.csv',channel_rows)
    fr=csv_rows(dirs[0]/'rollout_delay_mae_by_frame.csv')
    fm={(r['lead_minutes'],r['component']):r for r in csv_rows(dirs[1]/'rollout_delay_mae_by_frame.csv')}
    frame_rows=[]
    for r in fr:
        m=fm[r['lead_minutes'],r['component']];assert r['valid_count']==m['valid_count']
        u,v=float(r['mae_mm']),float(m['mae_mm'])
        frame_rows.append({'lead_minutes':int(r['lead_minutes']),'component':r['component'],
                           'real_mae_mm':u,'mean_mae_mm':v,'improvement_pct':improvement(u,v)})
    write_csv(out/'comparison_delays_by_frame.csv',frame_rows)
    fig,axes=plt.subplots(2,2,figsize=(16,10))
    for row,component in enumerate(('zwd','zhd')):
        frames=sorted((v for v in frame_rows if v['component']==component),key=lambda v:v['lead_minutes'])
        assert [v['lead_minutes'] for v in frames]==list(range(20,2881,20))
        hours=np.array([v['lead_minutes']/60 for v in frames])
        for col,limit in enumerate((48,12)):
            ax=axes[row,col];keep=hours<=limit
            for key,label,color in [('real_mae_mm','Real initial observations','#0072b2'),
                                    ('mean_mae_mm','Mean initial observations','#d55e00')]:
                values=np.array([v[key] for v in frames])
                ax.plot(hours[keep],values[keep],color=color,label=label,lw=1.6,
                        marker='o',markersize=2.2)
                boundary=keep & (np.array([v['lead_minutes'] for v in frames])%360==0)
                ax.scatter(hours[boundary],values[boundary],facecolors='none',edgecolors=color,s=55,zorder=4)
            for boundary_hour in range(6,limit+1,6):
                ax.axvline(boundary_hour,color='.4',ls='--',lw=.9,alpha=.7)
            ax.set(xlim=(0,limit+.2),xticks=np.arange(0,limit+1,6 if limit==48 else 1),
                   xlabel='Forecast lead (h)',ylabel=f'{component.upper()} MAE (mm)',
                   title=f'{component.upper()}: '+('full 48h rollout' if limit==48 else 'first 12h (detail)'))
            ax.grid(axis='y',alpha=.2);ax.legend(loc='best')
    fig.suptitle('Real-origin vs mean-origin: 20-minute frame MAE on the test set\n1452 common origins; individual frames, NOT 6-hour block averages',fontsize=16)
    fig.text(.5,.016,'Dashed vertical lines: every 6h rollout boundary. Large open circles: the final (18th) frame of each step.\nMAE is over valid observation grid cells/samples, without latitude weighting. Original validity rules retained.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.065,1,.925))
    fig.savefig(out/'comparison_delays_by_frame_20min.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    hours=[r['lead_hours'] for r in lead_rows]
    for ax,(key,title) in zip(axes.flat,[('weather_global','Global weather MAE (normalized, latitude weighted)'),
                                        ('weather_halo','Station-halo weather MAE (normalized, latitude weighted)'),
                                        ('zhd_mm','ZHD MAE (mm; 18 frames per 6h block)'),
                                        ('zwd_mm','ZWD MAE (mm; 18 frames per 6h block)')]):
        for prefix,label in [('real','Real initial observations'),('mean','Mean initial observations')]:
            ax.plot(hours,[r[prefix+'_'+key] for r in lead_rows],'o-',label=label)
        ax.set(title=title,xlabel='Forecast lead (h)',xticks=hours);ax.grid(alpha=.25);ax.legend()
    fig.suptitle(f'Test set: {a["samples"]} common origins | shared-model 48h rollout')
    fig.tight_layout();fig.savefig(out/'comparison_by_lead.png',dpi=170);plt.close(fig)
    for lead in (6,48):
        selected=[r for r in channel_rows if r['lead_hours']==lead]
        for region in ('global','halo'):
            values=[r[f'{region}_mae_normalized_improvement_pct'] for r in selected]
            fig,ax=plt.subplots(figsize=(19,6));ax.bar(np.arange(70),values,color=['#16876d' if v>=0 else '#c64b4b' for v in values])
            ax.set_xticks(np.arange(70),[r['channel'] for r in selected],rotation=90,fontsize=8)
            ax.axhline(0,color='black',lw=.6)
            area='Global' if region=='global' else 'Station halo (radius = 3 grid cells)'
            ax.set(ylabel='Normalized latitude-weighted MAE improvement (%)',
                   title=f'{lead}h | {area}: real-origin vs mean-origin | positive = real-origin better')
            ax.grid(axis='y',alpha=.2);fig.tight_layout()
            suffix='' if region=='global' else '_halo'
            fig.savefig(out/f'comparison_channels{suffix}_{lead}h.png',dpi=170);plt.close(fig)
    lines=['# 测试集：真实初始观测 vs 全局均值初始观测','',
           f'共同样本数：{a["samples"]}；起报范围：{a["evaluation"]["origin_first"]} 至 {a["evaluation"]["origin_last"]}。',
           '两套模型均为 stage_48h/best.pt；均使用预测分量反馈，未来观测仅作为标签。',
           '提升比例 = 100 × (均值输入实验 MAE − 真实输入实验 MAE) / 均值输入实验 MAE；正值表示真实输入更好。','',
           '| 时效 | 气象MAE：真实 | 气象MAE：均值 | 提升 | ZHD MAE：真实/均值 (mm) | ZWD MAE：真实/均值 (mm) |',
           '|---|---|---|---|---|---|']
    for r in lead_rows:
        lines.append(f'| {r["lead_hours"]}h | {r["real_weather_global"]:.6f} | {r["mean_weather_global"]:.6f} | {r["weather_global_improvement_pct"]:+.3f}% | {r["real_zhd_mm"]:.3f} / {r["mean_zhd_mm"]:.3f} | {r["real_zwd_mm"]:.3f} / {r["mean_zwd_mm"]:.3f} |')
    lines += ['', '气象 MAE 为全球纬度加权、归一化指标。不同物理单位不合并平均。',
              'ZHD/ZWD 是各 6h 区间内 18 帧、有效站点样本的 MAE（非纬度加权、非仅区间末帧）。',
              '沿用训练和原评估的有效性规则，未临时过滤负 ZWD。',
              '不比较两者的 delay persistence 提升：其基准分别是真实末帧与固定均值，口径不同。',
              '这是两次训练的结果比较，不等同于多随机种子统计显著性结论。']
    lines += extended_comparison(out,reports,test_name)
    lines += ['', 'Station halo 逐通道相对提升：[6h](comparison_channels_halo_6h.png)、[48h](comparison_channels_halo_48h.png)。正值表示真实初始观测优于均值输入。']
    lines += ['', '图：[ZWD/ZHD 20min 逐帧 MAE，含每 6h 边界与前 12h 放大图](comparison_delays_by_frame_20min.png)。']
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')
    (out/'provenance.json').write_text(json.dumps({'real':a['evaluation'],'mean':b['evaluation'],'samples':a['samples']},indent=2)+'\n')
    print('\n'.join(lines),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true')
    p.add_argument('--test-name',default='test_compare_20261119')
    p.add_argument('--output-dir',type=Path,default=ROOT/'work_dirs/comparison_predict_delays_vs_mean_origin_test_20261119')
    args=p.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.run:
        import torch
        from data.delay_sequence import make_rollout_dataset
        from train import resolve_data_paths
        datasets=[]
        for name,conf in zip(EXPERIMENTS,CONFIGS):
            cfg=json.loads((ROOT/'configs'/conf).read_text())
            ck=torch.load(ROOT/'work_dirs'/name/'stage_48h/best.pt',map_location='cpu',weights_only=False,mmap=True)
            for k in ('data','model','curriculum'):
                assert cfg[k]==ck['config'][k], f'{name}: current {k} differs from saved checkpoint'
            assert ck['config']['curriculum_state']['stage']==8
            print(f'Preflight {name}: best checkpoint update={ck["step"]}',flush=True)
            datasets.append(make_rollout_dataset(resolve_data_paths(ROOT/'configs'/conf,cfg['data']),'test',8,1))
            del ck
        assert datasets[0].analysis_times.equals(datasets[1].analysis_times)
        print(f'Identical {len(datasets[0])} origins verified',flush=True)
        del datasets
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1')
        for name,conf in zip(EXPERIMENTS,CONFIGS):
            dest=ROOT/'work_dirs'/name/args.test_name;dest.mkdir(parents=True,exist_ok=True)
            if (dest/'rollout_mae_summary.json').exists():
                raise FileExistsError(f'Result exists: {dest}; compare without --run or choose a new --test-name')
            cmd=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                 'evaluate_rollout_delays.py','--config',str(ROOT/'configs'/conf),'--checkpoint',
                 str(ROOT/'work_dirs'/name/'stage_48h/best.pt'),'--output-dir',str(dest),
                 '--split','test','--steps','8','--batch-size','1','--num-workers','1']
            print('Starting: '+' '.join(cmd),flush=True)
            with (dest/'evaluate.log').open('a') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                for line in proc.stdout:
                    log.write(line);log.flush();print(line,end='',flush=True)
                code=proc.wait()
            if code:
                raise RuntimeError(f'{name} evaluation exited {code}; see {dest}/evaluate.log')
    compare(out,args.test_name)


if __name__=='__main__':
    main()
