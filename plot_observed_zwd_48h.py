#!/usr/bin/env python3
"""Observed ZWD at one fixed origin: station means and 4 reproducible stations.

No model inference. Reads the selected raw NGL station cells in temporal blocks;
uses paired finite ZTD/ZWD validity as in component-label preparation. This is
an observation diagnostic, not the 1452-origin lead-aggregated test statistic.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import zarr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data.da_dataset import _decode_time
from data.delay_components import normalization

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--origin',default='2025-03-27 06:00:00')
    p.add_argument('--interval-minutes',type=int,default=20)
    p.add_argument('--seed',type=int,default=2000)
    p.add_argument('--output-dir',type=Path,default=ROOT/'work_dirs/comparison_predict_delays_vs_mean_origin_test_20261119')
    args=p.parse_args()
    if args.interval_minutes<=0 or args.interval_minutes%5 or 2880%args.interval_minutes:
        p.error('interval must be a positive multiple of 5 and divide 2880')
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    origin=pd.Timestamp(args.origin)
    wanted=pd.date_range(origin,origin+pd.Timedelta(hours=48),freq=f'{args.interval_minutes}min')
    source=ROOT/'datasets/global_ngl_0p25_5min_2023_2025.zarr'
    group=zarr.open_group(str(source),mode='r')
    times=_decode_time(group,'time');indices=times.get_indexer(wanted)
    assert np.all(indices>=0) and np.all(np.diff(indices)==args.interval_minutes//5)
    occupied=~np.asarray(group['mask'][:],bool)
    rr,cc=np.nonzero(occupied)
    names=np.asarray(group['station'][:])[occupied].astype(str)
    lat=np.asarray(group['station_lat'][:])[occupied]
    lon=(np.asarray(group['station_lon'][:])[occupied]+180)%360-180
    mean,std=normalization(group,'zwd')
    observations=np.full((len(wanted),len(rr)),np.nan,dtype=np.float64)
    # Limit peak memory: do not materialize a dense 145 x global-grid array.
    for start in range(0,len(wanted),18):
        stop=min(start+18,len(wanted));idx=indices[start:stop]
        sl=slice(int(idx[0]),int(idx[-1])+1,args.interval_minutes//5)
        wet=np.asarray(group['zwd'][sl],np.float32)[:,rr,cc]
        total=np.asarray(group['ztd'][sl],np.float32)[:,rr,cc]
        valid=np.isfinite(wet)&np.isfinite(total)
        observations[start:stop]=np.where(valid,wet.astype(np.float64)*std+mean,np.nan)
        print(f'Observation frames read: {stop}/{len(wanted)}',flush=True)
    finite=np.isfinite(observations)
    complete=finite.all(0)
    assert complete.sum()>=4 and finite.any(1).all()
    selected=np.random.default_rng(args.seed).choice(np.flatnonzero(complete),4,replace=False)
    dynamic=np.nanmean(observations,axis=1)
    fixed=observations[:,complete].mean(1)
    counts=finite.sum(1)
    hours=(wanted-origin).total_seconds().to_numpy()/3600
    boundaries=np.arange(6,49,6)
    def decorate(ax):
        for h in boundaries: ax.axvline(h,color='.45',ls='--',lw=.8,alpha=.6)
        ax.set(xlim=(0,48),xticks=np.arange(0,49,6),xlabel='Hours since origin T')
        ax.grid(axis='y',alpha=.2)
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True,gridspec_kw={'height_ratios':[3,1]})
    axes[0].plot(hours,dynamic,label='All valid stations at each time',color='#0072b2',lw=1.8)
    axes[0].plot(hours,fixed,label=f'Fixed complete-history subset ({int(complete.sum())} stations)',color='#d55e00',lw=1.5)
    axes[0].set_ylabel('Observed ZWD mean (mm)');axes[0].legend()
    axes[1].plot(hours,counts,color='#009e73');axes[1].set_ylabel('Valid station count')
    for ax in axes: decorate(ax)
    fig.suptitle(f'Observed ZWD: global station mean over one 48h window\n{wanted[0]} to {wanted[-1]} UTC | {args.interval_minutes}-minute sampling',fontsize=15)
    fig.text(.5,.012,'Unweighted station means, NOT global area means. Paired finite ZTD/ZWD mask; negative values retained.\nDashed lines: every 6h. This is one actual time window, not a mean over all test forecast origins.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.065,1,.92));fig.savefig(out/'observed_zwd_48h_station_mean.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(15,9))
    records=[]
    for ax,k in zip(axes.flat,selected):
        ax.plot(hours,observations[:,k],color='#0072b2',marker='.',markersize=3,lw=1.5)
        ax.set(title=f'{names[k]} | lat={lat[k]:.3f}, lon={lon[k]:.3f}',ylabel='Observed ZWD (mm)')
        decorate(ax)
        records.append({'station':names[k],'latitude':float(lat[k]),'longitude':float(lon[k]),
                        'source_row':int(rr[k]),'source_col':int(cc[k]),
                        'min_mm':float(observations[:,k].min()),'max_mm':float(observations[:,k].max())})
    fig.suptitle(f'Observed ZWD: 4 random stations (seed {args.seed})\n{wanted[0]} to {wanted[-1]} UTC | {args.interval_minutes}-minute sampling',fontsize=15)
    fig.text(.5,.012,'Sampled uniformly from stations valid at every plotted time; no selection by smoothness or error.\nDashed lines: every 6h. Independent y-axis ranges; raw physical ZWD, not model predictions.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.065,1,.92));fig.savefig(out/'observed_zwd_48h_four_stations.png',dpi=180);plt.close(fig)
    data=[]
    for j,t in enumerate(wanted):
        row={'time_utc':str(t),'lead_hours':float(hours[j]),'mean_all_valid_stations_mm':float(dynamic[j]),
             'mean_fixed_stations_mm':float(fixed[j]),'valid_station_count':int(counts[j])}
        row.update({names[k]+'_zwd_mm':float(observations[j,k]) for k in selected});data.append(row)
    with (out/'observed_zwd_48h.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    np.savez_compressed(out/'observed_zwd_48h_station_data.npz',time_ns=wanted.asi8,
                        station=names,lat=lat,lon=lon,zwd_mm=observations,complete=complete)
    report={'source':str(source),'origin_utc':str(origin),'end_utc':str(wanted[-1]),
            'interval_minutes':args.interval_minutes,'frames':len(wanted),'seed':args.seed,
            'complete_stations':int(complete.sum()),'valid_stations_min':int(counts.min()),
            'valid_stations_max':int(counts.max()),'selected_stations':records,
            'mean_all_valid_range_mm':[float(dynamic.min()),float(dynamic.max())],
            'mean_fixed_range_mm':[float(fixed.min()),float(fixed.max())],
            'note':'Single actual window; unweighted raw selected NGL station means (before ERA regridding). Paired finite ZTD/ZWD; no negative-value filtering.'}
    (out/'observed_zwd_48h_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
