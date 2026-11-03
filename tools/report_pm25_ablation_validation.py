"""Traceable descriptive analysis of all frozen PM2.5 runs; no tuning."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr

from run_pm25_ablation_validation import SEEDS, VARIANTS, LABELS, digest

FIGURE_SKILL = Path(r'C:\Users\LX\.codex\skills\nature-figure')
sys.path.insert(0, str(FIGURE_SKILL/'scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

DISPLAY = {'full': 'Guided UNO', 'no_cross_attention': 'No main cross-attention',
    'no_observation_residual': 'No local obs residual', 'no_spectral_loss': 'No spectral loss',
    'no_peak_loss': 'No peak loss', 'no_reliability': 'Equal reliability',
    'guided_fno_backbone': 'Guided FNO backbone', 'original_fno': 'Original FNO',
    'original_uno': 'Multi-scale UNO', 'bilinear': 'Bilinear', 'bias_offset': 'Offset correction',
    'linear': 'Linear correction', 'linear_idw': 'Linear + station IDW', 'qp': 'QP',
    'linear_qp': 'Linear background + QP'}
COLORS = {'full':'#266A8F','linear':'#BC7535','linear_idw':'#47967C',
    'qp':'#858A91','bilinear':'#B5BCC3','original_fno':'#9278A6','original_uno':'#6B9EB4'}


def label_fields(output):
    """Clarify ablation labels in NetCDF attributes without changing values."""
    import h5netcdf
    records=[]
    for method in VARIANTS:
        for seed in SEEDS:
            label=f'{method}_seed{seed}'
            path=output/'fields'/f'{label}.nc'
            controls=json.loads((output/'logs'/f'{label}_training.json').read_text(encoding='utf-8'))['ablation_controls']
            before=digest(path)
            with h5netcdf.File(path,'a') as ds:
                data_hash=sha256(ds.variables['pm25_rh35_gcc'][:].tobytes()).hexdigest()
                ds.attrs['algorithm']=f'{method}: official {controls["backbone"].upper()} backbone; configured observation encoder'
                ds.attrs['experiment_method']=method
                ds.attrs['experiment_seed']=seed
                ds.attrs['ablation_controls']=json.dumps(controls,sort_keys=True)
                ds.attrs['uses_target_period_training_station_observations']='true'
                ds.attrs['configuration_file']=f'configs/{label}.json'
            with h5netcdf.File(path,'r') as ds:
                assert data_hash==sha256(ds.variables['pm25_rh35_gcc'][:].tobytes()).hexdigest()
            records.append({'file':path.name,'before_file_sha256':before,'after_file_sha256':digest(path),
                'data_values_sha256':data_hash,'values_unchanged':True,
                'operation':'clarify ablation metadata only; weights, coordinates and predictions unchanged'})
    (output/'field_metadata_audit.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')


def number(v, digits=2):
    return '—' if pd.isna(v) else f'{v:.{digits}f}'


def save(fig, output, name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=output/'figures'/f'{name}.alignment.json',strict=True)
    fig.savefig(output/'figures'/f'{name}.pdf')
    fig.savefig(output/'figures'/f'{name}.svg')
    fig.savefig(output/'figures'/f'{name}.png',dpi=300)
    fig.savefig(output/'figures'/f'{name}.tiff',dpi=300)
    plt.close(fig)


def figures(output, runs, summary, deltas):
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':9,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
        'svg.fonttype':'none','legend.frameon':False})
    contract = {'backend':'python','archetype':'quantitative grids and sparse-station error maps',
        'purpose':'distinguish module effects, initialization variance and bias removal from spatial reconstruction',
        'uncertainty':'five seed observations and sample SD, not independent station-hour confidence intervals',
        'seed_selection':'all seeds; fixed seed17 only where explicitly labelled, never best-seed selection',
        'spatial_map':'sparse test stations only, no filled ground-truth surface',
        'source_data':['test_runs.csv','paired_ablation_deltas.csv','seed_summary.csv','predictions/*.csv','logs/*_curve.csv'],
        'exports':['PNG/TIFF 300dpi','editable PDF','editable SVG'],
        'journal_target':'internal research report, not a claimed Nature submission'}
    (output/'figure_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2),encoding='utf-8')
    neural = runs[runs.seed!=-1]
    fig, ax = plt.subplots(figsize=(7.2,5.2))
    methods = list(VARIANTS)
    for y,method in enumerate(methods):
        values = neural[neural.method==method].rmse.to_numpy()
        ax.errorbar(values.mean(),y,xerr=values.std(ddof=1),fmt='o',color='#266A8F',capsize=3)
        ax.scatter(values,np.full(5,y)+np.linspace(-.12,.12,5),color='#9BB9CA',s=18,zorder=3)
    ax.set(yticks=range(len(methods)),yticklabels=[DISPLAY[m] for m in methods],
        xlabel='Test-station RMSE (ug/m3)',title='All seeds: mean +/- sample SD (n = 5)')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.subplots_adjust(left=.31,right=.97,bottom=.13,top=.9)
    save(fig,output,'seed_rmse')

    fig, ax = plt.subplots(figsize=(7.2,4.6))
    effects = deltas[deltas.metric=='rmse']
    for y,row in enumerate(effects.itertuples()):
        values = np.array(list(json.loads(row.deltas_by_seed).values()))
        ax.scatter(values,np.full(5,y)+np.linspace(-.1,.1,5),color='#ABC7D5',s=22)
        ax.errorbar(row.mean_delta,y,xerr=row.sample_sd,fmt='o',color='#266A8F',capsize=3)
    ax.axvline(0,color='#4D5359',linestyle='--',lw=.8)
    ax.set(yticks=range(len(effects)),yticklabels=[DISPLAY[m] for m in effects.method],
        xlabel='Ablation RMSE - full RMSE (ug/m3)',title='Paired module effects; positive favors full model')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.subplots_adjust(left=.31,right=.97,bottom=.15,top=.9)
    save(fig,output,'paired_ablation')

    fig, ax = plt.subplots(figsize=(7.2,4.3))
    for method in ('full','original_fno','original_uno'):
        curves=[]
        for seed in SEEDS:
            curve = pd.read_csv(output/'logs'/f'{method}_seed{seed}_curve.csv')
            curves.append(curve.set_index('epoch').validation_rmse)
            ax.plot(curve.epoch,curve.validation_rmse,color=COLORS[method],alpha=.18,lw=.7)
        combined = pd.concat(curves,axis=1)
        ax.plot(combined.index,combined.mean(axis=1),color=COLORS[method],label=DISPLAY[method],lw=1.8)
    ax.set(xlabel='Epoch',ylabel='Validation-station RMSE (ug/m3)',title='Validation curves: all seed traces and available-run mean')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.subplots_adjust(left=.11,right=.97,bottom=.15,top=.9)
    save(fig,output,'convergence')

    # Validation curves for each one-factor variant, keeping the five traces.
    for method in list(VARIANTS)[:7]:
        fig,ax=plt.subplots(figsize=(6.2,3.8))
        for seed in SEEDS:
            curve=pd.read_csv(output/'logs'/f'{method}_seed{seed}_curve.csv')
            ax.plot(curve.epoch,curve.validation_rmse,label=f'Seed {seed}',lw=1)
        ax.set(xlabel='Epoch',ylabel='Validation RMSE (ug/m3)',title=DISPLAY[method])
        ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.subplots_adjust(left=.13,right=.97,bottom=.15,top=.9)
        save(fig,output,f'curve_{method}')

    selected=['bilinear','bias_offset','linear','linear_idw','qp','linear_qp','original_fno','original_uno','full']
    fig,ax=plt.subplots(figsize=(7.2,5))
    decomposition=[]
    for method in selected:
        sub=runs[runs.method==method]
        decomposition.append({'method':method,'bias_squared':float((sub.bias**2).mean()),
            'centered_mse':float((sub.centered_rmse**2).mean())})
    parts=pd.DataFrame(decomposition)
    parts.to_csv(output/'error_decomposition.csv',index=False)
    ax.barh(range(len(parts)),parts.bias_squared,color='#A9BDC8',label='Squared mean bias')
    ax.barh(range(len(parts)),parts.centered_mse,left=parts.bias_squared,color='#3F7996',label='Centered error variance')
    ax.set(yticks=range(len(parts)),yticklabels=[DISPLAY[m] for m in selected],
        xlabel='MSE (ug/m3)^2',title='MSE = bias squared + centered error variance')
    ax.invert_yaxis();ax.legend(loc='lower right',fontsize=8)
    fig.subplots_adjust(left=.29,right=.97,bottom=.13,top=.9)
    save(fig,output,'bias_decomposition')

    fig,ax=plt.subplots(figsize=(7.2,5))
    for y,method in enumerate(selected):
        values=runs[runs.method==method].spatial_anomaly_rmse.to_numpy()
        ax.scatter(values,np.full(len(values),y),s=25,color=COLORS.get(method,'#858A91'))
        if len(values)>1:
            ax.errorbar(values.mean(),y,xerr=values.std(ddof=1),fmt='o',capsize=3,color='#266A8F')
    ax.set(yticks=range(len(selected)),yticklabels=[DISPLAY[m] for m in selected],
        xlabel='Within-hour station spatial-anomaly RMSE (ug/m3)',
        title='Spatial variation at independent stations (not full-field truth)')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.subplots_adjust(left=.29,right=.97,bottom=.13,top=.9)
    save(fig,output,'spatial_structure')

    predictions=[]
    for seed in SEEDS:
        p=pd.read_csv(output/'predictions'/f'full_seed{seed}_test.csv',parse_dates=['time'])
        p['seed']=seed;predictions.append(p)
    pooled=pd.concat(predictions,ignore_index=True)
    site=sorted(pooled.station_id.astype(str).unique())[0]
    station=pooled[pooled.station_id.astype(str)==site]
    mean=station.groupby('time').predicted.mean()
    sd=station.groupby('time').predicted.std(ddof=1)
    observed=station.groupby('time')['observed'].first()
    fig,ax=plt.subplots(figsize=(7.2,3.8))
    ax.plot(observed.index,observed,color='#3A3E43',lw=1.3,label='Observed')
    ax.plot(mean.index,mean,color=COLORS['full'],lw=1.2,label='Guided UNO: seed mean')
    ax.fill_between(mean.index,mean-sd,mean+sd,color=COLORS['full'],alpha=.15,label='Seed SD')
    for method in ('linear','linear_idw'):
        p=pd.read_csv(output/'predictions'/f'{method}_seed-1_test.csv',parse_dates=['time'])
        p=p[p.station_id.astype(str)==site].sort_values('time')
        ax.plot(p.time,p.predicted,color=COLORS[method],lw=.9,label=DISPLAY[method])
    ax.set(xlabel='August 2019 (UTC)',ylabel='PM2.5 (ug/m3)',title=f'Fixed first sorted test station: {site}')
    ax.legend(ncol=2,fontsize=7,loc='upper right');ax.grid(alpha=.2)
    fig.subplots_adjust(left=.1,right=.97,bottom=.18,top=.88)
    save(fig,output,'station_timeseries')

    average=pooled.groupby(['station_id','time','latitude','longitude'],as_index=False).agg(
        observed=('observed','first'),predicted=('predicted','mean'))
    station_error=average.assign(error=lambda d:d.predicted-d.observed).groupby('station_id',as_index=False).agg(
        latitude=('latitude','first'),longitude=('longitude','first'),bias=('error','mean'))
    station_error.to_csv(output/'station_error_map_source.csv',index=False)
    fig,ax=plt.subplots(figsize=(6,4.6))
    limit=max(float(abs(station_error.bias).max()),1)
    dots=ax.scatter(station_error.longitude,station_error.latitude,c=station_error.bias,cmap='RdBu_r',
        vmin=-limit,vmax=limit,s=75,edgecolors='white',lw=.6)
    for row in station_error.itertuples():
        ax.annotate(str(row.station_id),(row.longitude,row.latitude),xytext=(4,5),textcoords='offset points',fontsize=7)
    fig.colorbar(dots,ax=ax,label='Mean prediction - observation (ug/m3)',fraction=.046,pad=.04)
    ax.set(xlabel='Longitude (approx. WGS84)',ylabel='Latitude (approx. WGS84)',title='Guided UNO: test-station mean error, averaged over 5 seeds')
    ax.margins(.2);ax.grid(alpha=.2)
    fig.subplots_adjust(left=.12,right=.89,bottom=.14,top=.9)
    save(fig,output,'station_error_map')

    # Fixed timestamp, all seeds, common color scale; these are predictions,
    # never presented as high-resolution ground truth.
    stamp = pd.Timestamp('2019-08-12 12:30')
    maps = {}
    for method in ('linear','linear_idw'):
        with xr.open_dataset(output/'fields'/f'{method}.nc',engine='h5netcdf') as ds:
            maps[method] = ds.pm25_rh35_gcc.sel(time=stamp).load()
    neural_maps=[]
    for seed in SEEDS:
        with xr.open_dataset(output/'fields'/f'full_seed{seed}.nc',engine='h5netcdf') as ds:
            neural_maps.append(ds.pm25_rh35_gcc.sel(time=stamp).load())
    maps['full'] = xr.concat(neural_maps,dim='seed').mean('seed')
    vmax = max(float(a.max()) for a in maps.values())
    for method,field in maps.items():
        fig,ax=plt.subplots(figsize=(6,4.6))
        image=ax.pcolormesh(field.lon,field.lat,field.values,cmap='viridis',vmin=0,vmax=vmax,shading='auto',rasterized=True)
        fig.colorbar(image,ax=ax,label='Predicted PM2.5 (ug/m3)',fraction=.046,pad=.04)
        ax.set(xlabel='Longitude (approx. WGS84)',ylabel='Latitude (approx. WGS84)',
            title=f'{DISPLAY[method]}: 12 Aug 2019 12:30 UTC')
        fig.subplots_adjust(left=.12,right=.89,bottom=.14,top=.9)
        save(fig,output,f'field_{method}')


def report(output,runs,summary,deltas):
    plan=json.loads((output/'experiment_plan.json').read_text(encoding='utf-8'))
    data=json.loads((output/'data_boundary.json').read_text(encoding='utf-8'))
    cross=json.loads((output/'september_data_status.json').read_text(encoding='utf-8'))
    baseline=json.loads((output/'baseline_fit.json').read_text(encoding='utf-8'))
    input_audit=json.loads((output/'independent_input_audit.json').read_text(encoding='utf-8'))
    lines=['# 神经网络消融与泛化验证报告','',
        '## 结论与证据边界','',
        '本轮完成9组神经模型配置×5个固定随机种子，共45次独立训练，以及6个传统/偏差校正基线。全部模型沿用官方FNO/UNO及已有编码器，没有增加复杂网络。所有种子均纳入统计，没有挑选最好的一次。',
        '本轮结果属于**探索性验证**：站点坐标基准仍缺直接来源证明，GEOS-CF RH35 PM2.5与地面监测定义有差异；沿用的测试站此前已在旧报告中评价，不能称为新采集、从未接触过的外部验证队列。',
        f'9月跨月份验证状态：**{cross["status"]}**。本报告不把8月相邻日期推理或微调冒充9月泛化，也不填写未执行的数值。','',
        '## 实验方案与防泄漏','',
        '- 固定40训练站、10验证站、10最终测试站；站点ID和划分文件哈希见`data_boundary.json`、`provenance.json`。',
        '- 训练为2019-08-01 00:30至08-07 23:30 UTC，共168小时；权重直接推理评价为08-08至08-14，共168小时。沿用上一报告的时段，**并非使用8月整月训练**。',
        '- 全部45次训练及未来时段推理场生成完毕，写入`scheme_frozen_before_test.json`后才计算最终测试站指标。验证站仅作Early Stopping及最佳权重选择。本轮没有基于测试集再调参。',
        '- 背景场和6个ERA5通道的标准化仅使用训练时段外生场；训练/验证/测试地面标签不参与这些统计。推理加载Checkpoint标准化参数。训练站值可进入编码器与损失，验证站仅用于模型选择，测试站不进入模型。',
        '- 本实验推理输入包括**目标时段40个训练站同期观测**，属于观测引导融合/重构，不是纯零样本预测。FNO/UNO也使用观测编码。可靠度完整率依据该推理窗口的训练站可用性计算，故是离线融合评估，尚未证明严格在线因果性。',
        '- 原始有限、三源对齐记录统一评价；所有方法的测试站—时刻键完全一致。没有高分辨率全场真值，不报告全场RMSE。',
        f'- 峰值误差是每小时独立站中最大观测与最大预测的绝对差均值，不是整幅栅格峰值真值。高值MAE阈值为训练期训练站90%分位数：{data["high_value_threshold"]:.2f} μg/m³。',
        '- 配置：AdamW，学习率0.002，weight decay=0.0001，Plateau，最多30轮，最少8轮后可早停，patience=5，min_delta=0.01，梯度裁剪1.0；UNO hidden=8、3层下/上采样；FNO hidden=8、2层；5个种子为17/29/43/71/101。',
        '- CPU每次4线程，最多4路独立训练并行。训练秒包含输入准备、训练、验证、函数内Checkpoint保存和结果生成，不包含外部JSON/CSV日志及NetCDF写盘；推理秒包含权重读取、输入准备、前向计算和验证站采样，不包含外部NetCDF写盘。并行资源竞争及依赖首次加载影响耗时，只作描述性比较。初始化种子、数据顺序、阈值和配置均固定。',
        '- 原有训练优化报告、Checkpoint和结果均逐文件哈希核对未修改。本轮初次串行试跑单独保留，未纳入45次正式批量统计。','',
        '执行偏离：FNO状态含额外字典，首次记录程序在4次训练和推理已保存后发生参数元素统计错误。已保留原日志和脚本，只修复记录并加载既有权重恢复这4次结果，没有重新设计或重训。受影响训练时间取原训练报告的函数内runtime_seconds；恢复推理时间单独重新测量。详见`execution_deviation.md`、`runtime_code_versions.json`。','',
        f'独立训练期检查：训练站PM2.5均值{input_audit["training_station_mean"]:.2f} μg/m³，GEOS同记录均值{input_audit["training_geos_mean"]:.2f} μg/m³，偏差{input_audit["training_geos_minus_station_bias"]:.2f} μg/m³。该7日窗口的偏差大于旧报告整月约89.73 μg/m³，因此校正系数必须从本轮训练数据拟合。训练站有效完整率为{min(input_audit["train_completeness"].values()):.1%}—{max(input_audit["train_completeness"].values()):.1%}。','',
        '## 消融开关与单因素控制','',
        '|配置|实际改变|保持内容|','|---|---|---|']
    descriptions={'full':'无','no_cross_attention':'cross_attention=False，仅关闭背景条件查询主MHA',
        'no_observation_residual':'observation_residual_gate=0','no_spectral_loss':'spectral_weight=0',
        'no_peak_loss':'peak_weight=0','no_reliability':'reliability_weighting=False，损失权重与质量先验都置1',
        'guided_fno_backbone':'backbone=fno，仅替换官方骨干'}
    for method in list(VARIANTS)[:7]:
        lines.append(f'|{LABELS[method]}|{descriptions[method]}|其余编码器、损失、数据和训练控制不变|')
    lines += ['',
        '**Attention边界**：局部观测残差内部也调用同一官方MHA。为保持“每次只改变一个模块”，去Cross Attention仅去背景查询主通路，保留局部残差中的Attention。它不是“所有Attention都移除”；若今后研究这一问题，需作为另一个预先声明的联合消融。',
        'FNO骨干对照保持可靠度、局部残差、频谱和峰值损失开启。原有FNO与原有UNO另作主要算法对比，因它们包含其他配置差异，不能用这两项对比单独归因于骨干。两骨干结构深度和参数量也不同，不是等参数对照。','',
        '## 多种子最终测试结果','',
        '以下均为5个固定种子的均值±样本标准差（ddof=1）；单位为μg/m³，相关系数无量纲。标准差仅反映初始化和优化变异，不代表空间/月份抽样置信区间。',
        '|方法|RMSE|MAE|Bias|相关系数|站点峰值误差|','|---|---:|---:|---:|---:|---:|']
    for method in VARIANTS:
        r=summary.loc[method]
        values=[f'{r[f"{m}_mean"]:.{3 if m=="correlation" else 2}f} ± {r[f"{m}_std"]:.{3 if m=="correlation" else 2}f}'
            for m in ['rmse','mae','bias','correlation','peak_error']]
        lines.append('|'+LABELS[method]+'|'+'|'.join(values)+'|')
    lines += ['','|方法|最佳轮次均值±SD|实际训练轮次均值±SD|训练秒均值±SD|RMSE最小—最大|',
        '|---|---:|---:|---:|---:|']
    for method in VARIANTS:
        r=summary.loc[method]
        lines.append(f'|{LABELS[method]}|{r.best_epoch_mean:.1f} ± {r.best_epoch_std:.1f}|'
            f'{r.epochs_completed_mean:.1f} ± {r.epochs_completed_std:.1f}|'
            f'{r.training_seconds_mean:.1f} ± {r.training_seconds_std:.1f}|{r.rmse_min:.2f}—{r.rmse_max:.2f}|')
    lines += ['','最佳轮次定义为验证RMSE选中Checkpoint的轮次；达到上限30轮不意味着证明已经数学收敛。完整逐种子结果、参数状态元素数和推理耗时见`test_runs.csv`。','',
        '### 初始化稳定性','',
        '|方法|RMSE变异系数(SD/均值)|解释范围|','|---|---:|---|']
    stability=[]
    for method in VARIANTS:
        r=summary.loc[method]
        cv=r.rmse_std/r.rmse_mean
        stability.append({'method':method,'rmse_mean':r.rmse_mean,'rmse_sample_sd':r.rmse_std,
            'rmse_coefficient_of_variation':cv,'rmse_min':r.rmse_min,'rmse_max':r.rmse_max})
        lines.append(f'|{LABELS[method]}|{cv:.1%}|固定数据下的优化/初始化波动，不代表跨月份稳定性|')
    pd.DataFrame(stability).to_csv(output/'stability.csv',index=False)
    paired=runs[runs.seed!=-1].pivot(index='seed',columns='method',values='rmse')
    lines += ['',f'完整模型在{int((paired.full<paired.original_fno).sum())}/5个种子上优于原有FNO，'
        f'在{int((paired.full<paired.original_uno).sum())}/5个种子上优于原有UNO；'
        '上述次数与标准差、范围共同报告，不把个别最好种子当作稳定优势。','',
        '## 各模块的实际贡献','',
        '配对差值=去模块模型−完整模型，同一随机种子作配对；正值表示完整模型更好。这里只报告描述性效应，不做5次重复不足以支持的“显著创新”宣称。',
        '|去掉/替换内容|RMSE配对差值均值±SD|完整模型较好次数|解释|','|---|---:|---:|---|']
    for row in deltas[deltas.metric=='rmse'].itertuples():
        interpretation='去除此模块平均更好；当前配置未显示正贡献' if row.mean_delta<0 else '保留模块平均更好；需结合种子波动'
        lines.append(f'|{LABELS[row.method]}|{row.mean_delta:.2f} ± {row.sample_sd:.2f}|{row.count_full_better}/5|{interpretation}|')
    lines += ['',
        '注意：此处的贡献限定在当前地区、7日训练窗口、网络规模和30轮上限。模块变化也会影响优化轨迹，有限轮次消融不能直接解释为结构的理论优劣。可靠度数据没有仪器QA或观测误差方差信息；可识别贡献主要来自有效记录完整率，不能验证真正的测量可靠度建模。','',
        '## 传统方法与偏差校正：网络学到了什么','',
        f'训练数据拟合线性系数：预测=max(0, {baseline["linear_slope"]:.6f}×GEOS双线性值 + {baseline["linear_intercept"]:.6f})；常数偏差校正为max(0, 背景 + {baseline["constant_offset"]:.6f})。系数仅来自训练期40训练站，未使用目标期验证/测试标签。',
        '线性+IDW在该线性校正后，用目标时段40个训练站的同期残差进行固定反距离平方插值，没有使用验证/测试站残差，也没有以测试指标选择插值参数。额外加入“校正背景+原有QP”诊断粗场约束是否在维持错误背景。',
        '|方法|RMSE|MAE|Bias|相关系数|峰值误差|训练期拟合/同期观测|',
        '|---|---:|---:|---:|---:|---:|---|']
    conditions={'bilinear':'无拟合/无同期观测','bias_offset':'训练站常数拟合/无同期观测',
        'linear':'训练站线性拟合/无同期观测','linear_idw':'训练站线性拟合/40训练站同期残差',
        'qp':'无历史拟合/40训练站同期观测','linear_qp':'训练站线性拟合/40训练站同期观测'}
    for method in conditions:
        r=runs[runs.method==method].iloc[0]
        values=[number(r[m],3 if m=='correlation' else 2) for m in ['rmse','mae','bias','correlation','peak_error']]
        lines.append('|'+LABELS[method]+'|'+'|'.join(values)+'|'+conditions[method]+'|')
    lines += ['',
        '公平性边界：同一站点划分、相同评价记录、网格、单位和算子；但各方法输入信息不同。神经模型有ERA5，QP和IDW没有；双线性没有训练站校正。最接近神经融合观测条件的简单基线是线性+同期训练站残差。对“ERA5增益”本轮未做专门消融，不能把所有增益归因于网络。','',
        '|方法|去整体偏差后RMSE|独立站每小时空间异常RMSE|空间异常相关|高值MAE|',
        '|---|---:|---:|---:|---:|']
    for method in ['bilinear','linear','linear_idw','qp','linear_qp','original_fno','original_uno','full']:
        s=runs[runs.method==method]
        values=[]
        for metric in ['centered_rmse','spatial_anomaly_rmse','spatial_anomaly_correlation','high_value_mae']:
            values.append(number(s[metric].mean(),3 if metric.endswith('correlation') else 2))
        lines.append('|'+LABELS[method]+'|'+'|'.join(values)+'|')
    full=summary.loc['full'];linear=runs[runs.method=='linear'].iloc[0];idw=runs[runs.method=='linear_idw'].iloc[0]
    raw=runs[runs.method=='bilinear'].iloc[0]
    full_runs=runs[runs.method=='full']
    mse_reduction=raw.rmse**2-float((full_runs.rmse**2).mean())
    bias_reduction=raw.bias**2-float((full_runs.bias**2).mean())
    fraction=bias_reduction/mse_reduction if mse_reduction>0 else None
    lines += ['',
        f'完整模型RMSE均值为{full.rmse_mean:.2f}，训练期线性校正为{linear.rmse:.2f}，线性+IDW为{idw.rmse:.2f}。',
        f'完整模型空间异常RMSE为{full.spatial_anomaly_rmse_mean:.2f}，线性+IDW为{idw.spatial_anomaly_rmse:.2f}；站点峰值误差分别为{full.peak_error_mean:.2f}与{idw.peak_error:.2f}。',
        f'相对原始双线性，完整模型平均MSE下降{mse_reduction:.2f} (μg/m³)²，其中Bias平方项下降{bias_reduction:.2f} (μg/m³)²。'
        +(f'偏差项占MSE下降的{fraction:.1%}。' if fraction is not None else '整体MSE没有下降，不能解释为误差改善。'),
        '该比例是可观测误差的代数分解，不是内部网络机制证明；若超过100%，表示去Bias误差项反而变差。判断应同时依据空间异常和峰值。若简单校正达到相近或更低误差，则现有结果不足以证明网络超出偏差校正的额外收益；只有空间异常和峰值指标也一致胜出，才支持独立站尺度上的额外结构重构能力。不能推及没有真值的站点之间整幅场。','',
        '## 收敛、误差图与可视化','',
        '![全部种子RMSE](figures/seed_rmse.png)','',
        '图1：九组配置全部五种子散点及均值±样本SD，未筛选种子。','',
        '![配对贡献](figures/paired_ablation.png)','',
        '图2：六组单因素消融的配对RMSE差值；完整逐种子差值见`paired_ablation_deltas.csv`。','',
        '![收敛曲线](figures/convergence.png)','',
        '图3：三种主要算法所有种子验证曲线，粗线为该轮仍有记录的均值。提前停止后不伪造继续训练曲线；七组消融各自五种子完整曲线另存`figures/curve_*.png`。','',
        '![误差分解](figures/bias_decomposition.png)','',
        '图4：MSE分解为Bias²及中心化误差方差；对神经网络先按每个种子分解，再对五种子取平均。','',
        '![独立站空间结构](figures/spatial_structure.png)','',
        '图5：每小时分别去掉独立站平均值后的空间异常误差；属于稀疏站点证据，未伪造全场真值。','',
        '![固定站点时序](figures/station_timeseries.png)','',
        '图6：固定排序第一的测试站全部评价期时序；蓝带为五种子SD，非预测置信区间。该站非按表现选取。','',
        '![稀疏站误差图](figures/station_error_map.png)','',
        '图7：完整模型先对种子预测取平均，再计算每个测试站的时段平均Bias。只在真实站点画点，不把稀疏误差插值成“真值面”。','',
        '![完整模型预测场](figures/field_full.png)','',
        '![线性加残差预测场](figures/field_linear_idw.png)','',
        '图8—9：预先固定2019-08-12 12:30 UTC，完整模型五种子预测场均值与线性+残差场使用统一色标；线性校正场另存`figures/field_linear.png`。均为预测，不是真实高分辨率场；细纹理不等于精度提高。','',
        '## 9月跨月份直接推理与微调','',
        cross['description'],'',
        '|试验|执行状态|结果|','|---|---|---|',
        '|8月权重→9月直接推理|缺少可兼容三源输入，未执行|未获得；不得用8月结果替代|',
        '|少量9月训练站微调→9月后段评价|同上，未执行|未获得；不填写假数值|','',
        '数据来源和逐项下载要求见`9月数据下载需求.md`；本地检查清单见`september_data_inventory.csv`。','',
        '预先设计：优先按同一40/10/10站点划分，加载本轮完整模型全部五种子的8月最佳权重，保持变量、单位、时区、网格和配置。直接融合使用9月同期40训练站观测，不使用测试站观测更新或选择参数。由于8月权重只训练7日，该试验必须明确写为“8月7日窗口训练→9月”，若另训8月整月则单独版本保存。',
        '可选微调：固定9月前3个完整可用日，仍只用40训练站更新、10验证站选权重；学习率0.0005、上限10轮、patience=3、minimum_epochs=4。直接与微调都在后续相同日期/测试站记录评价，不能把微调日期测试成绩计入后段指标。该协议在得到9月测试值前冻结；未验证前只是计划。','',
        '## 主要问题与下一步','',
        '1. 先确认RH35 PM2.5和地面PM2.5的测量/转换定义，并取得站点坐标基准直接证据；近90 μg/m³偏差不宜只靠网络强制纠正。',
        '2. 当前频谱Loss只对GEOS粗场作监督，可能强化有偏背景；粗场MSE约束也存在同样冲突。根据本轮负贡献结果再提出修改，而不是据测试分数反复调参。',
        '3. 小训练窗口、30轮上限及五种子波动限制结论；下一轮使用新的外部时段预先冻结更充分训练预算，不挑最好种子。',
        '4. 简单校正+残差插值是必要强基线。后续如检验气象、跨月份、缺测或实时可靠度贡献，应分别预声明新实验，不增加复杂网络。',
        '5. 先补齐9月三源数据和一致语义，再做跨月份直接融合与微调；缺失数据不以8月复制、模拟或城市平均观测替代。','',
        '## 复现与保存','',
        '- 冻结协议：`experiment_plan.json`；配置：`configs/`；最佳权重：`checkpoints/`；逐轮日志：`logs/`；全部未来预测场：`fields/`；逐站预测：`predictions/`。',
        '- 逐次指标：`test_runs.csv`；均值/SD及范围：`seed_summary.csv`；配对贡献：`paired_ablation_deltas.csv`；误差分解与源表：`error_decomposition.csv`和`station_error_map_source.csv`。',
        '- 数据/旧文件/代码哈希、依赖与命令：`provenance.json`；训练完成后的方案冻结：`scheme_frozen_before_test.json`；检查：`audit.json`。',
        '- 本轮源代码快照：`code/`；分析与绘图源文件：`code/report_pm25_ablation_validation.py`；图像均有可编辑PDF/SVG及PNG。','']
    (output/'神经网络消融与泛化验证报告.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    output=args.output
    if not (output/'audit.json').exists():
        raise RuntimeError('All training and final evaluation must finish first')
    runs=pd.read_csv(output/'test_runs.csv')
    summary=pd.read_csv(output/'seed_summary.csv',index_col='method')
    deltas=pd.read_csv(output/'paired_ablation_deltas.csv')
    if len(runs[runs.seed!=-1])!=45 or not (summary.n_seeds==5).all():
        raise RuntimeError('Missing seeds or runs; do not report a complete experiment')
    label_fields(output)
    figures(output,runs,summary,deltas)
    report(output,runs,summary,deltas)
    import shutil
    shutil.copy2(__file__,output/'code'/Path(__file__).name)
    print(output/'神经网络消融与泛化验证报告.md')


if __name__=='__main__':
    main()
