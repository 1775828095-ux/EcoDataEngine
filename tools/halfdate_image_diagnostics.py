"""Predeclared secondary structure/peak diagnostics, never a selection score."""
from pathlib import Path
import hashlib,sys
import numpy as np
import pandas as pd
import xarray as xr
from run_halfdate_reconstruction import OUT,ROOT,load,labels,config
from ecodata.algorithms.numerical_search import save


def register():
    if (OUT/'test_metrics.json').exists():raise ValueError('Register before final test analysis')
    save(OUT/'image_diagnostic_protocol.json',dict(
        used_for_selection=False,peak_threshold='Per-patch 90th percentile of QA-finite 2013-2014 NDVI only',
        texture='Five-point spatial Laplacian error only where all five reference pixels are finite; grid-index differences, not physical derivatives',
        temporal='Successive hidden-date NDVI change error divided by elapsed days, finite reference pairs only',
        missing_prediction='Fixed reference support; failures retained, no available-case advantage',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    p=OUT/'source_frozen/tools/halfdate_image_diagnostics.py';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(Path(__file__).read_bytes())


def laplace(a):return a[:,1:-1,:-2]+a[:,1:-1,2:]+a[:,:-2,1:-1]+a[:,2:,1:-1]-4*a[:,1:-1,1:-1]


def analyze():
    if not (OUT/'predictions_frozen.json').exists():raise ValueError('Predictions must be sealed')
    protocol=load(OUT/'image_diagnostic_protocol.json')
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==protocol['script_sha256']
    thresholds={};rows=[]
    for entry in load(OUT/'predictions_frozen.json'):
        case=entry['case'];patch=int(case.split('_')[0].removeprefix('patch'))
        if patch not in thresholds:
            with xr.open_dataset(OUT/'data'/f'patch{patch}_all_reference.nc') as ds:
                train=ds.ndvi.sel(time=slice('2013','2014')).values
                thresholds[patch]=float(np.quantile(train[np.isfinite(train)],.9))
        truth=labels(case,'test');y=np.asarray(truth.values,float)
        with xr.open_dataset(entry['path']) as ds:
            prediction=ds.load();p=np.asarray(prediction.ndvi.sel(time=truth.time).values,float)
        mask=np.isfinite(y);valid=np.isfinite(p[mask]);row=dict(case=case,method=entry['method'],reference_pixels=int(mask.sum()),
            prediction_coverage=float(valid.mean()),peak_threshold=thresholds[patch])
        if not valid.all():row['status']='INCOMPLETE_OUTPUT';rows.append(row);continue
        peak=mask&(y>=thresholds[patch]);d=p-y
        row.update(status='PASS',peak_pixels=int(peak.sum()),peak_MAE=float(np.mean(abs(d[peak]))) if peak.any() else None,
            peak_Bias=float(np.mean(d[peak])) if peak.any() else None)
        a,b=laplace(p),laplace(y);m=np.isfinite(b)
        row['texture_RMSE']=float(np.sqrt(np.mean((a[m]-b[m])**2))) if m.any() else None
        days=np.diff(pd.DatetimeIndex(truth.time.values).values).astype('timedelta64[s]').astype(float)/86400
        delta=(np.diff(p,axis=0)-np.diff(y,axis=0))/days[:,None,None];m=np.isfinite(delta)
        row['temporal_change_RMSE_per_day']=float(np.sqrt(np.mean(delta[m]**2))) if m.any() else None
        with xr.open_dataset(config(case,'test')['inputs'][0]['path']) as ds:
            original=ds.ndvi.load();restored=prediction.ndvi.sel(time=original.time)
            known=np.isfinite(original.values);row['visible_anchor_max_change']=float(np.max(abs(restored.values[known]-original.values[known])))
        rows.append(row)
    save(OUT/'image_diagnostics.json',dict(training_peak_thresholds=thresholds,rows=rows))
    pd.DataFrame(rows).to_csv(OUT/'image_diagnostics.csv',index=False,encoding='utf-8-sig')
    lines=['# 结构、峰值与可见观测保护诊断','',
        '这些指标只用于补充分析，没有参与选图或重新调参。高NDVI阈值来自2013—2014年；空间纹理使用五点差分，不是物理动力一致性。时序变化只使用相邻隐藏日期均有QA参考的像元。', '',
        f"已检查{len(rows)}份封存产品；完整预测支持{sum(r['status']=='PASS' for r in rows)}份。",'']
    good=[r for r in rows if r['status']=='PASS']
    if good:lines.append(f"可见原始有效观测最大改动：{max(r['visible_anchor_max_change'] for r in good):.12g}。")
    lines+=['','逐案例、种子和方法的峰值误差、纹理误差、变化率误差及覆盖率见image_diagnostics.csv；失败及缺失分组不填虚假数值。']
    path=ROOT/'docs/半日期隐藏时空重建_结构诊断_20261028.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':{'register':register,'analyze':analyze}[sys.argv[1]]()
