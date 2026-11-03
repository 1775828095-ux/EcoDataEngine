"""Pre-test secondary check: does coarse temporal information affect the graph?"""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import xarray as xr
from run_halfdate_reconstruction import OUT,ROOT,load,freeze,config,inputs,bench,labels
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.autonomous_field import measured
from ecodata.algorithms.numerical_search import save


def predict():
    freeze()
    if (OUT/'test_metrics.json').exists():raise ValueError('Addendum must freeze before test scoring')
    seal=OUT/'coarse_ablation_predictions.json'
    if seal.exists():raise ValueError('Secondary predictions already sealed')
    save(OUT/'coarse_ablation_protocol.json',dict(
        registered_before_test=True,selection_changed=False,
        question='Does the selected graph actually use independent coarse temporal variation?',
        intervention='Replace only visible MODIS time series with its per-pixel same-year mean, preserving coordinates/units/source identity',
        interpretation='Destructive diagnostic, not an alternative source or a new tuned algorithm',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    snapshot=OUT/'source_frozen/tools/halfdate_coarse_ablation.py'
    snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(Path(__file__).read_bytes())
    rows=[]
    for entry in load(OUT/'predictions_frozen.json'):
        if not entry['method'].startswith('random_'):continue
        c=entry['case'];cfg=config(c,'test');actual_inputs=inputs(cfg)
        p=Path(cfg['inputs'][1]['path'])
        with xr.open_dataset(p) as ds:coarse=ds.load()
        changed=coarse.copy(deep=True)
        changed.ndvi.values[:]=coarse.ndvi.mean('time',skipna=True).values
        folder=OUT/'coarse_ablation'/c/entry['method'];row=dict(case=c,method=entry['method'])
        try:
            folder.mkdir(parents=True,exist_ok=True)
            changed.to_netcdf(folder/'coarse_intervention.nc',engine='h5netcdf')
            variant=api.SearchInput(actual_inputs[1].filename,'background',(folder/'coarse_intervention.nc').read_bytes())
            receipt=api.execute_frozen_graph([actual_inputs[0],variant],cfg['request'],entry['graph'],folder)
            out=Path(receipt['product']);row.update(status='PASS',path=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),calls=receipt['cost']['used'])
            with xr.open_dataset(entry['path']) as a,xr.open_dataset(out) as b:
                d=a.ndvi.values-b.ndvi.values
                row.update(output_change_RMSE=float(np.sqrt(np.nanmean(d*d))),max_abs_change=float(np.nanmax(abs(d))))
        except Exception as exc:row.update(status='FAILED',error=str(exc))
        rows.append(row)
    save(seal,rows);print('COARSE ABLATION SEALED',len(rows),flush=True)


def score():
    freeze();rows=load(OUT/'coarse_ablation_predictions.json');original=load(OUT/'test_metrics.json')
    for r in rows:
        if r['status']!='PASS':continue
        p=Path(r['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
        ctx,_=bench(r['case'],'test')
        with xr.open_dataset(p) as ds:
            try:r['test_metrics']=measured(ds.load(),labels(r['case'],'test'),ctx)
            except Exception as exc:r['test_error']=str(exc);continue
        baseline=next(x for x in original if x['case']==r['case'] and x['method']==r['method'])
        if baseline['status']=='PASS':r['RMSE_worsening']=r['test_metrics']['objective']-baseline['objective']
    save(OUT/'coarse_ablation_results.json',rows)
    valid=[r for r in rows if 'RMSE_worsening' in r]
    lines=['# 粗源时序作用补充检查','',
        '在测试评分前封存：固定18个已选随机变异图，仅把可见MODIS时序替换为同像元全年均值，真实重跑；不改图、不用测试标签重新选择。此破坏性对照只检查粗源时序是否实际影响处理。','',
        f"封存{len(rows)}个对照；成功评分{len(valid)}个。",
        f"输出确实变化：{sum(r.get('max_abs_change',0)>1e-10 for r in rows)}/{len(rows)}。"]
    if valid:
        lines.append(f"去除粗源时序后，测试RMSE平均变化：{np.mean([r['RMSE_worsening'] for r in valid]):.8f}；正值表示原粗源时序有帮助。")
        lines.append(f"测试变差/变好/近似不变：{sum(r['RMSE_worsening']>1e-8 for r in valid)}/{sum(r['RMSE_worsening']<-1e-8 for r in valid)}/{sum(abs(r['RMSE_worsening'])<=1e-8 for r in valid)}。")
    lines+=['','额外真实调用全部记录在各deployment.json；该成本不归入原180次搜索预算，也不算搜索效率优势。此对照不能证明恢复未知真实细节，依然只对原固定QA参考支持评分。']
    path=ROOT/'docs/半日期隐藏时空重建_粗源作用补充检查_20261028.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':{'predict':predict,'score':score}[sys.argv[1]]()
