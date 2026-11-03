"""Independent artifact replay, isolation audit and qualified delivery."""
from collections import Counter
from pathlib import Path
import hashlib,json,sys,zipfile
import numpy as np
import pandas as pd
import xarray as xr
from run_halfdate_reconstruction import OUT,ROOT,load,freeze,cases,config,inputs,bench,labels,SEEDS
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.autonomous_field import measured
from ecodata.algorithms.numerical_search import save


def audit():
    freeze();checks=[];manifest=load(OUT/'predictions_frozen.json')
    for name,digest in load(OUT/'input_manifest.json').items():assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest,name
    for c in cases():
        periods=[]
        for phase in ['development','validation','test']:
            cfg=config(c,phase)
            with xr.open_dataset(cfg['inputs'][0]['path']) as ds:fine=ds.load()
            ref=labels(c,phase)
            assert not pd.DatetimeIndex(ref.time.values).isin(fine.time.values).any()
            year={'development':2015,'validation':2016,'test':2018}[phase]
            assert np.all(fine.time.dt.year==year) and np.all(ref.time.dt.year==year)
            assert len(cfg['inputs'])==2 and not any(k in cfg for k in ['evaluation','search_validation'])
            with xr.open_dataset(cfg['inputs'][1]['path']) as ds:assert np.all(ds.time.dt.year==year)
            periods.append(dict(phase=phase,visible_dates=len(fine.time),hidden_dates=len(ref.time),year=year))
        checks.append(dict(case=c,periods=periods,hidden_dates_absent=True,annual_derivatives_absent=True))
    replay=[]
    for n,entry in enumerate(manifest):
        path=Path(entry['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        folder=OUT/'independent_replay'/entry['case']/entry['method'];cfg=config(entry['case'],'test')
        if (folder/'deployment.json').exists():receipt=load(folder/'deployment.json')
        else:receipt=api.execute_frozen_graph(inputs(cfg),cfg['request'],entry['graph'],folder)
        with xr.open_dataset(path) as expected,xr.open_dataset(receipt['product']) as actual:
            xr.testing.assert_allclose(expected,actual)
        replay.append(dict(case=entry['case'],method=entry['method'],calls=receipt['cost']['used'],reference_used=receipt['reference_used']))
        if n%12==0:print('REPLAY',n,'/',len(manifest),flush=True)
    trajectories=[]
    for folder in sorted((OUT/'runs/validation').iterdir()):
        summary=load(folder/'summary.json');meta=load(folder/'metadata.json');cost=load(folder/'total_cost.json')
        assert cost['used']<=cost['limit']
        trace=load(folder/'mutation_trace.json')
        if meta['control']=='none':assert not trace
        eligible=[json.loads(l) for l in (folder/'attempts.jsonl').read_text().splitlines() if json.loads(l)['status']=='ELIGIBLE_OUTPUT']
        if summary.get('initial_best') and summary.get('best'):
            assert summary['best']['metrics']['objective']<=summary['initial_best']['metrics']['objective']+summary.get('objective_tie_resolution',1e-14)
        trajectories.append(dict(**meta,status=summary['status'],initial=(summary.get('initial_best') or {}).get('metrics',{}).get('objective'),
            final=(summary.get('best') or {}).get('metrics',{}).get('objective'),calls=cost['used'],seconds=cost['service_seconds'],
            eligible=len(eligible),transitions=len(trace),failed_calls=cost['failed_calls'],
            nodes=(summary.get('best') or {}).get('algorithm_nodes')))
    for c in cases():
        for s in SEEDS:
            pair=[r for r in trajectories if r['case']==c and r['seed']==s]
            assert len(pair)==2
            if all(r['initial'] is not None for r in pair):assert pair[0]['initial']==pair[1]['initial']
    save(OUT/'independent_audit.json',dict(status='PASS',isolation=checks,replay=replay,trajectories=trajectories,
        test_used_for_graph_selection=False,global_optimum_certified=False))
    print('AUDIT PASS',len(replay),len(trajectories),flush=True)


def report():
    freeze();assert load(OUT/'independent_audit.json')['status']=='PASS'
    summary=load(OUT/'effect_summary.json');metrics=load(OUT/'test_metrics.json');prepared=load(OUT/'prepared.json')
    trajectories=load(OUT/'independent_audit.json')['trajectories'];details=[]
    for c in cases():
        # Secondary strata derive only from visible coarse dynamics, threshold from development.
        magnitudes={}
        for phase in ['development','test']:
            cfg=config(c,phase)
            with xr.open_dataset(cfg['inputs'][1]['path']) as ds:series=ds.ndvi.mean(('lat','lon'),skipna=True).to_series()
            t=pd.DatetimeIndex(cfg['request']['target_times']);queue=series.index.union(t).sort_values()
            z=series.reindex(queue).interpolate(method='time').ffill().bfill().reindex(t).values
            days=np.diff(t.values).astype('timedelta64[s]').astype(float)/86400
            change=np.r_[0,abs(np.diff(z))/days]
            magnitudes[phase]=dict(zip(t.astype(str),change.tolist()))
        threshold=float(np.quantile(list(magnitudes['development'].values()),.75))
        refs=labels(c,'test');times=pd.DatetimeIndex(refs.time.values).astype(str)
        high=np.array([magnitudes['test'][t]>=threshold for t in times])
        for r in metrics:
            if r['case']==c and r['status']=='PASS':
                values=np.asarray(r['frame_RMSE']);details.append(dict(case=c,method=r['method'],threshold=threshold,
                    high_change_dates=times[high].tolist(),high_change_RMSE=float(values[high].mean()) if high.any() else None,
                    other_RMSE=float(values[~high].mean()) if (~high).any() else None))
    save(OUT/'high_change_secondary.json',details)
    graphs=[]
    for entry in load(OUT/'predictions_frozen.json'):
        graph=entry['graph'];used=set()
        def visit(k):
            if k in used:return
            used.add(k)
            for p in graph['records'][k]['parents']:visit(p)
        visit(graph['key']);operations=Counter(graph['records'][k]['operation'] for k in used)
        roots=[graph['records'][k]['parameters']['source'] for k in used if graph['records'][k]['operation']=='source']
        graphs.append(dict(case=entry['case'],method=entry['method'],operations=dict(operations),source_branches=roots,
            scientific_nodes=sum(v for k,v in operations.items() if k!='source')))
    save(OUT/'selected_graph_inventory.json',graphs)
    lines=['# 半日期隐藏高分辨率重建：实验交付','',
        '平台公共入口自行生成候选、真实执行并随机变异。2016选图冻结后，2018部署接口不接收测试标签。',
        '', '## 数据与实验', '',
        '同一流域三个48×48细像元区域；Landsat8约30m与独立MOD09Q1约250m/8天，统一计算NDVI。目标是缺失日期的细网格恢复，未降到1km。0.0003度细网格实际约29m×33m，不是精确30m正方形。',
        '2013—2014仅用于区域QA；2015开发；2016验证选图；2018测试。全年影像中每日期至少30%像元QA有效，交错与连续中段各隐藏floor(n/2)个符合条件日期，两端锚点保留。云覆盖像元不作真值；所有方案必须在同一固定有效参考支持评价。原季节影像/70%清晰门限因2016年仅两个日期而在效果实验前修订，保留旧记录。',
        '参考是QA清晰Landsat派生NDVI，不是地面观测；MODIS光谱及复合期内观测日差异保留，不宣称精确守恒。', '',
        '2017年首轮诊断结果完整保留：随机变异0.06229850、不变异0.06110817，变异较差；粗源插值对照因遗漏data_kind连续类型标记被拒绝，属于接入错误。补齐已知物理类型标记后重新冻结；2017不再充当新盲测，新增2018年作为最终留出年。没有根据2018误差调权重或选路线。', '',
        '| 方法 | 隐藏日期等权RMSE | MAE | 梯度误差 |', '|---|---:|---:|---:|']
    for method,r in summary.items():
        if r['status']=='PASS':lines.append(f"| {method} | {r['RMSE']:.8f} | {r['MAE']:.8f} | {r['gradient_RMSE']:.8f} |")
        else:lines.append(f'| {method} | 未完整完成 | — | — |')
    lines+=['','## 结果边界','']
    lines+=['| 案例 | 开发期可见/隐藏 | 验证期可见/隐藏 | 测试期可见/隐藏 |','|---|---:|---:|---:|']
    for case in prepared['cases']:
        counts=[str(len(case['phases'][p]['visible_dates']))+'/'+str(len(case['phases'][p]['hidden_dates'])) for p in ['development','validation','test']]
        lines.append('| '+case['case']+' | '+' | '.join(counts)+' |')
    lines+=['']
    if summary['random']['status']=='PASS':
        for name in ['none','development_fixed','fine_time_linear','anchor_residual']:
            r=summary[name]
            if r['status']=='PASS':lines.append(f"- 随机变异相对 {name} 的平均RMSE改善：{100*(1-summary['random']['RMSE']/r['RMSE']):.4f}%；负数表示更差。")
        for pattern in ['alternating','contiguous']:
            lines.append('- '+pattern+'：'+', '.join(name+'='+f"{np.mean([v for k,v in r['by_case'].items() if k.endswith(pattern)]):.8f}" for name,r in summary.items() if r['status']=='PASS'))
    for control in ['random','none']:
        selected=[r for r in trajectories if r['control']==control]
        lines.append(f"- {control}：每次平均实际调用{np.mean([r['calls'] for r in selected]):.2f}，平均服务时间{np.mean([r['seconds'] for r in selected]):.2f}秒；缓存不重复计调用。")
    random_graphs=[r for r in graphs if r['method'].startswith('random_')]
    lines+=['',f"随机变异最终图使用两个源分支：{sum(len(r['source_branches'])==2 for r in random_graphs)}/{len(random_graphs)}；不能仅凭下载了两源就声称发生了融合。",'',
        '全部逐种子结果、失败、真实调用账本、初始化及变异轨迹保留。开发最强固定来自发现图的跨案例真实重放，不是理论全空间最优。追加高变化时段分层只用开发阈值，不重新选图。',
        '单流域、三个空间相依小区域、一个测试年；重复种子不增加独立流域样本量。不能由此声称普遍最优、恢复未知真实细节或改善SWAT+/WRF。',
        '', '## 实现与验收','',
        '复用公共搜索、画像、状态去重、随机变异、DAG执行及质量门。补充公共栅格时间插值与可见锚点残差节点（不是STARFM），补充固定有限参考支持及无标签固定图部署。没有新增候选整图池；未知栅格响应仍标UNKNOWN，没有用另一套学习排序代替画像。',
        '首版开发的float32/float64误差计算差异已修正：统一双精度误差计算，使用按参考规模计算的舍入同分尺度，再比较图复杂度与成本；精确网格中心插值不再在相同完整网格上重复执行。首版结果全部留在diagnostic_float32目录，不能当作正式优势。正式数据、种子与预算保持不变后重跑。',
        '最终轮在开发年验收所有固定对照后才运行正式验证。原始残差控制仍保留；另外预先冻结残差加NDVI物理范围投影控制。范围投影在无违例的新输入上合法返回原值，搜索仍跳过已知恒等投影，不把这种部署边界当作算法失败。',
        '同样修复栅格补缺/时间插值的跨年幂等边界：新输入已经完整时可原样继续执行，搜索仍不添加这类无效节点。旧部署失败在2018参考未评分时发现并归档，修复后重跑开发与验证。',
        '独立审计：输入哈希及阶段隔离通过；所有封存产品用公共接口重新执行并逐值核对；正式运行初始化成对核对，失败不剔除。详见independent_audit.json。']
    text='\n'.join(lines)+'\n';path=ROOT/'docs/半日期隐藏时空重建_实验交付_20261028.md';path.write_text(text,encoding='utf-8')
    master=ROOT/'docs/EcoData Engine项目总交付报告.md'
    heading='## 69. 半日期隐藏时空重建（2026-10-28）'
    existing=master.read_text(encoding='utf-8')
    if heading in existing:
        import re
        (OUT/'master_before_ch69_final.md').write_text(existing,encoding='utf-8')
        start=existing.index(heading);following=re.search(r'(?m)^## \d+\. ',existing[start+len(heading):])
        end=start+len(heading)+following.start() if following else len(existing)
        master.write_text(existing[:start]+heading+'\n\n'+text+existing[end:],encoding='utf-8')
    else:
        with master.open('a',encoding='utf-8') as f:f.write('\n\n'+heading+'\n\n'+text)
    pd.DataFrame(metrics).to_csv(OUT/'test_metrics.csv',index=False,encoding='utf-8-sig')
    archive=ROOT/'output/半日期隐藏时空重建_实验交付_20261028.zip'
    import shutil
    (OUT/'logs').mkdir(exist_ok=True)
    for p in (ROOT/'output').glob('halfdate*.log'):shutil.copy2(p,OUT/'logs'/p.name)
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file() and 'pytest_' not in str(p) and 'before' not in p.parts:z.write(p,p.relative_to(ROOT))
        z.write(path,path.relative_to(ROOT));z.write(ROOT/'docs/半日期隐藏时空重建_冻结协议_20261028.md','protocol.md')
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None;count=len(z.namelist())
    save(OUT/'delivery_verified.json',dict(archive=str(archive),members=count,bytes=archive.stat().st_size,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),report=str(path)))
    print(text,flush=True)


if __name__=='__main__':{'audit':audit,'report':report}[sys.argv[1]]()
