"""Final independent readback and complete evidence packaging; no selection."""
from run_halfdate_reconstruction import ROOT,OUT,load,freeze,config,inputs
from pathlib import Path
import hashlib,re,zipfile,shutil
import numpy as np
import xarray as xr
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.numerical_search import save


def counts(name):
    path=ROOT/'output'/name;text=path.read_text(encoding='utf-8',errors='replace')
    match=re.search(r'(?m)^([.FEsxX]+)\s+\[100%\]',text)
    if not match:raise ValueError('No final pytest summary: '+name)
    return dict(passed=match[1].count('.'),failed=match[1].count('F'),errors=match[1].count('E'),log=str(path))


def main():
    freeze();audit=load(OUT/'independent_audit.json');assert audit['status']=='PASS'
    results=load(OUT/'effect_summary.json');assert results['random']['status']=='PASS' and results['none']['status']=='PASS'
    manifest=load(OUT/'predictions_frozen.json')
    for r in manifest:assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
    extra=[]
    for row in load(OUT/'coarse_ablation_results.json'):
        if row['status']!='PASS':extra.append(dict(case=row['case'],method=row['method'],status='FAILED_INTERVENTION_RETAINED'));continue
        entry=next(r for r in manifest if r['case']==row['case'] and r['method']==row['method'])
        cfg=config(row['case'],'test');base=inputs(cfg)
        variant=OUT/'coarse_ablation'/row['case']/row['method']/'coarse_intervention.nc'
        changed=api.SearchInput(base[1].filename,'background',variant.read_bytes())
        folder=OUT/'coarse_ablation_independent'/row['case']/row['method']
        receipt=(load(folder/'deployment.json') if (folder/'deployment.json').exists()
                 else api.execute_frozen_graph([base[0],changed],cfg['request'],entry['graph'],folder))
        with xr.open_dataset(row['path']) as expected,xr.open_dataset(receipt['product']) as actual:xr.testing.assert_allclose(expected,actual)
        extra.append(dict(case=row['case'],method=row['method'],status='PASS',calls=receipt['cost']['used']))
    tests=counts('halfdate_tests_idempotent_release.log')
    expanded=counts('halfdate_tests_final_public_regression.log')
    baseline=counts('halfdate_preexisting_test_check.log')
    assert tests['failed']==tests['errors']==0
    assert expanded['failed']==baseline['failed']==2 and expanded['errors']==0
    save(OUT/'verification_summary.json',dict(core_tests=tests,expanded_regression=expanded,
        preexisting_baseline=baseline,preexisting_reason='Unchanged raster upload test expects8 outputs; pre-turn frozen source and current source both return14',
        ablation_replay=extra,main_replay=len(audit['replay']),formal_runs=len(audit['trajectories']),
        source_and_input_hashes='PASS',test_year=2018,completed_date='2026-10-29'))
    control_names=list(load(OUT/'controls_frozen.json')['chosen'])
    complete_fixed=[name for name in control_names if results[name]['status']=='PASS']
    strongest=min(complete_fixed,key=lambda name:results[name]['RMSE'])
    primary=results['random']['RMSE'];none=results['none']['RMSE']
    pairs=[results['none']['by_case'][c]-results['random']['by_case'][c] for c in results['random']['by_case']]
    finding=dict(random_RMSE=primary,no_mutation_RMSE=none,
        improvement_over_none_percent=100*(1-primary/none),
        strongest_frozen_fixed_on_test=strongest,strongest_fixed_RMSE=results[strongest]['RMSE'],
        improvement_over_strongest_fixed_percent=100*(1-primary/results[strongest]['RMSE']),
        case_wins_over_none=int(np.sum(np.asarray(pairs)>1e-8)),total_cases=len(pairs),
        interpretation='Testwise strongest fixed is an evaluation comparator only; no test-based route selection')
    save(OUT/'delivery_findings.json',finding)
    supplement=['','## 最终验收补充（2026-10-29）','',
        f"最终留出年为2018。随机变异RMSE={primary:.8f}，不变异初始化选优={none:.8f}；相对改善{finding['improvement_over_none_percent']:.4f}%（负数为更差）。",
        f"预先冻结固定对照中，测试表现最低的是{strongest}，RMSE={results[strongest]['RMSE']:.8f}；随机变异相对改善{finding['improvement_over_strongest_fixed_percent']:.4f}%。此比较不反馈选图。",
        f"随机变异在6个区域×遮挡案例中优于不变异：{finding['case_wins_over_none']}/6。只有三个空间相依区域，不能把种子或像元当成大量独立案例。",
        f"定向检查{tests['passed']}项通过。扩展回归{expanded['passed']}项通过、2项失败；两项都是旧栅格上传测试写死8个输出，而当前和本轮修改前的冻结源码均返回14个，已独立复现，未改无关测试凑全绿。",
        f"主实验{len(audit['replay'])}份已封存产品独立重放，{len(audit['trajectories'])}次正式轨迹核查；粗源扰动补充重放{sum(r['status']=='PASS' for r in extra)}份。物理越界失败单列保留，没有丢弃失败后平均。",
        '峰值、空间纹理和时间变化误差见image_diagnostics.csv；可见观测保护、固定QA支持覆盖率另列。空间差分是纹理代理，不是动力平衡指标。',
        '不变异对照也会对初始化候选真实执行并按验证误差选择；本轮不能单独归因三画像软评分的优势。栅格响应仍存在UNKNOWN，未声称已学会预测所有数据上的算法效果。',
        '页面浏览器交互未在本轮验收；这里验证的是平台公共文件/API入口及共用数值执行链。',
        '2017负结果、float32诊断与未评分的部署边界诊断完整留在本地diagnostic目录；交付包包含其摘要，正式轮完整图、产品、轨迹和输入均在包内。源卫星大文件本地保留，包内包含所需标准化数据与原始来源/哈希记录。',
        '全部开发、验证、失败、部署和复核成本保留；旧中断重放的成本未完整落盘，不能声称诊断成本为零，也不把额外重放算作变异效率优势。']
    report=ROOT/'docs/半日期隐藏时空重建_实验交付_20261028.md'
    base_text=report.read_text(encoding='utf-8').split('\n## 最终验收补充（2026-10-29）')[0]
    text=base_text+'\n'.join(supplement)+'\n';report.write_text(text,encoding='utf-8')
    master=ROOT/'docs/EcoData Engine项目总交付报告.md';existing=master.read_text(encoding='utf-8')
    heading='## 69. 半日期隐藏时空重建（2026-10-28）';start=existing.index(heading)
    following=re.search(r'(?m)^## \d+\. ',existing[start+len(heading):])
    end=start+len(heading)+following.start() if following else len(existing)
    master.write_text(existing[:start]+heading+'\n\n'+text+existing[end:],encoding='utf-8')
    dest=OUT/'source_frozen/tools/finalize_halfdate_delivery.py';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(__file__).read_bytes())
    (OUT/'logs').mkdir(exist_ok=True)
    for p in (ROOT/'output').glob('halfdate*.log'):shutil.copy2(p,OUT/'logs'/p.name)
    archive=ROOT/'output/半日期隐藏时空重建_实验交付_20261028.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file() and 'pytest_' not in str(p) and p.name!='delivery_verified.json':z.write(p,p.relative_to(ROOT))
        for name in ['半日期隐藏时空重建_实验交付_20261028.md','半日期隐藏时空重建_冻结协议_20261028.md',
                     '半日期隐藏时空重建_粗源作用补充检查_20261028.md','半日期隐藏时空重建_结构诊断_20261028.md']:
            p=ROOT/'docs'/name;z.write(p,p.relative_to(ROOT))
        for tag in ['float32','metadata','runtime_contract']:
            folder=ROOT/'output'/('halfdate_reconstruction_20261028_diagnostic_'+tag)
            for name in ['effect_summary.json','test_metrics.json','deployment_failures.json','numeric_revision.json','withdrawn_report.md','protocol_v2.md']:
                p=folder/name
                if p.exists():z.write(p,'diagnostics/'+tag+'/'+name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;members=z.namelist();assert len(members)==len(set(members))
        assert z.read('docs/'+report.name)==report.read_bytes()
    save(OUT/'delivery_verified.json',dict(archive=str(archive),members=len(members),bytes=archive.stat().st_size,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),report=str(report),completed_date='2026-10-29',verification='PASS'))
    print(finding,flush=True);print('FINAL DELIVERY VERIFIED',len(members),archive.stat().st_size,flush=True)


if __name__=='__main__':main()
