"""Replay frozen selected products after a diagnostic-only seam correction."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,sys,difflib
import numpy as np
import xarray as xr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tools'))
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.composition_decision import Composition
import run_scheduled_fusion_study as study
OUT=study.OUT
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert (OUT/'results.json').exists();study.stores()
    source=ROOT/'src/ecodata/algorithms/composition_decision.py';old=OUT/'source_frozen'/source.relative_to(ROOT)
    difference=''.join(difflib.unified_diff(old.read_text(encoding='utf-8').splitlines(keepends=True),source.read_text(encoding='utf-8').splitlines(keepends=True),fromfile='frozen-scientific-source',tofile='released-source'))
    assert 'get_indexer' in difference and 'searchsorted' in difference
    (OUT/'boundary_diagnostic_patch.diff').write_text(difference,encoding='utf-8')
    groups=defaultdict(dict)
    for path in (OUT/'search').rglob('summary.json'):
        report=load(path);case=int(path.parts[-4]);groups[case][report['recommended_key']]=(report['best']['plan'],report['best']['product'])
    for path in (OUT/'test').rglob('evaluation.json'):
        report=load(path);case=int(path.parts[-3]);groups[case][path.parent.name]=(report['plan'],report['product'])
    checked=0;diagnostics=[];maximum_difference=0.;bitwise_exact=True
    for case,items in groups.items():
        inputs,req=study.bundle(case);assert 'max_seam_jump' not in req['composition'].get('coordination',{})
        parsed,_=api.parse_inputs(inputs);engine=Composition(parsed,req,api.platform_registry())
        for key,(plan,path) in items.items():
            genome=[]
            for slot,axis in zip(engine.slots,engine.axes):
                choice=next(p['choice'] for p in plan if (p['field'],p['window'],p['branch'])==slot)
                genome.append(next(i for i,x in enumerate(axis) if x==choice))
            product,_,_,checks=engine.execute(genome)
            with xr.open_dataset(path,engine='h5netcdf') as ds:
                previous=ds.load();delta=float(abs(product.air_temperature-previous.air_temperature).max())
                scale=max(1.,float(abs(previous.air_temperature).max()))
                numerical_bound=64*np.finfo(float).eps*scale
                xr.testing.assert_allclose(product,previous,rtol=0,atol=numerical_bound)
                maximum_difference=max(maximum_difference,delta);bitwise_exact &= delta==0
            measured=float(abs(product.air_temperature.isel(time=7)-product.air_temperature.isel(time=6)).max())
            seam=next(c['max_seam_jump'] for c in checks if 'max_seam_jump' in c)
            assert abs(seam-measured)<1e-9
            diagnostics.append(dict(case=case,key=key,actual_seam_jump_K=seam));checked+=1
    explanation=f'最终独立检查发现：输出时间队列不含时段边界时，旧max_seam_jump诊断会漏报。科学比较全部使用冻结源码，且没有启用该硬阈值；评分与验证目标也不使用该摘要。完成实验后仅修复边界邻居定位（保留原冻结源码/产物），定向测试确认声明的跳变门能够拒绝超限输出，并重放所有不同的冻结推荐及测试图。最大逐元素差{maximum_difference:.3g}K，在64倍机器精度乘以场值尺度的舍入检查范围内；不冒充逐比特一致。未根据测试成绩改算法、候选、评价容差或阈值。'
    receipt=dict(status='PASS',checked_unique_selected_products=checked,frozen_sha256=sha(old),released_sha256=sha(source),
        no_scientific_request_enabled_seam_gate=True,all_replayed_values_within_roundoff=True,bitwise_exact=bitwise_exact,max_abs_difference_K=maximum_difference,diagnostics=diagnostics,explanation=explanation)
    (OUT/'boundary_diagnostic_fix.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='PASS',replayed=checked),ensure_ascii=False))
if __name__=='__main__':main()
