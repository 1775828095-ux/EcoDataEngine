"""Readback/parity/evidence audit only: never starts search or reads sealed truth."""
from pathlib import Path
import sys,json,hashlib,time,io,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,xarray as xr
from ecodata.algorithms.numerical_search import save
from ecodata.interface.temporal_search_page import download_search_result
OUT=ROOT/'output/capability_matching_20261021'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k not in {'seconds','elapsed','cache_info'} and not k.endswith('_seconds')}
    if isinstance(x,list):return [clean(v) for v in x]
    return x


def equivalent(a,b):
    da,db=read(a/'decision.json'),read(b/'decision.json');sa,sb=read(a/'summary.json'),read(b/'summary.json')
    assert da['requirements']['required']==db['requirements']['required']
    assert da['requirements']['optional_capabilities']==db['requirements']['optional_capabilities']
    assert da['matches']==db['matches'] and sa['candidate_domain']==sb['candidate_domain']
    pa=[p for p in sa['proposals'] if 'probabilities' in p];pb=[p for p in sb['proposals'] if 'probabilities' in p]
    # Declaration evidence is intentionally not identical, but numerical
    # decisions and actual operations must be; do not compare descriptions.
    for x,y in zip(pa,pb):
        assert x['keys']==y['keys'] and x['chosen_index']==y['chosen_index']
        np.testing.assert_allclose(x['probabilities'],y['probabilities'],rtol=0,atol=0)
    assert len(pa)==len(pb) and sa['best']['graph']==sb['best']['graph']
    with xr.open_dataset(sa['best']['product'],engine='h5netcdf') as x,xr.open_dataset(sb['best']['product'],engine='h5netcdf') as y:
        for key in x.data_vars:np.testing.assert_array_equal(x[key].values,y[key].values)
    return dict(status='PASS',proposals=len(pa),best_graph=sa['best']['key'],same_products=True)


def run():
    started=time.perf_counter();audit=[]
    for chapter in ['problem_directed_20261021','local_problem_20261021']:
        for path in sorted((ROOT/'output'/chapter/'runs').glob('*/operation_transitions.json')):
            rows=read(path)['records']
            audit.append(dict(source=str(path),records=len(rows),single_action_flags=sum(bool(r.get('single_action_learning_label')) for r in rows),
                admitted=0,reason='Legacy transition lacks independently verified per-side information/parameter and explicit single-stage step signatures for this scoring protocol; raw gain is not normalized; repeated clients not independent'))
    for path,kind in [(ROOT/'output/lghap_reconstruction_20261020_v3/validation/metrics.csv','real product-reference'),(ROOT/'output/three_task_routing_20261020/all_metrics.csv','synthetic multi-objective, repeated policies')]:
        table=pd.read_csv(path)
        audit.append(dict(source=str(path),records=len(table),admitted=0,kind=kind,
            reason='Aggregate CSV has no row-level input/queue/information/parameter fingerprints; cannot certify the requested causal contrast merely from plan names. Not relabeled as independent test. No sealed reference opened.'))
    audit_seconds=time.perf_counter()-started
    save(OUT/'historical_evidence_audit.json',dict(records=audit,seconds=audit_seconds,
        new_collection='Only frozen six real LGHAP TRAIN reference fields, not historical sealed tests',
        scope='Explicitly inspected chapter42/43 transition pools and old raster metrics, not all project files'))
    comparisons={}
    for case in ['aligned','mismatch','temporal','temporal_instant','category','interval_mean']:
        version='_v2' if case=='interval_mean' else ''
        comparisons[case]=equivalent(OUT/'layers'/(case+'_declared'+version),OUT/'layers'/(case+'_recognized'+version))
    runs=OUT/'runs'
    comparisons['scored_layers']=equivalent(runs/'declared_toolkit_v4',runs/'recognized_toolkit_v4')
    comparisons['page_toolkit']=equivalent(runs/'recognized_page_v4',runs/'recognized_toolkit_v4')
    on=read(runs/'recognized_toolkit_v4/summary.json');off=read(runs/'recognized_off_toolkit_v4/summary.json')
    assert on['candidate_domain']==off['candidate_domain']
    po=[p for p in on['proposals'] if 'empirical_optional' in p];pf=[p for p in off['proposals'] if 'empirical_optional' in p]
    assert any(any(d['factor']!=1 for d in p['empirical_optional']['details']) for p in po)
    changed=[float(np.max(np.abs(np.asarray(a['probabilities'])-b['probabilities']))) for a,b in zip(po,pf) if a['keys']==b['keys']]
    assert max(changed)>0
    # Direct data-role fingerprints must remain unchanged across a one-source edit.
    aligned=read(OUT/'layers/aligned_recognized/local_port_contracts.json')['resources']
    mismatch=read(OUT/'layers/mismatch_recognized/local_port_contracts.json')['resources']
    assert aligned['source1']==mismatch['source1'] and aligned['source0']['fingerprint']!=mismatch['source0']['fingerprint']
    local=read(OUT/'layers/mismatch_recognized/decision.json')['requirements']['problems']
    affected=[p for p in local if p['rule']=='source_grid_mismatch']
    assert len(affected)==1 and affected[0]['source_ids']==['role:primary:aod']
    paired=read(OUT/'pairs.json');products={}
    for pair in paired:
        for side in ['before','after']:
            p=Path(pair[side]['product'])
            with xr.open_dataset(p,engine='h5netcdf') as ds:assert np.isfinite(ds.aod).all() and ds.aod.min()>=0 and ds.aod.attrs['units']=='1'
            products[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    package=download_search_result(runs/'recognized_page_v4')
    with zipfile.ZipFile(io.BytesIO(package)) as z:
        assert z.testzip() is None and any(n.endswith('product.nc') for n in z.namelist())
        archive=dict(status='PASS',bytes=len(package),members=len(z.namelist()),sha256=hashlib.sha256(package).hexdigest())
    model=read(OUT/'model_report.json')
    from ecodata.algorithms.capability_scoring import CapabilityScoringService
    scorer=CapabilityScoringService(model['model_id'])
    scoped=[scorer.score(r['features'],r['context']) for r in paired if r['group'] in model['holdout_groups']]
    benefit_regret=[]
    for row in model['heldout']:
        r,p=row['actual'],row['prediction'];benefit_regret.append(max(0,r)-(r if p>0 else 0))
    diagnostics=dict(estimated_quantity='dimensionless implementation-conditional relative RMSE gain',
        mean_optional_join_or_skip_regret=float(np.mean(benefit_regret)),
        ranking_unit='candidate relative benefit, not final whole-graph accuracy ranking',
        deployment_holdout_in_support=sum(s['estimated_relative_benefit'] is not None for s in scoped),deployment_holdout_total=len(scoped),
        deployment_holdout_RMSE_gain_MAE=None if not any(s['estimated_relative_benefit'] is not None for s in scoped) else 'see model report',
        domain_reason='Frozen 12-feature training-range gate abstains outside support; not relaxed after seeing holdout',
        independent_holdout_regions=1,independent_holdout_fields=2,heldout_pair_rows=12,
        conclusion='Raw RF diagnostic MAE smaller than constants; ranking regret equals global ranking. Holdout lies outside declared deployment support, so trusted scoring validation is INSUFFICIENT_EVIDENCE. No calibrated reliability or selection advantage.')
    save(OUT/'model_diagnostics.json',diagnostics)
    costs=dict(historical_evidence_audit_seconds=audit_seconds,verification_and_readback_seconds=time.perf_counter()-started-audit_seconds,new_pairs=read(OUT/'collection.json'),training_seconds=model['training_seconds'],
        runs={str(p.parent.relative_to(OUT)):dict(service=r.get('total_service_seconds'),search=r['seconds'],proposal=r['candidate_generation_seconds'],score=r.get('optional_benefit_inference_seconds'),
            preparation=r.get('preparation_costs'),counts=r['counts'],per_execution_costs=[v.get('costs') for v in r['rows'] if v.get('numerical_execution_attempted')])
            for p in sorted(OUT.glob('runs/*/summary.json'))+sorted(OUT.glob('layers/*/summary.json')) for r in [read(p)]},
        note='No speed advantage inferred; repeated acceptance clients and earlier v1/v2 runs cost extra, preserved separately')
    save(OUT/'costs.json',costs)
    save(OUT/'verification.json',dict(comparisons=comparisons,score_probability_max_change=max(changed),same_selected_on_off=on['best']['key']==off['best']['key'],
        background_preserved=True,affected_problems=affected,paired_product_sha256=products,download=archive,
        browser='NOT_RUN_THIS_ROUND; page handler and ZIP generation/readback only',
        hidden_reference='Independent DatasetReference only; public features allowlist has no targets/filename/task label; tested with altered evaluator',
        model_diagnostics=diagnostics))
    print(json.dumps(dict(comparisons=comparisons,score_max_change=max(changed),model=diagnostics,download=archive),ensure_ascii=False,indent=2))


if __name__=='__main__':run()
