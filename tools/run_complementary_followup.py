"""Development-only portrait calibration, frozen before new held-out dates."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,sys
import numpy as np
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests'),str(ROOT/'tools')]
from ecodata.algorithms import platform_search as api,attribute_registry as ar
from ecodata.algorithms.composition_response import family,scope
from ecodata.models import portrait_policy as policy
from test_full_screen_spatiotemporal import publish,prepare_case
import run_complementary_optimizer_study as study

OLD=ROOT/'output/complementary_20261024/study_v2'
OUT=ROOT/'output/complementary_20261024/study_v3'
DEV=[0,360,720,1080];TEST=[1600,1960,2320,2680]
SEEDS=[11,29,47,71,101]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
save=study.save


def calibrate():
    groups={};scope_value=None;calibration=[];all_predictions=[]
    for start in DEV:
        folder=OLD/'oracles'/str(start);matrix=load(folder/'independent_matrix.json')['rows']
        request=load(folder/'request.json');req=request['request']
        sources={}
        for item in request['visible_input_snapshots']:
            p=Path(item['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
            with xr.open_dataset(p,engine='h5netcdf') as ds:sources[item['source_id']]=ds.load()
        current_scope=scope(sources,req,req['composition']['reconstruction'])
        if scope_value is None:scope_value=current_scope
        assert scope_value==current_scope
        scale=float(sources['fine_slow'][req['variable']].std());buckets={}
        for row in matrix:
            key,alpha=family(row['plan'],'fine_slow')
            buckets.setdefault(key,[]).append((alpha,row['rmse']**2/scale**2,row['key']))
        fitted={}
        for key,values in buckets.items():
            x=np.array([v[0] for v in values]);y=np.array([v[1] for v in values])
            design=np.stack([x*x,(1-x)**2,2*x*(1-x)],axis=1)
            beta=np.linalg.lstsq(design,y,rcond=None)[0]
            moments=np.array([[beta[0],beta[2]],[beta[2],beta[1]]])
            assert np.max(abs(design@beta-y))<1e-8 and np.linalg.eigvalsh(moments).min()>-1e-9
            groups.setdefault(key,dict(second_moments=[]))['second_moments'].append(moments.tolist())
            for alpha,_,candidate in values:fitted[candidate]=float(np.sqrt(np.array([alpha,1-alpha])@moments@np.array([alpha,1-alpha])))
        calibration.append(dict(case=start,rows=matrix,predictions=fitted))
    profile=dict(schema_version=1,version='complementary-error-response-v1',training_partition='development',
        training_groups=DEV,scope=scope_value,families=groups,
        evidence='Raw error second moments including bias and cross covariance; equal group weights; normalized by visible fine-anchor standard deviation',
        training_sources=[dict(path=str(OLD/'oracles'/str(s)/'independent_matrix.json'),sha256=hashlib.sha256((OLD/'oracles'/str(s)/'independent_matrix.json').read_bytes()).hexdigest()) for s in DEV])
    # Use pooled moment means, not the same-case estimates, for prospective scoring.
    for group in calibration:
        for row in group['rows']:
            key,a=family(row['plan'],'fine_slow');v=np.array([a,1-a])
            group['predictions'][row['key']]=float(np.sqrt(v@np.mean(groups[key]['second_moments'],axis=0)@v))
            all_predictions.append(group['predictions'][row['key']])
    bad=max(all_predictions);choices=[]
    for weight in [0.,1.,2.,4.,8.,16.,32.]:
        errors=[];selected=[]
        for group in calibration:
            # The original measured score equals the sum of two half-weight risks.
            def score(row):return (2*row['score']+weight*(1-min(1,group['predictions'][row['key']]/bad)))/(2+weight)
            best=min(group['rows'],key=lambda r:(-round(score(r),9),r['key']))
            errors.append(best['rmse']);selected.append(best['key'])
        choices.append(dict(weight=weight,mean_rmse=float(np.mean(errors)),keys=selected))
    choice=min(choices,key=lambda r:(r['mean_rmse'],r['weight']))
    return profile,dict(weight=choice['weight'],bad=bad,grid=choices,selection='Lowest development grouped RMSE; ties choose smaller weight; no old/new test outcomes used')


def publish_followup(profile,parameters):
    publish()
    attrs=deepcopy(ar.load('complementary-state-v1')['attributes'])
    attrs.append(dict(id='st_expected_error',definition='Development-estimated normalized reconstruction error',tables=['data','algorithm','model'],unit='dimensionless',role='scoring',scoring_enabled=True,hard_allowed=False,provider='state_metric',property='expected_error',activation='constant_one',evidence=profile['evidence'],history=['Versioned development-only second moments; out-of-scope remains UNKNOWN']))
    api.publish_attribute_registry('complementary-state-v2',attrs,evidence=profile['evidence'])
    old=policy.load('complementary-wrf-v1')
    scenario=deepcopy(old['scenarios'][0])
    scenario['issues'].append(dict(attribute='st_expected_error',metric='expected_error',acceptable=0.,bad=parameters['bad'],importance=parameters['weight'],evidence=parameters['selection']))
    api.publish_model_portrait('complementary-wrf-v2','wrf',[scenario],evidence='Development-calibrated reconstruction proxy; no measured WRF response')


def main():
    OUT.mkdir(parents=True,exist_ok=True);ar.ROOT=OUT/'attributes';policy.ROOT=OUT/'models'
    profile,parameters=calibrate()
    profile_path=OUT/'response_portrait.json'
    if profile_path.exists():assert load(profile_path)==profile
    else:save(profile_path,profile)
    if not (ar.ROOT/'complementary-state-v2.json').exists():publish_followup(profile,parameters)
    frozen=OUT/'frozen_protocol.json'
    protocol=dict(development=DEV,fresh_test=TEST,seeds=SEEDS,parameters=parameters,
        response=dict(path=str(profile_path),sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest()),
        backends=['alns','basic','random','none'],candidate_count=56,
        fixed_key=load(OLD/'frozen_selection.json')['fixed_from_development_key'],
        original_negative_result=str(OLD/'results.json'),fresh_test_seen_at_freeze=False,
        scope='Controlled degradation of one real reanalysis; not independent sensors or WRF simulation')
    if frozen.exists():assert load(frozen)==protocol
    else:save(frozen,protocol)
    study.OUT=OUT
    results=[];fixed=[]
    for start in TEST:
        inputs,req,truth=study.data(start)
        req['scoring']['registry_version']='complementary-state-v2';req['portrait_policy']['version']='complementary-wrf-v2'
        req['composition']['response_profile']=protocol['response']
        for seed in SEEDS:
            for backend in protocol['backends']:
                report=study.run(req,inputs,truth,OUT/'test'/str(start)/backend/str(seed),backend,seed,'test')
                results.append(dict(case=start,backend=backend,seed=seed,**study.observed(report)))
        _,matrix=study.oracle(start,inputs,req,truth,'test')
        fixed.append(dict(case=start,rmse=next(r['rmse'] for r in matrix if r['key']==protocol['fixed_key']),oracle_rmse=min(r['rmse'] for r in matrix)))
        # Frozen original proxy ablation and source-only controls, same public service.
        original=deepcopy(req);original['scoring']['registry_version']='complementary-state-v1';original['portrait_policy']['version']='complementary-wrf-v1';original['composition'].pop('response_profile')
        report=study.run(original,inputs,truth,OUT/'controls'/str(start)/'original_score','alns',11,'test')
        results.append(dict(case=start,backend='original_score',seed=11,**study.observed(report)))
        for source in ['fine_slow','coarse_fast']:
            control=deepcopy(original)
            control['composition']['source_stages']['fine_slow'][0]['options']=[dict(algorithm='temporal.semantic_resample',parameters=dict(target_time_frequency='6h',temporal_interpolation='linear'))]
            control['composition']['source_stages']['coarse_fast'][0]['options']=[dict(algorithm='raster_task.spatial.bilinear')]
            control['composition']['fusion_options']=[dict(algorithm='prepare.select_source',source=source,evidence='Frozen source-only control')]
            report=study.run(control,inputs,truth,OUT/'controls'/str(start)/source,'none',11,'test')
            results.append(dict(case=start,backend=source,seed=11,**study.observed(report)))
        save(OUT/'progress.json',dict(completed_case=start,runs=results,fixed=fixed))
        print(json.dumps(dict(case=start,runs=len(results))),flush=True)
    summary={b:float(np.mean([r['rmse'] for r in results if r['backend']==b])) for b in [*protocol['backends'],'original_score','fine_slow','coarse_fast']}
    summary['fixed']=float(np.mean([r['rmse'] for r in fixed]));summary['oracle']=float(np.mean([r['oracle_rmse'] for r in fixed]))
    save(OUT/'results.json',dict(protocol=protocol,runs=results,fixed=fixed,mean_rmse=summary))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
