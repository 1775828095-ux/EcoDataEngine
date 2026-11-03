"""Independent artifact readback; supplemental mixed-gap engineering acceptance."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
import pandas as pd
import xarray as xr

import run_robit_single_source as study
from ecodata.algorithms import platform_search as api
from ecodata.models.adapters.swat_plus import SWATPlusAdapter

OUT=study.OUT


def read_json(path):return json.loads(path.read_text(encoding='utf-8'))


def mixed_acceptance():
    study.stores();folder=OUT/'mixed_gap_engineering_acceptance'
    before=hashlib.sha256((OUT/'frozen_validation_selections.json').read_bytes()).hexdigest()
    frame=study.raw_weather();part=frame[frame.time.between('1996-01-01','1997-12-31')].copy()
    ds=study.dataset(part);mask=np.zeros(len(part),bool)
    for start,size in [(25,1),(80,3),(130,14),(240,1),(300,3),(380,14),(500,3),(600,14)]:mask[start:start+size]=True
    ds.precipitation.values[mask]=np.nan
    chain=[dict(algorithm='reconstruction.daily_linear',parameters=dict(max_gap_days=3)),
           dict(algorithm='reconstruction.daily_month_mean',parameters={})]
    assert chain in study.plans()
    req=study.base_request(ds,[chain]);req['composition']['search'].update(backend='basic',generations=0,survivors=1)
    result=api.submit_search(study.inputs_for(ds),req,None,folder,mode='Profile',budget=dict(objective_limit=1,wall_seconds=60))
    assert result['best'] and result['best']['file_readback']=='PASS'
    trace=[t for t in result['best']['trace'] if t.get('algorithm','').startswith('reconstruction.daily_')]
    assert len(trace)==2
    filled=[int(round((t['before']['missing_fraction']-t['after']['missing_fraction'])*len(part))) for t in trace]
    assert min(filled)>0 and sum(filled)==int(mask.sum())
    with xr.open_dataset(result['best']['product']) as file:product=file.load()
    values=product.precipitation.values[:,0,0]
    np.testing.assert_array_equal(values[~mask],part.precipitation.to_numpy()[~mask])
    assert np.isfinite(values).all() and (values>=0).all()
    part['precipitation']=values
    config=dict(project_name='Robit_mixed_acceptance',start='1996-01-01',end='1997-12-31',
        station=dict(name='Robit',latitude=11.6,longitude=37.36,elevation=1790),weather=part.to_dict('list'),
        target_model_version='62.0.0',additional_weather=['solar_radiation','relative_humidity','wind_speed'])
    export=SWATPlusAdapter().save_package(folder/'swatplus_weather',config)
    reread=pd.read_csv(export/'station.pcp',sep=r'\s+',skiprows=3,header=None)
    np.testing.assert_allclose(reread[2],values,atol=.00051,rtol=0)
    assert before==hashlib.sha256((OUT/'frozen_validation_selections.json').read_bytes()).hexdigest()
    study.save(folder/'engineering_acceptance.json',dict(status='PASS',scope='Supplemental training-period mechanism check; not an independent effect test',
        stage_filled_days=filled,total_missing=int(mask.sum()),raw_observations_unchanged=True,
        public_entry='platform_search.submit_search',weather_readback='PASS',native_SWAT_run=False,
        frozen_validation_selection_sha256=before))


def audit():
    frozen=read_json(OUT/'frozen_validation_selections.json');test=read_json(OUT/'test_results.json')
    rain=pd.read_csv(OUT/'source/robpcp.txt',skiprows=1,header=None)[0].to_numpy()
    times=pd.date_range('1990-01-01',periods=len(rain));raw=pd.Series(rain,index=times)
    prior=xr.open_dataset(OUT/'training_state.nc').load()
    fit=raw.loc['1990-01-01':'1995-12-31']
    np.testing.assert_allclose(prior.month_mean,fit.groupby(fit.index.month).mean())
    assert abs(float(prior.scale)-float(fit.std(ddof=0)))<1e-12
    executions=0;failures=0;probes=0;score_intervals=[];numeric={}
    for file in sorted((OUT/'development').glob('*/summary.json'))+sorted((OUT/'search').glob('*/summary.json')):
        report=read_json(file);probes+=report.get('parallel_scoring',{}).get('probe_graph_executions',0)
        numeric[str(file.parent)]=len({r['numeric_fingerprint'] for r in report['rows'] if r['status']=='PASS'})
        labels=xr.open_dataset(file.parent/'search_validation.nc').load().precipitation
        visible_path=Path(read_json(file.parent/'request.json')['visible_input_snapshots'][0]['path'])
        visible=xr.open_dataset(visible_path).load().precipitation
        assert np.isnan(visible.values[np.isfinite(labels.values)]).all()
        for row in report['rows']:
            executions+=1
            if row['status']!='PASS':failures+=1;continue
            path=Path(row['product']);assert hashlib.sha256(path.read_bytes()).hexdigest()==row['product_sha256']
            values=xr.open_dataset(path).load().precipitation
            m=np.isfinite(labels.values);error=values.values[m]-labels.values[m]
            assert abs(float(np.sqrt(np.mean(error*error)))-row['search_validation']['RMSE'])<1e-10
            finite=np.isfinite(visible.values)
            np.testing.assert_array_equal(values.values[finite],visible.values[finite])
        scoring=file.parent/'parallel_scoring.json'
        if scoring.exists():
            receipt=read_json(scoring)
            for probe in receipt['probes']:
                assert probe['hidden_validation_or_test_labels_accessed'] is False
                hold=np.array(probe['withheld_indices']);assert np.isfinite(visible.values[:,0,0][hold]).all()
                np.testing.assert_allclose(probe['truth'],visible.values[:,0,0][hold])
            screen=read_json(file.parent/'initial_screening.json')
            score_intervals.append(len({tuple(r['score_interval']) for r in screen['rows']}))
    tested_products={};fixed=read_json(OUT/'fixed_selection.json')
    for row in test:
        path=Path(row['product']);ds=xr.open_dataset(path).load()
        summary=read_json(path.parents[2]/'summary.json')
        request=read_json(path.parents[2]/'request.json')
        if row['method'].startswith('fixed_'):
            expected=[dict(algorithm='reconstruction.daily_'+row['method'][6:],parameters={})]
        elif row['method']=='development_best_fixed_combo':
            expected=next(x['choice'] for x in fixed['best_graph_plan'] if x['branch']=='Robit_station')
        else:
            selected=next(x for x in frozen if (x['case'],x['method'],x['seed'])==(row['case'],row['method'],row['seed']))
            expected=next(x['choice'] for x in selected['plan'] if x['branch']=='Robit_station')
        assert expected==row['graph']
        assert (OUT/'frozen_validation_selections.json').stat().st_mtime<=path.stat().st_mtime
        source=xr.open_dataset(request['visible_input_snapshots'][0]['path']).load()
        missing=~np.isfinite(source.precipitation.values[:,0,0]);truth=raw.reindex(pd.to_datetime(ds.time.values)).to_numpy()
        predicted=ds.precipitation.values[:,0,0]
        error=predicted[missing]-truth[missing]
        assert abs(float(np.sqrt(np.mean(error*error)))-row['metrics']['RMSE'])<1e-10
        np.testing.assert_array_equal(predicted[~missing],truth[~missing])
        assert summary['reference_used_for_selection'] is False
        assert not (path.parents[2]/'search_validation.nc').exists()
        package=Path(row['weather_output']);actual=pd.read_csv(package/'station.pcp',sep=r'\s+',skiprows=3,header=None)
        np.testing.assert_allclose(actual[2],predicted,atol=.00051,rtol=0)
        days=pd.to_datetime(actual[0].astype(str),format='%Y')+pd.to_timedelta(actual[1]-1,unit='D')
        assert days.tolist()==list(pd.to_datetime(ds.time.values))
        tested_products[str(path)]=True
    assert failures==0 and min(score_intervals)>1
    result=dict(status='PASS',development_and_validation_graph_executions=executions,
        visible_probe_graph_executions=probes,failed_graph_executions=failures,
        distinct_test_products_readback=len(tested_products),test_rows=len(test),
        score_interval_distinct_counts=score_intervals,observations_preserved=True,training_state_verified=True,
        probe_truth_from_visible_days_only=True,validation_labels_absent_from_processing_inputs=True,
        test_graphs_frozen_before_test=True,SWAT_precipitation_values_and_dates_readback=True,
        native_SWAT_run=False,distinct_numerical_products_per_run=numeric,
        limitations=['Three time/scenario cases, not independent basins',
            'Pure gap-length strata often collapse multiple structural graphs to the same numeric product',
            'Model score is declared input suitability and probe volume bias, not measured SWAT response',
            'Supplemental mixed-gap check uses development years and does not replace held-out negative results'])
    study.save(OUT/'independent_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='distinct_numerical_products_per_run'},ensure_ascii=False,indent=2))


def replay_probes():
    """Replay frozen visible-only probes, without rerunning selection or using test."""
    study.stores()
    from ecodata.algorithms.composition_decision import Composition
    calls=0
    for file in sorted((OUT/'search').glob('*/parallel_scoring.json')):
        saved=read_json(file);protocol=read_json(file.parent/'request.json');parsed=[]
        for item in protocol['visible_input_snapshots']:
            with xr.open_dataset(item['path']) as ds:parsed.append(('primary',ds.load()))
        for item in protocol['resource_snapshots']:
            with xr.open_dataset(item['path']) as ds:parsed.append((item['role'],ds.load()))
        engine=Composition(parsed,protocol['request'],api.platform_registry());engine.prepare_parallel_scoring()
        replay=engine.parallel_scorer.receipt();calls+=replay['probe_graph_executions']
        truth=np.asarray(replay['probes'][0]['truth'])
        old={r['key']:r for r in saved['responses']}
        for row in replay['responses']:
            prediction=np.asarray(row['probe_prediction']);rmse=float(np.sqrt(np.mean((prediction-truth)**2)))
            assert abs(rmse-row['anchor_rmse'])<1e-12
            assert abs(row['anchor_rmse']-old[row['key']]['anchor_rmse'])<1e-12
            assert row['routes']==old[row['key']]['routes']
        study.save(file.parent/'replayed_probe_receipt.json',replay)
    study.save(OUT/'probe_replay_audit.json',dict(status='PASS',actual_probe_graph_reexecutions=calls,
        same_errors_and_routes_as_frozen_search=True,test_accessed=False,selection_rerun=False,
        scientific_benefit_claim=False,cost_scope='Additional audit executions; not counted as original search advantage'))
    print('Frozen visible-probe replay PASS',calls)


if __name__=='__main__':
    if sys.argv[1]=='mixed':mixed_acceptance()
    elif sys.argv[1]=='audit':audit()
    elif sys.argv[1]=='replay':replay_probes()
