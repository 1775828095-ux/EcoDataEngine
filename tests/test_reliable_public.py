import json
import numpy as np
import xarray as xr
from ecodata.algorithms import platform_search as api
from test_current_input import setup


def fixture(monkeypatch,tmp_path):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=8)
    fine=fine.isel(time=[0,1,2,5,6,7])
    inputs[0]=api.SearchInput('fine.nc','primary',fine.to_netcdf())
    q['response_scoring']=dict(enabled=True)
    q['reliable_selection']=dict(mode='full',delta=.01,evidence='Synthetic engineering tolerance test, not a learned guarantee')
    roots=dict(s0=dict(operation='source',parents=[],parameters=dict(source='source0')))
    short=dict(records=dict(roots,t=dict(operation='temporal.raster_linear',parents=['s0'],parameters={})),key='t')
    long=dict(records=dict(short['records'],f=dict(operation='spatial.denoise_tv',parents=['t'],parameters=dict(noise_multiplier=1.))),key='f')
    return fine,inputs,q,dict(short=short,long=long)


def test_public_bank_same_complete_folds_real_response_and_no_prefix_injection(monkeypatch,tmp_path):
    fine,inputs,q,bank=fixture(monkeypatch,tmp_path)
    result=api.evaluate_graph_bank(inputs,q,bank,tmp_path/'bank',budget=dict(objective_limit=300,probe_call_limit=0,proposal_limit=5000,wall_seconds=60))
    assert result['status']=='PASS' and result['selected_bank_id'] in bank
    assert len(result['bank_rows'])==2 and result['reliable_selection']['parsimony_enabled']
    assert result['self_validation_protocol']['fold_count']==3
    assert result['confirmation'] is None
    ids=None
    for row in result['bank_rows']:
        assert row['status']=='PASS' and row['metrics']['complete_required_folds']
        assert row['metrics']['fold_count']==3
        if ids is None:ids=row['metrics']['fold_ids']
        assert row['metrics']['fold_ids']==ids
        with xr.open_dataset(row['product']) as ds:
            np.testing.assert_array_equal(ds.ndvi.sel(time=fine.time).values,fine.ndvi.values)
    response=json.loads((tmp_path/'bank/response_scoring.json').read_text(encoding='utf-8'))
    assert any(r['status']=='MEASURED_TASK_FOLDS' for r in response['responses'].values())
    assert result['total_cost']['used']==len(result['total_cost']['events'])


def test_public_multifold_random_same_seed_page_and_bank_not_required(monkeypatch,tmp_path):
    _,inputs,q,_=fixture(monkeypatch,tmp_path)
    q['search_stopping']=dict(enabled=True,min_evaluations=2,patience_evaluations=3,no_novel_iterations=4,min_relative_gain=.01)
    budget=dict(objective_limit=200,probe_call_limit=0,proposal_limit=8000,wall_seconds=60)
    a=api.process_current_input(inputs,q,tmp_path/'a',seed=29,budget=budget)
    from ecodata.interface.search_page import process_search_upload
    b=process_search_upload(inputs,dict(q,decision_policy='current_input'),None,tmp_path/'b',seed=29,budget=budget)
    assert a['status']==b['status']=='PASS'
    assert a['best']['key']==b['best']['key'] and a['bank_rows']==[]
    assert a['best']['metrics']['fold_count']==3
    assert a['best']['matching']['status']=='MEASURED_TASK_FOLDS'
    with xr.open_dataset(a['product']) as x,xr.open_dataset(b['product']) as y:xr.testing.assert_equal(x.load(),y.load())


def test_public_history_metrics_are_real_complete_graph_responses(monkeypatch,tmp_path):
    import hashlib
    import pandas as pd
    from test_advanced_complements import field
    from ecodata.algorithms.attribute_registry import digest
    from ecodata.algorithms import development_validation as history
    fine,inputs,q,bank=fixture(monkeypatch,tmp_path)
    amended=[]
    for item,sensor in zip(inputs,['landsat-8','terra-modis']):
        from ecodata.algorithms.platform_search import parse_inputs
        parsed,_=parse_inputs([item]);ds=parsed[0][1];ds.attrs['sensor']=sensor
        amended.append(api.SearchInput(item.filename,item.role,ds.to_netcdf()))
    cases=[]
    for year in (2013,2014):
        resources=[]
        for role,sensor in [('primary','landsat-8'),('background','terra-modis')]:
            ds=field(n=8,t=8,source=f'{role}_{year}')
            if role=='background':ds.ndvi.values+=.015
            ds=ds.assign_coords(time=pd.date_range(f'{year}-01-01',periods=8,freq='8D'))
            ds.attrs.update(sensor=sensor,evidence_role='development')
            path=tmp_path/f'{role}_{year}.nc';ds.to_netcdf(path,engine='h5netcdf')
            resources.append(dict(role=role,path=str(path),sensor=sensor,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        cases.append(dict(id=str(year),year=year,sources=resources))
    registry=dict(evidence_role='development',evidence='Matched synthetic public-entry fixtures',cases=cases)
    path=tmp_path/'published.json';path.write_text(json.dumps(dict(schema_version=1,registry=registry,sha256=digest(registry))),encoding='utf-8')
    monkeypatch.setattr(history,'DEFAULT_REGISTRY',path)
    result=api.evaluate_graph_bank(amended,q,dict(short=bank['short']),tmp_path/'historical_bank',
        budget=dict(objective_limit=300,probe_call_limit=0,proposal_limit=5000,wall_seconds=60))
    assert result['status']=='PASS'
    protocol=result['self_validation_protocol']
    assert protocol['history_validation']['status']=='MATCHED_DEVELOPMENT_SIMULATIONS'
    metrics=result['best']['metrics']
    assert metrics['fold_count']==5 and metrics['objective_fold_count']==2
    assert len(metrics['diagnostic_fold_ids'])==3
    assert all(r['status']=='PASS' for r in result['best']['fold_responses'])
    assert metrics['portrait_response']['status']=='MEASURED_MULTI_FOLD'


def test_page_identity_changes_with_published_matching_policy(monkeypatch,tmp_path):
    from ecodata.algorithms import reliable_selection as selection
    from ecodata.algorithms import development_validation as history
    from ecodata.interface.search_page import request_fingerprint
    p=tmp_path/'policy.json';r=tmp_path/'registry.json'
    monkeypatch.setattr(selection,'POLICY',p);monkeypatch.setattr(history,'DEFAULT_REGISTRY',r)
    original=request_fingerprint([],{'variable':'ndvi'})
    p.write_text('{}')
    published=request_fingerprint([],{'variable':'ndvi'})
    assert original!=published
    r.write_text('{}')
    assert request_fingerprint([],{'variable':'ndvi'})!=published
