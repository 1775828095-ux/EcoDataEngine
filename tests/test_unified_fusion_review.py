"""Task-adaptive mechanism checks; synthetic inputs are not accuracy evidence."""
import json
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ecodata.algorithms import platform_search as api
from ecodata.evaluation.blocked_confirmation import review,engineering_review


def reference(shift=0,days=12,start='2000-01-01'):
    a=xr.DataArray(np.full((days,8,8),10.),dims=('time','lat','lon'),
        coords=dict(time=pd.date_range(start,periods=days),lat=np.arange(8.)+shift,lon=np.arange(8.)+shift),
        attrs=dict(units='K',split='search_validation'),name='t')
    return a,a.to_dataset().copy(deep=True)


@pytest.mark.parametrize('axes',[['space'],['space','time']])
def test_spatial_and_joint_review_require_actual_support_isolation(axes):
    a,x=reference();b,y=reference(shift=20,start='2001-01-01' if 'time' in axes else '2000-01-01')
    selection=api.BlockedSearchValidation(a,'t',axes=axes);confirmation=api.BlockedReference(b,'t',axes=axes)
    base=x.copy(deep=True);base.t.values+=2
    held_base=y.copy(deep=True);held_base.t.values+=2
    result=review(selection(x),selection(base),confirmation(y),confirmation(held_base),quality_pass=True)
    assert result['recommended']
    overlap=api.BlockedReference(a,'t',axes=axes)
    with pytest.raises(ValueError,match='overlap|disjoint'):review(selection(x),selection(base),overlap(x),overlap(base),quality_pass=True)


def test_weighted_fusion_only_reports_actual_engineering_checks():
    ref,ds=reference();ds['uncertainty']=xr.ones_like(ds.t)
    receipt=engineering_review(ds,'t')
    assert receipt['status']=='ENGINEERING_ONLY' and receipt['metrics']['coverage']==1
    assert not receipt['accuracy_recommended'] and not receipt['recommended']
    ds.uncertainty.values[0]= -1
    assert engineering_review(ds,'t')['status']=='ENGINEERING_FAILED'


def test_public_page_and_toolkit_cannot_omit_review(monkeypatch,tmp_path):
    from ecodata.interface.search_page import process_search_upload
    a,ds=reference();path=tmp_path/'input.nc';ds.to_netcdf(path)
    q=dict(task='multisource',variable='t',unit='K',semantic_type='continuous')
    calls=[]
    def backend(inputs,request,evaluation,folder,**options):
        calls.append(options['mode']);return dict(status='PASS',product=None)
    monkeypatch.setattr(api,'_submit_search',backend)
    page=process_search_upload([api.SearchInput('input.nc','primary',path.read_bytes())],q,None,tmp_path/'page')
    toolkit=api.submit_file_request(dict(inputs=[dict(path=str(path),role='primary')],request=q),tmp_path/'toolkit')
    assert calls==['PortraitMutation','PortraitMutation']
    assert page['recommendation_review']==toolkit['recommendation_review']
    assert not page['recommendation_review']['recommended']


def test_spatial_backend_publishes_common_review_and_frozen_graph(tmp_path):
    from test_spatial_relationship_fusion import fixture,BUDGET
    inputs,q,_=fixture('station_grid',True)
    result=api.submit_search(inputs,q,None,tmp_path/'search',budget=BUDGET,seed=11)
    assert result['status']=='PASS' and result['review_service'].endswith('blocked_confirmation')
    assert result['recommendation_review']['axes']==['space','time']
    assert (tmp_path/'search/selection_freeze.json').exists()
    assert (tmp_path/'search/recommendation_review.json').exists()
    assert result['delivered_graph']['key'] in result['delivered_graph']['records']


def test_outer_reservation_removes_alias_labels_and_features(monkeypatch,tmp_path):
    from ecodata.algorithms.task_validation import _fold,reserve_review
    import ecodata.algorithms.task_validation as planner
    from test_series_fusion import data,context
    ds=data(np.arange(24.)+1);raw=ds.to_netcdf()
    inputs=[api.SearchInput('a.nc','primary',raw),api.SearchInput('alias.nc','background',raw)]
    sources=[(0,'primary',ds),(1,'background',ds)]
    q=dict(task='multisource',variable='precipitation',unit='mm/day',semantic_type='continuous')
    folds=[]
    for start in [2,8,14]:
        mask=np.zeros(ds.precipitation.shape,bool);mask[start:start+2]=True
        folds.append(_fold(inputs,sources,0,ds,mask,q,'matched_temporal_block',{}))
    def replan(masked,request,ctx):
        parsed,_=api.parse_inputs(masked)
        assert np.isnan(parsed[0][1].precipitation.values[14:16]).all()
        return folds[:2],{}
    monkeypatch.setattr(planner,'plan',replan)
    inner,outer,profile,receipt=reserve_review(folds,inputs,q,context(ds,ds))
    assert receipt['status']=='RESERVED' and len(inner)==2
    for group in [profile,*[f['inputs'] for f in inner],outer['inputs']]:
        for _,source in api.parse_inputs(group)[0]:
            assert np.isnan(source.precipitation.values[14:16]).all()
    assert np.isfinite(outer['validation'].reference.values).all()


def test_application_variable_name_is_not_confused_with_declared_quantity():
    from ecodata.algorithms.application_contract import prepare
    from ecodata.algorithms.profile_matching import validate_model_context
    from ecodata.algorithms.contracts import AlgorithmContext
    from ecodata.core.schemas import EcoDataBundle
    ref,ds=reference();ds.t.attrs['physical_variable']='air_temperature'
    q=dict(task='multisource',variable='t',physical_variable='air_temperature',unit='K',semantic_type='continuous',soft_guidance=False,
        application_contract=dict(id='generic',version='1',field='t',unit='K',semantic_type='continuous',valid_range=None,issues=[],evidence='Explicit physical quantity'))
    _,model=prepare(q);context=AlgorithmContext((EcoDataBundle(ds),),'t','continuous')
    validate_model_context(context,model)
    ds.t.attrs['physical_variable']='precipitation_amount'
    with pytest.raises(ValueError,match='physical variable'):validate_model_context(context,model)


def test_internal_multisource_recovery_executes_reserved_outer_case(monkeypatch,tmp_path):
    from test_current_input import setup
    fine,coarse,inputs,q=setup(monkeypatch,tmp_path,t=180)
    clock=pd.DatetimeIndex(np.concatenate([pd.date_range(str(y)+'-01-01',periods=60).values for y in (2018,2019,2020)]))
    for ds in [fine,coarse]:ds.coords['time']=clock
    fine.ndvi.values[:]=fine.ndvi.values[0]
    coarse.ndvi.values[:]=fine.ndvi.values[0]+.015
    fine.ndvi.values[10:12]=np.nan
    inputs=[api.SearchInput('fine.nc','primary',fine.to_netcdf()),api.SearchInput('background.nc','background',coarse.to_netcdf())]
    q.update(target_times=[str(t) for t in clock],current_input_validation=dict(max_folds=4))
    q['parameters']['raster_target_times']=q['target_times']
    result=api.submit_search(inputs,q,None,tmp_path/'run',budget=dict(objective_limit=80,total_call_limit=120,proposal_limit=180,wall_seconds=60),seed=11)
    reservation=result['self_validation_protocol']['review_reservation']
    assert reservation['status']=='RESERVED',reservation
    gate=result['recommendation_review']
    assert 'evidence' in gate,gate
    assert gate['confirmation_used_for_search'] is False
    assert gate['execution'] and all(r['status']=='PASS' and r['trace'] and r['metrics']['RMSE']>=0 for r in gate['execution'])
    # Source-only graphs execute real boundary nodes without a scientific-kernel call.
    assert 'validation_audit' in result['total_cost']['phases'] or all(r['trace'][0]['operation']=='source' for r in gate['execution'])
    assert (tmp_path/'run/selection_freeze.json').exists()


def test_review_rejects_truncated_baseline_block_results():
    from copy import deepcopy
    from ecodata.evaluation.blocked_confirmation import compare
    a,ds=reference();evaluator=api.BlockedSearchValidation(a,'t')
    actual=evaluator(ds);bad=deepcopy(actual);bad['block_metrics'].pop()
    with pytest.raises(ValueError,match='block counts'):compare(actual,bad)


def test_parent_station_delivery_reports_actual_child_fallback(tmp_path):
    from ecodata.evaluation.blocked_confirmation import finalize,unavailable
    child=dict(delivery_status='BASELINE_RETAINED',recommendation_review=unavailable('Few dates'))
    result=finalize(dict(stations=[child]),{},tmp_path)
    assert result['delivery_status']=='BASELINE_RETAINED'
    assert not result['recommendation_review']['recommended']
