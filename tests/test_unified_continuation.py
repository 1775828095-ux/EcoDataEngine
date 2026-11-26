"""Best-effort orchestration, not environmental accuracy or simulator proof."""
import io
import json
import zipfile
from pathlib import Path

import pytest
import xarray as xr

from ecodata.core.continuation import decision, receipt, check_decision
from test_desktop_simple_flow import window


def test_ds_advice_cannot_waive_required_science():
    row=decision('alignment','A:temperature','unit unknown',advice=dict(action='SKIP_STEP'))
    assert row['action']=='BLOCK_STEP' and not row['advice_applied']
    optional=decision('advice','A:extra','optional plot',required=False,advice=dict(action='SKIP_STEP'))
    assert optional['action']=='SKIP_STEP' and optional['advice_applied']
    with pytest.raises(ValueError):
        decision('alignment','A','missing',recovery=dict(action='SKIP_STEP',verified=True))


def test_proposals_are_scoped_not_global_permission():
    from ecodata.desktop.discovery import blocking_checks,executable_checks
    rows=[dict(candidate='C1',action='align_space',status='通过'),
          dict(candidate='C2',action='fusion',status='条件不足',block_code='unit_unknown'),
          dict(candidate='C3',action='pending_goal',status='条件不足')]
    assert blocking_checks(rows)==[rows[1]]
    assert executable_checks(rows)==[rows[0]]
    assert check_decision(rows[2],rows)['action']=='SKIP_STEP'


def upload(name,raw):
    b=io.BytesIO(raw);b.name=name;return b


def test_bad_auxiliary_source_does_not_discard_valid_source(tmp_path):
    from contextlib import ExitStack
    from ecodata.models.project_binding import ProjectFiles
    from ecodata.models.engineering.preparation import source_fields
    good=xr.Dataset({'v':('time',[280.,281.],dict(units='K',physical_variable='air_temperature',temporal_semantics='instantaneous'))},
        coords=dict(time=['2020-01-01','2020-01-02']),attrs=dict(source_id='good',engineering_role='source'))
    bad=good.copy(deep=True);bad.attrs['source_id']='bad';bad.v.attrs.pop('units')
    audit=[]
    with ExitStack() as stack:
        files=ProjectFiles([upload('good.nc',good.to_netcdf()),upload('bad.nc',bad.to_netcdf())],stack)
        sources=source_fields(files,{},audit)
    assert [s['source_id'] for s in sources]==['good']
    failed=next(r for r in audit if r['resource']=='bad.nc')
    assert failed['continuation']['blocks_only']==['bad.nc:v']
    assert '单位' in failed['continuation']['reason']


def test_missing_metadata_in_one_variable_preserves_other_variable():
    from contextlib import ExitStack
    from ecodata.models.project_binding import ProjectFiles
    from ecodata.models.engineering.preparation import source_fields
    ds=xr.Dataset({'good':('time',[280.],dict(units='K',physical_variable='air_temperature',temporal_semantics='instantaneous')),
                   'bad':('time',[100000.],dict(physical_variable='air_pressure',temporal_semantics='instantaneous'))},
        attrs=dict(source_id='one-source',engineering_role='source'))
    audit=[]
    with ExitStack() as stack:
        sources=source_fields(ProjectFiles([upload('source.nc',ds.to_netcdf())],stack),{},audit)
    assert [s['field'] for s in sources]==['good']
    assert next(r for r in audit if r.get('field')=='bad')['used'] is False


def test_selected_operations_keep_success_and_research_failure(tmp_path,monkeypatch):
    from ecodata.desktop import selected_processing as service, pair_processing, research_dispatch
    monkeypatch.setattr(service,'assert_current',lambda scan:None)
    def process(scan,key,task,folder,model,progress):
        if key=='bad':raise ValueError('missing unit')
        return dict(files=[('actual.txt',b'actual product')],source_formats=['CSV'],target_format='Text',
            original_report={},local=dict(artifacts=[],reason='actual completed'))
    monkeypatch.setattr(pair_processing,'run',process)
    monkeypatch.setattr(research_dispatch,'run',lambda *args:(_ for _ in ()).throw(ValueError('missing reference')))
    tasks=[(key,dict(recipes=[],route='pair_processing')) for key in ('bad','good')]
    result=service.run(dict(research_tasks={'data_only':[{}]}),tasks,tmp_path,model_id='data_only')
    assert result['processing_completed'] and result['continuation']['status']=='PARTIAL'
    assert result['files']==[('group-2-actual.txt',b'actual product')]
    assert result['continuation']['blocked_steps']==['bad','研究附加步骤']
    data=service.bundle(result)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert z.read('group-2-actual.txt')==b'actual product'


def test_no_viable_operation_exports_diagnostic_not_fake_product(tmp_path,monkeypatch):
    from ecodata.desktop import selected_processing as service, pair_processing
    monkeypatch.setattr(service,'assert_current',lambda scan:None)
    monkeypatch.setattr(pair_processing,'run',lambda *args:(_ for _ in ()).throw(ValueError('unit UNKNOWN')))
    result=service.run({},[('bad',dict(recipes=[],route='pair_processing'))],tmp_path,model_id='data_only')
    assert not result['processing_completed'] and result['files']==[]
    assert result['continuation']['status']=='BLOCKED'
    with zipfile.ZipFile(io.BytesIO(service.bundle(result))) as z:
        assert 'desktop_issues.json' in z.namelist()
    from ecodata.desktop.output_review import describe_output,cards
    check=describe_output(result,None)
    assert not any(r.get('selected') or r.get('metrics') or r.get('performed') for r in cards(result,check))


@pytest.mark.parametrize('model,operation,relative,input_name,config_name',[
    ('swat_plus','soil','swat_plus/soil','A_official_valid_subset.csv','config.json'),
    ('camx','emissions','camx/emissions','A_synthetic_LATLON_hourly.csv','config_LATLON.json'),
    ('cmaq','emissions','cmaq/emissions','A_EDGAR_CO_2014_Paris.nc','config_EDGAR.json'),
    ('calpuff','precipitation','calpuff/precipitation','A_NOAA_USCRN_20210701.csv','config_NOAA.json')])
def test_project_attachment_failure_keeps_real_converted_components(tmp_path,monkeypatch,model,operation,relative,input_name,config_name):
    from ecodata.models.engineering import native
    from ecodata.algorithms.automatic_preparation import prepare_uploads
    monkeypatch.setattr(native,'finish_project',lambda *args,**kwargs:(_ for _ in ()).throw(ValueError('缺少工程运行依赖')))
    base=Path(__file__).parents[1]/'examples/ai_round1'/relative
    raw=(base/input_name).read_bytes()
    request=dict(schema='ecodata.model-task.v1',model_id=model,operation=operation,input=input_name,
        configuration=json.loads((base/config_name).read_text()))
    source=upload(input_name,raw)
    result=prepare_uploads([upload('request.json',json.dumps(request).encode()),source],model,tmp_path)
    m=result['manifest']
    assert source.getvalue()==raw and m['status']=='COMPONENT_EXPORTED'
    assert m['continuation']['status']=='PARTIAL' and 'project_writeback' in m['continuation']['blocked_steps']
    assert m['final_check']['format']['status']=='PASS'
    assert m['final_check']['project']['status']=='NOT_REQUESTED'
    assert m['final_check']['native']['status']=='NOT_RUN'
    with zipfile.ZipFile(result['archive']) as z:
        assert z.testzip() is None
        for entry in m['output_inventory']:assert z.read(entry['path'])


def test_final_receipt_never_turns_missing_model_into_conversion_failure():
    from ecodata.core.continuation import final_events
    events,done=final_events(dict(format=dict(status='PASS'),project=dict(status='NOT_REQUESTED'),
                                native=dict(status='NOT_RUN',reason='no matching executable')))
    r=receipt(events,done)
    assert r['blocked_steps']==[] and r['completed_steps']==['format']
    assert r['decisions'][0]['required'] is False


def test_bad_pressure_branch_keeps_real_temperature_search_and_writeback(tmp_path,monkeypatch):
    """Synthetic input, real public search/kernel/writeback; no mocked fitness."""
    import numpy as np
    from test_unified_engineering_ports import request
    from ecodata.algorithms import platform_search
    from ecodata.algorithms.automatic_preparation import prepare_uploads
    monkeypatch.setattr(platform_search,'CURRENT_INPUT_BUDGET',dict(objective_limit=40,proposal_limit=100,total_call_limit=200,wall_seconds=8))
    with zipfile.ZipFile(request('wrf')) as z:files={n:z.read(n) for n in z.namelist()}
    with xr.open_dataset(io.BytesIO(files['project/unusual.nc'])) as opened:native=opened.load()
    native['pressure']=(native.custom_field.dims,np.full(native.custom_field.shape,100000.,dtype='f4'),
        dict(units='Pa',physical_variable='air_pressure',temporal_semantics='instantaneous'))
    files['project/unusual.nc']=native.to_netcdf(engine='scipy',format='NETCDF3_CLASSIC')
    with xr.open_dataset(io.BytesIO(files['source.nc'])) as opened:bad=opened.load()
    bad.attrs['source_id']='pressure_source'
    bad.arbitrary_name.attrs.update(units='kg',physical_variable='air_pressure')
    files['bad_pressure.nc']=bad.to_netcdf(engine='scipy')
    result=prepare_uploads([upload(n,raw) for n,raw in files.items()],'wrf',tmp_path)
    assert result['manifest']['status']=='PROJECT_UPDATED'
    assert result['manifest']['continuation']['status']=='PARTIAL'
    assert any('air_pressure' in r for r in result['manifest']['continuation']['blocked_steps'])
    with zipfile.ZipFile(result['archive']) as z:
        with xr.open_dataset(io.BytesIO(z.read('project/unusual.nc'))) as actual:
            np.testing.assert_array_equal(actual.pressure,native.pressure)
            assert np.any(actual.custom_field.values!=native.custom_field.values)
        assert z.read('bad_pressure.nc')==files['bad_pressure.nc']
    tasks=result['manifest']['preparation']['tasks']
    assert len(tasks)==1 and tasks[0]['port']['quantity']=='air_temperature'
    assert tasks[0]['execution']['result']['selected_graph']


def test_desktop_verified_step_is_not_vetoed_by_other_candidate(window,monkeypatch):
    from test_desktop_continuation_gate import setup_recipe
    from ecodata.desktop import __main__ as desktop
    monkeypatch.setattr(desktop,'assert_current',lambda scan:None)
    row=setup_recipe(window)
    window.update_task_checks([dict(candidate='C1',action='fusion',status='通过'),
        dict(candidate='C2',action='fusion',status='条件不足',block_code='unit_unknown',reason='unit unknown')])
    assert window.original_recommendation_allowed
    calls=[]
    monkeypatch.setattr(window,'execute_now',lambda:calls.append(window.selected_recommendations))
    window.continue_recommendations()
    assert calls==[[row]]


def test_wrf_existing_writer_exports_without_full_project(tmp_path):
    from ecodata.algorithms.automatic_preparation import prepare_uploads
    base=Path(__file__).parents[1]/'examples/ai_round1/wrf/daily'
    name='A_sst_three_days.nc';raw=(base/name).read_bytes()
    req=dict(schema='ecodata.model-task.v1',model_id='wrf',operation='sst',input=name,
        configuration=json.loads((base/'config.json').read_text()))
    result=prepare_uploads([upload(name,raw),upload('task.json',json.dumps(req).encode())],'wrf',tmp_path)
    assert result['manifest']['status']=='COMPONENT_EXPORTED'
    assert any(r['path'].endswith('.wps') for r in result['manifest']['output_inventory'])
    assert result['manifest']['final_check']['project']['status']=='NOT_REQUESTED'
    assert not result['manifest']['native_model_executed']


def test_data_dispatch_attempts_independent_tasks_and_preserves_reasons(tmp_path,monkeypatch):
    from ecodata.models import task_dispatch as dispatch
    candidates=[(dict(file=name,quantity='air_temperature'),[]) for name in ('one.csv','two.csv')]
    monkeypatch.setattr(dispatch,'discover',lambda uploads:(candidates,[]))
    result=dispatch.dispatch_preparation([], 'camx',tmp_path)
    assert result['manifest']['status']=='TASK_BLOCKED'
    assert result['manifest']['continuation']['blocked_steps']==['one.csv:air_temperature','two.csv:air_temperature']
    assert result['manifest']['preparation']['tasks']==[]
    with zipfile.ZipFile(result['archive']) as z:
        handoff=json.loads(z.read('handoff.json'))
        assert len(handoff['continuation']['decisions'])==2 and z.testzip() is None
