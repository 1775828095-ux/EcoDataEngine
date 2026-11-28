"""Shared desktop behavior and native trial dispatch; no accuracy claims."""
import os
from pathlib import Path
from uuid import uuid4

import pytest
from test_desktop_simple_flow import window


def test_windows_secure_credential_roundtrip():
    from ecodata.core import credentials
    if os.name!='nt':pytest.skip('Windows credential backend')
    target='EcoDataEngine/tests/'+str(uuid4())
    try:
        assert credentials.load(target)==''
        credentials.save('synthetic-test-value',target)
        assert credentials.load(target)=='synthetic-test-value'
    finally:credentials.forget(target)
    assert credentials.load(target)==''


def test_verified_key_is_saved_but_failed_verification_is_not(window,monkeypatch):
    from ecodata.core import credentials
    from ecodata.desktop import service
    saved=[];monkeypatch.setattr(credentials,'save',saved.append)
    monkeypatch.setattr(window,'work',lambda fn,done:done(fn(lambda _:None)))
    monkeypatch.setattr(service,'verify_api_key',lambda key:None)
    window.api_key.setText('synthetic-new-key');window.connect_api()
    assert saved==['synthetic-new-key'] and window.saved_api_key=='synthetic-new-key'
    assert window.api_key.text()=='' and '安全记住' in window.api_status.text()
    def failed(key):raise ValueError('连接不通过')
    monkeypatch.setattr(service,'verify_api_key',failed)
    window.api_key.setText('synthetic-bad-key');window.connect_api()
    assert saved==['synthetic-new-key']


def test_saved_key_restores_and_edit_remains_open(window,monkeypatch):
    monkeypatch.delenv('DEEPSEEK_API_KEY')
    window.saved_api_key='synthetic-saved-key'
    assert window.effective_api_key()=='synthetic-saved-key'
    window.edit_api_key();window.api_key.setText('replacement')
    assert not window.api_editor.isHidden()
    assert '密钥' in window.api_badge.text()


def test_live_progress_is_on_first_page_and_bounded(window):
    window.auto_run=True;window.tabs.setCurrentIndex(0)
    window.show_progress('检查字段、单位与维度 · source.nc')
    assert 'source.nc' in window.workflow_status.text()
    assert not window.workflow_steps.isHidden()
    assert '检查字段' in window.workflow_steps.toPlainText()
    for i in range(30):window.show_progress(f'真实进度事件{i}')
    assert window.workflow_steps.document().blockCount()<=12


def test_other_model_context_does_not_hash_input(tmp_path,monkeypatch):
    import json
    from ecodata.models import input_context
    store=tmp_path/'contexts';store.mkdir()
    (store/'unrelated.json').write_text(json.dumps({'model_id':'other'}))
    monkeypatch.setattr(input_context,'STORE',store)
    source=tmp_path/'input.zip';source.write_bytes(b'input')
    monkeypatch.setattr(input_context.hashlib,'file_digest',lambda *a:pytest.fail('Unnecessary full input hash'))
    assert input_context.find(source,'camx') is None


def test_station_compare_uses_joint_reader_not_generic_field_sampling():
    from ecodata.desktop.recipe_trials import required
    assert required({'route':'station_grid','action':'compare_station_grid'})
    assert required({'route':'station_grid','action':'fusion'})
    assert not required({'route':'pair_processing','action':'convert_units'})


def test_wind_component_not_silently_treated_as_speed():
    from ecodata.desktop.station_grid import task_for
    grid={'format':'NetCDF'};obs={'format':'CSV','fields':[{'name':n} for n in ['station','lon','lat','time']]}
    field={'name':'u10','dimensions':['time','x','y'],'unit':'m/s'}
    station={'name':'WND.wind_speed','unit':'m/s'}
    with pytest.raises(ValueError,match='风分量不能'):
        task_for({'rows':[grid,obs]},((0,'A',field),(1,'B',station)),'fusion',
                 {'status':'equivalent','quantity':'wind_speed'})


def test_camx_temperature_header_and_noaa_units_match_without_online_hint():
    from ecodata.desktop.discovery import pair_units

    camx = dict(name="t2", long_name="temperature at 2 m", unit="K")
    noaa = dict(name="TMP.air_temperature", standard_name="air_temperature", unit="degC")
    report = pair_units({}, camx, {}, noaa)
    assert report["quantity"] == "air_temperature"
    assert report["status"] == "convertible"
    assert report["offset"] == pytest.approx(273.15)
    assert pair_units({}, dict(camx, long_name="standard deviation of temperature at 2 m"), {},
                      noaa)["status"] == "physical_identity_unconfirmed"
    assert pair_units({}, dict(camx, standard_name="surface_temperature"), {},
                      noaa)["status"] == "quantity_conflict"


def test_comparison_does_not_fit_an_unrequested_bias(monkeypatch):
    from types import SimpleNamespace
    import pandas as pd
    from ecodata.desktop import station_grid as sg,indexed_table_recipe,recipe_trials
    source=SimpleNamespace(held={'A'},groups={'A':[]},category='station',
                           header=['value'],read=lambda key:[[1]]*6)
    monkeypatch.setattr(indexed_table_recipe,'table',lambda *args:source)
    frame=pd.DataFrame({'background':[10.,11.,12.,13.,14.,15.],'observed':[9.,10.,11.,12.,13.,14.]})
    monkeypatch.setattr(sg,'load',lambda *a,**kw:{sg.EXACT:frame})
    monkeypatch.setattr(sg,'corrected_frame',lambda *a:pytest.fail('Comparison must not fit correction'))
    task=dict(route='station_grid',action='compare_station_grid',table_index=0,
              table_name='value',obs_unit='K',quantity='air_temperature',unit='K')
    scan={'rows':[dict(format='CSV',fields=[dict(name='value')])]}
    report=recipe_trials.review(scan,'C1:compare_station_grid',task,[sg.EXACT,sg.MATCH])
    assert report['status']=='checked' and report['count']==6
    assert report['sampled_scores'][0]['candidate']['RMSE']==report['sampled_scores'][0]['baseline']['RMSE']


def test_joint_reader_failure_is_not_automatically_passed(monkeypatch):
    from ecodata.desktop import recipe_trials,station_grid,release_gate
    def fail(*a,**kw):raise ValueError('missing actual time axis')
    monkeypatch.setattr(station_grid,'review',fail)
    task=dict(route='station_grid',action='compare_station_grid',unit='K',quantity='air_temperature')
    review=recipe_trials.review({},'C1:compare_station_grid',task,[])
    assert review['status']=='pending' and 'time axis' in review['reason']
    row=dict(task='C1:compare_station_grid',steps=['station.time.exact'],
             local_holdout_review=[dict(review,field='F')])
    row['execution_release']=release_gate.assess({'rows':[{'format':'NetCDF','name':'input'}]},row,task,
                                               [(0,'F',{'name':'temperature','unit':'K'})])
    assert not release_gate.selectable(row)


def test_descriptive_json_does_not_count_as_actual_source_header():
    import json
    from types import SimpleNamespace
    from ecodata.desktop.file_agent import uninspected_described_sources
    tools=SimpleNamespace(rows={'observation.csv':{'format':'CSV'}},
        documents={'metadata.json':json.dumps({'file':'X:/not-local/grid.nc','variables':{'t':{'units':'K'}}})},
        document_locators={'metadata.json':'docs/metadata.json'})
    assert uninspected_described_sources(tools)==[{'described_file':'grid.nc','evidence_document':'docs/metadata.json'}]
    tools.rows['package.zip!/inputs/grid.nc']={'format':'NetCDF','fields':[{'name':'t'}]}
    assert uninspected_described_sources(tools)==[]


def test_unrelated_inspection_failure_cannot_veto_checked_plan(window,monkeypatch):
    from test_step2_clear_flow import rows
    monkeypatch.setattr(window,'original_background_ready',lambda:False)
    window.render_recommendations({'recommendations':[rows()[-1]]})
    window.tabs.setCurrentIndex(0)
    window.recommendation_review_busy=True
    window.refresh_recommendation_action()
    window.continue_automatically({'failures':[{'reason':'unrelated description unreadable','batch':1}]},[])
    assert window.tabs.currentIndex()==1
    assert window.recommendation_checks[0].isChecked()


def test_compound_observation_sampling_reuses_codec_and_quality(tmp_path):
    import csv
    from ecodata.desktop import sampling_plan,sampling_risk
    station=next(str(i) for i in range(1000) if sampling_plan.station_reserved(str(i)))
    path=tmp_path/'observations.csv'
    with path.open('w',newline='') as file:
        writer=csv.writer(file);writer.writerow(['STATION','DATE','TMP'])
        for i,cell in enumerate(['+0123,1','+9999,1','+0345,3']):
            writer.writerow([station,f'2020-01-01T0{i}:00:00',cell])
    row={'name':path.name,'format':'CSV','ref':(path,'',None,None),'bytes':path.stat().st_size}
    field=dict(name='TMP.air_temperature',source_column='TMP',noaa_component=True,unit='degC')
    held=sampling_plan.sample(row,field,holdout=True)
    assert 12.3 in held['values'] and 34.5 not in held['values'] and 999.9 not in held['values']
    review=sampling_risk.review_field(row,field,basis=held['basis'],partition=held['partition'])
    assert review['status']=='checked'


def test_automatic_supplement_budget_rejects_large_copy_before_network(tmp_path):
    from ecodata.desktop import fallback_evidence as fallback
    manifest=[{'parts':[{}]*362}]
    budget=fallback.automatic_budget(manifest)
    assert not budget['allowed'] and budget['parts']==362
    assert fallback.automatic_budget([{'parts':[{}]*4}])['allowed']
    assert not fallback.automatic_budget([{'parts':[{}]*2}],used_requests=3)['allowed']
    with pytest.raises(ValueError,match='联网预算'):
        fallback.transmit(tmp_path/'not-even-opened.zip',manifest,'unused',api_key='unused',
                          automatic=True,send=lambda *a:pytest.fail('Budget must stop network'))


def test_over_budget_supplement_does_not_block_other_ready_plan(window,monkeypatch):
    from ecodata.desktop import fallback_evidence as fallback
    from test_step2_clear_flow import rows
    failed,_,ready=rows();failed['automatic_fallback_started']=True
    window.scan={'rows':[{'name':'source','format':'CSV'}]};window.gated={}
    monkeypatch.setattr(window,'original_background_ready',lambda:False)
    window.render_recommendations({'recommendations':[failed,ready]})
    monkeypatch.setattr(fallback,'eligible',lambda row:True)
    monkeypatch.setattr(fallback,'approved_fields',lambda *a:[(0,{'name':'value','unit':'K'})])
    monkeypatch.setattr(fallback,'task_scoped_fields',lambda scan,task,fields:fields)
    def build(scan,fields,archive,*a,**kw):
        archive.write_bytes(b'bounded-test-copy')
        return [{'records':100,'bytes':100,'parts':[{}]*362}]
    monkeypatch.setattr(fallback,'build',build)
    monkeypatch.setattr(fallback,'digest_file',lambda *a:'test')
    monkeypatch.setattr(fallback,'transmit',lambda *a,**kw:pytest.fail('No oversized automatic send'))
    monkeypatch.setattr(window,'work',lambda fn,done,**kw:done(fn(lambda _:None)))
    window.recommendation_review_busy=True
    window.review_full_fallback(failed,automatic=True)
    assert not window.recommendation_review_busy and window.search_button.isEnabled()
    assert failed['execution_release']['status']=='failed'
    assert failed['supplementary_budget']['allowed'] is False
