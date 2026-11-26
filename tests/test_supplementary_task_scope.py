import numpy as np
import xarray as xr

from ecodata.desktop import fallback_evidence, recommendations
from ecodata.desktop.release_gate import allowed, selectable


def test_full_supplement_uses_common_task_domain_not_global_pixels(tmp_path,monkeypatch):
    fields=[];rows=[]
    for i,axis in enumerate((np.arange(-80.,81.),np.arange(0.,5.))):
        path=tmp_path/f'source-{i}.nc'
        xr.Dataset({'rain':(('time','lat','lon'),np.ones((2,len(axis),len(axis))))},
            coords=dict(time=np.array(['2020-01-01','2020-01-02'],dtype='datetime64[ns]'),lat=axis,lon=axis)).to_netcdf(path)
        rows.append(dict(name=path.name,format='NetCDF',ref=(path,'',None,None),bytes=path.stat().st_size))
        fields.append((i,dict(name='rain',unit='mm')))
    scoped=fallback_evidence.task_scoped_fields(dict(rows=rows),dict(route='independent_grid_search'),fields)
    assert 'evidence_indexers' not in fields[0][1]
    records=list(fallback_evidence.numerical_rows(rows[0],scoped[0][1]))
    assert len(records)==2*7*7
    assert min(r[1] for r in records)==79 and max(r[1] for r in records)==85
    assert scoped[0][1]['evidence_scope']['extent']['lat']==[0.,4.]
    assert len(list(fallback_evidence.numerical_rows(rows[1],scoped[1][1])))==50
    monkeypatch.setattr(fallback_evidence,'assert_current',lambda scan:None)
    archive=tmp_path/'evidence.zip'
    manifest=fallback_evidence.build(dict(rows=rows),scoped,archive,include_native=False)
    calls=[]
    def send(messages,tools,key,progress):
        calls.append(messages)
        return dict(content='{"recipe":"R1","reason":"functional response"}'),dict(id='not-online',usage={})
    result=fallback_evidence.transmit(archive,manifest,fallback_evidence.digest_file(archive),api_key='not-online',
        recipes=[dict(id='R1')],send=send)
    assert result['received']==result['total']==len(calls)
    assert result['candidate']=='R1'


def test_supplementary_continuation_never_inherits_exposed_confirmation():
    previous=dict(task='C1:fusion',id='R1',steps=['spatial.blend'],
        execution_release=dict(status='failed'),local_holdout_review=[dict(basis='recipe_numeric_rejected',
            acceptance_sampling=dict(failures=[dict(rule='recipe_score_below_single_source',unscored=False)]))])
    task=dict(route='independent_grid_search',mutation_reservation=dict(status='RESERVED'),mutation_confirmation_times=['2020-01-02'])
    row=recommendations.supplementary_search_continuation(previous,{('C1','fusion'):task})
    assert selectable(row) and not allowed(row)
    assert task['mutation_reservation']['status']=='INSUFFICIENT'
    assert not task['mutation_confirmation_times']
    assert row['previous_recommendation_review']==previous
    assert previous['execution_release']['status']=='failed'


def test_supplementary_does_not_admit_contract_failure():
    previous=dict(task='C1:fusion',id='R1',execution_release=dict(status='failed'),
        local_holdout_review=[dict(basis='joint_recipe_acceptance_rejected')])
    assert recommendations.supplementary_search_continuation(previous,{('C1','fusion'):dict(route='independent_grid_search')}) is None
