"""Synthetic contract/selection tests, not environmental accuracy evidence."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ecodata.algorithms.cross_scale import align_variables
from ecodata.core.exceptions import SchemaError
from ecodata.algorithms.spatial_selection_protocol import blocked_validation


def field(name, n, unit):
    axis=np.linspace(.125,.875,n)
    ds=xr.Dataset({name:(('lat','lon'),np.ones((n,n))*3)},coords=dict(lat=axis,lon=axis),attrs=dict(crs='EPSG:4326',source_id=name))
    ds[name].attrs.update(units=unit,data_kind='continuous',spatial_semantics='cell_mean')
    return ds


def test_cross_variable_preserves_units_and_does_not_blend():
    a,b=field('temperature',8,'K'),field('rainfall',4,'mm')
    outputs,report=align_variables([a,b],['temperature','rainfall'])
    assert set(outputs[0].data_vars)=={'temperature'} and set(outputs[1].data_vars)=={'rainfall'}
    assert outputs[0].temperature.attrs['units']=='K'
    assert outputs[1].rainfall.attrs['units']=='mm'
    xr.testing.assert_equal(outputs[0].lat,outputs[1].lat)
    assert not report['physical_variables_blended'] and not report['reference_accuracy_assessed']
    assert all(np.allclose(ds[list(ds.data_vars)[0]],3) for ds in outputs)


@pytest.mark.parametrize('attr,value',[('units',''),('data_kind','unknown'),('spatial_semantics','cell_total')])
def test_unknown_and_cell_total_not_guessed(attr,value):
    a,b=field('a',8,'K'),field('b',4,'m')
    a.a.attrs[attr]=value
    with pytest.raises(SchemaError):align_variables([a,b],['a','b'])


def test_same_grid_is_identity_and_other_variable_does_not_change():
    a,b=field('a',4,'K'),field('b',4,'m')
    outputs,report=align_variables([a,b],['a','b'])
    assert all(r['operations'][0]['method']=='identity' for r in report['variables'])
    a.a.values[:]=8
    changed,_=align_variables([a,b],['a','b'])
    xr.testing.assert_equal(outputs[1],changed[1])


def test_cross_variable_explicit_target_and_time_support():
    a,b=field('a',8,'K'),field('b',4,'mm')
    for ds in [a,b]:
        ds.update({v:ds[v].expand_dims(time=pd.date_range('2000-01-01',periods=3)) for v in ds.data_vars})
        ds[list(ds.data_vars)[0]].attrs['temporal_semantics']='instantaneous'
    out,_=align_variables([a,b],['a','b'],target_grid=dict(crs='EPSG:4326',latitude_step=.1,longitude_step=.1,extent='intersection'))
    assert out[0].sizes==out[1].sizes
    b.b.attrs['temporal_semantics']='accumulation'
    with pytest.raises(SchemaError,match='bounds'):align_variables([a,b],['a','b'])


def test_blocked_loss_does_not_reward_dense_region_or_sources():
    y=np.zeros((3,10));p=np.zeros_like(y);p[:,-1]=10
    groups={str(i):0 if i<9 else 1 for i in range(10)}
    r=blocked_validation(p,y,pd.date_range('2000-01-01',periods=3),range(10),groups)
    assert r['objective']==5 and r['block_count']==6 and r['source_bonus']==0
    assert r['pooled']['RMSE']<r['objective']
    assert r['robustness_status']=='MULTIPLE_DATES_AND_REGIONS'
    p[0,0]=np.nan
    with pytest.raises(ValueError,match='Incomplete'):blocked_validation(p,y,pd.date_range('2000-01-01',periods=3),range(10),groups)


def test_hours_are_not_independent_dates_and_outer_labels_not_required():
    r=blocked_validation(np.ones((4,2)),np.zeros((4,2)),pd.date_range('2000-01-01',periods=4,freq='h'),[1,2],{'1':0,'2':1})
    assert r['date_count']==1 and r['block_count']==2
    assert r['robustness_status']=='INSUFFICIENT_DATE_OR_REGION_REPLICATION'


def test_desktop_and_toolkit_use_same_alignment_service(tmp_path):
    from ecodata.algorithms.platform_search import submit_file_request
    from ecodata.desktop.scan import Scanner
    from ecodata.desktop.pair_processing import run_cross_variable
    files=[]
    for name,n,unit in [('temperature',8,'K'),('rainfall',4,'mm')]:
        p=tmp_path/(name+'.nc');field(name,n,unit).to_netcdf(p);files.append(p)
    scan,_=Scanner().scan(str(tmp_path))
    bindings=[(next(i for i,r in enumerate(scan['rows']) if r['name']==p.name),p.stem) for p in files]
    page=run_cross_variable(scan,'C1',dict(fields=bindings,recipes=[['scale.semantic_stack']]),tmp_path/'page',None,None)
    toolkit=submit_file_request(dict(inputs=[dict(path=str(p),role='primary' if i==0 else 'auxiliary') for i,p in enumerate(files)],
        request=dict(task='cross_variable_alignment',variables=[p.stem for p in files])),tmp_path/'toolkit')
    assert page['original_report']['variables']==toolkit['variables']
    for a,b in zip(page['original_report']['products'],toolkit['products']):
        with xr.open_dataset(a['path']) as left,xr.open_dataset(b['path']) as right:xr.testing.assert_identical(left,right)


def test_project_netcdf_member_is_seekable_and_cleaned_after_failure(tmp_path,monkeypatch):
    from contextlib import ExitStack
    from pathlib import Path
    import io,zipfile
    from ecodata.models.project_binding import ProjectFiles
    from ecodata.interface.engineering_upload import LocalUpload
    p=tmp_path/'inputs.zip'
    with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('native.nc',field('temperature',4,'K').to_netcdf())
    before=p.read_bytes();opened_paths=[]
    original=xr.open_dataset
    def capture(path,**kwargs):
        opened_paths.append(Path(path));return original(path,**kwargs)
    monkeypatch.setattr(xr,'open_dataset',capture)
    with ExitStack() as stack:
        files=ProjectFiles([LocalUpload(p.name,p)],stack)
        with pytest.raises(RuntimeError,match='consumer failed'):
            with files.netcdf('native.nc') as ds:
                assert ds.temperature.shape==(4,4)
                assert opened_paths[-1].is_file()
                raise RuntimeError('consumer failed')
    assert opened_paths and not opened_paths[-1].exists() and p.read_bytes()==before


def test_ds_alignment_suggestion_passes_common_gate_but_cross_variable_blend_does_not(tmp_path):
    from ecodata.desktop.scan import Scanner
    from ecodata.desktop import discovery
    a,b=field('temperature',8,'K'),field('rainfall',4,'mm')
    a.temperature.attrs['standard_name']='air_temperature'
    b.rainfall.attrs['standard_name']='precipitation_amount'
    for name,ds in [('a.nc',a),('b.nc',b)]:ds.to_netcdf(tmp_path/name)
    scan,_=Scanner().scan(str(tmp_path))
    summary,_=discovery.discover(scan,'data_only')
    details={f['id']:f for source in summary['sources'] for f in source['fields']}
    bound=[]
    for i,row in enumerate(scan['rows']):
        name='temperature' if row['name']=='a.nc' else 'rainfall'
        j=next(j for j,f in enumerate(row['fields']) if f['name']==name)
        bound.append((i,f'D{i+1}.F{j+1}',row['fields'][j]))
    pair,local=discovery.describe_pair(scan,'C1',*bound,'data_only',details)
    summary['candidates']=[pair]
    checked,gated=discovery.check(scan,{'C1':local},dict(decisions=[dict(candidate='C1',action='align_space')]),summary)
    assert checked[0]['status']=='通过', checked
    assert gated['C1','align_space']['route']=='cross_variable_alignment'
    checked,_=discovery.check(scan,{'C1':local},dict(decisions=[dict(candidate='C1',action='fusion')]),summary)
    assert checked[0]['status']!='通过' and checked[0]['block_code']=='quantity_conflict'
