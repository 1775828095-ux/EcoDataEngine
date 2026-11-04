from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ecodata.models.task_dispatch import discover,normalise,dispatch_preparation


def inputs(quantity='air_temperature',column='temperature_K',unit='K',missing=True):
    lon,lat=np.meshgrid(np.linspace(-100,-98,5),np.linspace(35,37,5))
    attrs=dict(standard_name=quantity,units=unit)
    ds=xr.Dataset({'anything':(('y','x'),280+lon*.1+lat*.05,attrs)},
        coords=dict(longitude=(('y','x'),lon,dict(units='degrees_east')),latitude=(('y','x'),lat,dict(units='degrees_north'))),
        attrs=dict(Conventions='CF-1.8',analysis_time_utc='2001-02-03T12:00:00Z'))
    frame=pd.DataFrame(dict(station_id=[str(i) for i in range(5)],time_utc=['2001-02-03_12:00:00']*5,
        latitude=[35.2,35.2,36.8,36.8,36],longitude=[-99.8,-98.2,-99.8,-98.2,-99],levels=1,
        **{column:[273.,274.,275.,276.,np.nan if missing else 277.],column+'_qc':[0,0,0,0,-88 if missing else 0]}))
    b=BytesIO(ds.to_netcdf());b.name='random.nc'
    c=BytesIO(frame.to_csv(index=False).encode());c.name='random.csv'
    return [b,c]


def test_requirements_follow_quantity_not_model_or_filename():
    for quantity,column,unit in [('air_temperature','temperature_K','K'),('air_pressure','pressure_Pa','Pa')]:
        c,_=discover(inputs(quantity,column,unit));assert len(c)==1
        assert c[0][0]['quantity']==quantity
        model,refs,target,orig,accepted,report=normalise(c[0][0],c[0][1][0])
        assert quantity in model and report['natural_gap_targets']==1
        assert refs.variable.eq(quantity).all() and report['valid_conditioning_rows']==4
        assert orig[column].isna().all() and not report['natural_gap_truth_available']


def test_unrelated_changes_and_aliases_do_not_create_new_tasks():
    u=inputs();a,_=discover(u)
    u[0].name='renamed.nc';u[1].name='renamed.csv'
    copy=BytesIO(u[1].getvalue());copy.name='alias.csv'
    b,audit=discover([copy,*reversed(u)])
    assert len(a)==len(b)==1 and a[0][0]['quantity']==b[0][0]['quantity']
    assert a[0][0]['sha256']==b[0][0]['sha256']
    assert any(r['action']=='DUPLICATE_ALIAS' for r in audit)


def test_no_mask_is_created_for_complete_observations():
    c,_=discover(inputs(missing=False));assert c==[]


def test_unknown_timestamp_and_unit_conflicts_are_explicit():
    c,_=discover(inputs());table,fields=c[0]
    fields[0]['ds'].attrs.pop('analysis_time_utc')
    with pytest.raises(ValueError,match='analysis_time_utc'):normalise(table,fields[0])
    c,_=discover(inputs(unit='kg'));table,fields=c[0]
    with pytest.raises(Exception):normalise(table,fields[0])


def test_no_model_specific_task_gate(tmp_path):
    # Planning rejection is data/contract-based for every target model.
    for model in ['wrf','camx','swat_plus']:
        folder=tmp_path/model;folder.mkdir()
        with pytest.raises(ValueError,match='当前数据任务分派未支持'):
            dispatch_preparation(inputs(),model,folder,task=dict(type='unknown'))
        assert (folder/'task_dispatch.json').exists()


def test_sst_handler_declines_other_task_without_blanket_rejection(tmp_path):
    import zipfile
    from existing_air_converters.wrf_bundle_pipeline import prepare
    from netCDF4 import Dataset
    p=tmp_path/'state.nc'
    with Dataset(p,'w') as ds:
        ds.TITLE='OUTPUT FROM REAL_EM V3.9';ds.createDimension('Time',1)
        for n in ['XLAT','XLONG','LANDMASK','Times']:ds.createVariable(n,'f4',('Time',))[:]=1
    b=BytesIO();b.name='engineering.zip'
    with zipfile.ZipFile(b,'w') as z:z.writestr('wrfinput_d01',p.read_bytes())
    assert prepare([b],tmp_path/'out') is None


def test_production_entry_consumes_shared_dispatch_before_model_writer(tmp_path):
    from ecodata.algorithms.automatic_preparation import prepare_uploads
    for model in ['wrf','camx']:
        with pytest.raises(ValueError,match='当前数据任务分派未支持'):
            prepare_uploads(inputs(),model,tmp_path/model,task=dict(type='unsupported'))
        assert (tmp_path/model/'task_dispatch.json').exists()
