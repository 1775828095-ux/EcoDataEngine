"""Synthetic native dependency binding; not an environmental accuracy study."""
from pathlib import Path
import hashlib,zipfile
from copy import deepcopy
import numpy as np
import pytest
from netCDF4 import Dataset
from ecodata.models.engineering.native import prepare_indexed_products
from test_project_binding import case,uploaded


def fixture(tmp_path):
    source=tmp_path/'source.nc'
    with Dataset(source,'w',format='NETCDF3_CLASSIC') as ds:
        for name,size in [('TSTEP',3),('LAY',1),('ROW',2),('COL',2),('VAR',1),('DATE-TIME',2)]:ds.createDimension(name,size)
        ds.CAMx_NAME='2D METEOROLOGY';ds.CPROJ=2;ds.GDTYP=2;ds.ITZON=0
        for name,value in dict(NCOLS=2,NROWS=2,XORIG=0.,YORIG=0.,XCELL=1000.,YCELL=1000.,P_ALP=33.,P_BET=45.,XCENT=-97.,YCENT=40.).items():ds.setncattr(name,value)
        ds.createVariable('TFLAG','i4',('TSTEP','VAR','DATE-TIME'))[:]=[[[2020001,0]],[[2020001,10000]],[[2020001,20000]]]
        for name,dim in [('X','COL'),('Y','ROW')]:
            v=ds.createVariable(name,'f4',(dim,));v.units='km';v[:]=[.5,1.5]
        v=ds.createVariable('t2','f4',('TSTEP','LAY','ROW','COL'));v.units='K';v.long_name='temperature at 2 m';v[:]=280.
        ds.createVariable('untouched','f4',('ROW','COL'))[:]=123.
    raw=source.read_bytes();files=case(root='unrelated_project');files['unrelated_project/inputs/met.nc']=raw
    update=dict(source_sha256=hashlib.sha256(raw).hexdigest(),variable='t2',unit='K',task='public_task',
        delivery_status='ENHANCEMENT_RECOMMENDED',points=[dict(index=[1,0,1,1],expected=280.,value=283.)])
    return [uploaded(files)],update


def test_delivered_predictions_reach_the_actual_native_parent(tmp_path):
    uploads,update=fixture(tmp_path)
    results=prepare_indexed_products(uploads,'camx',tmp_path/'writeback',[update])
    assert len(results)==1 and results[0]['manifest']['native_writeback_verified']
    assert results[0]['manifest']['processed_product_binding']['assigned_positions']==1
    with zipfile.ZipFile(results[0]['archive']) as archive:
        path=tmp_path/'readback.nc';path.write_bytes(archive.read('unrelated_project/inputs/met.nc'))
        with Dataset(path) as ds:
            assert ds['t2'][1,0,1,1]==283 and ds['t2'][0,0,0,0]==280
            assert np.all(ds['untouched'][:]==123)
    assert results[0]['manifest']['temporary_cleanup']['selective_resources']['temporary_directory_absent']


def test_background_mismatch_is_not_silently_written(tmp_path):
    uploads,update=fixture(tmp_path);update['points'][0]['expected']=999.
    with pytest.raises(ValueError,match='背景与原生父资源不一致'):
        prepare_indexed_products(uploads,'camx',tmp_path/'writeback',[update])


def test_diagnostic_passthrough_does_not_override_selected_product(tmp_path):
    uploads,update=fixture(tmp_path);update['purpose']='selected_product'
    comparison=deepcopy(update);comparison.update(purpose='comparison_passthrough',delivery_status='BASELINE_RETAINED')
    comparison['points'][0]['value']=280.
    result,=prepare_indexed_products(uploads,'camx',tmp_path/'writeback',[comparison,update])
    assert result['manifest']['processed_product_binding']['purposes']==['selected_product']
    with zipfile.ZipFile(result['archive']) as archive:
        path=tmp_path/'check.nc';path.write_bytes(archive.read('unrelated_project/inputs/met.nc'))
        with Dataset(path) as ds:assert ds['t2'][1,0,1,1]==283.


def test_bound_native_project_does_not_infer_missing_inputs_from_partial_artifact(tmp_path):
    from ecodata.desktop.output_review import describe_output
    uploads,update=fixture(tmp_path)
    result,=prepare_indexed_products(uploads,'camx',tmp_path/'writeback',[update])
    check=describe_output(dict(manifest=result['manifest'],local=dict(artifacts=[dict(name='point.nc',fields=['t2'],dimensions={})])),
        dict(id='camx',inputs=[dict(name='generic chemistry role',required=True,fields=[dict(name='PM25')])]))
    assert check['requirements']==[]
    assert not check['native_ready']
