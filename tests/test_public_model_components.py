"""Public dispatch regressions; synthetic changes are format tests, not skill claims."""
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import netcdf_file

from ecodata.algorithms.automatic_preparation import prepare_uploads
from ecodata.models.component_service import capabilities
from ecodata.models.converter_contract import load_converter

ROOT = Path(__file__).parents[1]


def uploads(model='camx', operation='emissions', change=None):
    base = ROOT/'examples/ai_round1/camx/emissions'
    request = dict(schema='ecodata.model-task.v1', model_id=model,
                   operation=operation, input='data.csv',
                   configuration=json.loads((base/'config_LATLON.json').read_text()))
    if change:
        change(request)
    a = io.BytesIO(json.dumps(request).encode()); a.name = 'request.json'
    b = io.BytesIO((base/'A_synthetic_LATLON_hourly.csv').read_bytes()); b.name = 'data.csv'
    return [a, b]


@pytest.mark.parametrize('model', ['swat_plus', 'camx', 'cmaq', 'calpuff'])
def test_registered_contract_has_no_case_dependencies(model):
    contract = load_converter(model).requirements()
    assert contract['component_dispatch'] and contract['dependencies'] == []
    assert contract['coordinates'] == {} and len(capabilities(model)) >= 4


@pytest.mark.parametrize('change,reason', [
    (lambda r: r.update(model_id='cmaq'), '目标模型'),
    (lambda r: r.update(operation='unimplemented'), '未实现'),
    (lambda r: r.update(input='../private.csv'), '越界'),
    (lambda r: r.update(input='absent.csv'), '缺失'),
    (lambda r: r.update(input_sha256='0'*64), '指纹'),
    (lambda r: r.update(command='run.exe'), '程序'),
    (lambda r: r['configuration'].update(mode='boundary_concentrations'), 'mode冲突'),
    (lambda r: r['configuration'].update(executable='run.exe'), '程序'),
    (lambda r: r.update(configuration=None), '科学配置'),
])
def test_public_rejects_unsafe_or_conflicting_contract(tmp_path, change, reason):
    with pytest.raises(ValueError, match=reason):
        prepare_uploads(uploads(change=change), 'camx', tmp_path)


def test_new_location_grid_and_values_reach_existing_writer(tmp_path):
    results = []
    for shift in [0, 7]:
        def change(r):
            r['configuration']['grid']['xorig'] += shift
        source = uploads(change=change)
        result = prepare_uploads(source, 'camx', tmp_path/str(shift))
        assert result['manifest']['status'] == 'COMPONENT_EXPORTED'
        assert not result['manifest']['native_ready']
        assert not result['manifest']['native_model_executed']
        with zipfile.ZipFile(result['archive']) as z:
            assert z.testzip() is None
            name = next(n for n in z.namelist() if n.endswith('.nc'))
            with netcdf_file(io.BytesIO(z.read(name)), mmap=False) as ds:
                results.append((float(ds.XORIG), ds.variables['NO'][:].copy()))
    assert results[1][0] - results[0][0] == 7
    np.testing.assert_array_equal(results[0][1], results[1][1])


def test_swat_soils_public_output_independently_matches_official(tmp_path):
    from test_swat_plus_soil import parse
    base = ROOT/'examples/ai_round1/swat_plus/soil'
    req = dict(schema='ecodata.model-task.v1',model_id='swat_plus',operation='soil',
               input='soil.csv', configuration=json.loads((base/'config.json').read_text()))
    a=io.BytesIO(json.dumps(req).encode());a.name='request.json'
    b=io.BytesIO((base/'A_official_valid_subset.csv').read_bytes());b.name='soil.csv'
    digest=hashlib.sha256(b.getvalue()).hexdigest()
    result=prepare_uploads([a,b],'swat_plus',tmp_path)
    actual=(tmp_path/'components/soils.sol').read_bytes()
    expected=parse((base/'B2_official_soils.sol').read_bytes());expected.pop('soil_05')
    assert parse(actual)==expected
    assert hashlib.sha256(b.getvalue()).hexdigest()==digest
    assert result['manifest']['preparation']['tasks'][0]['source']['sha256']==digest


def test_native_patch_requires_actual_target_binding(tmp_path):
    import xarray as xr
    source=xr.Dataset({'v':('x',[1.],dict(units='K'))})
    raw=source.to_netcdf()
    a=io.BytesIO(raw);a.name='source.nc'
    b=io.BytesIO(raw);b.name='target.nc'
    task=dict(type='model_component',operation='update_field',input='source.nc',
              configuration=dict(target='target.nc',target_sha256=hashlib.sha256(raw).hexdigest(),variable='v'))
    with pytest.raises(ValueError,match='native_reference_sha256'):
        prepare_uploads([a,b],'cmaq',tmp_path,task=task)


@pytest.mark.parametrize('model', ['camx','cmaq'])
def test_native_mask_consumes_real_product_and_preserves_other_cells(tmp_path,model):
    import xarray as xr
    target=xr.Dataset({'v':('x',[280.,281.],dict(units='K')),
                       'untouched':('x',[10.,20.])}).to_netcdf(engine='scipy',format='NETCDF3_CLASSIC')
    digest=hashlib.sha256(target).hexdigest()
    product=xr.Dataset({'v':('x',[300.,310.],dict(units='K')),
                        'mask':('x',[True,False])},attrs=dict(native_reference_sha256=digest)).to_netcdf()
    a=io.BytesIO(product);a.name='source.nc'
    b=io.BytesIO(target);b.name='target.nc'
    task=dict(type='model_component',operation='update_field',input='source.nc',
              configuration=dict(target='target.nc',target_sha256=digest,variable='v',mask_variable='mask'))
    result=prepare_uploads([a,b],model,tmp_path,task=task)
    with zipfile.ZipFile(result['archive']) as z, xr.open_dataset(io.BytesIO(z.read('components/native_updated.nc'))) as actual:
        np.testing.assert_array_equal(actual.v,[300.,281.])
        np.testing.assert_array_equal(actual.untouched,[10.,20.])
    assert hashlib.sha256(b.getvalue()).hexdigest()==digest


@pytest.mark.parametrize('strict', [False, True])
def test_native_script_review_does_not_block_data_only_tasks(tmp_path, strict):
    from types import SimpleNamespace
    from ecodata.models.archive_preflight import inspect_upload_archives
    blob=io.BytesIO();blob.name='project.zip'
    with zipfile.ZipFile(blob,'w') as z:z.writestr('run.csh','#!/bin/csh\necho not executed')
    converter=SimpleNamespace(probe_sources=lambda _:dict(issues=[dict(file='run.csh',message='unreviewed')]))
    if strict:
        with pytest.raises(ValueError,match='unreviewed'):
            inspect_upload_archives([blob],converter,tmp_path,native_required=True)
    else:
        r=inspect_upload_archives([blob],converter,tmp_path,native_required=False)
        assert r['status']=='CONTINUE' and r['native_contract_issues']


@pytest.mark.parametrize('model,label,operation,relative,input_name,config_name', [
    ('swat_plus','SWAT+','soil','swat_plus/soil','A_official_valid_subset.csv','config.json'),
    ('camx','CAMx','emissions','camx/emissions','A_synthetic_LATLON_hourly.csv','config_LATLON.json'),
    ('cmaq','CMAQ','emissions','cmaq/emissions','A_EDGAR_CO_2014_Paris.nc','config_EDGAR.json'),
    ('calpuff','CALPUFF','precipitation','calpuff/precipitation','A_NOAA_USCRN_20210701.csv','config_NOAA.json'),
])
def test_four_models_page_and_toolkit_share_real_writers(tmp_path,model,label,operation,relative,input_name,config_name):
    from streamlit.testing.v1 import AppTest
    from ecodata.interface.engineering_upload import local_folder_uploads
    base=ROOT/'examples/ai_round1'/relative
    inp=tmp_path/'input';inp.mkdir()
    (inp/input_name).write_bytes((base/input_name).read_bytes())
    req=dict(schema='ecodata.model-task.v1',model_id=model,operation=operation,input=input_name,
             configuration=json.loads((base/config_name).read_text()))
    (inp/'request.json').write_text(json.dumps(req),encoding='utf8')
    direct=prepare_uploads(local_folder_uploads(inp,include_all=True),model,tmp_path/'direct')
    app=AppTest.from_file(ROOT/'src/ecodata/interface/app.py').run(timeout=30)
    app.selectbox[0].select(label).run()
    app.text_input[0].set_value(str(inp)).run()
    app.button[0].click().run(timeout=30)
    assert not app.exception and not app.error
    assert app.get('download_button')[0].label=='下载模型组件与质量记录'
    page=app.session_state['workbench_result']['data']
    assert page['manifest']['status']==direct['manifest']['status']=='COMPONENT_EXPORTED'
    with zipfile.ZipFile(page['archive']) as p, zipfile.ZipFile(direct['archive']) as d:
        # Final-check receipt IDs are intentionally unique per invocation.
        # Compare the real payload names and verify both receipt outcomes.
        payload=lambda z:[n for n in z.namelist() if not n.startswith('_ecodata/final_check/')]
        assert payload(p)==payload(d)
        for key in ('format','project','native'):
            assert page['manifest']['final_check'][key]['status']==direct['manifest']['final_check'][key]['status']
        for name in payload(p):
            if name.endswith('.json'):continue
            if name.endswith('.nc'):
                # I/O API records creation/write time: compare all scientific
                # values and metadata, excluding only these wall-clock fields.
                with netcdf_file(io.BytesIO(p.read(name)),mmap=False) as x, netcdf_file(io.BytesIO(d.read(name)),mmap=False) as y:
                    assert x.dimensions==y.dimensions
                    ignore={'CDATE','CTIME','WDATE','WTIME'}
                    assert set(x._attributes)==set(y._attributes)
                    for key in set(x._attributes)-ignore:
                        np.testing.assert_array_equal(x._attributes[key],y._attributes[key])
                    assert set(x.variables)==set(y.variables)
                    for key in x.variables:
                        assert x.variables[key]._attributes==y.variables[key]._attributes
                        np.testing.assert_array_equal(x.variables[key][:],y.variables[key][:])
            else:assert p.read(name)==d.read(name)
