from io import BytesIO
from pathlib import Path
import json
import zipfile

import pytest
import numpy as np
import xarray as xr
from ecodata.models.project_binding import bind_project, bound_resource
from ecodata.algorithms.automatic_preparation import prepare_uploads
from ecodata.interface.engineering_upload import LocalUpload, local_folder_uploads


def uploaded(files):
    b = BytesIO()
    with zipfile.ZipFile(b, 'w') as z:
        for k, v in files.items():
            z.writestr(k, v)
    b.name = 'arbitrary-name.zip'
    return b


def case(root='project', reference='inputs/met.nc'):
    controls = """&CAMx_Control
 Restart=.false.,
 Run_Message='not a version contract',
 Chemistry_Parameters='inputs/chem',
 Photolyis_Rates='inputs/rates',
 Ozone_Column='inputs/ozone',
 Initial_Conditions='inputs/icon',
 Boundary_Conditions='inputs/bcon',
 Met2D_Grid(1)='%s',
 Root_Output_Name='outputs/future',
 Master_Grid_Restart='not-present-and-disabled',
/""" % reference
    ds = xr.Dataset({'t2': (('y', 'x'), np.full((2, 3), 290.))})
    files = {root+'/control.any': controls, root+'/inputs/met.nc': ds.to_netcdf()}
    files.update({root+'/inputs/'+n: b'boundary fixture' for n in ['chem','rates','ozone','icon','bcon']})
    return files


def test_two_unrelated_projects_and_real_downstream_data():
    for root in ['region_1', 'completely-different/region_2']:
        uploads = [uploaded(case(root))]
        b = bind_project(uploads, 'camx')
        assert len(b['bindings']) == 6 and b['payload_hashes_verified']
        assert not b['native_schema_verified']
        assert b['inactive'] and not any('future' in r['target'] for r in b['bindings'])
        with bound_resource(uploads, b, 'met2d_grid', (0,)) as stream:
            with xr.open_dataset(BytesIO(stream.read())) as ds:
                assert float(ds.t2.mean()) == 290.
        changed = case(root)
        changed[root+'/inputs/met.nc'] = b'changed bytes'
        with pytest.raises(ValueError, match='已变化'):
            with bound_resource([uploaded(changed)], b, 'met2d_grid', (0,)):
                pass


def test_absolute_relocation_is_evidence_based():
    b = bind_project([uploaded(case(reference='C:/another-machine/data/inputs/met.nc'))], 'camx')
    row = next(r for r in b['bindings'] if r['parameter'] == 'met2d_grid')
    assert row['reason'] == 'unique_path_suffix:2'


@pytest.mark.parametrize('ref', ['missing.nc', '../inputs/met.nc', '/private/unknown.nc'])
def test_missing_or_unsafe_reference(ref):
    with pytest.raises(ValueError):
        bind_project([uploaded(case(reference=ref))], 'camx')


def test_ambiguous_suffix_and_multiple_controls_rejected():
    files = case(reference='C:/machine/inputs/met.nc')
    files['other/inputs/met.nc'] = b'other'
    with pytest.raises(ValueError, match='歧义'):
        bind_project([uploaded(files)], 'camx')
    files = case(); files.update(case('second'))
    with pytest.raises(ValueError, match='多个工程'):
        bind_project([uploaded(files)], 'camx')


def test_wrfs_share_binding_not_case_paths():
    files = {'region/anything': '&time_control\n restart=.false.,\n/\n&domains\n max_dom=1,\n/',
             'region/wrfinput_d01': b'initial', 'region/wrfbdy_d01': b'boundary'}
    b = bind_project([uploaded(files)], 'wrf')
    assert {r['parameter'] for r in b['bindings']} == {'initial_state','boundary_state'}


def test_directory_and_zip_shared_entry_and_download(tmp_path):
    files = case()
    for name, raw in files.items():
        p = tmp_path/'input'/name;p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw.encode() if isinstance(raw,str) else raw)
    a = prepare_uploads([uploaded(files)], 'camx', tmp_path/'zip')
    b = prepare_uploads(local_folder_uploads(tmp_path/'input', include_all=True), 'camx', tmp_path/'dir')
    for result in (a,b):
        assert result['manifest']['status'] == 'ENGINEERING_BOUND'
        assert not result['manifest']['native_ready']
        with zipfile.ZipFile(result['archive']) as z:
            assert z.testzip() is None
            binding = json.loads(z.read('project_binding.json'))
            assert len(binding['bindings']) == 6
    aa=json.loads((tmp_path/'zip/project_binding.json').read_text(encoding='utf8'))
    bb=json.loads((tmp_path/'dir/project_binding.json').read_text(encoding='utf8'))
    assert aa['bindings'] == bb['bindings']


def test_unsafe_member_and_restarts_and_unknown_formats():
    files=case();files['../outside']=b'bad'
    with pytest.raises(ValueError,match='不安全'):
        bind_project([uploaded(files)],'camx')
    files=case();files['project/control.any']=files['project/control.any'].replace('Restart=.false.', 'Restart=.true.')
    with pytest.raises(ValueError,match='缺失'):
        bind_project([uploaded(files)],'camx')
    assert bind_project([uploaded({'data.csv':b'x,y\n1,2'})], 'camx') is None
    assert bind_project([uploaded(case())], 'cmaq') is None


def test_embedded_job_is_not_a_native_control():
    files=case()
    files['project/control.any']='! leading comment\n'+files['project/control.any']
    files['job.sh']='#!/bin/sh\ncat << EOF\n'+files['project/control.any']+'\nEOF'
    b=bind_project([uploaded(files)],'camx')
    assert b['control']['path']=='project/control.any'
    assert 'job.sh' in b['unbound_files']


def test_main_page_reports_binding_without_claiming_conversion(tmp_path):
    from streamlit.testing.v1 import AppTest
    files=case()
    for name, raw in files.items():
        p=tmp_path/'input'/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(raw.encode() if isinstance(raw,str) else raw)
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'src/ecodata/interface/app.py').run(timeout=30)
    app.selectbox[0].select('CAMx').run()
    app.text_input[0].set_value(str(tmp_path/'input')).run()
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert any('已自动解析上传工程' in item.value for item in app.info)
    assert app.get('download_button')[0].label=='下载工程绑定记录'
