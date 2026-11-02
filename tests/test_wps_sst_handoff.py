import io,json,zipfile,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from ecodata.round1 import convert
from ecodata.core.exceptions import SchemaError
from ecodata.io.structures import read_structure
from ecodata.models.adapters.wps_sst import export_sst
from examples.ai_round1.independent import wps

CASE=Path(__file__).resolve().parents[1]/'examples/ai_round1/wrf'
def test_noaa_constant_handoff_roundtrips_real_grid_time_and_units(tmp_path):
    config=json.loads((CASE/'config_constant_snapshot.json').read_text(encoding='utf-8'))
    source=(CASE/'A_sst_subset.nc').read_bytes()
    files,report=convert('wrf',source,config)
    legacy=next(n for n in files if n.endswith('.wps'))
    with zipfile.ZipFile(io.BytesIO(files['WPS_metgrid_SST_constant.zip'])) as z:
        assert set(z.namelist())=={'ECODATA_SST_CONSTANT','namelist.metgrid.fragment','manifest.json','README.txt'}
        payload=z.read('ECODATA_SST_CONSTANT');assert payload==files[legacy]
        manifest=json.loads(z.read('manifest.json'))
        assert manifest['snapshot_time_UTC']=='2020-01-01T12:00:00Z'
        assert manifest['intermediate_sha256']==hashlib.sha256(payload).hexdigest()
        assert "constants_name = './ECODATA_SST_CONSTANT'" in z.read('namelist.metgrid.fragment').decode()
        assert not manifest['official_metgrid_run_verified'] and not manifest['complete_WRF_input']
    path=tmp_path/'ECODATA_SST_CONSTANT';path.write_bytes(payload);parsed=wps(path)
    original=read_structure('source.nc',source)
    np.testing.assert_allclose(parsed['values'],np.asarray(original.sst.values,dtype=float).squeeze()+273.15,rtol=0,atol=2e-5)
    np.testing.assert_allclose(parsed['projection'][:4],[30.125,300.125,.25,.25],rtol=0,atol=1e-5)
    assert parsed['date']=='2020-01-01_12:00:00' and parsed['field']=='SST' and parsed['unit']=='K' and parsed['level']==200100
    assert report['wps_handoff']==manifest and report['source_sha256']==hashlib.sha256(source).hexdigest()

def test_default_conversion_remains_without_constant_assumption():
    cfg=json.loads((CASE/'config.json').read_text())
    files,report=convert('wrf',(CASE/'A_sst_subset.nc').read_bytes(),cfg)
    assert 'WPS_metgrid_SST_constant.zip' not in files and 'wps_handoff' not in report

@pytest.mark.parametrize('handoff_config',[{},'constant_snapshot',{'mode':'unknown'},{'mode':'constant_snapshot','pick_first':True}])
def test_ambiguous_handoff_configuration_is_rejected(handoff_config):
    cfg=json.loads((CASE/'config.json').read_text());cfg['wps_handoff']=handoff_config
    with pytest.raises(SchemaError,match='wps_handoff'):
        convert('wrf',(CASE/'A_sst_subset.nc').read_bytes(),cfg)

def test_constant_handoff_never_silently_collapses_two_valid_hourly_frames():
    ds=read_structure('source.nc',(CASE/'A_sst_subset.nc').read_bytes())
    ds=ds.isel(time=[0,0]).assign_coords(time=pd.to_datetime(['2020-01-01T12:00','2020-01-01T13:00']))
    cfg=json.loads((CASE/'config_constant_snapshot.json').read_text())
    with pytest.raises(SchemaError,match='一个源时刻'):export_sst(ds,cfg)
