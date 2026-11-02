import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from examples.ai_round1.independent import wps
from ecodata.core.exceptions import SchemaError
from ecodata.io.structures import read_structure
from ecodata.models.adapters.wps_sst import export_sst
from ecodata.round1 import convert

CASE = Path(__file__).resolve().parents[1] / 'examples/ai_round1/wrf/daily'


def load():
    return (CASE/'A_sst_three_days.nc').read_bytes(), json.loads((CASE/'config.json').read_text())


def test_real_daily_files_independently_read_dates_units_grid_values(tmp_path):
    source, config = load()
    ds = read_structure('source.nc', source)
    files, report = convert('wrf', source, config)
    assert report['source_interval_seconds'] == 86400
    assert 'no temporal interpolation' in report['time_processing']
    assert 'WPS_metgrid_SST_constant.zip' not in files
    assert sorted(n for n in files if n.endswith('.wps')) == [
        f'SST_2020-01-0{i}_12.wps' for i in range(1, 4)]
    for i in range(3):
        name = f'SST_2020-01-0{i+1}_12.wps'
        path = tmp_path/name; path.write_bytes(files[name]); parsed = wps(path)
        assert parsed['date'] == f'2020-01-0{i+1}_12:00:00'
        assert parsed['field'] == 'SST' and parsed['unit'] == 'K'
        assert parsed['level'] == 200100 and parsed['xfcst'] == 0
        np.testing.assert_allclose(parsed['projection'][:4], [30.125, 300.125, .25, .25], rtol=0, atol=1e-6)
        assert parsed['values'].shape == (4, 4)
        expected = ds.sst.isel(time=i).squeeze().values.astype(float) + 273.15
        np.testing.assert_allclose(parsed['values'], expected, rtol=0, atol=2e-5)


def test_hourly_sequence_remains_hourly_and_single_remains_single(tmp_path):
    source, config = load(); ds = read_structure('source.nc', source)
    ds = ds.assign_coords(time=pd.date_range('2020-01-01T12:00', periods=3, freq='h'))
    files, report = export_sst(ds, config)
    assert report['source_interval_seconds'] == 3600
    assert 'README_daily_SST.txt' not in files
    for i in range(3):
        name=f'SST_2020-01-01_{12+i}.wps'
        single, _ = export_sst(ds.isel(time=slice(i,i+1)), config)
        assert files[name] == single[name]
        path=tmp_path/name; path.write_bytes(files[name]); parsed=wps(path)
        assert parsed['date'] == f'2020-01-01_{12+i}:00:00'
        np.testing.assert_allclose(parsed['values'], ds.sst.isel(time=i).squeeze().values.astype(float)+273.15, rtol=0, atol=2e-5)


@pytest.mark.parametrize('kind', ['missing_day','mixed_interval','half_hour','constant_series'])
def test_no_silent_temporal_changes(kind):
    source, config=load(); ds=read_structure('source.nc',source)
    if kind=='missing_day': ds=ds.isel(time=[0,2])
    elif kind=='mixed_interval': ds=ds.assign_coords(time=pd.to_datetime(['2020-01-01T12:00','2020-01-02T12:00','2020-01-02T13:00']))
    elif kind=='half_hour': ds=ds.assign_coords(time=pd.date_range('2020-01-01T12:30',periods=3,freq='D'))
    else: config['wps_handoff']={'mode':'constant_snapshot'}
    with pytest.raises(SchemaError): export_sst(ds,config)


def test_daily_real_ui_reports_time_method():
    source, config=load()
    def upload(label, **kwargs):
        b=source if label=='导入源数据' else json.dumps(config).encode()
        return SimpleNamespace(getvalue=lambda:b)
    with patch('streamlit.file_uploader',side_effect=upload):
        app=AppTest.from_string('from ecodata.interface.round1_page import render_round1_page\nrender_round1_page()',default_timeout=30).run()
        app.selectbox[0].set_value('wrf').run()
        assert any('不做时间插值' in n.value for n in app.info)
        app.button[0].click().run()
        assert not app.exception and not app.error
        files, report=app.session_state['round1_result']
        assert report['source_interval_seconds']==86400
        assert len([n for n in files if n.endswith('.wps')])==3
        assert len(app.get('download_button'))==len(files)+1
