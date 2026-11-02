import io
import json
import zipfile
import datetime as dt
from pathlib import Path

import pytest
from ecodata.models.swat_weather_project import attach_weather, project_stations
from ecodata.round1 import convert

ROOT = Path(__file__).resolve().parents[1]


def test_explicit_short_period_preserves_upload_and_reports_time_change():
    original = project()
    original['time.sim'] = b'time\nday_start yrc_start day_end yrc_end step\n1 2021 365 2021 0\n'
    source = pack(original)
    with pytest.raises(ValueError, match='天气日期不足'):
        attach_weather(source, weather(), 'a')
    result, report = attach_weather(source, weather(), 'a',
        simulation_period=(dt.date(2021,1,1),dt.date(2021,1,2)))
    assert unpack(source) == original
    assert unpack(result)['time.sim'].splitlines()[2] == b'1 2021 2 2021 0'
    assert report['original_simulation_period'] == ['2021-01-01','2021-12-31']
    assert report['simulation_period'] == ['2021-01-01','2021-01-02']
    assert next(row for row in report['file_manifest'] if row['file']=='time.sim')['action']=='modified'
    assert unpack(result)['soils.sol'] == original['soils.sol']
    with pytest.raises(ValueError, match='天气日期不足'):
        attach_weather(source, weather(), 'a', simulation_period=(dt.date(2021,1,1),dt.date(2021,1,3)))
    with pytest.raises(ValueError, match='顺序正确'):
        attach_weather(source, weather(), 'a', simulation_period=(dt.date(2021,1,2),dt.date(2021,1,1)))


def pack(files, prefix=''):
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w') as z:
        for n, v in files.items():
            z.writestr(prefix+n, v)
    return out.getvalue()


def unpack(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        return {n: z.read(n) for n in z.namelist()}


def project(sim=False):
    weather = 'sim' if sim else 'old.pcp'
    files = {'file.cio': b'title\nclimate weather-sta.cli null null pcp.cli tmp.cli null null null null\n',
             'time.sim': b'time\nday_start yrc_start day_end yrc_end step\n1 2021 2 2021 0\n',
             'weather-sta.cli': f'title\nname wgn pcp tmp slr hmd wnd wnd_dir atmo_dep\na null {weather} old.tmp sim sim sim null null\nb null {weather} old.tmp sim sim sim null null\n'.encode(),
             'pcp.cli': b'title\nfilename\nold.pcp\n', 'tmp.cli': b'title\nfilename\nold.tmp\n',
             'old.pcp': b'old rain', 'old.tmp': b'old temp', 'soils.sol': b'keep physical parameters'}
    return files


def weather(extra=False):
    files = {'station.pcp': b'title\nNBYR TSTEP LAT LONG ELEV\n1 0 1 2 3\n2021 1 9.1\n2021 2 0\n',
             'station.tem': b'title\nNBYR TSTEP LAT LONG ELEV\n1 0 1 2 3\n2021 1 0 -17\n2021 2 2 -10\n'}
    if extra:
        files['station.slr'] = files['station.pcp']
    return files


def test_first_index_preserves_other_stations():
    original = project()
    original.pop('pcp.cli')
    original.pop('tmp.cli')
    original['file.cio'] = original['file.cio'].replace(b'pcp.cli tmp.cli', b'null null')
    result, _ = attach_weather(pack(original), weather(), 'a')
    files = unpack(result)
    assert set(files['pcp.cli'].decode().splitlines()[2:]) == {'ecodata_a.pcp', 'old.pcp'}
    assert set(files['tmp.cli'].decode().splitlines()[2:]) == {'ecodata_a.tmp', 'old.tmp'}
    assert files['old.pcp'] == original['old.pcp']
    assert files['weather-sta.cli'].decode().splitlines()[3] == original['weather-sta.cli'].decode().splitlines()[3]


def test_uppercase_index_from_raw_builder_preserves_original_references():
    original=project()
    for index in ('pcp.cli','tmp.cli'):
        original[index]=original[index].replace(b'filename',b'FILENAME')
    result,_=attach_weather(pack(original),weather(),'a')
    files=unpack(result)
    assert files['pcp.cli'].splitlines()[1]==b'FILENAME'
    assert b'old.pcp' in files['pcp.cli'].splitlines()[2:]
    assert files['ecodata_a.pcp']==weather()['station.pcp']
    assert files['old.pcp']==original['old.pcp']


def test_rebuild_explicit_missing_index_from_all_station_references():
    from ecodata.models.swat_weather_project import repair_missing_weather_indices
    original = project()
    original['file.cio'] = original['file.cio'].replace(b'pcp.cli', b'local-rain-list.cli')
    original.pop('pcp.cli')
    repaired, report = repair_missing_weather_indices(pack(original))
    files = unpack(repaired)
    assert files.keys() - original.keys() == {'local-rain-list.cli'}
    assert all(files[n] == raw for n, raw in original.items())
    assert files['local-rain-list.cli'].splitlines()[2:] == [b'old.pcp']
    assert report['reconstructed_indices'][0]['file'] == 'local-rain-list.cli'
    attached, attachment = attach_weather(pack(original), weather(), 'a')
    result = unpack(attached)
    assert b'old.pcp' in result['local-rain-list.cli'].splitlines()[2:]
    assert b'ecodata_a.pcp' in result['local-rain-list.cli'].splitlines()[2:]
    assert attachment['index_repairs']['reconstructed_indices']


@pytest.mark.parametrize('nested,sim,extra', [(False,False,False), (True,True,True)])
def test_distinct_project_structures(nested, sim, extra):
    original = project(sim)
    payload = pack(original, 'watershed/' if nested else '')
    assert project_stations(payload) == ['a', 'b']
    result, report = attach_weather(payload, weather(extra), 'a')
    files = unpack(result)
    assert files['ecodata_a.pcp'] == weather()['station.pcp']
    assert files['ecodata_a.tmp'] == weather()['station.tem']
    assert files['old.pcp'] == original['old.pcp']
    assert files['soils.sol'] == original['soils.sol']
    assert files['time.sim'] == original['time.sim']
    row = files['weather-sta.cli'].decode().splitlines()[2].split()
    assert row[2:4] == ['ecodata_a.pcp', 'ecodata_a.tmp']
    assert files['weather-sta.cli'].decode().splitlines()[3].split() == original['weather-sta.cli'].decode().splitlines()[3].split()
    assert 'ecodata_a.pcp' in files['pcp.cli'].decode()
    if extra:
        assert files['file.cio'].decode().splitlines()[1].split()[6] == 'slr.cli'
    else:
        assert files['file.cio'] == original['file.cio']
    assert report['weather_records'] == 2


@pytest.mark.parametrize('fault,message', [('missing','天气引用缺失'), ('date','天气日期不足'),
                                          ('station','请选择'), ('index','索引重建缺少'),
                                          ('traversal','不安全路径'), ('duplicate','重复文件名')])
def test_reject_bad_projects(fault, message):
    files = project()
    station = 'a'
    if fault == 'missing': del files['old.tmp']
    if fault == 'date': files['time.sim'] = files['time.sim'].replace(b'2 2021 0', b'3 2021 0')
    if fault == 'station': station = 'no-such-station'
    if fault == 'index':
        files['file.cio'] = files['file.cio'].replace(b'pcp.cli', b'custom.cli')
        del files['old.pcp']
    if fault == 'traversal': files['../bad.txt'] = b'bad'
    if fault == 'duplicate': files['OLD.TMP'] = b'bad'
    with pytest.raises(ValueError, match=message):
        attach_weather(pack(files), weather(), station)


def test_missing_date_rejected():
    files = weather()
    files['station.pcp'] = files['station.pcp'].replace(b'2021 2', b'2021 3')
    with pytest.raises(ValueError, match='日期缺失'):
        attach_weather(pack(project()), files, 'a')


def test_referenced_custom_names_and_byte_preservation():
    original = project()
    names = {'weather-sta.cli': 'local-stations.cli', 'pcp.cli': 'rain-list.cli',
             'tmp.cli': 'temperature-list.cli', 'time.sim': 'period.sim'}
    cio = original['file.cio']
    for old, new in names.items():
        cio = cio.replace(old.encode(), new.encode())
        original[new] = original.pop(old)
    original['file.cio'] = cio + b'simulation period.sim print.prt null object.cnt null\n'
    result, report = attach_weather(pack(original), weather(), 'a')
    files = unpack(result)
    assert project_stations(result) == ['a', 'b']
    assert files['file.cio'] == original['file.cio']
    assert b'ecodata_a.pcp' in files['rain-list.cli']
    changed = {r['file'] for r in report['file_manifest'] if r['action'] != 'unchanged'}
    assert changed == {'local-stations.cli', 'rain-list.cli', 'temperature-list.cli',
                       'ecodata_a.pcp', 'ecodata_a.tmp'}
    for name in original.keys() - changed:
        assert files[name] == original[name]
    assert files['local-stations.cli'].splitlines()[3] == original['local-stations.cli'].splitlines()[3]
    assert report['resolved_layout']['time_file'] == 'period.sim'


@pytest.mark.parametrize('value', [b'nan', b'inf', b'-99'])
def test_invalid_weather_values_not_attached(value):
    incoming = weather()
    incoming['station.pcp'] = incoming['station.pcp'].replace(b'9.1', value)
    with pytest.raises(ValueError, match='数值'):
        attach_weather(pack(project()), incoming, 'a')


def test_empty_weather_not_attached():
    incoming = weather()
    incoming['station.pcp'] = b'title\nNBYR TSTEP LAT LONG ELEV\n1 0 1 2 3\n\n'
    with pytest.raises(ValueError, match='没有日记录'):
        attach_weather(pack(project()), incoming, 'a')


@pytest.mark.parametrize('old,new,message', [
    (b'NBYR TSTEP LAT LONG ELEV', b'NBYR TSTEP LONG LAT ELEV', '表头'),
    (b'1 0 1 2 3', b'1 1 1 2 3', 'TSTEP'),
    (b'1 0 1 2 3', b'1 0 91 2 3', '经纬度'),
    (b'1 0 1 2 3', b'1 0 1 2 nan', '经纬度'),
    (b'1 0 1 2 3', b'2 0 1 2 3', '年份数'),
])
def test_weather_metadata_contract(old, new, message):
    incoming = weather()
    incoming['station.pcp'] = incoming['station.pcp'].replace(old, new)
    with pytest.raises(ValueError, match=message):
        attach_weather(pack(project()), incoming, 'a')


@pytest.mark.parametrize('missing_index', [False, True])
def test_current_page_shows_manifest_and_blocks_dates_outside_new_weather(missing_index):
    from datetime import date
    from types import SimpleNamespace
    from unittest.mock import patch
    from streamlit.testing.v1 import AppTest
    from ecodata.interface import swat_runner
    original = project()
    if missing_index:
        original.pop('pcp.cli')
    uploads = {'上传完整工程 ZIP（包含file.cio及其实际引用文件）': pack(original),
               '可选：上传刚从平台下载的日天气转换ZIP': pack(weather())}
    def upload(label, **kwargs):
        raw = uploads.get(label)
        return SimpleNamespace(getvalue=lambda: raw) if raw else None
    checks = dict(errors=[], warnings=[], declared_major=62,
                  dates=dict(yrc_start=2021, day_start=1, yrc_end=2021, day_end=2))
    with patch('streamlit.file_uploader', side_effect=upload), \
            patch.object(swat_runner, 'engine_path', return_value=Path(__file__)), \
            patch.object(swat_runner, 'check_project', return_value=checks), \
            patch.object(swat_runner, 'recent_jobs', return_value=[]), \
            patch.object(swat_runner, 'start_job') as start:
        app = AppTest.from_string('''
from ecodata.interface.swat_run_page import render_swat_run_page
render_swat_run_page()
''', default_timeout=30).run()
        next(s for s in app.selectbox if '哪个现有站点' in s.label).set_value('a').run()
        next(c for c in app.checkbox if '确认新天气' in c.label).check().run()
        assert not app.exception and not app.error
        assert any('接入核查清单JSON' in d.label for d in app.get('download_button'))
        if missing_index:
            assert any('按工程原站点引用重建缺失索引：pcp.cli' in i.value for i in app.info)
        next(d for d in app.date_input if d.label == '结束日期').set_value(date(2021, 1, 3)).run()
        next(c for c in app.checkbox if '我的工程适用于' in c.label).check().run()
        next(b for b in app.button if b.label == '启动 SWAT+').click().run()
        assert not app.exception
        assert any('超出本次新天气覆盖范围' in e.value for e in app.error)
        start.assert_not_called()


def test_current_page_blocks_incomplete_wgn_before_station_confirmation():
    from types import SimpleNamespace
    from unittest.mock import patch
    from streamlit.testing.v1 import AppTest
    from ecodata.interface import swat_runner
    files = project()
    files['file.cio'] = files['file.cio'].replace(b'weather-sta.cli null null', b'weather-sta.cli generator.cli null')
    files['weather-sta.cli'] = files['weather-sta.cli'].replace(b'a null ', b'a g ')
    files['generator.cli'] = ('fixture\ng 1 2 3 10\n'+' '.join('field'+str(i) for i in range(14))+'\n'+' '.join(['1']*14)+'\n').encode()
    uploads = {'上传完整工程 ZIP（包含file.cio及其实际引用文件）': pack(files),
               '可选：上传刚从平台下载的日天气转换ZIP': pack(weather())}
    def upload(label, **kwargs):
        raw = uploads.get(label)
        return SimpleNamespace(getvalue=lambda: raw) if raw else None
    with patch('streamlit.file_uploader', side_effect=upload), \
            patch.object(swat_runner, 'engine_path', return_value=Path(__file__)), \
            patch.object(swat_runner, 'recent_jobs', return_value=[]), \
            patch.object(swat_runner, 'start_job') as start:
        app = AppTest.from_string('''
from ecodata.interface.swat_run_page import render_swat_run_page
render_swat_run_page()
''', default_timeout=30).run()
        assert not app.exception
        assert any('12个月统计，实际只有1行' in e.value for e in app.error)
        assert not any('哪个现有站点' in s.label for s in app.selectbox)
        start.assert_not_called()


def test_ui_attachment_calls_existing_writer():
    from types import SimpleNamespace
    from unittest.mock import patch
    from streamlit.testing.v1 import AppTest
    case = ROOT/'examples/ai_round1/swat_plus'
    original = project()
    original['time.sim'] = original['time.sim'].replace(b'2 2021', b'365 2021')
    uploads = {'导入源数据': (case/'daymet_2021_complete.nc').read_bytes(),
               '导入字段、单位与版本配置JSON': (case/'config_ui.json').read_bytes(),
               '上传已有流域工程ZIP（不是天气转换ZIP）': pack(original)}
    def upload(label, **kwargs):
        value = uploads.get(label)
        return SimpleNamespace(getvalue=lambda: value) if value else None
    with patch('streamlit.file_uploader', side_effect=upload):
        # This test exercises the attachment helper's writer and invalidation
        # contract. The current full-page continuation has its own tests in
        # test_conversion_continuation.py; it no longer renders this helper.
        app = AppTest.from_string('''
from pathlib import Path
import json
import streamlit as st
from ecodata.round1 import convert
from ecodata.interface.round1_page import render_weather_attachment
root = Path(ROOT_PATH)
files, report = convert('swat_plus', (root/'examples/ai_round1/swat_plus/daymet_2021_complete.nc').read_bytes(), json.loads((root/'examples/ai_round1/swat_plus/config_ui.json').read_text()))
st.session_state['round1_result'] = (files, report)
render_weather_attachment(files)
'''.replace('ROOT_PATH', repr(str(ROOT))), default_timeout=30).run()
        next(s for s in app.selectbox if '哪个站点' in s.label).set_value('a').run()
        app.checkbox[0].check().run()
        next(b for b in app.button if b.label == '自动接入并检查工程').click().run()
        assert not app.exception and not app.error
        result, report = app.session_state['swat_attach_result']
        assert unpack(result)['ecodata_a.pcp'] == app.session_state['round1_result'][0]['station.pcp']
        assert report['weather_records'] == 365
        # A different target must invalidate the previous download.
        next(s for s in app.selectbox if '哪个站点' in s.label).set_value('b').run()
        assert not any('接好天气' in d.label for d in app.get('download_button'))
