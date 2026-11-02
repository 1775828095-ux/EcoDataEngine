import copy
import csv
import io
import json
import zipfile
import pytest
from ecodata.interface.swat_build import read_inputs


def bundle(prefix='raw/'):
    config = {'target_version': '62.0.0', 'accept_reference_parameters': True,
              'stream_cells': 200, 'channel_cells': 100, 'snap_metres': 300, 'slope_limits': [5, 15]}
    files = {}
    for key in ('dem', 'landuse', 'soil', 'outlets', 'landuse_lookup', 'soil_lookup', 'usersoil', 'wgn_stations', 'wgn_months'):
        config[key] = prefix + key + ('.shp' if key == 'outlets' else '.tif' if key in {'dem', 'landuse', 'soil'} else '.csv')
        files[config[key]] = b'synthetic structural fixture, not a real raster'
    for ext in ('.shx', '.dbf', '.prj'):
        files[config['outlets'][:-4] + ext] = b'synthetic shape component'
    for key in ('landuse_lookup', 'soil_lookup'):
        files[config[key]] = b'ID,NAME\n1,parameter1\n'
    files[config['usersoil']] = (','.join('c' + str(i) for i in range(152)) + '\n' + ','.join('1' for _ in range(152)) + '\n').encode()
    files[config['wgn_stations']] = b'id,name,lat,lon,elev,rain_yrs\n1,test,12,38,1000,20\n'
    fields = 'tmp_max_ave tmp_min_ave tmp_max_sd tmp_min_sd pcp_ave pcp_sd pcp_skew wet_dry wet_wet pcp_days pcp_hhr slr_ave dew_ave wnd_ave'.split()
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(['id', 'wgn_id', 'month', *fields])
    for month in range(1, 13):
        writer.writerow([month, 1, month, *([1] * 14)])
    files[config['wgn_months']] = out.getvalue().encode()
    return files, config


def pack(files):
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w') as z:
        for name, content in files.items():
            z.writestr(name, content)
    return out.getvalue()


@pytest.mark.parametrize('prefix', ['', 'case/inputs/'])
def test_structures(prefix):
    files, config = bundle(prefix)
    assert read_inputs(pack(files), config) == files


@pytest.mark.parametrize('weather_config',[{'target_model_version':'61.0.2'},None])
def test_new_build_rejects_mismatched_weather_version_before_conversion(weather_config):
    from unittest.mock import patch
    from ecodata.interface.swat_build import start_build
    files,config=bundle()
    with patch('ecodata.round1.convert') as convert:
        with pytest.raises(ValueError,match='天气转换目标版本'):
            start_build(pack(files),config,b'not executed',weather_config)
        convert.assert_not_called()


@pytest.mark.parametrize('fault',[None,'depth_missing','depth_invalid','sidecar_missing'])
def test_explicit_stream_burn_inputs(fault):
    files,cfg=bundle()
    cfg.update(burn_streams='raw/streams.shp',burnin_depth_metres=50)
    for ext in ('.shp','.shx','.dbf','.prj'):files['raw/streams'+ext]=b'structural fixture only'
    if fault=='depth_missing':cfg.pop('burnin_depth_metres')
    if fault=='depth_invalid':cfg['burnin_depth_metres']=True
    if fault=='sidecar_missing':files.pop('raw/streams.prj')
    if fault:
        with pytest.raises(ValueError):read_inputs(pack(files),cfg)
    else:assert read_inputs(pack(files),cfg)==files


@pytest.mark.parametrize('fault,match', [('missing', '缺少'), ('months', '12个月'),
    ('unsafe', '不安全'), ('template', '已经写好'), ('soil', '152列'), ('version', '62.0.0'),
    ('defaults', '确认'), ('threshold', '正整数'), ('nan', '无效字段')])
def test_rejections(fault, match):
    files, config = bundle()
    if fault == 'missing':
        files.pop(config['soil'])
    elif fault == 'months':
        files[config['wgn_months']] = b'\n'.join(files[config['wgn_months']].splitlines()[:-1])
    elif fault == 'unsafe':
        files['../bad.csv'] = b'bad'
    elif fault == 'template':
        files['file.cio'] = b'not raw'
    elif fault == 'soil':
        files[config['usersoil']] = b'name,depth\nsoil,1\n'
    elif fault == 'version':
        config['target_version'] = '60.5.7'
    elif fault == 'defaults':
        config['accept_reference_parameters'] = False
    elif fault == 'threshold':
        config['snap_metres'] = 0
    elif fault == 'nan':
        files[config['wgn_months']] = files[config['wgn_months']].replace(b',1,1,1,1,1,', b',1,1,nan,1,1,', 1)
    with pytest.raises(ValueError, match=match):
        read_inputs(pack(files), config)


def test_new_route_ui():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string('from ecodata.interface.swat_run_page import render_swat_run_page\nrender_swat_run_page()').run()
    app.radio(key='swat_route').set_value('没有工程，从原始资料开始').run()
    assert not app.exception
    assert any('试验功能' in w.value for w in app.warning)
    assert next(b for b in app.button if b.label == '检查资料并建立新工程').disabled


def test_raw_route_does_not_show_existing_project_result():
    from unittest.mock import patch
    from streamlit.testing.v1 import AppTest
    with patch('ecodata.interface.swat_run_page._render_job') as render:
        app = AppTest.from_string('from ecodata.interface.swat_build_page import render_build_page\nrender_build_page()')
        app.session_state['swat_active_job'] = 'other-route-result'
        app.run()
        assert not app.exception
        render.assert_not_called()


def test_raw_input_change_clears_old_build_and_result():
    from unittest.mock import patch
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string('from ecodata.interface.swat_build_page import render_build_page\nrender_build_page()')
    app.session_state['swat_build_inputs_fingerprint'] = 'previous-inputs'
    app.session_state['swat_build_job'] = 'previous-build'
    app.session_state['swat_build_active_job'] = 'previous-run'
    with patch('ecodata.interface.swat_build_page.render_build_job') as build, patch('ecodata.interface.swat_run_page._render_job') as render:
        app.run()
        assert not app.exception
        build.assert_not_called()
        render.assert_not_called()


def test_delineation_failure_stops_before_gdal(tmp_path):
    from ecodata.interface.swat_gis_worker import require_delineation_output
    for path in ('', str(tmp_path / 'missing.tif')):
        with pytest.raises(RuntimeError, match='0 snapped: 1 failed'):
            require_delineation_output(path, '0 snapped: 1 failed')


def test_delineation_output_exists(tmp_path):
    from ecodata.interface.swat_gis_worker import require_delineation_output
    output = tmp_path / 'basin.tif'
    output.write_bytes(b'structural fixture only')
    require_delineation_output(str(output), '1 snapped')


def test_failed_snap_reports_main_stream_distance_without_changing_threshold():
    from ecodata.interface.swat_gis_worker import require_delineation_output
    diagnostics = {'snap_metres': 300, 'points': [
        {'id': 7, 'stream_distance_m': 534.306, 'channel_distance_m': 167.759}]}
    with pytest.raises(RuntimeError, match='7=534.306'):
        require_delineation_output('', '0 snapped: 1 failed', diagnostics)
    assert diagnostics['snap_metres'] == 300


def test_existing_route_defaults_to_verified_editor_and_warns_for_gnu():
    from unittest.mock import patch
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    with patch('ecodata.interface.swat_runner.engine_path', return_value=Path('trusted.exe')):
        app = AppTest.from_string('from ecodata.interface.swat_run_page import render_swat_run_page\nrender_swat_run_page()').run()
        assert app.selectbox(key='run_engine').value == 'editor62'
        app.selectbox(key='run_engine').set_value('gnu62').run()
        assert not app.exception
        assert any('浮点异常' in w.value for w in app.warning)


def test_worker_failure_reaches_ui_status_without_generic_command_only(tmp_path,monkeypatch):
    import subprocess
    from ecodata.interface import swat_build
    reason='出口到主河网534.306米，超过300米；不移动出口'
    def fail(*args,**kwargs):
        (tmp_path/'worker_error.json').write_text(json.dumps({'error':reason}),encoding='utf8')
        raise subprocess.CalledProcessError(1,args[0])
    monkeypatch.setattr(swat_build.subprocess,'run',fail)
    job={'directory':str(tmp_path),'id':'test','state':'running'}
    swat_build._work(job,{}, {})
    assert job['state']=='failed' and job['error']==reason
    assert json.loads((tmp_path/'build_status.json').read_text(encoding='utf8'))['error']==reason
