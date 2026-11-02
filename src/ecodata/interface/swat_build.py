"""Raw-data SWAT+ build coordinator. Official tools run in private workspaces."""
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import sqlite3
import subprocess
import threading
import uuid
import zipfile

TOOLS = Path(r'D:\EcoDataEngine\tools')
RUNS = Path(r'D:\EcoDataEngine\runs\swat_build')
JOBS = {}
LOCK = threading.Lock()
PATHS = ('dem', 'landuse', 'soil', 'outlets', 'landuse_lookup', 'soil_lookup',
         'usersoil', 'wgn_stations', 'wgn_months')
WGN_FIELDS = ('tmp_max_ave', 'tmp_min_ave', 'tmp_max_sd', 'tmp_min_sd',
              'pcp_ave', 'pcp_sd', 'pcp_skew', 'wet_dry', 'wet_wet',
              'pcp_days', 'pcp_hhr', 'slr_ave', 'dew_ave', 'wnd_ave')


def editor_wgn_csv(stations_raw, months_raw):
    """Map named source fields to Editor two_file's positional CSV contract.

    The official importer ignores the first ID column, then reads its model
    order: generator FK, month, fourteen values. A raw database dump commonly
    places wgn_id LAST; sending that dump directly silently shifts every value.
    Scientific values and station IDs are copied as strings without defaults.
    """
    stations = list(csv.DictReader(io.StringIO(stations_raw.decode('utf-8-sig'))))
    months = list(csv.DictReader(io.StringIO(months_raw.decode('utf-8-sig'))))
    ids = [r['id'] for r in stations]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('WGN统计站ID为空或重复')
    for ident in ids:
        rows = [r for r in months if r['wgn_id'] == ident]
        if len(rows) != 12 or sorted(int(r['month']) for r in rows) != list(range(1, 13)):
            raise ValueError(f'WGN统计站{ident}需要不重复的12个月')
    if any(r['wgn_id'] not in ids for r in months):
        raise ValueError('WGN月份引用不存在的统计站')
    station_out, month_out = io.StringIO(), io.StringIO()
    writer = csv.writer(station_out, lineterminator='\n')
    cols = ('id', 'name', 'lat', 'lon', 'elev', 'rain_yrs')
    writer.writerow(cols)
    writer.writerows([[r[k] for k in cols] for r in stations])
    writer = csv.writer(month_out, lineterminator='\n')
    writer.writerow(['id', 'weather_wgn_cli_id', 'month', *WGN_FIELDS])
    for number, row in enumerate(months, 1):
        if not all(math.isfinite(float(row[k])) for k in WGN_FIELDS):
            raise ValueError('WGN月统计包含非有限值')
        writer.writerow([row.get('id', str(number)), row['wgn_id'], row['month'], *[row[k] for k in WGN_FIELDS]])
    return station_out.getvalue().encode('utf-8'), month_out.getvalue().encode('utf-8')


def read_inputs(payload, config):
    """Validate archive and explicit settings before any tool or model starts."""
    if not isinstance(config, dict):
        raise ValueError('建工程配置必须是对象')
    allowed = {*PATHS, 'plant', 'urban', 'stream_cells', 'channel_cells', 'snap_metres',
               'slope_limits', 'accept_reference_parameters', 'target_version', 'burn_streams', 'burnin_depth_metres'}
    if set(config) - allowed:
        raise ValueError('不支持的建工程设置：' + ', '.join(sorted(set(config) - allowed)))
    for key in (*PATHS, 'stream_cells', 'channel_cells', 'snap_metres', 'slope_limits'):
        if key not in config:
            raise ValueError('缺少建工程资料或设置：' + key)
    if config.get('target_version') != '62.0.0':
        raise ValueError('当前建工程目标仅支持SWAT+62.0.0')
    if config.get('accept_reference_parameters') is not True:
        raise ValueError('需确认使用随工具提供的参考参数；它们不是当地实测或率定参数')
    if ('burn_streams' in config) != ('burnin_depth_metres' in config):
        raise ValueError('河网刻蚀需要同时提供burn_streams与明确burnin_depth_metres，不猜测深度')
    if 'burn_streams' in config and (type(config['burnin_depth_metres']) is not int or not 0 < config['burnin_depth_metres'] <= 1000):
        raise ValueError('burnin_depth_metres必须为1..1000米整数；它是明确的预处理设置')
    for key in ('stream_cells', 'channel_cells', 'snap_metres'):
        n = config[key]
        if type(n) is not int or not 0 < n <= 100000000:
            raise ValueError(key + '必须为正整数')
    if config['channel_cells'] > config['stream_cells']:
        raise ValueError('河道阈值不能大于子流域河网阈值')
    limits = config['slope_limits']
    if not isinstance(limits, list) or any(type(v) not in (int, float) or not 0 < v < 1000 for v in limits) or limits != sorted(set(limits)):
        raise ValueError('坡度分级必须是递增正数列表，单位百分比；空列表代表不分级')
    if len(payload) > 100 * 1024**2:
        raise ValueError('原始资料包不能超过100 MiB')
    files = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries = archive.infolist()
        if len(entries) > 3000 or sum(i.file_size for i in entries) > 100 * 1024**2:
            raise ValueError('原始资料解压大小或数量超限')
        seen = set()
        for item in entries:
            name = item.filename
            p = PurePosixPath(name)
            reserved = {'con', 'prn', 'aux', 'nul', *(f'com{i}' for i in range(1, 10)), *(f'lpt{i}' for i in range(1, 10))}
            if (not p.parts or p.is_absolute() or '..' in p.parts or '\\' in name or any(c in name for c in ':<>"|?*')
                    or any(ord(c) < 32 for c in name)
                    or any(v.endswith(('.', ' ')) or v.split('.')[0].lower() in reserved for v in p.parts)
                    or (item.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError('资料包有不安全路径')
            if item.is_dir():
                continue
            if name.casefold() in seen:
                raise ValueError('资料包存在重名文件')
            seen.add(name.casefold())
            if p.suffix.lower() in {'.exe', '.dll', '.bat', '.cmd', '.ps1', '.py', '.sh', '.lnk', '.com', '.scr'}:
                raise ValueError('原始资料包不能包含程序或脚本')
            if p.name.lower() in {'file.cio', 'hru.con', 'aquifer.con'}:
                raise ValueError('从零建工程入口不接受已经写好的模型工程文件')
            files[name] = archive.read(item)
    for key in (*PATHS, *[k for k in ('plant', 'urban', 'burn_streams') if k in config]):
        path = config[key]
        if not isinstance(path, str) or path != PurePosixPath(path).as_posix():
            raise ValueError(key + '必须为ZIP内相对路径')
        # ArcInfo grids are directories containing hdr.adf; GeoTIFF is a file.
        if path not in files and not (key in {'dem', 'landuse', 'soil'} and path + '/hdr.adf' in files):
            raise ValueError('资料包缺少：' + key + ' → ' + path)
    outlet = config['outlets']
    if not outlet.endswith('.shp') or any(outlet[:-4] + ext not in files for ext in ('.shx', '.dbf', '.prj')):
        raise ValueError('出口需要完整shp/shx/dbf/prj文件')
    if 'burn_streams' in config:
        burn=config['burn_streams']
        if not burn.endswith('.shp') or any(burn[:-4]+ext not in files for ext in ('.shx','.dbf','.prj')):
            raise ValueError('刻蚀河网需要完整shp/shx/dbf/prj线图层')
    soil_rows = list(csv.reader(io.StringIO(files[config['usersoil']].decode('utf-8-sig'))))
    if not soil_rows or any(len(r) != 152 for r in soil_rows if r):
        raise ValueError('本批仅支持完整152列usersoil CSV；分层双表尚未接入，不能自动填参数')
    for key in ('landuse_lookup', 'soil_lookup'):
        rows = list(csv.reader(io.StringIO(files[config[key]].decode('utf-8-sig'))))
        if len(rows) < 2 or any(len(r) != 2 or any(not v.strip() for v in r) for r in rows if r):
            raise ValueError(key + '需要两列完整CSV（分类编号、参数名称）')
        ids = [int(r[0]) for r in rows[1:]]
        if len(ids) != len(set(ids)):
            raise ValueError(key + '分类编号重复')
    months = list(csv.DictReader(io.StringIO(files[config['wgn_months']].decode('utf-8-sig'))))
    stations = list(csv.DictReader(io.StringIO(files[config['wgn_stations']].decode('utf-8-sig'))))
    ids = [r['id'] for r in stations]
    if not ids or len(ids) != len(set(ids)) or any(r.get('wgn_id') not in ids for r in months):
        raise ValueError('WGN站点与月份引用不明确')
    for station in ids:
        rows = [r for r in months if r.get('wgn_id') == station]
        if len(rows) != 12 or sorted(int(r['month']) for r in rows) != list(range(1, 13)):
            raise ValueError('WGN站点' + station + '必须提供不重复的12个月')
    fields = ('tmp_max_ave', 'tmp_min_ave', 'tmp_max_sd', 'tmp_min_sd', 'pcp_ave', 'pcp_sd', 'pcp_skew',
              'wet_dry', 'wet_wet', 'pcp_days', 'pcp_hhr', 'slr_ave', 'dew_ave', 'wnd_ave')
    for row in months:
        for field in fields:
            if field not in row or not math.isfinite(float(row[field])) or float(row[field]) == -99:
                raise ValueError('WGN月统计缺少或无效字段：' + field)
    return files


def qgis_environment():
    root = TOOLS / 'QGIS340/QGIS 3.40.9'
    env = os.environ.copy()
    env.update(OSGEO4W_ROOT=str(root), PYTHONHOME=str(root / 'apps/Python312'),
               PYTHONPATH=os.pathsep.join([str(root / 'apps/qgis-ltr/python')]),
               QGIS_PREFIX_PATH=str(root / 'apps/qgis-ltr'),
               QT_PLUGIN_PATH=os.pathsep.join([str(root / 'apps/qgis-ltr/qtplugins'), str(root / 'apps/qt5/plugins')]),
               QT_QPA_PLATFORM='offscreen', GDAL_DATA=str(root / 'share/gdal'),
               PROJ_DATA=str(root / 'share/proj'), PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    env['PATH'] = os.pathsep.join([str(root / 'bin'), str(root / 'apps/qgis-ltr/bin'), str(root / 'apps/qt5/bin'), env['PATH']])
    return root / 'bin/python.exe', env


def start_build(payload, config, weather, weather_config):
    files = read_inputs(payload, config)
    if not isinstance(weather_config, dict) or str(weather_config.get('target_model_version', '62.0.0')) != '62.0.0':
        raise ValueError('新建工程目标为62.0.0，天气转换目标版本必须同为62.0.0；请修改配置后重新转换，不自动迁移版本')
    from ecodata.round1 import convert
    converted, report = convert('swat_plus', weather, weather_config)
    if set(report['additional_weather']) != {'solar_radiation', 'relative_humidity', 'wind_speed'}:
        raise ValueError('从零建工程首批要求完整降水、最高最低温、辐射、湿度和风速；不隐式生成缺失天气')
    with LOCK:
        if any(j['state'] == 'running' for j in JOBS.values()):
            raise ValueError('已有建工程任务运行中，请先等待')
        ident = uuid.uuid4().hex
        directory = RUNS / ident
        directory.mkdir(parents=True, exist_ok=False)
        raw = directory / 'raw'
        for name, content in files.items():
            dest = raw / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
        weather_dir = directory / 'weather'
        weather_dir.mkdir()
        for name, content in converted.items():
            (weather_dir / name).write_bytes(content)
        worker_config = {k: config[k] for k in ('stream_cells', 'channel_cells', 'snap_metres', 'slope_limits')}
        worker_config.update({k: str(raw / config[k]) for k in (*PATHS, *[k for k in ('plant', 'urban', 'burn_streams') if k in config])})
        if 'burn_streams' in config:worker_config['burnin_depth_metres']=config['burnin_depth_metres']
        worker_config.update(output=str(directory / 'project'), support_dir=str(TOOLS / 'SWATPlus'), plugin_dir=str(TOOLS / 'SWATPlus/QSWATPlus'))
        (directory / 'build_config.json').write_text(json.dumps(worker_config), encoding='utf-8')
        (directory / 'inputs.json').write_text(json.dumps({'configuration': config, 'weather_configuration': weather_config,
            'raw_zip_sha256': hashlib.sha256(payload).hexdigest(), 'weather_sha256': hashlib.sha256(weather).hexdigest(),
            'reference_parameters': 'Official Editor 4.0.2 defaults, explicitly accepted; not site calibration'}, ensure_ascii=False, indent=2), encoding='utf-8')
        job = dict(id=ident, state='running', stage='准备', directory=str(directory))
        JOBS[ident] = job
        _save_status(job)
        threading.Thread(target=_work, args=(job, worker_config, weather_config), daemon=True).start()
    return ident


def _work(job, config, weather_config):
    directory = Path(job['directory'])
    api = TOOLS / 'SWATPlusEditor402/resources/app.asar.unpacked/static/api_dist/swatplus_api.exe'
    def run(args, stage, env=None):
        job['stage'] = stage
        _save_status(job)
        with (directory / 'build.log').open('ab') as log:
            log.write(('\nSTAGE: ' + stage + '\n').encode('utf-8'))
            subprocess.run([str(a) for a in args], stdout=log, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, timeout=900, check=True, env=env,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    try:
        python, env = qgis_environment()
        worker = Path(__file__).with_name('swat_gis_worker.py')
        run([python, worker, directory / 'build_config.json'], '从原始GIS划分流域与HRU', env)
        project = directory / 'project'
        db = project / 'Watershed.sqlite'
        run([api, 'setup_project', '--project_db_file', db, '--editor_version', '4.0.2', '--project_name', 'Watershed',
             '--datasets_db_file', project / 'swatplus_datasets.sqlite', '--is_lte', 'n', '--constant_ps', 'n', '--copy_datasets_db', 'y'], '导入工程参数与连接')
        station_csv, month_csv = editor_wgn_csv(Path(config['wgn_stations']).read_bytes(), Path(config['wgn_months']).read_bytes())
        station_path, month_path = directory / 'editor_wgn_stations.csv', directory / 'editor_wgn_months.csv'
        station_path.write_bytes(station_csv)
        month_path.write_bytes(month_csv)
        run([api, 'import_weather', '--project_db_file', db, '--import_type', 'wgn', '--import_method', 'two_file',
             '--file1', station_path, '--file2', month_path, '--delete_existing', 'y', '--create_stations', 'n'], '导入用户WGN统计')
        with sqlite3.connect(db) as conn:
            conn.execute('UPDATE project_config SET weather_data_dir=?,input_files_dir=?,weather_data_format=?',
                         (str(directory / 'weather'), str(project / 'Scenarios/Default/TxtInOut'), 'plus'))
        run([api, 'import_weather', '--project_db_file', db, '--import_type', 'observed', '--delete_existing', 'y', '--create_stations', 'y'], '关联平台天气')
        import datetime as dt
        first, last = (dt.date.fromisoformat(weather_config[k]) for k in ('start', 'end'))
        with sqlite3.connect(db) as conn:
            conn.execute('UPDATE time_sim SET day_start=?,yrc_start=?,day_end=?,yrc_end=?,step=0',
                         (first.timetuple().tm_yday, first.year, last.timetuple().tm_yday, last.year))
        run([api, 'write_files', '--project_db_file', db, '--swat_version', '62.0.0'], '写出模型输入')
        from ecodata.interface import swat_runner as runner
        inputs = {p.name: p.read_bytes() for p in (project / 'Scenarios/Default/TxtInOut').iterdir() if p.is_file()}
        report = runner.check_project(inputs)
        if report['errors']:
            raise ValueError('；'.join(report['errors']))
        with zipfile.ZipFile(directory / 'project.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            for name, content in inputs.items():
                z.writestr(name, content)
        job.update(state='ready', stage='已生成工程，尚未运行模型', report=report)
    except Exception as exc:
        reason = str(exc)
        if isinstance(exc, subprocess.CalledProcessError) and job['stage'] == '从原始GIS划分流域与HRU':
            detail = directory / 'worker_error.json'
            if detail.is_file():
                try:
                    reason = json.loads(detail.read_text(encoding='utf-8'))['error']
                except (ValueError, KeyError):
                    pass  # Original process error and full log remain available.
        job.update(state='failed', error=reason)
    finally:
        _save_status(job)


def _save_status(job):
    directory=Path(job['directory'])
    temporary=directory/'build_status.tmp'
    temporary.write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(directory/'build_status.json')


def status(ident):
    with LOCK:
        if ident in JOBS:return dict(JOBS[ident])
    import re
    if not isinstance(ident,str) or not re.fullmatch('[0-9a-f]{32}',ident):raise ValueError('无效建工程任务号')
    directory=RUNS/ident
    record=json.loads((directory/'build_status.json').read_text(encoding='utf-8'))
    if record.get('id')!=ident or Path(record.get('directory','')).resolve()!=directory.resolve():raise ValueError('建工程记录与保存位置不匹配')
    return record
