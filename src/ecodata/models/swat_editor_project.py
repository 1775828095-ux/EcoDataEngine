"""Public GIS/Editor database -> official SWAT+ input export, on copied inputs."""
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
import csv
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import zipfile

from .project_binding import ProjectFiles

EDITOR_COMMIT = '3606ac4c3b54585072a557382b8e3b60b44dd354'
WGN_FIELDS='tmp_max_ave tmp_min_ave tmp_max_sd tmp_min_sd pcp_ave pcp_sd pcp_skew wet_dry wet_wet pcp_days pcp_hhr slr_ave dew_ave wnd_ave'.split()


def _config(path):
    with sqlite3.connect('file:'+path.as_posix()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
        if 'project_config' not in tables:
            return None
        rows = db.execute('select * from project_config').fetchall()
        if len(rows) != 1:
            raise ValueError('SWAT+工程数据库需唯一project_config')
        return dict(rows[0]), tables


def validate_wgn_database(path, table):
    """The official importer requires table/table_mon and wgn_id linkage."""
    if not isinstance(table,str) or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',table) is None:
        raise ValueError('WGN统计表名未声明或不安全')
    with sqlite3.connect('file:'+path.as_posix()+'?mode=ro',uri=True) as db:
        parent={r[1] for r in db.execute('pragma table_info('+table+')')}
        child={r[1] for r in db.execute('pragma table_info('+table+'_mon)')}
        if not {'id','name','lat','lon','elev','rain_yrs'}<=parent or not {'wgn_id','month'}<=child:
            raise ValueError('WGN数据库不符合官方统计表/月份外键契约')
        ids=[r[0] for r in db.execute('select id from '+table)]
        if not ids:raise ValueError('WGN统计表为空，不能生成统计资料')
        fields=WGN_FIELDS
        if not set(fields)<=child:raise ValueError('WGN数据库缺少14项真实月统计字段')
        for ident in ids:
            values=list(db.execute('select month,'+','.join(fields)+' from '+table+'_mon where wgn_id=?',(ident,)))
            months=[r[0] for r in values]
            if sorted(months)!=list(range(1,13)):
                raise ValueError('WGN统计站缺少唯一的12个月资料：'+str(ident))
            if any(not isinstance(v,(int,float)) or not math.isfinite(v) for r in values for v in r[1:]):
                raise ValueError('WGN月统计字段缺失或非有限，不能编造统计值')


def find_wgn_csv(files, root):
    """Recognise official two-table content and keys, never case filenames."""
    stations=[];monthly=[]
    for name in sorted(files.files):
        if not name.lower().endswith('.csv') or not PurePosixPath(name).is_relative_to(root):continue
        with files.files[name][0]() as stream:
            reader=csv.DictReader(stream.read().decode('utf-8-sig').splitlines())
            columns=set(reader.fieldnames or [])
            if {'id','name','lat','lon','elev','rain_yrs'}<=columns:
                stations.append((name,list(reader)))
            elif {'id','month',*WGN_FIELDS}<=columns:
                key=next((k for k in ['wgn_id','weather_wgn_cli_id','weather_wgn_cli'] if k in columns),None)
                if key:monthly.append((name,list(reader),key))
    if not stations and not monthly:return None
    if len(stations)!=1 or len(monthly)!=1:raise ValueError('WGN CSV统计站/月表配对不唯一或缺表，不能按文件名猜选')
    stat,rows=stations[0];month,values,key=monthly[0]
    ids=[r['id'] for r in rows]
    if not ids or len(set(ids))!=len(ids) or len({r['name'] for r in rows})!=len(rows):
        raise ValueError('WGN CSV站点为空或身份重复')
    if {r[key] for r in values}!=set(ids):raise ValueError('WGN CSV月份外键与统计站不对应')
    for row in rows:
        latitude,longitude,elevation,years=[float(row[k]) for k in ['lat','lon','elev','rain_yrs']]
        if not all(math.isfinite(x) for x in [latitude,longitude,elevation,years]) or not -90<=latitude<=90 or not -180<=longitude<=180 or years<1:
            raise ValueError('WGN CSV站点坐标或统计年数不合法')
        selected=[r for r in values if r[key]==row['id']]
        if sorted(int(r['month']) for r in selected)!=list(range(1,13)):
            raise ValueError('WGN CSV统计站需完整且唯一的12个月资料')
        if any(not math.isfinite(float(r[k])) for r in selected for k in WGN_FIELDS):
            raise ValueError('WGN CSV缺少有限的真实月统计，不填造资料')
    return dict(stations=stat,monthly=month,station_rows=rows,month_rows=values,foreign_key=key)


def prepare_editor_project(uploads, model_id, folder, progress=None):
    if model_id != 'swat_plus':
        return None
    folder = Path(folder)
    with ExitStack() as stack:
        files = ProjectFiles(uploads, stack)
        if any(PurePosixPath(n).name == 'file.cio' for n in files.files):
            return None
        sqlite_names = [n for n in sorted(files.files) if files.prefix(n,16)==b'SQLite format 3\x00']
        if not sqlite_names:
            return None
        probe = folder/'editor_probe'; probe.mkdir()
        candidates = []
        for i,name in enumerate(sqlite_names):
            path=probe/(str(i)+'.sqlite')
            with files.files[name][0]() as src, path.open('wb') as dst:
                shutil.copyfileobj(src,dst,8*1024**2)
            facts=_config(path)
            if facts is not None:
                candidates.append((name,*facts))
        if len(candidates)!=1:
            raise ValueError('需唯一SWAT+编辑工程数据库，不按文件名猜选')
        name,config,tables=candidates[0]
        if config.get('gis_type')!='qgis' or not str(config.get('gis_version','')).startswith('4.'):
            raise ValueError('当前官方导出适配仅核验QSWAT+4.x数据库；原始版本不改写')
        if config.get('use_gwflow'):
            raise ValueError('当前导出缺少GWFlow外部文件安全绑定契约，不生成不完整成功包')
        if config.get('editor_version') not in {None,'4.0.0'}:
            raise ValueError('当前官方导出契约为Editor4.0.0，不自动升级其他编辑版本')
        datasets,_=files.resolve(name,config.get('reference_db') or '')
        wgn=None
        if config.get('wgn_db'):
            wgn,_=files.resolve(name,config['wgn_db'])
            with files.files[wgn][0]() as src,(probe/'wgn.sqlite').open('wb') as dst:
                shutil.copyfileobj(src,dst,8*1024**2)
            validate_wgn_database(probe/'wgn.sqlite',config.get('wgn_table_name'))
        wgn_csv=None if wgn else find_wgn_csv(files,PurePosixPath(name).parent)
        # Declared project paths are rebased only into this uploaded workspace.
        weather_source=None;weather_format=None
        declared=config.get('weather_data_dir')
        if declared:
            weather_format=config.get('weather_data_format')
            if weather_format not in {'2012','plus'}:
                raise ValueError('工程天气格式未知，不能推测索引语义')
            index='pcp.txt' if weather_format=='2012' else 'pcp.cli'
            target,_=files.resolve(name,str(declared).replace('\\','/').rstrip('/')+'/'+index)
            weather_source=str(PurePosixPath(target).parent)
        else:
            parents={str(PurePosixPath(n).parent) for n in files.files if PurePosixPath(n).name=='pcp.txt'
                     and files.prefix(n).lstrip().startswith(b'ID,NAME,LAT,LONG,ELEVATION')}
            if len(parents)==1:
                weather_source=parents.pop();weather_format='2012'
        if not weather_source or weather_format not in {'2012','plus'}:
            raise ValueError('编辑工程缺少唯一可识别天气目录/格式，不能凭空生成气象输入')
        # The official weather importer uses names from these files as output
        # paths; validate before calling it, not after it has written files.
        for resource in files.files:
            if str(PurePosixPath(resource).parent)!=weather_source:continue
            if weather_format=='2012' and files.prefix(resource).startswith(b'ID,NAME,LAT,LONG,ELEVATION'):
                with files.files[resource][0]() as stream:
                    records=list(csv.DictReader(stream.read().decode('utf-8-sig').splitlines()))
                for record in records:
                    station=record.get('NAME','')
                    if re.fullmatch(r'[A-Za-z0-9_-]+',station) is None:
                        raise ValueError('天气索引NAME不是安全的站点文件名：'+station)
                    target=(PurePosixPath(weather_source)/(station+'.txt')).as_posix()
                    if target not in files.files:raise ValueError('天气索引引用缺失：'+target)
            elif weather_format=='plus' and resource.endswith('.cli'):
                with files.files[resource][0]() as stream:
                    names=stream.read().decode('utf-8-sig').splitlines()[2:]
                for item in names:
                    item=item.strip()
                    if item and (PurePosixPath(item).name!=item or '\\' in item or ':' in item):
                        raise ValueError('天气CLI包含不安全文件引用：'+item)
        repo=Path(__file__).resolve().parents[3]
        upstream=repo/'external/swatplus-editor';runtime=repo/'.venv-swat-editor/Scripts/python.exe'
        if not runtime.is_file():
            raise ValueError('缺少官方Editor隔离环境；运行integrations/swat_editor/setup.ps1')
        revision=subprocess.run(['git','-C',str(upstream),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
        if revision!=EDITOR_COMMIT:
            raise ValueError('官方Editor版本与已验证4.0.0导出契约不符')
        project=folder/'editor_project';project.mkdir()
        sources=[]
        for resource in sorted(files.files):
            target=project/resource;target.parent.mkdir(parents=True,exist_ok=True)
            with files.files[resource][0]() as src,target.open('wb') as dst:
                shutil.copyfileobj(src,dst,8*1024**2)
            sources.append(files.identity(resource))
        csv_inputs=None
        if wgn_csv:
            imported=project/PurePosixPath(name).parent/'_ecodata_wgn_import';imported.mkdir()
            stat=imported/'stations.csv';month=imported/'monthly.csv'
            # The official importer is positional: reorder columns, retain every
            # actual number and original file, and explicitly map the FK alias.
            with stat.open('w',newline='',encoding='utf8') as stream:
                writer=csv.DictWriter(stream,fieldnames=['id','name','lat','lon','elev','rain_yrs'],extrasaction='ignore')
                writer.writeheader();writer.writerows(wgn_csv['station_rows'])
            with month.open('w',newline='',encoding='utf8') as stream:
                writer=csv.writer(stream);writer.writerow(['id','weather_wgn_cli','month',*WGN_FIELDS])
                for row in wgn_csv['month_rows']:
                    writer.writerow([row['id'],row[wgn_csv['foreign_key']],row['month'],*[row[k] for k in WGN_FIELDS]])
            csv_inputs=[str(stat.resolve()),str(month.resolve())]
    job=dict(api=str(upstream/'src/api'),project=str((project/name).resolve()),
             datasets=str((project/datasets).resolve()),project_name=config.get('project_name'),
             wgn=str((project/wgn).resolve()) if wgn else None,wgn_table=config.get('wgn_table_name'),
             wgn_csv=csv_inputs,
             initialise=not config.get('imported_gis') or 'hru_data_hru' not in tables,
             weather_source=str((project/weather_source).resolve()),weather_format=weather_format)
    if not job['project_name']:
        raise ValueError('工程数据库缺少project_name，不用文件名生成科学工程标识')
    request=folder/'editor_export_request.json';request.write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf8')
    if progress:progress('调用官方SWAT+ Editor：导入工程数据库及天气，导出模型输入；不运行求解器')
    with (folder/'editor_export.log').open('wb') as log:
        run=subprocess.run([str(runtime),str(Path(__file__).with_name('swat_editor_worker.py')),str(request)],
            stdout=log,stderr=subprocess.STDOUT,timeout=180,env=dict(os.environ,PYTHONIOENCODING='utf8',PYTHONNOUSERSITE='1'),
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
    if run.returncode:
        tail=(folder/'editor_export.log').read_text(encoding='utf8',errors='replace')[-1800:]
        raise ValueError('官方SWAT+工程导出失败，原件未改；'+tail)
    native=project/PurePosixPath(name).parent/'Scenarios/Default/TxtInOut'
    from .swat_weather_project import _layout
    payloads={p.name:p.read_bytes() for p in native.iterdir() if p.is_file()}
    _layout(payloads)  # Actual file.cio and weather-index references, not just existence.
    from .project_inventory import prepare_project_inventory
    from ecodata.interface.engineering_upload import LocalUpload
    inventory_folder=folder/'exported_inventory';inventory_folder.mkdir()
    prepare_project_inventory([LocalUpload(p.relative_to(project).as_posix(),p)
        for p in native.iterdir() if p.is_file()],model_id,inventory_folder)
    inventory=json.loads((inventory_folder/'project_inventory.json').read_text(encoding='utf8'))
    bindings=[b for p in inventory['projects'] for b in p['bindings']]
    missing=[b for b in bindings if b['status']!='BOUND']
    errors=[e for p in inventory['projects'] for e in p.get('weather_dependencies',{}).get('errors',[])]
    if not bindings:
        raise ValueError('官方导出缺少可验证输入引用，不能通过验收；见exported_inventory')
    manifest=dict(protocol='ecodata.converter.v1',model_id=model_id,
        status='CONVERSION_FAILED' if missing or errors else 'PROJECT_UPDATED',native_ready=False,
        native_model_executed=False,complete_model_input=False,preparation=dict(route='official_swat_editor_export_v1',
        upstream=dict(url='https://github.com/swat-model/swatplus-editor',commit=revision,editor_version='4.0.0',target_input_revision='62.0.0'),
        source_project=name,datasets=datasets,source_files=sources,official_gis_defaults_used=job['initialise'],
        wgn_import=dict(method='database' if wgn else 'two_file' if wgn_csv else 'not_available',
                        source_files=[wgn] if wgn else [wgn_csv['stations'],wgn_csv['monthly']] if wgn_csv else []),
        original_parameters_not_reconstructed=True,weather_format=weather_format),
        reference_checks=dict(bound_count=len(bindings)-len(missing),missing=missing,errors=errors,unknown=[p['unknown'] for p in inventory['projects']]),
        native_writeback_verified=not missing and not errors,note='官方Editor输入导出及引用检查；未运行模型，不证明科学标定；辅助天气尚未融合')
    if missing or errors:
        manifest.update(conversion_error='官方已导出TxtInOut，但工程仍缺输入依赖：'+
            '、'.join([b['original'] for b in missing]+errors)+'；未填造统计资料，下载仅供检查，不可直接运行',
            exported_inputs_status='PARTIAL')
    (folder/'handoff.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    archive=folder/'EcoData-editor-exported-project.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in project.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(project).as_posix())
        for p in [request,folder/'handoff.json',folder/'editor_export.log']:
            z.write(p,'_ecodata/'+p.name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip():raise ValueError('导出工程ZIP校验失败')
    return dict(manifest=manifest,archive=str(archive))
