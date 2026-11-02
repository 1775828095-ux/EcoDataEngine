"""Managed WRF execution for the registered FNL case; no uploaded code execution."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import threading
import uuid
import zipfile

import xarray as xr
from ecodata.core.exceptions import SchemaError
from ecodata.interface.wrf_assembly import assemble_pressure_update, read_fields
from ecodata.models.adapters.wps_fields import export_pressure

ROOT=Path(__file__).resolve().parents[3]
SOURCE=Path(os.environ.get('ECODATA_WRF_CASE_SOURCE',r'C:\Users\ASUS\Documents\ChatGPT\测试重\WRF完整案例准备_20261017'))
RUNS=ROOT/'outputs'/'wrf_jobs'
_LOCK=threading.Lock()
_ACTIVE=None


def read_update_zip(payload: bytes) -> dict[str,bytes]:
    if len(payload)>128*1024**2:
        raise SchemaError('平台更新包超过128 MiB。')
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            items=archive.infolist()
            if len(items)>100 or sum(f.file_size for f in items)>128*1024**2:
                raise SchemaError('更新包解压大小或文件数超限。')
            files={}
            for info in items:
                name=info.filename
                if (info.is_dir() or '/' in name or '\\' in name or ':' in name
                        or name in ('.','..') or stat.S_ISLNK(info.external_attr>>16)
                        or any(ord(c)<32 for c in name)):
                    raise SchemaError('更新包只允许平铺数据文件。')
                if name.casefold() in files:
                    raise SchemaError('更新包有重复文件名。')
                files[name.casefold()]=(name,archive.read(info))
            fields={name:data for name,data in files.values() if name.endswith('.wps')}
            if not fields:
                raise SchemaError('更新包没有WPS场文件。')
            return fields
    except (zipfile.BadZipFile,RuntimeError,NotImplementedError) as exc:
        raise SchemaError('无法读取平台更新ZIP。') from exc


def _directory(ident):
    if not re.fullmatch('[0-9a-f]{32}',ident):
        raise SchemaError('无效WRF任务编号。')
    return RUNS/ident


def status(ident):
    path=_directory(ident)/'status.json'
    return json.loads(path.read_text(encoding='utf-8'))


def _save(folder,value):
    temp=folder/'status.tmp'
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(folder/'status.json')


def _linux(path):
    path=Path(path).resolve()
    if not path.drive or len(path.drive)!=2:
        raise SchemaError('WRF运行目录需位于本地盘符。')
    return '/mnt/'+path.drive[0].lower()+'/'+path.as_posix()[3:]


def start(route: str, update_zip: bytes | None=None,mode=None):
    global _ACTIVE
    if route not in ('existing','from_zero'):
        raise SchemaError('未知WRF路线。')
    with _LOCK:
        if _ACTIVE is not None:
            raise SchemaError('已有WRF任务正在运行，请等待完成。')
        ident=uuid.uuid4().hex
        _ACTIVE=ident
    folder=_directory(ident)
    try:
        folder.mkdir(parents=True,exist_ok=False)
        if mode=='sst':
            from ecodata.interface.registered_conversion import member
            raw=member(member(update_zip,'WPS_metgrid_SST_constant.zip'),'ECODATA_SST_CONSTANT')
            reference=SOURCE.parent/'WRF_SST接收检验_20261018/04_界面开发检验/ECODATA_SST_CONSTANT'
            if raw!=reference.read_bytes():raise SchemaError('SST不属于已核验的OISST单快照案例；不推测其他日期/区域兼容性')
            (folder/'ECODATA_SST_CONSTANT').write_bytes(raw)
            (folder/'assembly.json').write_text(json.dumps(dict(route=route,mode=mode,input_sha256=hashlib.sha256(raw).hexdigest(),source=str(reference),simulation_seconds=36,temporal_policy='Explicit constant snapshot, not hourly observations'),indent=2),encoding='utf8')
            shutil.copyfile(Path(__file__).with_name('wrf_sst_worker.py'),folder/'worker.py')
            _save(folder,dict(state='running',route=route,id=ident,simulation_seconds=36))
            threading.Thread(target=_run,args=(ident,route),daemon=True).start();return ident
        terrain_report=None
        if mode=='geogrid_tiles':
            from ecodata.interface.registered_conversion import member
            with zipfile.ZipFile(io.BytesIO(update_zip)) as outer:
                selected=[name for name in ('geogrid_HGT_M.zip','geogrid_LANDUSEF.zip') if name in outer.namelist()]
            if len(selected)!=1:raise SchemaError('地理转换包须明确包含一个HGT_M或LANDUSEF瓦片包')
            field='LANDUSEF' if selected[0]=='geogrid_LANDUSEF.zip' else 'HGT_M'
            raw=member(update_zip,selected[0])
            case_name='WRF真实比例土地利用' if field=='LANDUSEF' else 'WRF真实地理瓦片原生接收'
            reference=(SOURCE.parent/'夜间续跑_20261025'/case_name/selected[0]).read_bytes()
            def tile_files(data):
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    names=z.namelist()
                    if len(names)!=2 or len(set(names))!=2 or any(not re.fullmatch(r'index|[0-9]+-[0-9]+\.[0-9]+-[0-9]+',n) for n in names):raise SchemaError('地形包必须为index及单瓦片')
                    return {n:z.read(n) for n in names}
            current=tile_files(raw)
            if current!=tile_files(reference):raise SchemaError('地形数据不属于已核验登记范围；不自动重采样或改用示例')
            geographic_folder=folder/('landuse' if field=='LANDUSEF' else 'terrain');geographic_folder.mkdir()
            for name,data in current.items():(geographic_folder/name).write_bytes(data)
            terrain_report=dict(field=field,sha256=hashlib.sha256(raw).hexdigest(),scope='Current exact original MODIS21 percent crop; every category preserved' if field=='LANDUSEF' else 'Current exact GMTED benchmark crop; companion geographic fields and meteorology reused')
            update_zip=None
        elif mode is not None:raise SchemaError('未登记此WPS类别的工程接入')
        config=json.loads((SOURCE/'07_本次用户上传/config_FNL_pressure.json').read_text(encoding='utf-8'))
        reference=SOURCE/'06_官方参考运行/files'
        radius=read_fields((reference/'FILE_2023-03-31_00').read_bytes())[0].grid[-1]
        config['earth_radius_km']=radius
        if update_zip is None:
            with xr.open_dataset(SOURCE/'07_本次用户上传/A_FNL_20230331_5times_33levels.nc',engine='scipy') as ds:
                updates,report=export_pressure(ds,config)
            updates={n:v for n,v in updates.items() if n.endswith('.wps')}
            origin='registered source NetCDF converted by platform'
        else:
            updates=read_update_zip(update_zip)
            origin='user uploaded platform output'
        assembled=[]
        stamps=set()
        for data in updates.values():
            stamp=read_fields(data)[0].time[:13]
            if stamp in stamps:
                raise SchemaError('更新包有重复时刻。')
            stamps.add(stamp)
            source=reference/('FILE_'+stamp)
            if not source.is_file():
                raise SchemaError('当前登记案例不含此时刻；不能自动换用其他案例资料。')
            merged,report=assemble_pressure_update(source.read_bytes(),data)
            (folder/('MERGED_'+stamp+'.wps')).write_bytes(merged)
            assembled.append(report)
        expected={'2023-03-31_'+h for h in ('00','06','12','18')}|{'2023-04-01_00'}
        if stamps!=expected:
            raise SchemaError('登记案例要求2023-03-31 00时至04-01 00时共5个六小时时刻。')
        (folder/'assembly.json').write_text(json.dumps(dict(route=route,origin=origin,configuration=config,files=assembled,current_terrain=terrain_report),ensure_ascii=False,indent=2),encoding='utf-8')
        shutil.copyfile(Path(__file__).with_name('wrf_native_worker.py'),folder/'worker.py')
        _save(folder,dict(state='running',route=route,id=ident,simulation_seconds=180))
        threading.Thread(target=_run,args=(ident,route),daemon=True).start()
        return ident
    except Exception:
        with _LOCK:_ACTIVE=None
        raise


def _run(ident,route):
    global _ACTIVE
    folder=_directory(ident)
    seconds=status(ident).get('simulation_seconds',180)
    try:
        command=['wsl.exe','-d','EcoData-WRF-Ubuntu2204','-u','root','--','python3',_linux(folder/'worker.py'),'--workspace',_linux(folder),'--job-id',ident]
        if route=='from_zero':command.append('--from-zero')
        with (folder/'runner.log').open('wb') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=5400)
        if result.returncode:
            raise RuntimeError(f'WRF链返回{result.returncode}；请查看运行日志。')
        _plot(folder)
        _save(folder,dict(state='succeeded',route=route,id=ident,simulation_seconds=seconds))
    except Exception as exc:
        _save(folder,dict(state='failed',route=route,id=ident,error=str(exc)))
    finally:
        with _LOCK:_ACTIVE=None


def _plot(folder):
    import matplotlib
    import netCDF4
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    output=next(folder.glob('*wrfout*'))
    # netCDF-C on Windows may reject Unicode file paths; Python reads the file.
    native=netCDF4.Dataset('memory',memory=output.read_bytes())
    with xr.open_dataset(xr.backends.NetCDF4DataStore(native)) as ds:
        data=ds['T2'].isel(Time=-1).values-273.15
        lon=ds['XLONG'].isel(Time=-1).values
        lat=ds['XLAT'].isel(Time=-1).values
        fig=Figure(figsize=(8,6));FigureCanvasAgg(fig)
        ax=fig.subplots();mesh=ax.pcolormesh(lon,lat,data,shading='auto',cmap='coolwarm')
        fig.colorbar(mesh,ax=ax,label='2 m air temperature (degC)')
        stamp=b''.join(ds['Times'].isel(Time=-1).values.tolist()).decode('ascii') if ds['Times'].ndim==2 else str(ds['Times'].isel(Time=-1).values)
        ax.set(xlabel='Longitude',ylabel='Latitude',title='WRF native output: '+stamp+' UTC')
        fig.tight_layout();fig.savefig(folder/'temperature.png',dpi=150)


def result_path(ident,name):
    if name not in ('temperature.png','runner.log','assembly.json','native_existing_outputs.json','native_from_zero_outputs.json'):
        raise SchemaError('未知WRF结果文件。')
    return _directory(ident)/name
