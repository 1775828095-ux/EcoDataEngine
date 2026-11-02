"""WPS v5 intermediate format, IPROJ=0, SST supplementary field ONLY."""
import struct
import io
import json
import zipfile
import hashlib
import numpy as np
import pandas as pd
from ecodata.core.exceptions import SchemaError


def _text(value,width):
    try: data=str(value).encode('ascii')
    except UnicodeEncodeError as exc:raise SchemaError('WPS头文本必须ASCII') from exc
    if len(data)>width:raise SchemaError('WPS头文本过长；不截断')
    return data.ljust(width,b' ')


def _record(payload):
    marker=struct.pack('>i',len(payload))
    return marker+payload+marker


def _constant_snapshot_package(payload, *, source_filename, snapshot_time, field, source_unit):
    """WPS metgrid reads constants_name literally; no hourly dates are fabricated."""
    native_name='ECODATA_SST_CONSTANT'
    lat=field.coords[field.dims[-2]].values
    lon=field.coords[field.dims[-1]].values
    manifest={
        'interface':'WPS 4.6.0 metgrid constants_name / intermediate v5 IPROJ0',
        'mode':'constant_snapshot', 'snapshot_time_UTC':snapshot_time,
        'temporal_assumption':'One source analysis held fixed; NOT hourly observations or interpolation',
        'original_export_filename':source_filename, 'native_filename':native_name,
        'intermediate_sha256':hashlib.sha256(payload).hexdigest(),
        'field':'SST', 'units':'K', 'source_units':source_unit, 'xlvl':200100.0,
        'shape_y_x':[len(lat),len(lon)], 'crs':'EPSG:4326',
        'latitude_centers':[float(lat[0]),float(lat[-1])],
        'longitude_centers':[float(lon[0]),float(lon[-1])],
        'official_metgrid_run_verified':False, 'complete_WRF_input':False,
    }
    fragment=(
        '! Assignment fragment only: merge into an existing &metgrid group.\n'
        '! Preserve the actual fg_name and all domain/geogrid settings.\n'
        '&metgrid\n'
        " constants_name = './ECODATA_SST_CONSTANT',\n"
        '/\n'
    )
    readme=(
        'WPS 4.6.0 fixed SST snapshot supplement\n'
        'Extract into a dedicated WPS working directory. The intermediate file\n'
        'ECODATA_SST_CONSTANT has no .wps suffix and is addressed literally by\n'
        'constants_name. Merge namelist.metgrid.fragment into your EXISTING\n'
        'namelist.wps; it is not a complete runnable namelist. Preserve/append any\n'
        'other constants and keep the complete atmospheric fg_name inputs.\n'
        'This explicitly holds one SST analysis fixed; it does not create hourly\n'
        'SST or change the timestamp inside the intermediate file. Confirm this\n'
        'choice is appropriate for the intended simulation window.\n'
        'All water points in the WPS domain/interpolation stencil must be covered\n'
        'by valid source SST. The small example is a format test, not global SST.\n'
        'If another fg_name source also supplies SST, resolve source precedence\n'
        'in the actual project and inspect FLAG_SST/SST in met_em output.\n'
        'Required: compatible WPS/METGRID.TBL, geo_em and full atmospheric fields.\n'
        'Unverified: metgrid.exe -> real.exe -> wrf.exe; this package alone cannot\n'
        'initialize WRF. See manifest.json for source time, grid and data hash.\n'
    )
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr(native_name,payload)
        z.writestr('namelist.metgrid.fragment',fragment)
        z.writestr('manifest.json',json.dumps(manifest,indent=2,ensure_ascii=False))
        z.writestr('README.txt',readme)
    return archive.getvalue(),manifest


def export_sst(dataset,config):
    if config.get('version')!='WPS-4.6-intermediate-v5-SST':raise SchemaError('仅支持WPS4.6中间格式v5的SST切片')
    if config.get('time_basis')!='UTC':raise SchemaError('需明确NetCDF时间为UTC，不猜测本地时间')
    handoff=config.get('wps_handoff')
    if handoff is not None and handoff != {'mode':'constant_snapshot'}:
        raise SchemaError('wps_handoff仅支持显式mode=constant_snapshot；不猜测时序模式或忽略未知选项')
    var=config.get('variable','sst');lat=config.get('latitude','lat');lon=config.get('longitude','lon');time=config.get('time','time')
    if any(n not in dataset for n in (var,lat,lon,time)):raise SchemaError('缺少SST/经纬度/时间')
    field=dataset[var]
    if field.dtype.kind not in 'iuf':raise SchemaError('SST必须为实数物理变量')
    for dim in field.dims:
        if dim not in (time,lat,lon):
            if field.sizes[dim]!=1 or dim not in config.get('squeeze_dimensions',[]):raise SchemaError('额外维度需为明确允许的单层维度；不静默选层')
            field=field.isel({dim:0},drop=True)
    if set(field.dims)!={time,lat,lon}:raise SchemaError('SST需time×lat×lon')
    if config.get('crs')!='EPSG:4326':raise SchemaError('仅支持明确EPSG:4326规则经纬度网格；不将投影坐标改名')
    for axis in (lat,lon):
        v=dataset[axis].values
        if v.ndim!=1 or len(v)<2 or not np.isfinite(v).all() or not (np.all(np.diff(v)>0) or np.all(np.diff(v)<0)) or not np.allclose(np.diff(v),np.diff(v)[0],rtol=0,atol=1e-6):raise SchemaError('经纬度需至少2点、严格单调等间距')
    if float(dataset[lat].min())<-90 or float(dataset[lat].max())>90 or float(dataset[lon].min())<-180 or float(dataset[lon].max())>360:raise SchemaError('经纬度范围不合法')
    field=field.sortby(lat).sortby(lon).transpose(time,lat,lon)
    field=field.astype(np.float64) # Avoid rounding the Celsius offset before final float32 encoding.
    unit=field.attrs.get('units','')
    if unit in {'degC','degrees_Celsius','Celsius'}:field=field+273.15
    elif unit!='K':raise SchemaError(f'SST单位{unit!r}不能确认')
    if not np.isfinite(field).all():raise SchemaError('SST含掩码/缺失；当前切片不填海陆缺失')
    if float(field.min())<250 or float(field.max())>330:raise SchemaError('SST超出工程温度范围250..330K')
    try:times=pd.to_datetime(field[time].values)
    except (ValueError,TypeError) as exc:raise SchemaError('仅支持明确UTC的公历时间') from exc
    if times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing:raise SchemaError('时间缺失/重复/乱序')
    interval_seconds = None
    if len(times)>1:
        differences = pd.Series(times).diff().dropna()
        if differences.eq(pd.Timedelta(hours=1)).all():
            interval_seconds = 3600
        elif differences.eq(pd.Timedelta(days=1)).all():
            interval_seconds = 86400
        else:
            raise SchemaError('SST时间需连续整点逐小时或逐日序列；存在缺时刻或不支持的间隔，不自动填补、复制或插值')
    if handoff is not None and len(times)!=1:
        raise SchemaError('固定SST快照交接只接受一个源时刻；不自动选第一帧或平均')
    files={};errors=[]
    for index,t in enumerate(times):
        if t.minute or t.second or t.microsecond or t.nanosecond:raise SchemaError('当前文件命名仅支持UTC整点；不静默移动时间')
        hdate=t.strftime('%Y-%m-%d_%H:%M:%S')
        nx,ny=field.sizes[lon],field.sizes[lat]
        source=config.get('map_source','EcoData SST')
        header=(_text(hdate,24)+struct.pack('>f',0)+_text(source,32)+_text('SST',9)+_text('K',25)+_text('Sea surface temperature supplementary field',46)+struct.pack('>fiii',200100.,nx,ny,0))
        projection=_text('SWCORNER',8)+struct.pack('>5f',float(field[lat][0]),float(field[lon][0]),float(field[lat][1]-field[lat][0]),float(field[lon][1]-field[lon][0]),6367.47)
        array=field.isel({time:index}).values
        slab=array.astype('>f4').tobytes(order='C') # x varies fastest, matching Fortran slab(nx,ny).
        errors.append(float(np.max(np.abs(array-array.astype('f4').astype('f8')))))
        payload=b''.join(_record(p) for p in (struct.pack('>i',5),header,projection,struct.pack('>i',0),slab))
        # ':' is invalid on Windows; rename to SST:YYYY-MM-DD_HH on WPS host.
        files['SST_'+t.strftime('%Y-%m-%d_%H')+'.wps']=payload
    report={'scope':'WPS4.6 intermediate v5 IPROJ0 SST supplement only','records':len(times),'shape':[ny,nx],
                  'source_interval_seconds':interval_seconds,
                  'time_processing':'none; original UTC timestamps retained; no temporal interpolation or replication',
                  'output_timestamps_UTC':[t.strftime('%Y-%m-%dT%H:%M:%SZ') for t in times],
                  'complete_model_input':False,'native_model_input_verified':False,'float32_max_error_K':max(errors),
                  'format_source':'NCAR WPS guide; WPS v4.6.0 ungrib/output.F',
                  'required_next_steps':['On WPS host rename SST_YYYY-MM-DD_HH.wps to SST:YYYY-MM-DD_HH',
                    'Use constants_name/fg_name as appropriate; GEOGRID and complete atmospheric forcing required',
                    'metgrid.exe -> real.exe -> wrf.exe; this SST field alone cannot initialize WRF']}
    if interval_seconds == 86400:
        report['required_next_steps'] = [
            'On a WPS host rename each SST_YYYY-MM-DD_HH.wps to SST:YYYY-MM-DD_HH; use SST in fg_name, not constants_name',
            'For these supplied times use matching start_date/end_date and interval_seconds=86400; this does not change the source daily-product meaning',
            'For subdaily metgrid times, these daily files alone are insufficient: explicitly choose a scientifically justified temporal method or supply suitable higher-frequency SST; this converter does neither',
            'Provide matching geo_em, METGRID.TBL, full atmospheric inputs and domain SST coverage; metgrid/real/WRF execution remains unverified']
        files['README_daily_SST.txt'] = ('WPS 4.6 daily SST supplement ONLY\n'
            'One file per original UTC source timestamp; no temporal interpolation, replication or missing-date filling.\n'
            'Rename SST_YYYY-MM-DD_HH.wps to SST:YYYY-MM-DD_HH on the WPS host.\n'
            'Add SST to the existing metgrid fg_name; retain complete atmospheric sources. Do not use constants_name for this daily series.\n'
            'Align start_date/end_date with source times and use interval_seconds=86400 for these timestamps.\n'
            'For subdaily processing, first explicitly justify and configure a temporal method outside this converter, or supply suitable higher-frequency SST. Daily data are not hourly observations.\n'
            'Requires geo_em, matching METGRID.TBL, full atmospheric fields and sufficient SST coverage.\n'
            'No official metgrid, real.exe or WRF run is verified.\n').encode('ascii')
    if handoff is not None:
        filename=next(iter(files))
        package,manifest=_constant_snapshot_package(files[filename],source_filename=filename,
            snapshot_time=times[0].strftime('%Y-%m-%dT%H:%M:%SZ'),field=field,source_unit=unit)
        files['WPS_metgrid_SST_constant.zip']=package
        report['wps_handoff']=manifest
        report['required_next_steps']=['Extract WPS_metgrid_SST_constant.zip in the WPS working directory',
            'Merge constants_name fragment into the existing namelist.wps; retain actual fg_name and other constants',
            'Check full domain SST coverage and the fixed snapshot assumption; GEOGRID and complete atmospheric forcing required',
            'metgrid.exe -> real.exe -> wrf.exe remain unverified']
    return files,report
