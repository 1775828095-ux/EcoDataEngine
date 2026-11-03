"""Static regular-ll NetCDF to WPS4.6 geogrid index + binary tile.

Upstream geography input only; does not execute geogrid or create geo_em.
"""
import io,re,zipfile
import numpy as np
from ecodata.core.exceptions import SchemaError
from ecodata.models.adaptation import unit_transform
from ecodata.models.adapters.wps_fields import _axis,_check_unit_declaration,_units


def export_geography(ds,cfg):
    if cfg.get('version')!='WPS-4.6-geogrid-tile' or cfg.get('crs')!='EPSG:4326':raise SchemaError('仅WPS4.6规则经纬度geogrid原始瓦片；需crs=EPSG:4326')
    if ds.attrs.get('crs') and ds.attrs['crs']!='EPSG:4326':raise SchemaError('数据内crs与配置冲突')
    field=cfg.get('field');category=field=='LANDUSEF'
    if field not in {'HGT_M','LANDUSEF'}:raise SchemaError('当前仅HGT_M地形及LANDUSEF已分类土地利用，不推算其他地理变量')
    if not category and cfg.get('elevation_vertical_reference')!='MSL':raise SchemaError('HGT_M需elevation_vertical_reference=MSL，不能把椭球高默认为海拔')
    source=cfg.get('source')
    if not isinstance(source,dict) or not all(source.get(k) for k in ('kind','description','license')):raise SchemaError('需source.kind/description/license，合成数据需注明')
    axes=cfg.get('coordinates')
    if not isinstance(axes,dict) or set(axes)!={'latitude','longitude'} or len(set(axes.values()))!=2:raise SchemaError('coordinates需不同的一维latitude/longitude')
    lat,lon=axes['latitude'],axes['longitude'];a=_axis(ds,lat,'latitude');b=_axis(ds,lon,'longitude')
    for role,name in axes.items():
        declared=cfg.get('coordinate_units',{}).get(role,ds[name].attrs.get('units',''))
        _check_unit_declaration(ds[name].attrs.get('units',''),declared,role)
        if _units(declared) not in {'degree','degrees'}:raise SchemaError('瓦片坐标需角度单位')
    a=np.sort(a);b=np.sort(b)
    if a.min()<-90 or a.max()>90 or b.min()<-180 or b.max()>180 or np.ptp(b)>=360:raise SchemaError('瓦片经纬度越界或重复地球闭环')
    nx,ny=len(b),len(a)
    if nx>99999 or ny>99999 or nx*ny>10000000:raise SchemaError('本单瓦片上限1000万格，每轴不超过99999；不自动截取')
    name=cfg.get('variable')
    if name not in ds.data_vars:raise SchemaError('variable没有对应源地理变量')
    variable=ds[name]
    if len(variable.dims)!=2 or set(variable.dims)!={lat,lon} or variable.dtype.kind not in 'iuf':raise SchemaError('源需静态二维实数格网，不能丢弃时间或额外层')
    actual=variable.attrs.get('units','');declared=cfg.get('source_units',actual)
    if not declared:raise SchemaError('地理源物理单位不明')
    _check_unit_declaration(actual,declared,field)
    if category and declared!='1':raise SchemaError('土地类别码需单位1，不把百分比等连续量改成类别')
    scale,offset=unit_transform(declared,'1' if category else 'm')
    raw=np.asarray(variable.sortby(lat).sortby(lon).transpose(lat,lon),dtype=float)
    markers=cfg.get('missing_values')
    if not isinstance(markers,list):raise SchemaError('需明确missing_values列表')
    if not np.isfinite(raw).all():raise SchemaError('源瓦片有NaN/缺测；不填补')
    for marker in markers+list(np.atleast_1d(variable.attrs.get('missing_value',[])))+list(np.atleast_1d(variable.attrs.get('_FillValue',[]))):
        try:numeric=float(marker)
        except (ValueError,TypeError):raise SchemaError('需明确数值缺测标记')
        if np.any(raw==numeric):raise SchemaError('源瓦片含缺测标记；不填补')
    values=raw*scale+offset
    wordsize=cfg.get('wordsize');endian=cfg.get('endian')
    if type(wordsize)!=int or wordsize not in {2,4} or endian not in {'big','little'}:raise SchemaError('当前wordsize需2或4，endian需big/little')
    quant=cfg.get('scale_factor');tolerance=cfg.get('max_absolute_quantization_error')
    if isinstance(quant,bool) or isinstance(tolerance,bool):raise SchemaError('缩放与误差阈值不能为布尔')
    try:quant=float(quant);tolerance=float(tolerance)
    except (ValueError,TypeError) as exc:raise SchemaError('需显式scale_factor及允许量化误差max_absolute_quantization_error') from exc
    if not np.isfinite([quant,tolerance]).all() or quant<=0 or tolerance<0:raise SchemaError('缩放须正且量化误差阈值非负')
    if not np.isfinite(np.float32(quant)) or np.float32(quant)<=0:raise SchemaError('scale_factor不能安全表示为官方reader的float32')
    if not np.isfinite(values).all():raise SchemaError('换算溢出')
    extra={}
    if category:
        info=cfg.get('landuse')
        if not isinstance(info,dict) or set(info)!={'category_min','category_max','mminlu','iswater','islake','isice','isurban'}:raise SchemaError('LANDUSEF需类别范围、分类系统及水/湖/冰/城市类别码，不猜测分类')
        if any(type(info[k])!=int for k in ('category_min','category_max','iswater','islake','isice','isurban')) or not 1<=info['category_min']<=info['category_max']<=32767:raise SchemaError('土地类别范围需1..32767整数')
        if any(not info['category_min']<=info[k]<=info['category_max'] for k in ('iswater','islake','isice','isurban')):raise SchemaError('特殊地类编码未位于类别范围')
        if not isinstance(info['mminlu'],str) or not re.fullmatch('[A-Za-z0-9_-]{1,32}',info['mminlu']):raise SchemaError('mminlu须明确安全分类系统名')
        if quant!=1 or tolerance!=0 or np.any(values!=np.rint(values)) or np.any((values<info['category_min'])|(values>info['category_max'])):raise SchemaError('类别码不允许量化或超范围；scale_factor=1，允许误差=0')
        extra=info
    stored=np.rint(values/quant)
    limit=2**(wordsize*8-1)
    if np.any(stored<=-limit) or np.any(stored>=limit):raise SchemaError('数值不适合所选signed字长，拒绝溢出及官方reader最小负数边界')
    # Official C reader returns float32 and scales there; include that loss.
    decoded=stored.astype('f4')*np.float32(quant)
    error=float(np.max(np.abs(decoded.astype(float)-values)))
    if not np.isfinite(decoded).all() or error>tolerance:raise SchemaError(f'瓦片量化/float32最大绝对误差{error}超过事先给定阈值{tolerance}；不放宽')
    dtype=np.dtype(('>' if endian=='big' else '<')+f'i{wordsize}')
    filename=f'00001-{nx:05d}.00001-{ny:05d}'
    # missing_value is format metadata, never a value inserted into the grid.
    nodata=-9999
    if np.any(decoded==nodata):raise SchemaError('真实值与目标missing_value=-9999冲突，拒绝写出')
    meta=dict(type='categorical' if category else 'continuous',signed='yes',projection='regular_ll',dx=b[1]-b[0],dy=a[1]-a[0],known_x=1.,known_y=1.,known_lat=a[0],known_lon=b[0],wordsize=wordsize,endian=endian,row_order='bottom_top',tile_x=nx,tile_y=ny,tile_z=1,tile_bdr=0,scale_factor=quant,missing_value=nodata)
    meta.update(extra)
    lines=[f'{k}={format(v,".17g") if isinstance(v,(float,np.floating)) else v}' for k,v in meta.items()]
    lines+=['units="category"' if category else 'units="m"',f'description="EcoDataEngine user supplied {field}"']
    index=('\n'.join(lines)+'\n').encode('ascii');tile=stored.astype(dtype).tobytes(order='C')
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for member,content in [('index',index),(filename,tile)]:
            info=zipfile.ZipInfo(member,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,content)
    notes=f'WPS4.6 geogrid上游{field}数据集（index与原始整数瓦片），规则经纬度单瓦片；未生成geo_em。需把ZIP解压至独立地理目录，并在已有GEOGRID.TBL对应{field}节配置绝对路径/分辨率及有依据的插值规则。需完整域覆盖及其他必需地理数据；本平台不运行geogrid，不填出界格，不改变土地分类或推导物理参数。量化最大绝对误差={error}，源缺测策略=拒绝。'
    return {f'geogrid_{field}.zip':archive.getvalue(),'README_geogrid.txt':notes.encode('utf8')},dict(scope=notes,field=field,source=source,shape_y_x=[ny,nx],index=meta,native_tile=filename,unit_conversion=dict(source=declared,target='category' if category else 'm',scale=scale,offset=offset),max_absolute_quantization_error=error,time_processing='static; no time axis',coordinate_processing='axis sorting only; no regridding',complete_model_input=False,model_run_verified=False)
