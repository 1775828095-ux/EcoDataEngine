"""CALMET 5.8.5 READGE GEO.DAT: explicit UTM/WGS-84 static grids only."""
import numpy as np
from pyproj import CRS
from ecodata.core.exceptions import SchemaError
from ecodata.models.adaptation import unit_transform

FIELDS={'landuse':'1','elevation':'m','roughness':'m','albedo':'1','bowen':'1','soil_heat_flux':'1','anthropogenic_heat':'W/m^2','leaf_area_index':'1'}


def export_geography(ds,cfg):
    if cfg.get('target_version')!='CALMET-5.8.5-GEO.DAT-2.0':raise SchemaError('地理格式需target_version=CALMET-5.8.5-GEO.DAT-2.0')
    if cfg.get('coordinate_location')!='cell_center':raise SchemaError('源x/y必须明确为cell_center；文件原点将写格边')
    if cfg.get('elevation_vertical_reference')!='MSL':raise SchemaError('地形高程需明确elevation_vertical_reference=MSL；不把椭球高当海拔')
    if not isinstance(cfg.get('source'),dict) or not cfg['source'].get('description'):raise SchemaError('需记录地理参数来源，合成输入须明确标注')
    if not isinstance(cfg.get('missing_values'),list):raise SchemaError('需明确missing_values列表')
    try:crs=CRS.from_user_input(cfg.get('crs'));epsg=crs.to_epsg()
    except Exception as exc:raise SchemaError('必须声明WGS84 UTM CRS') from exc
    if epsg is None or not (32601<=epsg<=32660 or 32701<=epsg<=32760):raise SchemaError('当前地理profile仅支持WGS84 UTM南/北半球')
    if ds.attrs.get('crs') and not CRS.from_user_input(ds.attrs['crs']).equals(crs,ignore_axis_order=True):raise SchemaError('数据内crs与声明冲突')
    zone=epsg%100;hemisphere='N' if epsg<32700 else 'S'
    axes=cfg.get('coordinates',{})
    if set(axes)!={'x','y'} or len(set(axes.values()))!=2:raise SchemaError('coordinates需明确不同x/y轴；仅静态二维输入')
    coords={};steps={}
    for role in ('x','y'):
        name=axes[role]
        if name not in ds.coords or ds[name].dims!=(name,):raise SchemaError(f'{role}需同名一维格心坐标')
        raw=np.asarray(ds[name],dtype=float);units=cfg.get('coordinate_units',{}).get(role,ds[name].attrs.get('units'))
        if units not in {'m','km'}:raise SchemaError(f'{role}单位需明确m或km')
        if ds[name].attrs.get('units',units)!=units:raise SchemaError(f'{role}源单位与配置冲突')
        scale,offset=unit_transform(units,'km');a=raw*scale+offset
        if not 2<=len(a)<=265 or not np.isfinite(a).all():raise SchemaError('当前5.8.5编译上限下每轴需2..265个有效坐标')
        d=np.diff(a)
        if not (np.all(d>0) or np.all(d<0)) or not np.allclose(d,d[0],rtol=0,atol=1e-9):raise SchemaError('网格需严格单调等间距；不补格或重采样')
        coords[role]=np.sort(a);steps[role]=abs(d[0])
    if not np.isclose(steps['x'],steps['y'],rtol=0,atol=1e-9):raise SchemaError('CALMET5.8.5 READGE仅单一DGRID，需方形网格')
    origin=[coords[r][0]-steps[r]/2 for r in ('x','y')];geometry=origin+[steps['x'],steps['y']]
    # Fixed f12.3 kilometre header: reject rather than shift a sub-metre grid.
    if any(abs(v-round(v,3))>1e-9 or len(f'{v:12.3f}')!=12 for v in geometry):raise SchemaError('GEO.DAT表头为km/f12.3；格边和格距须能以1m精度无损表达')
    mapping=cfg.get('fields')
    if not isinstance(mapping,dict) or set(mapping)!=set(FIELDS):raise SchemaError('需完整映射landuse/elevation/roughness/albedo/bowen/soil_heat_flux/anthropogenic_heat/leaf_area_index八个真实字段')
    values={};transforms={};x,y=axes['x'],axes['y']
    numeric_markers=[]
    for v in cfg['missing_values']:
        try:numeric_markers.append(float(v))
        except (TypeError,ValueError):pass
    for field,target in FIELDS.items():
        setting=mapping[field]
        if not isinstance(setting,dict) or set(setting)-{'variable','source_units'} or setting.get('variable') not in ds.data_vars:raise SchemaError(f'{field}缺源变量或映射无效')
        a=ds[setting['variable']]
        if set(a.dims)!={x,y} or len(a.dims)!=2 or a.dtype.kind not in 'iuf':raise SchemaError(f'{field}必须是完整二维实数网格，无时间或额外维度')
        raw=np.asarray(a.values)
        if not np.isfinite(raw).all():raise SchemaError(f'{field}含缺测或非有限值')
        for marker in numeric_markers+list(np.atleast_1d(a.attrs.get('missing_value',[])))+list(np.atleast_1d(a.attrs.get('_FillValue',[]))):
            if np.any(raw==marker):raise SchemaError(f'{field}含源定义缺测标记')
        source=setting.get('source_units',a.attrs.get('units',''))
        if not source:raise SchemaError(f'{field}单位不明')
        if a.attrs.get('units'):
            scale,offset=unit_transform(a.attrs['units'],source)
            if not np.isclose(scale,1,rtol=0,atol=1e-12) or not np.isclose(offset,0,rtol=0,atol=1e-12):raise SchemaError(f'{field}源单位与配置冲突')
        scale,offset=unit_transform(source,target)
        # READGE reads north->south rows, west->east columns.
        result=np.asarray(a.sortby(y,ascending=False).sortby(x).transpose(y,x),dtype=float)*scale+offset
        if not np.isfinite(result).all():raise SchemaError(f'{field}换算溢出')
        values[field]=result;transforms[field]=dict(source_units=source,target_units=target,scale=scale,offset=offset)
    land=values['landuse'];categories=cfg.get('landuse_categories');water=cfg.get('water_category_range')
    if not isinstance(categories,list) or not 1<=len(categories)<=52 or any(type(v)!=int or v<0 for v in categories) or len(set(categories))!=len(categories):raise SchemaError('需明确1..52个唯一整数土地类别码')
    if not isinstance(water,list) or len(water)!=2 or any(type(v)!=int for v in water) or water[0]>water[1]:raise SchemaError('需明确水体类别码闭区间water_category_range')
    if not np.equal(land,np.round(land)).all() or not np.isin(land,categories).all():raise SchemaError('landuse有非整数或未列入类别表的编码')
    if np.any(values['roughness']<=0) or np.any(values['leaf_area_index']<0) or np.any(values['anthropogenic_heat']<0):raise SchemaError('粗糙度需正值，叶面积指数与人为热通量需非负')
    if any(np.any((values[k]<0)|(values[k]>1)) for k in ('albedo','soil_heat_flux')):raise SchemaError('反照率与土壤热通量参数需0..1')
    if np.any(values['bowen']==-1):raise SchemaError('Bowen比-1使热通量计算分母为0')
    nx,ny=len(coords['x']),len(coords['y'])
    lines=[f'{"GEO.DAT":16}{"2.0":16}{"EcoDataEngine explicit grids":64}',f'{1:4d}','User supplied static fields; no defaults or land-use lookup.',f'{"UTM":8}',f'{zone:4d}{hemisphere:4}',f'{"WGS-84":8}{"":12}',f'{nx:8d}{ny:8d}'+''.join(f'{v:12.3f}' for v in geometry),f'{"KM":4}','1',f'{len(categories)} {water[0]} {water[1]}',' '.join(map(str,categories))]
    def rows(a,integer=False):return [' '.join(str(int(v)) if integer else format(v,'.17g') for v in row) for row in a]
    lines+=rows(land,True)+['1.0']+rows(values['elevation'])
    for name in ('roughness','albedo','bowen','soil_heat_flux','anthropogenic_heat','leaf_area_index'):lines+=['2']+rows(values[name])
    scope='CALMET5.8.5 GEO.DAT 2.0 / WGS84 UTM方格静态资料；八字段全部来自用户，无土地参数默认值。需已有CALMET.INP网格/投影完全匹配；不生成气象场、控制文件或CALPUFF工程。'
    return {'GEO.DAT':('\n'.join(lines)+'\n').encode('ascii'),'README_GEO.txt':scope.encode('utf8')},dict(scope=scope,shape_y_x=[ny,nx],crs=crs.to_string(),origin_km=origin,grid_spacing_km=steps['x'],units=FIELDS,transformations=transforms,source=cfg['source'],time_processing='static; no time axis',complete_model_input=False,model_run_verified=False)
