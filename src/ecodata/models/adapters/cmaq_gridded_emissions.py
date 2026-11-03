"""CMAQ 5.5 GRDDED3 emissions from explicitly mapped, already allocated data.

No chemical speciation, spatial allocation, hourly-profile generation or vertical
injection is inferred. netCDF4 and the existing Pint engine own storage/units.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pyproj import Geod
from ecodata.core.exceptions import SchemaError
from ecodata.models.adaptation import unit_transform

CATALOG = json.loads(Path(__file__).with_name('cmaq_emission_species.json').read_text())


def _griddesc(grid):
    """CMAS I/O API ASCII description of this already-validated single grid."""
    projection='ECO_LATLON' if grid['GDTYP']==1 else 'ECO_LAMBERT'
    parameters=' '.join(format(grid[k],'.17g') for k in ('P_ALP','P_BET','P_GAM','XCENT','YCENT'))
    geometry=' '.join(format(grid[k],'.17g') for k in ('XORIG','YORIG','XCELL','YCELL'))
    return (f"' '\n'{projection}'\n{grid['GDTYP']} {parameters}\n' '\n"
            f"'{grid['GDNAM']}'\n'{projection}' {geometry} {grid['NCOLS']} {grid['NROWS']} 1\n' '\n").encode('ascii')


def _number(value, field, integer=False):
    if isinstance(value, (bool, np.bool_)):
        raise SchemaError(f'{field}不能为布尔值')
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise SchemaError(f'{field}需有限数值') from exc
    if not np.isfinite(result) or (integer and result != int(result)):
        raise SchemaError(f'{field}需有限' + ('整数' if integer else '数值'))
    return int(result) if integer else result


def _grid(config, dataset, axes):
    g = config.get('grid')
    numeric = ('GDTYP','P_ALP','P_BET','P_GAM','XCENT','YCENT','XORIG','YORIG','XCELL','YCELL','NCOLS','NROWS')
    if not isinstance(g, dict) or set(numeric + ('GDNAM',)) - set(g):
        raise SchemaError('grid需完整I/O API投影、原点、格距、行列数和GDNAM，不从经纬度猜测投影')
    grid = {k:_number(g[k], k, k in {'GDTYP','NCOLS','NROWS'}) for k in numeric}
    if grid['GDTYP'] not in {1,2}:
        raise SchemaError('本入口只验证GDTYP=1经纬度或2 Lambert规则网格')
    name = g['GDNAM']
    if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,16}',name):
        raise SchemaError('GDNAM需1–16字符安全ASCII网格名')
    grid['GDNAM'] = name
    if any(grid[k] <= 0 for k in ('NCOLS','NROWS','XCELL','YCELL')):
        raise SchemaError('行列数、格距必须为正；禁止隐式改变网格大小')
    if grid['NCOLS']*grid['NROWS'] > 10000000:
        raise SchemaError('本入口单层上限1000万格，不隐式截取')
    if not isinstance(axes,dict) or not all(axes.get(k) for k in ('x','y')):
        raise SchemaError('axes需明确x/y源坐标名称')
    expected_unit = 'degree' if grid['GDTYP']==1 else 'm'
    if config.get('coordinate_units') != expected_unit:
        raise SchemaError(f'coordinate_units需明确为{expected_unit}，与目标投影匹配')
    normalized = dataset
    for role, nkey, okey, dkey in (('x','NCOLS','XORIG','XCELL'),('y','NROWS','YORIG','YCELL')):
        name = axes[role]
        if name not in dataset.coords or dataset[name].dims != (name,):
            raise SchemaError(f'{role}必须为一维同名坐标，不支持曲线网格或自动重投影')
        values = np.asarray(dataset[name], dtype=float)
        if len(values)!=grid[nkey] or not np.isfinite(values).all() or len(np.unique(values)) != len(values):
            raise SchemaError(f'{role}坐标有缺失/重复或行列数不符')
        if len(values)>1 and not (np.all(np.diff(values)>0) or np.all(np.diff(values)<0)):
            raise SchemaError(f'{role}坐标必须单调，不能静默重新组织非规则网格')
        normalized = normalized.sortby(name)
        expected = grid[okey]+(np.arange(grid[nkey])+.5)*grid[dkey]
        tolerance = 1e-7 if expected_unit=='degree' else 1e-3
        if not np.allclose(np.sort(values), expected, atol=tolerance, rtol=0):
            raise SchemaError(f'{role}坐标与grid格心不符，不进行隐式重采样（容差{tolerance}{expected_unit}）')
        actual_unit = str(dataset[name].attrs.get('units',''))
        permitted = {'degree','degrees','degrees_east','degrees_north'} if expected_unit=='degree' else {'m','meter','metre','meters','metres'}
        if actual_unit not in permitted:
            raise SchemaError(f'{role}源坐标units不明确或与{expected_unit}不符')
    if grid['GDTYP']==1:
        if (grid['XORIG'] < -180 or grid['XORIG']+grid['NCOLS']*grid['XCELL'] >180 or
                grid['YORIG'] < -90 or grid['YORIG']+grid['NROWS']*grid['YCELL']>90):
            raise SchemaError('经纬度网格边界超范围；跨日期变更线需上游明确处理')
    else:
        # The coordinate numbers alone do not identify a Lambert projection.
        if abs(grid['XCENT']-grid['P_GAM'])>1e-9:
            raise SchemaError('当前Lambert需XCENT=P_GAM；偏移中心需上游明确重投影，不能猜测')
        crs = config.get('source_crs')
        if not crs:
            raise SchemaError('Lambert输入需source_crs，不能只给x/y数字')
        from pyproj import CRS
        radius = _number(config.get('earth_radius_m'), 'earth_radius_m')
        if radius <= 0:
            raise SchemaError('Lambert earth_radius_m必须为正且与项目GRIDDESC一致')
        try:
            target = CRS.from_proj4(f'+proj=lcc +lat_1={grid["P_ALP"]} +lat_2={grid["P_BET"]} +lat_0={grid["YCENT"]} +lon_0={grid["P_GAM"]} +R={radius} +units=m')
            if not CRS.from_user_input(crs).equals(target, ignore_axis_order=True):
                raise SchemaError('source_crs与grid Lambert/球体参数不符，不隐式重投影')
            if dataset.attrs.get('crs') and not CRS.from_user_input(dataset.attrs['crs']).equals(target, ignore_axis_order=True):
                raise SchemaError('数据内crs与配置投影冲突')
        except (ValueError, TypeError) as exc:
            raise SchemaError(f'投影配置无效：{exc}') from exc
    return grid, normalized


def _time(config, dataset, axes):
    mode = config.get('time_mode')
    if mode == 'static_inventory':
        if not isinstance(config.get('static_mean_description'),str) or not config['static_mean_description'].strip():
            raise SchemaError('static_inventory必须说明原始平均期及静态使用假设；不是逐小时排放')
        if axes.get('time'):
            raise SchemaError('静态模式不接受时间维；先选择并保留单个明确平均期，不能丢掉已有时间序列')
        return None, np.zeros((1,2), dtype=np.int32), 0, 0, 0
    if mode != 'hourly' or not axes.get('time'):
        raise SchemaError('time_mode需hourly（真实逐小时）或static_inventory（明确平均期静态文件）')
    name = axes['time']
    if name not in dataset.coords or dataset[name].dims != (name,):
        raise SchemaError('hourly需一维真实时间坐标')
    if dataset[name].dtype.kind != 'M':
        raise SchemaError('hourly需已解码datetime64时间；不能把无单位数值当纳秒日期')
    if config.get('source_timezone') != 'UTC':
        raise SchemaError('本入口要求source_timezone=UTC；本地时间请先用通用时间模块明确转换')
    try:
        times = pd.DatetimeIndex(pd.to_datetime(dataset[name].values))
    except (ValueError, TypeError) as exc:
        raise SchemaError('时间坐标不是Gregorian可解析日期') from exc
    if not len(times) or times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
        raise SchemaError('时间缺失、重复或倒序；不补造日期')
    if any(t.minute or t.second or t.microsecond or t.nanosecond for t in times):
        raise SchemaError('hourly必须UTC整点，不静默取整')
    if len(times)>1 and not np.all(np.diff(times.values)==np.timedelta64(3600,'s')):
        raise SchemaError('真实逐小时数据必须连续，不能复制或插值缺失小时')
    flags = np.array([(int(t.strftime('%Y%j')),int(t.strftime('%H%M%S'))) for t in times],dtype=np.int32)
    return name, flags, int(flags[0,0]), int(flags[0,1]), 10000


def _area(config, dataset, grid, axes):
    method = config.get('area_method')
    if method == 'source_variable':
        name = config.get('cell_area_variable')
        if name not in dataset:
            raise SchemaError('cell_area_variable未对应真实网格面积变量')
        area = dataset[name]
        if set(area.dims) != {axes['y'],axes['x']}:
            raise SchemaError('网格面积必须为同网格二维y/x变量')
        scale, offset = unit_transform(area.attrs.get('units',''), 'm**2')
        if offset:
            raise SchemaError('面积单位换算不能有偏移')
        values = np.asarray(area.transpose(axes['y'],axes['x']),dtype=float)*scale
    elif method == 'geodesic_WGS84' and grid['GDTYP']==1:
        geod=Geod(ellps='WGS84');values=np.empty((grid['NROWS'],grid['NCOLS']))
        for j in range(grid['NROWS']):
            south=grid['YORIG']+j*grid['YCELL'];north=south+grid['YCELL']
            for i in range(grid['NCOLS']):
                west=grid['XORIG']+i*grid['XCELL'];east=west+grid['XCELL']
                values[j,i]=abs(geod.polygon_area_perimeter([west,east,east,west],[south,south,north,north])[0])
    else:
        raise SchemaError('通量→格排放需真实面积变量，或明确选择经纬度网格geodesic_WGS84；不能猜测面积')
    if not np.isfinite(values).all() or (values<=0).any():
        raise SchemaError('网格面积有缺失/非正值')
    return values


def export_gridded_emissions(dataset, config):
    if config.get('version') != 'CMAQ-5.5' or config.get('mechanism') != CATALOG['mechanism']:
        raise SchemaError('当前网格排放只核对CMAQ5.5/cb6r3_ae7_aq；其他机制需独立物种表适配')
    if config.get('layer_mode') != 'surface':
        raise SchemaError('当前仅真实已分配到地面层的排放；不猜测高架源层分配')
    source = config.get('source')
    if not isinstance(source,dict) or not all(isinstance(source.get(k),str) and source[k] for k in ('kind','description')):
        raise SchemaError('source需明确kind和description，不将合成数据称为实测')
    fields = config.get('fields')
    if not isinstance(fields,dict) or not fields:
        raise SchemaError('需明确目标物种→源变量与物理单位映射')
    if len(fields)>2048:
        raise SchemaError('物种数超出本入口上限2048')
    axes = config.get('axes')
    grid, dataset = _grid(config,dataset,axes)
    time_name, flags, sdate, stime, step = _time(config,dataset,axes)
    dims = ([time_name] if time_name else [])+[axes['y'],axes['x']]
    vertical = config.get('vertical')
    if not isinstance(vertical,dict) or set(('VGTYP','VGTOP','VGLVLS'))-set(vertical):
        raise SchemaError('需明确vertical.VGTYP/VGTOP/VGLVLS，与目标气象工程一致')
    vgt=_number(vertical['VGTYP'],'VGTYP',True);top=_number(vertical['VGTOP'],'VGTOP')
    levels=np.asarray(vertical['VGLVLS'],dtype=float)
    if vgt!=7 or top<=0 or levels.shape!=(2,) or not np.isfinite(levels).all() or not 0<=levels[1]<levels[0]<=1:
        raise SchemaError('当前仅支持VGTYP=7、正VGTOP、两个递减0..1 sigma-pressure边界的单地面层')
    arrays={};units={};operations={};area=None
    used=[]
    for target, mapping in fields.items():
        if not isinstance(mapping,dict) or not re.fullmatch(r'[A-Z][A-Z0-9_]{0,15}',str(target)):
            raise SchemaError(f'无效物种名或字段配置：{target}')
        kind=mapping.get('type');name=mapping.get('source')
        accepted=CATALOG['gas_molecular_weights_g_mol'] if kind=='gas' else CATALOG['aerosol_mass'] if kind=='aerosol_mass' else []
        if target not in accepted:
            raise SchemaError(f'{target}不是所选机制{kind}物种；不把PM25/NOx等总量自动拆分')
        if name not in dataset.data_vars:
            raise SchemaError(f'{target}缺源变量{name!r}')
        field=dataset[name]
        if set(field.dims)!=set(dims) or len(field.dims)!=len(dims):
            raise SchemaError(f'{target}维度必须为{dims}，不丢弃时间/垂直层/额外维度')
        if field.dtype.kind not in 'iuf':raise SchemaError(f'{target}必须为实数变量，不能为布尔/字符串')
        raw=np.asarray(field.transpose(*dims),dtype=float)
        markers=config.get('missing_values')
        if not isinstance(markers,list):
            raise SchemaError('missing_values必须明确列表，即使无缺测标记也需[]')
        if not np.isfinite(raw).all() or (raw<0).any():
            raise SchemaError(f'{target}含缺失/NaN/负排放；不截零')
        for marker in markers:
            if np.any(raw==_number(marker,'missing_values')):
                raise SchemaError(f'{target}含源缺测标记{marker}')
        declared=mapping.get('source_units');actual=field.attrs.get('units')
        if not declared or actual != declared:
            raise SchemaError(f'{target}声明单位{declared!r}与源units{actual!r}不一致；不只改标签')
        basis=mapping.get('basis')
        if basis not in {'cell_rate','surface_flux'}:
            raise SchemaError(f'{target}basis需cell_rate或surface_flux，不能把浓度当排放')
        outunit='moles/s' if kind=='gas' else 'g/s'
        normalized_unit='mol/s' if kind=='gas' else 'g/s'
        denominator='/m**2' if basis=='surface_flux' else ''
        try:
            scale,offset=unit_transform(declared,normalized_unit+denominator)
            transform_note={'scale':scale,'offset':offset}
        except SchemaError:
            if kind!='gas':
                raise SchemaError(f'{target}单位无法变为质量排放率，不推算排放')
            scale,offset=unit_transform(declared,'g/s'+denominator)
            mw=_number(mapping.get('molecular_weight_g_mol'),f'{target}.molecular_weight_g_mol')
            if mw<=0 or not mapping.get('molecular_weight_source'):
                raise SchemaError(f'{target}质量→摩尔需正摩尔质量及其依据，不从物种名称猜测')
            scale/=mw;transform_note={'scale':scale,'offset':offset,'molecular_weight_g_mol':mw,'molecular_weight_source':mapping['molecular_weight_source']}
        if offset!=0:
            raise SchemaError('排放率单位转换不应有偏移')
        values=raw*scale
        if basis=='surface_flux':
            if area is None:area=_area(config,dataset,grid,axes)
            values*=area
            transform_note['area_method']=config['area_method']
        if not time_name:values=values[None,:,:]
        converted=values[:,None,:,:].astype(np.float32)
        if ((values[:,None,:,:]!=0)&(converted==0)).any():raise SchemaError(f'{target}float32下溢，不能静默变为0')
        if not np.isfinite(converted).all():
            raise SchemaError(f'{target}输出float32溢出')
        arrays[target]=converted;units[target]=outunit;operations[target]=transform_note;used.append(name)
    if len(used)!=len(set(used)):
        raise SchemaError('同一源变量不能隐式复制为多个机制物种；需要上游明确物种分配')
    stamp=datetime.now(timezone.utc)
    ncf=Dataset('CMAQ_emissions','w',format='NETCDF3_CLASSIC',memory=65536)
    try:
        for name,size in {'TSTEP':None,'DATE-TIME':2,'LAY':1,'VAR':len(fields),'ROW':grid['NROWS'],'COL':grid['NCOLS']}.items():ncf.createDimension(name,size)
        integer_meta={'FTYPE':1,'CDATE':int(stamp.strftime('%Y%j')),'CTIME':int(stamp.strftime('%H%M%S')),'WDATE':int(stamp.strftime('%Y%j')),'WTIME':int(stamp.strftime('%H%M%S')),
                      'SDATE':sdate,'STIME':stime,'TSTEP':step,'NTHIK':1,'NCOLS':grid['NCOLS'],'NROWS':grid['NROWS'],'NLAYS':1,'NVARS':len(fields),'GDTYP':grid['GDTYP'],'VGTYP':vgt}
        for key,val in integer_meta.items():ncf.setncattr(key,np.int32(val))
        for key in ('P_ALP','P_BET','P_GAM','XCENT','YCENT','XORIG','YORIG','XCELL','YCELL'):ncf.setncattr(key,np.float64(grid[key]))
        ncf.setncattr('VGTOP',np.float32(top));ncf.setncattr('VGLVLS',levels.astype(np.float32))
        strings={'IOAPI_VERSION':'EcoDataEngine I/O API compatible netCDF; native library NOT run'.ljust(80),'EXEC_ID':'EcoDataEngine'.ljust(80),'GDNAM':grid['GDNAM'].ljust(16),'UPNAM':'ECODATA_EMIS'.ljust(16),
                 'VAR-LIST':''.join(k.ljust(16) for k in fields),'FILEDESC':'CMAQ 5.5 gridded surface emissions; no spatial/temporal/species allocation'.ljust(4800),'HISTORY':str(source['description'])[:4800].ljust(4800)}
        for key,val in strings.items():ncf.setncattr(key,val)
        tflag=ncf.createVariable('TFLAG','i4',('TSTEP','VAR','DATE-TIME'));tflag.long_name='TFLAG'.ljust(16);tflag.units='<YYYYDDD,HHMMSS>'.ljust(16);tflag.var_desc='Timestep-valid flags; zero for explicit time-independent inventory'.ljust(80)
        tflag[:]=np.repeat(flags[:,None,:],len(fields),axis=1)
        for name,values in arrays.items():
            variable=ncf.createVariable(name,'f4',('TSTEP','LAY','ROW','COL'));variable.long_name=name.ljust(16);variable.units=units[name].ljust(16);variable.var_desc=(name+' surface emission rate')[:80].ljust(80);variable[:]=values
        content=bytes(ncf.close());ncf=None
    finally:
        if ncf is not None:ncf.close()
    notes=('CMAQ5.5 cb6r3_ae7_aq GRDDED3地面排放文件，目标逻辑流GR_EMIS_XXX。'
           '需要与MCIP网格/垂直定义和DESID物种映射匹配；不生成气象、初始/边界场或控制工程。'
           'hourly仅保留真实逐小时；static_inventory保存明确平均期的时间不变排放，TSTEP/SDATE/STIME/TFLAG=0，'
           '不是实测逐小时排放，应用到仿真时需用户明确静态清单时间使用假设。'
           '没有化学物种拆分、时间分配、空间重采样或高架源注入。')
    return {'emissions.CMAQ55.nc':content,'GRIDDESC':_griddesc(grid),'README_CMAQ_emissions.txt':notes.encode('utf8')}, {
        'target':'CMAQ 5.5 GR_EMIS_XXX / I/O API GRDDED3','mechanism':config['mechanism'], 'grid':grid,'time_mode':config['time_mode'],
        'times_utc': [pd.Timestamp(t).isoformat() for t in dataset[time_name].values] if time_name else [],'time_interpolation':'none',
        'variable_units':units,'conversion_operations':operations,'surface_layer_only':True,'source':source,
        'unmapped_source_variables':[k for k in dataset.data_vars if k not in used and k!=config.get('cell_area_variable')],
        'numeric_storage':'IEEE float32; relative rounding bound 1e-7 except subnormal values', 'scope':notes,
        'griddesc_scope':'Single emission grid only; horizontal geometry matches NetCDF. No vertical/meteorological data in GRIDDESC.',
        'verification':'native CMAQ/I/O API library not executed; external independent parsing required'}
