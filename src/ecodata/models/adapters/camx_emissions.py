"""CAMx 7.40/7.10 native NetCDF 2-D, hourly, already-speciated emissions.

Metadata follows CAMx ncf_areaprep/ncf_chk_griddef, not generic IO-API
renaming. Does not spatially allocate an inventory or invent chemical species.
"""
from datetime import timezone, timedelta
import io
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pyproj import CRS, Transformer
from ecodata.core.exceptions import SchemaError
from ecodata.models.adaptation import unit_transform

MECHANISM = 'CB6r5_CF3_SIMPLE_NoEvap'
# Official CAMx 7.40 CHEMPARAM files differ only in the SOAP3 option and
# description: gas/aerosol identifiers and all species records are identical.
# This selects a compatible input vocabulary; it does not convert chemistry.
SUPPORTED_MECHANISMS = frozenset({MECHANISM, 'CB6r5_CF3_COMPLX'})
# Species identifiers, not physical values, from the v7.40 official CHEMPARAM.
GASES = set('AACD ACET ALD2 ALDX BENZ BZO2 C2O3 CAT1 ACG2 ACG1 BCG2 BCG1 CO CRES CRO CRON CXO3 DMS ECH4 EPOX EPX2 ETH ETHA ETHY ETOH FACD FORM GLY GLYD H2O2 HCL HCO3 HIO3 HNO3 HO2 HOI HONO HPLD I I2 I2O2 INO3 INTR IO IOLE ISO2 ISOP ISPD ISPX IVOA IXOY KET MEO2 MEOH MEPX MGLY N2O5 NH3 NO NO2 NO3 NTR1 NTR2 O O1D O3 OH OIO OLE OPAN OPEN OPO3 PACD PAN PANX PAR PNA PRPA RO2 ROOH ROR SO2 SQT SULF SVOA TERP APIN TO2 TOL XLO2 XO2 XO2H XO2N XOPN XPAR XPRP XYL CGLY'.split())
AEROSOLS = set('CCRS CPRM FCRS FPRM NA PCL PEC PH2O PNH4 PNO3 PSO4 HOA AOA2 AOA1 AOA0 BOA2 BOA1 BOA0'.split())
# Species vocabulary from published CAMx7.1.chemparam.CB05_CF2, Zenodo
# 14944768. Keep version/mechanism pairs strict: no chemistry translation.
CB05_GASES = set('O1D O OH HO2 C2O3 XO2 XO2N CXO3 MEO2 TO2 ROR HCO3 CRO ETHA ETOH FACD FORM H2O2 HNO3 HONO IOLE ISOP ISPD MEOH MEPX MGLY N2O5 NO NO2 NO3 NTR O3 OLE OPEN PACD PAN PANX PAR PNA ROOH SO2 SULF TERP TOL XYL NH3 HCL SQT IVOA CG1 CG2 CG3 CG4 AACD ALD2 ALDX CO CRES ETH'.split())
CB05_AEROSOLS = set('PNO3 PSO4 PNH4 POA SOA1 SOA2 SOA3 SOA4 SOPA SOPB PEC FPRM FCRS CPRM CCRS NA PCL PH2O'.split())


def _number(value, name, integer=False):
    if isinstance(value, bool):
        raise SchemaError(f'{name}必须为数值，不能为布尔值')
    try:
        val = float(value)
    except (ValueError, TypeError) as exc:
        raise SchemaError(f'{name}必须为有限数值') from exc
    if not np.isfinite(val) or (integer and val != int(val)):
        raise SchemaError(f'{name}必须为有限' + ('整数' if integer else '数值'))
    return int(val) if integer else val


def _grid(grid):
    keys = {'projection', 'ncols', 'nrows', 'xorig', 'yorig', 'xcell', 'ycell',
            'coordinate_units', 'xcent', 'ycent', 'p_alp', 'p_bet', 'itzon'}
    if not isinstance(grid, dict) or set(grid) != keys:
        raise SchemaError('grid必须明确' + '/'.join(sorted(keys)))
    g = {k: _number(grid[k], k, k in {'ncols', 'nrows', 'itzon'})
         for k in keys - {'projection', 'coordinate_units'}}
    if g['ncols'] < 1 or g['nrows'] < 1 or g['ncols'] * g['nrows'] > 10000000:
        raise SchemaError('网格数必须为正；本入口上限1000万格，禁止隐式裁剪')
    if not -12 <= g['itzon'] <= 14:
        raise SchemaError('ITZON为UTC以西的整小时，范围-12..14；不使用夏令时')
    if not -180 <= g['xcent'] <= 180 or not -90 <= g['ycent'] <= 90:
        raise SchemaError('投影中心经纬度无效')
    projection = grid['projection']
    if projection == 'LATLON':
        if grid['coordinate_units'] != 'degree':
            raise SchemaError('LATLON网格必须明确degree单位')
        g.update(cproj=0, gdtyp=1)
        if g['p_alp'] != 0 or g['p_bet'] != 0:
            raise SchemaError('LATLON的p_alp/p_bet必须明确为0')
    elif projection == 'LCC':
        if grid['coordinate_units'] not in {'m', 'km'}:
            raise SchemaError('LCC网格必须明确m或km单位')
        fac = 1000 if grid['coordinate_units'] == 'km' else 1
        for k in ('xorig', 'yorig', 'xcell', 'ycell'):
            g[k] *= fac
        if not (-90 < g['p_alp'] < 90 and -90 < g['p_bet'] < 90) or abs(g['p_alp'] + g['p_bet']) < 1e-8:
            raise SchemaError('LCC标准纬线无效，不能位于南北极或互相抵消')
        if g['p_alp']*g['p_bet']<=0:
            raise SchemaError('CAMx LCC标准纬线必须位于同一半球且不为0')
        g.update(cproj=2, gdtyp=2)
    else:
        raise SchemaError('本入口仅支持明确的LATLON或CAMx球体LCC投影；其他投影需另行适配')
    if g['xcell'] <= 0 or g['ycell'] <= 0:
        raise SchemaError('网格xcell/ycell必须为正；row从南向北、col从西向东')
    x = g['xorig'] + (np.arange(g['ncols']) + .5) * g['xcell']
    y = g['yorig'] + (np.arange(g['nrows']) + .5) * g['ycell']
    xx, yy = np.meshgrid(x, y)
    if projection == 'LATLON':
        lon, lat = xx, yy
        if (g['xorig'] < -180 or g['xorig'] + g['ncols'] * g['xcell'] > 180
                or g['yorig'] < -90 or g['yorig'] + g['nrows'] * g['ycell'] > 90):
            raise SchemaError('经纬度网格边界越界；不自动跨日期变更线')
    else:
        # CAMx lcpgeo uses tangent LCC when the latitudes differ by <=0.1 deg.
        lat2=g['p_alp'] if abs(g['p_alp']-g['p_bet'])<=.1 else g['p_bet']
        crs = CRS.from_proj4(f'+proj=lcc +lat_1={g["p_alp"]} +lat_2={lat2} +lat_0={g["ycent"]} +lon_0={g["xcent"]} +R=6370000 +units=m +no_defs')
        lon, lat = Transformer.from_crs(crs, CRS.from_proj4('+proj=longlat +R=6370000'), always_xy=True).transform(xx, yy)
        if not np.isfinite(lon).all() or not np.isfinite(lat).all():
            raise SchemaError('LCC网格反投影失败')
    return g, x, y, lon, lat


def export_gridded_emissions(frame, config):
    """Map an explicit complete time×row×col table to native CAMx emissions."""
    required = {'mode', 'version', 'mechanism', 'fields', 'species', 'grid',
                'row_col_base', 'missing_values', 'time_basis', 'source'}
    if not isinstance(config, dict) or not required <= set(config) or set(config) - required - {'encoding'}:
        raise SchemaError('CAMx排放配置必须明确mode/version/mechanism/fields/species/grid/row_col_base/missing_values/time_basis/source')
    if config['mode'] != 'gridded_emissions' or config['version'] not in {'CAMx-7.40','CAMx-7.10'}:
        raise SchemaError('本入口支持CAMx-7.40或CAMx-7.10 mode=gridded_emissions，须显式匹配机制')
    legacy = config['version'] == 'CAMx-7.10'
    supported = {'CB05_CF2'} if legacy else SUPPORTED_MECHANISMS
    if not isinstance(config['mechanism'], str) or config['mechanism'] not in supported:
        raise SchemaError('版本与化学机制不匹配：7.10仅CB05_CF2；7.40仅CB6r5_CF3_SIMPLE_NoEvap/CB6r5_CF3_COMPLX；不进行机制互换')
    gases,aerosols = (CB05_GASES,CB05_AEROSOLS) if legacy else (GASES,AEROSOLS)
    if config['time_basis'] != 'interval_start_hourly':
        raise SchemaError('必须为interval_start_hourly：时间为每小时平均排放率区间起点，不将瞬时值或累计量冒充小时排放率')
    source = config['source']
    if not isinstance(source, dict) or not {'kind', 'url', 'license'} <= set(source) or not all(isinstance(source[k],str) and source[k].strip() for k in ('kind','url','license')):
        raise SchemaError('source需真实声明kind、url、license；合成或格式参考应明确标注')
    fields = config['fields']
    if not isinstance(fields, dict) or set(fields) != {'time', 'row', 'col'}:
        raise SchemaError('fields需映射time、row、col')
    if not all(isinstance(v,str) and v.strip() for v in fields.values()):
        raise SchemaError('fields映射必须为非空列名字符串')
    sp = config['species']
    if not isinstance(sp, list) or not sp:
        raise SchemaError('species需至少一个明确的模型物种，不推测物种拆分')
    names, columns, scales, units = [], [], [], []
    for spec in sp:
        if not isinstance(spec, dict) or set(spec) != {'name', 'column', 'phase', 'unit'}:
            raise SchemaError('每个species必须明确name/column/phase/unit')
        if not all(isinstance(spec[k],str) and spec[k].strip() for k in ('name','column','phase','unit')):
            raise SchemaError('species的name/column/phase/unit必须为非空字符串')
        name = spec['name']
        if not isinstance(name,str) or not re.fullmatch(r'[A-Z][A-Z0-9_]{0,9}', name) or name in {'X','Y','TFLAG','ETFLAG'}:
            raise SchemaError('模型物种名须为1..10字符大写标识符，不能与保留变量重名')
        if spec['phase'] not in {'gas', 'aerosol'}:
            raise SchemaError(f'{name}需明确gas或aerosol')
        if name not in (gases if spec['phase']=='gas' else aerosols):
            raise SchemaError(f'{name}不属于指定{config["version"]}机制的{spec["phase"]}物种；不自动把NOx/PM25改名或拆分')
        target = 'mol/hour' if spec['phase'] == 'gas' else 'g/hour'
        try:
            factor, offset = unit_transform(spec['unit'], target)
        except Exception as exc:
            raise SchemaError(f'{name}单位不能转换为{target}；质量与摩尔转换须由项目提供依据，本入口不猜分子量') from exc
        if offset != 0 or factor <= 0:
            raise SchemaError(f'{name}排放率单位转换必须为正比例')
        names.append(name); columns.append(spec['column']); scales.append(factor)
        units.append('mol hr-1' if spec['phase'] == 'gas' else 'g hr-1')
    mapped = [*fields.values(), *columns]
    if len(set(names)) != len(names) or len(set(mapped)) != len(mapped):
        raise SchemaError('物种名或字段映射重复，不允许覆盖')
    if len(frame.columns) != len(set(frame.columns)) or set(frame.columns) != set(mapped):
        raise SchemaError('CSV字段需与配置映射一致，存在缺列或未映射列；不静默丢弃')
    if not isinstance(config['missing_values'], list):
        raise SchemaError('missing_values须明确列出数据源缺测标记（无标记时[]）')
    if frame.empty or frame.isna().any().any():
        raise SchemaError('排放表为空或有缺失值；禁止补造')
    for marker in config['missing_values']:
        if frame.astype(str).eq(str(marker)).any().any():
            raise SchemaError(f'出现缺测标记{marker}；禁止填零')
    g, x, y, lon, lat = _grid(config['grid'])
    base = config['row_col_base']
    if type(base) is not int or base not in {0,1}:
        raise SchemaError('row_col_base必须为0或1，不能猜测格点索引')
    indices = []
    for k in ('row','col'):
        try: v=pd.to_numeric(frame[fields[k]],errors='raise').to_numpy(dtype=float)-base
        except (ValueError,TypeError) as exc: raise SchemaError(f'{k}包含非数值') from exc
        if not np.isfinite(v).all() or (v%1!=0).any() or (v<0).any() or (v>=g['nrows' if k=='row' else 'ncols']).any():
            raise SchemaError(f'{k}索引不是有效完整网格中的整数')
        indices.append(v.astype(int))
    raw = frame[fields['time']].astype(str)
    if not raw.str.contains(r'(?:Z|[+-]\d{2}:?\d{2})$',regex=True).all():
        raise SchemaError('每个时间必须带明确UTC偏移或Z，不能猜测时区')
    try: utc = pd.to_datetime(raw,utc=True,errors='raise',format='mixed')
    except (ValueError,TypeError) as exc: raise SchemaError('时间格式无效') from exc
    local = utc.dt.tz_convert(timezone(timedelta(hours=-g['itzon'])))
    if (local.dt.minute.ne(0)|local.dt.second.ne(0)|local.dt.microsecond.ne(0)|local.dt.nanosecond.ne(0)).any():
        raise SchemaError('CAMx逐小时排放必须位于整点，不能自动移动时间')
    times = sorted(local.unique())
    if len(times)>1 and any((b-a).total_seconds()!=3600 for a,b in zip(times,times[1:])):
        raise SchemaError('小时序列不连续；禁止复制、插值或补造缺失小时')
    keys = pd.MultiIndex.from_arrays([local, *indices],names=['time','row','col'])
    expected = pd.MultiIndex.from_product([times,range(g['nrows']),range(g['ncols'])],names=keys.names)
    if keys.has_duplicates or len(keys)!=len(expected) or len(expected.difference(keys)):
        raise SchemaError('存在重复或缺失的时间×网格记录；所有显式零排放也需提供')
    arrays = []
    errors = []
    for name,col,factor in zip(names,columns,scales):
        try:
            raw_values=pd.to_numeric(frame[col],errors='raise').to_numpy(dtype=float)
            for marker in config['missing_values']:
                try: numeric_marker=float(marker)
                except (ValueError,TypeError):continue
                if np.equal(raw_values,numeric_marker).any():
                    raise SchemaError(f'{name}出现数据源缺测标记{marker}；不填零')
            v = raw_values*factor
        except (ValueError,TypeError) as exc: raise SchemaError(f'{name}包含非数值') from exc
        if not np.isfinite(v).all() or (v<0).any():
            raise SchemaError(f'{name}含非有限或负排放；不截断')
        if np.max(v)>np.finfo(np.float32).max:
            raise SchemaError(f'{name}超出CAMx float32范围')
        ordered=pd.Series(v,index=keys).reindex(expected).to_numpy().reshape(len(times),1,g['nrows'],g['ncols'])
        arr=ordered.astype(np.float32)
        if ((arr==0)&(ordered!=0)).any():
            raise SchemaError(f'{name}低于float32范围；禁止静默下溢为0')
        arrays.append(arr); errors.append(float(np.max(np.abs(ordered-arr.astype(float)))))
    def flags(t): return [int(t.strftime('%Y%j')), int(t.strftime('%H%M%S'))]
    tf=np.repeat(np.asarray([flags(t) for t in times],dtype='i4')[:,None,:],len(names),axis=1)
    ef=np.repeat(np.asarray([flags(t+pd.Timedelta(hours=1)) for t in times],dtype='i4')[:,None,:],len(names),axis=1)
    with tempfile.TemporaryDirectory(prefix='ecodata_camx_') as td:
        path=Path(td)/'emissions.nc'
        with Dataset(path,'w',format='NETCDF3_CLASSIC') as out:
            for dim,size in [('TSTEP',None),('DATE-TIME',2),('VAR',len(names)),('LAY',1),('ROW',g['nrows']),('COL',g['ncols'])]:out.createDimension(dim,size)
            attrs={k.upper():g[k] for k in ('xorig','yorig','xcell','ycell','xcent','ycent','p_alp','p_bet','itzon')}
            attrs.update(CPROJ=g['cproj'],GDTYP=g['gdtyp'],IUTM=0,ISTAG=0,NCOLS=g['ncols'],NROWS=g['nrows'],NLAYS=1,NVARS=len(names),
                         SDATE=int(tf[0,0,0]),STIME=int(tf[0,0,1]),TSTEP=10000,NSTEPS=len(times),
                         P_GAM=g['xcent'],CAMx_NAME='EMISSIONS',NOTE='EcoDataEngine: supplied speciated hourly gridded rates',
                         FILEDESC=config['version']+' native area emissions; no invented species or values',FTYPE=1,NTHIK=1,
                         GDNAM=config['version']+' grid',UPNAM='EcoDataEngine',**{'VAR-LIST':''.join(n.ljust(16) for n in names)})
            out.setncatts(attrs)
            for key,vals in [('TFLAG',tf),('ETFLAG',ef)]:
                v=out.createVariable(key,'i4',('TSTEP','VAR','DATE-TIME'));v.units='YYYYDDD,HHMMSS';v.long_name=key;v.var_desc=key;v[:]=vals
            for key,vals,dims,unit in [('X',x,('COL',),'degrees' if g['cproj']==0 else 'km'),('Y',y,('ROW',),'degrees' if g['cproj']==0 else 'km'),
                                       ('longitude',lon,('ROW','COL'),'degrees east'),('latitude',lat,('ROW','COL'),'degrees north')]:
                v=out.createVariable(key,'f8',dims);v.units=unit;v.long_name=key;v.var_desc='cell centre '+key;v[:]=vals/1000 if key in {'X','Y'} and g['cproj']==2 else vals
            for name,arr,unit in zip(names,arrays,units):
                v=out.createVariable(name,'f4',('TSTEP','LAY','ROW','COL'));v.units=unit;v.long_name=name;v.var_desc=name+' supplied emission rate';v.coordinates='longitude latitude';v[:]=arr
        payload=path.read_bytes()
    report={'target':config['version'],'output_type':'native NetCDF 2D hourly gridded emissions candidate','mechanism':config['mechanism'],
            'species':names,'units':dict(zip(names,units)),'shape':[len(times),1,g['nrows'],g['ncols']],
            'grid':g,'row_order':'south to north','column_order':'west to east','ITZON_hours_west_of_UTC':g['itzon'],
            'time_range':[times[0].isoformat(),(times[-1]+pd.Timedelta(hours=1)).isoformat()],
            'float32_max_absolute_rounding_error':dict(zip(names,errors)),'missing_policy':'reject; no fill/clipping',
            'scope':'Already chemically speciated complete hourly grid-cell emission rates. No inventory allocation, speciation or temporal disaggregation.',
            'complete_model_input':False,'camx_run_verified':False,
            'next_steps':'Use Gridded_Emissions in a matching CAMx.in with the identical grid, standard time zone, mechanism and complete episode; other CAMx inputs still required.'}
    readme='CAMx 7.40二维逐小时网格排放候选文件。需匹配CAMx.in的网格、Time_Zone、化学机制及完整模拟时段。\n输入为已完成化学分种的每格小时平均排放率；本入口未生成空间分配、化学拆分、小时分配或其他模型输入。\n独立解析不代表CAMx已读取或完整工程可运行。X/Y坐标为km（LCC）或degree（LATLON），投影全局网格属性为m（LCC）或degree（LATLON），按官方接口分别处理。\n'
    if legacy:readme=readme.replace('CAMx 7.40','CAMx 7.10 CB05_CF2')
    return {('emissions.CAMx7.10.nc' if legacy else 'emissions.CAMx7.40.nc'):payload,'README_CAMx_emissions.txt':readme.encode('utf8')},report
