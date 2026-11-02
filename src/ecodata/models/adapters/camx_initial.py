"""Explicit 3-D concentration tables to CAMx 7.40 INITIAL NetCDF.

Original implementation of the published ncf_cncprep/ncf_readcnc interface.
No model code, interpolation, chemical allocation or missing-species defaults.
"""
from datetime import timezone,timedelta
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from ecodata.core.exceptions import SchemaError
from ecodata.models.adaptation import unit_transform
from ecodata.models.adapters.camx_emissions import _grid,_number,GASES,AEROSOLS,SUPPORTED_MECHANISMS


def export_initial(frame,cfg):
    if cfg.get('version')!='CAMx-7.40' or not isinstance(cfg.get('mechanism'),str) or cfg['mechanism'] not in SUPPORTED_MECHANISMS:raise SchemaError('仅支持CAMx7.40的CB6r5_CF3_SIMPLE_NoEvap或CB6r5_CF3_COMPLX INITIAL；必须与项目CHEMPARAM一致，不进行机制互换')
    source=cfg.get('source')
    if not isinstance(source,dict) or not all(source.get(k) for k in ('kind','description','license')):raise SchemaError('source需kind/description/license，合成数据须如实声明')
    fields=cfg.get('fields');species=cfg.get('species');markers=cfg.get('missing_values')
    if not isinstance(fields,dict) or set(fields)!={'time','row','col','layer'}:raise SchemaError('fields需time/row/col/layer映射')
    if not isinstance(species,list) or not species or not isinstance(markers,list):raise SchemaError('需明确species及missing_values列表')
    layers=_number(cfg.get('layer_count'),'layer_count',True)
    if not 1<=layers<=200 or not isinstance(cfg.get('layer_definition'),str) or not cfg['layer_definition'].strip():raise SchemaError('需1..200真实层及layer_definition来源说明；不能猜测层厚')
    if type(cfg.get('index_base'))!=int or cfg['index_base'] not in {0,1}:raise SchemaError('index_base需0或1，适用于row/col/layer')
    g,x,y,lon,lat=_grid(cfg.get('grid'));names=[];columns=[];factors=[];units=[]
    for spec in species:
        if not isinstance(spec,dict) or set(spec)!={'name','column','phase','unit'}:raise SchemaError('species每项需name/column/phase/unit')
        name=spec['name'];phase=spec['phase'];unit=spec['unit']
        if phase not in {'gas','aerosol'} or name not in (GASES if phase=='gas' else AEROSOLS):raise SchemaError(f'{name}不在所选机制{phase}物种表；不拆分总量')
        if phase=='gas':
            choices={'ppm':1.,'ppmV':1.,'ppb':.001,'ppbV':.001,'mol/mol':1e6}
            if unit not in choices:raise SchemaError('气相浓度需ppmV/ppbV/mol/mol；不无依据地做质量体积到摩尔混合比转换')
            factor=choices[unit];target='ppm'
        else:
            factor,offset=unit_transform(unit,'ug/m^3');target='ug/m3'
            if offset or factor<=0:raise SchemaError('气溶胶质量浓度单位需正比例转换')
        names.append(name);columns.append(spec['column']);factors.append(factor);units.append(target)
    mapped=[*fields.values(),*columns]
    if len(set(mapped))!=len(mapped) or len(set(names))!=len(names) or set(frame.columns)!=set(mapped):raise SchemaError('缺列、未映射列或重复物种/字段')
    required=cfg.get('required_species')
    if not isinstance(required,list) or not required or len(set(required))!=len(required) or set(required)-set(names):raise SchemaError('必须明确项目required_species并全部提供；不让目标读取器补默认物种')
    if frame.empty or frame.isna().any().any():raise SchemaError('初始场为空或含缺测')
    rawtime=frame[fields['time']].astype(str)
    if not rawtime.str.contains(r'(?:Z|[+-]\d{2}:?\d{2})$',regex=True).all():raise SchemaError('时间必须带明确UTC偏移')
    try:times=pd.to_datetime(rawtime,utc=True,format='mixed');end=pd.Timestamp(cfg.get('valid_until'))
    except Exception as exc:raise SchemaError('初始时刻或valid_until无效') from exc
    if times.nunique()!=1 or end.tzinfo is None:raise SchemaError('单初始快照要求唯一时刻及带时区valid_until')
    instant=times.iloc[0]
    if not pd.Timedelta(0)<end-instant<=pd.Timedelta(days=1):raise SchemaError('需用户明确0..24小时valid_until以满足CAMx时间区间接口；不能隐式延长')
    localzone=timezone(timedelta(hours=-g['itzon']));start=instant.tz_convert(localzone);end=end.tz_convert(localzone)
    if any(t.minute or t.second or t.microsecond or t.nanosecond for t in (start,end)):raise SchemaError('当前初始场profile要求整点区间边界')
    indices=[]
    for role,size in [('layer',layers),('row',g['nrows']),('col',g['ncols'])]:
        try:a=pd.to_numeric(frame[fields[role]],errors='raise').to_numpy(dtype=float)-cfg['index_base']
        except (ValueError,TypeError) as exc:raise SchemaError(f'{role}需整数索引') from exc
        if not np.isfinite(a).all() or np.any(a%1) or np.any(a<0) or np.any(a>=size):raise SchemaError(f'{role}索引范围/数值无效')
        indices.append(a.astype(int))
    index=pd.MultiIndex.from_arrays(indices);expected=pd.MultiIndex.from_product([range(layers),range(g['nrows']),range(g['ncols'])])
    if index.has_duplicates or len(index)!=len(expected) or len(expected.difference(index)):raise SchemaError('初始场layer×row×col有缺格/缺层或重复；不补值')
    arrays={};errors={}
    for name,col,factor in zip(names,columns,factors):
        try:a=pd.to_numeric(frame[col],errors='raise').to_numpy(dtype=float)
        except (ValueError,TypeError) as exc:raise SchemaError(f'{name}浓度不是数值') from exc
        for marker in markers:
            try:v=float(marker)
            except (ValueError,TypeError):continue
            if np.any(a==v):raise SchemaError(f'{name}含缺测标记')
        a=a*factor
        if not np.isfinite(a).all() or np.any(a<0):raise SchemaError(f'{name}含缺测/溢出/负浓度；不截零')
        ordered=pd.Series(a,index=index).reindex(expected).to_numpy().reshape(1,layers,g['nrows'],g['ncols'])
        arr=ordered.astype('f4')
        if not np.isfinite(arr).all() or np.any((ordered!=0)&(arr==0)):raise SchemaError('float32溢出或下溢')
        arrays[name]=arr;errors[name]=float(np.max(np.abs(ordered-arr)))
    def flags(t):return np.repeat(np.array([[[int(t.strftime('%Y%j')),int(t.strftime('%H%M%S'))]]],dtype='i4'),len(names),axis=1)
    tf,ef=flags(start),flags(end);seconds=int((end-start).total_seconds());hours=seconds//3600
    # The Windows diskless NetCDF writer may leave non-deterministic padding
    # bytes in its returned memory buffer. Use its ordinary file backend, as
    # the existing CAMx emissions writer does, then read the finalized file.
    temporary=tempfile.TemporaryDirectory(prefix='ecodata_camx_initial_')
    path=Path(temporary.name)/'initial.nc'
    nc=None
    try:
        nc=Dataset(path,'w',format='NETCDF3_CLASSIC')
        for dim,size in [('TSTEP',None),('DATE-TIME',2),('VAR',len(names)),('LAY',layers),('ROW',g['nrows']),('COL',g['ncols'])]:nc.createDimension(dim,size)
        attrs={k.upper():g[k] for k in ('xorig','yorig','xcell','ycell','xcent','ycent','p_alp','p_bet','itzon')}
        attrs.update(CPROJ=g['cproj'],GDTYP=g['gdtyp'],IUTM=0,ISTAG=0,NCOLS=g['ncols'],NROWS=g['nrows'],NLAYS=layers,NVARS=len(names),
            SDATE=int(tf[0,0,0]),STIME=int(tf[0,0,1]),TSTEP=hours*10000,NSTEPS=1,P_GAM=g['xcent'],CAMx_NAME='INITIAL',
            NOTE='Supplied initial concentrations; no interpolation',FILEDESC='CAMx 7.40 INITIAL; user-declared validity interval',FTYPE=1,NTHIK=1,
            GDNAM='CAMx7.40 grid',UPNAM='EcoDataEngine',**{'VAR-LIST':''.join(n.ljust(16) for n in names)})
        nc.setncatts(attrs)
        for key,vals in [('TFLAG',tf),('ETFLAG',ef)]:
            v=nc.createVariable(key,'i4',('TSTEP','VAR','DATE-TIME'));v.units='YYYYDDD,HHMMSS';v.long_name=key;v.var_desc=key;v[:]=vals
        for name,unit in zip(names,units):
            v=nc.createVariable(name,'f4',('TSTEP','LAY','ROW','COL'));v.units=unit;v.long_name=name;v.var_desc=name+' supplied initial concentration';v[:]=arrays[name]
        nc.close();nc=None
        raw=path.read_bytes()
    finally:
        if nc is not None:nc.close()
        temporary.cleanup()
    scope='CAMx7.40 INITIAL三维浓度快照；仅用户给定物种/层/网格，不推导初始场。valid_until为用户明确提供的原生区间标签，不构造时间序列；需真实气象层和CAMx.in匹配。未执行CAMx读取或仿真。'
    return {'initial.CAMx7.40.nc':raw,'README_CAMx_initial.txt':scope.encode('utf8')},dict(scope=scope,source=source,mechanism=cfg['mechanism'],
        grid=g,layers=layers,layer_definition=cfg['layer_definition'],species=names,units=dict(zip(names,units)),time=start.isoformat(),valid_until=end.isoformat(),
        time_processing='no interpolation; explicit source validity interval',float32_max_absolute_error=errors,complete_model_input=False,model_run_verified=False)
