"""User-supplied complete boundary concentrations to CMAQ5.5 BNDARY3.

Not the BCON/profile interpolation algorithm. Source points must already lie on
the target external perimeter; no vertical or horizontal concentration inference.
"""
from datetime import datetime,timezone
import numpy as np
from netCDF4 import Dataset
from ecodata.core.exceptions import SchemaError
from ecodata.models.adaptation import unit_transform
from ecodata.models.adapters.cmaq_gridded_emissions import _grid,_time,_griddesc,_number,CATALOG


def external_perimeter(nx, ny):
    """I/O API BNDARY3, NTHIK=1, using interior indices 1..nx and 1..ny."""
    return ([(i,0) for i in range(1,nx+2)] + [(nx+1,j) for j in range(1,ny+2)]
            + [(i,ny+1) for i in range(nx+1)] + [(0,j) for j in range(ny+1)])


def export_boundary(ds,cfg):
    if cfg.get('version')!='CMAQ-5.5' or cfg.get('mechanism')!=CATALOG['mechanism']:raise SchemaError('仅支持CMAQ5.5/cb6r3_ae7_aq边界资料直接编码')
    source=cfg.get('source')
    if not isinstance(source,dict) or not all(source.get(k) for k in ('kind','description','license')):raise SchemaError('需source.kind/description/license')
    axes=cfg.get('axes')
    if not isinstance(axes,dict) or set(axes)!={'time','layer','perimeter','x','y'} or len(set(axes.values()))!=5:raise SchemaError('axes需不同time/layer/perimeter/x/y映射')
    grid,ds=_grid(cfg,ds,axes)
    if cfg.get('time_mode')!='hourly' or ds[axes['time']].dtype.kind!='M':raise SchemaError('边界场需已解码UTC逐小时，不能把数值时间当纳秒')
    time,flags,sdate,stime,step=_time(cfg,ds,axes)
    if len(flags)<2:raise SchemaError('边界场需至少两个真实连续小时，覆盖区间两端；不复制单时刻')
    vert=cfg.get('vertical')
    if not isinstance(vert,dict) or set(vert)!={'VGTYP','VGTOP','VGLVLS'}:raise SchemaError('vertical需VGTYP/VGTOP/VGLVLS，与来源属性和气象一致')
    levels=np.asarray(vert['VGLVLS'],dtype=float);top=_number(vert['VGTOP'],'VGTOP')
    if vert['VGTYP']!=7 or top<=0 or levels.ndim!=1 or not 2<=len(levels)<=201 or not np.isfinite(levels).all() or not np.all(np.diff(levels)<0) or levels[0]!=1 or levels[-1]<0:raise SchemaError('当前垂直profile需VGTYP7、正VGTOP、从1递减到非负的层界面')
    nlay=len(levels)-1;layer=axes['layer'];perim=axes['perimeter']
    for key in ('VGTYP','VGTOP','VGLVLS'):
        if key not in ds.attrs or np.asarray(ds.attrs[key]).shape!=np.asarray(vert[key]).shape or not np.allclose(ds.attrs[key],vert[key],rtol=0,atol=1e-7):raise SchemaError(f'源{key}缺失或与配置冲突；不重标垂直层')
    if layer not in ds.coords or ds[layer].dims!=(layer,) or not np.array_equal(ds[layer],np.arange(1,nlay+1)):raise SchemaError('layer须按源VGLVLS从地面向上编号1..NLAYS；不推算层')
    if perim not in ds.coords or ds[perim].dims!=(perim,) or len(np.unique(ds[perim]))!=ds.sizes[perim]:raise SchemaError('perimeter需唯一一维标识')
    nx,ny=grid['NCOLS'],grid['NROWS']
    expected=external_perimeter(nx,ny)
    pm=cfg.get('perimeter_mapping')
    if not isinstance(pm,dict) or set(pm)!={'column','row','x','y'}:raise SchemaError('perimeter_mapping需column/row/x/y四个源变量；含外圈0索引')
    vectors={}
    for role,name in pm.items():
        if name not in ds or ds[name].dims!=(perim,):raise SchemaError(f'边界{role}需perimeter一维源变量')
        a=np.asarray(ds[name],dtype=float)
        if not np.isfinite(a).all():raise SchemaError('周界位置存在缺测')
        vectors[role]=a
    if any(np.any(vectors[k]%1) for k in ('column','row')):raise SchemaError('周界column/row需整数')
    pairs=list(zip(vectors['column'].astype(int),vectors['row'].astype(int)))
    if len(pairs)!=len(expected) or len(set(pairs))!=len(pairs) or set(pairs)!=set(expected):raise SchemaError('外圈NTHIK=1周界有缺格/重复/位置错误')
    for role,origin,spacing,idx in [('x','XORIG','XCELL','column'),('y','YORIG','YCELL','row')]:
        unit=ds[pm[role]].attrs.get('units')
        if unit!=cfg['coordinate_units']:raise SchemaError(f'周界{role}单位与grid不符')
        expected_pos=grid[origin]+(vectors[idx]-.5)*grid[spacing]
        if not np.allclose(vectors[role],expected_pos,rtol=0,atol=1e-7 if grid['GDTYP']==1 else 1e-3):raise SchemaError(f'周界{role}真实坐标与声明格心不符')
    order={p:i for i,p in enumerate(pairs)};permutation=[order[p] for p in expected]
    fields=cfg.get('fields');required=cfg.get('required_species');markers=cfg.get('missing_values')
    if not isinstance(fields,dict) or not fields or not isinstance(required,list) or not required or set(required)-set(fields) or len(required)!=len(set(required)):raise SchemaError('需fields及无缺项的required_species项目清单；不补物种')
    if not isinstance(markers,list):raise SchemaError('需明确missing_values列表')
    arrays={};units={};errors={};used=[]
    for name,m in fields.items():
        if not isinstance(m,dict) or set(m)!={'source','type','source_units'}:raise SchemaError('物种配置需source/type/source_units')
        kind=m['type'];accepted=CATALOG['gas_molecular_weights_g_mol'] if kind=='gas' else CATALOG['aerosol_mass'] if kind=='aerosol_mass' else []
        if name not in accepted or m['source'] not in ds.data_vars:raise SchemaError(f'{name}物种或源变量无效')
        field=ds[m['source']];dims=(time,layer,perim)
        if field.dtype.kind not in 'iuf':raise SchemaError(f'{name}需实数浓度，不能丢弃复数或转换布尔值')
        if len(field.dims)!=3 or set(field.dims)!=set(dims):raise SchemaError(f'{name}需time×layer×perimeter完整维度')
        if field.attrs.get('units')!=m['source_units']:raise SchemaError(f'{name}源单位与声明冲突')
        raw=np.asarray(field.transpose(*dims),dtype=float)[:,:,permutation]
        for marker in markers+list(np.atleast_1d(field.attrs.get('missing_value',[])))+list(np.atleast_1d(field.attrs.get('_FillValue',[]))):
            if np.any(raw==_number(marker,'缺测标记')):raise SchemaError(f'{name}含缺测标记')
        if not np.isfinite(raw).all() or np.any(raw<0):raise SchemaError(f'{name}含缺测或负浓度；不截零')
        if kind=='gas':
            factors={'ppmV':1.,'ppm':1.,'ppbV':.001,'ppb':.001,'mol/mol':1e6}
            if m['source_units'] not in factors:raise SchemaError('气相需混合比单位；不以未知温压换算质量浓度')
            scale=factors[m['source_units']];target='ppmV'
        else:
            scale,offset=unit_transform(m['source_units'],'ug/m^3');target='ug m-3'
            if offset:raise SchemaError('浓度单位不可带偏移')
        values=raw*scale;arr=values.astype('f4')
        if not np.isfinite(arr).all() or np.any((values!=0)&(arr==0)):raise SchemaError('浓度float32溢出或下溢')
        arrays[name]=arr;units[name]=target;used.append(m['source']);errors[name]=float(np.max(np.abs(values-arr)))
    if len(set(used))!=len(used):raise SchemaError('同一源变量不能复制成多个机制物种')
    nc=Dataset('cmaq_boundary','w',format='NETCDF3_CLASSIC',memory=65536)
    try:
        for name,size in {'TSTEP':None,'DATE-TIME':2,'LAY':nlay,'VAR':len(fields),'PERIM':len(expected)}.items():nc.createDimension(name,size)
        stamp=datetime.now(timezone.utc)
        integers=dict(FTYPE=2,CDATE=int(stamp.strftime('%Y%j')),CTIME=int(stamp.strftime('%H%M%S')),WDATE=int(stamp.strftime('%Y%j')),WTIME=int(stamp.strftime('%H%M%S')),SDATE=sdate,STIME=stime,TSTEP=step,NTHIK=1,NCOLS=nx,NROWS=ny,NLAYS=nlay,NVARS=len(fields),GDTYP=grid['GDTYP'],VGTYP=7)
        for name,v in integers.items():nc.setncattr(name,np.int32(v))
        for name in ('P_ALP','P_BET','P_GAM','XCENT','YCENT','XORIG','YORIG','XCELL','YCELL'):nc.setncattr(name,np.float64(grid[name]))
        nc.VGTOP=np.float32(top);nc.VGLVLS=levels.astype('f4')
        nc.setncatts({'IOAPI_VERSION':'EcoDataEngine compatible BNDARY3; native library not executed'.ljust(80),'EXEC_ID':'EcoDataEngine'.ljust(80),'GDNAM':grid['GDNAM'].ljust(16),'UPNAM':'ECODATA_BCON'.ljust(16),'VAR-LIST':''.join(k.ljust(16) for k in fields),'FILEDESC':'CMAQ5.5 directly supplied boundary concentrations; no interpolation'.ljust(4800),'HISTORY':source['description'][:4800].ljust(4800)})
        v=nc.createVariable('TFLAG','i4',('TSTEP','VAR','DATE-TIME'));v.units='<YYYYDDD,HHMMSS>'.ljust(16);v.long_name='TFLAG'.ljust(16);v.var_desc='UTC time flags'.ljust(80);v[:]=np.repeat(flags[:,None,:],len(fields),axis=1)
        for name,a in arrays.items():
            v=nc.createVariable(name,'f4',('TSTEP','LAY','PERIM'));v.units=units[name].ljust(16);v.long_name=name.ljust(16);v.var_desc=(name+' supplied boundary concentration').ljust(80);v[:]=a
        content=bytes(nc.close());nc=None
    finally:
        if nc is not None:nc.close()
    scope='CMAQ5.5 BNDARY3显式外圈浓度，NTHIK=1；用户须已给目标网格/垂直层/UTC小时全部浓度与位置。这不是BC_PROFILE，也不从静态剖面推算三维边界；需匹配MCIP与项目完整机制，未执行BCON/CCTM。'
    return {'boundary.CMAQ55.nc':content,'GRIDDESC':_griddesc(grid),'README_CMAQ_boundary.txt':scope.encode('utf8')},dict(scope=scope,source=source,grid=grid,vertical=vert,perimeter_count=len(expected),fields=list(fields),units=units,float32_max_absolute_error=errors,time_processing='none; preserve UTC hours',complete_model_input=False,model_run_verified=False)
