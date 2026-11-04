"""Bound input + actual composite raster -> existing fusion -> native writer.

Format/profile adapters declare physical meaning; no case paths, dimensions,
dates, archive hashes or preferred algorithms are used for dispatch.
"""
from contextlib import ExitStack
from pathlib import Path
import json,re
import numpy as np,pandas as pd,xarray as xr
from .project_binding import ProjectFiles
from .scientific_schema import identity
from .native_meteorology import native_times,save


def reference_profile(ds):
    """Reuse the existing MODIS LAI QC definition as a product-format contract."""
    if ds.attrs.get('product')!='MCD15A3H':return None
    names={'Lai_500m','Lai_500m_raw','FparLai_QC_raw','FparExtra_QC_raw'}
    if not names<=set(ds):raise ValueError('已识别LAI产品但缺原始数值或QA证据')
    a=ds.Lai_500m
    if a.attrs.get('units')!='1' or float(a.attrs.get('source_scale_factor',np.nan))!=.1:
        raise ValueError('LAI单位/尺度解码未确认')
    raw=ds.Lai_500m_raw.values
    valid_raw=(raw>=0)&(raw<=100)
    if not np.allclose(a.values[valid_raw],raw[valid_raw]*.1,rtol=1e-6,atol=1e-6):raise ValueError('LAI解码值与原始尺度不一致')
    q=np.asarray(ds.FparLai_QC_raw,dtype='uint8');extra=np.asarray(ds.FparExtra_QC_raw,dtype='uint8')
    valid=(valid_raw&np.isfinite(a)&((q&1)==0)&((q&4)==0)&(((q>>3)&3)==0)&(((q>>5)&7)<=1)&(extra!=255))
    start=pd.Timestamp(ds.attrs['time_coverage_start']);end=pd.Timestamp(ds.attrs['time_coverage_end'])
    if start.tzinfo is None or end.tzinfo is None or start>=end:raise ValueError('参考合成区间时区或先后关系未知')
    return dict(field='Lai_500m',quantity='leaf_area_index',unit='1',valid=np.asarray(valid),
        start=start.tz_convert('UTC').tz_localize(None),end=end.tz_convert('UTC').tz_localize(None),
        policy='interval composite as static reference; retain native hourly anomalies; not hourly observations',
        quality='existing MODIS LAI QC definition; invalid pixels not filled; passed-pixel coverage retained')


def fusion_inputs(sat,profile,met,grid,variable,source_id,folder):
    from pyproj import CRS,Transformer
    from rasterio.transform import from_origin
    from rasterio.warp import reproject,Resampling
    from scipy.spatial import Delaunay
    from existing_air_converters.raster_support import aggregate_complete_blocks,remap_valid_fraction,covered_cells
    a=met[variable];nt,nz,nr,nc=a.shape
    if a.dims!=('TSTEP','LAY','ROW','COL') or nz!=1:raise ValueError('参考场更新只适用有明确单层网格支持的输入端口')
    times=native_times(met)
    if times.has_duplicates or not times.is_monotonic_increasing or times[0]<profile['start'] or times[-1]>=profile['end']:
        raise ValueError('原生时刻不被参考合成区间完整覆盖，不扩展卫星时间支持')
    h=met.attrs;radius=re.search(r'EARTH RADIUS ASSUMED IN MCIP:\s*([0-9.]+)\s*m',str(h.get('FILEDESC','')))
    if int(h.get('GDTYP',-1))!=2 or not radius:raise ValueError('需要实际MCIP LCC参数及地球半径，不猜默认投影')
    crs=CRS.from_proj4(f'+proj=lcc +lat_1={h["P_ALP"]} +lat_2={h["P_BET"]} +lat_0={h["YCENT"]} +lon_0={h["P_GAM"]} +R={radius[1]} +units=m')
    xx,yy=np.meshgrid(h['XORIG']+(np.arange(nc)+.5)*h['XCELL'],h['YORIG']+(np.arange(nr)+.5)*h['YCELL'])
    lon,lat=Transformer.from_crs(crs,4326,always_xy=True).transform(xx,yy)
    error=max(float(np.max(abs(np.asarray(grid.LAT).reshape(nr,nc)-lat))),float(np.max(abs(np.asarray(grid.LON).reshape(nr,nc)-lon))))
    if error>1e-4:raise ValueError('MCIP网格声明与绑定GRID经纬度不一致')
    if any(met.attrs.get(k)!=grid.attrs.get(k) for k in ('NCOLS','NROWS','XORIG','YORIG','XCELL','YCELL')):
        raise ValueError('绑定的GRID与MET网格元数据冲突')
    dx=float(sat.x[1]-sat.x[0]);dy=float(sat.y[0]-sat.y[1]);valid=profile['valid'];field=sat[profile['field']].values
    if dx<=0 or dy<=0 or not np.allclose(np.diff(sat.x),dx) or not np.allclose(-np.diff(sat.y),dy):raise ValueError('参考栅格不是已支持的规则北向网格')
    src_crs=CRS.from_user_input(sat.attrs['crs'])
    if not src_crs.is_projected or any(abs(x.unit_conversion_factor-1)>1e-9 for x in src_crs.axis_info):raise ValueError('参考栅格聚合要求实际米制投影')
    transform=from_origin(float(sat.x[0])-dx/2,float(sat.y[0])+dy/2,dx,dy)
    target_transform=from_origin(h['XORIG'],h['YORIG']+nr*h['YCELL'],h['XCELL'],h['YCELL'])
    kwargs=dict(src_transform=transform,src_crs=src_crs,dst_transform=target_transform,dst_crs=crs)
    destination=np.full((nr,nc),np.nan)
    reproject(np.where(valid,field,np.nan),destination,src_nodata=np.nan,dst_nodata=np.nan,resampling=Resampling.average,**kwargs)
    destination=np.flipud(destination);fraction=np.flipud(remap_valid_fraction(valid,(nr,nc),kwargs))
    bounds=(float(sat.x.min())-dx/2,float(sat.y.min())-dy/2,float(sat.x.max())+dx/2,float(sat.y.max())+dy/2)
    inside=np.flipud(covered_cells(target_transform,(nr,nc),crs,bounds,src_crs,np.flipud(fraction>0)))
    eligible=(fraction>0)&np.isfinite(destination)&inside
    if 'VEG' in met:eligible &= met.VEG.values[0,0]>0
    aggregate,good,block_transform,scale=aggregate_complete_blocks(field,valid,transform,max(h['XCELL'],h['YCELL']),require_all=False)
    sr,sc=np.where(good)
    if len(sr)<12:raise ValueError('真实参考尺度单元不足12，不补造数值')
    sx,sy=block_transform*(sc+.5,sr+.5);slon,slat=Transformer.from_crs(src_crs,4326,always_xy=True).transform(sx,sy)
    eligible &= Delaunay(np.column_stack([slon,slat])).find_simplex(np.column_stack([lon.ravel(),lat.ravel()])).reshape(nr,nc)>=0
    rows,cols=np.where(eligible)
    if len(rows)<12:raise ValueError('质量、覆盖及有效域检查后的目标不足12格')
    bx,by=Transformer.from_crs(src_crs,crs,always_xy=True).transform(sx,sy)
    br,bc=np.where((xx>=min(bx)-h['XCELL'])&(xx<=max(bx)+h['XCELL'])&(yy>=min(by)-h['YCELL'])&(yy<=max(by)+h['YCELL']))
    if Delaunay(np.column_stack([lon[br,bc],lat[br,bc]])).find_simplex(np.column_stack([slon,slat])).min()<0:raise ValueError('背景不覆盖参考源支持域，拒绝外推')
    from ecodata.engines.units import UnitEngine
    background=UnitEngine().convert(a,profile['unit']).values
    attrs=dict(units=profile['unit'],physical_variable=profile['quantity'],physical_definition='reference comparability assumption; vegetation consistency not independently verified',
        spatial_semantics='grid_estimate',temporal_semantics='instantaneous',data_kind='continuous',estimate_kind=profile['policy'])
    datasets=[]
    for role,values,yloc,xloc in [('primary',aggregate[sr,sc],slat,slon),('background',background[0,0,br,bc],lat[br,bc],lon[br,bc])]:
        d=xr.Dataset({variable:(('time','location'),values[None,:],attrs)},coords=dict(time=[times[0]],lat=('location',yloc),lon=('location',xloc)),
            attrs=dict(crs='EPSG:4326',source_id=role+':'+source_id,original_temporal_support=str([profile['start'],profile['end']]) if role=='primary' else 'native timestamp',application_assumption=profile['policy']))
        datasets.append((role,d))
    datasets.append(('target',xr.Dataset(coords=dict(time=[times[0]],lat=('location',lat[rows,cols]),lon=('location',lon[rows,cols])),attrs=dict(crs='EPSG:4326'))))
    for _,d in datasets:d.lat.attrs['units']='degrees_north';d.lon.attrs['units']='degrees_east'
    save(folder/'reference_support.json',dict(policy=profile['policy'],source_interval=[str(profile['start']),str(profile['end'])],
        native_times=[str(t) for t in times],coordinate_error_degrees=error,source_QA_passed=int(valid.sum()),source_pixels=valid.size,
        source_scale=scale,target_cells=len(rows),background_cells=len(br),effective_coverage=fraction[rows,cols].tolist(),
        accuracy_assessed=False,partial_coverage='means of passed pixels only; actual coverage retained, no imputation'))
    return datasets,rows,cols,background,times


def prepare_bound_reference(uploads,binding,folder,progress=None):
    from .engineering.native import extract_native_field
    from .component_service import prepare_components
    from ecodata.interface.engineering_upload import LocalUpload
    from ecodata.algorithms.platform_search import SearchInput,process_current_input
    folder=Path(folder);stage=folder/'native_reference'
    with ExitStack() as stack:
        files=ProjectFiles(uploads,stack);sources=[]
        for name in sorted(files.files):
            if not files.prefix(name,8).startswith((b'CDF',b'\x89HDF')):continue
            with files.files[name][0]() as stream,xr.open_dataset(stream) as ds:
                if ds.attrs.get('product')=='MCD15A3H':sources.append((name,ds.load(),files.identity(name)))
        if not sources:return None
        sources=list({r[2]['sha256']:r for r in sources}.values())
        if len(sources)!=1:raise ValueError('多个参考产品尚无明确拼接/独立来源契约，不猜选')
        pairs=[r for r in binding['bindings'] if r['parameter']=='MET_CRO_2D']
        if len(pairs)!=1:raise ValueError('当前单次参考更新需要唯一气象时间端口，未静默截取多日工程')
        metrow=pairs[0];gridrows=[r for r in binding['bindings'] if r['parameter']=='GRID_CRO_2D' and r['day']==metrow['day']]
        if len(gridrows)!=1:raise ValueError('GRID端口没有同日唯一绑定')
        with files.files[metrow['target']][0]() as stream,xr.open_dataset(stream,decode_times=False) as ds:met=ds.load()
        with files.files[gridrows[0]['target']][0]() as stream,xr.open_dataset(stream,decode_times=False) as ds:grid=ds.load()
        from datetime import datetime,timedelta
        parameters=binding['parameters']
        clock=parameters.get('CTM_STTIME',{})
        if not metrow['day'] or clock.get('unknown') or not clock.get('text','').isdigit():
            raise ValueError('控制文件运行起点仍为UNKNOWN，不猜气象时刻')
        t=int(clock['text'])
        start=datetime.strptime(metrow['day'],'%Y%j')+timedelta(hours=t//10000,minutes=t//100%100,seconds=t%100)
        if native_times(met)[0]!=start:raise ValueError('控制运行起点与绑定气象TFLAG不一致')
        name,sat,source=sources[0];profile=reference_profile(sat)
        variables=[v for v in met.data_vars if identity(v,met[v].attrs)[0]==profile['quantity']]
        if len(variables)!=1:raise ValueError('参考物理变量未唯一匹配原生字段')
        variable=variables[0]
    stage.mkdir()
    if progress:progress('绑定实际MCIP/GRID及卫星参考；复用现有QA、尺度适配与公共数值融合')
    data,rows,cols,background,times=fusion_inputs(sat,profile,met,grid,variable,source['sha256'],stage)
    inputs=[]
    for role,ds in data:
        p=stage/(role+'.nc');ds.to_netcdf(p,engine='h5netcdf');inputs.append(SearchInput(p.name,role,p.read_bytes()))
    request=dict(task='multisource',processing_goal='spatial_fusion',variable=variable,unit=profile['unit'],semantic_type='continuous',
        physical_variable=profile['quantity'],temporal_semantics='instantaneous',nonnegative=True,allow_training=True,
        mutation_control='random',soft_guidance=False,observation_crs='EPSG:4326')
    result=process_current_input(inputs,request,stage/'search',progress=progress)
    if result.get('status')!='PASS' or not result.get('product'):raise ValueError('公共引擎没有返回合格参考场产品')
    with xr.open_dataset(result['product']) as d:values=d[variable].values[0]
    increment=values-background[0,0,rows,cols]
    updated=background.copy();updated[:,0,rows,cols]+=increment[None,:]
    if not np.isfinite(updated).all() or (updated<0).any() or (updated>10).any():raise ValueError('参考增量导致非法值，未通过截断凑成功')
    product=extract_native_field(uploads,binding['model_id'],metrow['target'],variable)
    from ecodata.engines.units import UnitEngine
    converted=UnitEngine().affine(profile['unit'],met[variable].attrs['units'])
    product[variable].values[:]=updated*converted[0]+converted[1]
    product.attrs['native_processing_policy']=profile['policy']
    p=stage/'processed.nc';product.to_netcdf(p)
    task=dict(type='model_component',operation='update_project',input=p.name,configuration={})
    output=stage/'writeback';output.mkdir()
    prepared=prepare_components([*uploads,LocalUpload(p.name,p)],binding['model_id'],output,task=task,progress=progress)
    save(stage/'processing.json',dict(bound_resource=metrow,reference_source=source,policy=profile['policy'],
        result=result,static_increment_cells=len(rows),native_times=len(times),accuracy_assessed=False,native_solver_executed=False))
    # Keep execution/support evidence inside the real native deliverable.
    import zipfile
    with zipfile.ZipFile(prepared['archive'],'a',zipfile.ZIP_DEFLATED) as archive:
        for item in [stage/'processing.json',stage/'reference_support.json',folder/'project_binding.json']:
            if item.exists():archive.write(item,'_ecodata/reference/'+item.name)
    return prepared
