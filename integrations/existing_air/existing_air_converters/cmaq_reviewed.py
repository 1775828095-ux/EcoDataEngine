"""Reviewed engineering binding + unchanged public fusion + native LAI patch.

Case settings are an explicit internal configuration authorized by the user.
No uploaded scripts execute and no old result replaces current processed data.
"""
from pathlib import Path, PurePosixPath
import hashlib, io, json, shutil, zipfile
import numpy as np
import xarray as xr

ROUTE='cmaq_reviewed_lai_v1'

def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf8')

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def configuration():
    return json.loads(Path(__file__).with_name('cmaq_reviewed_project.json').read_text(encoding='utf8'))

def remap_quality(valid,shape,kwargs):
    from .raster_support import remap_quality as common
    return common(valid,shape,kwargs)

def probe(prefixes):
    cfg=configuration(); expected=cfg['reviewed_control_prefixes']
    scripts=[(n,b) for n,b in prefixes if n.lower().endswith('.csh')]
    if scripts and all(expected.get(n)==hashlib.sha256(b).hexdigest() for n,b in scripts):
        return dict(model_id='cmaq',issues=[],controls=[n for n,b in scripts],
            scope='人工转写的工程控制头匹配；后续仍须核对整个输入ZIP哈希',
            native_model_executed=False,source_modified=False)
    return None

def contract(manifest):
    if manifest.get('preparation',{}).get('route')!=ROUTE:return None
    cfg=configuration();dims=cfg['native_shape']
    return dict(schema='ecodata.converter.requirements.v1',model_id='cmaq',
        role='meteorology_vegetation',scope='Reviewed CMAQ5.5 CRACMM2/STAGE LAI input update; accuracy not evaluated',
        fields=[dict(name='LAI',unit='1',required=True,data_type='continuous')],
        coordinates=dict(time=dict(datetime=True,interval_seconds=cfg['interval_seconds'],
                start=cfg['start'],end=cfg['end'],dimension=True),
            row=dict(size=dims[2],start=0,step=1,dimension=True),col=dict(size=dims[3],start=0,step=1,dimension=True)),
        dependencies=[],metadata=dict(time_basis='UTC'),reference_fields=[],required_references=[],
        deferred_checks=['写回原生METCRO2D，保留原时间标记/变量/头', '独立PseudoNetCDF回读', '不等于原生模型积分'])

def prepare(uploads,folder,progress=None):
    if len(uploads)!=1 or not str(uploads[0].name).lower().endswith('.zip'):return None
    cfg=configuration(); upload=uploads[0]; folder=Path(folder)
    path=getattr(upload,'path',None)
    if path is None:
        path=folder/'uploaded_engineering.zip'
        with Path(path).open('wb') as f:f.write(upload.getbuffer() if hasattr(upload,'getbuffer') else upload.getvalue())
    path=Path(path)
    # Do not claim a case match based on a filename or a few field aliases.
    if path.stat().st_size!=cfg['archive_bytes']:return None
    source_sha=digest(path)
    if source_sha!=cfg['archive_sha256']:return None
    if progress:progress('核对人工转写的CMAQ工程与原始ZIP，读取本次卫星和气象数据')
    inputs=folder/'reviewed_inputs';inputs.mkdir()
    records=[]
    with zipfile.ZipFile(path) as archive:
        for key in ('meteorology','grid','satellite'):
            member=cfg[key]; raw=archive.read(member)
            target=inputs/(key+'.nc');target.write_bytes(raw)
            records.append(dict(role=key,member=member,path=target.relative_to(folder).as_posix(),
                sha256=digest(target),bytes=len(raw)))
    sat=xr.open_dataset(inputs/'satellite.nc',engine='h5netcdf').load()
    met=xr.open_dataset(inputs/'meteorology.nc').load()
    grid=xr.open_dataset(inputs/'grid.nc').load()
    from datetime import datetime
    flags=met.TFLAG.values
    if not np.array_equal(flags,np.broadcast_to(flags[:,:1,:],flags.shape)):raise ValueError('各变量的原生时标不同')
    times=np.array([np.datetime64(datetime.strptime(f'{int(day):07d}{int(clock):06d}', '%Y%j%H%M%S'),'ns') for day,clock in flags[:,0,:]])
    if times[0]!=np.datetime64('2018-07-01T00:00:00') or not np.all(np.diff(times)==np.timedelta64(1,'h')):raise ValueError('原生时标与已转写工程设置不符')
    if met.LAI.dims!=('TSTEP','LAY','ROW','COL') or met.LAI.shape!=(25,1,105,100):raise ValueError('人工工程配置与实际LAI维度不符')
    if met.LAI.attrs['units'].strip()!='m2 m-2':raise ValueError('模型LAI单位变化，拒绝自动覆盖')
    if sat.Lai_500m.attrs.get('units')!='1':raise ValueError('卫星LAI单位不符')
    if sat.attrs.get('product')!='MCD15A3H':raise ValueError('卫星产品不同，质量位配置不能沿用')
    if not sat.attrs['time_coverage_start'].startswith('2018-06-30') or not sat.attrs['time_coverage_end'].startswith('2018-07-04'):
        raise ValueError('静态参考授权时段与本次卫星数据不符')
    from pyproj import CRS,Transformer
    from rasterio.warp import reproject,Resampling
    from rasterio.transform import from_origin
    h=met.attrs
    # Radius is explicitly recorded in this MCIP file; no library default is used.
    if 'EARTH RADIUS ASSUMED IN MCIP:  6370000.000 m' not in h['FILEDESC']:raise ValueError('没有已核对的MCIP地球半径证据')
    crs=CRS.from_proj4(f'+proj=lcc +lat_1={h["P_ALP"]} +lat_2={h["P_BET"]} +lat_0={h["YCENT"]} +lon_0={h["P_GAM"]} +R=6370000 +units=m')
    x=h['XORIG']+(np.arange(100)+.5)*h['XCELL'];y=h['YORIG']+(np.arange(105)+.5)*h['YCELL']
    xx,yy=np.meshgrid(x,y);lon,lat=Transformer.from_crs(crs,4326,always_xy=True).transform(xx,yy)
    native_lat=np.asarray(grid.LAT.values).reshape(105,100);native_lon=np.asarray(grid.LON.values).reshape(105,100)
    coordinate_error=max(float(np.max(abs(lat-native_lat))),float(np.max(abs(lon-native_lon))))
    if coordinate_error>1e-4:raise ValueError('推导的MCIP网格与GRIDCRO2D实际经纬度不一致')
    q=np.asarray(sat.FparLai_QC_raw.values,dtype='uint8')
    extra=np.asarray(sat.FparExtra_QC_raw.values,dtype='uint8')
    valid=(np.isfinite(sat.Lai_500m.values)&(sat.Lai_500m.values>=0)&(sat.Lai_500m.values<=10)
        &((q&1)==0)&((q&4)==0)&(((q>>3)&3)==0)&(((q>>5)&7)<=1)&(extra!=255))
    dx=float(sat.x[1]-sat.x[0]);dy=float(sat.y[0]-sat.y[1])
    if dx<=0 or dy<=0 or not np.allclose(np.diff(sat.x),dx) or not np.allclose(-np.diff(sat.y),dy):raise ValueError('卫星栅格并非已核对的规则网格')
    transform=from_origin(float(sat.x[0])-dx/2,float(sat.y[0])+dy/2,dx,dy)
    dest_transform=from_origin(h['XORIG'],h['YORIG']+105*h['YCELL'],h['XCELL'],h['YCELL'])
    destination=np.full((105,100),np.nan,dtype='float64')
    kwargs=dict(src_transform=transform,src_crs=sat.attrs['crs'],dst_transform=dest_transform,dst_crs=crs.to_wkt())
    reproject(np.where(valid,sat.Lai_500m.values,np.nan),destination,src_nodata=np.nan,dst_nodata=np.nan,resampling=Resampling.average,**kwargs)
    # Preserve strict QC diagnostics AND effective coverage. Never relabel a
    # failed pixel as good. Partial means remain explicitly partial estimates.
    quality=remap_quality(valid,(105,100),kwargs)
    destination=np.flipud(destination);quality=np.flipud(quality)
    from .raster_support import covered_cells,remap_valid_fraction
    fraction=np.flipud(remap_valid_fraction(valid,(105,100),kwargs))
    source_bounds=(float(sat.x.min())-dx/2,float(sat.y.min())-dy/2,float(sat.x.max())+dx/2,float(sat.y.max())+dy/2)
    inside=np.flipud(covered_cells(dest_transform,(105,100),crs.to_wkt(),source_bounds,sat.attrs['crs'],np.flipud(fraction>0)))
    eligible=(fraction>0)&np.isfinite(destination)&inside&(met.VEG.values[0,0]>0)
    # Retain source geometry and actual roles. Native MCIP is a model background,
    # not a set of hard observations at every destination. Aggregating the full
    # source at the engineering scale also avoids quadratic all-pixel searches.
    from .raster_support import aggregate_complete_blocks
    from scipy.spatial import Delaunay
    sat_crs=CRS.from_user_input(sat.attrs['crs'])
    if not sat_crs.is_projected or any(abs(a.unit_conversion_factor-1)>1e-9 for a in sat_crs.axis_info):
        raise ValueError('源栅格不是米制投影，不能按工程米制尺度自动聚合')
    aggregate,good_blocks,block_transform,scale_receipt=aggregate_complete_blocks(
        sat.Lai_500m.values,valid,transform,max(float(h['XCELL']),float(h['YCELL'])),require_all=False)
    sr,sc=np.where(good_blocks)
    if len(sr)<12:raise ValueError('具有真实有效像元的完整源尺度单元不足12个')
    sx,sy=block_transform*(sc+.5,sr+.5)
    slon,slat=Transformer.from_crs(sat_crs,4326,always_xy=True).transform(sx,sy)
    # Only supported true MCIP targets enter the engine; no arbitrary query shift.
    hull=Delaunay(np.column_stack([slon,slat]))
    eligible &= hull.find_simplex(np.column_stack([lon.ravel(),lat.ravel()])).reshape(105,100)>=0
    rows,cols=np.where(eligible)
    if len(rows)<12:raise ValueError('严格质量、源覆盖及插值支持检查后，匹配网格不足12格，未降低门槛')
    bx,by=Transformer.from_crs(sat_crs,crs,always_xy=True).transform(sx,sy)
    background_mask=(xx>=min(bx)-h['XCELL'])&(xx<=max(bx)+h['XCELL'])&(yy>=min(by)-h['YCELL'])&(yy<=max(by)+h['YCELL'])
    br,bc=np.where(background_mask)
    if Delaunay(np.column_stack([lon[br,bc],lat[br,bc]])).find_simplex(np.column_stack([slon,slat])).min()<0:
        raise ValueError('模型背景没有覆盖源尺度单元，拒绝外推')
    # Both are application estimates under the approved static-reference policy;
    # original temporal support is retained, never hourly satellite observations.
    attrs=dict(units='1',physical_variable='leaf_area_index',physical_definition='leaf area per ground area; chain-test comparability assumption, vegetation consistency not independently verified',
        spatial_semantics='grid_estimate',temporal_semantics='instantaneous',data_kind='continuous',
        estimate_kind='application snapshot estimate under approved static-reference assumption; not an instantaneous satellite observation')
    datasets=[]
    for role,values,resource_lat,resource_lon in [
        ('primary',aggregate[sr,sc],slat,slon),
        ('background',met.LAI.values[0,0,br,bc],lat[br,bc],lon[br,bc])]:
        d=xr.Dataset({'LAI':(('time','location'),values[None,:],attrs)},
            coords=dict(time=[times[0]],lat=('location',resource_lat),lon=('location',resource_lon)),
            attrs=dict(crs='EPSG:4326',source_id=role+':'+source_sha,
                original_temporal_support=sat.attrs['time_support'] if role=='primary' else 'MCIP first timestamp',
                application_assumption=cfg['temporal_policy']))
        d.lat.attrs['units']='degrees_north';d.lon.attrs['units']='degrees_east'
        datasets.append((role,d))
    target=xr.Dataset(coords=dict(time=[times[0]],lat=('location',lat[rows,cols]),lon=('location',lon[rows,cols])),attrs=dict(crs='EPSG:4326'))
    target.lat.attrs['units']='degrees_north';target.lon.attrs['units']='degrees_east'
    datasets.append(('target',target))
    decision=dict(route=ROUTE,source_archive=str(path),source_archive_sha256=source_sha,
        sources=records,approved_assumption=cfg['temporal_policy'],approval_evidence=cfg['approval_evidence'],
        mcip_grid_coordinate_error_degrees=coordinate_error,
        satellite_collection=sat.attrs.get('collection','UNKNOWN'),
        quality_policy='NASA MODIS C6/C6.1 QC: MODLAND good; live detector; clear cloud bits; main algorithm SCF 0/1; retain original QC arrays',
        quality_reference='https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MCD15A3H_c61.fs',
        satellite_valid_pixels=int(valid.sum()),satellite_total_pixels=int(valid.size),updated_native_cells=int(eligible.sum()),
        resource_roles=dict(primary='Satellite retrieval estimates aggregated on source-aligned grid',background='MCIP model estimates',target='Actual native MCIP cell centres'),
        scale_adapter=scale_receipt,satellite_conditioning_locations=len(sr),background_locations=len(br),
        native_target_effective_coverage_fraction=fraction[rows,cols].tolist(),
        native_targets_all_contributors_good=int(np.sum(quality[rows,cols]==1)),
        partial_coverage_policy='Use passed-pixel means with disclosed coverage, not full-cell measurements; no invalid-pixel filling or accuracy claim',
        spatial_method='QC-passed pixel means on source-aligned blocks at nominal engineering scale -> unchanged automatic spatial search -> actual MCIP centres. GDAL target coverage and strict-min mask are diagnostics; partial effective coverage is explicit. Not certified geodesic conservation.',
        scientific_limitations=['No independent LAI truth','Collection not returned by source service; not inferred',
          'Vegetation-fraction/landcover consistency not independently validated','Four-day composite is a static reference, not hourly observations',
          'Aggregated source cells may have partial QC coverage; representativeness is unknown',
          'Self holdout metrics do not establish predictive accuracy'],
        original_engineering_read_only=True,original_scripts_executed=False)
    save(folder/'reviewed_binding.json',decision)
    from ecodata.algorithms.platform_search import SearchInput,process_current_input
    candidates=[]
    for role,d in datasets:
        p=folder/f'LAI_{role}.nc';d.to_netcdf(p,engine='h5netcdf');candidates.append(SearchInput(p.name,role,p.read_bytes()))
    request=dict(task='multisource',processing_goal='spatial_fusion',variable='LAI',unit='1',
        semantic_type='continuous',physical_variable='leaf_area_index',temporal_semantics='instantaneous',
        nonnegative=True,allow_training=True,mutation_control='random',soft_guidance=False,observation_crs='EPSG:4326')
    if progress:progress(f'平台原融合引擎：{len(sr)}个卫星源单元、{len(br)}个模型背景单元，处理{len(rows)}个真实目标网格')
    result=process_current_input(candidates,request,folder/'fusion_search',progress=progress)
    if not result.get('product'):raise ValueError('平台融合引擎没有返回合格产物')
    fused=xr.open_dataset(result['product'],engine='h5netcdf').load()
    values=fused.LAI.values[0]
    increment=values-met.LAI.values[0,0,rows,cols]
    graph=result.get('selected_graph',{}).get('records',{})
    satellite_explicit=any(n.get('operation')=='spatial.source' and n.get('parameters',{}).get('index')==0 for n in graph.values())
    # In self-reference mode spatial.observations conditions on the PRIMARY
    # retrieved grid, never on independent station truth. Include that real edge.
    satellite_conditioning=bool(result.get('self_validation_protocol',{}).get('self_gridded_reference')) and any(
        n.get('operation')=='spatial.observations' for n in graph.values())
    satellite_in_graph=satellite_explicit or satellite_conditioning
    background_in_graph=any(n.get('operation')=='spatial.source' and n.get('parameters',{}).get('index')==1 for n in graph.values())
    effect=dict(satellite_in_selected_graph=satellite_in_graph,
        model_background_in_selected_graph=background_in_graph,
        satellite_conditioning_used=satellite_conditioning,
        spatial_search_selected_multiple_sources=bool(satellite_in_graph and background_in_graph),
        final_update_uses_native_hourly_variation=True,
        changed_snapshot_cells=int(np.sum(abs(increment)>1e-6)),
        max_absolute_snapshot_increment=float(np.max(abs(increment))),
        qualification='Automatic spatial selection may prefer one source. Static-increment application retains the original MCIP hourly anomalies; no forced blend, graph substitution or independent-truth claim.')
    save(folder/'numerical_effect.json',effect)
    updated=met.LAI.values[:,0].astype('float64').copy()
    updated[:,rows,cols]+=increment[None,:]
    if not np.isfinite(updated).all() or (updated<0).any() or (updated>10).any():raise ValueError('静态增量写回会产生不合理LAI；未截断或补值凑成功')
    product=xr.Dataset({'LAI':(('time','row','col'),updated,dict(units='1',standard_name='leaf_area_index',
        temporal_semantics='application_static_increment_retaining_original_MCiP_time_variation'))},
        coords=dict(time=times,row=np.arange(105),col=np.arange(100),lat=(('row','col'),lat),lon=(('row','col'),lon)),
        attrs=dict(crs='EPSG:4326',time_basis='UTC',approved_static_reference=cfg['temporal_policy']))
    product.lat.attrs['units']='degrees_north';product.lon.attrs['units']='degrees_east'
    p=folder/'LAI_product.nc';product.to_netcdf(p,engine='h5netcdf')
    np.savez(folder/'update_cells.npz',rows=rows,cols=cols)
    receipt=dict(route=ROUTE,reviewed_binding=decision,original_data_modified=False,
        fusion_decision=dict(status='PLATFORM_ENGINE_RETURNED',algorithm_search_executed=bool(result.get('counts',{}).get('evaluated',0)),
            selected_graph=result.get('selected_graph'),numerical_effect=effect,accuracy_improvement_assessed=False),
        tasks=[dict(role='meteorology_vegetation',field='LAI',unit='1',result=result)],
        full_model_ready=False,native_model_executed=False,
        scope='Authorized static vegetation-reference chain test; native format/readback separate from full CMAQ run')
    save(folder/'preparation.json',receipt)
    if progress:progress(('空间搜索选中多来源方案' if background_in_graph else '空间搜索选中卫星单源估计，未强制混入模型背景')+'；保留模型原有逐小时变化，写回原生METCRO2D并独立回读')
    from ecodata.models.conversion_bridge import finish_preparation
    return finish_preparation(folder,'cmaq',[p],receipt)

def convert(manifest_path,output_dir):
    from netCDF4 import Dataset
    from PseudoNetCDF import pncopen
    manifest_path=Path(manifest_path);base=manifest_path.parent;out=Path(output_dir);out.mkdir(exist_ok=True)
    manifest=json.loads(manifest_path.read_text(encoding='utf8'));receipt=manifest['preparation']
    cfg=configuration();item=manifest['products'][0];product=(base/item['path']).resolve()
    if not product.is_relative_to(base.resolve()) or digest(product)!=item['sha256']:raise ValueError('本次融合产品路径或哈希不符')
    ds=xr.open_dataset(product,engine='h5netcdf').load();source=base/'reviewed_inputs/meteorology.nc'
    if digest(source)!=next(r['sha256'] for r in receipt['reviewed_binding']['sources'] if r['role']=='meteorology'):
        raise ValueError('原生METCRO2D副本发生变化')
    if ds.LAI.dims!=('time','row','col') or ds.LAI.attrs.get('units')!='1':raise ValueError('产品变量维度顺序或单位不符')
    from datetime import datetime
    with Dataset(source) as native:
        flags=native['TFLAG'][:,0,:]
        expected=np.array([np.datetime64(datetime.strptime(f'{int(day):07d}{int(clock):06d}', '%Y%j%H%M%S'),'ns') for day,clock in flags])
    if not np.array_equal(ds.time.values,expected) or not np.array_equal(ds.row.values,np.arange(105)) or not np.array_equal(ds.col.values,np.arange(100)):
        raise ValueError('产品时标或网格索引与原生工程不一致')
    target=out/'METCRO2D_12NE3_20180701.nc'
    cells=np.load(base/'update_cells.npz');mask=np.zeros((105,100),bool);mask[cells['rows'],cells['cols']]=True
    from .native_netcdf_update import update_variable
    shared_update=update_variable(source,target,'LAI',ds.LAI.values[:,None,:,:],allowed_mask=mask[None,None,:,:])
    checks=[]
    with Dataset(source) as before,Dataset(target) as after:
        if before.ncattrs()!=after.ncattrs() or any(not np.array_equal(before.getncattr(k),after.getncattr(k)) for k in before.ncattrs()):raise ValueError('原生全局头发生未授权变化')
        for name in before.variables:
            a,b=before[name],after[name]
            if a.dimensions!=b.dimensions or a.ncattrs()!=b.ncattrs() or any(not np.array_equal(a.getncattr(k),b.getncattr(k)) for k in a.ncattrs()):raise ValueError('变量结构/属性被改变：'+name)
            if name!='LAI':
                if not np.array_equal(a[:],b[:],equal_nan=True):raise ValueError('非目标变量被改变：'+name)
            elif not np.array_equal(a[:,0,:,:][:,~mask],b[:,0,:,:][:,~mask]):raise ValueError('卫星范围外LAI被改变')
            checks.append(dict(variable=name,preserved=name!='LAI'))
        native_values=after['LAI'][:,0,:,:].copy()
        if not np.allclose(native_values,ds.LAI.values,rtol=1e-7,atol=1e-6):raise ValueError('原生数值与本次融合产物不符')
        changed=int(np.sum(before['LAI'][:]!=after['LAI'][:]))
        anomaly_error=float(np.max(abs((native_values-native_values[:1])-(before['LAI'][:,0,:,:]-before['LAI'][0:1,0,:,:]))))
        if anomaly_error>1e-6:raise ValueError('原生小时变化未按静态增量假设保留')
    reader=pncopen(str(target),format='ioapi')
    try:
        if not np.array_equal(reader.variables['LAI'][:][:,0],native_values):raise ValueError('独立IOAPI读取器LAI不符')
    finally:reader.close()
    source_archive=Path(receipt['reviewed_binding']['source_archive'])
    if digest(source_archive)!=cfg['archive_sha256']:raise ValueError('原始工程ZIP已改变')
    project=base/'connected_project';project.mkdir(exist_ok=False)
    copied=[]
    # Materialise the complete original engineering payload in THIS job. All
    # decompression/CRC and the replacement occur inside the platform plugin.
    with zipfile.ZipFile(source_archive) as archive:
        for member in archive.infolist():
            if member.is_dir():continue
            relative=PurePosixPath(member.filename.replace('\\','/'))
            if relative.is_absolute() or '..' in relative.parts or ':' in str(relative) or (member.external_attr>>16)&0o170000==0o120000:
                raise ValueError('工程成员路径不安全')
            dest=(project/str(relative)).resolve()
            if not dest.is_relative_to(project.resolve()):raise ValueError('工程成员越界')
            dest.parent.mkdir(parents=True,exist_ok=True)
            hasher=hashlib.sha256()
            with archive.open(member) as src,dest.open('xb') as dst:
                while chunk:=src.read(4*1024*1024):dst.write(chunk);hasher.update(chunk)
            copied.append(dict(member=str(relative),bytes=member.file_size,original_sha256=hasher.hexdigest()))
    actual_target=project/cfg['meteorology'];shutil.copyfile(target,actual_target)
    if digest(actual_target)!=digest(target):raise ValueError('本次工程内替换文件哈希不符')
    save(out/'connected_project_inventory.json',dict(project=str(project),files=copied,
        replacement_member=cfg['meteorology'],replacement_sha256=digest(actual_target),
        original_zip_crc_verified_during_streamed_reads=True,
        scripts_executed=False,scope='完整原工程资料已在本次目录展开并替换目标输入；未绑定/执行CCTM程序'))
    # The complete source project remains user-owned. Export the changed native
    # input plus an exact-member update map, not a false "complete runnable ZIP".
    application=dict(source_archive_sha256=cfg['archive_sha256'],source_archive=receipt['reviewed_binding']['source_archive'],
        connected_project=str(project),connected_native_input=str(actual_target),
        replacement_member=cfg['meteorology'],replacement_file=target.name,replacement_sha256=digest(target),
        original_scripts_preserved=True,full_project_bundled=False,
        scientific_settings_modified=False,authorized_input_modified='LAI only; recorded static-reference increment',
        download_contents='Updated native input plus exact-member update map and verification; complete engineering folder retained locally')
    save(out/'engineering_update.json',application)
    validation=dict(status='PASS',native_format='CMAQ IOAPI METCRO2D',reader='PseudoNetCDF ioapi',
        generic_native_template_checks=shared_update,
        source_product_sha256=item['sha256'],native_sha256=digest(target),changed_values=changed,
        checks=checks,all_other_fields_preserved=True,outside_update_cells_preserved=True,
        original_TFLAG_preserved=True,native_model_executed=False,full_model_ready=False,
        native_hourly_anomaly_preserved=True,max_hourly_anomaly_error=anomaly_error,
        numerical_effect=receipt['fusion_decision']['numerical_effect'],
        prediction_accuracy_assessed=False,connected_project=str(project),full_source_payload_crc_verified=True,
        scope='Actual fused product -> native input -> complete local engineering copy + independent readback; not full model integration')
    save(out/'validation.json',validation)
    # A small real visualization of the current two fields, never a simulated UI.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with Dataset(source) as native:old=native['LAI'][0,0].copy()
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,a,title in zip(axes,[old,native_values[0],native_values[0]-old],['Original MCIP LAI','Updated LAI','Update (LAI units)']):
        m=ax.imshow(a,origin='lower');ax.set_title(title);fig.colorbar(m,ax=ax,shrink=.7)
    fig.savefig(out/'LAI_compare.png',dpi=150);plt.close(fig)
    return dict(status='PASS',files=[target.name,'engineering_update.json','validation.json','LAI_compare.png','connected_project_inventory.json'],validation=validation)
