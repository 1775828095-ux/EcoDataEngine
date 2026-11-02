"""Explicit raster-to-CALMET geography, inside the platform; no hidden defaults."""
import io,json,re,hashlib
from contextlib import ExitStack
import numpy as np


def from_rasters(payload,config,control):
    import xarray as xr
    from pyproj import CRS
    from rasterio.io import MemoryFile
    from rasterio.merge import merge
    from rasterio.warp import transform_bounds,reproject,Resampling
    from rasterio.transform import from_origin
    from ecodata.interface.wrf_external_project import read_archive
    from ecodata.interface.calpuff7_results import control_values
    from ecodata.models.adapters.calmet_geography import FIELDS,export_geography
    files=read_archive(payload)
    if config.get('schema')!='ecodata.calmet.raw-geography.v1':raise ValueError('原始GIS配置版本不符')
    if config.get('dem_resampling')!='average' or config.get('landcover_resampling')!='mode':raise ValueError('需明确DEM average、土地类别mode；不推测重采样')
    if config.get('elevation_units')!='m' or config.get('vertical_datum') not in {'MSL','EGM2008'}:raise ValueError('需明确米制MSL或EGM2008正高；不转换高程基准')
    if not isinstance(config.get('source'),dict) or not config['source'].get('description'):raise ValueError('缺少原始GIS来源与参数选择说明')
    vals=control_values(control.decode('ascii'))
    def one(k):
        if len(vals.get(k,[]))!=1:raise ValueError('CALMET网格缺少或重复：'+k)
        return vals[k][0]
    if one('DATUM')!='WGS-84' or one('PMAP')!='UTM':raise ValueError('原始GIS入口当前需WGS84 UTM控制，不自动更改投影')
    zone=int(one('IUTMZN'));hemi=one('UTMHEM')
    if not 1<=zone<=60 or hemi not in {'N','S'}:raise ValueError('UTM分区无效')
    crs=CRS.from_epsg((32600 if hemi=='N' else 32700)+zone)
    nx,ny=int(one('NX')),int(one('NY'));step=float(one('DGRIDKM'))*1000
    west,south=float(one('XORIGKM'))*1000,float(one('YORIGKM'))*1000
    if not 2<=nx<=265 or not 2<=ny<=265 or not np.isfinite([west,south,step]).all() or step<=0:raise ValueError('原始GIS目标网格无效')
    bounds=(west,south,west+nx*step,south+ny*step)
    target=from_origin(west,bounds[3],step,step)
    source_records=[]
    def raster(names,method):
        if not isinstance(names,list) or not names or len(names)!=len(set(names)) or any(n not in files for n in names):raise ValueError('原始栅格文件清单缺失或重复')
        with ExitStack() as stack:
            srcs=[]
            for n in names:
                mem=stack.enter_context(MemoryFile(files[n]));src=stack.enter_context(mem.open())
                if src.count!=1 or not src.crs or src.transform.b or src.transform.d:raise ValueError('需有CRS的单波段非旋转栅格：'+n)
                if srcs and (src.crs!=srcs[0].crs or src.res!=srcs[0].res or src.dtypes!=srcs[0].dtypes):raise ValueError('同组原始瓦片需相同投影、分辨率和精度；不隐式重采样拼图')
                srcs.append(src);source_records.append(dict(name=n,sha256=hashlib.sha256(files[n]).hexdigest(),crs=str(src.crs),resolution=list(src.res),bounds=list(src.bounds),nodata=src.nodata))
            bbox=transform_bounds(crs,srcs[0].crs,*bounds,densify_pts=41)
            # Bound memory before materializing only the platform-needed window.
            count=((bbox[2]-bbox[0])/srcs[0].res[0]+4)*((bbox[3]-bbox[1])/srcs[0].res[1]+4)
            if not 1<count<=80_000_000:raise ValueError('原始GIS读取窗口超过受控规模')
            # Anchor crop to the source affine origin, including point-grid
            # half-pixel offsets. Global-zero alignment can add an empty column.
            from math import floor,ceil
            origin=srcs[0].transform;dx,dy=srcs[0].res
            aligned=(origin.c+floor((bbox[0]-origin.c)/dx)*dx,
                     origin.f+floor((bbox[1]-origin.f)/dy)*dy,
                     origin.c+ceil((bbox[2]-origin.c)/dx)*dx,
                     origin.f+ceil((bbox[3]-origin.f)/dy)*dy)
            mosaic,trans=merge(srcs,bounds=aligned,target_aligned_pixels=False,dtype='float64',nodata=np.nan,masked=True)
            a=mosaic[0]
            if np.ma.getmaskarray(a).any() or not np.isfinite(a).all():raise ValueError('目标外包窗口存在原始缺测/覆盖缺口，不插补')
            destination=np.full((ny,nx),np.nan,dtype='float64')
            reproject(np.asarray(a),destination,src_transform=trans,src_crs=srcs[0].crs,dst_transform=target,dst_crs=crs,dst_nodata=np.nan,resampling=method)
        if not np.isfinite(destination).all():raise ValueError('目标网格未完全覆盖，不补值')
        return destination
    elevation=raster(config.get('dem_files'),Resampling.average)
    cover=raster(config.get('landcover_files'),Resampling.mode)
    if not np.equal(cover,np.round(cover)).all():raise ValueError('土地覆盖类别不是整数')
    table_file=config.get('parameter_table_file')
    if table_file not in files:raise ValueError('缺少明确的MAKEGEO参考参数原文件')
    table={}
    for line in files[table_file].decode('ascii').splitlines():
        m=re.search(r'!\s*X\s*=\s*(.*?)!',line)
        if m:
            row=[float(v.strip()) for v in m[1].split(',')]
            if len(row)!=8 or not np.isfinite(row).all() or row[0]!=int(row[0]) or int(row[0]) in table:raise ValueError('参考参数行无效或重复')
            table[int(row[0])]=row[1:]
    cross=config.get('class_crosswalk')
    if not isinstance(cross,dict) or not config.get('crosswalk_description'):raise ValueError('需明确土地分类对照和适用假设')
    used=sorted(int(v) for v in np.unique(cover));lookup={}
    for c in used:
        ref=cross.get(str(c))
        if type(ref)!=int or ref not in table:raise ValueError('土地类别未明确匹配参考参数：'+str(c))
        lookup[c]=table[ref]
    rows=np.array([lookup[int(v)] for v in cover.ravel()])
    physical={'elevation':elevation,'landuse':rows[:,-1].reshape(ny,nx)}
    for i,n in enumerate(('roughness','albedo','bowen','soil_heat_flux','anthropogenic_heat','leaf_area_index')):physical[n]=rows[:,i].reshape(ny,nx)
    x=west+(np.arange(nx)+.5)*step;y=bounds[3]-(np.arange(ny)+.5)*step
    ds=xr.Dataset({n:(('y','x'),a,{'units':FIELDS[n]}) for n,a in physical.items()},coords={'x':('x',x,{'units':'m'}),'y':('y',y,{'units':'m'})},attrs={'crs':str(crs)})
    cfg=dict(target_version='CALMET-5.8.5-GEO.DAT-2.0',coordinate_location='cell_center',elevation_vertical_reference=config['vertical_datum'],source=config['source'],missing_values=[],crs=str(crs),coordinates={'x':'x','y':'y'},coordinate_units={'x':'m','y':'m'},fields={n:dict(variable=n,source_units=u) for n,u in FIELDS.items()},landuse_categories=sorted(int(v) for v in np.unique(physical['landuse'])),water_category_range=config.get('water_category_range'))
    generated,validation=export_geography(ds,cfg)
    report=dict(source_files=source_records,raw_zip_sha256=hashlib.sha256(payload).hexdigest(),parameter_table_sha256=hashlib.sha256(files[table_file]).hexdigest(),target_crs=str(crs),target_bounds_m=bounds,shape_y_x=[ny,nx],input_classes=used,class_crosswalk=cross,crosswalk_description=config['crosswalk_description'],dem_resampling='average',landcover_resampling='mode',vertical_datum=config['vertical_datum'],vertical_transformation='none; global orthometric datum is not certified equal to local mean sea level',parameters='Explicit uploaded reference table, not locally measured or calibrated',source_bytes_modified=False,missing_values_filled=False,validation=validation)
    return dict(generated,**{'geography.nc':ds.to_netcdf(engine='scipy'),'geography_config.json':json.dumps(cfg,ensure_ascii=False,indent=2).encode('utf8'),'raw_geography_report.json':json.dumps(report,ensure_ascii=False,indent=2).encode('utf8')}),report


def attach_raw_geography(project_payload,raster_files,config):
    """Assemble uploads in-platform, discarding the old prepared geography."""
    import zipfile
    from ecodata.interface.calpuff_project import unpack
    data,provenance=unpack(project_payload,'rebuild',raw_geography_inputs=True)
    raw=io.BytesIO()
    if any(not re.fullmatch(r'[A-Za-z0-9_.-]+',n) or '..' in n for n in raster_files):raise ValueError('原始GIS文件需安全单层文件名')
    with zipfile.ZipFile(raw,'w',zipfile.ZIP_DEFLATED) as z:
        for n,b in raster_files.items():z.writestr(n,b)
    # Validate before creating a native job; the same operation executes again there.
    _,report=from_rasters(raw.getvalue(),config,data['CALMET.INP'])
    data.pop('geography.nc',None);data.pop('geography_config.json',None)
    data['geography_raw.zip']=raw.getvalue();data['geography_raw_config.json']=json.dumps(config,ensure_ascii=False,indent=2).encode('utf8')
    provenance['sources'].append(config['source']['description'])
    provenance['defaults'].append('Raw-GIS route uses explicitly uploaded MAKEGEO table/crosswalk; no hidden surface parameters')
    provenance['limitations'].append('Copernicus DSM/land-cover product epoch and explicit crosswalk assumptions are not local terrain surveys or parameter calibration; original point/receptor elevations retained')
    data['provenance.json']=json.dumps(provenance,ensure_ascii=False,indent=2).encode('utf8')
    result=io.BytesIO()
    with zipfile.ZipFile(result,'w',zipfile.ZIP_DEFLATED) as z:
        for n,b in data.items():z.writestr(n,b)
    return result.getvalue(),report
