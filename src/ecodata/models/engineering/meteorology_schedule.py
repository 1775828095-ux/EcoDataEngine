"""Execute actual CAMx meteorology ports over a bound multi-run/grid schedule.

The format driver supplies geometry; collocation, residuals and quality are
shared services. No package, site, date, domain count or numerical winner is fixed.
"""
from contextlib import ExitStack
from pathlib import Path
import hashlib,json,shutil,zipfile
import numpy as np
from pyproj import CRS,Transformer
from ..project_binding import ProjectFiles
from ..native_meteorology import save,native_times,geometry,native_axes,materialize_project,write_native_copy,_scope_observations
from .collocation import process_collocations


def prepare_schedule(uploads,binding,folder,decoded,progress=None):
    from existing_air_converters.native_netcdf_update import _open_native
    folder=Path(folder);stage=folder/'meteorology_schedule';stage.mkdir()
    save(folder/'project_binding.json',binding)
    rows=[r for r in binding['bindings'] if r['parameter']=='met2d_grid' and r.get('status','BOUND')=='BOUND']
    if not rows:raise ValueError('运行计划没有实际二维气象端口')
    files_by_path={};geometry_by_grid={};overlap={};products=[];reports=[]
    with ExitStack() as stack:
        files=ProjectFiles(uploads,stack)
        for i,row in enumerate(rows):
            if row['target'] in files_by_path:continue
            source=stage/f'source_{i}.nc'
            with files.files[row['target']][0]() as src,source.open('wb') as dst:shutil.copyfileobj(src,dst,8*1024**2)
            with source.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
            if digest!=row['sha256']:raise ValueError('气象绑定后内容改变')
            with _open_native(source,'r') as ds:
                if 't2' not in ds.variables or str(ds['t2'].units).strip()!='K' or str(ds['t2'].long_name).strip()!='temperature at 2 m':
                    raise ValueError('二维温度端口物理定义未确认')
                if ds['t2'].dimensions!=('TSTEP','LAY','ROW','COL') or ds['t2'].shape[1]!=1:
                    raise ValueError('不能把多层字段当地面二维温度；需另行明确垂直操作')
                times=native_times(ds);lon,lat=geometry(ds);x,y=native_axes(ds);bg=np.asarray(ds['t2'][:,0,:,:]);a={k:ds.getncattr(k) for k in ds.ncattrs()}
                if not np.isfinite(bg).all():raise ValueError('背景缺失；当前局部残差模式不能补造采样值')
            run_index=row.get('run_index',0)
            run=binding['runs'][run_index] if binding.get('runs') else dict(parameters=binding['parameters']['camx_control'])
            params=run['parameters'];index=row['indexes'][0] if row['indexes'] else 0
            grid_id=str(index);grid_signature={k:a[k] for k in ['NCOLS','NROWS','XORIG','YORIG','XCELL','YCELL','P_ALP','P_BET','XCENT','YCENT']}
            if grid_id in geometry_by_grid and geometry_by_grid[grid_id]!=grid_signature:raise ValueError('同一网格跨运行段几何变化，未声明移动网格契约')
            geometry_by_grid[grid_id]=grid_signature
            from datetime import datetime,timedelta
            if 'start_date_hour' in params:
                year,month,day,hour=params['start_date_hour'];start=datetime(year,month,day)+timedelta(hours=int(hour)//100,minutes=int(hour)%100)
                year,month,day,hour=params['end_date_hour'];end=datetime(year,month,day)+timedelta(hours=int(hour)//100,minutes=int(hour)%100)
                if times[0]>start or times[-1]<end:raise ValueError('实际气象时间不能覆盖声明运行区间')
            if index==0 and (int(a['NCOLS'])!=params['master_grid_columns'] or int(a['NROWS'])!=params['master_grid_rows']):
                raise ValueError('主网格形状与控制声明冲突')
            if index:
                parent=geometry_by_grid.get('0')
                if parent is None:raise ValueError('嵌套网格缺主网格几何')
                ratio=params['nest_meshing_factor'][index-1]
                if not np.isclose(parent['XCELL']/a['XCELL'],ratio) or not np.isclose(parent['YCELL']/a['YCELL'],ratio):
                    raise ValueError('原生嵌套倍率与控制声明冲突')
                for axis,size,cell in [('X','NCOLS','XCELL'),('Y','NROWS','YCELL')]:
                    origin=axis+'ORIG'
                    if a[origin]<parent[origin]-1e-6 or a[origin]+a[size]*a[cell]>parent[origin]+parent[size]*parent[cell]+1e-6:
                        raise ValueError('原生子网格超出父网格；未修改声明来凑通过')
            for t,plane in zip(times,bg):
                key=(grid_id,str(t))
                sha=hashlib.sha256(plane.tobytes()).hexdigest()
                if key in overlap and overlap[key]!=sha:raise ValueError('跨文件重复网格时刻内容冲突，不能猜选或平均')
                overlap[key]=sha
            lat2=a['P_ALP'] if abs(a['P_ALP']-a['P_BET'])<=.1 else a['P_BET']
            crs=CRS.from_proj4(f'+proj=lcc +lat_1={a["P_ALP"]} +lat_2={lat2} +lat_0={a["YCENT"]} +lon_0={a["XCENT"]} +R=6370000 +units=m')
            tx=Transformer.from_crs(4326,crs,always_xy=True)
            valid_x=(x>a['XORIG'])&(x<a['XORIG']+a['NCOLS']*a['XCELL'])
            valid_y=(y>a['YORIG'])&(y<a['YORIG']+a['NROWS']*a['YCELL'])
            support=valid_y[:,None]&valid_x[None,:]
            obs,qa=_scope_observations(decoded,times)
            if progress:progress(f'处理实际运行段{run_index}、网格{index}：同刻观测局部校正，未向全域外推')
            product,report=process_collocations(bg,times,x,y,obs,tx.transform,lon,lat,variable='air_temperature',unit='K',support_mask=support)
            report.update(resource=row['target'],grid_id=grid_id,run_index=run_index,geometry=grid_signature,observation_quality=qa,
                coordinate_assumption='station geographic coordinates interpreted as EPSG:4326; not survey verification',
                comparability_assumption='NOAA near-surface air temperature used against native 2 m field; actual sensor exposure/elevation representativeness not verified',
                storage_shape=list(bg.shape[1:]),computational_shape=[int(a['NROWS']),int(a['NCOLS'])],buffer_cells_preserved=True,
                unchanged_fields='wind, 3D temperature, moisture, dynamics and all other native variables')
            save(stage/f'processing_{i}.json',report);reports.append(report)
            if product is not None:
                values,ti,cells,out=product;out.to_netcdf(stage/f'product_{i}.nc',engine='h5netcdf')
                products.append((source,row,values,ti,cells))
            files_by_path[row['target']]=source
    if not products:raise ValueError('全部网格均无可处理的合法观测支持，未交付空融合')
    project=folder/'native_project'
    if progress:progress('按通用依赖计划生成完整工程副本；写回各网格并逐变量回读')
    mapping,inventory=materialize_project(uploads,binding,project);validations=[]
    for source,row,values,ti,cells in products:
        validation=write_native_copy(source,project/mapping[row['target']],'t2',values,ti,cells)
        validations.append(dict(validation,resource=row['target'],run_index=row.get('run_index',0),processed_times=len(ti),processed_cells=len(cells)))
    for item in inventory:
        with (project/item['path']).open('rb') as f:item['output_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
        if item['source'] not in {row['target'] for _,row,_,_,_ in products} and item['output_sha256']!=item['sha256']:
            raise ValueError('非目标依赖发生变化')
    save(project/'inventory.json',inventory);save(folder/'native_validation.json',validations)
    manifest=dict(protocol='ecodata.converter.v1',model_id=binding['model_id'],status='PROJECT_UPDATED',
        native_ready=False,native_writeback_verified=True,native_model_executed=False,complete_model_input=False,
        preparation=dict(route='bound_multirun_multigrid_meteorology_v1',tasks=reports,local_only=True),validation=validations,
        note='真实局部残差校正与原生写回；不是全域空间重建、动力平衡认证、独立精度证明或CAMx求解')
    save(folder/'handoff.json',manifest)
    archive=folder/'EcoData-updated-project.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in project.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(project).as_posix())
        for p in [folder/'handoff.json',folder/'project_binding.json',folder/'native_validation.json',*stage.glob('*.json'),*stage.glob('product_*.nc')]:
            z.write(p,'_ecodata/'+p.relative_to(folder).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('最终工程CRC失败')
        for item in inventory:
            with z.open(item['path']) as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
            if actual!=item['output_sha256']:raise ValueError('下载工程依赖回读不一致')
    return dict(manifest=manifest,archive=str(archive))
