"""Native meteorology adapters to the existing public spatial fusion service.

File/field contracts are format-scoped. No location, date or winning algorithm
is encoded here. Native writing edits a copy, preserving every unrelated value.
"""
from contextlib import ExitStack
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
import hashlib
import io
import json
import shutil
import zipfile

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from scipy.spatial import Delaunay

from ecodata.models.project_binding import ProjectFiles, bound_resource


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf8')


def read_observations(uploads):
    """Existing decoded NOAA schema: units/UTC/QA are explicit field names.

Other CSV tables remain unclassified; never infer quantities from file names.
    """
    readings, locations = [], []
    with ExitStack() as stack:
        files = ProjectFiles(uploads, stack)
        for name in sorted(files.files):
            if not name.lower().endswith('.csv'):
                continue
            header = files.prefix(name).splitlines()[0].decode('utf-8-sig').split(',')
            observation = {'station', 'time_utc', 'temperature_C', 'temperature_quality', 'report_type'} <= set(header)
            location = {'station', 'latitude', 'longitude', 'elevation_m'} <= set(header)
            if not observation and not location:
                continue
            with files.files[name][0]() as stream:
                frame = pd.read_csv(stream, dtype={'station':str, 'temperature_quality':str})
            frame['source_member'] = name
            (readings if observation else locations).append(frame)
    if not readings:
        return None
    if not locations:
        raise ValueError('气温观测存在但缺少明确的站点坐标表')
    obs = pd.concat(readings, ignore_index=True)
    stations = pd.concat(locations, ignore_index=True)
    if stations.duplicated('station').any():
        raise ValueError('站点元数据重复，不能自动选择坐标')
    obs = obs.merge(stations[['station','latitude','longitude']], on='station', how='left', validate='many_to_one')
    if obs[['latitude','longitude']].isna().any().any():
        raise ValueError('气温记录没有对应站点坐标，未静默丢弃')
    if obs.duplicated(['station','time_utc']).any():
        raise ValueError('站点时间重复，不平均不同报告')
    obs['time'] = pd.to_datetime(obs.time_utc, utc=True, errors='raise').dt.tz_localize(None)
    # NOAA Global Hourly quality 1/5: passed all quality checks. SOD is not an
    # instantaneous report; its missing QA/value must never enter calibration.
    good = obs.temperature_quality.isin(['1','5']) & obs.report_type.isin(['FM-12','FM-15','FM-16'])
    good &= np.isfinite(pd.to_numeric(obs.temperature_C, errors='coerce'))
    report = dict(schema='NOAA_GLOBAL_HOURLY_DECODED_TEMPERATURE_V1',
                  total=len(obs), qa_accepted=int(good.sum()), qa_rejected=int((~good).sum()),
                  unit_conversion='temperature_C + 273.15 -> K',
                  time_basis='explicit time_utc field',
                  coordinate_interpretation='EPSG:4326 operational convention for supplied geographic station metadata; survey datum not independently verified',
                  rejected=obs.loc[~good,['station','time_utc','temperature_quality','report_type']].to_dict('records'))
    obs = obs.loc[good].copy()
    obs['value'] = obs.temperature_C.astype(float) + 273.15
    return obs, report


def native_times(ds):
    values = np.asarray(ds['TFLAG'][:,0,:], dtype=int)
    return pd.DatetimeIndex([datetime.strptime(str(day), '%Y%j') +
        timedelta(hours=int(clock)//10000, minutes=(int(clock)//100)%100, seconds=int(clock)%100)
        for day, clock in values])


def geometry(ds):
    """Reuse CAMx's existing sphere-LCC grid implementation with actual header."""
    from ecodata.models.adapters.camx_emissions import _grid
    a = {k:ds.getncattr(k) for k in ds.ncattrs()}
    if int(a.get('GDTYP',-1)) != 2 or int(a.get('CPROJ',-1)) != 2 or int(a.get('ITZON',99)) != 0:
        raise ValueError('当前原生气象适配支持明确UTC的CAMx球体LCC网格')
    g, _, _, lon, lat = _grid(dict(projection='LCC', ncols=int(a['NCOLS']), nrows=int(a['NROWS']),
        xorig=a['XORIG'], yorig=a['YORIG'], xcell=a['XCELL'], ycell=a['YCELL'],
        coordinate_units='m', xcent=a['XCENT'], ycent=a['YCENT'], p_alp=a['P_ALP'],
        p_bet=a['P_BET'], itzon=0))
    expected_x=(g['xorig']+(np.arange(g['ncols'])+.5)*g['xcell'])/1000
    expected_y=(g['yorig']+(np.arange(g['nrows'])+.5)*g['ycell'])/1000
    if not np.allclose(ds['X'][:],expected_x) or not np.allclose(ds['Y'][:],expected_y):
        raise ValueError('原生X/Y与投影网格中心不一致')
    return lon, lat


def write_native_copy(source, target, variable, updates, time_indices, cell_indices):
    """Lossless container copy + restricted native-variable patch and full readback."""
    shutil.copyfile(source, target)
    with Dataset(source) as original, Dataset(target, 'r+') as native:
        if native[variable].dimensions != ('TSTEP','LAY','ROW','COL') or native[variable].shape[1] != 1:
            raise ValueError('写回仅支持明确单层二维气象场')
        if updates.shape != (len(time_indices),len(cell_indices)) or not np.isfinite(updates).all():
            raise ValueError('写回预测形状错误或存在非法值')
        for k, ti in enumerate(time_indices):
            plane=np.asarray(native[variable][ti,0,:,:]).copy()
            plane.flat[cell_indices]=updates[k]
            native[variable][ti,0,:,:]=plane
    with Dataset(source) as original, Dataset(target) as native:
        if original.data_model!=native.data_model or original.ncattrs()!=native.ncattrs():
            raise ValueError('原生格式或全局属性发生变化')
        for key in original.ncattrs():
            np.testing.assert_equal(original.getncattr(key), native.getncattr(key))
        changed=0
        for name in original.variables:
            left,right=original[name],native[name]
            if left.dimensions!=right.dimensions or left.dtype!=right.dtype or left.ncattrs()!=right.ncattrs():
                raise ValueError('原生字段定义发生变化：'+name)
            for key in left.ncattrs():np.testing.assert_equal(left.getncattr(key),right.getncattr(key))
            a,b=left[:],right[:]
            if name!=variable:
                np.testing.assert_equal(a,b)
            else:
                expected=np.array(a,copy=True)
                for k,ti in enumerate(time_indices):expected[ti,0].flat[cell_indices]=updates[k]
                np.testing.assert_equal(expected,b)
                changed=int(np.count_nonzero(np.asarray(a)!=np.asarray(b)))
        return dict(status='PASS',native_data_model=native.data_model,changed_values=changed,
                    edited_variable=variable,other_variables_and_metadata_equal=True,
                    time_flags_equal=True,unprocessed_support_equal=True,
                    native_model_executed=False)


def materialize_project(uploads, binding, folder):
    """Shared binding resources become a portable, complete dependency directory."""
    import f90nml
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    mapping={};inventory=[]
    with ExitStack() as stack:
        files=ProjectFiles(uploads,stack)
        for i, resource in enumerate(binding['resources']):
            name=f'inputs/{i:03d}_{PurePosixPath(resource["path"]).name}'
            target=folder/name;target.parent.mkdir(exist_ok=True)
            h=hashlib.sha256()
            with files.files[resource['path']][0]() as stream, target.open('wb') as dest:
                for block in iter(lambda:stream.read(8*1024**2),b''):
                    h.update(block);dest.write(block)
            if h.hexdigest()!=resource['sha256']:raise ValueError('绑定后输入变化：'+resource['path'])
            mapping[resource['path']]=name
            inventory.append(dict(path=name,sha256=h.hexdigest(),source=resource['path']))
        with files.files[binding['control']['path']][0]() as stream:
            raw=stream.read()
        if hashlib.sha256(raw).hexdigest()!=binding['control']['sha256']:raise ValueError('原控制已变化')
        (folder/'original_control.txt').write_bytes(raw)
        nml=f90nml.read(io.StringIO(raw.decode('utf-8-sig')))
        group=nml['camx_control']
        for row in binding['bindings']:
            if not row['indexes']:group[row['parameter']]=mapping[row['target']]
            else:
                value=group[row['parameter']]
                for index in row['indexes'][:-1]:value=value[index]
                value[row['indexes'][-1]]=mapping[row['target']]
        # Output paths are sandboxed; all scientific settings remain original.
        group['root_output_name']='outputs/CAMx'
        (folder/'outputs').mkdir()
        nml.write(folder/'CAMx.in')
        reread=f90nml.read(folder/'CAMx.in')
        for key,value in nml['camx_control'].items():
            if reread['camx_control'][key]!=value:raise ValueError('控制文件回读不一致：'+key)
    return mapping,inventory


def prepare_meteorology(uploads, binding, folder, progress=None):
    decoded=read_observations(uploads)
    if decoded is None:return None
    rows=[r for r in binding['bindings'] if r['parameter']=='met2d_grid']
    if len(rows)!=1:raise ValueError('气象处理需唯一二维气象绑定，嵌套网格尚未支持')
    from ecodata.algorithms.platform_search import SearchInput, process_current_input
    folder=Path(folder);stage=folder/'meteorology';stage.mkdir(exist_ok=False)
    obs,report=decoded;save(stage/'observation_quality.json',report)
    source=stage/'original_met2d.nc'
    with bound_resource(uploads,binding,'met2d_grid',tuple(rows[0]['indexes'])) as stream, source.open('wb') as dest:
        shutil.copyfileobj(stream,dest,8*1024**2)
    with Dataset(source) as ds:
        if 't2' not in ds.variables or str(ds['t2'].units).strip()!='K' or str(ds['t2'].long_name).strip()!='temperature at 2 m':
            raise ValueError('未确认原生2米气温定义和K单位，不猜测其他变量')
        times=native_times(ds);lon,lat=geometry(ds);background=np.asarray(ds['t2'][:,0,:,:])
    coords=obs.drop_duplicates('station')[['longitude','latitude']].to_numpy(float)
    if len(coords)<3 or not np.isfinite(coords).all():raise ValueError('至少需要三个合法气温站点')
    obs=obs[obs.time.isin(times)].copy()
    counts=obs.groupby('time').station.nunique()
    selected_times=pd.DatetimeIndex(counts[counts>=3].index).sort_values()
    if not len(selected_times):raise ValueError('无至少三站共同覆盖的原生时刻；不跨小时借值')
    # Per-time convex support: no continental correction from five local stations.
    xy=np.column_stack([lon.ravel(),lat.ravel()]);masks=[]
    for time in selected_times:
        points=obs.loc[obs.time.eq(time),['longitude','latitude']].to_numpy(float)
        masks.append(Delaunay(points).find_simplex(xy)>=0)
    common=np.logical_and.reduce(masks);cells=np.flatnonzero(common)
    if not len(cells):raise ValueError('站点共同有效域内无原生格点中心')
    obs=obs[obs.time.isin(selected_times)].copy()
    frame=pd.DataFrame(dict(time=obs.time,station_id=obs.station,longitude=obs.longitude,latitude=obs.latitude,
        value=obs.value,unit='K',variable='air_temperature',split='train',temporal_semantics='instantaneous',crs='EPSG:4326'))
    attrs=dict(units='K',physical_variable='air_temperature',physical_definition='air temperature at 2 m',
               temporal_semantics='instantaneous',spatial_semantics='grid_estimate',data_kind='continuous')
    model=xr.Dataset({'air_temperature':(('time','point'),background.reshape(len(times),-1),attrs)},
        coords=dict(time=times,point=np.arange(len(xy)),lon=('point',xy[:,0]),lat=('point',xy[:,1])),
        attrs=dict(crs='EPSG:4326',source_id='bound_met2d:'+rows[0]['sha256']))
    target=xr.Dataset(coords=dict(time=selected_times,point=np.arange(len(cells)),
        lon=('point',xy[cells,0]),lat=('point',xy[cells,1])),attrs=dict(crs='EPSG:4326'))
    inputs=[SearchInput('background.nc','primary',model.to_netcdf()),
            SearchInput('observations.csv','observations',frame.to_csv(index=False).encode()),
            SearchInput('target.nc','target',target.to_netcdf())]
    for item in inputs:(stage/item.filename).write_bytes(item.content)
    request=dict(task='multisource',processing_goal='spatial_fusion',variable='air_temperature',unit='K',
        semantic_type='continuous',physical_variable='air_temperature',temporal_semantics='instantaneous',
        allow_training=True,mutation_control='random',soft_guidance=False,nonnegative=True,
        observation_crs='EPSG:4326',target_times=[str(t) for t in selected_times])
    budget=dict(objective_limit=240,total_call_limit=240,proposal_limit=120,wall_seconds=90,probe_call_limit=0)
    save(stage/'protocol.json',dict(request=request,budget=budget,seed=11,observation_quality=report,
        native_times=[str(t) for t in times],processed_times=[str(t) for t in selected_times],
        exact_time_match_only=True,processed_cell_count=len(cells),total_cell_count=len(xy),
        spatial_scope='intersection of per-time station convex hulls; all other native values preserved',
        limitations=['Station geographic datum operational EPSG:4326; not independent survey validation',
                     'Only 2m temperature; wind height/rotation contracts unresolved, preserved unchanged',
                     'No 3D temperature adjustment or claim of dynamically balanced meteorology']))
    if progress:progress('调用平台已有站点/背景比较、真实留站评价与随机变异')
    result=process_current_input(inputs,request,stage/'search',budget=budget,seed=11,progress=progress)
    if result.get('status')!='PASS' or not result.get('product'):raise ValueError('平台气象处理未产生合格产品')
    with xr.open_dataset(result['product']) as ds:
        updated=ds.air_temperature.values.copy()
    if progress:progress('复制绑定工程，保留全部依赖；写回并逐变量核对原生气象')
    project=folder/'native_project';mapping,inventory=materialize_project(uploads,binding,project)
    target_path=project/mapping[rows[0]['target']]
    ti=times.get_indexer(selected_times)
    validation=write_native_copy(source,target_path,'t2',updated,ti,cells)
    validation.update(native_CAMx_execution='NOT_RUN',station_accuracy='NOT_INDEPENDENTLY_VERIFIED',
        wind_and_3d_unchanged=True,selection=result['selection_comparison'],
        post_selection=result['post_selection_assessment'],processed_times=len(ti),processed_cells=len(cells))
    save(folder/'native_validation.json',validation)
    for item in inventory:
        with (project/item['path']).open('rb') as stream:item['output_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
    save(project/'inventory.json',inventory)
    # Native file update is a real output; full dependency directory stays local.
    # The download is a delta + control + evidence, explicitly not all 12GB inputs.
    manifest=dict(protocol='ecodata.converter.v1',model_id='camx',status='CONVERTED',native_ready=False,
        native_schema_verified=False,native_writeback_verified=True,native_model_executed=False,complete_project_directory=str(project.resolve()),
        preparation=dict(route='bound_native_meteorology_v1',tasks=[],fusion_decision=dict(
            status='MULTISOURCE_GRAPH_EFFECT_UNVERIFIED' if result['selection_comparison']['selected_family']=='MULTISOURCE' else 'SINGLE_SOURCE_SELECTED',
            selected_family=result['selection_comparison']['selected_family'])),
        conversion=dict(status='PASS',validation=validation),
        note='本地native_project含全部绑定依赖；下载为原生气象更新包及控制/证据，原生模拟未运行，不能称全物理一致性已证明')
    save(folder/'handoff.json',manifest)
    archive=folder/'EcoData-meteorology-update.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        z.write(target_path,'native_update/'+target_path.name)
        for p in [folder/'handoff.json',folder/'project_binding.json',folder/'native_validation.json',
                  project/'CAMx.in',project/'inventory.json',stage/'protocol.json',stage/'observation_quality.json']:
            z.write(p,p.relative_to(folder).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('下载更新包CRC失败')
    return dict(manifest=manifest,archive=str(archive))
