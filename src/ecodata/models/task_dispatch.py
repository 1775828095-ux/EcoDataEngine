"""Data-task dispatch before native-format writers (not model-specific policy).

CF/Pint own physical matching; public spatial service owns algorithm selection.
Decoded observation schemas declare fields and QC, never preferred algorithms.
"""
from contextlib import ExitStack
from pathlib import Path
import hashlib
import io
import json
import zipfile

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import Delaunay

from ecodata.models.project_binding import ProjectFiles
from ecodata.models.scientific_schema import identity
from ecodata.models.source_catalog import header


# Format-level contracts: these column names explicitly encode units.
OBSERVATION_COLUMNS = {
    'temperature_K': ('air_temperature', 'K'),
    'pressure_Pa': ('air_pressure', 'Pa'),
    'relative_humidity_percent': ('relative_humidity', '%'),
    'dewpoint_K': ('dew_point_temperature', 'K'),
}


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf8')


def discover(uploads):
    """Only actual field declarations and table schemas enter task matching."""
    fields, tables, audit = [], [], []
    seen = {}
    with ExitStack() as stack:
        files = ProjectFiles(uploads, stack)
        for name in sorted(files.files):
            prefix = files.prefix(name)
            if name.lower().endswith('.csv'):
                lines = prefix.splitlines()
                columns = set(lines[0].decode('utf-8-sig').split(',')) if lines else set()
                if {'station_id','time_utc','latitude','longitude','levels'} <= columns:
                    for field, (quantity, unit) in OBSERVATION_COLUMNS.items():
                        if {field,field+'_qc'} <= columns:
                            digest = files.identity(name)['sha256']
                            alias = (digest,field)
                            if alias in seen:
                                audit.append(dict(file=name,action='DUPLICATE_ALIAS',same_as=seen[alias],field=field))
                                continue
                            seen[alias] = name
                            with files.files[name][0]() as stream:
                                frame = pd.read_csv(stream, dtype={'station_id':str})
                            tables.append(dict(file=name,frame=frame,field=field,quantity=quantity,unit=unit,sha256=digest))
                continue
            if not prefix.startswith((b'CDF',b'\x89HDF\r\n\x1a\n')):
                continue
            try:
                facts = header(prefix)
            except (ValueError,OSError,RuntimeError):
                if not prefix.startswith(b'\x89HDF\r\n\x1a\n'):
                    audit.append(dict(file=name,action='HEADER_DEFERRED'))
                    continue
                # HDF metadata is not necessarily contiguous in its first 64K.
                # Reuse xarray/h5netcdf's seekable reader, not full array loading.
                with files.files[name][0]() as stream, xr.open_dataset(stream,engine='h5netcdf') as opened:
                    facts=dict(attributes=dict(opened.attrs),variables={n:dict(attributes=dict(a.attrs)) for n,a in opened.variables.items()})
            # Raw native states stay with their existing format drivers. CF
            # standardised inputs can use task services without native engines.
            if not str(facts['attributes'].get('Conventions','')).startswith('CF-'):
                continue
            for field, item in facts['variables'].items():
                attrs = item['attributes']
                quantity,basis = identity(field, attrs)
                if basis not in {'standard_name','physical_variable'} or not attrs.get('units'):
                    continue
                with files.files[name][0]() as stream:
                    raw=stream.read()
                digest=hashlib.sha256(raw).hexdigest()
                alias=(digest,field)
                if alias in seen:
                    audit.append(dict(file=name,action='DUPLICATE_ALIAS',same_as=seen[alias],field=field))
                    continue
                seen[alias]=name
                with xr.open_dataset(io.BytesIO(raw)) as opened:
                    ds=opened.load()
                fields.append(dict(file=name,field=field,quantity=quantity,unit=attrs['units'],ds=ds,sha256=digest))
    candidates=[]
    for table in tables:
        if not table['frame'][table['field']].isna().any():
            audit.append(dict(file=table['file'],field=table['field'],action='NO_NATURAL_GAP'))
            continue
        matches=[field for field in fields if field['quantity']==table['quantity']]
        if matches:
            candidates.append((table,matches))
        else:
            audit.append(dict(file=table['file'],field=table['field'],action='NO_COMPATIBLE_BACKGROUND',quantity=table['quantity']))
    return candidates,audit


def normalise(table, field):
    """Expose the same ports as any other spatial task, with explicit evidence."""
    from ecodata.engines.units import UnitEngine
    import cf_xarray  # noqa: F401; register the existing CF coordinate accessor
    ds=field['ds'];a=ds[field['field']]
    try:
        lon=ds.cf['longitude'];lat=ds.cf['latitude']
    except KeyError:
        if not {'longitude','latitude'} <= set(ds.coords):
            raise ValueError('CF背景没有可确认的经纬度坐标')
        lon,lat=ds.longitude,ds.latitude
    if lon.shape!=lat.shape or lon.dims!=a.dims[-lon.ndim:]:
        raise ValueError('背景坐标与变量空间维度不一致')
    if 'time' in a.dims:
        times=pd.DatetimeIndex(ds.time.values)
    else:
        stamp=ds.attrs.get('analysis_time_utc')
        if not stamp or pd.Timestamp(stamp).tzinfo is None:
            raise ValueError('静态背景需要明确带时区的analysis_time_utc；不从文件名猜测')
        times=pd.DatetimeIndex([pd.Timestamp(stamp).tz_convert('UTC').tz_localize(None)])
        a=a.expand_dims(time=times)
    a=UnitEngine().convert(a,table['unit'])
    if len(times)>1 and (times.has_duplicates or not times.is_monotonic_increasing):
        raise ValueError('背景时间不唯一或不递增')
    quantity=table['quantity'];frame=table['frame'].copy();value=table['field']
    # time_utc is an explicit format field, not a guessed local timezone.
    frame['time']=pd.to_datetime(frame.time_utc.str.replace('_','T',regex=False),utc=True,errors='raise').dt.tz_localize(None)
    frame=frame[frame.levels.eq(1)&frame.time.isin(times)].copy()
    if not len(frame):raise ValueError('观测与背景无精确共同时间；未默认为±1小时平均')
    if frame[['latitude','longitude']].isna().any().any():raise ValueError('观测缺少坐标')
    qc=frame[value+'_qc']
    valid=frame[value].notna()&qc.eq(0)
    # Multiple reports at one coordinate are not independent validation sites.
    colocated=frame.loc[valid].duplicated(['time','longitude','latitude'],keep=False)
    excluded=frame.loc[valid].loc[colocated].index
    valid.loc[excluded]=False
    observations=frame.loc[valid].copy()
    if observations.station_id.isna().any() or observations.duplicated(['station_id','time']).any():
        raise ValueError('观测站点身份缺失或同站同时刻重复')
    target=frame[frame[value].isna()].copy()
    if not len(target):raise ValueError('共同时间没有待恢复的自然缺失；不自动制造遮挡')
    xy=np.column_stack([lon.values.ravel(),lat.values.ravel()])
    inside=Delaunay(xy).find_simplex(target[['longitude','latitude']].to_numpy(float))>=0
    target['inside_background']=inside
    target['output_row']=np.arange(len(target))
    accepted=target[target.inside_background].copy()
    if not len(accepted):raise ValueError('所有缺失目标均在背景空间支持范围之外')
    attrs=dict(units=table['unit'],physical_variable=quantity,physical_definition='Declared '+quantity+'; near-surface observation/background comparison',
        temporal_semantics='instantaneous',spatial_semantics='grid_estimate',data_kind='continuous')
    model=xr.Dataset({quantity:(('time','point'),a.values.reshape(len(times),-1),attrs)},
        coords=dict(time=times,point=np.arange(len(xy)),lon=('point',xy[:,0]),lat=('point',xy[:,1])),
        attrs=dict(crs='EPSG:4326',source_id='declared_cf:'+field['sha256']))
    reference=pd.DataFrame(dict(time=observations.time,station_id=observations.station_id,longitude=observations.longitude,
        latitude=observations.latitude,value=observations[value],unit=table['unit'],variable=quantity,split='train',
        temporal_semantics='instantaneous',crs='EPSG:4326'))
    template=xr.Dataset(coords=dict(time=times,point=np.arange(len(accepted)),lon=('point',accepted.longitude.to_numpy()),
        lat=('point',accepted.latitude.to_numpy())),attrs=dict(crs='EPSG:4326'))
    report=dict(task='observation_gap_recovery',quantity=quantity,unit=table['unit'],times=[str(t) for t in times],
        exact_time_only=True,valid_conditioning_rows=len(reference),natural_gap_targets=len(target),inside_targets=len(accepted),
        excluded_collocated_reports=len(excluded),nonzero_qc_not_used_as_training=int((~qc.eq(0)).sum()),
        original_qc_preserved=True,natural_gap_truth_available=False,
        limitations=['Near-surface height/terrain representativeness not independently corrected',
                    'Geographic angular coordinates interpreted under explicit operational EPSG:4326 convention; not survey-datum certification',
                    'Model background may have assimilated related observations'],
        source=dict(background=field['file'],background_sha256=field['sha256'],table=table['file'],table_sha256=table['sha256']))
    return model,reference,template,target,accepted,report


def dispatch_preparation(uploads, model_id, folder, progress=None, task=None):
    """A data-task stage independent of the downstream target model."""
    candidates,audit=discover(uploads)
    folder=Path(folder)
    plan=dict(version='data_task_dispatch_v1',target_model=model_id,audit=audit,
              candidate_tasks=[dict(task='observation_gap_recovery',quantity=t['quantity'],table=t['file'],
                backgrounds=[f['file'] for f in fs]) for t,fs in candidates])
    save(folder/'task_dispatch.json',plan)
    if task:
        if task.get('type')!='observation_gap_recovery':
            raise ValueError('当前数据任务分派未支持：'+str(task.get('type')))
        candidates=[(t,fs) for t,fs in candidates if t['quantity']==task.get('quantity')]
        if not candidates:raise ValueError('声明任务没有满足契约的观测/背景资源，详见task_dispatch.json')
    if not candidates:return None
    from ecodata.algorithms.platform_search import SearchInput, process_current_input
    results=[];members=[]
    for index,(table,matches) in enumerate(candidates):
        if len(matches)!=1:raise ValueError('同物理量存在多个背景，需明确时间/来源用途，未猜选：'+table['quantity'])
        stage=folder/f'task-{index+1}';stage.mkdir(exist_ok=False)
        model,observations,target,original,accepted,report=normalise(table,matches[0])
        save(stage/'contract.json',report)
        quantity=table['quantity']
        inputs=[SearchInput('field.nc','primary',model.to_netcdf()),
                SearchInput('observations.csv','observations',observations.to_csv(index=False).encode()),
                SearchInput('target.nc','target',target.to_netcdf())]
        request=dict(task='multisource',processing_goal='spatial_fusion',variable=quantity,unit=table['unit'],
            semantic_type='continuous',physical_variable=quantity,temporal_semantics='instantaneous',
            allow_training=True,mutation_control='random',soft_guidance=False,observation_crs='EPSG:4326',
            nonnegative=quantity in {'air_temperature','air_pressure','relative_humidity'},target_times=report['times'])
        budget=dict(objective_limit=240,total_call_limit=240,proposal_limit=40,wall_seconds=60,probe_call_limit=0)
        save(stage/'request.json',dict(request=request,budget=budget,seed=11))
        if progress:progress('按数据任务调用公共数值处理：'+quantity+' 缺失恢复')
        result=process_current_input(inputs,request,stage/'search',budget=budget,seed=11,progress=progress)
        if result.get('status')!='PASS':raise ValueError('公共数值处理未通过：'+str(result.get('status')))
        with xr.open_dataset(result['product']) as product:
            estimates=product[quantity].values
        output=original.copy();output['estimate']=np.nan;output['estimate_status']='OUTSIDE_BACKGROUND_SUPPORT'
        ti=pd.DatetimeIndex(report['times']).get_indexer(accepted.time)
        output.loc[accepted.index,'estimate']=estimates[ti,np.arange(len(accepted))]
        output.loc[accepted.index,'estimate_status']='MODEL_ESTIMATE_NOT_OBSERVATION'
        path=stage/'recovered_observations.csv';output.to_csv(path,index=False)
        readback=pd.read_csv(path)
        if not np.allclose(readback.loc[readback.estimate.notna(),'estimate'],output.loc[output.estimate.notna(),'estimate']):
            raise ValueError('恢复产品回读不一致')
        results.append(dict(task='observation_gap_recovery',field=quantity,status='PASS',report=report,
            result=result,product=str(path),native_model_updated=False))
        members.extend([path,stage/'contract.json',stage/'request.json',Path(result['product']),
            stage/'search/summary.json',stage/'search/selection_comparison.json'])
    manifest=dict(protocol='ecodata.converter.v1',model_id=model_id,status='TASK_PROCESSED',native_ready=False,
        preparation=dict(route='data_task_dispatch_v1',tasks=results),native_model_executed=False,
        note='观测缺失恢复结果，不是模型原生初始场；原观测值及QC未改写，估计值另列')
    save(folder/'handoff.json',manifest)
    archive=folder/'EcoData-task-products.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in [folder/'handoff.json',folder/'task_dispatch.json',*members]:
            z.write(p,p.relative_to(folder).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('任务输出包CRC失败')
    return dict(manifest=manifest,archive=str(archive))
