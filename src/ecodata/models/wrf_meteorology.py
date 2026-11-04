"""WRF ports -> existing public fusion/sampling -> common native writeback.

Daily SST analyses initialise sea cells under an explicit retrospective daily
analysis contract; they are not instantaneous observations or forecast evidence.
"""
from contextlib import ExitStack
from pathlib import Path
import io
import zipfile

import numpy as np
import pandas as pd
import xarray as xr

from .project_binding import ProjectFiles
from .native_meteorology import _load_observations, _scope_observations, fuse_temperature, save


def wrf_times(values):
    labels=[''.join(np.asarray(v).astype('U').ravel()) for v in values]
    clock=pd.DatetimeIndex(pd.to_datetime(labels,format='%Y-%m-%d_%H:%M:%S',errors='raise'))
    if clock.empty or clock.has_duplicates or not clock.is_monotonic_increasing:
        raise ValueError('WRF Times需非空、唯一递增，不能猜测重复时刻')
    return clock


def sst_sources(uploads):
    from existing_air_converters.wrf_bundle_pipeline import _oisst
    records=[];sources=[]
    with ExitStack() as stack:
        files=ProjectFiles(uploads,stack)
        for name in sorted(files.files):
            if not files.prefix(name,8).startswith((b'CDF',b'\x89HDF')):continue
            with files.files[name][0]() as stream,xr.open_dataset(stream) as d:
                if 'sst' not in d or 'Optimum Interpolation' not in str(d.attrs.get('title','')):continue
                record=dict(resource=files.identity(name))
                # Multi-date derived subsets can carry stale one-day metadata.
                # Do not repair their time semantics by guessing from a filename.
                if d.sizes.get('time')!=1:
                    records.append(dict(record,status='NOT_USED',reason='daily interval metadata not unique for multi-date subset'));continue
                start=d.attrs.get('time_coverage_start');end=d.attrs.get('time_coverage_end')
                if not start or not end:raise ValueError('日SST缺少实际覆盖区间')
                lo=pd.Timestamp(start);hi=pd.Timestamp(end)
                if lo.tzinfo is None or hi.tzinfo is None:raise ValueError('日SST覆盖区间缺少时区')
                lo=lo.tz_convert('UTC').tz_localize(None);hi=hi.tz_convert('UTC').tz_localize(None)
                stamp=pd.Timestamp(d.time.values[0])
                if not lo<=stamp<=hi or hi-lo>pd.Timedelta(days=1):raise ValueError('日SST时刻与覆盖区间冲突')
            with files.files[name][0]() as stream:raw=stream.read()
            ds=_oisst(raw)
            with xr.open_dataset(io.BytesIO(raw),decode_times=False,mask_and_scale=False) as encoded:
                a=encoded.sst;low=a.attrs.get('valid_min');high=a.attrs.get('valid_max')
                if low is None or high is None:raise ValueError('OISST缺少原始有效值范围')
                valid=(a>=low)&(a<=high)
                # _oisst also normalises longitude ordering and singleton zlev.
                mask=valid.squeeze(drop=True).assign_coords(lon=((valid.lon+180)%360)-180).sortby('lon').sortby('lat')
                ds['sst']=ds.sst.where(mask)
            records.append(dict(record,status='AVAILABLE',start=str(lo),end=str(hi),no_independent_truth=True))
            sources.append((name,ds,lo,hi))
    return sources,records


def sample_daily_sst(sources, times, lon, lat):
    """Existing sampler; equal aliases must agree, missing coverage stays masked."""
    from ecodata.algorithms.spatial_fusion_task import sample
    from ecodata.engines.units import UnitEngine
    xy=np.column_stack([lon.ravel(),lat.ravel()]);result=np.full((len(times),len(xy)),np.nan);used=[]
    for i,t in enumerate(times):
        for name,ds,start,end in sources:
            if not start<=t<=end:continue
            field=ds[['sst']].copy()
            field['sst']=UnitEngine().convert(field.sst,'K')
            field.sst.attrs.update(spatial_semantics='grid_estimate')
            field.attrs['crs']='EPSG:4326'
            values=sample(field,'sst',pd.DatetimeIndex(field.time.values),xy)[0]
            overlap=np.isfinite(result[i])&np.isfinite(values)
            if not np.allclose(result[i,overlap],values[overlap],atol=1e-5,rtol=0):
                raise ValueError('同日SST分析重叠值冲突，不平均来源')
            valid=np.isfinite(values);result[i,valid]=values[valid];used.append(dict(source=name,native_time=str(t),source_interval=[str(start),str(end)]))
    return result,used


def prepare_wrf_meteorology(uploads,binding,folder,progress=None):
    from .engineering.native import extract_native_field
    from .component_service import prepare_components
    from ecodata.interface.engineering_upload import LocalUpload
    folder=Path(folder);stage=folder/'wrf_meteorology';stage.mkdir()
    decoded=_load_observations(uploads);sources,source_records=sst_sources(uploads)
    if decoded is None and not sources:return None
    decisions=[];products=[]
    rows=[r for r in binding['bindings'] if r['parameter']=='initial_state']
    for row in rows:
        with ExitStack() as stack:
            files=ProjectFiles(uploads,stack)
            with files.files[row['target']][0]() as stream,xr.open_dataset(stream,decode_times=False) as d:
                times=wrf_times(d.Times.values)
                lon=np.asarray(d.XLONG[0]);lat=np.asarray(d.XLAT[0])
                if lon.shape!=lat.shape:raise ValueError('WRF经纬度支持不一致')
                sea=np.asarray(d.LANDMASK.values)==0
                if 'LAKEMASK' in d:sea &= np.asarray(d.LAKEMASK.values)==0
                if 'SEAICE' in d:sea &= np.asarray(d.SEAICE.values)==0
                if 'T2' in d and str(d.T2.attrs.get('units','')).strip()!='K':
                    raise ValueError('WRF T2单位未知或不是K；不能将未知单位直接参与气温融合')
                t2=np.asarray(d.T2.values) if 'T2' in d else None
        if decoded is not None and t2 is not None:
            obs,report=_scope_observations(decoded,times)
            common=obs.loc[obs.time.isin(times)]
            if common.station.nunique()<3:
                decisions.append(dict(target=row['target'],variable='T2',status='NOT_APPLICABLE',
                    accepted_stations=int(common.station.nunique()),required=3,
                    reason='existing leave-location-out fusion requires at least three concurrent stations; no global extrapolation from one'))
            else:
                dst=stage/('temperature_'+str(row['indexes'][0]));dst.mkdir()
                save(dst/'observation_quality.json',report)
                updated,ti,cells,result=fuse_temperature(obs,report,t2,times,lon,lat,'bound_wrf:'+row['sha256'],dst,progress)
                product=extract_native_field(uploads,'wrf',row['target'],'T2')
                for i,index in enumerate(ti):product.T2.values[index].flat[cells]=updated[i]
                products.append((row,'T2',product))
                decisions.append(dict(target=row['target'],variable='T2',status='PROCESSED',selection=result.get('selection_comparison'),cells=len(cells)))
        if sources:
            product=extract_native_field(uploads,'wrf',row['target'],'SST')
            if str(product.SST.attrs.get('units','')).strip()!='K':raise ValueError('WRF SST需要明确K单位')
            sampled,used=sample_daily_sst(sources,times,lon,lat)
            values=sampled.reshape(product.SST.shape);mask=sea&np.isfinite(values)
            if mask.any():
                product.SST.values[mask]=values[mask]
                products.append((row,'SST',product))
            decisions.append(dict(target=row['target'],variable='SST',status='PROCESSED' if mask.any() else 'NO_VALID_OVERLAP',
                processed_cells=int(mask.sum()),total_cells=int(mask.size),sources=used,
                operation='existing spatial_fusion_task.sample -> native SST replacement only on finite ice-free ocean support',
                temporal_contract='retrospective daily analysis initialisation within stated UTC coverage; not instantaneous SST or a forecast experiment',
                accuracy_gain='NOT_EVALUATED',mask_policy='retain land, native ice, explicit lake mask when present, and invalid OISST; no inferred lake geometry'))
    save(stage/'plan.json',dict(version='wrf_bound_meteorology_v1',sources=source_records,tasks=decisions,
        solver_executed=False,observations_not_relabelled_as_sst=True))
    if not products:
        return None  # Binder returns the explicit plan as remaining conditions.
    current=list(uploads);result=None
    for i,(row,variable,product) in enumerate(products):
        output=stage/f'product_{i}.nc';product.to_netcdf(output,engine='scipy',format='NETCDF3_CLASSIC')
        request=dict(type='model_component',operation='update_project',input=output.name,configuration={})
        if progress:progress('将本次处理产品写回绑定的WRF输入：'+variable)
        destination=stage/f'writeback_{i}';destination.mkdir()
        result=prepare_components([*current,LocalUpload(output.name,output)],'wrf',destination,task=request,progress=progress)
        current=[LocalUpload('updated_project.zip',Path(result['archive']))]
    # Keep the public plan in the deliverable, not only the local run folder.
    with zipfile.ZipFile(result['archive'],'a',zipfile.ZIP_DEFLATED) as z:
        z.write(stage/'plan.json','_ecodata/wrf_meteorology_plan.json')
    return result
