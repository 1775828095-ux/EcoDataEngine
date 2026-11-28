"""Evidence-adaptive evaluation over existing spatial kernels, never source bonuses.

Reduced-scale tests assess reconstruction of a product, not environmental truth.
Only cell means have the reviewed area operator; unknown support stays unknown.
"""
from hashlib import sha256
from time import perf_counter
import numpy as np
import pandas as pd
import xarray as xr
from ecodata.algorithms.raster_reconstruction_task import edges, overlap_weights
from ecodata.algorithms.spatial_selection_protocol import split_locations, error_metrics

VERSION='fusion_evidence_v2_target_domain'


def remap(source, target, variable):
    """Apply the existing spherical cell-mean observation operator in either direction."""
    ay=overlap_weights(edges(source.lat),edges(target.lat),True)
    ax=overlap_weights(edges(source.lon),edges(target.lon))
    a=source[variable].transpose('time','lat','lon').values
    return np.stack([np.asarray(ay@frame@ax.T) for frame in a])


def degrade(source, variable, factor=2, extent=None):
    coords={}
    for axis in ('lat','lon'):
        b=edges(source[axis])
        low,high=extent[axis] if extent else (b[0],b[-1])
        count=int(np.floor((high-low)/(float(source[axis][1]-source[axis][0])*factor)+1e-9))
        if count<2:raise ValueError('Fewer than two degraded cells on '+axis)
        bounds=np.linspace(low,high,count+1)
        coords[axis]=(bounds[:-1]+bounds[1:])/2
    target=xr.Dataset(coords=coords)
    out=xr.Dataset({variable:(('time','lat','lon'),remap(source,target,variable))},
        coords=dict(time=source.time,**coords),attrs=dict(source.attrs))
    out[variable].attrs=dict(source[variable].attrs)
    return out


def prepare(task, times, seed):
    """Geometry selects the finest native reference; no errors select a source."""
    started=perf_counter()
    plan=dict(version=VERSION,mode='COARSE_PRODUCT_RECONSTRUCTION_ONLY',
        accuracy_established=False,source_bonus=0,metric='RMSE',
        selection_meaning='Coarse product reconstruction only; fine-scale ranking unsupported',
        missing_evidence=[],independence='UNKNOWN',cost_seconds=0.)
    plan['sources']=[dict(source_id=d.attrs['source_id'],role='primary' if i==0 else 'background',
        physical_variable=d[task.request['variable']].attrs.get('physical_variable','UNKNOWN'),
        unit=d[task.request['variable']].attrs.get('units','UNKNOWN'),
        support=d[task.request['variable']].attrs.get('spatial_semantics','UNKNOWN'),
        temporal_semantics=d[task.request['variable']].attrs.get('temporal_semantics','UNKNOWN'),
        crs=d.attrs.get('crs','UNKNOWN'),crs_evidence=d.attrs.get('crs_evidence','UNKNOWN'),
        independence=d.attrs.get('independence','UNKNOWN'),training_role='Input only; no fine assessment labels')
        for i,d in enumerate(task.native_fields or [])]
    plan['method_sources']=['https://github.com/liangjiandeng/DLPan-Toolbox',
        'https://github.com/rvalavi/blockCV']
    plan['reuse']='Existing spherical area operator and sklearn spatial block splitter; no imagery metric or third-party optimizer copied'
    if not task.self_reference:
        plan.update(mode='OBSERVATION_HOLDOUT',selection_meaning='Held-location observation error at declared observation support')
        plan['missing_evidence']=['Upstream independence of gridded sources and observations must be established separately']
        return plan,None
    native=task.native_fields
    if not native or any(d[task.request['variable']].attrs.get('spatial_semantics')!='cell_mean' for d in native):
        plan['missing_evidence']=['Reviewed native cell means unavailable; no invented degradation operator']
        return plan,None
    variable=task.request['variable']
    try:
        clipped=[]
        for source in native:
            ds=source.sel(time=times)
            for axis in ('lat','lon'):
                halo=2*float(ds[axis][1]-ds[axis][0])
                ds=ds.sel({axis:slice(float(task.template[axis][0])-halo,float(task.template[axis][-1])+halo)})
            if not np.isfinite(ds[variable]).all():raise ValueError('Native degradation region contains missing cells; no truth filling')
            clipped.append(ds)
        reference_index=min(range(len(clipped)),key=lambda i:(float(np.diff(clipped[i].lat)[0]*np.diff(clipped[i].lon)[0]),str(clipped[i].attrs['source_id'])))
        extent={axis:(max(edges(d[axis])[0] for d in clipped),min(edges(d[axis])[-1] for d in clipped)) for axis in ('lat','lon')}
        degraded=[degrade(d,variable,extent=extent) for d in clipped]
        reference=clipped[reference_index]
        for axis in ('lat','lon'):
            low=max(edges(d[axis])[0] for d in degraded)
            high=min(edges(d[axis])[-1] for d in degraded)
            b=edges(reference[axis]);keep=(b[:-1]>=low-1e-9)&(b[1:]<=high+1e-9)
            reference=reference.isel({axis:np.flatnonzero(keep)})
            if reference.sizes[axis]<2:raise ValueError('Insufficient common native reference footprint on '+axis)
        from ecodata.spatial.regridding import contained_cell_grid
        before_domain=int(reference.sizes['lat']*reference.sizes['lon'])
        reference=contained_cell_grid(reference,task.template)
        domain=dict(rule='Complete reference cell support inside application target edges; halo remains input-only',
            reference_cells_before=before_domain,reference_cells_retained=int(reference.sizes['lat']*reference.sizes['lon']),
            application_bounds={axis:[float(edges(task.template[axis])[0]),float(edges(task.template[axis])[-1])] for axis in ('lat','lon')},
            evaluation_bounds={axis:[float(edges(reference[axis])[0]),float(edges(reference[axis])[-1])] for axis in ('lat','lon')},
            reference_axes={axis:reference[axis].values.tolist() for axis in ('lat','lon')},
            outside_cells=0)
        # Conditioning is the degraded primary, never retained fine reference labels.
        conditioning=degraded[0]
        for axis in ('lat','lon'):
            b=edges(conditioning[axis]);low=max(edges(d[axis])[0] for d in degraded);high=min(edges(d[axis])[-1] for d in degraded)
            conditioning=conditioning.isel({axis:np.flatnonzero((b[:-1]>=low-1e-9)&(b[1:]<=high+1e-9))})
            if conditioning.sizes[axis]<2:raise ValueError('Insufficient common degraded conditioning cells')
        from ecodata.algorithms.spatial_fusion_task import positions
        query=positions(reference)[0];sxy=positions(conditioning)[0]
        sources=[np.concatenate([remap(d,reference,variable).reshape(len(times),-1),
                                 remap(d,conditioning,variable).reshape(len(times),-1)],axis=1) for d in degraded]
        values=reference[variable].values.reshape(len(times),-1)
        meta=pd.DataFrame(dict(station_id=[str(i) for i in range(len(query))],longitude=query[:,0],latitude=query[:,1]))
        split=split_locations(meta,seed)
        selected=np.array(split['selection_ids'],int);test=np.array(split['test_ids'],int)
        labels=values[:,selected]
        support=sha256(VERSION.encode()+repr((selected.tolist(),[str(t) for t in times])).encode()+query.tobytes()+np.isfinite(labels).tobytes()).hexdigest()
        from ecodata.evaluation.intermediate_scale import prepare_grid
        intermediate=prepare_grid(reference,degraded[reference_index])
        intermediate_plan=dict(status='READY' if intermediate is not None else 'INTERMEDIATE_GRID_UNAVAILABLE',
            reference='Same withheld native finer product used for fine scoring',
            coarse_endpoint='Degraded grid of the same reference product in the reduced-scale experiment',
            operator='Shared spherical area-weighted cell means; only complete cells inside each evaluation split')
        if intermediate is not None:
            intermediate_plan['steps_degrees']=intermediate['steps_degrees']
        plan.update(mode='REDUCED_SCALE_PRODUCT_RECONSTRUCTION',reference_source=reference.attrs['source_id'],
            reference_role='Original native finer product, not independent environmental truth',
            factor=2,operator='Existing spherical area overlap, complete native cells, no interpolation truth',
            degradation_note='Nominal factor 2; common edge footprint and integer cell counts determine actual axis factors',
            degraded_axes=[dict(source_id=d.attrs['source_id'],lat=d.lat.values.tolist(),lon=d.lon.values.tolist()) for d in degraded],
            selection_meaning='RMSE against withheld native finer product after degrading ALL input sources',
            reference_shape=list(values.shape),degraded_shapes=[list(d[variable].shape) for d in degraded],
            target_grid_shape=[reference.sizes['lat'],reference.sizes['lon']],
            fixed_support_sha256=support,location_split=split,conditioning_count=len(sxy),
            spatial_domain=domain,intermediate_scale=intermediate_plan,
            selection_count=int(np.isfinite(labels).sum()),test_count=int(np.isfinite(values[:,test]).sum()),
            limitations=['Scale transfer is an assumption; degradation is not the true sensor error model',
                'Shared upstream observations may correlate product errors',
                'Fine reference values are absent from assessment conditioning and source arrays; only their prescribed aggregates are inputs',
                'Geographic blocks are not a proof of independent repeats; one date cannot establish temporal robustness'],
            native_input_sha256=[sha256(d[variable].values.tobytes()).hexdigest() for d in clipped],
            degraded_input_sha256=[sha256(d[variable].values.tobytes()).hexdigest() for d in degraded])
        cohort=dict(y=conditioning[variable].values.reshape(len(times),-1),query_count=len(query),
            sources=sources,aux=[],sxy=sxy,allxy=np.concatenate([query,sxy]))
        return plan,dict(cohort=cohort,reference=values,selected=selected,test=test,split=split,query=query,
                         intermediate_grid=intermediate)
    except ValueError as exc:
        plan['missing_evidence']=[str(exc)]
        return plan,None
    finally:
        plan['cost_seconds']=perf_counter()-started


def fixed_recipe_cohort(native, template, variable, *, seed=0):
    """Use the search observation operator for a pre-search fixed recipe trial.

    This is a product reconstruction diagnostic, not a final confirmation.
    The fixed recipe receives only degraded inputs, never reference labels.
    """
    from types import SimpleNamespace

    sources=[]
    for index, ds in enumerate(native):
        source=ds.copy(deep=False)
        source.attrs=dict(ds.attrs,source_id=ds.attrs.get('source_id',f'source-{index}'))
        sources.append(source)
    task=SimpleNamespace(native_fields=sources,self_reference=True,template=template,
                         request=dict(variable=variable))
    protocol,reconstruction=prepare(task,pd.DatetimeIndex(sources[0].time.values),seed)
    if reconstruction is None:
        return protocol,None
    count=reconstruction['cohort']['query_count']
    selected=reconstruction['selected']
    inputs=[values[:,:count][:,selected] for values in reconstruction['cohort']['sources']]
    return protocol,dict(inputs=inputs,reference=reconstruction['reference'][:,selected])


def paired_report(single,multi,reference,indices,split):
    """Paired descriptive blocks, not a pixel-level confidence interval."""
    a=error_metrics(single[:,indices],reference[:,indices]);b=error_metrics(multi[:,indices],reference[:,indices])
    blocks=[]
    groups=split['block_by_station']
    for group in sorted(set(groups[str(i)] for i in indices)):
        subset=np.array([i for i in indices if groups[str(i)]==group],int)
        sa=error_metrics(single[:,subset],reference[:,subset]);mb=error_metrics(multi[:,subset],reference[:,subset])
        blocks.append(dict(block=group,single=sa,multi=mb,RMSE_change=mb['RMSE']-sa['RMSE']))
    return dict(single=a,multi=b,RMSE_change_percent=100*(b['RMSE']-a['RMSE'])/a['RMSE'] if a['RMSE']>0 else None,
        blocks=blocks,improved_blocks=sum(x['RMSE_change']<0 for x in blocks),
        uncertainty='INSUFFICIENT_INDEPENDENT_REPLICATION; no pixel bootstrap or significance claim')
