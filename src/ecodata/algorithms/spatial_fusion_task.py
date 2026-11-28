"""Spatial relationship adapter for the public random-mutation service.

Paired locations stay paired; gridded estimates never become station observations.
This task adds ports to the existing DAG executor, cost ledger and random driver.
It is for continuous spatial estimation, not categorical or conservative remapping.
"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS
from scipy.spatial import cKDTree, Delaunay
from threadpoolctl import threadpool_limits
from ecodata.algorithms.dag import Artifact, Port, Node, NodeAdapter, NumericalDAGExecutor, DAGError
from ecodata.algorithms.numerical_search import save
from ecodata.algorithms.search_accounting import SearchAccount, SearchBudgetExceeded
from ecodata.algorithms.search_stopping import SearchStopper, policy
from ecodata.fusion.point_weights import distance_weights, weighted_values
from ecodata.fusion.residual import residual_difference, residual_addition
from ecodata.algorithms.quality_aware import fit_affine_frames,fit_ratio_frames
from ecodata.evaluation.quality_gates import RawInputQualityGate,ResultQualityGate
from ecodata.io.structures import table_frame
from ecodata.algorithms.spatial_selection_protocol import split_locations, fixed_support, xskillscore_assessment

VERSION = 'spatial_relationship_fusion_v1_5_date_region'
COMPONENTS = {
    'spatial.source': dict(kind='source',backend='xarray / SciPy cKDTree sampling'),
    'spatial.observations': dict(kind='source',backend='Explicit retained-observation matrix; held labels absent'),
    'spatial.auxiliary': dict(kind='source',backend='Explicit independently unit-labelled covariate sampling'),
    'spatial.idw': dict(kind='estimate',backend='fusion.point_weights; same distance kernel as fusion.idw'),
    'spatial.aux_affine': dict(kind='fit',backend='quality_aware.fit_affine_frames, retained observations only'),
    'spatial.affine': dict(kind='fit',backend='quality_aware.fit_affine_frames'),
    'spatial.ratio': dict(kind='fit',backend='NumPy retained-pair ratio of sums'),
    'spatial.residual_idw': dict(kind='combine',backend='fusion.residual_difference / residual_addition + metric IDW'),
    'spatial.blend': dict(kind='combine',backend='declared convex source weighting; not inverse-uncertainty weighting'),
}


def graph_components(registry):
    """Existing task operations and their actual named-port arities, no rankings."""
    from ecodata.algorithms.spatial_registry_binding import operations
    components={
        'spatial.idw':dict(source_arity=1,parameter_domains=dict(power=[1.,2.,3.])),
        'spatial.affine':dict(source_arity=2,parameter_domains={}),
        'spatial.ratio':dict(source_arity=2,parameter_domains={}),
        'spatial.residual_idw':dict(source_arity=2,parameter_domains=dict(power=[1.,2.,3.])),
        'spatial.blend':dict(source_arity=2,parameter_domains=dict(weight=[.25,.5,.75])),
        **{identifier:dict(source_arity=1,parameter_domains={}) for identifier in operations(registry)}}
    for identifier,contract in components.items():
        names=(['left','right'] if identifier=='spatial.blend' else ['field','observations']
            if identifier in {'spatial.affine','spatial.ratio','spatial.residual_idw'} else ['observations'])
        contract.update(input_ports={name:'spatial_conditioning_matrix' if name=='observations' else 'spatial_estimate_pair' for name in names},
            output_port='spatial_estimate_pair',target_geometry='Actual paired query and retained positions bound by this task',
            applicability='Actual retained inputs and additional resource requirements checked at execution')
    return components


def blend_fields(left, right, weight):
    return weight * left + (1 - weight) * right


def applies(parsed, request):
    if request.get('task') not in {'spatial', 'multisource'} or request.get('adapter'):
        return False
    if request.get('processing_goal') == 'spatial_fusion':
        return True
    if any(role == 'observations' for role, _ in parsed):
        return True
    variable = request.get('variable')
    return any(role == 'primary' and variable in ds and 'time' in ds[variable].dims
               and ds[variable].dims != ('time','lat','lon') for role, ds in parsed)


def positions(ds):
    if not ds.attrs.get('crs') or not CRS.from_user_input(ds.attrs['crs']).equals(CRS.from_epsg(4326)):
        raise ValueError('Spatial fusion currently requires explicit EPSG:4326; project coordinates first')
    if not {'lat','lon'} <= set(ds.coords):
        raise ValueError('Explicit spatial coordinates required')
    if ds.lat.dims == ('lat',) and ds.lon.dims == ('lon',):
        x,y = np.meshgrid(ds.lon.values, ds.lat.values)
        return np.column_stack([x.ravel(),y.ravel()]), ('lat','lon'), (len(ds.lat),len(ds.lon))
    if ds.lat.ndim == ds.lon.ndim == 1 and ds.lat.dims == ds.lon.dims:
        return np.column_stack([ds.lon.values,ds.lat.values]), ds.lat.dims, ds.lat.shape
    raise ValueError('Rectilinear grid or paired locations required; curvilinear grids need an explicit adapter')


def _values(ds, variable, times):
    _, dims, shape = positions(ds)
    a = ds[variable]
    if a.dims == dims:
        return np.broadcast_to(a.values.reshape(1,-1), (len(times),int(np.prod(shape))))
    if a.dims != ('time',*dims):
        raise ValueError('Value axes must match declared time and spatial coordinates')
    if not pd.DatetimeIndex(times).isin(ds.time.values).all():
        if a.attrs.get('temporal_semantics')=='instantaneous':
            native=pd.DatetimeIndex(ds.time.values)
            if native.has_duplicates or not native.is_monotonic_increasing or len(native)<2:
                raise ValueError('Temporal interpolation needs unique increasing observation dates')
            # Linear sampling of an instantaneous covariate only. Interval totals
            # and means cannot enter this branch; no temporal extrapolation.
            return a.interp(time=times,method='linear').values.reshape(len(times),-1)
        raise ValueError('Source does not cover target dates; no implicit resampling of time intervals')
    return a.sel(time=times).values.reshape(len(times),-1)


def sample(ds, variable, times, query):
    """Sample covariates, explicitly retaining their spatial support."""
    xy, dims, _ = positions(ds)
    values = _values(ds,variable,times)
    if not np.isfinite(xy).all() or len(np.unique(xy,axis=0)) != len(xy):
        raise ValueError('Nonfinite or duplicate source locations')
    if ds[variable].attrs.get('spatial_semantics') == 'cell_mean':
        index = cKDTree(xy).query(query)[1]
        if not np.allclose(xy[index],query,rtol=0,atol=1e-9):
            raise ValueError('Changing cell-average support requires conservative remapping, not point sampling')
        return values[:,index]
    if dims == ('lat','lon'):
        a=xr.DataArray(values.reshape(len(times),len(ds.lat),len(ds.lon)),dims=('time','lat','lon'),
            coords=dict(time=times,lat=ds.lat,lon=ds.lon))
        return a.interp(lon=xr.DataArray(query[:,0],dims='query'),
                        lat=xr.DataArray(query[:,1],dims='query'),method='linear').values
    # This selects supplied locations, not unknown native pixel centres.
    distance,index = cKDTree(xy).query(query)
    inside = distance < 1e-9
    if len(xy)>=3 and np.linalg.matrix_rank(xy-xy.mean(axis=0))==2:
        inside |= Delaunay(xy).find_simplex(query)>=0
    result = values[:,index].copy()
    result[:,~inside] = np.nan
    return result


def mosaic_fields(fields, variable, times):
    """Same declared product pieces are a mosaic, not independent measurements.

    No inferred group from filenames. Unknown identity/processing, time mismatch
    and conflicting overlap stop this operation; no averaging resolves conflicts.
    """
    groups={};result=[];relations=[];order={ds.attrs['source_id']:i for i,ds in enumerate(fields)}
    for ds in fields:
        group=ds.attrs.get('mosaic_group')
        if group:groups.setdefault(group,[]).append(ds)
        else:result.append(ds)
    for group,members in groups.items():
        if len(members)==1:result.extend(members);continue
        if any(not ds.attrs.get('mosaic_evidence') or not ds.attrs.get('source_product') or not ds.attrs.get('processing_level') for ds in members):
            raise ValueError('Mosaic requires evidenced common product and processing identity')
        if len({(ds.attrs['source_product'],ds.attrs['processing_level']) for ds in members})!=1:
            raise ValueError('Mosaic product/processing identities conflict')
        points=[];columns=[];lookup={};overlap=0
        for ds in members:
            if ds[variable].attrs.get('spatial_semantics')!='grid_estimate' or 'time' not in ds or not np.array_equal(ds.time.values,times.values):
                raise ValueError('Mosaic needs common native dates and gridded-estimate support')
            xy,_,_=positions(ds);values=_values(ds,variable,times)
            for i,location in enumerate(xy):
                key=tuple(location)
                if key not in lookup:
                    lookup[key]=len(points);points.append(location);columns.append(values[:,i].copy())
                else:
                    j=lookup[key];a=columns[j];b=values[:,i];both=np.isfinite(a)&np.isfinite(b)
                    if not np.allclose(a[both],b[both],rtol=1e-12,atol=1e-12):
                        raise ValueError('Mosaic overlap conflicts; values cannot be silently averaged or overwritten')
                    a[~np.isfinite(a)]=b[~np.isfinite(a)];overlap+=1
        xy=np.asarray(points)
        members_ids=[ds.attrs['source_id'] for ds in members]
        identity='mosaic:'+sha256(json.dumps([group,members_ids]).encode()).hexdigest()[:20]
        ds=xr.Dataset({variable:(('time','point'),np.column_stack(columns))},
            coords=dict(time=times,point=np.arange(len(xy)),lat=('point',xy[:,1]),lon=('point',xy[:,0])),
            attrs=dict(members[0].attrs,source_id=identity,mosaic_members=json.dumps(members_ids)))
        ds[variable].attrs=dict(members[0][variable].attrs)
        if 'time_bounds' in members[0]:ds['time_bounds']=members[0].time_bounds
        result.append(ds)
        relations.append(dict(operation='same_product_spatial_mosaic',group=group,members=members_ids,
            output_source_id=identity,output_locations=len(xy),identical_overlap_locations=overlap,
            conflict_resolution='reject conflicting finite values; fill unknown from identical-product piece',
            evidence=[m.attrs['mosaic_evidence'] for m in members]))
    result.sort(key=lambda ds:min(order[sid] for sid in json.loads(ds.attrs['mosaic_members'])) if 'mosaic_members' in ds.attrs else order[ds.attrs['source_id']])
    return result,relations


@dataclass
class SpatialTask:
    fields: list
    auxiliary: list
    observations: pd.DataFrame
    template: xr.Dataset
    times: pd.DatetimeIndex
    request: dict
    audit: list
    self_reference: bool = False
    relationships: list = None
    evaluation_observations: pd.DataFrame = None
    reference_grid: xr.Dataset = None
    reference_operator: object = None
    target_alignment: list = None
    native_fields: list = None

    @classmethod
    def build(cls, parsed, request, audit):
        if request.get('semantic_type') != 'continuous' or not request.get('unit') or request['unit']=='UNKNOWN':
            raise ValueError('Continuous variable and confirmed unit required for spatial estimation')
        from ecodata.models.scientific_schema import compare_units, convert_matched_field, identity

        variable = request['variable']; fields=[]; auxiliary=[]; observations=[]; targets=[]; primary=None
        requested_quantity=identity(request.get('physical_variable',''),{})[0] or request.get('physical_variable')
        for role, ds in parsed:
            if role=='observations':
                frame=table_frame(ds).copy()
                if 'crs' not in frame and ds.attrs.get('crs'):frame['crs']=ds.attrs['crs']
                if {'unit','value','variable'} <= set(frame.columns):
                    frame=frame.reset_index(drop=True)
                    if not frame.unit.eq(request['unit']).all():
                        frame['value']=frame.value.astype(float)
                    for (name,source_unit), group in frame.groupby(['variable','unit'],sort=False):
                        quantity=identity(name,{})[0] or name
                        if requested_quantity and quantity!=requested_quantity:
                            raise ValueError('Observation physical variable conflict')
                        if source_unit!=request['unit']:
                            report=compare_units(requested_quantity or quantity,request['unit'],quantity,source_unit)
                            if report["status"] not in {"equivalent","convertible"}:
                                raise ValueError('Observation physical identity or unit mismatch')
                            frame.loc[group.index,'value']=group.value*report["scale"]+report["offset"]
                            frame.loc[group.index,'unit']=request['unit']
                        frame.loc[group.index,'variable']=quantity
                observations.append(frame)
            elif role=='target':
                targets.append(ds)
            elif role in {'primary','background'}:
                if variable not in ds:
                    raise ValueError('Same-variable source missing target; other variables must be auxiliary')
                physical_label=ds[variable].attrs.get('physical_variable')
                identity_label=identity(physical_label,{})[0] or physical_label
                if ds[variable].attrs.get('units') != request['unit']:
                    quantity=requested_quantity or identity_label
                    ds=ds.assign({variable:convert_matched_field(ds[variable],quantity,request['unit'],
                                                                 verified_quantity=identity_label)})
                if physical_label and physical_label!=identity_label:
                    ds=ds.assign({variable:ds[variable].assign_attrs(physical_variable=identity_label)})
                support = ds[variable].attrs.get('spatial_semantics')
                if support not in {'point','grid_estimate','cell_mean'}:
                    raise ValueError('Unknown spatial support or unsupported cell totals/flux')
                positions(ds)
                if role=='primary':
                    if primary is not None: raise ValueError('One primary source required')
                    primary=ds
                if support=='point':
                    if ds[variable].attrs.get('units')!=request['unit'] or ds[variable].attrs.get('data_kind') not in {None,'continuous'}:
                        raise ValueError('Point source unit or data kind conflicts with request')
                    xy,dims,_=positions(ds)
                    if 'time' not in ds[variable].dims: raise ValueError('Point observation dates required')
                    t=pd.DatetimeIndex(ds.time.values); a=_values(ds,variable,t)
                    if ds[variable].attrs.get('temporal_semantics')=='accumulation':
                        if ('time_bounds' not in ds or ds.time_bounds.shape!=(len(t),2)
                                or not np.array_equal(ds.time_bounds.values[:,0],t.values)
                                or not np.all(ds.time_bounds.values[:,1]-ds.time_bounds.values[:,0]==np.timedelta64(1,'D'))):
                            raise ValueError('Accumulated point observations require explicit matching daily bounds')
                    observations.append(pd.DataFrame(dict(time=np.repeat(t,len(xy)),
                        station_id=np.tile([f'{ds.attrs.get("source_id","point")}:{i}' for i in range(len(xy))],len(t)),
                        longitude=np.tile(xy[:,0],len(t)),latitude=np.tile(xy[:,1],len(t)),value=a.ravel(),
                        unit=request['unit'],variable=ds[variable].attrs.get('physical_variable'),split='train',
                        temporal_semantics=ds[variable].attrs.get('temporal_semantics'),crs='EPSG:4326',
                        interval_seconds=86400 if ds[variable].attrs.get('temporal_semantics')=='accumulation' and 'time_bounds' in ds else np.nan)))
                else: fields.append(ds)
            elif role=='auxiliary': auxiliary.append(ds)
            else: raise ValueError('Unsupported spatial-fusion resource role: '+role)
        if primary is None and not observations: raise ValueError('A spatial field or station observations required')
        if len(targets)>1: raise ValueError('Ambiguous target geometry')
        if (request.get('target_lat') is None)!=(request.get('target_lon') is None):
            raise ValueError('Both target grid axes are required; use a target dataset for paired positions')
        if targets: template=targets[0]
        elif request.get('target_lat') is not None and request.get('target_lon') is not None:
            template=xr.Dataset(coords=dict(lat=request['target_lat'],lon=request['target_lon']),attrs=dict(crs='EPSG:4326'))
        elif primary is not None: template=primary
        else: raise ValueError('Stations need an explicit target geometry')
        xy,_,_=positions(template)
        if not np.isfinite(xy).all() or (abs(xy[:,0])>180).any() or (abs(xy[:,1])>90).any():
            raise ValueError('Invalid target coordinates')
        if request.get('target_times') is not None: times=pd.DatetimeIndex(request['target_times'])
        elif primary is not None and 'time' in primary: times=pd.DatetimeIndex(primary.time.values)
        elif 'time' in template: times=pd.DatetimeIndex(template.time.values)
        else: times=pd.DatetimeIndex(sorted(pd.to_datetime(pd.concat(observations).time).unique()))
        if not len(times) or times.has_duplicates or not times.is_monotonic_increasing:
            raise ValueError('Unique increasing target dates required')
        physical=requested_quantity
        for ds in fields:
            a=ds[variable]; attrs=a.attrs
            if attrs.get('units')!=request['unit'] or not attrs.get('physical_variable') or not attrs.get('physical_definition'):
                raise ValueError('Same-variable sources require unit, physical variable and definition')
            physical=physical or attrs['physical_variable']
            if attrs['physical_variable']!=physical: raise ValueError('Physical variable conflict')
            if fields and attrs.get('physical_definition')!=fields[0][variable].attrs.get('physical_definition'):
                raise ValueError('Physical definition conflict')
            if request.get('temporal_semantics') and attrs.get('temporal_semantics')!=request['temporal_semantics']:
                raise ValueError('Temporal semantics conflict')
            values=_values(ds,variable,times)
            if np.isinf(values).any() or (request.get('nonnegative') and (values<0).any()):
                raise ValueError('Invalid source values; missing sentinels require documented input decoding')
            report=RawInputQualityGate().validate(ds,variable=variable,requires_spatial=True,expected_unit=request['unit'])
            if report['status']=='BLOCK':raise ValueError('Raw input quality gate blocked spatial source: '+str(report))
            if attrs.get('temporal_semantics')=='accumulation':
                if 'time_bounds' not in ds or ds.time_bounds.dims!=('time','bounds'):
                    raise ValueError('Accumulations require explicit interval bounds')
                b=ds.time_bounds.sel(time=times).values
                if not np.array_equal(b[:,0],times.values) or not np.all(b[:,1]-b[:,0]==np.timedelta64(1,'D')):
                    raise ValueError('Reviewed accumulation path requires matching daily intervals')
        native_fields=sorted(fields,key=lambda ds:0 if ds is primary else 1)
        reference_grid=None;reference_operator=None;target_alignment=[]
        if request.get('target_grid'):
            if observations or auxiliary or primary is None or any(d[variable].attrs.get('spatial_semantics')!='cell_mean' for d in fields):
                raise ValueError('Explicit cell-grid reconstruction currently requires same-variable cell-mean fields; station/auxiliary target operators must be declared separately')
            from ecodata.spatial.regridding import resolve_target_grid,GridRegridder,RegridConfig,require_target_grid,contained_cell_grid
            from ecodata.core.schemas import EcoDataBundle
            from ecodata.algorithms.raster_reconstruction_task import edges,overlap_weights
            from scipy import sparse
            target=resolve_target_grid(fields,request['target_grid'])
            if targets or request.get('target_lat') is not None:
                for axis in ('lat','lon'):
                    if not np.array_equal(template[axis],target[axis]):raise ValueError('Conflicting target grid declarations')
            template=target
            reference_grid=contained_cell_grid(primary,target)
            ay=overlap_weights(edges(target.lat),edges(reference_grid.lat),True)
            ax=overlap_weights(edges(target.lon),edges(reference_grid.lon))
            reference_operator=sparse.kron(ay,ax,format='csr')
            aligned=[];primary_index=fields.index(primary) if len(fields)==1 else next(i for i,d in enumerate(fields) if d is primary)
            for ds in fields:
                clipped=ds
                for axis in ('lat','lon'):
                    halo=abs(float(ds[axis][1]-ds[axis][0]))*2
                    clipped=clipped.sel({axis:slice(float(target[axis][0])-halo,float(target[axis][-1])+halo)})
                result=GridRegridder(RegridConfig('conservative',tuple(target.lat.values),tuple(target.lon.values),variable)).apply(EcoDataBundle(clipped))
                out=result.data.dataset
                if 'time_bounds' in ds:out['time_bounds']=ds.time_bounds
                require_target_grid(out,target,variable)
                aligned.append(out);target_alignment.append(dict(operation='spatial.regrid.conservative',status='PASS',source_id=ds.attrs['source_id'],record=result.record.as_dict()))
            fields=aligned;primary=fields[primary_index]
            request=dict(request,target_lat=target.lat.values.tolist(),target_lon=target.lon.values.tolist(),
                completed_preparation=[*request.get('completed_preparation',[]),*target_alignment])
        self_reference=not observations
        if self_reference:
            if primary is None: raise ValueError('No visible validation anchors')
            reference=reference_grid if reference_grid is not None else primary
            pxy,_,_=positions(reference); values=_values(reference,variable,times)
            observations=[pd.DataFrame(dict(time=np.repeat(times,len(pxy)),station_id=np.tile(np.arange(len(pxy)).astype(str),len(times)),
                longitude=np.tile(pxy[:,0],len(times)),latitude=np.tile(pxy[:,1],len(times)),value=values.ravel(),
                unit=request['unit'],variable=physical,split='train',temporal_semantics=primary[variable].attrs.get('temporal_semantics')))]
        obs=pd.concat(observations,ignore_index=True).copy()
        required={'time','station_id','longitude','latitude','value','unit','variable','split','temporal_semantics'}
        if not required<=set(obs) or not obs.split.isin(['train','test']).all() or not obs.split.eq('train').any():
            raise ValueError('Declared training observations required; only train/test splits are supported, test never enters fitting')
        obs.time=pd.to_datetime(obs.time,errors='raise'); obs.station_id=obs.station_id.astype(str)
        observation_crs=request.get('observation_crs',request.get('parameters',{}).get('observation_crs',request.get('crs')))
        if 'crs' in obs:
            if obs.crs.isna().any() or any(not CRS.from_user_input(c).equals(CRS.from_epsg(4326)) for c in obs.crs.unique()):
                raise ValueError('Station coordinates need explicit EPSG:4326')
        elif not observation_crs or not CRS.from_user_input(observation_crs).equals(CRS.from_epsg(4326)):
            if not self_reference:raise ValueError('Observation CRS unknown; confirm before fusion')
        if obs.duplicated(['station_id','time']).any(): raise ValueError('Duplicate station/date')
        if obs[['longitude','latitude']].isna().any().any() or obs.groupby('station_id')[['longitude','latitude']].nunique().gt(1).any().any():
            raise ValueError('Station identity must bind stable finite coordinates')
        station_xy=obs.drop_duplicates('station_id')[['longitude','latitude']].to_numpy(float)
        if (not np.isfinite(station_xy).all() or len(np.unique(station_xy,axis=0))!=len(station_xy)
                or (abs(station_xy[:,0])>180).any() or (abs(station_xy[:,1])>90).any()):
            raise ValueError('Invalid or duplicate station coordinates')
        if not obs.unit.eq(request['unit']).all() or obs.variable.isna().any() or obs.variable.nunique()!=1:
            raise ValueError('Observation quantity/unit unknown or conflicting')
        physical=physical or obs.variable.iloc[0]
        if not obs.variable.eq(physical).all(): raise ValueError('Observation/background physical variable conflict')
        temporal=request.get('temporal_semantics') or (fields[0][variable].attrs.get('temporal_semantics') if fields else obs.temporal_semantics.iloc[0])
        if temporal not in {'instantaneous','accumulation'}:
            raise ValueError('Temporal support '+repr(temporal)+' is unsupported or UNKNOWN: spatial estimation currently accepts instantaneous values or explicitly bounded daily accumulations; no guessed interval/window')
        if not obs.temporal_semantics.eq(temporal).all() or any(ds[variable].attrs.get('temporal_semantics')!=temporal for ds in fields):
            raise ValueError('Source temporal semantics differ')
        request=dict(request,temporal_semantics=temporal)
        if request.get('temporal_semantics') and not obs.temporal_semantics.eq(request['temporal_semantics']).all():
            raise ValueError('Observation temporal support conflicts with task')
        if request.get('temporal_semantics')=='accumulation' and not self_reference:
            if 'interval_seconds' not in obs or not obs.interval_seconds.eq(86400).all():
                raise ValueError('Daily accumulated observations need declared 86400-second intervals')
        if np.isinf(obs.value).any() or (request.get('nonnegative') and (obs.value<0).any()):
            raise ValueError('Invalid observation values')
        evaluation=obs.loc[obs.split.eq('test')].copy()
        obs=obs.loc[obs.split.eq('train')].copy()
        if not evaluation.empty:
            if set(evaluation.station_id)&set(obs.station_id):
                raise ValueError('Final test locations must be disjoint from training locations')
            if not set(evaluation.time)<=set(times):
                raise ValueError('Final test dates outside application time support')
        if obs.station_id.nunique()<3: raise ValueError('At least three spatial locations required for leave-location-out validation')
        for ds in auxiliary:
            positions(ds)
            if any(not a.attrs.get('units') for n,a in ds.data_vars.items() if n!='time_bounds'):
                raise ValueError('Auxiliary quantities require their own units')
        fields.sort(key=lambda ds:0 if ds is primary else 1)
        identities=[ds.attrs.get('source_id') for ds in fields]
        if any(not sid for sid in identities) or len(set(identities))!=len(identities):
            raise ValueError('Spatial fields require distinct stable source identities')
        if self_reference:
            root=primary.attrs.get('source_id')
            for ds in [*fields[1:],*auxiliary]:
                lineage=ds.attrs.get('derived_from',[])
                if isinstance(lineage,str):
                    try:lineage=json.loads(lineage)
                    except json.JSONDecodeError:lineage=[lineage]
                if root in lineage or ds.attrs.get('source_id')==root:
                    raise ValueError('Derived reference alias requires replay after masking; cannot leak held anchors')
        request=dict(request,physical_variable=physical)
        fields,relationships=mosaic_fields(fields,variable,times)
        return cls(fields,auxiliary,obs,template,times,request,audit,self_reference,relationships,evaluation,
                   reference_grid,reference_operator,target_alignment,native_fields)

    def query_grid(self,query):
        return positions(self.template)[0] if self.reference_grid is not None else query

    def observe_grid(self,values,query):
        if self.reference_grid is None:return values
        xy,_,_=positions(self.reference_grid)
        index=cKDTree(xy).query(query)[1]
        if not np.allclose(xy[index],query,rtol=0,atol=1e-9):raise ValueError('Evaluation locations must be original reference cells')
        if values.shape[1]!=self.reference_operator.shape[1]:raise ValueError('Candidate did not execute the complete target grid')
        return (self.reference_operator@values.T).T[:,index]

    def cohort(self, train_ids, query, times):
        stations=self.observations.drop_duplicates('station_id').set_index('station_id').loc[train_ids]
        sxy=stations[['longitude','latitude']].to_numpy(float)
        y=self.observations.pivot(index='time',columns='station_id',values='value').reindex(index=times,columns=train_ids).to_numpy(float)
        allxy=np.concatenate([query,sxy]); sources=[]; aux=[]
        for ds in self.fields:
            if self.self_reference and ds is self.fields[0]:
                # Replay the primary after removing held locations, including all dates.
                original=self.reference_grid if self.reference_grid is not None else ds
                xy,_,_=positions(original); visible=np.isclose(xy[:,None,:],sxy[None,:,:],atol=1e-9,rtol=0).all(axis=2).any(axis=1)
                vals=_values(original,self.request['variable'],times)[:,visible]
                sources.append(weighted_values(vals,distance_weights(xy[visible],allxy)))
            elif self.reference_grid is not None:
                targetxy=positions(self.template)[0]
                if not np.array_equal(query,targetxy):raise ValueError('Candidate query is not the declared target grid')
                fine=_values(ds,self.request['variable'],times)
                sources.append(np.concatenate([fine,self.observe_grid(fine,sxy)],axis=1))
            else: sources.append(sample(ds,self.request['variable'],times,allxy))
        for ds in self.auxiliary:
            for name in ds.data_vars:
                if name!='time_bounds': aux.append((name,sample(ds,name,times,allxy),ds[name].attrs['units']))
        return dict(y=y,query_count=len(query),sources=sources,aux=aux,sxy=sxy,allxy=allxy)


class SpatialSearch:
    def __init__(self,task,folder,budget,seed,model=None,progress=None):
        self.progress=progress
        self.task=task; self.folder=Path(folder); self.request=task.request; self.rng=np.random.default_rng(seed)
        self.seed=seed
        self.budget=budget
        from ecodata.algorithms.platform_search import platform_registry
        from ecodata.algorithms.spatial_registry_binding import operations
        self.registry=platform_registry()
        self.registry_operations=operations(self.registry) if self.request.get('registry_expansion',True) else {}
        self.components=dict(COMPONENTS,**{identifier:dict(kind='estimate',backend=spec.backend)
            for identifier,spec in self.registry_operations.items()})
        self.account=SearchAccount(budget['objective_limit'],budget['wall_seconds'],total_calls=budget.get('total_call_limit'))
        # Selected graph + fixed single-source comparator require a protected
        # post-selection allowance, in addition to final product deployment.
        self.account.protect_deployment(96,min(30.,budget['wall_seconds']*.2))
        self.account.configure_deployment(32,min(30.,budget['wall_seconds']*.2))
        declaration=policy(self.request)
        if not declaration.get('calibration_scope_match',False) and not self.request.get('search_stopping'):
            declaration=dict(enabled=False,evidence='Existing stopping thresholds not calibrated for this spatial task; numerical/time budget applies')
        self.stopper=SearchStopper(declaration,np.finfo(float).eps)
        self.records={};self.rows=[];self.transitions=[];self.attempts=0;self.novel_executions=0;self.numerical_rounds=0
        self.recommendation_receipts=[];self.recommended_parents=[]
        self.promotion_baseline=None;self.promotion_support=False;self.folds=[];self.executors=[];self.state_seen={}
        meta=task.observations.drop_duplicates('station_id').sort_values(['longitude','latitude','station_id'])
        explicit=task.evaluation_observations is not None and not task.evaluation_observations.empty
        self.location_protocol=split_locations(meta,seed,explicit_test=explicit)
        selection_ids=self.location_protocol['selection_ids']
        groups=[held for _,held in self.location_protocol['folds']]
        # Last date block is confirmation only and cannot influence search/stop.
        from ecodata.algorithms.spatial_selection_protocol import split_dates
        self.search_times,self.confirm_times,self.date_protocol=split_dates(task.times)
        self.groups=groups
        if 'confirmation_times' in self.request:
            requested=pd.DatetimeIndex(self.request['confirmation_times'])
            if requested.hasnans or requested.has_duplicates:
                raise ValueError('Invalid pre-reserved confirmation times')
            self.confirm_times=task.times.intersection(requested)
            self.search_times=task.times.difference(self.confirm_times)
            self.date_protocol.update(version='explicit_reserved_dates',
                selection_times=[str(t) for t in self.search_times],confirmation_times=[str(t) for t in self.confirm_times],
                selection_dates=int(self.search_times.normalize().nunique()),confirmation_dates=int(self.confirm_times.normalize().nunique()),
                sufficient=self.search_times.normalize().nunique()>=3 and self.confirm_times.normalize().nunique()>=3,
                date_overlap=bool(len(self.search_times.normalize().intersection(self.confirm_times.normalize()))))
            reservation=self.request.get('recommendation_isolation',{})
            if reservation.get('status')=='RESERVED':
                previous=pd.DatetimeIndex(reservation['recommendation_review_times'])
                if len(previous.intersection(self.confirm_times)):
                    raise ValueError('Recommendation and mutation confirmation units overlap')
                if len(self.confirm_times)!=len(requested):
                    raise ValueError('Reserved confirmation times are absent from actual search input')
        # Outer test values cannot affect even tie resolution or stopping.
        visible=task.observations[task.observations.station_id.isin(selection_ids)&task.observations.time.isin(self.search_times)]
        finite=visible.value.to_numpy(float);finite=finite[np.isfinite(finite)]
        if not len(finite):raise ValueError('No finite selection labels')
        self.resolution=np.finfo(float).eps*max(1,len(finite))*max(1.,float(np.max(abs(finite))))
        for retained,held in self.location_protocol['folds']:
            query=meta[meta.station_id.isin(held)][['longitude','latitude']].to_numpy(float)
            reference=task.observations.pivot(index='time',columns='station_id',values='value').reindex(index=self.search_times,columns=held).to_numpy(float)
            cohort=task.cohort(retained,task.query_grid(query),self.search_times)
            if not np.isfinite(reference).any(): raise ValueError('Spatial holdout has no finite labels')
            self.folds.append(dict(held=list(held),train=retained,query=query,reference=reference,cohort=cohort,
                support_sha256=fixed_support(reference,held,self.search_times)))
            self.executors.append(self.executor(cohort,'search'))
        self.reconstruction=None;self.reconstruction_predictions={}
        from ecodata.evaluation.fusion_protocol import prepare
        reviewed_native=bool(task.native_fields) and all(d[self.request['variable']].attrs.get('spatial_semantics')=='cell_mean' for d in task.native_fields)
        evaluation_policy=self.request.get('evaluation_policy','evidence_adaptive_v1' if task.reference_grid is not None or not task.self_reference or reviewed_native else 'legacy_primary_grid_v1')
        if evaluation_policy not in {'evidence_adaptive_v1','legacy_primary_grid_v1'}:
            raise ValueError('Unknown spatial evaluation policy')
        self.evaluation_plan,self.reconstruction=prepare(task,self.search_times,seed) if evaluation_policy=='evidence_adaptive_v1' else (
            dict(version='legacy_primary_grid_v1',mode='PRIMARY_GRID_RECONSTRUCTION_PROXY' if task.self_reference else 'OBSERVATION_HOLDOUT',
                 accuracy_established=False,selection_meaning='Legacy held-location proxy; not fine-scale accuracy',source_bonus=0),None)
        if self.reconstruction is not None:
            self.reconstruction_executor=self.executor(self.reconstruction['cohort'],'search')
            self.resolution=np.finfo(float).eps
        self.review_axes=['space','time'] if len(task.times.normalize().unique())>=6 or 'confirmation_times' in self.request else ['space']
        if self.progress:
            self.progress('评价依据：'+self.evaluation_plan['mode']+'；'+self.evaluation_plan['selection_meaning'])
        save(self.folder/'evaluation_protocol.json',self.evaluation_plan)
        self.leaves=[self.intern('spatial.source',[],dict(index=i)) for i in range(len(task.fields))]
        self.leaves += [self.intern('spatial.idw',[],dict(power=p)) for p in (1.,2.,3.)]
        self.leaves += [self.intern(identifier,[],{}) for identifier in self.registry_operations]
        if self.request.get('allow_training'):
            for i in range(len(self.folds[0]['cohort']['aux'])):self.leaves.append(self.intern('spatial.aux_affine',[],dict(index=i)))
        self.portraits=None
        if self.request.get('spatial_soft_guidance',False):
            if model is None or not model.get('state_policy'):
                raise ValueError('Spatial soft guidance requires an explicit registered model portrait policy')
            from ecodata.algorithms.spatial_portraits import SpatialPortraits
            self.portraits=SpatialPortraits(self,model)
        self.candidate_pool=self.request.get('spatial_candidate_pool',1)
        if type(self.candidate_pool) is not int or not 1<=self.candidate_pool<=16:
            raise ValueError('Spatial candidate pool must be an integer in 1..16')

    def candidate_sources(self,key):
        identities=set()
        for node in self.nodes(key):
            if node.operation=='spatial.source':
                identities.add('field:'+self.task.fields[node.parameters['index']].attrs['source_id'])
            elif node.operation=='spatial.observations':
                identities.add('field:'+self.task.fields[0].attrs['source_id'] if self.task.self_reference else 'conditioning_observations')
            elif node.operation=='spatial.auxiliary':
                identities.add('auxiliary:'+str(node.parameters['index']))
        return sorted(identities)

    def apply_prepared_recommendation(self):
        """Present only the real retained cohort, never held labels or raw aliases."""
        from ecodata.algorithms.registered_graph import has_recommender,apply_prepared_recommendation
        if not has_recommender():return
        from ecodata.algorithms.contracts import AlgorithmContext
        from ecodata.core.schemas import EcoDataBundle
        cohort=self.folds[0]['cohort'];xy=cohort['allxy'];sources=[]
        for original,values in zip(self.task.fields,cohort['sources']):
            dataset=xr.Dataset({self.request['variable']:(('time','point'),values)},
                coords=dict(time=self.search_times,lat=('point',xy[:,1]),lon=('point',xy[:,0])),
                attrs=dict(crs='EPSG:4326',source_id=original.attrs['source_id'],
                    feature_scope='Actual first retained cohort preparation; held labels excluded'))
            dataset[self.request['variable']].attrs=dict(original[self.request['variable']].attrs)
            dataset[self.request['variable']].attrs.update(spatial_semantics='point',
                support_evidence='Prepared field estimates sampled at paired positions; not original cell means or measured station labels')
            sources.append(EcoDataBundle(dataset))
        times=len(self.search_times);stations=len(cohort['sxy'])
        observations=pd.DataFrame(dict(time=np.repeat(self.search_times,stations),
            longitude=np.tile(cohort['sxy'][:,0],times),latitude=np.tile(cohort['sxy'][:,1],times),
            value=cohort['y'].ravel(),unit=self.request['unit'],split='train'))
        context=AlgorithmContext(tuple(sources),self.request['variable'],'continuous',observations=observations,
            parameters=dict(paired_target_positions=xy.tolist()),
            model_constraints=dict(unit=self.request['unit'],nonnegative=self.request.get('nonnegative',False)))
        proposal_request=dict(self.request)
        proposal_request.setdefault('operation','spatial_estimation')
        updated,seconds=apply_prepared_recommendation(self.registry,context,proposal_request,self.folder,
            components=graph_components(self.registry),resources=['observations'])
        self.request.clear();self.request.update(updated)
        self.account.started+=seconds

    def comparison_seeds(self):
        # These are scored candidates, never a forced fusion/final weighting.
        # Keep the existing random driver and component implementations intact.
        sources=[k for k in self.leaves if self.records[k]['operation']=='spatial.source']
        seeds=[]
        for source in sources:
            k=self.intern('spatial.residual_idw',[source],dict(power=2.))
            if len(self.candidate_sources(k))>1:seeds.append(k)
        for left,right in zip(sources,sources[1:]):
            seeds.append(self.intern('spatial.blend',[left,right],dict(weight=.5)))
        return list(dict.fromkeys(seeds))

    def validate_recommendations(self):
        """Compile ordered recipes into real typed parent edges, then score them."""
        from ecodata.algorithms.registered_graph import validate
        for proposal in self.request.get('recommended_graphs',[]):
            receipt=dict(recommendation=proposal,status='REJECTED',keys=[])
            try:
                if proposal.get('scope',{}).get('variable')!=self.request['variable']:
                    raise ValueError('Recommended graph variable scope mismatch')
                sources=[k for k in self.leaves if self.records[k]['operation']=='spatial.source']
                graph=validate(proposal['graph'],self.registry,len(sources),components=graph_components(self.registry),resources=['observations'])
                bindings={f'source:{i}':key for i,key in enumerate(sources)}
                bindings['observations']=self.intern('spatial.observations',[],{})
                for node in graph['nodes']:
                    operation=node['algorithm']
                    if operation not in self.components:raise ValueError('Algorithm needs a different actual output port binding: '+operation)
                    if len(node['parents'])!=len(self.input_names(operation)):raise ValueError('Task port arity mismatch')
                    parameters=node['parameters']
                    if operation in {'spatial.idw','spatial.residual_idw'}:parameters={'power':2.,**parameters}
                    if operation=='spatial.blend':parameters={'weight':.5,**parameters}
                    bindings[node['id']]=self.intern(operation,[bindings[p] for p in node['parents']],parameters)
                key=bindings[graph['output']]
                self.execute(key,'recommended_graph')
                row=next((r for r in self.rows if r['key']==key),None)
                receipt.update(keys=[key],status=row['status'] if row else 'BUDGET_STOP',error=row.get('error') if row else None)
                if row and row['status']=='PASS':self.recommended_parents.append(key)
            except (ValueError,KeyError,TypeError) as exc:receipt['error']=str(exc)
            self.recommendation_receipts.append(receipt)
        for recipe in self.request.get('recommended_recipes',[]):
            receipt=dict(recipe=recipe,status='REJECTED',keys=[])
            try:
                current=None
                for operation in recipe['steps']:
                    if operation in {'engineering.port_binding','engineering.precipitation_adaptation'}:
                        if current is not None:raise ValueError('Engineering preparation must precede numerical nodes')
                        evidence=[e for e in self.request.get('completed_preparation',[])
                                  if e.get('operation')==operation and e.get('status')=='PASS']
                        if not evidence:raise ValueError('Engineering preparation has not actually passed: '+operation)
                        receipt.setdefault('preparation',[]).extend(evidence)
                        continue
                    if operation=='spatial.regrid.conservative':
                        if current is not None:raise ValueError('Grid preparation must precede spatial estimation')
                        evidence=self.request.get('completed_preparation',[])
                        if not any(e.get('operation')==operation and e.get('status')=='PASS' for e in evidence):
                            raise ValueError('Recommended conservative regrid has not actually been executed')
                        receipt['preparation']=evidence
                        continue
                    if operation=='spatial.blend':
                        if len(self.task.fields)!=2:raise ValueError('Recommended blend requires exactly two bound same-variable fields')
                        left=current or self.leaves[0]
                        current=self.intern(operation,[left,self.leaves[1]],dict(weight=.5))
                    elif operation=='spatial.idw':
                        if current is not None:raise ValueError('IDW does not consume an upstream estimated field')
                        current=self.intern(operation,[],dict(power=2.))
                    elif operation in {'spatial.residual_idw','spatial.affine','spatial.ratio'}:
                        if operation!='spatial.residual_idw' and not self.request.get('allow_training'):
                            raise ValueError('Recommended fitted operation requires explicit training permission')
                        if current is None:
                            if not self.task.fields:raise ValueError('Recommended correction needs an actual background parent')
                            current=self.leaves[0]
                        current=self.intern(operation,[current],dict(power=2.) if operation=='spatial.residual_idw' else {})
                    else:raise ValueError('Unsupported recommended operation for spatial ports: '+operation)
                if current is None:
                    receipt['status']='PREPARATION_ONLY'
                else:
                    for node in self.nodes(current):
                        if len(node.inputs)!=len(self.input_names(node.operation)):
                            raise ValueError('Recommended graph has missing required parent ports')
                    cached=any(r['key']==current for r in self.rows)
                    self.execute(current,'recommended_initial')
                    row=next((r for r in self.rows if r['key']==current),None)
                    receipt.update(keys=[current],cache_hit=cached,status=row['status'] if row else 'NOT_EXECUTED_BUDGET',
                        metrics=row.get('metrics') if row else None,error=row.get('error') if row else None)
                    if row and row['status']=='PASS':
                        self.recommended_parents.append(current)
            except (ValueError,KeyError,IndexError) as exc:receipt['error']=str(exc)
            self.recommendation_receipts.append(receipt)

    def recommendation_parent(self):
        while self.recommended_parents:
            key=self.recommended_parents.pop(0)
            return key
        # Measured structure tradeoffs are actual search parents, not log-only
        # alternatives. Visit each once; the existing proposal/call budget holds.
        frontier=self.reference_frontier()
        visited=getattr(self,'structure_parents_used',set())
        for key in frontier.get('keys',[]):
            if key!=self.best()['key'] and key not in visited:
                visited.add(key);self.structure_parents_used=visited
                return key
        return None

    def reference_frontier(self):
        from ecodata.algorithms.reliable_selection import reference_frontier
        mode=self.request.get('spatial_objective_mode','reference_pareto_v1')
        if mode=='rmse_v1':return dict(status='EXPLICIT_RMSE_ONLY',keys=[])
        if mode!='reference_pareto_v1':raise ValueError('Unknown spatial objective mode')
        return reference_frontier(self.rows)

    def recommendation_mutation_family(self,key):
        return 'recommendation_parameters' if self.records[key]['operation'] in {'spatial.blend','spatial.idw','spatial.residual_idw'} else 'structure'

    def comparison_report(self,best):
        families={}
        for name in ['SINGLE_SOURCE','MULTISOURCE']:
            rows=[r for r in self.rows if r.get('source_family')==name]
            passed=[r for r in rows if r['status']=='PASS']
            value=min((r['metrics']['objective'] for r in passed),default=None)
            winner=min((r for r in passed if r['metrics']['objective']<=value+self.resolution),
                       key=lambda r:(r['algorithm_nodes'],r['key'])) if passed else None
            families[name]=dict(evaluated=len(rows),passed=len(passed),
                best_key=winner['key'] if winner else None,metrics=winner['metrics'] if winner else None,
                coarse_reference_diagnostic=winner.get('coarse_reference_diagnostic') if winner else None,
                blocked_validation=winner.get('blocked_validation') if winner else None)
        intermediate_scores={
            name: (row['metrics'] or {}).get('intermediate_scale', dict(status='NOT_EVALUATED'))
            for name,row in families.items()
        }
        single=intermediate_scores['SINGLE_SOURCE']
        multi=intermediate_scores['MULTISOURCE']
        intermediate_comparison=dict(status='NOT_COMPARABLE', scores=intermediate_scores)
        if single['status']==multi['status']=='PASS' and single['support_sha256']==multi['support_sha256']:
            intermediate_comparison.update(status='PASS', count=single['count'],
                RMSE_change=multi['RMSE']-single['RMSE'])
        ablations=[]
        for source in ['field:'+d.attrs['source_id'] for d in self.task.fields]:
            rows=[r for r in self.rows if r['status']=='PASS' and source not in r['source_identities']]
            winner=min(rows,key=lambda r:(r['metrics']['objective'],r['algorithm_nodes'],r['key'])) if rows else None
            ablations.append(dict(omitted_source=source,status='PASS' if winner else 'NOT_EVALUATED',
                best_key=winner['key'] if winner else None,metrics=winner['metrics'] if winner else None,
                scope='Best among actually evaluated graphs excluding this source; not exhaustive'))
        paired=None
        if self.reconstruction is not None and all(f['best_key'] for f in families.values()):
            from ecodata.evaluation.fusion_protocol import paired_report
            r=self.reconstruction
            paired=paired_report(self.reconstruction_predictions[families['SINGLE_SOURCE']['best_key']],
                self.reconstruction_predictions[families['MULTISOURCE']['best_key']],r['reference'],r['selected'],r['split'])
        paired_blocks=None
        if all(f.get('blocked_validation') for f in families.values()):
            single={ (b['date'],b['region']):b for b in families['SINGLE_SOURCE']['blocked_validation']['blocks'] }
            multi={ (b['date'],b['region']):b for b in families['MULTISOURCE']['blocked_validation']['blocks'] }
            if single.keys()!=multi.keys():raise ValueError('Candidate date-region comparison queues differ')
            changes=[dict(date=k[0],region=k[1],RMSE_change=multi[k]['metrics']['RMSE']-single[k]['metrics']['RMSE']) for k in sorted(single)]
            paired_blocks=dict(blocks=changes,improved_blocks=sum(b['RMSE_change']<0 for b in changes),
                mean_RMSE_change=float(np.mean([b['RMSE_change'] for b in changes])),
                uncertainty='Descriptive paired blocks; adjacent dates/regions are not independent replicates')
        return dict(protocol='spatial_selection_assessment_v2',evaluation_protocol=self.evaluation_plan,
            decision_after_scoring=True,selection_metric=best['metrics'].get('objective_name','RMSE'),
            fixed_support_sha256=[f['support_sha256'] for f in self.folds],
            families=families,intermediate_scale=intermediate_comparison,selected_key=best['key'],selected_family=best['source_family'],
            comparison_status='BOTH_FAMILIES_SCORED' if all(f['passed'] for f in families.values()) else 'COMPARISON_INCOMPLETE_OR_SINGLE_SOURCE_ONLY',
            reference_basis='PRIMARY_GRID_RECONSTRUCTION_PROXY' if self.task.self_reference else 'HELD_STATION_OBSERVATIONS',
            selection_reference_basis=self.evaluation_plan['mode'],source_ablations=ablations,paired_selection=paired,paired_date_regions=paired_blocks,
            accuracy_improvement='NOT_ESTABLISHED',
            source_identity_scope='Declared source identities, not proof of statistical independence')

    def post_selection_assessment(self,keys):
        """Evaluate frozen winners only. Results never feed best()/stopper."""
        if self.reconstruction is not None:
            from ecodata.algorithms.spatial_selection_protocol import error_metrics
            r=self.reconstruction;index=r['test'];schemes={}
            times=self.search_times
            predictions=self.reconstruction_predictions
            traces={}
            if 'time' in self.review_axes:
                if not len(self.confirm_times):
                    return dict(status='NO_SEPARATE_DATE_BLOCK',used_for_selection=False,schemes={})
                from ecodata.evaluation.fusion_protocol import prepare
                _,outer=prepare(self.task,self.confirm_times,self.seed)
                if outer is None or not np.array_equal(outer['test'],index) or not np.array_equal(outer['query'],r['query']):
                    return dict(status='CONFIRMATION_SUPPORT_UNAVAILABLE',used_for_selection=False,schemes={})
                r=outer;times=self.confirm_times;predictions={}
                executor=self.executor(r['cohort'],'validation_audit')
                for key in set(keys.values())-{None}:
                    try:
                        executed=executor.run(self.nodes(key),key)
                        values=executed['outputs']['field'].value[:,:r['cohort']['query_count']]
                        if self.request.get('nonnegative'):values=np.maximum(values,0.)
                        if not np.isfinite(values[np.isfinite(r['reference'])]).all():
                            raise ValueError('Incomplete reserved confirmation output')
                        predictions[key]=values;traces[key]=executed['trace']
                    except (SearchBudgetExceeded,DAGError,ValueError) as exc:
                        traces[key]=dict(error=str(exc))
            for name,key in keys.items():
                if key is not None and key in predictions and len(index):
                    from ecodata.algorithms.spatial_selection_protocol import blocked_validation
                    schemes[name]=dict(status='PASS',key=key,metrics=error_metrics(predictions[key][:,index],r['reference'][:,index]),
                        blocked_validation=blocked_validation(predictions[key][:,index],r['reference'][:,index],times,index,r['split']['block_by_station']),
                        trace=traces.get(key))
                    from ecodata.evaluation.spatial_structure import paired_grid
                    schemes[name]['metrics']['spatial_structure']=paired_grid(predictions[key],r['reference'],r['query'],times,index,
                        self.request['unit'],threshold=self.request.get('event_threshold'))
                    from ecodata.evaluation.intermediate_scale import score as score_intermediate
                    schemes[name]['metrics']['intermediate_scale']=score_intermediate(
                        r.get('intermediate_grid'),predictions[key],r['reference'],index)
            return dict(status='PASS' if schemes else 'NO_SEPARATE_LOCATION_TEST',used_for_selection=False,
                source='FROZEN_REDUCED_SCALE_OUTER_BLOCKS',reference_basis=self.evaluation_plan['mode'],
                statistical_independence_verified=False,accuracy_improvement='NOT_ESTABLISHED',schemes=schemes,
                metric_backend='Shared error_metrics; cached genuine numerical predictions, no new fitted graphs',
                test_ids=index.tolist(),conditioning='Degraded primary only; fine test labels never used in fitting',
                review_support=dict(locations=list(map(str,index)),times=[str(t) for t in times],reference_key=sha256(r['reference'][:,index].tobytes()).hexdigest()))
        task=self.task;explicit=task.evaluation_observations
        if explicit is not None and not explicit.empty:
            frame=explicit;origin='EXPLICIT_TEST_LOCATIONS'
        else:
            frame=task.observations[task.observations.station_id.isin(self.location_protocol['test_ids'])]
            origin=self.location_protocol['reason']
        if 'time' in self.review_axes:frame=frame[frame.time.isin(self.confirm_times)]
        report=dict(used_for_selection=False,source=origin,metric_backend='xskillscore',
            reference_basis='PRIMARY_GRID_RECONSTRUCTION_PROXY' if task.self_reference else 'STATION_OBSERVATIONS_UPSTREAM_INDEPENDENCE_UNKNOWN',
            statistical_independence_verified=False,accuracy_improvement='NOT_ESTABLISHED')
        if frame.empty:return dict(report,status='NO_SEPARATE_LOCATION_TEST')
        meta=frame.drop_duplicates('station_id').sort_values(['longitude','latitude','station_id'])
        ids=meta.station_id.tolist();query=meta[['longitude','latitude']].to_numpy(float)
        times=pd.DatetimeIndex(sorted(frame.time.unique()))
        reference=frame.pivot(index='time',columns='station_id',values='value').reindex(index=times,columns=ids).to_numpy(float)
        cohort=task.cohort(self.location_protocol['selection_ids'],task.query_grid(query),times)
        report.update(test_station_ids=ids,conditioning_station_ids=self.location_protocol['selection_ids'],
            reference_sha256=sha256(reference.tobytes()).hexdigest(),support_sha256=fixed_support(reference,ids,times),
            schemes={},times=[str(t) for t in times])
        from ecodata.algorithms.spatial_selection_protocol import spatial_blocks,blocked_validation
        groups={str(i):int(group) for i,group in zip(ids,spatial_blocks(meta))}
        report['review_support']=dict(locations=list(map(str,ids)),times=[str(t) for t in times],reference_key=sha256(reference.tobytes()).hexdigest())
        executor=self.executor(cohort,'validation_audit')
        scored={}
        for name,key in keys.items():
            if key is None:continue
            if key in scored:
                report['schemes'][name]=dict(scored[key])
                continue
            try:
                executed=executor.run(self.nodes(key),key)
                prediction=executed['outputs']['field'].value[:,:cohort['query_count']]
                if self.request.get('nonnegative'):prediction=np.maximum(prediction,0.)
                prediction=task.observe_grid(prediction,query)
                bounds=self.request.get('parameters',{}).get('target_valid_range')
                mask=np.isfinite(reference)
                if bounds is not None and ((prediction[mask]<bounds[0])|(prediction[mask]>bounds[1])).any():
                    raise ValueError('Assessment prediction outside declared application range')
                report['schemes'][name]=dict(status='PASS',key=key,metrics=xskillscore_assessment(prediction,reference),trace=executed['trace'],
                    blocked_validation=blocked_validation(prediction,reference,times,ids,groups))
            except Exception as exc:
                report['schemes'][name]=dict(status='NOT_EVALUATED_RESOURCE_LIMIT' if isinstance(exc,SearchBudgetExceeded) or getattr(exc,'details',{}).get('budget_exhausted') else 'FAILED',key=key,error=str(exc))
            scored[key]=report['schemes'][name]
        report['status']='PASS' if report['schemes'] and all(s['status']=='PASS' for s in report['schemes'].values()) else 'INCOMPLETE'
        if report['status']=='PASS' and 'selected' in report['schemes'] and 'single_source_baseline' in report['schemes']:
            report['RMSE_reduction_vs_single_on_this_reference']=(report['schemes']['single_source_baseline']['metrics']['RMSE']-report['schemes']['selected']['metrics']['RMSE'])
        return report

    def intern(self,op,parents,parameters):
        parents=list(parents)
        if op in self.registry_operations and not parents:
            parents=[self.intern('spatial.observations',[],{})]
        if op in {'spatial.idw','spatial.aux_affine','spatial.affine','spatial.ratio','spatial.residual_idw'}:
            observations=self.intern('spatial.observations',[],{})
            if op=='spatial.idw' and not parents:parents=[observations]
            elif op=='spatial.aux_affine' and not parents:
                parents=[self.intern('spatial.auxiliary',[],dict(index=parameters['index'])),observations]
            elif op in {'spatial.affine','spatial.ratio','spatial.residual_idw'} and len(parents)==1:
                parents.append(observations)
        if op=='spatial.blend' and parents[0]==parents[1]:return parents[0]
        if op in {'spatial.affine','spatial.ratio'} and self.records.get(parents[0],{}).get('operation')==op:return parents[0]
        row=dict(operation=op,parents=list(parents),parameters=parameters)
        key=sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()[:20];self.records[key]=row
        return key

    def nodes(self,key):
        result=[];seen=set()
        def visit(k):
            if k in seen:return
            seen.add(k);r=self.records[k]
            for p in r['parents']:visit(p)
            names=self.input_names(r['operation'])
            result.append(Node(k,r['operation'],dict(zip(names,[(p,'field') for p in r['parents']])),r['parameters']))
        visit(key);return result

    def bind_graph(self,graph):
        visiting=set();rebound={}
        def bind(key):
            if key in rebound:return rebound[key]
            if key in visiting:raise ValueError('Cycle in spatial graph')
            visiting.add(key);row=graph['records'][key]
            if row['operation'] not in self.components:raise ValueError('Unknown spatial component')
            rebound[key]=self.intern(row['operation'],[bind(p) for p in row['parents']],row['parameters'])
            visiting.remove(key);return rebound[key]
        return bind(graph['key'])

    def input_names(self,op):
        if op in self.registry_operations:return ['observations']
        if op in {'spatial.source','spatial.observations','spatial.auxiliary'}:return []
        if op=='spatial.idw':return ['observations']
        if op=='spatial.aux_affine':return ['covariate','observations']
        if op=='spatial.blend':return ['left','right']
        return ['field','observations']

    def executor(self,cohort,phase):
        port=Port('spatial_estimate_pair',('units',)); nq=cohort['query_count']; allxy=cohort['allxy']; sxy=cohort['sxy']
        weight_cache={}
        def weights(power):
            if power not in weight_cache:weight_cache[power]=distance_weights(sxy,allxy,power)
            return weight_cache[power]
        def operation(op):
            def run(inputs,p,external):
                token=self.account.before(op,phase,p,requires_training=op in {'spatial.affine','spatial.ratio','spatial.aux_affine'})
                try:
                    detail={}
                    output_kind='spatial_estimate_pair'
                    y=inputs['observations'].value if 'observations' in inputs else None
                    if op in self.registry_operations:
                        from ecodata.algorithms.spatial_registry_binding import execute
                        result,detail=execute(self.registry.get(op),y,cohort,self.request)
                    elif op=='spatial.observations':
                        result=cohort['y'].copy();output_kind='spatial_conditioning_matrix'
                        detail.update(coordinates=sxy.tolist(),role='retained observations only')
                    elif op=='spatial.auxiliary':
                        result=cohort['aux'][p['index']][1].copy();output_kind='spatial_covariate_matrix'
                        detail.update(name=cohort['aux'][p['index']][0],units=cohort['aux'][p['index']][2],role='covariate; never a target observation')
                    elif op=='spatial.source': result=cohort['sources'][p['index']].copy()
                    elif op=='spatial.idw': result=weighted_values(y,weights(p['power']))
                    elif op=='spatial.aux_affine':
                        x=inputs['covariate'].value;slope,intercept,failed=fit_affine_frames(x[:,nq:],y)
                        result=slope*x+intercept
                        if failed: raise ValueError('Auxiliary affine response cannot be identified in every date; no invented coefficient')
                    elif op=='spatial.blend':
                        a=inputs['left'].value;b=inputs['right'].value;w=p['weight']
                        if not np.isfinite(w) or not 0<=w<=1:raise ValueError('Convex blend weight must be in [0,1]')
                        result=blend_fields(a,b,w)
                    else:
                        a=inputs['field'].value; station=a[:,nq:]
                        if op=='spatial.affine':
                            slope,intercept,failed=fit_affine_frames(station,y);result=slope*a+intercept
                            detail['unidentifiable_dates_identity']=failed
                            detail.update(coefficient_sha256=sha256(slope.tobytes()+intercept.tobytes()).hexdigest(),
                                slope_range=[float(slope.min()),float(slope.max())],intercept_range=[float(intercept.min()),float(intercept.max())])
                        elif op=='spatial.ratio':
                            ratio,ok=fit_ratio_frames(station,y);result=ratio*a
                            detail['unidentifiable_dates_identity']=int((~ok).sum())
                        elif op=='spatial.residual_idw':
                            delta=residual_difference(y,station);correction=weighted_values(delta,weights(p['power']))
                            result=residual_addition(a,correction)
                            # Explicit missing-background fallback, not a zero-rain assumption.
                            result=np.where(np.isfinite(a),result,weighted_values(y,weights(p['power'])))
                        else:raise ValueError('Unregistered spatial operation')
                    self.account.after(token)
                    metadata=dict(operation=op,units=self.request['unit'],**detail) if 'units' not in detail else dict(operation=op,**detail)
                    return dict(field=Artifact(output_kind,result,metadata))
                except Exception as exc:
                    self.account.after(token,error=exc);raise
            return run
        adapters={}
        observations=Port('spatial_conditioning_matrix',('units',));covariate=Port('spatial_covariate_matrix',('units',))
        for op in self.components:
            names=self.input_names(op)
            output=observations if op=='spatial.observations' else covariate if op=='spatial.auxiliary' else port
            ports={n:observations if n=='observations' else covariate if n=='covariate' else port for n in names}
            adapters[op,'run']=NodeAdapter(ports,{'field':output},operation(op),version=VERSION,
                decision_contract=dict(self.components[op],scope='paired spatial-estimate task component; not a new scientific method'))
        return NumericalDAGExecutor(adapters,version=VERSION)

    def execute(self,key,phase='initial'):
        if any(r['key']==key for r in self.rows): return
        nodes=self.nodes(key)
        if len(nodes)>32:return
        if not self.account.can_search(len(nodes)*(len(self.folds)+(self.reconstruction is not None)),self.account.reserve):return
        identities=self.candidate_sources(key)
        row=dict(key=key,phase=phase,algorithm_nodes=len(nodes),source_identities=identities,
            source_family='MULTISOURCE' if len(identities)>1 else 'SINGLE_SOURCE',
            fixed_support_sha256=[f['support_sha256'] for f in self.folds]+(
                [self.evaluation_plan['fixed_support_sha256']] if self.reconstruction is not None else []))
        if self.portraits is not None:row['candidate_prior']=self.portraits.score(key)
        predictions=[];labels=[];traces=[];blocked_parts=[]
        try:
            for fold,executor in zip(self.folds,self.executors):
                executed=executor.run(nodes,key);trace=executed['trace'];raw=executed['outputs']['field'].value
                if self.task.reference_grid is not None:
                    expected=(len(self.search_times),fold['cohort']['query_count']+len(fold['train']))
                    if raw.shape!=expected:raise ValueError('Candidate violates complete target-grid output shape')
                a=raw[:,:fold['cohort']['query_count']]
                if self.request.get('nonnegative'):a=np.maximum(a,0.)
                if self.task.reference_grid is not None:
                    if not np.isfinite(a).all():raise ValueError('Candidate has incomplete target-grid coverage')
                    bounds=self.request.get('parameters',{}).get('target_valid_range')
                    if bounds is not None and ((a<bounds[0])|(a>bounds[1])).any():raise ValueError('Candidate target grid violates declared physical bounds')
                    row['target_grid_check']=dict(status='PASS',shape=[len(self.search_times),self.task.template.sizes['lat'],self.task.template.sizes['lon']],
                        scoring='Area aggregation to original held reference cells; no upsampled truth')
                a=self.task.observe_grid(a,fold['query'])
                mask=np.isfinite(fold['reference'])
                if not np.isfinite(a[mask]).all():raise ValueError('Incomplete predictions on fixed held-location cohort')
                bounds=self.request.get('parameters',{}).get('target_valid_range')
                if bounds is not None and ((a[mask]<bounds[0])|(a[mask]>bounds[1])).any():
                    raise ValueError('Candidate outside declared application range')
                predictions.append(a[mask]);labels.append(fold['reference'][mask]);traces.append(trace)
                from ecodata.algorithms.spatial_selection_protocol import blocked_validation
                blocked_parts.append(blocked_validation(a,fold['reference'],self.search_times,fold['held'],
                    self.location_protocol['block_by_station']))
            values=np.concatenate(predictions);truth=np.concatenate(labels)
            coarse_error=values-truth
            row['coarse_reference_diagnostic']=dict(RMSE=float(np.sqrt(np.mean(coarse_error**2))),MAE=float(np.mean(abs(coarse_error))),
                Bias=float(coarse_error.mean()),count=len(coarse_error),used_for_selection=self.reconstruction is None,
                meaning='Held primary coarse product reconstruction, not independent truth')
            if self.reconstruction is not None:
                r=self.reconstruction;cohort=r['cohort']
                executed=self.reconstruction_executor.run(nodes,key)
                raw=executed['outputs']['field'].value
                if raw.shape!=(len(self.search_times),cohort['query_count']+len(cohort['sxy'])):
                    raise ValueError('Reduced-scale candidate output shape mismatch')
                predicted=raw[:,:cohort['query_count']]
                if self.request.get('nonnegative'):predicted=np.maximum(predicted,0.)
                mask=np.isfinite(r['reference'])
                if not np.isfinite(predicted[mask]).all():raise ValueError('Incomplete common reduced-scale evaluation support')
                row['reconstruction_trace']=executed['trace']
                self.reconstruction_predictions[key]=predicted
                values=predicted[:,r['selected']][np.isfinite(r['reference'][:,r['selected']])]
                truth=r['reference'][:,r['selected']][np.isfinite(r['reference'][:,r['selected']])]
                self.folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(self.folder/(key+'-evaluation.npz'),prediction=predicted,reference=r['reference'],
                    selection_ids=r['selected'],test_ids=r['test'],reference_coordinates=r['query'])
            digest=sha256(values.tobytes()).hexdigest()
            if digest in self.state_seen:
                row['same_validation_predictions_as']=self.state_seen[digest]
            else:self.state_seen[digest]=key
            error=values-truth;rmse=float(np.sqrt(np.mean(error**2)))
            row.update(status='PASS',evaluation_mode=self.evaluation_plan['mode'],
                metrics=dict(objective=rmse,RMSE=rmse,MAE=float(np.mean(abs(error))),Bias=float(error.mean()),
                             coverage=1.,failure_rate=0.,count=len(error)),traces=traces)
            from ecodata.algorithms.spatial_selection_protocol import blocked_validation
            selection_policy=self.request.get('selection_policy','date_region_v1')
            if selection_policy not in {'date_region_v1','pooled_rmse_v1'}:raise ValueError('Unknown selection policy')
            if self.reconstruction is not None:
                r=self.reconstruction
                report=blocked_validation(predicted[:,r['selected']],r['reference'][:,r['selected']],self.search_times,
                    r['selected'],r['split']['block_by_station'])
            else:
                blocks=[block for part in blocked_parts for block in part['blocks']]
                report=dict(objective=float(np.mean([b['metrics']['RMSE'] for b in blocks])),blocks=blocks,
                    objective_name='equal_date_region_RMSE',source_bonus=0,
                    date_count=len({b['date'] for b in blocks}),region_count=len({b['region'] for b in blocks}),
                    robustness_status='MULTIPLE_DATES_AND_REGIONS' if len({b['date'] for b in blocks})>=3 and len({b['region'] for b in blocks})>=2 else 'INSUFFICIENT_DATE_OR_REGION_REPLICATION')
            row['blocked_validation']=report
            if self.reconstruction is not None:
                from ecodata.evaluation.spatial_structure import paired_grid
                r=self.reconstruction
                row['metrics']['spatial_structure']=paired_grid(predicted,r['reference'],r['query'],self.search_times,r['selected'],self.request['unit'],
                    threshold=self.request.get('event_threshold'))
                from ecodata.evaluation.intermediate_scale import score as score_intermediate
                row['metrics']['intermediate_scale']=score_intermediate(
                    r.get('intermediate_grid'),predicted,r['reference'],r['selected'])
            row['metrics']['objective']=report['objective'] if selection_policy=='date_region_v1' else rmse
            row['metrics']['objective_name']='equal_date_region_RMSE' if selection_policy=='date_region_v1' else 'pooled_RMSE'
            row['reconstruction_reference_used_for_selection']=self.reconstruction is not None
        except SearchBudgetExceeded: return
        except Exception as exc:
            if getattr(exc,'details',{}).get('budget_exhausted'):return
            row.update(status='FAILED',error=str(exc))
        self.rows.append(row);self.novel_executions+=1
        if self.progress:
            best=self.best()
            stage={'initial':'初始候选','family_baseline':'单源/融合对照','mutation':'变异候选','confirmed_initial':'推荐初始图'}.get(phase,phase)
            route=' → '.join(n.operation for n in nodes)
            result=(f"RMSE {row['metrics']['RMSE']:.5g}，MAE {row['metrics']['MAE']:.5g}，样本 {row['metrics']['count']}"
                    if row['status']=='PASS' else '失败：'+row['error'])
            current=f"；当前最佳 RMSE {best['metrics']['RMSE']:.5g}" if best else '；暂无通过候选'
            self.progress(f"{stage} #{len(self.rows)} · {key} · {route} · {result}{current}")

    def best(self):
        rows=[r for r in self.rows if r['status']=='PASS']
        if not rows:return None
        value=min(r['metrics']['objective'] for r in rows)
        near=[r for r in rows if r['metrics']['objective']<=value+self.resolution]
        front=self.reference_frontier().get('keys',[])
        eligible=[r for r in near if r['key'] in front] or near
        return min(eligible,key=lambda r:(r['algorithm_nodes'],r['key']))

    def exhausted(self):
        return bool(self.stopper.reason or self.attempts>=self.budget['proposal_limit'] or self.account.exhausted()
                    or not self.account.can_search(1,self.account.reserve))

    def destroy(self,key,rng):return dict(reset=bool(rng.integers(2)))

    def candidate(self,current,family,rng,destruction=None):
        self.attempts+=1;row=self.records[current]
        if family=='recommendation_parameters':
            parameter='weight' if row['operation']=='spatial.blend' else 'power'
            domain=[.25,.5,.75] if parameter=='weight' else [1.,2.,3.]
            values=[v for v in domain if v!=row['parameters'][parameter]]
            return self.intern(row['operation'],row['parents'],dict(row['parameters'],**{parameter:float(rng.choice(values))}))
        if family in {'algorithm','source'} or (destruction and destruction['reset']):
            child=str(rng.choice(self.leaves))
        elif family=='parameters' and row['operation'] in {'spatial.idw','spatial.residual_idw','spatial.blend'}:
            p=dict(row['parameters']);p.update(weight=float(rng.choice([.25,.5,.75]))) if row['operation']=='spatial.blend' else p.update(power=float(rng.choice([1.,2.,3.])))
            child=self.intern(row['operation'],row['parents'],p)
        else:
            options=['spatial.residual_idw','spatial.blend','delete']
            if self.request.get('allow_training'):options+=['spatial.affine','spatial.ratio']
            op=str(rng.choice(options))
            anchor=str(rng.choice([current,*self.leaves]))
            if op=='delete':
                candidates=[p for p in row['parents'] if self.records[p]['operation'] not in {'spatial.observations','spatial.auxiliary'}]
                child=str(rng.choice(candidates)) if candidates else current
            elif op=='spatial.blend':child=self.intern(op,[anchor,str(rng.choice(self.leaves))],dict(weight=float(rng.choice([.25,.5,.75]))))
            else:child=self.intern(op,[anchor],dict(power=float(rng.choice([1.,2.,3.]))) if op=='spatial.residual_idw' else {})
        return child

    def propose(self,current,family,strategy,rng,destruction=None):
        choices=[]
        for _ in range(self.candidate_pool):
            if self.exhausted():break
            choices.append(self.candidate(current,family,rng,destruction))
        if not choices:return None
        random_child=choices[0];child=random_child
        prior=None
        if strategy=='portrait' and self.portraits is not None:
            child=min(choices,key=lambda k:(self.portraits.order_key(k),k))
            prior=self.portraits.score(child)
        before=len(self.rows);self.execute(child,'mutation')
        self.transitions.append(dict(parent=current,child=child,family=family,
            strategy=strategy if self.portraits is not None else 'random',
            actual_execution=len(self.rows)>before,candidates=choices,random_child=random_child,
            guidance_changed_choice=child!=random_child,candidate_prior=prior))
        return child

    def product(self,key):
        task=self.task;xy,dims,shape=positions(task.template)
        ids=task.observations.drop_duplicates('station_id').station_id.tolist()
        cohort=task.cohort(ids,xy,task.times)
        self.account.begin_deployment();executor=self.executor(cohort,'deploy')
        executed=executor.run(self.nodes(key),key);trace=executed['trace'];a=executed['outputs']['field'].value[:,:len(xy)]
        projected=int(np.sum(a<0)) if self.request.get('nonnegative') else 0
        if self.request.get('nonnegative'):a=np.maximum(a,0.)
        if not np.isfinite(a).all():raise ValueError('Final field incomplete; no silent source-only substitution')
        bounds=self.request.get('parameters',{}).get('target_valid_range')
        if bounds is not None and ((a<bounds[0])|(a>bounds[1])).any():
            raise ValueError('Final estimate outside declared application range')
        out=xr.Dataset(coords={n:task.template[n] for n in task.template.coords if n!='time' and 'time' not in task.template[n].dims})
        out=out.assign_coords(time=task.times)
        variable=self.request['variable'];out[variable]=(('time',*dims),a.reshape(len(task.times),*shape))
        out[variable].attrs.update(units=self.request['unit'],physical_variable=self.request['physical_variable'],
            physical_definition=task.fields[0][variable].attrs['physical_definition'] if task.fields else self.request['physical_variable'],
            temporal_semantics=self.request.get('temporal_semantics','instantaneous'),spatial_semantics='grid_estimate',
            data_kind='continuous',estimate_kind='spatial model estimate, not additional observed stations')
        stations=cohort['sxy']
        outside=np.ones(len(xy),bool)
        if len(stations)>=3 and np.linalg.matrix_rank(stations-stations.mean(axis=0))==2:
            outside=Delaunay(stations).find_simplex(xy)<0
        out['outside_station_hull']=(dims,outside.reshape(shape).astype('int8'))
        out.outside_station_hull.attrs.update(long_name='Outside conditioning-location convex hull; no accuracy guarantee',units='1')
        out['conditioning_location_count']=('time',np.isfinite(cohort['y']).sum(axis=1).astype('int32'))
        out.conditioning_location_count.attrs.update(units='1',long_name='Available input conditioning locations on each date')
        if self.request.get('temporal_semantics')=='accumulation':
            out['time_bounds']=(('time','bounds'),np.stack([task.times.values,(task.times+pd.Timedelta(days=1)).values],axis=1))
        out.attrs.update(crs='EPSG:4326',processing_stage='spatial_estimation',fusion_method=VERSION,
            selected_graph=key,uncertainty_status='UNKNOWN',nonnegative_projection_count=projected,
            point_geometry='paired positions' if dims!=('lat','lon') else 'rectilinear analysis grid',
            source_lineage=json.dumps([ds.attrs.get('source_id','UNKNOWN') for ds in task.fields]),
            sampling_operator='Rectilinear linear / nearest supplied paired position inside hull; instantaneous covariates may interpolate time linearly without extrapolation; no interval-total interpolation or native-cell equivalence',
            observation_role='contemporaneous model conditioning; no forecasting claim')
        quality=ResultQualityGate().validate(task.fields[0] if task.fields else out,out,variable=variable,
            nonnegative=bool(self.request.get('nonnegative')),expected_output_unit=self.request['unit'])
        if quality['status']=='BLOCK':raise ValueError('Result quality gate blocked spatial output: '+str(quality))
        if task.reference_grid is not None:
            from ecodata.spatial.regridding import require_target_grid
            out.attrs['target_grid_validation']=json.dumps(require_target_grid(out,task.template,variable))
            out[variable].attrs.update(spatial_semantics='cell_mean',estimate_kind='Reconstructed target-cell mean estimate; not independently observed fine detail')
            out.attrs['sampling_operator']='Sources conservatively reconstructed before graph; every candidate executes full target field; area-mean observation operator scores original reference cells'
        return out,trace


@threadpool_limits.wrap(limits=1)
def submit(parsed,request,audit,folder,*,budget,seed=11,progress=None):
    from ecodata.algorithms.composition_optimizers import random_graph_search
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists():raise ValueError('Immutable spatial request already exists')
    model=None
    if request.get('model_profile') or request.get('application_contract'):
        from ecodata.algorithms.profile_matching import prepare_model_request
        request,model=prepare_model_request(request)
        if set(model['matching_requirements']['hard'])-{'nonnegative'}:
            raise ValueError('Unsupported hard model requirement for spatial-estimation adapter')
        if 'nonnegative' in model['matching_requirements']['hard']:request=dict(request,nonnegative=True)
    if request.get('preferences'):
        raise ValueError('Custom multiobjective preference scoring is not yet supported for the new spatial ports')
    task=SpatialTask.build(parsed,request,audit)
    request=task.request
    save(folder/'request.json',dict(request=request,inputs=audit,budget=budget,seed=seed,version=VERSION))
    if progress:progress('搜索准备：按固定评价队列比较真实候选；评分不等于独立实测精度，搜索不保证全局最优')
    search=SpatialSearch(task,folder,budget,seed,model=model,progress=progress);search.budget=budget
    search.apply_prepared_recommendation()
    save(folder/'request.json',dict(request=request,inputs=audit,budget=budget,seed=seed,version=VERSION))
    search.validate_recommendations()
    if progress:
        for receipt in search.recommendation_receipts:
            progress('推荐契约核验：'+json.dumps(receipt,ensure_ascii=False))
    leaves=sorted(search.leaves,key=lambda k:(search.portraits.order_key(k),k)) if search.portraits else search.leaves
    seed_receipt=[]
    requested=request.get('initial_algorithms',[])
    if not isinstance(requested,list) or any(x not in {'spatial.idw','spatial.residual_idw','spatial.blend'} for x in requested):
        raise ValueError('Unsupported initial algorithm ID')
    for operation in dict.fromkeys(requested):
        keys=[k for k in [*leaves,*search.comparison_seeds()] if search.records[k]['operation']==operation]
        if operation=='spatial.idw':
            keys=[k for k in keys if search.records[k]['parameters'].get('power')==2. and task.observations is not None and not task.observations.empty]
        if not keys:raise ValueError('Initial algorithm prerequisites not met: '+operation)
        key=keys[0]
        search.execute(key,'confirmed_initial')
        row=next((r for r in search.rows if r['key']==key),None)
        if row is None:raise ValueError('Insufficient search budget for confirmed initial candidate')
        seed_receipt.append(dict(algorithm=operation,key=key,status=row['status'],metrics=row.get('metrics'),error=row.get('error')))
    if requested:save(folder/'confirmed_initial.json',seed_receipt)
    for key in leaves:
        if search.exhausted():break
        # New Registry neighbours remain reachable in mutation. Do not consume
        # the whole numerical budget evaluating every newly discovered leaf
        # before the first graph edit (legacy comparison seeds stay explicit).
        if search.records[key]['operation'] in search.registry_operations:continue
        search.execute(key)
    for key in search.comparison_seeds():
        if search.exhausted():break
        search.execute(key,'family_baseline')
    initial=search.best()
    if not initial:raise ValueError('No executable spatial estimator passed every spatial validation fold')
    initial_key=initial['key'];search.stopper.start(initial)
    backend=request.get('mutation_control','random')
    if backend not in {'random','none'}:raise ValueError('Spatial task supports random mutation or explicit no-mutation control')
    if progress:progress('开始变异搜索；缓存复用不重复记为新候选评价' if backend=='random' else '本请求关闭变异，仅比较实际候选')
    optimizer=random_graph_search(search,request,dict(portraits_enabled=search.portraits is not None)) if backend=='random' else dict(backend='none')
    best=search.best()
    if progress:progress(f"搜索停止 · 已评价 {len(search.rows)} 图，提议 {search.attempts} 次，真实节点调用 {search.account.receipt()['used']}；预算内最佳 {best['key']}，RMSE {best['metrics']['RMSE']:.5g}。正在生成产品及回读。")
    comparison=search.comparison_report(best)
    frontier=search.reference_frontier()
    structure_key=frontier.get('minimum_gradient_error_key')
    intermediate_key=frontier.get('minimum_intermediate_error_key')
    save(folder/'selection_freeze.json',dict(key=best['key'],baseline_key=comparison['families']['SINGLE_SOURCE']['best_key'],
        graph=search.records,review_axes=search.review_axes,confirmation_read=False,
        reference_frontier=frontier,structure_key=structure_key,intermediate_key=intermediate_key))
    out,trace=search.product(best['key']);path=folder/'product.nc';out.to_netcdf(path,engine='h5netcdf')
    with xr.open_dataset(path,engine='h5netcdf') as readback:xr.testing.assert_equal(out,readback.load())
    recommended_keys=list(dict.fromkeys(key for receipt in search.recommendation_receipts if receipt.get('status')=='PASS' for key in receipt.get('keys',[])))
    assessment=search.post_selection_assessment(dict(selected=best['key'],
        single_source_baseline=comparison['families']['SINGLE_SOURCE']['best_key'],
        multisource_candidate=comparison['families']['MULTISOURCE']['best_key'],
        structure_preferred=structure_key,
        **{'ds_recommendation_'+str(i):key for i,key in enumerate(recommended_keys[:2])}))
    confirmation=[]
    # Independent date-block confirmation is post-selection only; no promotion.
    if len(search.confirm_times):
        try:
            for fold in search.folds:
                cohort=task.cohort(fold['train'],task.query_grid(fold['query']),search.confirm_times)
                executor=search.executor(cohort,'validation_audit')
                executed=executor.run(search.nodes(best['key']),best['key']);a=executed['outputs']['field'].value[:,:len(fold['held'])]
                if request.get('nonnegative'):a=np.maximum(a,0.)
                if task.reference_grid is not None:
                    fine=executed['outputs']['field'].value[:,:cohort['query_count']]
                    if request.get('nonnegative'):fine=np.maximum(fine,0.)
                    a=task.observe_grid(fine,fold['query'])
                ref=task.observations.pivot(index='time',columns='station_id',values='value').reindex(index=search.confirm_times,columns=fold['held']).to_numpy(float)
                mask=np.isfinite(ref)
                if not np.isfinite(a[mask]).all():raise ValueError('Incomplete confirmation support')
                confirmation.extend((a[mask]-ref[mask]).tolist())
            confirm=dict(RMSE=float(np.sqrt(np.mean(np.square(confirmation)))),count=len(confirmation),used_for_selection=False)
        except (SearchBudgetExceeded,DAGError) as exc:
            if isinstance(exc,SearchBudgetExceeded) or getattr(exc,'details',{}).get('budget_exhausted'):
                confirm=dict(status='NOT_EVALUATED_RESOURCE_LIMIT',used_for_selection=False)
            else:confirm=dict(status='FAILED',error=str(exc),used_for_selection=False)
    else:confirm=dict(status='NO_SEPARATE_DATE_BLOCK',used_for_selection=False)
    from ecodata.evaluation.blocked_confirmation import review_spatial
    review=review_spatial(search,comparison,assessment,best)
    baseline_key=comparison['families']['SINGLE_SOURCE']['best_key']
    delivered_graph=dict(key=best['key']);candidate_product=None;delivery_status='ENHANCEMENT_RECOMMENDED' if review['recommended'] else 'EXPLORATORY_ONLY'
    if not review['recommended'] and baseline_key:
        if baseline_key==best['key']:delivery_status='BASELINE_RETAINED'
        else:
            candidate_product=folder/'exploratory-candidate.nc';candidate_product.write_bytes(path.read_bytes())
            try:
                fallback,fallback_trace=search.product(baseline_key)
                fallback.to_netcdf(path,engine='h5netcdf')
                with xr.open_dataset(path) as restored:xr.testing.assert_allclose(fallback,restored.load())
                delivered_graph=dict(key=baseline_key);trace=fallback_trace;delivery_status='BASELINE_RETAINED'
            except (SearchBudgetExceeded,DAGError,ValueError) as exc:
                path.write_bytes(candidate_product.read_bytes())
                review['delivery_error']=str(exc);delivery_status='EXPLORATORY_ONLY_NO_SAFE_BASELINE'
    save(folder/'recommendation_review.json',review)
    delivered_key=delivered_graph['key']
    delivered_graph['records']={n.id:search.records[n.id] for n in search.nodes(delivered_key)}
    retained_products=[];exported={delivered_key:str(path)}
    alternatives=[('baseline',baseline_key),('structure_preferred',structure_key)]
    alternatives.extend(('ds_recommendation_'+str(i),key) for i,key in enumerate(recommended_keys[:2]))
    alternatives.append(('intermediate_preferred',intermediate_key))
    for label,key in alternatives:
        if key is None:continue
        try:
            if key not in exported:
                product,product_trace=search.product(key)
                alternative=folder/(label+'.nc');product.to_netcdf(alternative,engine='h5netcdf')
                with xr.open_dataset(alternative) as restored:xr.testing.assert_allclose(product,restored.load())
                exported[key]=str(alternative)
            retained_products.append(dict(role=label,key=key,status='QUALITY_VALID',product=exported[key],
                default=key==delivered_key,enhancement_confirmed=False,selection_scope='search_validation_only',
                confirmation=assessment.get('schemes',{}).get(label,{}).get('metrics')))
        except (SearchBudgetExceeded,DAGError,ValueError) as exc:
            retained_products.append(dict(role=label,key=key,status='NOT_EXPORTED',reason=str(exc)))
    groups=[dict(held_station_ids=f['held'],training_station_ids=f['train'],reference_sha256=sha256(f['reference'].tobytes()).hexdigest()) for f in search.folds]
    result=dict(status='PASS',version=VERSION,product=str(path),best=best,initial_key=initial_key,
        recommendation_review=review,delivery_status=delivery_status,delivered_graph=delivered_graph,retained_products=retained_products,
        delivered_best=next(r for r in search.rows if r['key']==delivered_key),
        candidate_product=str(candidate_product) if candidate_product else None,
        recommendation_validation=search.recommendation_receipts,
        confirmed_initial=seed_receipt,optimizer=optimizer,counts=dict(evaluated=len(search.rows),proposals=search.attempts),
        stop_reason=search.stopper.reason or ('BUDGET_OR_PROPOSAL_LIMIT' if backend=='random' else 'NO_MUTATION_CONTROL'),
        selected_graph=dict(key=best['key'],records={n.id:search.records[n.id] for n in search.nodes(best['key'])}),
        deployment_trace=trace,current_input_deployment=dict(status='PASS'),total_cost=search.account.receipt(),
        selection_comparison=comparison,post_selection_assessment=assessment,
        self_validation_protocol=dict(tasks=['spatial_block_selection','untouched_location_assessment','date_block_confirmation'],folds=groups,
            search_start=str(search.search_times.min()),search_end=str(search.search_times.max()),
            confirmation_start=str(search.confirm_times.min()) if len(search.confirm_times) else None,
            self_gridded_reference=task.self_reference,
            external_test_used=bool(task.evaluation_observations is not None and not task.evaluation_observations.empty and assessment['status']=='PASS'),
            location_split=search.location_protocol,
            explicit_test_used=bool(task.evaluation_observations is not None and not task.evaluation_observations.empty),
            test_used_for_selection=False,
            deployment_refit='Frozen winning graph refitted on all declared training locations; explicit test remains excluded',
            reference_scope='held visible grid estimates' if task.self_reference else 'held visible station observations',
            statistical_independence='UNKNOWN: upstream gridded products may include the target stations'),
        confirmation=confirm,task_acceptance=dict(status='MECHANISM_AND_OUTPUT_PASS',task_quality_achieved=None),
        final_validation_best_delivered=True,evolution_used=any(t['actual_execution'] for t in search.transitions),
        limitations=['No certified global optimum','No downstream native model run','Unknown uncertainty and model sensitivity remain unknown',
                    'Sparse station cross-validation is not independent product truth','Point estimates are not conservative cell averages'],
        portrait_scope='Experimental existing three-portrait exploration heuristic; final selection by held-location RMSE' if search.portraits else 'Hard quantity/unit/support/model contracts active; no invented soft benefit ranking for new spatial ports',
        soft_guidance=dict(enabled=search.portraits is not None,candidate_pool=search.candidate_pool,
            changed_choices=sum(t['guidance_changed_choice'] for t in search.transitions)))
    if search.portraits is not None:
        state=model['state_policy']
        result['portrait_strategy']=dict(version=state['policy_version'],scenario=state['scenario']['id'],
            candidate_order=leaves,portrait_proposals=sum(t['strategy']=='portrait' for t in search.transitions),
            changed_choices=sum(t['guidance_changed_choice'] for t in search.transitions))
    result['component_catalog']=COMPONENTS
    result['evaluation_protocol']=search.evaluation_plan
    result['reference_frontier']=frontier
    result['structure_parent_keys']=sorted(getattr(search,'structure_parents_used',set()))
    result['choice_scope']='ERROR_STRUCTURE_TRADEOFF_NO_UNIVERSAL_WINNER' if frontier.get('tradeoff') else 'ERROR_INCUMBENT_WITH_STRUCTURE_CHECK' if frontier.get('keys') else 'STRUCTURE_NOT_RANKABLE'
    result['overall_optimum_claim']=False
    result['evaluation_protocol']['date_partition']=search.date_protocol
    result['evaluation_cost']=dict(preparation_seconds=search.evaluation_plan.get('cost_seconds',0.),
        numerical_calls='Included in total_cost search ledger; coarse diagnostic and reduced scale run separately')
    if search.reconstruction is not None:
        result['self_validation_protocol'].update(tasks=['reduced_scale_product_selection','frozen_reduced_scale_outer_blocks','coarse_primary_diagnostic'],
            coarse_diagnostic_folds=groups,folds=[],location_split=search.reconstruction['split'],
            selection_protocol_file='evaluation_protocol.json',
            reference_scope='Withheld native finer product at reduced scale, not independent observations')
    result['input_relationships']=task.relationships
    if task.reference_grid is not None:
        from ecodata.spatial.regridding import require_target_grid
        result['target_grid_contract']=dict(requirement=request['target_grid'],lat=task.template.lat.values.tolist(),lon=task.template.lon.values.tolist(),
            output=require_target_grid(out,task.template,request['variable']),preparation=task.target_alignment,
            preparation_seconds=sum(r['record']['elapsed_seconds'] for r in task.target_alignment),
            reference_shape=dict(task.reference_grid.sizes),reference_role='Original native cell means; no interpolated target truth',
            area_operator_shape=list(task.reference_operator.shape),candidate_checks=[dict(key=r['key'],status=r['status'],check=r.get('target_grid_check')) for r in search.rows])
    save(folder/'selection_comparison.json',comparison);save(folder/'post_selection_assessment.json',assessment)
    save(folder/'graphs.json',search.records);save(folder/'candidate_results.json',search.rows);save(folder/'transitions.json',search.transitions)
    if search.portraits is not None:save(folder/'portraits.json',dict(profiles=search.portraits.profiles,scores=search.portraits.cache))
    save(folder/'validation_protocol.json',result['self_validation_protocol']);save(folder/'total_cost.json',result['total_cost']);save(folder/'summary.json',result)
    return result


@threadpool_limits.wrap(limits=1)
def execute_graph(parsed,request,audit,graph,folder):
    """Fixed reference/deployment through the same ports; never used to seed search."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'deployment.json').exists():raise ValueError('Immutable spatial deployment exists')
    if request.get('model_profile') or request.get('application_contract'):
        from ecodata.algorithms.profile_matching import prepare_model_request
        request,model=prepare_model_request(request)
        if set(model['matching_requirements']['hard'])-{'nonnegative'}:
            raise ValueError('Unsupported hard model requirement for spatial-estimation adapter')
        if 'nonnegative' in model['matching_requirements']['hard']:request=dict(request,nonnegative=True)
    task=SpatialTask.build(parsed,request,audit)
    budget=dict(objective_limit=1000,total_call_limit=1000,wall_seconds=300,proposal_limit=1)
    engine=SpatialSearch(task,folder,budget,11);key=engine.bind_graph(graph)
    graph=dict(key=key,records={n.id:engine.records[n.id] for n in engine.nodes(key)})
    for node in engine.nodes(key):
        if node.operation=='spatial.source' and not 0<=node.parameters['index']<len(task.fields):
            raise ValueError('Fixed graph source binding differs')
        if node.operation in {'spatial.affine','spatial.ratio','spatial.aux_affine'} and not request.get('allow_training'):
            raise ValueError('Fixed fitting requires authorized training')
    out,trace=engine.product(key);path=folder/'product.nc';out.to_netcdf(path,engine='h5netcdf')
    with xr.open_dataset(path) as saved:xr.testing.assert_equal(out,saved.load())
    receipt=dict(status='PASS',product=str(path),graph=graph,trace=trace,cost=engine.account.receipt(),
        selection_performed=False,reference_used=False,inputs=audit)
    save(folder/'deployment.json',receipt)
    return receipt


@threadpool_limits.wrap(limits=1)
def evaluate_bank(parsed,request,audit,graphs,folder,*,budget,seed):
    """Matched validation of fixed controls only; cannot replace live search."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    task=SpatialTask.build(parsed,request,audit);engine=SpatialSearch(task,folder,budget,seed);engine.budget=budget
    keys={}
    for name,graph in graphs.items():
        key=engine.bind_graph(graph);keys[name]=key;engine.execute(key,'fixed_control_audit')
    result=dict(selection_performed=False,product=None,control_keys=keys,
        controls={name:next((r for r in engine.rows if r['key']==key),dict(status='NOT_EVALUATED_RESOURCE_LIMIT')) for name,key in keys.items()},
        total_cost=engine.account.receipt(),external_test_used=False)
    result['evaluation_protocol']=engine.evaluation_plan
    result['post_selection_assessment']=engine.post_selection_assessment(keys)
    result['fixed_graphs']=graphs
    result['per_date']=[]
    result['counts']=dict(evaluated=len(engine.rows),proposals=engine.attempts)
    result['full_domain_controls']={}
    if engine.reconstruction is not None:
        from ecodata.algorithms.spatial_selection_protocol import error_metrics
        r=engine.reconstruction
        for name,key in keys.items():
            if key not in engine.reconstruction_predictions:continue
            prediction=engine.reconstruction_predictions[key]
            result['full_domain_controls'][name]=dict(key=key,used_for_selection=False,
                metrics=error_metrics(prediction,r['reference']))
            for date,time in enumerate(engine.search_times):
                for split,index in [('selection_region',r['selected']),('frozen_outer_region',r['test']),
                                    ('all_target_cells',np.arange(r['reference'].shape[1]))]:
                    if not len(index):continue
                    result['per_date'].append(dict(control=name,key=key,date=str(time),region=split,
                        used_for_selection=False,metrics=error_metrics(prediction[date:date+1,index],r['reference'][date:date+1,index])))
    save(folder/'summary.json',result)
    return result
