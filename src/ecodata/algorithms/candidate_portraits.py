"""Visible-data/registered-algorithm/model prior, independent of validation loss.

Increment-based affinity is an engineering smoothness heuristic, NOT a learned
error prediction or a Lipschitz guarantee. Unknown transfers remain unknown.
"""
import numpy as np
from ecodata.algorithms.portrait_rules import algorithm_properties
from ecodata.algorithms.portrait_state import risk


def data_profile(context,index=0):
    source=context.sources[index].dataset
    a=np.asarray(source[context.variable].values,float)
    known=a[np.isfinite(a)];scale=float(np.std(known)) if len(known) else 0.
    scale=scale or 1.
    missing=~np.isfinite(a);distances={};support={}
    from scipy.ndimage import distance_transform_edt
    spatial=[];supported=0
    for frame in a:
        absent=~np.isfinite(frame)
        if np.isfinite(frame).any():
            spatial.extend(distance_transform_edt(absent)[absent].tolist());supported+=int(absent.sum())
    distances['spatial']=float(np.mean(spatial)) if spatial else 1.
    support['spatial']=supported/int(missing.sum()) if missing.any() else 1.
    days=(source.time.values-source.time.values[0])/np.timedelta64(1,'D')
    cadence=float(np.median(np.diff(days))) if len(days)>1 else 1.
    time=days[:,None,None]
    left=np.maximum.accumulate(np.where(~missing,time,-np.inf),axis=0)
    right=np.minimum.accumulate(np.where(~missing,time,np.inf)[::-1],axis=0)[::-1]
    distance=np.minimum(time-left,right-time);usable=missing&np.isfinite(distance)
    distances['temporal']=float(np.mean(distance[usable])/cadence) if usable.any() else 1.
    support['temporal']=int(usable.sum())/int(missing.sum()) if missing.any() else 1.
    variation={}
    for name,axes in [('spatial',(1,2)),('temporal',(0,))]:
        changes=np.concatenate([(np.abs(np.diff(a,axis=axis))*cadence/np.diff(days)[:,None,None] if axis==0 and len(days)>1 else np.abs(np.diff(a,axis=axis))).ravel() for axis in axes])
        changes=changes[np.isfinite(changes)]
        variation[name]=float(np.mean(changes)/scale) if len(changes) else None
    grid=bool(context.target_lat is not None and context.target_lon is not None and
        any(not np.array_equal(source[c],target) for c,target in [('lat',context.target_lat),('lon',context.target_lon)]))
    return dict(missing_fraction=float(1-np.isfinite(a).mean()),negative_fraction=float(np.mean(known<0)) if len(known) else None,
        grid_gap=grid,variation=variation,support_distance=distances,support_fraction=support,scale=scale,source_id=source.attrs.get('source_id'),
        basis='Visible finite adjacent increments / visible standard deviation; missingness and requested geometry; no validation or test labels')


class CandidatePortraits:
    def __init__(self,owner,context):
        self.owner=owner;self.data=data_profile(context,getattr(owner,'request',{}).get('_current_primary_index',0));self.cache={}
        self.issues=owner.field['model']['state_policy']['scenario']['issues']

    def score(self,key):
        if key in self.cache:return self.cache[key]
        from ecodata.algorithms.autonomous_field import Framewise
        from ecodata.algorithms.capabilities import describe_algorithm
        from ecodata.algorithms.raster_gap_prediction import GridGapRecovery,DeclaredRangeProjection
        from ecodata.algorithms.raster_temporal import RasterTimeInterpolation
        from ecodata.algorithms.series_fusion import SeriesNonnegative
        actions=set();algorithms=[];nonnegative=None
        for node in self.owner.nodes(key):
            record=self.owner.records[node]
            if record['operation']=='source':continue
            plugin=self.owner.registry.get(record['operation'])
            if isinstance(plugin,Framewise):plugin=plugin.plugin
            caps=set(describe_algorithm(plugin.spec).capabilities)
            properties=algorithm_properties(plugin)
            if 'spatial_missing_recovery' in caps or properties['transfer'] in {'nearest','bilinear','pchip','cubic'} or 'continuous_spatial_resampling' in caps:actions.add('spatial')
            if 'temporal_missing_recovery' in caps:actions.add('temporal')
            if 'spatiotemporal_missing_recovery' in caps:actions.update(['spatial','temporal'])
            convex=(properties['transfer'] in {'nearest','bilinear','pchip'} or
                isinstance(plugin,(RasterTimeInterpolation,GridGapRecovery)) and plugin.method in {'linear','nearest'})
            if isinstance(plugin,SeriesNonnegative) or properties['transfer']=='clip_nonnegative':nonnegative=True
            elif isinstance(plugin,DeclaredRangeProjection):nonnegative=self.owner.context.parameters.get('target_valid_range',[-1,1])[0]>=0
            elif not convex:nonnegative=None
            algorithms.append(dict(operation=record['operation'],capabilities=sorted(caps),transfer=properties,convex_value_transfer=bool(convex),
                pointwise_projection=isinstance(plugin,(SeriesNonnegative,DeclaredRangeProjection)) or properties['transfer']=='clip_nonnegative',
                evidence='Existing registered capabilities and reviewed transfer whitelist; no performance invented'))
        result=score_profile(self.data,algorithms,actions,nonnegative,self.issues,self.owner.field['model'].get('scoring_parameters',{}))
        self.cache[key]=result;return result

    def order_key(self,key):
        value=self.score(key)
        return (-value['score_interval'][0],value['unknown_terms'],self.owner.node_count(key))


def score_profile(data,algorithms,actions,nonnegative,issues,parameters):
    """Shared existing three-portrait heuristic; never a prediction of accuracy."""
    affinities={axis:data['support_fraction'][axis]/(1+data['variation'][axis]*data['support_distance'][axis]) for axis in actions if data['variation'][axis] is not None}
    terms=[];total=sum(i['importance'] for i in issues)
    for issue in issues:
        setting=parameters.get(issue['attribute'],{})
        metric=issue['metric'];weight=issue['importance']/total*setting.get('multiplier',1.);value=None;activation=1.
        if metric=='missing_fraction':
            activation=max(data['missing_fraction'],float(data['grid_gap']))
            if data['grid_gap']:affinity=affinities.get('spatial')
            else:affinity=max(affinities.values()) if affinities else None
            value=affinity
        elif metric=='negative_fraction':
            activation=risk(data['negative_fraction'],issue) if data['negative_fraction'] is not None else 1.
            value=1. if nonnegative is True else None
        if activation<setting.get('activation_threshold',0.):activation=0.
        bounds=[weight*activation*value]*2 if value is not None else [0.,weight*activation]
        terms.append(dict(dimension=issue['attribute'],metric=metric,model_weight=weight,model_evidence=issue['evidence'],
            activation=activation,affinity=value,interval=bounds,status='VISIBLE_PROFILE_HEURISTIC' if value is not None else 'UNKNOWN',
            basis='Model importance x observed problem degree x registered-action affinity; repeated actions counted once'))
    result=dict(version='candidate_portraits_v1',data=data,algorithms=algorithms,model_terms=terms,
        score_interval=[sum(t['interval'][i] for t in terms) for i in (0,1)],
        accuracy_prediction='UNKNOWN',validation_loss_used=False,
        score_scope='Candidate exploration priority only; final graph must pass actual required-fold evaluation',
        unknown_terms=sum(t['status']=='UNKNOWN' and t['activation']>0 for t in terms))
    return result
