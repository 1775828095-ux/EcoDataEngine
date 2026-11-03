"""Scoped empirical algorithm portraits, never a replacement ranking model.

For linear mixtures, error second moments include both bias and correlation:
E[e(a)^2] = a^2 Mhh + (1-a)^2 Mcc + 2a(1-a) Mhc.
All matrices must be fitted before evaluation and tied to visible-data scope.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ecodata.algorithms.attribute_registry import digest


def scope(sources,request,reconstruction):
    fine=sources[reconstruction['fine_source']];coarse=sources[reconstruction['coarse_source']]
    def relative_times(ds):
        t=pd.DatetimeIndex(ds.time.values)
        return ((t-t[0]).total_seconds()/3600).tolist()
    target=pd.DatetimeIndex(request['target_times']);origin=pd.Timestamp(fine.time.values[0])
    return dict(variable=request['variable'],unit=request['unit'],
        physical_variable=fine[request['variable']].attrs.get('physical_variable'),
        spatial_semantics=fine[request['variable']].attrs.get('spatial_semantics'),
        temporal_semantics=fine[request['variable']].attrs.get('temporal_semantics'),
        grids={k:{a:sources[k][a].values.tolist() for a in ('lat','lon')} for k in (reconstruction['fine_source'],reconstruction['coarse_source'])},
        fine_hours=relative_times(fine),coarse_hours=relative_times(coarse),
        target_hours=((target-origin).total_seconds()/3600).tolist())


def family(plan,fine_id):
    if len({p['window'] for p in plan})!=1:return None,None
    fusion=next(p['choice'] for p in plan if p['branch']=='fusion')
    if any(p['choice'] for p in plan if p['branch']=='post'):return None,None
    algorithm=fusion['algorithm']
    if algorithm=='fusion.spatiotemporal_detail':alpha=fusion['detail_strength']
    elif algorithm=='fusion.primary_weighted_mean':
        alpha=fusion['primary_share'] if fusion['primary']==fine_id else 1-fusion['primary_share']
    else:return None,None
    branches=[dict(branch=p['branch'],choice=p['choice']) for p in plan if p['branch'] not in {'fusion','post'}]
    return digest(dict(branches=branches,fusion=algorithm)),alpha


class ResponsePortrait:
    def __init__(self,declaration,sources,request,reconstruction):
        path=Path(declaration['path']);raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=declaration['sha256']:raise ValueError('Response portrait digest mismatch')
        self.profile=json.loads(raw)
        if self.profile['schema_version']!=1 or self.profile['training_partition']!='development':
            raise ValueError('Development-only response portrait required')
        self.supported=self.profile['scope']==scope(sources,request,reconstruction)
        self.fine_id=reconstruction['fine_source']
        self.scale=float(sources[self.fine_id][request['variable']].std())
        self.evidence=dict(version=self.profile['version'],sha256=declaration['sha256'],
            kind='development error-response estimate; not observed test error',
            supported=self.supported,training_groups=self.profile['training_groups'])

    def estimate(self,plan):
        key,alpha=family(plan,self.fine_id)
        if not self.supported or self.scale<=1e-12 or key not in self.profile['families']:return None
        matrices=np.asarray(self.profile['families'][key]['second_moments'],float)
        if matrices.shape!=(len(self.profile['training_groups']),2,2) or not np.isfinite(matrices).all() or np.linalg.eigvalsh(matrices).min() < -1e-9:
            raise ValueError('Invalid error-response second moments')
        v=np.array([alpha,1-alpha]);mse=np.einsum('i,kij,j->k',v,matrices,v)
        return float(np.sqrt(max(0,float(mse.mean()))))
