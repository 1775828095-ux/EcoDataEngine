"""Paired-location adapter for the existing candidate-portrait heuristic.

Only retained fold inputs and search dates are measured. No held labels or
confirmation values are read; model preferences must be explicitly registered.
"""
import numpy as np
from scipy.spatial import cKDTree
from ecodata.algorithms.candidate_portraits import score_profile


class SpatialPortraits:
    def __init__(self, search, model):
        self.search=search;self.model=model;self.cache={};self.profiles=[]
        for fold in search.folds:
            c=fold['cohort'];y=c['y'];finite=y[np.isfinite(y)]
            scale=float(np.std(finite)) if len(finite) else 0.;scale=scale or 1.
            # Neighbours are geographic, never adjacent columns of a point table.
            distances,indices=cKDTree(c['sxy']).query(c['sxy'],k=2)
            changes=np.abs(y-y[:,indices[:,1]]);known=changes[np.isfinite(changes)]
            variation=float(known.mean()/scale) if len(known) else None
            nearest=cKDTree(c['sxy']).query(c['allxy'][:c['query_count']])[0]
            spacing=float(np.median(distances[:,1]));distance=float(nearest.mean()/spacing) if spacing>0 else 1.
            self.profiles.append(dict(missing_fraction=float(1-np.isfinite(y).mean()),
                negative_fraction=float(np.mean(finite<0)) if len(finite) else None,
                grid_gap=True,variation={'spatial':variation},support_distance={'spatial':distance},
                support_fraction={'spatial':float(np.mean(np.isfinite(y).any(axis=1)))},scale=scale,
                basis='Retained station geographic neighbours and query geometry on search dates only'))

    def score(self,key):
        if key in self.cache:return self.cache[key]
        actions=set();algorithms=[]
        for node in self.search.nodes(key):
            op=node.operation
            if op in {'spatial.observations','spatial.auxiliary'}:continue
            # These are explicit spatial estimators; calibration/blend alone
            # establishes no new spatial capability or accuracy guarantee.
            if op in {'spatial.idw','spatial.residual_idw','spatial.aux_affine'}:actions.add('spatial')
            algorithms.append(dict(operation=op,contract=self.search.records[node.id],
                evidence='Executed typed spatial component; no inferred accuracy response'))
        issues=self.model['state_policy']['scenario']['issues']
        parameters=self.model.get('scoring_parameters',{})
        results=[score_profile(p,algorithms,actions,
            True if self.search.request.get('nonnegative') else None,issues,parameters) for p in self.profiles]
        result=dict(version='spatial_candidate_portraits_v1',folds=results,
            score_interval=np.mean([r['score_interval'] for r in results],axis=0).tolist(),
            unknown_terms=sum(r['unknown_terms'] for r in results),
            model_policy=self.model['state_policy'],accuracy_prediction='UNKNOWN',validation_loss_used=False,
            scope='Existing heuristic transferred experimentally; not calibrated SWAT sensitivity')
        self.cache[key]=result;return result

    def order_key(self,key):
        r=self.score(key)
        return (-r['score_interval'][0],r['unknown_terms'],len(self.search.nodes(key)))
