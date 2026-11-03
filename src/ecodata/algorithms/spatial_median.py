"""Small, deterministic SciPy filter; smoothing can also worsen reconstruction."""
import numpy as np
from scipy.ndimage import median_filter
from ecodata.algorithms.contracts import AlgorithmSpec,Compatibility,AlgorithmResult
from ecodata.core.schemas import EcoDataBundle


class SpatialMedian:
    spec=AlgorithmSpec('spatial.median_filter','SciPy 3x3 spatial median','1','spatial_filter','available',
        ('regular_grid',),('continuous',),capabilities=('spatial_noise_suppression',),
        backend='scipy.ndimage.median_filter; fixed 3x3 reflect boundary; no temporal mixing')

    def compatibility(self,context):
        a=context.sources[0].dataset[context.variable]
        ok=context.semantic_type=='continuous' and a.dims==('lat','lon') and min(a.shape)>=3 and np.isfinite(a).all()
        return Compatibility(bool(ok),() if ok else ('Finite continuous 2D spatial grid of at least 3x3 required',))

    def run(self,context):
        if not self.compatibility(context).eligible:raise ValueError('Unsupported spatial median input')
        ds=context.sources[0].dataset.copy(deep=True)
        ds[context.variable].values=median_filter(ds[context.variable].values,size=3,mode='reflect')
        return AlgorithmResult(EcoDataBundle(ds),self.spec.id,dict(size=3,boundary='reflect',training=False))
