"""Metric IDW weights for paired positions, not a Cartesian target grid."""
import numpy as np
from pyproj import CRS, Transformer
from scipy.spatial.distance import cdist


def inverse_distance_weights(distances, power):
    return 1. / np.maximum(distances, 1e-8) ** power


def distance_weights(source, target, power=2.):
    source, target = np.asarray(source, float), np.asarray(target, float)
    if (source.ndim != 2 or target.ndim != 2 or source.shape[1] != 2 or target.shape[1] != 2
            or not len(source) or not np.isfinite(source).all() or not np.isfinite(target).all()
            or not np.isfinite(power) or power <= 0):
        raise ValueError('Finite longitude/latitude pairs and positive IDW power required')
    crs = CRS.from_proj4(f'+proj=aeqd +lat_0={target[:,1].mean()} +lon_0={target[:,0].mean()} +datum=WGS84 +units=m')
    transform = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    s = np.column_stack(transform.transform(source[:,0], source[:,1]))
    t = np.column_stack(transform.transform(target[:,0], target[:,1]))
    distances = cdist(t, s)
    return inverse_distance_weights(distances, power)


def weighted_values(values, weights):
    """Time x source -> time x target; missing inputs have zero weight."""
    valid = np.isfinite(values)
    numerator = np.where(valid, values, 0.) @ weights.T
    denominator = valid.astype(float) @ weights.T
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
