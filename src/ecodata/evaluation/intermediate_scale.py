"""Area-weighted scoring strictly between the native and degraded reference grids."""
from hashlib import sha256

import numpy as np
from scipy.sparse import kron

from ecodata.algorithms.raster_reconstruction_task import edges, overlap_weights


def prepare_grid(fine, coarse):
    """Choose an intermediate grid from geometry alone, over the fine footprint."""
    weights = []
    steps = {}
    for axis in ('lat', 'lon'):
        fine_edges = edges(fine[axis])
        coarse_edges = edges(coarse[axis])
        fine_step = float(np.diff(fine_edges).mean())
        coarse_step = float(np.diff(coarse_edges).mean())
        span = float(fine_edges[-1] - fine_edges[0])
        counts = [count for count in range(1, len(fine_edges) - 1)
                  if fine_step * (1 + 1e-9) < span / count < coarse_step * (1 - 1e-9)]
        if not counts:
            return None
        target_step = np.sqrt(fine_step * coarse_step)
        count = min(counts, key=lambda n: abs(np.log(span / n / target_step)))
        bounds = np.linspace(fine_edges[0], fine_edges[-1], count + 1)
        weights.append(overlap_weights(fine_edges, bounds, latitude=axis == 'lat'))
        steps[axis] = dict(fine=fine_step, intermediate=span / count, coarse=coarse_step, cells=count)

    operator = kron(*weights, format='csr')
    missing = operator.copy()
    missing.data.fill(1)
    geometry = b''.join(fine[dim].values.tobytes() for dim in ('time', 'lat', 'lon') if dim in fine.coords)
    return dict(operator=operator, missing=missing, steps_degrees=steps, geometry=sha256(geometry).digest())


def score(grid, prediction, reference, indices):
    """Aggregate paired errors only for cells fully covered by this phase's labels."""
    if grid is None:
        return dict(status='INTERMEDIATE_GRID_UNAVAILABLE')
    valid = np.zeros(reference.shape, dtype=bool)
    valid[:, indices] = np.isfinite(reference[:, indices])
    complete = (grid['missing'] @ (~valid).T).T == 0
    count = int(complete.sum())
    if not count:
        return dict(status='NO_COMPLETE_INTERMEDIATE_CELLS', count=0)
    errors = np.zeros(reference.shape, dtype=float)
    errors[valid] = prediction[valid] - reference[valid]
    aggregated = (grid['operator'] @ errors.T).T[complete]
    if not np.isfinite(aggregated).all():
        return dict(status='INCOMPLETE_INTERMEDIATE_PREDICTION', count=count)
    digest = sha256(grid['geometry'] + repr(grid['steps_degrees']).encode()
                    + valid.tobytes() + complete.tobytes()).hexdigest()
    return dict(status='PASS', RMSE=float(np.sqrt(np.mean(aggregated ** 2))), count=count,
                support_sha256=digest)
