"""Fixed spatial selection/assessment splits shared by all model adapters.

Uses scikit-learn's group splitters; no model names or scientific weights.
Geographic blocks reduce neighbour leakage but are not a decorrelation proof.
"""
import hashlib
import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def spatial_blocks(meta):
    xy = meta[['longitude', 'latitude']].to_numpy(float)
    if len(xy)<6:
        return np.arange(len(xy))  # Too sparse for occupied multi-point blocks.
    side = max(2, min(4, int(np.sqrt(len(xy) / 4))))
    codes = []
    for axis in xy.T:
        edges = np.unique(np.quantile(axis, np.linspace(0, 1, side + 1)))[1:-1]
        codes.append(np.searchsorted(edges, axis, side='right'))
    return codes[0] * (side + 1) + codes[1]


def split_locations(meta, seed, *, explicit_test=False):
    """Only coordinates/identities enter splitting; never reference values."""
    ids = meta.station_id.to_numpy()
    blocks = spatial_blocks(meta)
    selection = np.arange(len(ids))
    held = np.array([], dtype=int)
    reason = 'EXPLICIT_TEST_LOCATIONS' if explicit_test else 'INSUFFICIENT_LOCATIONS_FOR_OUTER_BLOCK'
    if not explicit_test and len(ids) >= 10 and len(np.unique(blocks)) >= 4:
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                         random_state=int(seed)).split(selection, groups=blocks))
        if len(train) >= 3 and len(np.unique(blocks[train])) >= 2:
            selection, held = train, test
            reason = 'AUTOMATIC_SPATIAL_BLOCK_HOLDOUT'
    remaining = blocks[selection]
    n_splits = min(6, len(np.unique(remaining)))
    if n_splits < 2:
        raise ValueError('At least two occupied geographic blocks required for spatial validation')
    splitter = GroupKFold(n_splits=n_splits)
    folds = [(ids[selection[a]].tolist(), ids[selection[b]].tolist())
             for a, b in splitter.split(selection, groups=remaining)]
    return dict(selection_ids=ids[selection].tolist(), test_ids=ids[held].tolist(),
                folds=folds, reason=reason, backend='sklearn GroupShuffleSplit / GroupKFold',
                block_by_station={str(i):int(b) for i,b in zip(ids, blocks)},
                limitation='Quantile geographic blocks (singleton locations for fewer than six); correlation length and upstream independence unknown')


def fixed_support(reference, ids, times):
    mask = np.isfinite(reference)
    payload = repr((list(ids), [str(t) for t in times])).encode() + mask.tobytes()
    return hashlib.sha256(payload).hexdigest()


def error_metrics(prediction, reference):
    mask = np.isfinite(reference)
    if not mask.any():
        raise ValueError('No finite reference on fixed assessment support')
    if not np.isfinite(prediction[mask]).all():
        raise ValueError('Incomplete prediction on fixed assessment support')
    error = prediction[mask] - reference[mask]
    return dict(RMSE=float(np.sqrt(np.mean(error**2))),
                MAE=float(np.mean(abs(error))), Bias=float(error.mean()), count=int(mask.sum()))
