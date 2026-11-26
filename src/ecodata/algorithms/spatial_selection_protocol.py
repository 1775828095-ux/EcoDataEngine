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


def xskillscore_assessment(prediction, reference):
    """Score frozen predictions on the same finite reference points."""
    import xarray as xr
    import xskillscore as xs

    mask = np.isfinite(reference)
    if prediction.shape != reference.shape or not mask.any():
        raise ValueError("No matching finite reference on fixed assessment support")
    if not np.isfinite(prediction[mask]).all():
        raise ValueError("Incomplete prediction on fixed assessment support")
    actual = xr.DataArray(reference[mask], dims=["matched_sample"])
    estimate = xr.DataArray(prediction[mask], dims=["matched_sample"])
    return dict(RMSE=float(xs.rmse(estimate, actual).item()),
                MAE=float(xs.mae(estimate, actual).item()),
                Bias=float(xs.me(estimate, actual).item()), count=int(mask.sum()))


def blocked_validation(prediction,reference,times,location_ids,block_by_station):
    """Equal date x region loss; validation only, no source-family preference.

    These are descriptive blocks, not independent statistical repeats. Missing
    references never become truth; failed predictions cannot drop hard cases.
    """
    prediction=np.asarray(prediction);reference=np.asarray(reference)
    if prediction.shape!=reference.shape or reference.shape!=(len(times),len(location_ids)):
        raise ValueError('Blocked validation queue mismatch')
    rows=[]
    groups=np.asarray([block_by_station[str(i)] for i in location_ids])
    import pandas as pd
    days=pd.DatetimeIndex(times).normalize()
    for date in days.unique():
        selected=days==date
        for group in sorted(set(groups)):
            keep=groups==group
            y=reference[selected][:,keep];p=prediction[selected][:,keep]
            if not np.isfinite(y).any():continue
            metrics=error_metrics(p,y)
            rows.append(dict(date=str(date),region=int(group),metrics=metrics))
    if not rows:raise ValueError('No actual date-region validation references')
    losses=np.array([row['metrics']['RMSE'] for row in rows])
    return dict(objective=float(losses.mean()),objective_name='equal_date_region_RMSE',
        pooled=error_metrics(prediction,reference),blocks=rows,block_count=len(rows),
        date_count=len(set(row['date'] for row in rows)),region_count=len(set(row['region'] for row in rows)),
        RMSE_block_std=float(losses.std()),worst_block_RMSE=float(losses.max()),
        source_bonus=0,uncertainty='Descriptive date-region variability, not a confidence interval',
        robustness_status='MULTIPLE_DATES_AND_REGIONS' if len(set(row['date'] for row in rows))>=3 and len(set(groups))>=2 else 'INSUFFICIENT_DATE_OR_REGION_REPLICATION')
