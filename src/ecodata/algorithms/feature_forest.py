"""Shared station-group OOF kernel, extracted from EPATask without retuning."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold


def fit_station_forest(table, features, *, compute_oof=True):
    x = table[list(features)].to_numpy(float)
    y = table.value.to_numpy(float)
    if not np.isfinite(x).all() or table.station_id.nunique() < 4:
        raise ValueError("Insufficient independent stations or missing covariates")
    config = dict(n_estimators=50, max_depth=6, min_samples_leaf=3, random_state=11, n_jobs=1)
    oof = np.full(len(table), np.nan)
    folds = []
    for fit, hold in (GroupKFold(n_splits=min(5, table.station_id.nunique())).split(x, y, groups=table.station_id) if compute_oof else []):
        model = RandomForestRegressor(**config).fit(x[fit], y[fit])
        oof[hold] = model.predict(x[hold])
        folds.append(dict(fit_stations=sorted(table.iloc[fit].station_id.unique()),
                          held_stations=sorted(table.iloc[hold].station_id.unique())))
    return RandomForestRegressor(**config).fit(x, y), oof, folds
