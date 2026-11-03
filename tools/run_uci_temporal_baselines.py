"""Fixed observed-only masks; retrospective and strictly causal baselines."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('D:/codex/数据/公开数据三层预审_20261018/01_UCI_Beijing/official_original/extracted/PRSA_Data_20130301-20170228'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    predictions, masks, timings = [], [], []
    for file in sorted(a.data.glob('*.csv')):
        df = pd.read_csv(file)
        time = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        y = pd.Series(df['PM2.5'].to_numpy(float), index=time)
        station = str(df.station.iloc[0])
        train = y.loc[:'2015-12-31 23:00'].dropna()
        history = train.groupby(train.index.hour).median()
        # KNN uses calendar features only, not masked truth or future donors.
        def features(index):
            return np.c_[np.sin(index.hour*2*np.pi/24), np.cos(index.hour*2*np.pi/24),
                         np.sin(index.dayofyear*2*np.pi/366), np.cos(index.dayofyear*2*np.pi/366)]
        knn = KNeighborsRegressor(n_neighbors=24, weights='distance').fit(features(train.index), train.to_numpy())
        for split, start, end in [('validation', '2016-01-01', '2016-12-31 23:00'),
                                  ('test', '2017-01-01', '2017-02-28 23:00')]:
            target = y.loc[start:end]
            for scenario, length in [('random', 1), ('block6', 6), ('block24', 24), ('block72', 72)]:
                rng = np.random.default_rng(11)
                candidates = rng.permutation(max(0, len(target)-length-1)) + 1
                selected = set()
                for pos in candidates:
                    block = target.iloc[pos:pos+length]
                    if len(block) == length and block.notna().all() and not any(i in selected for i in range(pos-1,pos+length+1)):
                        selected.update(range(pos,pos+length))
                    if len(selected) >= (120 if scenario == 'random' else 3*length):
                        break
                idx = target.index[sorted(selected)]
                masked = y.copy()
                masked.loc[idx] = np.nan
                for t in idx:
                    masks.append(dict(station=station, split=split, scenario=scenario, time=str(t), seed=11))
                for mode, method in [('bidirectional', 'linear'), ('past_only', 'last_observation'),
                                     ('past_only', 'historical_hour_median'), ('past_only', 'calendar_knn')]:
                    clock = perf_counter()
                    if method == 'linear':
                        estimate = masked.interpolate(method='time', limit_area='inside').loc[idx].to_numpy()
                    elif method == 'last_observation':
                        estimate = masked.ffill().loc[idx].to_numpy()
                    elif method == 'historical_hour_median':
                        estimate = np.array([history.loc[t.hour] for t in idx])
                    else:
                        estimate = knn.predict(features(idx))
                    timings.append(dict(station=station, split=split, scenario=scenario, mode=mode,
                                        method=method, seconds=perf_counter()-clock))
                    for t, value in zip(idx, estimate):
                        predictions.append(dict(station=station, split=split, scenario=scenario, mode=mode,
                                                method=method, time=str(t), truth=float(y.loc[t]), prediction=float(value)))
    out = pd.DataFrame(predictions)
    assert np.isfinite(out[['truth', 'prediction']]).all().all()
    out.to_csv(a.output/'predictions.csv', index=False)
    pd.DataFrame(masks).to_csv(a.output/'fixed_masks.csv', index=False)
    pd.DataFrame(timings).to_csv(a.output/'timings.csv', index=False)
    metrics = []
    for key, group in out.groupby(['split','scenario','mode','method']):
        error = group.prediction-group.truth
        metrics.append(dict(zip(['split','scenario','mode','method'], key), n=len(group),
                            RMSE=float(np.sqrt(np.mean(error**2))), MAE=float(abs(error).mean()), Bias=float(error.mean())))
    pd.DataFrame(metrics).to_csv(a.output/'metrics.csv', index=False)
    reread = pd.read_csv(a.output/'predictions.csv')
    assert len(reread) == len(out) and np.allclose(reread.prediction, out.prediction)
    (a.output/'configuration.json').write_text(json.dumps(dict(seed=11, train='2013-03-01/2015-12-31',
        validation='2016', test='2017-01-01/2017-02-28', variable='PM2.5', unit='ug/m3',
        natural_missing_not_scored=True, artificial_masks_observed_only=True,
        knn='24 neighbors on hour/day seasonal coordinates, donors training dates only',
        retrospective='time interpolation can use later observations; NOT forecasting',
        causal='ffill uses only earlier available observations; medians/KNN fit on 2013-2015 only',
        no_model_selection_on_test=True, readback='PASS'), indent=2), encoding='utf-8')
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == '__main__':
    main()
