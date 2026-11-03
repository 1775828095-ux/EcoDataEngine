"""Four frozen EPA DAG configurations, fixed-cohort descriptive comparison.

Only the prior non-test table is opened. No downloader, optimiser or test labels.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.append(str(ROOT / '.venv-golem/Lib/site-packages'))
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import Delaunay
import xarray as xr
from ecodata.algorithms.dag import Artifact, NumericalDAGExecutor
from ecodata.algorithms.epa_task import EPATask, UNIT
from ecodata.algorithms.registry import create_default_registry
from run_three_source_ready_smoke import golem_graph

METHODS = ('fusion.idw', 'station.pykrige_ordinary', 'route_a', 'route_b')
POOL_SHA = '3200321ef18415e80e501bdb49325bc32783ab6fd8c5945ffe91262f49c18d28'


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qualify(train, validation, covariates):
    """Outcome-blind common domain, matching the existing output quality gate."""
    audit = validation[['station_id', 'time']].copy()
    audit['reason'] = 'QUALIFIED'
    if len(train) < 8:
        audit['reason'] = 'INPUT_STATIONS_LT_8'
        return audit
    features = train[['aod', 'temperature_2m']].to_numpy(float)
    if not np.isfinite(features).all() or np.linalg.matrix_rank(np.c_[np.ones(len(train)), features]) < 3:
        audit['reason'] = 'COMMON_UK_DRIFT_CONTRACT_UNSATISFIED'
        return audit
    # Same metre-based hull as EPATask.quality, not a longitude/latitude hull.
    tx = Transformer.from_crs(4326, 32611, always_xy=True)
    sx, sy = tx.transform(train.longitude.to_numpy(), train.latitude.to_numpy())
    hull = Delaunay(np.c_[sx, sy])
    qx, qy = tx.transform(validation.longitude.to_numpy(), validation.latitude.to_numpy())
    inside = hull.find_simplex(np.c_[qx, qy]) >= 0
    yy, xx = np.meshgrid(covariates.lat.values, covariates.lon.values, indexing='ij')
    gx, gy = tx.transform(xx.ravel(), yy.ravel())
    grid_mask = (hull.find_simplex(np.c_[gx, gy]) >= 0).reshape(xx.shape)
    grid_mask &= np.isfinite(covariates.aod.values) & np.isfinite(covariates.temperature_2m.values)
    support = xr.DataArray(np.where(grid_mask, 1., np.nan), dims=('lat', 'lon'), coords=covariates.coords)
    stencil = support.interp(lat=xr.DataArray(validation.latitude.to_numpy(), dims='station'),
                             lon=xr.DataArray(validation.longitude.to_numpy(), dims='station')).notnull().values
    audit.loc[~inside, 'reason'] = 'OUTSIDE_INPUT_STATION_HULL'
    audit.loc[inside & ~stencil, 'reason'] = 'INCOMPLETE_COMMON_READBACK_STENCIL'
    return audit


def summarize(predictions, runs, cohort_size, raw_size):
    """No survivor-only accuracy: full-cohort completion is required to rank."""
    rows = []
    for method in METHODS:
        sample = predictions[predictions.method.eq(method)]
        attempts = [r for r in runs if r['method'] == method]
        good = sample.status.eq('PASS') & np.isfinite(sample.prediction.to_numpy(dtype=float))
        complete = len(sample) == cohort_size and bool(good.all()) and cohort_size > 0
        error = sample.loc[good, 'prediction']-sample.loc[good, 'value']
        projections = [r.get('quality', {}).get('nonnegative_projection', {}) for r in attempts]
        fit_seconds = sum(node['seconds'] for r in attempts for node in r.get('trace', [])
                          if node.get('operation') == 'epa.rf_fit' and not node.get('cache_hit'))
        total_seconds = sum(r['seconds'] for r in attempts)
        rows.append(dict(method=method, cohort_n=cohort_size, predicted_n=int(good.sum()),
            cohort_coverage=float(good.sum()/cohort_size) if cohort_size else 0.,
            raw_validation_coverage=float(good.sum()/raw_size) if raw_size else 0.,
            execution_dates=len(attempts), failed_dates=sum(r['status'] != 'PASS' for r in attempts),
            failure_rate=sum(r['status'] != 'PASS' for r in attempts)/len(attempts) if attempts else None,
            rank_eligible=complete, RMSE=float(np.sqrt((error**2).mean())) if complete else None,
            MAE=float(error.abs().mean()) if complete else None, Bias=float(error.mean()) if complete else None,
            seconds=total_seconds, fit_node_seconds=fit_seconds, non_fit_wall_seconds=total_seconds-fit_seconds,
            projected_pixels=sum(p.get('count', 0) for p in projections),
            finite_grid_pixels=sum(r.get('valid_pixels', 0) for r in attempts),
            grid_coverage=sum(r.get('valid_pixels', 0) for r in attempts)/sum(r.get('grid_cells', 0) for r in attempts)
                if sum(r.get('grid_cells', 0) for r in attempts) else None,
            projected_fraction=sum(p.get('count', 0) for p in projections)/sum(r.get('valid_pixels', 0) for r in attempts)
                if sum(r.get('valid_pixels', 0) for r in attempts) else None,
            minimum_before_projection=min((p['before_min'] for p in projections if 'before_min' in p), default=None),
            maximum_projection=max((p.get('max_correction', 0) for p in projections), default=0),
            quality_PASS=sum(r.get('quality', {}).get('status') == 'PASS' for r in attempts),
            quality_WARN=sum(r.get('quality', {}).get('status') == 'WARN' for r in attempts),
            quality_BLOCK_or_failure=sum(r['status'] != 'PASS' for r in attempts)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=ROOT/'output/three_experiments_20261019')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    base, out = args.base, args.output
    out.mkdir(parents=True, exist_ok=False)
    release = base/'complete_data_release'
    split = pd.read_csv(release/'station_split.csv')
    config = json.loads((release/'configuration.json').read_text(encoding='utf-8'))
    pool = ROOT/'output/capability_final_acceptance/frozen_generic_pool.json'
    assert sha(pool) == POOL_SHA
    table = pd.read_csv(release/'matched_non_test_samples.csv')
    # Fail before any fitting if the supposedly non-test table contains test IDs.
    sealed = set(config['test_stations'])
    assert not set(table.station_id) & sealed
    assert set(table.split) <= {'train', 'validation'}
    role = split.set_index('station_id').split
    assert table.split.eq(table.station_id.map(role)).all()
    assert table.unit.eq(UNIT).all() and not table.duplicated(['station_id', 'time']).any()
    history = table[table.split.eq('train') & table.time.between('2022-01-01', '2022-03-31')].copy()
    reference = pd.read_csv(base/'delivery_evidence/prepared_train.csv')
    assert len(history) == len(reference) == 290
    validation = table[table.split.eq('validation') & table.time.between('2022-04-01', '2022-08-31')].copy()
    reference_val = pd.read_csv(base/'delivery_evidence/prepared_validation.csv')
    assert set(zip(validation.station_id, validation.time)) == set(zip(reference_val.station_id, reference_val.time))
    snapshot = json.loads((release/'source_snapshot.json').read_text(encoding='utf-8'))
    by_date = {str(pd.Timestamp(Path(r['file']).stem[4:]).date()): r for r in snapshot}
    weather_path = base/'anonymous/era5_daily_2022.nc'
    weather = xr.open_dataset(weather_path).load()
    inputs = [release/'station_split.csv', release/'configuration.json', release/'matched_non_test_samples.csv',
              release/'source_snapshot.json', weather_path, pool]
    dump(out/'analysis_plan.json', dict(methods=METHODS, seed=11, train_end='2022-03-31',
        validation_window=['2022-04-01', '2022-08-31'], raw_validation_n=len(validation),
        primary='Pooled RMSE/MAE/Bias on full fixed common cohort; no method-specific deletion',
        qualification='Outcome-blind common input contracts, metre hull and full readback stencil; >=8 input stations',
        ranking='Only methods completing every qualified sample may enter full-cohort ranking',
        secondary='Coverage/rejection/failure/quality/projection/time and descriptive station/date errors',
        inference='Descriptive fixed-validation comparison; no p-values or independent-sample claims',
        training='Same available history for all; RF uses historical targets, IDW/OK/UK use current input stations',
        nonnegative_projection=config['nonnegative_projection'], test_targets_opened=False,
        cache='Shared existing DAG cache; RF fitting once, prediction and failed-execution costs counted',
        files=[dict(path=str(p), sha256=sha(p)) for p in inputs]))
    day_inputs, audit = {}, []
    # Entire cohort is fixed and saved BEFORE any numerical candidate execution.
    for date, labels in validation.groupby('time', sort=True):
        record = by_date[date]
        file = Path(record['file'])
        assert sha(file) == record['sha256']
        with xr.open_dataset(file) as opened:
            cov = opened[['aod']].load()
        cov['temperature_2m'] = weather.temperature_2m.sel(time=date).interp(lat=cov.lat, lon=cov.lon)
        cov.temperature_2m.attrs.update(units='degC', data_kind='continuous')
        cov.attrs.update(crs='EPSG:4326', provenance_verified='true', epa_local_standard_date=date)
        train = table[table.split.eq('train') & table.time.eq(date)].copy()
        audit.append(qualify(train, labels, cov))
        day_inputs[date] = (train, labels, cov)
    audit = pd.concat(audit, ignore_index=True)
    audit.to_csv(out/'sample_qualification.csv', index=False)
    qualified_keys = set(map(tuple, audit.loc[audit.reason.eq('QUALIFIED'), ['station_id', 'time']].to_numpy()))
    cohort = validation[[tuple(k) in qualified_keys for k in validation[['station_id','time']].to_numpy()]].copy()
    cohort.to_csv(out/'fixed_validation_cohort.csv', index=False)
    history.to_csv(out/'training_history.csv', index=False)
    dump(out/'cohort_freeze.json', dict(sha256=sha(out/'fixed_validation_cohort.csv'),
        n=len(cohort), dates=cohort.time.nunique(), stations=cohort.station_id.nunique(),
        fixed_before_candidate_execution=True))
    if cohort.empty:
        dump(out/'conclusion.json', dict(status='INSUFFICIENT_EVIDENCE', reason='Empty qualified common cohort'))
        return
    # Dependencies are pre-imported; no warm-up on validation targets.
    import sklearn.ensemble
    import pykrige.uk
    registry = create_default_registry()
    first = day_inputs[cohort.time.min()][2]
    template = xr.Dataset({'pm25': xr.full_like(first.aod, np.nan)}, attrs={'crs': 'EPSG:4326'})
    template.pm25.attrs.update(units=UNIT, data_kind='continuous')
    task = EPATask(registry, template, features=tuple(config['features']), train_end='2022-03-31',
                   sealed_test_stations=tuple(sealed), nonnegative_projection=config['nonnegative_projection'])
    executor = NumericalDAGExecutor(task.adapters(), version='epa-task-1')
    dump(out/'graph_contracts.json', {m:[asdict(n) for n in task.graph(m)] for m in METHODS})
    runs, predictions = [], []
    for date, labels in cohort.groupby('time', sort=True):
        train, _, cov = day_inputs[date]
        assert np.array_equal(cov.lat, template.lat) and np.array_equal(cov.lon, template.lon)
        external = dict(stations=Artifact('epa_train', train), history=Artifact('epa_history', history),
                        covariates=Artifact('epa_covariates', cov))
        for method in METHODS:
            folder = out/f'{date}_{method}'
            clock = perf_counter()
            try:
                result = task.objective(method, external, labels, folder,
                    golem_graph=golem_graph(task.graph(method)), executor=executor)
                sample = pd.read_csv(folder/'validation_predictions.csv')
                assert set(zip(sample.station_id, sample.time)) == set(zip(labels.station_id, labels.time))
                with xr.open_dataset(folder/'result.nc') as reopened:
                    result['valid_pixels'] = int(reopened.valid_domain.sum())
                    result['grid_cells'] = reopened.pm25.size
                    assert not np.isinf(reopened.pm25.values).any() and float(reopened.pm25.min()) >= 0
                sample['method'], sample['status'], sample['failure_reason'] = method, 'PASS', ''
                dump(folder/'quality_report.json', result['quality'])
            except Exception as exc:
                result = dict(status='FAIL', seconds=perf_counter()-clock, error=str(exc),
                    traceback=traceback.format_exc(), trace=getattr(exc, 'trace', []))
                sample = labels.copy()
                sample['prediction'], sample['method'], sample['status'], sample['failure_reason'] = np.nan, method, 'FAIL', str(exc)
                folder.mkdir(exist_ok=True)
                dump(folder/'failure.json', result)
            result.update(method=method, date=date, requested_n=len(labels), grid_cells=template.pm25.size)
            runs.append(result)
            predictions.append(sample)
            dump(folder/'execution.json', result)
            dump(out/'runs.json', runs)
        print(f'{date}: four configurations executed on {len(labels)} fixed samples', flush=True)
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(out/'predictions.csv', index=False)
    outcomes = []
    for method in METHODS:
        frame = validation.merge(audit, on=['station_id','time'], validate='one_to_one').merge(
            predictions[predictions.method.eq(method)][['station_id','time','prediction','status']],
            on=['station_id','time'], how='left', validate='one_to_one')
        frame['method'] = method
        frame.loc[frame.reason.ne('QUALIFIED'), 'status'] = 'NOT_ELIGIBLE'
        assert not frame.status.isna().any()
        outcomes.append(frame)
    pd.concat(outcomes, ignore_index=True).to_csv(out/'all_validation_outcomes.csv', index=False)
    pd.DataFrame([dict(method=r['method'], date=r['date'], finite_pixels=r.get('valid_pixels', 0),
                       grid_cells=r['grid_cells']) for r in runs]).to_csv(out/'grid_coverage.csv', index=False)
    summary = summarize(predictions, runs, len(cohort), len(validation))
    pd.DataFrame(summary).to_csv(out/'comparison.csv', index=False)
    dump(out/'comparison.json', summary)
    good = predictions[predictions.status.eq('PASS')]
    details = []
    for (method, station), g in good.groupby(['method','station_id']):
        error = g.prediction-g.value
        details.append(dict(method=method, station_id=station, n=len(g), dates=g.time.nunique(),
                            RMSE=float(np.sqrt((error**2).mean())), MAE=float(error.abs().mean()), Bias=float(error.mean())))
    pd.DataFrame(details).to_csv(out/'by_station.csv', index=False)
    # Paired daily errors preserve identical targets and expose temporal variability.
    daily = good.assign(error=good.prediction-good.value).groupby(['method','time']).error.agg(
        RMSE=lambda z: float(np.sqrt((z**2).mean())), MAE=lambda z: float(abs(z).mean()), Bias='mean').reset_index()
    daily.to_csv(out/'by_date.csv', index=False)
    cached_fit = [n for r in runs for n in r.get('trace', []) if n.get('operation') == 'epa.rf_fit']
    full = [s for s in summary if s['rank_eligible']]
    dump(out/'conclusion.json', dict(executions=len(runs), failures=sum(r['status']!='PASS' for r in runs),
        cohort_n=len(cohort), dates=cohort.time.nunique(), stations=cohort.station_id.nunique(),
        raw_validation_n=len(validation), raw_dates=validation.time.nunique(),
        descriptive_best=min(full,key=lambda r:r['RMSE'])['method'] if full else None,
        best_is_confirmed_generalizable=False, formal_ABC_ready=False,
        decision='PAUSE formal ABC; four configurations are exhaustively enumerable, sparse fixed spatial validation',
        input_rejections=audit.reason.value_counts().to_dict(),
        actual_rf_fit_nodes=sum(not n['cache_hit'] for n in cached_fit),
        rf_fit_cache_hits=sum(n['cache_hit'] for n in cached_fit),
        test_targets_opened=False, frozen_pool_unchanged=sha(pool)==POOL_SHA,
        cohort_unchanged=sha(out/'fixed_validation_cohort.csv')==json.loads((out/'cohort_freeze.json').read_text())['sha256'],
        elapsed_total_seconds=perf_counter()-started, objective_counters=task.counters))
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    main()
