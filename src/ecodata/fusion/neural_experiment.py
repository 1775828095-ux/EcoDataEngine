"""One split/grid/evaluator for interpolation, existing QP/Kriging and operators."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

from ecodata.core.exceptions import SchemaError
from ecodata.core.schemas import EcoDataBundle
from ecodata.engines.units import UnitEngine
from ecodata.fusion.neural_reconstruction import (
    NeuralReconstructionConfig, _main_variable, _station_indices, run_neural_reconstruction, split_stations,
    default_target_coordinates,
)

METHODS = ('bilinear', 'ordinary_kriging', 'qp', 'fno', 'multiscale_uno', 'observation_guided_uno')


def _correlation(actual, expected):
    if np.std(actual) == 0 or np.std(expected) == 0:
        return None
    return float(np.corrcoef(np.ravel(actual), np.ravel(expected))[0, 1])


def evaluate_reconstruction(field, stations, reference=None):
    variable = _main_variable(field)
    points = _station_indices(stations, field.time.values, field.lat.values, field.lon.values)
    rows = []
    for it, _, _, observed, station, lon, lat, _ in points:
        value = float(field[variable].isel(time=it).interp(lat=lat, lon=lon))
        rows.append({'station_id': station, 'time': pd.Timestamp(field.time.values[it]),
                     'latitude': lat, 'longitude': lon, 'observed': observed, 'predicted': value})
    predictions = pd.DataFrame(rows)
    if not np.isfinite(predictions.predicted).all():
        raise SchemaError('验证站采样产生缺失值')
    error = predictions.predicted - predictions.observed
    peaks = predictions.groupby('time')[['observed', 'predicted']].max()
    metrics = {'rmse': float(np.sqrt(np.mean(error ** 2))), 'mae': float(np.mean(abs(error))),
               'bias': float(np.mean(error)), 'station_correlation': _correlation(predictions.predicted, predictions.observed),
               'validation_peak_error': float(np.mean(abs(peaks.predicted - peaks.observed))),
               'validation_station_count': int(predictions.station_id.nunique()),
               'validation_record_count': len(predictions), 'spatial_correlation': None,
               'peak_amplitude_mae': None, 'full_field_rmse': None}
    if reference is not None:
        truth = reference[_main_variable(reference)].sel(time=field.time).transpose('time', 'lat', 'lon')
        if not np.array_equal(truth.lat, field.lat) or not np.array_equal(truth.lon, field.lon):
            raise SchemaError('参考真值必须在相同目标网格；不静默插值真值')
        truth = UnitEngine().convert(truth, field[variable].attrs['units'])
        actual, expected = field[variable].values, truth.values
        if not np.isfinite(expected).all():
            raise SchemaError('参考真值包含缺失值，需先明确有效区域掩膜')
        metrics.update(full_field_rmse=float(np.sqrt(np.mean((actual-expected)**2))),
                       peak_amplitude_mae=float(np.mean(abs(actual.max(axis=(1,2))-expected.max(axis=(1,2))))))
        correlations = [_correlation(a, b) for a, b in zip(actual, expected)]
        finite = [value for value in correlations if value is not None]
        metrics['spatial_correlation'] = float(np.mean(finite)) if finite else None
    return predictions, metrics


def run_reconstruction_comparison(background, stations, *, auxiliaries=(), evaluation_truth=None,
                                  config=NeuralReconstructionConfig(), methods=METHODS):
    if not set(methods).issubset(METHODS):
        raise SchemaError('未知基线方法')
    source_variable = _main_variable(background)
    source = background[source_variable].transpose('time', 'lat', 'lon')
    if background.attrs.get('crs') != 'EPSG:4326' or not source.attrs.get('units'):
        raise SchemaError('背景需要标准化EPSG:4326及明确单位')
    stations = stations.copy()
    engine = UnitEngine()
    if 'unit' not in stations:
        raise SchemaError('站点缺少unit')
    for unit in stations.unit.unique():
        selected = stations.unit == unit
        stations.loc[selected, 'value'] = engine.convert(xr.DataArray(stations.loc[selected, 'value'].to_numpy(), attrs={'units': unit}), source.attrs['units']).values
        if 'observation_error' in stations:
            error = engine.convert(xr.DataArray(stations.loc[selected, 'observation_error'].to_numpy(), attrs={'units': unit}), source.attrs['units'])
            zero = engine.convert(xr.DataArray(0., attrs={'units': unit}), source.attrs['units'])
            stations.loc[selected, 'observation_error'] = error.values - zero.values
    stations['unit'] = source.attrs['units']
    train, held = split_stations(stations, config.validation_station_count, config.seed)
    qa_exclusions = len(stations) - len(train) - len(held)
    before_time_filter = len(train) + len(held)
    train['time'], held['time'] = pd.to_datetime(train.time), pd.to_datetime(held.time)
    train, held = train[train.time.isin(source.time.values)].copy(), held[held.time.isin(source.time.values)].copy()
    if train.duplicated(['station_id','time']).any() or held.duplicated(['station_id','time']).any():
        raise SchemaError('同一站点/时间重复，需先明确合并规则')
    fixed_stations = pd.concat([train.assign(split='train'), held.assign(split='validation')], ignore_index=True)
    if evaluation_truth is not None:
        lat, lon = evaluation_truth.lat.values, evaluation_truth.lon.values
    else:
        lat, lon = default_target_coordinates(source)
    template = source.interp(lat=lat, lon=lon, kwargs={'fill_value': 'extrapolate'}).to_dataset(name=source_variable)
    template.attrs = dict(background.attrs)
    template.attrs.pop('grid_resolution_km', None)
    for axis in ('lat', 'lon'):
        template[axis].attrs.pop('bounds', None)
    time_bounds = source.time.attrs.get('bounds')
    if time_bounds in background.data_vars:
        template[time_bounds] = background[time_bounds].copy()
    # Verify held locations once before running any baseline.
    _station_indices(held, source.time.values, lat, lon)
    fields, results, reports, predictions = {}, [], {}, []
    for method in methods:
        started = perf_counter()
        try:
            diagnostics = {}
            if method == 'bilinear':
                field = template.copy(deep=True)
            elif method == 'ordinary_kriging':
                from ecodata.engines.fusion import SpatialFusionEngine, KrigingOptions
                gx, gy = np.meshgrid(lon, lat)
                projection = f'+proj=aeqd +lat_0={float(np.mean(lat))} +lon_0={float(np.mean(lon))} +datum=WGS84 +units=m'
                transformer = Transformer.from_crs('EPSG:4326', projection, always_xy=True)
                targets = np.asarray(transformer.transform(gx.ravel(), gy.ravel()))
                values, fits = [], []
                for stamp in pd.to_datetime(source.time.values):
                    rows = train[train.time == stamp]
                    positions = np.asarray(transformer.transform(rows.longitude.to_numpy(), rows.latitude.to_numpy()))
                    error_variance = rows.observation_error.to_numpy()**2 if 'observation_error' in rows else None
                    estimated, _, fit, _ = SpatialFusionEngine(KrigingOptions(bounded_variogram_fit=True)).estimate(positions, rows.value.to_numpy(), targets,
                        split=np.repeat('train', len(rows)), observation_error=error_variance)
                    values.append(estimated.reshape(len(lat), len(lon)))
                    fits.append(fit)
                field = template.copy(deep=True)
                field[source_variable].values = np.stack(values)
                diagnostics = {'fits': fits, 'spatial_distance': 'local AEQD metres', 'uses_background': False}
            elif method == 'qp':
                from ecodata.optimization.downscaling import ConstrainedDownscaler, DownscalingConfig
                coarse = background.copy(deep=True)
                # Numerical QP eligibility metadata; physical identity is taken
                # from the input, never inferred from filename.
                physical = source.attrs.get('physical_variable', source_variable)
                coarse[source_variable].attrs.update(data_kind='continuous', physical_variable=physical)
                observations = train.assign(variable=physical, split='train')
                result = ConstrainedDownscaler(DownscalingConfig(target_lat=tuple(lat), target_lon=tuple(lon),
                    variable=source_variable, adaptive=False)).apply(EcoDataBundle(coarse), observations)
                field = result.data.dataset
                diagnostics = result.record.as_dict()
            else:
                result = run_neural_reconstruction(background, fixed_stations, auxiliaries=auxiliaries,
                    evaluation_truth=evaluation_truth, config=replace(config, backend=method))
                field, diagnostics = result.field, result.report
            variable = _main_variable(field)
            if not np.isfinite(field[variable]).all():
                raise SchemaError('输出包含非有限值，禁止作为成功实验')
            clipped = int((field[variable].values < 0).sum())
            field[variable].values = np.maximum(field[variable].values, 0)
            station_predictions, metrics = evaluate_reconstruction(field, held, evaluation_truth)
            metrics['seconds'] = perf_counter() - started
            warning = None
            spread = float(train.value.std())
            minimum, maximum = float(field[variable].min()), float(field[variable].max())
            if minimum < float(train.value.min()) - 3*spread or maximum > float(train.value.max()) + 3*spread:
                warning = '输出超出训练观测范围±3倍标准差，可能存在不稳定空间外推；不是已证实物理错误'
            row = {'method': method, 'status': 'WARN' if warning else 'PASS', **metrics}
            fields[method] = field
            predictions.append(station_predictions.assign(method=method))
            reports[method] = {'metrics': metrics, 'diagnostics': diagnostics, 'negative_values_clipped': clipped,
                               'output_range': {'min':minimum, 'max':maximum}, 'range_warning': warning}
        except (SchemaError, ImportError, ValueError, RuntimeError) as exc:
            row = {'method': method, 'status': 'FAIL', 'reason': str(exc), 'seconds': perf_counter()-started}
            reports[method] = row
        results.append(row)
    return {'fields': fields, 'comparison': pd.DataFrame(results),
            'predictions': pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
            'report': {'created_utc': datetime.now(timezone.utc).isoformat(), 'config': asdict(config),
                       'training_station_ids': sorted(train.station_id.astype(str).unique()),
                       'validation_station_ids': sorted(held.station_id.astype(str).unique()),
                       'reference_use': 'evaluation only; shared target coordinates',
                       'dropped_noncommon_time_records': before_time_filter-len(fixed_stations),
                       'qa_excluded_records': qa_exclusions,
                       'qa_exclusions': 'INVALID/MISSING removed before shared split',
                       'baselines': reports, 'background_metadata': dict(background.attrs),
                       'station_metadata': dict(stations.attrs), 'auxiliary_metadata': [dict(ds.attrs) for ds in auxiliaries]}}


def save_comparison(experiment, directory, *, input_paths=()):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier run.
    experiment['comparison'].to_csv(directory/'comparison.csv', index=False)
    experiment['predictions'].to_csv(directory/'validation_predictions.csv', index=False)
    for method, field in experiment['fields'].items():
        field.to_netcdf(directory/f'{method}.nc', engine='h5netcdf')
        reread = xr.load_dataset(directory/f'{method}.nc', engine='h5netcdf')
        xr.testing.assert_allclose(field, reread)
    report = experiment['report']
    report['input_sources'] = [{'filename': Path(path).name, 'sha256': sha256(Path(path).read_bytes()).hexdigest()} for path in input_paths]
    report['dependencies'] = {name: version(name) for name in ('numpy', 'xarray', 'torch', 'neuraloperator', 'gstools')}
    (directory/'experiment_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
