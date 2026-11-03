"""Frozen, paired-seed real-data ablations; test labels accessed after all fits.

Reuses the existing reconstruction, QP and preprocessing. No new network.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import torch
import xarray as xr

from ecodata.fusion.neural_reconstruction import (
    NeuralReconstructionConfig, default_target_coordinates, run_neural_reconstruction,
)
from ecodata.fusion.neural_experiment import evaluate_reconstruction
from run_real_pm25_three_source import PROJECT, GROUND, GEOS, ERA5, SPLIT, prepare_inputs, run_qp
from run_real_pm25_training_optimization import subset, dump

SEEDS = [17, 29, 43, 71, 101]
VARIANTS = {
    'full': {},
    'no_cross_attention': {'cross_attention': False},
    'no_observation_residual': {'observation_residual_gate': 0.0},
    'no_spectral_loss': {'spectral_weight': 0.0},
    'no_peak_loss': {'peak_weight': 0.0},
    'no_reliability': {'reliability_weighting': False},
    'guided_fno_backbone': {'backbone': 'fno'},
    'original_fno': {'backend': 'fno'},
    'original_uno': {'backend': 'multiscale_uno'},
}
LABELS = {'full': '完整观测引导UNO', 'no_cross_attention': '去背景查询Cross Attention',
    'no_observation_residual': '去局部观测残差', 'no_spectral_loss': '去频谱Loss',
    'no_peak_loss': '去峰值约束', 'no_reliability': '去可靠度加权',
    'guided_fno_backbone': '仅替换FNO骨干', 'original_fno': '原有FNO',
    'original_uno': '原有Multi-scale UNO', 'bilinear': '双线性',
    'bias_offset': '训练期常数偏差校正', 'linear': '训练期线性校正',
    'linear_idw': '线性校正+同期训练站IDW残差', 'qp': '原有QP',
    'linear_qp': '线性校正背景+原有QP'}


def digest(path):
    h = sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def measure(predictions, threshold):
    p = predictions.copy()
    e = p.predicted - p.observed
    peaks = p.groupby('time')[['observed', 'predicted']].max()
    # Sparse independent-station spatial anomalies, not full-field truth.
    anomalies = p[['observed', 'predicted']] - p.groupby('time')[['observed', 'predicted']].transform('mean')
    high = p.observed >= threshold
    spatial_corr = anomalies.corr().iloc[0, 1]
    return {'rmse': float(np.sqrt(np.mean(e**2))), 'mae': float(abs(e).mean()),
        'bias': float(e.mean()), 'correlation': float(p[['observed', 'predicted']].corr().iloc[0, 1]),
        'peak_error': float(abs(peaks.predicted - peaks.observed).mean()),
        'centered_rmse': float(np.sqrt(np.mean((e-e.mean())**2))),
        'spatial_anomaly_rmse': float(np.sqrt(np.mean((anomalies.predicted-anomalies.observed)**2))),
        'spatial_anomaly_correlation': float(spatial_corr),
        'high_value_mae': float(abs(e[high]).mean()) if high.any() else None,
        'high_value_count': int(high.sum()), 'station_count': int(p.station_id.nunique()),
        'record_count': len(p)}


def save_field(field, path):
    field.to_netcdf(path, engine='h5netcdf')
    with xr.open_dataset(path, engine='h5netcdf') as ds:
        a = ds.pm25_rh35_gcc.values
        if not np.isfinite(a).all() or (a < 0).any():
            raise RuntimeError(f'Field reread failed: {path}')


def bilinear(background):
    lat, lon = default_target_coordinates(background.pm25_rh35_gcc)
    field = background.interp(lat=lat, lon=lon, kwargs={'fill_value': 'extrapolate'})
    field.pm25_rh35_gcc.values = np.maximum(field.pm25_rh35_gcc.values, 0)
    return field


def sample_values(field, frame):
    # Vectorized sampling at actual locations, identical linear operator.
    dim = 'station_record'
    return field.pm25_rh35_gcc.interp(
        time=xr.DataArray(pd.to_datetime(frame.time).values, dims=dim),
        lat=xr.DataArray(frame.latitude.values, dims=dim),
        lon=xr.DataArray(frame.longitude.values, dims=dim)).values


def simple_baselines(train_bg, train_stations, future_bg, future_stations, output):
    start = perf_counter()
    train = train_stations[train_stations.split == 'train']
    x = sample_values(bilinear(train_bg), train)
    y = train.value.to_numpy()
    slope, intercept = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0]
    offset = float(np.mean(y-x))
    dump(output/'baseline_fit.json', {'linear_slope': slope, 'linear_intercept': intercept,
        'constant_offset': offset, 'fit_records': len(y), 'fit_station_ids': sorted(train.station_id.unique()),
        'fit_period': ['2019-08-01 00:30 UTC', '2019-08-07 23:30 UTC'],
        'target_period_or_test_used': False, 'nonnegative_clip': True,
        'idw_power': 2, 'idw_space': 'regional approximate km: cos(40deg)*longitude, latitude',
        'idw_fit': 'no fitted hyperparameters; contemporaneous train observations only'})
    fit_seconds = perf_counter()-start
    start = perf_counter()
    raw = bilinear(future_bg)
    raw_seconds = perf_counter()-start
    start = perf_counter()
    offset_field = raw.copy(deep=True)
    offset_field.pm25_rh35_gcc.values = np.maximum(raw.pm25_rh35_gcc.values + offset, 0)
    offset_seconds = perf_counter()-start
    start = perf_counter()
    corrected = raw.copy(deep=True)
    corrected.pm25_rh35_gcc.values = np.maximum(slope*raw.pm25_rh35_gcc.values + intercept, 0)
    linear_seconds = perf_counter()-start
    start = perf_counter()
    fusion = corrected.copy(deep=True)
    gx, gy = np.meshgrid(raw.lon.values, raw.lat.values)
    cosine = np.cos(np.deg2rad(40))
    grid = np.column_stack([gx.ravel()*cosine, gy.ravel()])*111.2
    live = future_stations[future_stations.split == 'train']
    for it, stamp in enumerate(pd.to_datetime(raw.time.values)):
        rows = live[live.time == stamp]
        coords = np.column_stack([rows.longitude.to_numpy()*cosine, rows.latitude])*111.2
        distance = cdist(grid, coords)
        weights = 1 / np.maximum(distance, .01)**2
        weights /= weights.sum(axis=1, keepdims=True)
        residual = rows.value.to_numpy() - sample_values(corrected, rows)
        fusion.pm25_rh35_gcc.values[it] = np.maximum(
            corrected.pm25_rh35_gcc.values[it] + (weights @ residual).reshape(gx.shape), 0)
    fields = {'bilinear': raw, 'bias_offset': offset_field, 'linear': corrected, 'linear_idw': fusion}
    elapsed = {'bilinear': raw_seconds, 'bias_offset': raw_seconds+offset_seconds,
        'linear': raw_seconds+linear_seconds, 'linear_idw': raw_seconds+linear_seconds+perf_counter()-start}
    baseline_times = {name: {'training_seconds': fit_seconds if name != 'bilinear' else 0,
        'inference_seconds': elapsed[name]} for name in fields}
    for name, field in fields.items():
        save_field(field, output/'fields'/f'{name}.nc')
    qp, qp_report = run_qp(future_bg, live)
    fields['qp'] = qp
    baseline_times['qp'] = {'training_seconds': 0, 'inference_seconds': qp_report['runtime_seconds']}
    dump(output/'logs'/'qp.json', qp_report)
    coarse = future_bg.copy(deep=True)
    coarse.pm25_rh35_gcc.values = np.maximum(slope*coarse.pm25_rh35_gcc.values+intercept, 0)
    qp_linear, report = run_qp(coarse, live)
    fields['linear_qp'] = qp_linear
    baseline_times['linear_qp'] = {'training_seconds': fit_seconds, 'inference_seconds': report['runtime_seconds']}
    dump(output/'logs'/'linear_qp.json', report)
    for name in ('qp', 'linear_qp'):
        save_field(fields[name], output/'fields'/f'{name}.nc')
    return fields, baseline_times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-index', type=int)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for folder in ('checkpoints', 'logs', 'fields', 'predictions', 'configs', 'code', 'figures'):
        (output/folder).mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    base = NeuralReconstructionConfig(backend='observation_guided_uno', epochs=30,
        hidden_channels=8, n_layers=2, scheduler='plateau', warmup_epochs=4,
        minimum_epochs=8, early_stopping_patience=5, early_stopping_min_delta=.01,
        gradient_clip_norm=1., mixed_precision=True, variable_family='pollutant_concentration')
    plan = {'project_type': 'standalone_research_task', 'stage': 'exploratory_validation',
        'seeds': SEEDS, 'variants': VARIANTS, 'base_config': asdict(base),
        'train_period': ['2019-08-01 00:30', '2019-08-07 23:30'],
        'future_period': ['2019-08-08 00:30', '2019-08-14 23:30'],
        'time_zone': 'UTC aligned centers; original ground labels Asia/Shanghai preceding hour',
        'primary_metric': 'independent-test-station RMSE, after all configurations frozen',
        'secondary_metrics': ['mae','bias','correlation','peak_error','spatial_anomaly_rmse',
            'spatial_anomaly_correlation','centered_rmse','high_value_mae','best_epoch','epochs_completed','training_seconds'],
        'seed_analysis': 'paired seeds, mean and sample SD ddof=1, report all 5; no seed selection',
        'peak_definition': 'hourly maximum over observed independent stations, not grid full-field maxima',
        'high_value_threshold': '90th percentile of training-period training stations only',
        'early_stopping': 'validation RMSE; same patience/min-delta and max epochs for all',
        'test_policy': 'all 45 fits and inference fields completed, weights frozen, then test evaluation',
        'data_conditions': {'neural': 'GEOS + ERA5 + synchronous 40 train-station observations',
            'qp': 'GEOS + synchronous 40 train-station observations; no ERA5',
            'linear_idw': 'training-period GEOS/station linear fit + synchronous 40 train residuals',
            'linear': 'training-period station/GEOS fit; no target-period observations',
            'bilinear': 'GEOS only'},
        'cross_attention_ablation_scope': 'background-conditioned main MHA only; independent local residual MHA retained',
        'backbone_ablation_scope': 'guided encoder/reliability/losses unchanged; official UNO replaced with original official FNO',
        'normalization': 'background/ERA5 training-period inputs only; train observations only in losses/encoder; inference loads frozen statistics',
        'coordinate_status': 'stored station coordinates approximately WGS84, direct datum proof absent',
        'prior_test_exposure': 'same stations have prior published exploratory scores; not a fresh untouched external confirmatory cohort',
        'missingness': 'finite three-source matched records; same evaluation keys required across all methods',
        'full_field_truth': False, 'torch_threads': 4, 'parallel_workers': args.workers,
        'timing_scope': 'wall time under at most four concurrent CPU fits; descriptive, not isolated latency benchmark',
        'September_stage': 'only after ablations and seed testing: inventory, sources and compatibility; no fabricated results'}
    plan_path = output/'experiment_plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding='utf-8')) != plan:
            raise RuntimeError('Frozen protocol mismatch; use a new output directory')
    else:
        dump(plan_path, plan)
    sources = [Path(__file__), PROJECT/'src/ecodata/fusion/neural_reconstruction.py',
        PROJECT/'src/ecodata/fusion/observation_encoder.py', PROJECT/'src/ecodata/fusion/neural_experiment.py',
        PROJECT/'tools/run_real_pm25_three_source.py', PROJECT/'tools/run_real_pm25_training_optimization.py']
    for source in sources:
        snapshot = output/'code'/source.name
        if snapshot.exists() and digest(source) != digest(snapshot):
            raise RuntimeError(f'Code changed after freeze: {source}')
        if not snapshot.exists():
            shutil.copy2(source, snapshot)
    if not (output/'provenance.json').exists():
        old = PROJECT/'output/real_pm25_training_optimization_20261012T024744Z'
        dump(output/'provenance.json', {'data': [{'path': str(p), 'sha256': digest(p)} for p in [GROUND,GEOS,ERA5,SPLIT]],
            'old_result_protected': [{'path': str(p), 'sha256': digest(p)} for p in sorted(old.rglob('*')) if p.is_file()],
            'code': [{'path': str(p), 'sha256': digest(p)} for p in sources],
            'python': sys.version, 'platform': platform.platform(),
            'packages': {p:version(p) for p in ['torch','neuraloperator','numpy','pandas','xarray','scipy','focal-frequency-loss']},
            'hardware': {'device': 'cuda' if torch.cuda.is_available() else 'cpu', 'torch_threads': 4},
            'command': f'.venv/Scripts/python.exe -u tools/run_pm25_ablation_validation.py --output "{output}"'})
    background, era, stations, _ = prepare_inputs('month',24)
    bg, aux, train_stations = subset(background,era,stations,*plan['train_period'])
    future_bg, future_aux, future_stations = subset(background,era,stations,*plan['future_period'])
    usable = train_stations[train_stations.split.isin(['train','validation'])].copy()
    future_usable = future_stations[future_stations.split.isin(['train','validation'])].copy()
    threshold = float(train_stations.loc[train_stations.split == 'train','value'].quantile(.9))
    dump(output/'data_boundary.json', {'train_station_ids': sorted(usable[usable.split=='train'].station_id.unique()),
        'validation_station_ids': sorted(usable[usable.split=='validation'].station_id.unique()),
        'test_station_ids': sorted(stations[stations.split=='test'].station_id.unique()),
        'high_value_threshold': threshold, 'train_inputs_shape': dict(bg.sizes),
        'train_records': len(usable[usable.split=='train']), 'validation_records': len(usable[usable.split=='validation']),
        'test_observations_passed_to_training': False, 'full_field_truth_passed': False})
    if args.worker_index is None and args.workers > 1:
        processes, handles = [], []
        for index in range(args.workers):
            handle = (output/'logs'/f'worker{index}_console.log').open('w',encoding='utf-8')
            handles.append(handle)
            processes.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                '--output',str(output),'--worker-index',str(index),'--workers',str(args.workers)],
                stdout=handle,stderr=subprocess.STDOUT))
        statuses = [p.wait() for p in processes]
        for handle in handles:
            handle.close()
        if any(statuses):
            raise RuntimeError(f'Workers failed: {statuses}; inspect worker console logs')
    records = []
    task_index = -1
    for method, changes in VARIANTS.items():
        for seed in SEEDS:
            task_index += 1
            if args.worker_index is not None and task_index % args.workers != args.worker_index:
                continue
            label = f'{method}_seed{seed}'
            checkpoint = output/'checkpoints'/f'{label}.pt'
            completed = output/'logs'/f'{label}_completed.json'
            config = replace(base, **changes, seed=seed, save_checkpoint_path=str(checkpoint))
            dump(output/'configs'/f'{label}.json', asdict(config))
            if completed.exists():
                record = json.loads(completed.read_text(encoding='utf-8'))
                if digest(checkpoint) != record['checkpoint_sha256']:
                    raise RuntimeError(f'Checkpoint modified: {label}')
                records.append(record)
                print(f'RESUME {label}',flush=True)
                continue
            print(f'START {label}',flush=True)
            training_log = output/'logs'/f'{label}_training.json'
            recovered = checkpoint.exists() and training_log.exists()
            if recovered:
                training_report = json.loads(training_log.read_text(encoding='utf-8'))
                seconds = training_report['runtime_seconds']
                print(f'RECOVER saved training {label}; only inference/bookkeeping repeated',flush=True)
            else:
                start = perf_counter()
                result = run_neural_reconstruction(bg,usable,auxiliaries=[aux],config=config)
                seconds = perf_counter()-start
                training_report = result.report
                dump(training_log,training_report)
                pd.DataFrame({'epoch':np.arange(1,len(result.loss_history)+1),
                    'loss':result.loss_history,'validation_rmse':training_report['validation_rmse_history'],
                    'learning_rate':training_report['learning_rate_history'],
                    'gradient_norm':training_report['gradient_norm_history']}).to_csv(output/'logs'/f'{label}_curve.csv',index=False)
            control = training_report['training_control']
            start = perf_counter()
            inferred = run_neural_reconstruction(future_bg,future_usable,auxiliaries=[future_aux],
                config=replace(config,training_mode='inference',load_checkpoint_path=str(checkpoint),save_checkpoint_path=None))
            inference_seconds = perf_counter()-start
            dump(output/'logs'/f'{label}_inference.json',inferred.report)
            save_field(inferred.field,output/'fields'/f'{label}.nc')
            inferred.station_predictions.to_csv(output/'predictions'/f'{label}_validation.csv',index=False)
            record = {'method':method,'seed':seed,'training_seconds':seconds,'inference_seconds':inference_seconds,
                'best_epoch':control['best_epoch'],'epochs_completed':control['epochs_completed'],
                'best_validation_rmse':control['best_validation_rmse'],'checkpoint_sha256':digest(checkpoint),
                'parameter_count':sum(v.numel() for name in ['model_state','encoder_state']
                    for v in torch.load(checkpoint,weights_only=False)[name].values() if isinstance(v,torch.Tensor)),
                'training_recovered_without_refit':recovered,
                'future_validation': measure(inferred.station_predictions,threshold)}
            dump(completed,record)
            records.append(record)
            progress = 'training_progress.csv' if args.worker_index is None else f'worker{args.worker_index}_progress.csv'
            pd.DataFrame([{k:v for k,v in r.items() if k!='future_validation'} for r in records]).to_csv(output/progress,index=False)
            print(f'DONE {label}: epochs={control["epochs_completed"]}, val={control["best_validation_rmse"]:.3f}, seconds={seconds:.1f}',flush=True)
    if args.worker_index is not None:
        return
    pd.DataFrame([{**{k:v for k,v in r.items() if k!='future_validation'}, **r['future_validation']} for r in records]).to_csv(output/'validation_runs.csv',index=False)
    # All weights and algorithms are now immutable; no decisions based on test.
    dump(output/'scheme_frozen_before_test.json', {'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'plan_sha256':digest(plan_path),'runs':len(records),'test_metrics_seen_in_this_run':False,
        'selection': 'all predeclared methods and seeds; no winner selected',
        'checkpoints': {str(p):digest(p) for p in sorted((output/'checkpoints').glob('*.pt'))}})
    baseline_fields, baseline_times = simple_baselines(bg,train_stations,future_bg,future_stations,output)
    final = future_stations[future_stations.split=='test'].copy()
    rows = []
    expected_keys = None
    def evaluate(field, method, seed, details):
        nonlocal expected_keys
        predicted,_ = evaluate_reconstruction(field,final)
        keys = predicted[['station_id','time']].sort_values(['station_id','time']).reset_index(drop=True)
        if expected_keys is None:
            expected_keys = keys
        elif not keys.equals(expected_keys):
            raise RuntimeError('Unequal station/time evaluation keys')
        predicted.to_csv(output/'predictions'/f'{method}_seed{seed}_test.csv',index=False)
        rows.append({'method':method,'seed':seed,**details,**measure(predicted,threshold)})
    for record in records:
        method,seed = record['method'],record['seed']
        with xr.open_dataset(output/'fields'/f'{method}_seed{seed}.nc',engine='h5netcdf') as opened:
            evaluate(opened.load(),method,seed,{k:v for k,v in record.items() if k not in ['method','seed','future_validation']})
    for method,field in baseline_fields.items():
        evaluate(field,method,-1,baseline_times[method])
    result = pd.DataFrame(rows)
    result.to_csv(output/'test_runs.csv',index=False)
    neural = result[result.seed != -1]
    measures = ['rmse','mae','bias','correlation','peak_error','centered_rmse','spatial_anomaly_rmse',
        'spatial_anomaly_correlation','high_value_mae','best_epoch','epochs_completed','training_seconds','inference_seconds']
    summary = neural.groupby('method')[measures].agg(['mean','std','min','max'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary['n_seeds'] = neural.groupby('method').size()
    summary.to_csv(output/'seed_summary.csv')
    full = neural[neural.method=='full'].set_index('seed')
    differences=[]
    for method in list(VARIANTS)[1:7]:
        ablation = neural[neural.method==method].set_index('seed')
        for metric in ['rmse','mae','peak_error','spatial_anomaly_rmse']:
            delta = ablation[metric] - full[metric]
            differences.append({'method':method,'metric':metric,'mean_delta':delta.mean(),
                'sample_sd':delta.std(ddof=1),'min_delta':delta.min(),'max_delta':delta.max(),
                'count_full_better':int((delta>0).sum()),'n':len(delta),
                'deltas_by_seed':json.dumps(delta.to_dict())})
    pd.DataFrame(differences).to_csv(output/'paired_ablation_deltas.csv',index=False)
    provenance = json.loads((output/'provenance.json').read_text(encoding='utf-8'))
    protection = [{'path':r['path'],'unchanged':digest(r['path'])==r['sha256']} for r in provenance['old_result_protected']]
    if not all(r['unchanged'] for r in protection):
        raise RuntimeError('Previous results changed')
    dump(output/'audit.json',{'old_result_protection':protection,'paired_seed_count':len(SEEDS),
        'completed_train_runs':len(records),'equal_test_keys':True,'test_records':len(expected_keys),
        'no_full_field_rmse':True,'normalization_test_observations_used':False,
        'test_freeze_file_sha256':digest(output/'scheme_frozen_before_test.json')})
    print(f'PHASE_ABLATION_COMPLETE {output}',flush=True)


if __name__ == '__main__':
    main()
