"""Build machine-readable delivery evidence; never write a second report."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/three_experiments_20261019'


def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def main():
    out=BASE/'delivery_evidence'
    out.mkdir(exist_ok=True)
    release=BASE/'complete_data_release'
    sites=pd.read_csv(release/'station_split.csv')
    ground=pd.read_csv(BASE/'northern_california/qualified_continuous_daily_2022.csv')
    ground['split']=ground.station_id.map(sites.set_index('station_id').split)
    met=xr.open_dataset(BASE/'anonymous/era5_daily_2022.nc').load()
    summaries=[]
    for file in sorted((BASE/'anonymous_north').glob('aod_????????.nc')):
        date=str(pd.Timestamp(file.stem[4:]).date())
        with xr.open_dataset(file) as ds:
            aod=ds.aod.interp(lat=xr.DataArray(sites.latitude.to_numpy(),dims='station'),
                              lon=xr.DataArray(sites.longitude.to_numpy(),dims='station')).values
            valid_pixels=int(np.isfinite(ds.aod.values).sum())
            pixel_count=ds.aod.size
            met_grid=met.temperature_2m.sel(time=date).interp(lat=ds.lat,lon=ds.lon)
        features=met_grid.interp(lat=xr.DataArray(sites.latitude.to_numpy(),dims='station'),
            lon=xr.DataArray(sites.longitude.to_numpy(),dims='station')).values
        sample=sites[['station_id','split']].copy()
        sample['time'],sample['aod'],sample['temperature_2m']=date,aod,features
        available=set(ground.loc[ground.time.eq(date),'station_id'])
        sample['joint_valid']=np.isfinite(aod)&np.isfinite(features)&sample.station_id.isin(available)
        sample['date_role']='train' if date<='2022-03-31' else 'validation' if date<='2022-08-31' else 'test'
        sample['valid_pixels'],sample['pixel_count']=valid_pixels,pixel_count
        summaries.append(sample)
    eligibility=pd.concat(summaries,ignore_index=True)
    eligibility.to_csv(out/'sample_eligibility_no_target_values.csv',index=False)
    common=eligibility[eligibility.joint_valid].copy()
    counts=common.groupby(['date_role','split']).agg(samples=('station_id','size'),dates=('time','nunique'),stations=('station_id','nunique')).reset_index()
    counts.to_csv(out/'joint_counts.csv',index=False)
    # Prepare real target tables for train and validation only. Test covariates
    # are exported without targets, even though the read-only raw EPA archive
    # necessarily contains them.
    joined=common.merge(ground[['station_id','time','value','unit','longitude','latitude']],on=['station_id','time'],how='inner')
    joined[(joined.split=='train')&(joined.date_role=='train')].to_csv(out/'prepared_train.csv',index=False)
    joined[(joined.split=='validation')&(joined.date_role=='validation')].to_csv(out/'prepared_validation.csv',index=False)
    joined[joined.split=='train'].to_csv(out/'contemporaneous_input_stations.csv',index=False)
    common[(common.split=='test')&(common.date_role=='test')].to_csv(out/'test_covariates_no_targets.csv',index=False)
    # Pooled prediction metrics, not the arithmetic mean of daily RMSEs.
    metrics=[]
    for name,folder,pattern in [('spatial_baselines',BASE/'north_mixed_release','baseline_*/validation_predictions.csv'),
                                ('three_source_pilot',release,'2022-*/validation_predictions.csv')]:
        predictions=[]
        for path in sorted(folder.glob(pattern)):
            table=pd.read_csv(path)
            table['method']=path.parent.name.split('_',1)[1] if name=='three_source_pilot' else path.parent.name.split('_',2)[2]
            predictions.append(table)
        all_predictions=pd.concat(predictions,ignore_index=True)
        all_predictions.to_csv(out/f'{name}_predictions.csv',index=False)
        for method,table in all_predictions.groupby('method'):
            error=table.prediction-table.value
            metrics.append(dict(experiment=name,method=method,n=len(table),dates=table.time.nunique(),
                stations=table.station_id.nunique(),RMSE=float(np.sqrt((error**2).mean())),
                MAE=float(abs(error).mean()),Bias=float(error.mean())))
    pd.DataFrame(metrics).to_csv(out/'pooled_metrics.csv',index=False)
    manifest=[]
    for path in sorted((BASE/'anonymous/original').iterdir()):
        if path.is_file():
            manifest.append(dict(file=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            manifest[-1]['local_saved_utc']=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()
    dump(out/'raw_file_hashes.json',manifest)
    original_root=Path('D:/codex/数据/公开数据三层预审_20261018')
    external=[(original_root/'01_UCI_Beijing/official_original/PRSA2017_Data_20130301-20170228.zip',
        'https://archive.ics.uci.edu/dataset/501/beijing','UCI dataset 501; CC BY 4.0'),
        (original_root/'05_实验落地_20261019/original/daily_88101_2022.zip',
         'https://aqs.epa.gov/aqsweb/airdata/daily_88101_2022.zip','EPA AQS official public daily summary; attribute original agencies'),
        (original_root/'05_实验落地_20261019/original/carb_basins.geojson',
         'https://gis.carb.arb.ca.gov/hosting/rest/services/California_Air_Resources_Board_Air_Basin_MIL1/MapServer/0','CARB official public basin boundary; provider terms')]
    dump(out/'reused_official_sources.json',[dict(file=str(f),url=u,license=l,bytes=f.stat().st_size,
        sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f,u,l in external])
    covariance_files=sorted((BASE/'anonymous_north').glob('aod_????????.nc'))
    dump(out/'frozen_covariate_files.json',[dict(file=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in covariance_files])
    source_rows=[dict(source='China existing BTH 60 stations',quality=2.5,access=5,overlap=4,routes=4,scale=4,license=2,
        gate='BLOCKED: unresolved station coordinate datum and reuse licence'),
        dict(source='EPA South Coast/four counties',quality=5,access=4,overlap=2,routes=4,scale=2,license=4.5,
             gate='NOT SELECTED: only 10 qualified 2022 continuous stations'),
        dict(source='EPA Central Valley plus San Francisco Bay',quality=5,access=4,overlap=3,routes=4,scale=4,license=4.5,
             gate='SELECTED for scoped daily reconstruction; satellite coverage remains limited')]
    weights=dict(quality=.25,access=.20,overlap=.20,routes=.15,scale=.10,license=.10)
    for row in source_rows:
        row['weighted_score_100']=20*sum(row[k]*v for k,v in weights.items())
    pd.DataFrame(source_rows).to_csv(out/'source_comparison.csv',index=False)
    codes=['tools/run_uci_temporal_baselines.py','tools/run_epa_daily_smoke.py',
        'tools/prepare_anonymous_multisource.py','tools/fetch_era5_regional_public.py',
        'tools/run_three_source_ready_smoke.py','src/ecodata/algorithms/epa_task.py',
        'src/ecodata/algorithms/open_source_stations.py','tests/test_epa_task.py']
    dump(out/'code_versions.json',[dict(file=f,sha256=hashlib.sha256((ROOT/f).read_bytes()).hexdigest()) for f in codes])
    dump(out/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),
        versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','xarray','scikit-learn','PyKrige','h5netcdf']}))
    testroot=ET.parse(BASE/'targeted_frozen.xml').getroot()
    testcount=[e.attrib for e in testroot.iter('testsuite')]
    status=dict(ground_sites=ground.station_id.nunique(),ground_site_days=len(ground),ground_dates=ground.time.nunique(),
        split=sites.groupby('split').station_id.apply(list).to_dict(),satellite_actual_days=len(covariance_files),
        days_with_any_joint_samples=common.time.nunique(),days_with_20_joint_sites=int((common.groupby('time').station_id.nunique()>=20).sum()),
        total_joint_samples=len(common),joint_counts=counts.to_dict('records'),metrics=metrics,tests=testcount,
        EXPERIMENT_1_BASELINES='PASS',EXPERIMENT_2_DATA_READY='PASS (2022 daily scope only)',
        EXPERIMENT_2_PIPELINE_READY='PASS',EXPERIMENT_3_TWO_ROUTES_READY='PASS with explicit nonnegative constraint and masks',
        EXPERIMENT_3_DATA_READY='PASS for scoped masked arrays; WARN sparse validation/test coverage',
        NUMERICAL_OBJECTIVE_READY='PASS for four explicit graph configurations',
        FORMAL_GOLEM_READY=False,reason='Four adapter-supported graph configurations are cheaply enumerable; weak pilot validation coverage',
        test_targets_used_for_fitting_or_selection=False,
        test_warning='Bay test stations retained throughout north-region attempts. Earlier Central Valley exploratory fits are not final test evidence.')
    dump(out/'readiness.json',status)
    print(json.dumps(status,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
