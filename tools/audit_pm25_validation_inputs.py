"""Independent training-only data boundary and checkpoint metadata audit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from run_real_pm25_three_source import GROUND,GEOS,ERA5,SPLIT
from run_pm25_ablation_validation import VARIANTS


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    split=pd.read_csv(SPLIT,dtype={'station_id':str})
    ids={name:set(split.loc[split.split==name,'station_id']) for name in ['train','validation','test']}
    assert len(split)==60 and split.station_id.is_unique
    assert all(ids[x].isdisjoint(ids[y]) for x,y in [('train','validation'),('train','test'),('validation','test')])
    coords={name:set(map(tuple,split.loc[split.split==name,['longitude','latitude']].to_numpy())) for name in ids}
    assert all(coords[x].isdisjoint(coords[y]) for x,y in [('train','validation'),('train','test'),('validation','test')])
    frame=pd.read_parquet(GROUND)
    frame['station_id']=frame.station_id.astype(str)
    frame['geos_average_center_utc']=pd.to_datetime(frame.geos_average_center_utc)
    # No test label statistics are calculated here.
    train=frame[frame.station_id.isin(ids['train']) & frame.geos_average_center_utc.between(
        '2019-08-01 00:30','2019-08-07 23:30') & frame.three_source_complete & frame.pm25_as_stored.notna()].copy()
    assert not train.duplicated(['station_id','geos_average_center_utc']).any()
    difference=train.pm25_rh35_gcc-train.pm25_as_stored
    completeness=train.groupby('station_id').geos_average_center_utc.nunique()/168
    times=pd.date_range('2019-08-01 00:30','2019-08-07 23:30',freq='h')
    with xr.open_dataset(GEOS,engine='h5netcdf') as bg,xr.open_dataset(ERA5,engine='h5netcdf') as era:
        assert pd.Index(times).isin(bg.time.values).all()
        assert pd.Index(times+pd.Timedelta(minutes=30)).isin(era.valid_time.values).all()
        summary={'split_ids_unique_and_disjoint':True,'all_split_coordinates_disjoint':True,
            'train_rows':len(train),'test_labels_summarized':False,
            'training_station_mean':float(train.pm25_as_stored.mean()),
            'training_geos_mean':float(train.pm25_rh35_gcc.mean()),
            'training_geos_minus_station_bias':float(difference.mean()),
            'training_geos_minus_station_rmse':float(np.sqrt(np.mean(difference**2))),
            'train_completeness':completeness.to_dict(),
            'reliability_limitation':'no measurement QA or error variance supplied; weighting tests availability completeness only',
            'grid':{'lat':bg.lat.values.tolist(),'lon':bg.lon.values.tolist()},
            'geos_units':bg.pm25_rh35_gcc.attrs.get('units'),
            'era_units':{name:era[name].attrs.get('units') for name in ['t2m','d2m','u10','v10','sp','tp']},
            'time_alignment':{'geos':'hour mean centered HH:30 UTC','ground':'preceding-hour mean; local end label UTC+8',
                'era':'valid_time = GEOS center +30 minutes; tp ending-hour accumulation'},
            'checkpoint_checks':[]}
    for path in sorted((a.output/'checkpoints').glob('*.pt')):
        c=torch.load(path,map_location='cpu',weights_only=False)
        r=json.loads((a.output/'logs'/f'{path.stem}_training.json').read_text(encoding='utf-8'))
        assert set(r['training_station_ids'])==ids['train']
        assert set(r['validation_station_ids'])==ids['validation']
        assert c['channel_statistics']==r['normalization']['channel_statistics']
        assert r['test_station_records_excluded']==0
        assert not r['has_independent_full_field_evaluation']
        summary['checkpoint_checks'].append({'file':path.name,'station_boundary_ok':True,
            'checkpoint_and_training_normalization_match':True,'channels':c['channel_names']})
    summary['single_factor_config_checks']=[]
    frozen=json.loads((a.output/'experiment_plan.json').read_text(encoding='utf-8'))['base_config']
    for path in sorted((a.output/'configs').glob('*.json')):
        name=path.stem.split('_seed')[0]
        if name not in list(VARIANTS)[:7]:
            continue
        config=json.loads(path.read_text(encoding='utf-8'))
        changed={k:v for k,v in config.items() if k not in ['seed','save_checkpoint_path'] and v!=frozen[k]}
        assert changed==VARIANTS[name],(path.name,changed)
        summary['single_factor_config_checks'].append({'file':path.name,'changed_keys':sorted(changed),
            'one_module_or_full_model':True})
    (a.output/'independent_input_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Training-only bias',summary['training_geos_minus_station_bias'])
    print('Completeness range',completeness.min(),completeness.max())
    print('Checkpoints audited',len(summary['checkpoint_checks']))


if __name__=='__main__':main()
