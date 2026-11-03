"""Robit mutation supplement through the public platform, never test feedback."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import sys
from copy import deepcopy

from ecodata.algorithms import platform_search as api
from ecodata.algorithms.numerical_search import save
from ecodata.models import portrait_policy

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'output/robit_era5_fusion_20261027'
OUT=ROOT/'output/robit_era5_mutation_20261028'
VERSION='robit-era5-multisource-swatplus-v1'
load=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    attribute=ROOT/'output/robit_single_source_20261026/attributes/robit-daily-state-v1.json'
    target=ROOT/'data/attribute_registries'/attribute.name
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():assert target.read_bytes()==attribute.read_bytes()
    else:target.write_bytes(attribute.read_bytes())
    if not (portrait_policy.ROOT/(VERSION+'.json')).exists():
        old=load(ROOT/'output/robit_single_source_20261026/models/robit-swatplus-input-v1.json')['snapshot']
        scenario=deepcopy(old['scenarios'][0]);scenario['task']='multisource'
        portrait_policy.publish(VERSION,'swat_plus',[scenario],evidence='Reuse original Robit preparation issues and equal experimental priorities for multisource recovery; not measured SWAT sensitivity')
    hashes={}
    for kind in ('random','short','long'):
        config=load(OLD/'cases'/kind/'config_v5.json')
        config['request'].update(model_profile=dict(id='swat_plus',role='weather',field='precipitation'),
             portrait_policy=dict(version=VERSION,scenario='daily-recovery'),scoring=dict(registry_version='robit-daily-state-v1'))
        save(OUT/'cases'/kind/'config.json',config)
        for item in config['inputs']:hashes[item['path']]=hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()
    save(OUT/'input_hashes.json',hashes)
    save(OUT/'protocol.json',dict(prior=load(OLD/'protocol_v6.json'),date='2026-10-28',
          mode='PortraitMutation',optimizer='Existing ALNS 7 + AlphaUCB + HillClimbing, open-DAG adapter',
          seed_order=[11,29,47],controls=['alns','random','none'],test_already_inspected_in_prior_round=True,
          same_data_supplement_not_new_blind_validation=True,reference_used_for_portrait_screening=False))
    print('PREPARED',flush=True)


def run(kind,seed,control='alns',diagnostic=False,self_validate=False):
    folder=OUT/('diagnostic_v6' if diagnostic else 'runs')/(kind+'_'+str(seed)+'_'+control+('_self' if self_validate else ''))
    if (folder/'summary.json').exists():return load(folder/'summary.json')
    config=load(OUT/'cases'/kind/'config.json')
    config.update(mode='PortraitMutation',seed=seed,budget=load(OLD/'protocol_v6.json')['budget'])
    config['request']['mutation_control']=control
    if self_validate:
        config['request'].pop('portrait_policy',None);config['request'].pop('scoring',None)
    if not self_validate:config['search_validation']=dict(path=str(OLD/'cases'/kind/('development_labels.nc' if diagnostic else 'validation_labels.nc')))
    save(folder/'client_config.json',config)
    return api.submit_file_request(config,folder)


def freeze():
    target=OUT/'source_manifest.json'
    if target.exists():
        for path,digest in load(target).items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
        return
    paths=list((ROOT/'src/ecodata').rglob('*.py'))+list((ROOT/'src/ecodata').rglob('*.json'))
    paths+=list((ROOT/'data/model_portraits').glob('*.json'))+list((ROOT/'data/attribute_registries').glob('*.json'))
    paths+=[ROOT/'tools'/name for name in ['run_robit_era5_mutation.py','evaluate_robit_era5_mutation.py','audit_robit_era5_mutation.py']]
    hashes={}
    for path in paths:
        relative=path.relative_to(ROOT);destination=OUT/'source_frozen'/relative
        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(path.read_bytes())
        hashes[relative.as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    save(target,hashes)
    print('SOURCE FROZEN',len(hashes),flush=True)


if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='prepare':prepare()
    elif stage=='diagnostic':
        r=run('short',11,diagnostic=True)
        print(json.dumps({k:r.get(k) for k in ['status','executed_candidates','mutation_new_executions','evolution_used','elapsed_seconds','optimizer']},ensure_ascii=False))
    elif stage=='self':
        r=run('short',11,diagnostic=True,self_validate=True);print(r['status'],r['self_validation_from_visible'])
    elif stage=='batch':
        freeze()
        with ProcessPoolExecutor(max_workers=3) as pool:
            jobs={pool.submit(run,kind,seed,control):(kind,seed,control) for kind in ('random','short','long')
                  for seed in (11,29,47) for control in ('alns','random','none')}
            for job in as_completed(jobs):
                r=job.result();print(*jobs[job],r['status'],r['executed_candidates'],r['mutation_new_executions'],r['best']['metrics']['objective'] if r['best'] else None,flush=True)
