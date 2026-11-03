"""Read back bounded acceptance artifacts; never executes search."""
from pathlib import Path
import json,hashlib,shutil,zipfile,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,xarray as xr
from ecodata.algorithms.numerical_search import save
OUT=ROOT/'output/problem_directed_20261021'

def load(path):return json.loads(path.read_text(encoding='utf-8'))
def scrub(value):
    if isinstance(value,dict):return {k:scrub(v) for k,v in value.items() if k not in {'seconds','decision_seconds','elapsed'}}
    if isinstance(value,list):return [scrub(v) for v in value]
    return value

def main():
    records={};pairs={}
    for task in ['spatial','temporal','multisource']:
        runs=[]
        for client in ['toolkit','page']:
            folder=OUT/'runs'/f'{task}_{client}_bounded_v4';d=load(folder/'summary.json')
            transitions=load(folder/'operation_transitions.json')['records']
            for r in d['rows']:
                if r['status']=='PASS':
                    with xr.open_dataset(r['product'],engine='h5netcdf') as ds:assert np.isfinite(ds[d['task']=='multisource' and 'pm25' or ('aod' if task=='spatial' else 'tp')]).all()
            records[task+'_'+client]=dict(counts=d['counts'],objective=d['best']['objective'],seconds=d['seconds'],candidate_generation_seconds=d['candidate_generation_seconds'],
                preparation_costs=d['preparation_costs'],paired_rewards=sum(r['reward'] is not None for r in transitions),
                attributable_labels=sum(r['single_action_learning_label'] for r in transitions),actual_actions=sorted({r['action']['type'] for r in transitions}),
                new_graphs=d['unique_valid'],source= str(folder))
            runs.append((folder,d))
        left,right=runs
        assert scrub(load(left[0]/'decision.json'))==scrub(load(right[0]/'decision.json'))
        assert left[1]['candidate_domain']==right[1]['candidate_domain']
        assert scrub(left[1]['proposals'])==scrub(right[1]['proposals'])
        assert left[1]['best']['graph']==right[1]['best']['graph']
        with xr.open_dataset(left[1]['best']['product'],engine='h5netcdf') as a,xr.open_dataset(right[1]['best']['product'],engine='h5netcdf') as b:xr.testing.assert_equal(a,b)
        pairs[task]='PASS: same semantics, domain, proposed actions/probabilities, selected graph and numerical product'
    downloaded=Path('C:/Users/LX/Downloads/platform_search (1).zip')
    with zipfile.ZipFile(downloaded) as archive:
        assert archive.testzip() is None
        listing=archive.namelist();assert any(x.endswith('product.nc') for x in listing)
    before=load(OUT/'before_manifest.json');sources={}
    for relative in before:
        if relative.startswith('src/'):
            path=ROOT/relative;destination=OUT/'source_executed'/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
            sources[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    for relative in ['tools/accept_problem_directed.py','tools/verify_problem_directed.py','tests/test_problem_directed.py']:
        path=ROOT/relative;destination=OUT/'source_executed'/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
        sources[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    save(OUT/'verification.json',dict(runs=records,client_pairs=pairs,browser=dict(request_directory='output/platform_search_requests/eafde9acb0f94ce69ee64ab5bac3ebd4',download=str(downloaded),zip_sha256=hashlib.sha256(downloaded.read_bytes()).hexdigest(),members=listing),source_fingerprints=sources,q_model_trained=False))
    print(json.dumps(records,indent=2));print(pairs)

if __name__=='__main__':main()
