"""Complete finite-domain oracle after choices freeze; reuse verified real products.

No search trajectory is extended. Cached rows retain their original file and
run provenance; genuinely missing graphs execute through public Composition.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from pathlib import Path
import json

import h5py
import numpy as np
import xarray as xr

from run_common_reference_study import OUT,bundle,feedback,specs
from run_scheduled_fusion_study import load,save,sha
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.composition_decision import Composition
from ecodata.algorithms.attribute_registry import digest


def build(identifier):
    assert (OUT/'frozen_selections.json').exists(),'Never complete oracle before selection freeze'
    folder=OUT/'oracles'/identifier
    if (folder/'summary.json').exists():return load(folder/'summary.json')['reuse_receipt']
    inputs,req=bundle(identifier);parsed,_=api.parse_inputs(inputs)
    engine=Composition(parsed,req,api.platform_registry());reference=feedback(identifier)
    cache={}
    for path in (OUT/'search'/identifier).glob('*/*/summary.json'):
        d=load(path);request_receipt=load(path.parent/'request.json');runreq=request_receipt['request']
        def numerical_request(value):
            value=deepcopy(value)
            for k in ('search','parallel_scoring'):value['composition'].pop(k,None)
            return value
        assert numerical_request(runreq)==numerical_request(req)
        expected=load(OUT/'cases'/identifier/'config.json')['inputs']
        assert sorted((x['role'],x['sha256']) for x in expected)==sorted((x['role'],x['sha256']) for x in request_receipt['inputs'])
        for row in d['rows']:
            if row['status']!='PASS' or row['index'] in cache:continue
            assert row['key']==digest(engine.plan(engine.decode(row['index'])))
            assert sha(Path(row['product']))==row['product_sha256']
            cache[row['index']]=dict(row,oracle_origin='REUSED_REAL_EXECUTION',source_summary=str(path))
    rows=[];new=0
    for index in range(engine.count):
        assert engine.preflight(engine.decode(index))['status']!='REJECT'
        if index in cache:
            row=cache[index]
            with xr.open_dataset(row['product'],engine='h5netcdf') as raw:metric=reference(raw.load())
            np.testing.assert_allclose(metric['RMSE'],row['search_validation']['RMSE'],rtol=1e-9,atol=1e-9)
            row['search_validation']=metric
        else:
            genome=engine.decode(index);plan=engine.plan(genome);key=digest(plan)
            ds,_,trace,checks=engine.execute(genome)
            for obj in [ds,*ds.variables.values()]:obj.attrs={k:(str(v) if isinstance(v,(bool,dict)) else v) for k,v in obj.attrs.items() if v is not None}
            local=folder/'executions'/key;local.mkdir(parents=True,exist_ok=True);product=local/'product.nc'
            ds.to_netcdf(product,engine='h5netcdf')
            with xr.open_dataset(product,engine='h5netcdf') as raw:xr.testing.assert_allclose(ds,raw.load())
            row=dict(index=index,key=key,plan=plan,status='PASS',product=str(product),product_sha256=sha(product),
                search_validation=reference(ds),oracle_origin='ORACLE_COMPLETION_ONLY',trace=trace,coordination=checks)
            save(local/'result.json',row);new+=1
        rows.append(row)
    best=min(rows,key=lambda r:(r['search_validation']['RMSE'],r['index']))
    receipt=dict(case=identifier,total=engine.count,reused_real_products=len(cache),new_oracle_only_executions=new,
        search_choices_previously_frozen=True,mutation_trajectories_extended=False)
    save(folder/'summary.json',dict(rows=rows,final_validation_rmse=best['search_validation']['RMSE'],recommended_key=best['key'],
        certification='EXACT_VALIDATION_RMSE_OPTIMUM_IN_DECLARED_DOMAIN',reuse_receipt=receipt))
    return receipt


if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(build,x['id']) for x in specs()]):print(json.dumps(f.result()),flush=True)
