"""2015 development-input connection acceptance, no scientific effect test."""
from pathlib import Path
from copy import deepcopy
import json,hashlib
import numpy as np
import xarray as xr
from ecodata.algorithms import platform_search as api

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/complementary_five_20261029'


def run():
    old=ROOT/'output/complementary_library_20261029/real_2015'
    request=deepcopy(json.loads((old/'mechanism_config.json').read_text(encoding='utf-8'))['request'])
    request.pop('protect_observed_source',None);request['allow_training']=False
    request['parameters'].pop('calibration_source_id',None)
    paths=[old/'fine_visible_2015.nc',old/'coarse_2015.nc']
    with xr.open_dataset(paths[1]) as d:times=[str(t) for t in d.time.values]
    request.update(target_times=times);request['parameters'].update(raster_target_times=times,grid_gap_recovery=True)
    records=dict(s0=dict(operation='source',parents=[],parameters=dict(source='source0')),
        s1=dict(operation='source',parents=[],parameters=dict(source='source1')),
        d=dict(operation='reconstruction.dineof',parents=['s0'],parameters=dict(max_modes=4,internal_seed=0)),
        ft=dict(operation='temporal.raster_linear',parents=['d'],parameters={}),
        w=dict(operation='temporal.whittaker',parents=['ft'],parameters=dict(smoothing_lambda=1.)),
        ct=dict(operation='temporal.raster_linear',parents=['s1'],parameters={}),
        up=dict(operation='spatial.bilinear',parents=['ct'],parameters={}),
        g=dict(operation='fusion.guided_filter',parents=['w','up'],parameters=dict(guide_index=0,radius_pixels=2,relative_eps=.001)),
        physical=dict(operation='constraint.declared_range_projection',parents=['g'],parameters={}))
    graph=dict(records=records,key='physical')
    config=dict(inputs=[dict(path=str(p),role=role) for p,role in zip(paths,['primary','background'])],request=request,graph=graph,
        purpose='Fixed interface acceptance only; no optimizer template or accuracy advantage claim')
    (OUT/'real_development_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    sources=[api.SearchInput(p.name,role,p.read_bytes()) for p,role in zip(paths,['primary','background'])]
    result=api.execute_frozen_graph(sources,request,graph,OUT/'real_development')
    replay=api.execute_frozen_graph(sources,request,graph,OUT/'real_independent_replay')
    with xr.open_dataset(result['product']) as a,xr.open_dataset(replay['product']) as b:
        xr.testing.assert_allclose(a,b);assert np.isfinite(a.ndvi).all();shape=list(a.ndvi.shape)
    report=dict(status='PASS',scientific_advantage='NOT_EVALUATED',formal_test_used=False,development_year=2015,
        graph=graph,product=result['product'],shape=shape,registered_calls=result['cost']['used'],
        replay_matches=True,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        new_methods_executed=[e['algorithm'] for e in result['cost']['events'] if e['algorithm'] in ['reconstruction.dineof','temporal.whittaker','fusion.guided_filter']],
        other_methods='AROSICS and STARFM separately checked with controlled images and full public generated-graph deployment; NDVI is not relabelled as reflectance')
    (OUT/'real_development_acceptance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('REAL DEVELOPMENT AND INDEPENDENT REPLAY PASS',shape,result['cost']['used'])


if __name__=='__main__':run()
