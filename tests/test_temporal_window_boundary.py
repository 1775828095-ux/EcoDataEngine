"""An unpredicted boundary timestamp must not hide a declared seam gate."""
from copy import deepcopy
from pathlib import Path
import sys
import xarray as xr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from scheduled_fusion_data import case
from ecodata.algorithms import platform_search as api
from test_full_screen_spatiotemporal import publish,stores


def test_missing_boundary_timestamp_still_checks_real_neighbor_jump(tmp_path):
    publish();inputs,req,_=case(0,synthetic=True)
    req['composition']['source_stages']['coarse_fast']=[dict(id='grid',options=[dict(algorithm='raster_task.spatial.bilinear')],evidence='Single interpolation mechanism test')]
    req['composition']['fusion_options']=req['composition']['fusion_options'][:1]
    req['composition']['post_stages']=[];req['composition']['search'].update(backend='none',generations=0,survivors=1)
    result=api.submit_search(inputs,req,None,tmp_path/'ungated',mode='Profile',budget=dict(objective_limit=1,wall_seconds=120))
    with xr.open_dataset(result['best']['product'],engine='h5netcdf') as ds:
        jump=float(abs(ds.air_temperature.isel(time=7)-ds.air_temperature.isel(time=6)).max())
    assert jump>0
    checked=deepcopy(req);checked['composition']['coordination']=dict(max_seam_jump=jump/2,evidence='Synthetic gate sensitivity test, not an empirical physical threshold')
    gated=api.submit_search(inputs,checked,None,tmp_path/'gated',mode='Profile',budget=dict(objective_limit=1,wall_seconds=120))
    assert gated['best'] is None
    assert 'max_seam_jump' in gated['rows'][0]['error']
