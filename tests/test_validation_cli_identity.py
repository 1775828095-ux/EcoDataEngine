"""python -m must instantiate the canonical, strictly checked validation class."""
from pathlib import Path
import json,runpy,sys
import xarray as xr
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from scheduled_fusion_data import case
from test_full_screen_spatiotemporal import publish,stores


def test_module_cli_uses_canonical_validation_identity(tmp_path,monkeypatch):
    publish();inputs,req,truth=case(0,synthetic=True)
    c=req['composition'];c['source_stages']['coarse_fast']=[dict(id='grid',options=[dict(algorithm='raster_task.spatial.bilinear')],evidence='CLI mechanism test')]
    c['fusion_options']=c['fusion_options'][:1];c['post_stages']=[];c['search'].update(backend='none',generations=0,survivors=1)
    files=[]
    for item in inputs:
        p=tmp_path/item.filename;p.write_bytes(item.content);files.append(dict(path=str(p),role=item.role))
    labels=truth.copy();labels.attrs['split']='search_validation';labels.to_dataset(name=req['variable']).to_netcdf(tmp_path/'labels.nc',engine='h5netcdf')
    config=dict(inputs=files,request=req,mode='Profile',search_validation=dict(path=str(tmp_path/'labels.nc')),budget=dict(objective_limit=1,wall_seconds=120))
    path=tmp_path/'config.json';path.write_text(json.dumps(config),encoding='utf-8');folder=tmp_path/'cli'
    monkeypatch.setattr(sys,'argv',['platform_search',str(path),str(folder)])
    runpy.run_module('ecodata.algorithms.platform_search',run_name='__main__',alter_sys=True)
    report=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    assert report['best'] and report['search_validation_used_for_selection']
    assert not report['independent_test_used_for_selection']
