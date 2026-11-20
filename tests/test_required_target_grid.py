"""Target-grid behavior tests; synthetic fields are not fine-scale truth."""
from test_desktop_simple_flow import window
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from types import SimpleNamespace
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.spatial_fusion_task import SpatialTask,SpatialSearch
from ecodata.spatial.regridding import resolve_target_grid,require_target_grid,target_grid_spec

SPEC=dict(crs='EPSG:4326',latitude_step=.1,longitude_step=.1,extent='intersection')
BUDGET=dict(objective_limit=130,total_call_limit=180,proposal_limit=60,wall_seconds=45)


def fields():
    result=[]
    for n in (5,9):
        lat=30+np.linspace(0,2,n);lon=110+np.linspace(0,2,n);xx,yy=np.meshgrid(lon,lat)
        a=np.stack([xx+2*yy+t for t in range(4)])
        ds=xr.Dataset(dict(temperature=(('time','lat','lon'),a)),coords=dict(time=pd.date_range('2000-01-01',periods=4),lat=lat,lon=lon),attrs=dict(crs='EPSG:4326',source_id=str(n)))
        ds.temperature.attrs.update(units='K',physical_variable='temperature',physical_definition='cell temperature',spatial_semantics='cell_mean',temporal_semantics='instantaneous',data_kind='continuous')
        result.append(ds)
    return result


def task_and_inputs():
    ds=fields();items=[api.SearchInput(str(i)+'.nc','primary' if i==0 else 'background',d.to_netcdf()) for i,d in enumerate(ds)]
    q=dict(task='multisource',variable='temperature',unit='K',physical_variable='temperature',semantic_type='continuous',temporal_semantics='instantaneous',processing_goal='spatial_fusion',target_grid=SPEC,search_stopping=dict(enabled=False))
    parsed,audit=api.parse_inputs(items)
    return SpatialTask.build(parsed,q,audit),items,q


def test_geometry_hard_gate_rejects_original_coarse_shape_and_bad_crs():
    ds=fields();target=resolve_target_grid(ds,SPEC)
    with pytest.raises(ValueError,match='硬约束'):require_target_grid(ds[0],target,'temperature')
    with pytest.raises(ValueError):target_grid_spec(dict(SPEC,latitude_step=0))
    with pytest.raises(ValueError):target_grid_spec(dict(SPEC,crs='EPSG:3857'))


def test_complete_grid_is_executed_before_scoring_original_reference_cells(tmp_path):
    task,_,_=task_and_inputs();engine=SpatialSearch(task,tmp_path,BUDGET,11)
    assert task.reference_grid.sizes['lat']==3 and task.template.sizes['lat']>5
    assert len(engine.folds[0]['query'])<engine.folds[0]['cohort']['query_count']
    engine.execute(engine.leaves[0])
    row=engine.rows[0]
    assert row['status']=='PASS' and row['target_grid_check']['status']=='PASS'
    assert row['target_grid_check']['shape'][1:]==[task.template.sizes['lat'],task.template.sizes['lon']]
    assert row['coarse_reference_diagnostic']['count']==sum(np.isfinite(f['reference']).sum() for f in engine.folds)
    assert row['metrics']['count']==engine.evaluation_plan['selection_count']
    assert row['evaluation_mode']=='REDUCED_SCALE_PRODUCT_RECONSTRUCTION'
    product,_=engine.product(row['key']);require_target_grid(product,task.template,'temperature')


def test_wrong_candidate_shape_cannot_receive_fitness_or_be_selected(tmp_path):
    task,_,_=task_and_inputs();engine=SpatialSearch(task,tmp_path,BUDGET,11)
    class Wrong:
        def run(self,*a):return dict(trace=[],outputs=dict(field=SimpleNamespace(value=np.zeros((4,9)))))
    engine.executors[0]=Wrong();engine.execute(engine.leaves[0])
    assert engine.rows[0]['status']=='FAILED' and 'target-grid' in engine.rows[0]['error']
    assert 'metrics' not in engine.rows[0] and engine.best() is None


def test_public_search_evaluates_and_delivers_only_one_fine_grid(tmp_path):
    from ecodata.interface.search_page import process_search_upload
    task,items,q=task_and_inputs()
    result=process_search_upload(items,q,None,tmp_path/'run',budget=BUDGET,seed=11)
    assert result['status']=='PASS'
    assert result['target_grid_contract']['reference_role'].startswith('Original native')
    rows=json.loads((tmp_path/'run/candidate_results.json').read_text())
    assert len(rows)>1 and all(r.get('target_grid_check',{}).get('status')=='PASS' for r in rows if r['status']=='PASS')
    with xr.open_dataset(result['product']) as out:require_target_grid(out,task.template,'temperature')


def test_page_target_grid_is_explicit_and_invalidates_old_checks(window):
    window.set_decision_model('data_only')
    assert window.task_requirements()=={}
    window.checked_decisions=[{'status':'PASS'}]
    window.target_grid_input.setText('0.1,0.1')
    assert window.task_requirements()['target_grid']==SPEC
    assert not window.checked_decisions
    window.target_grid_input.setText('0,0.1')
    with pytest.raises(ValueError):window.task_requirements()
    window.set_decision_model('swat_plus')
    assert window.task_requirements()=={}


def test_supported_intersection_does_not_create_boundary_nodata():
    from ecodata.spatial.regridding import GridRegridder,RegridConfig
    from ecodata.core.schemas import EcoDataBundle
    ds=fields();target=resolve_target_grid(ds,SPEC)
    for source in ds:
        result=GridRegridder(RegridConfig('conservative',tuple(target.lat.values),tuple(target.lon.values),'temperature')).apply(EcoDataBundle(source))
        require_target_grid(result.data.dataset,target,'temperature')
        assert np.isfinite(result.data.dataset.temperature).all()
