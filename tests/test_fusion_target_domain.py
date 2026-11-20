"""Geometry-only functional cases; no environmental accuracy evidence."""
import numpy as np
import xarray as xr
import pytest
from test_required_target_grid import task_and_inputs
from ecodata.evaluation.fusion_protocol import prepare
from ecodata.spatial.regridding import contained_cell_grid
from ecodata.algorithms.raster_reconstruction_task import edges


def test_center_inside_but_partial_cell_outside_is_excluded():
    source=xr.Dataset(coords=dict(lat=[-.1,.4,.9,1.4,1.9,2.4],lon=[-.1,.4,.9,1.4,1.9,2.4]))
    target=xr.Dataset(coords=dict(lat=[.25,.75,1.25,1.75],lon=[.25,.75,1.25,1.75]))
    result=contained_cell_grid(source,target)
    assert np.allclose(result.lat,[.4,.9,1.4])
    assert np.allclose(result.lon,[.4,.9,1.4])
    assert 1.9 not in result.lat


def test_impossible_complete_support_is_explicitly_rejected():
    source=xr.Dataset(coords=dict(lat=[0.,1.,2.],lon=[0.,1.,2.]))
    target=xr.Dataset(coords=dict(lat=[.9,1.1],lon=[.9,1.1]))
    with pytest.raises(ValueError,match='fully cover'):contained_cell_grid(source,target)


def test_assessment_and_coarse_observation_share_target_support_contract():
    task,_,_=task_and_inputs();plan,r=prepare(task,task.times,11)
    assert r is not None and plan['version']=='fusion_evidence_v2_target_domain'
    audit=plan['spatial_domain']
    assert audit['reference_cells_retained']<audit['reference_cells_before']
    assert audit['outside_cells']==0
    for axis in ('lat','lon'):
        lower,upper=audit['application_bounds'][axis];b=edges(audit['reference_axes'][axis])
        assert np.all(b>=lower-1e-9) and np.all(b<=upper+1e-9)
    expected=contained_cell_grid(task.native_fields[0],task.template)
    np.testing.assert_array_equal(expected.lat,task.reference_grid.lat)
    np.testing.assert_array_equal(expected.lon,task.reference_grid.lon)
    assert set(r['selected']).isdisjoint(r['test'])
    assert sorted([*r['selected'],*r['test']])==list(range(r['reference'].shape[1]))


def test_domain_and_split_are_reproducible_and_do_not_read_values():
    task,_,_=task_and_inputs();p,a=prepare(task,task.times,11)
    task.native_fields[1]=task.native_fields[1].copy(deep=True)
    task.native_fields[1].temperature.values[:]+=5
    q,b=prepare(task,task.times,11)
    assert p['spatial_domain']==q['spatial_domain'] and p['location_split']==q['location_split']
    assert p['fixed_support_sha256']==q['fixed_support_sha256']
    assert not np.array_equal(a['reference'],b['reference'])


def test_same_count_different_coordinates_invalidate_support_fingerprint():
    task,_,_=task_and_inputs();p,_=prepare(task,task.times,11)
    task.native_fields=[d.assign_coords(lon=d.lon+2) for d in task.native_fields]
    task.template=task.template.assign_coords(lon=task.template.lon+2)
    q,_=prepare(task,task.times,11)
    assert p['selection_count']==q['selection_count']
    assert p['fixed_support_sha256']!=q['fixed_support_sha256']


def test_fixed_bank_reports_all_dates_and_never_calls_search(tmp_path,monkeypatch):
    from ecodata.algorithms import platform_search as api
    from ecodata.algorithms.spatial_fusion_task import SpatialSearch,evaluate_bank
    from test_required_target_grid import BUDGET
    import ecodata.algorithms.composition_optimizers as optimizers
    def forbidden(*args,**kwargs):raise AssertionError('Fixed assessment must not search')
    monkeypatch.setattr(optimizers,'random_graph_search',forbidden)
    task,inputs,request=task_and_inputs()
    builder=SpatialSearch(task,tmp_path/'builder',BUDGET,11)
    single=builder.leaves[1];multi=builder.intern('spatial.blend',builder.leaves[:2],dict(weight=.75))
    graphs={name:dict(key=key,records=builder.records) for name,key in [('single',single),('multi',multi)]}
    parsed,audit=api.parse_inputs(inputs)
    result=evaluate_bank(parsed,request,audit,graphs,tmp_path/'audit',budget=BUDGET,seed=11)
    assert result['selection_performed'] is False
    assert result['counts']==dict(evaluated=2,proposals=0)
    assert result['control_keys']==dict(single=single,multi=multi)
    assert len(result['per_date'])==2*4*3
    assert all(row['used_for_selection'] is False for row in result['per_date'])
    assert all(row['used_for_selection'] is False for row in result['full_domain_controls'].values())
    assert result['evaluation_protocol']['spatial_domain']['outside_cells']==0
