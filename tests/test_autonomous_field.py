"""Public field entry, unaligned source contracts and true numerical edges."""
import io,json
from dataclasses import replace
import numpy as np
import pytest
import xarray as xr
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.autonomous_field import Framewise,build_context
from ecodata.algorithms.registry import AlgorithmRegistry
from ecodata.algorithms.plugins import TemporalSemanticPlugin
from ecodata.algorithms.spatiotemporal_detail import SpatiotemporalDetail
from ecodata.interface.raster_upload import ShapePreservingRaster
from ecodata.algorithms.composition_decision import PrimaryWeightedMean
from ecodata.core.schemas import EcoDataBundle


def example():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
    from scheduled_fusion_data import case
    inputs,request,truth=case(0,synthetic=True)
    request={k:v for k,v in request.items() if k not in ['composition','scoring','portrait_policy','model_profile']}
    with xr.open_dataset(io.BytesIO(inputs[1].content)) as f:times=f.time.values.copy()
    request.update(target_times=[str(t) for t in times],complementary_observation_operator=True)
    truth.attrs['split']='search_validation'
    return inputs,request,truth


def test_public_entry_discovers_real_multisource_edges_and_recomputes_objective(tmp_path,monkeypatch):
    reg=AlgorithmRegistry()
    for p in [TemporalSemanticPlugin(),ShapePreservingRaster(),SpatiotemporalDetail(),PrimaryWeightedMean()]:reg.register(p)
    monkeypatch.setattr(api,'platform_registry',lambda:reg)
    inputs,request,truth=example()
    result=api.submit_search(inputs,request,None,tmp_path,mode='AutonomousField',
        budget=dict(objective_limit=90,proposal_limit=10000,wall_seconds=30),validation=api.SearchValidation(truth,'air_temperature'))
    assert result['status']=='PASS' and result['handwritten_pool'] is False
    assert result['complete_space'] is False
    rows=[json.loads(line) for line in (tmp_path/'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
    assert any(r['algorithm']=='fusion.spatiotemporal_detail' and r.get('status')=='ELIGIBLE_OUTPUT' for r in rows)
    with xr.open_dataset(result['product']) as f:prediction=f.air_temperature.sel(time=truth.time).values
    actual=float(np.sqrt(np.mean((prediction-truth.values)**2,axis=(1,2))).mean())
    np.testing.assert_allclose(actual,result['best']['metrics']['objective'])
    assert result['kernel_calls']>=result['adapter_invocations']


def test_unknown_reference_does_not_get_a_substitute_accuracy_score(tmp_path):
    inputs,request,truth=example()
    with xr.open_dataset(io.BytesIO(inputs[0].content)) as f:two=f.isel(time=[0,2]).load()
    inputs[0]=api.SearchInput('fine.nc','primary',two.to_netcdf())
    with pytest.raises(ValueError,match='search_validation'):
        api.submit_search(inputs,request,None,tmp_path,mode='AutonomousField')
    truth.attrs['split']='test'
    with pytest.raises(ValueError,match='never final test'):api.SearchValidation(truth,'air_temperature')


def test_unaligned_inputs_still_rejected_by_ordinary_entry():
    inputs,request,_=example();parsed,_=api.parse_inputs(inputs)
    with pytest.raises(ValueError,match='align explicitly'):api.make_context(parsed,request)
    context,_,_=api.make_context(parsed,request,allow_unaligned_time=True)
    assert len(context.sources)==2


def test_batched_frame_adapter_matches_original_per_frame_values():
    from ecodata.algorithms.raster_reconstruction_task import RasterBaselineAdapter
    from ecodata.algorithms.plugins import RegridPlugin
    inputs,request,_=example();context,_,_=build_context(inputs,request)
    context=replace(context,sources=(context.sources[1],))
    upstream=RasterBaselineAdapter(RegridPlugin('bilinear',('continuous',)))
    adapter=Framewise(upstream);batch=adapter.run(context).data.dataset.air_temperature
    parts=[]
    for i,time in enumerate(context.sources[0].dataset.time.values):
        parts.append(upstream.run(adapter.frame(context,i)).data.dataset.expand_dims(time=[time]))
    np.testing.assert_allclose(batch,xr.concat(parts,dim='time').air_temperature,rtol=1e-12,atol=1e-12)
    assert adapter.frame_calls==1


def test_public_entry_hides_visible_anchor_without_external_labels(tmp_path,monkeypatch):
    reg=AlgorithmRegistry()
    for p in [TemporalSemanticPlugin(),ShapePreservingRaster(),SpatiotemporalDetail()]:reg.register(p)
    monkeypatch.setattr(api,'platform_registry',lambda:reg)
    inputs,request,_=example()
    result=api.submit_search(inputs,request,None,tmp_path,mode='AutonomousField',
        budget=dict(objective_limit=80,proposal_limit=10000,wall_seconds=20))
    assert result['status']=='PASS' and result['self_validation_from_visible_anchors']
    assert result['deployment_adapter_calls']>0
    with xr.open_dataset(tmp_path/'internal_validation.nc') as f:assert f.sizes['time']==1
    assert result['test_used_for_selection'] is False


def test_numeric_score_rejects_changed_physical_support():
    from ecodata.algorithms.autonomous_field import measured
    inputs,request,truth=example();context,_,_=build_context(inputs,request)
    output=truth.to_dataset(name='air_temperature').copy(deep=True)
    output.air_temperature.attrs['spatial_semantics']='point'
    with pytest.raises(ValueError,match='physical contract'):measured(output,truth,context)
