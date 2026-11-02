"""Independent physical expectations, rather than self-roundtrip alone."""
from pathlib import Path
import numpy as np
import pytest
from ecodata.models import RequirementStore
from ecodata.models.scenario import recognise_inputs,suggest_roles
from ecodata.models.adapters.cmaq import execute_scenario,plan_scenario

def case():
    directory=Path(__file__).resolve().parents[1]/'examples/cmaq_case/input'
    inputs=recognise_inputs([(p.name,p.read_bytes()) for p in directory.iterdir()],{'pm25_station.csv':{'temporal_semantics':'mean'}})
    profile=RequirementStore().get('cmaq');roles=suggest_roles(profile,inputs)
    return profile,inputs,roles,next(i.dataset for i in inputs if i.filename==roles['meteorology'])

def test_explicit_wind_mapping_takes_priority_over_optional_vector_components():
    profile,inputs,roles,met=case()
    met['wind']=met.temperature*0+23.
    met.wind.attrs={'units':'m/s'}
    result=execute_scenario(profile,inputs,roles)
    np.testing.assert_allclose(result.output.wind,23.,rtol=0,atol=1e-10)

@pytest.mark.parametrize('field',['humidity','wind','u10m'])
def test_plan_rejects_extra_vertical_dimensions_for_every_meteorological_field(field):
    profile,inputs,roles,met=case()
    if field=='wind':
        met['wind']=met.temperature*0+1;met.wind.attrs={'units':'m/s'}
    if field=='u10m' and 'wind' in met:
        del met['wind'] # Force the configured mapping to use derived components.
    met[field]=met[field].expand_dims(layer=[1,2])
    _,plan=plan_scenario(profile,inputs,roles)
    assert not plan['can_process']
    assert any('垂直层' in issue for issue in plan['errors'])

def test_plan_rejects_mixed_time_mean_and_instantaneous_fields():
    profile,inputs,roles,met=case()
    met.humidity.attrs['temporal_semantics']='instantaneous'
    _,plan=plan_scenario(profile,inputs,roles)
    assert not plan['can_process']
    assert any('时间语义' in issue for issue in plan['errors'])

def test_projected_observations_reproject_to_same_geographic_fusion_result():
    from rasterio.warp import transform
    profile,inputs,roles,met=case()
    expected=execute_scenario(profile,inputs,roles).output.PM25.values.copy()
    observation=next(i.dataset for i in inputs if i.filename==roles['observation'])
    x,y=transform('EPSG:4326','EPSG:3857',observation.longitude.values,observation.latitude.values)
    observation['longitude']=(observation.longitude.dims,x)
    observation['latitude']=(observation.latitude.dims,y)
    observation.attrs['crs']='EPSG:3857'
    actual=execute_scenario(profile,inputs,roles).output.PM25.values
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-9)
