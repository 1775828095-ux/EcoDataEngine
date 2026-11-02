from copy import deepcopy
import pytest
from ecodata.core.exceptions import SchemaError
from ecodata.models.adapters.wasp import WASPAdapter, WASPInpWriter


def minimal_config():
    """Explicit artificial fixture, not Narragansett or an accepted native case."""
    series=[{'time':'2009-01-01','value':1.0},{'time':'2009-01-02','value':1.0}]
    segment=dict(id=1,name='test',bottom_segment=0,type=1,flow_option=4,volume=100.,
                 volume_multiplier=0.,volume_exponent=0.,depth_multiplier=1.,depth_exponent=0.,
                 length=100.,slope=.001,width=1.,roughness=.04,depth_q0=.01,
                 surface_elevation=0.,bottom_elevation=0.,side_slope=0.,initial_depth=1.,weir_height=0.,weir_type=0.)
    return dict(module_type=11,start='2009-01-01',end='2009-01-02',transport_only=True,
                state_variables=[dict(tag=8,name='DO',unit='mg/L',initial={'1':5.0},solids_field=0,solids_density=1.,max_concentration=100.,dissolved_fraction=1.)],
                segments=[segment],bed_volume_option=0,bed_compaction_step=0.,
                flow_paths=[dict(field=1,name='test',unit='m^3/s',routes=[{'from':0,'to':1,'fraction':1.},{'from':1,'to':0,'fraction':1.}],series=series)],
                boundaries=[dict(variable='DO',segment_id=1,unit='mg/L',series=deepcopy(series))],
                loads=[dict(variable='DO',source_id='test',segment_id=1,unit='kg/day',series=deepcopy(series))],
                time_functions=[],kinetic_parameters={},solver=dict(mass_balance=0,solution=1,negative=0,restart=0,time_optimization=1,wq_linkage=0,topt=.9,min_step=.003,max_step=1.,print_interval=0.))


def test_package_never_claims_native_on_file_generation():
    import json
    files=WASPAdapter().package(minimal_config())
    assert {'model.inp','manifest.json','validation_report.json','timeseries/point_source_load.csv'}<=set(files)
    report=json.loads(files['validation_report.json'])
    assert not report['native_model_input_verified'] and not report['model_format_matched']
    assert b'Segment Information' in files['model.inp']


@pytest.mark.parametrize('change', ['segment','initial','kinetics','unit','time','variable','hyd'])
def test_missing_physical_configuration_blocks(change):
    c=minimal_config()
    if change=='segment':c['loads'][0]['segment_id']=12
    if change=='initial':c['state_variables'][0]['initial']={}
    if change=='kinetics':c['kinetic_parameters']=None
    if change=='unit':c['loads'][0]['unit']='mg/L'
    if change=='time':c['flow_paths'][0]['series']=c['flow_paths'][0]['series'][1:]
    if change=='variable':c['loads'][0]['variable']='NH3'
    if change=='hyd':c['hydrodynamics_file']='missing.hyd'
    with pytest.raises(SchemaError):WASPInpWriter().write(c)
    assert 'model.inp' not in WASPAdapter().package(c)


def test_no_overwrite_existing_package(tmp_path):
    adapter=WASPAdapter();adapter.save_package(tmp_path,minimal_config())
    with pytest.raises(SchemaError):adapter.save_package(tmp_path,minimal_config())
