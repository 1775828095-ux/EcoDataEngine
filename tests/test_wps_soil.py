import copy,json
from pathlib import Path
import numpy as np
import pytest
from ecodata.io.structures import read_structure
from ecodata.models.adapters.wps_fields import export_pressure
from ecodata.core.exceptions import SchemaError
from test_wps_fields import read_records

BASE=Path(__file__).parents[1]/'examples/ai_round1/wrf/soil'
def load(real=False):
    a,c=('A_NOAA_topsoil_20240101.nc','config_NOAA.json') if real else ('A_synthetic_four_layers.nc','config_synthetic.json')
    return read_structure('source.nc',(BASE/a).read_bytes()),json.loads((BASE/c).read_text())

@pytest.mark.parametrize('real',[True,False])
def test_soil_independent_readback(real):
    ds,cfg=load(real);files,report=export_pressure(ds,cfg)
    dates=report['timestamps_UTC'];assert len(dates)==(2 if real else 1)
    for name,payload in files.items():
        if not name.endswith('.wps'):continue
        records=read_records(payload);assert len(records)==(2 if real else 8)
        index=0 if '_00.' in name or not real else 1
        axes=cfg['coordinates']
        for row in records:
            label=row['field'];mapping=cfg['fields'][label]
            array=ds[mapping['variable']].sortby(axes['latitude']).sortby(axes['longitude']).transpose(axes['time'],axes['latitude'],axes['longitude'])
            expected=array.values[index].astype('f8')
            if not real:expected=expected+273.15 if label.startswith('ST') else expected/100
            assert row['level']==200100
            assert row['units']==('K' if label.startswith('ST') else 'fraction')
            assert row['date']==dates[index].replace('T','_').replace('Z','')
            np.testing.assert_allclose(row['values'],expected,rtol=1e-7,atol=3e-5 if label.startswith('ST') else 1e-8)
        assert not report['time_processing'].startswith('interpol')

@pytest.mark.parametrize('error',['depth','metadata','basis','mass','missing','range','nan','underflow'])
def test_soil_invalid_input_rejected(error):
    ds,cfg=load();label='SM000010';m=cfg['fields'][label];var=m['variable']
    if error=='depth':m['source_depth_cm']=[0,20]
    if error=='metadata':ds[var].attrs['soil_depth_cm']=[0,20]
    if error=='basis':m.pop('soil_moisture_basis')
    if error=='mass':ds[var].attrs['units']='kg/kg';m['source_units']='kg/kg'
    if error=='missing':m.pop('source_depth_cm')
    if error=='range':ds[var][:]=101
    if error=='nan':ds[var][0,0,0]=np.nan
    if error=='underflow':ds[var]=ds[var].astype('float64');ds[var][:]=1e-50;ds[var].attrs['units']='%'
    with pytest.raises(SchemaError):export_pressure(ds,cfg)
