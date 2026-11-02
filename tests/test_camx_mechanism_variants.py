"""Synthetic interface invariants, not independent real inventory evidence."""
import copy
import io
import numpy as np
import pytest
from scipy.io import netcdf_file
from ecodata.round1 import convert
from ecodata.core.exceptions import SchemaError
from test_camx_emissions import case as emissions_case
from test_camx_initial import case as initial_case
from test_camx_boundary import case as boundary_case


def fixture(kind, alternate):
    if kind == 'emissions':
        frame, cfg = emissions_case()
        if alternate:
            cfg['grid'].update(projection='LCC', coordinate_units='km',
                               xorig=0, yorig=0, xcell=12, ycell=12,
                               xcent=-100, ycent=40, p_alp=30, p_bet=60)
        return frame, cfg, 'emissions.CAMx7.40.nc'
    if kind == 'initial':
        frame, cfg, _ = initial_case(alternate)
        return frame, cfg, 'initial.CAMx7.40.nc'
    frame, cfg = boundary_case('LCC' if alternate else 'LATLON')
    return frame, cfg, 'boundary.CAMx7.40.nc'


@pytest.mark.parametrize('kind', ['emissions', 'initial', 'boundary'])
@pytest.mark.parametrize('alternate', [False, True])
def test_both_mechanisms_preserve_every_variable_and_metadata(kind, alternate):
    frame, cfg, name = fixture(kind, alternate)
    outputs = []
    for mechanism in ['CB6r5_CF3_SIMPLE_NoEvap', 'CB6r5_CF3_COMPLX']:
        cfg = copy.deepcopy(cfg)
        cfg['mechanism'] = mechanism
        files, report = convert('camx', frame.to_csv(index=False).encode(), cfg)
        assert report['mechanism'] == mechanism
        outputs.append(files[name])
    # Mechanism selection never silently respeciates or alters supplied values.
    with netcdf_file(io.BytesIO(outputs[0]), mmap=False) as a, netcdf_file(io.BytesIO(outputs[1]), mmap=False) as b:
        assert a.dimensions == b.dimensions
        assert a._attributes == b._attributes
        assert a.variables.keys() == b.variables.keys()
        for key in a.variables:
            assert a.variables[key]._attributes == b.variables[key]._attributes
            np.testing.assert_array_equal(a.variables[key][:], b.variables[key][:])


@pytest.mark.parametrize('kind', ['emissions', 'initial', 'boundary'])
@pytest.mark.parametrize('mechanism', ['CB6r5_CF3_COMPLX_NoEvap', 'CB7r1', None, []])
def test_unverified_mechanisms_are_rejected(kind, mechanism):
    frame, cfg, _ = fixture(kind, False)
    cfg['mechanism'] = mechanism
    with pytest.raises(SchemaError, match='机制|CAMx7.40'):
        convert('camx', frame.to_csv(index=False).encode(), cfg)
