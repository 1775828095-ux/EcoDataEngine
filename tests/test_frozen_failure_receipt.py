import json
import numpy as np
import pytest
from ecodata.algorithms import platform_search as api
from test_current_input import setup


def test_fixed_numerical_failure_retains_cost_and_negative_trace(monkeypatch,tmp_path):
    fine,_,inputs,q=setup(monkeypatch,tmp_path,t=6);fine.ndvi.values[2]=np.nan
    inputs[0]=api.SearchInput('fine.nc','primary',fine.to_netcdf())
    graph=dict(records=dict(s0=dict(operation='source',parents=[],parameters=dict(source='source0')),
        a=dict(operation='reconstruction.grid_gap_linear',parents=['s0'],parameters={})),key='a')
    folder=tmp_path/'failed'
    with pytest.raises(Exception,match='four real visible'):api.execute_frozen_graph(inputs,q,graph,folder)
    receipt=json.loads((folder/'deployment.json').read_text(encoding='utf-8'))
    assert receipt['status']=='FAILED' and receipt['product'] is None
    assert receipt['cost']['used']==receipt['cost']['failed_calls']==1
    assert receipt['trace'][-1]['status']=='FAIL'
    assert not receipt['reference_used'] and not receipt['selection_performed']
    assert json.loads((folder/'total_cost.json').read_text(encoding='utf-8'))['used']==1


def test_fixed_post_execution_gate_failure_keeps_successful_call_cost(monkeypatch,tmp_path):
    _,_,inputs,q=setup(monkeypatch,tmp_path,t=6)
    from ecodata.algorithms import current_input
    def gate(*args):raise ValueError('Injected output quality gate rejection')
    monkeypatch.setattr(current_input,'validate_output',gate)
    graph=dict(records=dict(s0=dict(operation='source',parents=[],parameters=dict(source='source0')),
        a=dict(operation='temporal.raster_linear',parents=['s0'],parameters={})),key='a')
    folder=tmp_path/'gate'
    with pytest.raises(ValueError,match='quality gate'):api.execute_frozen_graph(inputs,q,graph,folder)
    receipt=json.loads((folder/'deployment.json').read_text(encoding='utf-8'))
    assert receipt['status']=='FAILED' and receipt['product'] is None
    assert receipt['cost']['used']==1 and receipt['cost']['failed_calls']==0
