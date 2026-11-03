import json
import numpy as np
import pytest
import xarray as xr
from ecodata.algorithms import reliable_calibration as module
from ecodata.algorithms import reliable_selection as selection


def write(path,value):
    path.write_text(json.dumps(value),encoding='utf-8')


def run(tmp_path,group,role='development'):
    folder=tmp_path/group;folder.mkdir()
    truth=xr.DataArray(np.zeros((2,2,2)),dims=('time','lat','lon'),
        coords=dict(time=np.array(['2020-01-01','2020-01-02'] if group=='a' else ['2021-01-01','2021-01-02'],dtype='datetime64[ns]'),lat=[0,1],lon=[0,1]),
        attrs=dict(units='1',split=role),name='ndvi')
    reference=folder/'labels.nc';truth.to_dataset().to_netcdf(reference,engine='h5netcdf')
    policy=selection.make_policy({},[0.,2.],calibration={
        'scope':{},'delta':0.,'evidence_role':'development','evidence':'Synthetic test evidence'})
    rows=[]
    for key,nodes,internal,external in [('simple',1,.101,.1),('complex',2,.1,.2),('third',3,.2,.5)]:
        product=folder/(key+'.nc');(truth+external).assign_attrs(units='1').to_dataset().to_netcdf(product,engine='h5netcdf')
        rows.append(dict(bank_id=key,key=key,status='PASS',selection_eligible=True,
            metrics=dict(objective=internal),algorithm_nodes=nodes,product=str(product)))
    write(folder/'request.json',dict(request=dict(variable='ndvi',unit='1',evidence_role=role,group=group)))
    write(folder/'summary.json',dict(reliable_selection=policy,bank_rows=rows))
    write(folder/'graphs.json',dict(s0=dict(operation='source',parents=[],parameters=dict(source='source0')),
        simple=dict(operation='temporal.raster_linear',parents=['s0'],parameters={}),
        complex=dict(operation='spatial.denoise_tv',parents=['simple'],parameters={})))
    return dict(folder=str(folder),reference=str(reference),group=group)


def test_calibration_reads_real_products_and_publishes_frozen_reference(tmp_path):
    result=module.calibrate([run(tmp_path,'a'),run(tmp_path,'b')],'fixture',tmp_path/'published.json')
    assert result['policy']['delta']>0
    assert all(h['regret_improvement']>0 for h in result['report']['held_group_audit'])
    graph=result['policy']['common_reference_graph']
    assert graph['bank_id']=='simple' and graph['key']=='simple'
    assert set(graph['records'])=={'s0','simple'}
    assert json.loads((tmp_path/'published.json').read_text())==result['policy']
    with pytest.raises(ValueError,match='overwrite'):
        module.calibrate([], 'different',tmp_path/'published.json')


def test_calibration_rejects_test_labels_and_no_group_support_keeps_zero(tmp_path):
    item=run(tmp_path,'a',role='test')
    with pytest.raises(ValueError,match='development'):
        module.calibrate([item],'bad',tmp_path/'bad.json')
    result=module.calibrate([run(tmp_path,'b')],'one',tmp_path/'one.json')
    assert result['policy']['delta']==0
    assert result['report']['status']=='INSUFFICIENT_GROUPS_ZERO_TOLERANCE'
