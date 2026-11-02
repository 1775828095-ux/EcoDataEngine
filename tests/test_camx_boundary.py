import io,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.io import netcdf_file
from ecodata.round1 import convert
from ecodata.core.exceptions import SchemaError
BASE=Path(__file__).parents[1]/'examples/ai_round1/camx/boundary'
def case(label='LATLON'):
    return pd.read_csv(BASE/f'A_synthetic_{label}.csv'),json.loads((BASE/f'config_{label}.json').read_text(encoding='utf8'))
def export(frame,cfg):return convert('camx',frame.to_csv(index=False).encode(),cfg)

def test_six_hour_source_intervals_preserve_values_and_reject_gaps():
    frame,cfg=case();col=cfg['fields']['time']
    times=frame[col].unique();start=pd.Timestamp(times[0])
    frame[col]=frame[col].replace({times[1]:(start+pd.Timedelta(hours=6)).isoformat()})
    cfg.update(interval_seconds=21600,valid_until=(start+pd.Timedelta(hours=12)).isoformat())
    files,report=export(frame,cfg)
    with netcdf_file(io.BytesIO(files['boundary.CAMx7.40.nc']),'r',mmap=False) as nc:
        assert nc.TSTEP==60000 and nc.NSTEPS==2
        assert nc.variables['O3'].shape[0]==2
        np.testing.assert_array_equal(nc.variables['ETFLAG'].data[0],nc.variables['TFLAG'].data[1])
    assert report['hours']==12 and report['intervals']==2
    cfg['valid_until']=(start+pd.Timedelta(hours=18)).isoformat()
    with pytest.raises(SchemaError,match='缺时刻'):export(frame,cfg)
    cfg['valid_until']=(start+pd.Timedelta(hours=13)).isoformat()
    with pytest.raises(SchemaError,match='整除'):export(frame,cfg)

@pytest.mark.parametrize('label,nx,ny,nl,dates,hours',[('LATLON',4,3,2,[2024060,2024061],[230000,0]),('LCC',3,4,3,[2024060,2024060],[170000,180000])])
def test_independent_full_array_and_edge_value_read(label,nx,ny,nl,dates,hours):
    frame,cfg=case(label);files,r=export(frame,cfg)
    with netcdf_file(io.BytesIO(files['boundary.CAMx7.40.nc']),'r',mmap=False) as nc:
        assert nc.CAMx_NAME==b'BOUNDARY' and nc.NSTEPS==2 and nc.NCOLS==nx and nc.NROWS==ny and nc.NLAYS==nl
        assert nc.variables['O3'].dimensions==('TSTEP','LAY','ROW','COL')
        assert nc.variables['O3'].units==b'ppm' and nc.variables['PSO4'].units==b'ug/m3'
        values=np.empty((2,nl,ny,nx))
        for t in range(2):
            for k in range(nl):values[t,k]=10+100*t+10*k+np.arange(ny*nx).reshape(ny,nx)
        np.testing.assert_allclose(nc.variables['O3'].data,values*.001,rtol=1e-7,atol=1e-9)
        np.testing.assert_allclose(nc.variables['PSO4'].data,values*.1,rtol=1e-7,atol=1e-6)
        np.testing.assert_array_equal(nc.variables['TFLAG'].data[:,0,:],np.array(list(zip(dates,hours))))
        np.testing.assert_array_equal(nc.variables['ETFLAG'].data[0],nc.variables['TFLAG'].data[1])
        # All four native lateral faces (without making corner defaults).
        assert np.isfinite(nc.variables['O3'].data[:,:,[0,-1],:]).all()
        assert np.isfinite(nc.variables['O3'].data[:,:,:,[0,-1]]).all()
    assert r['model_run_verified'] is False and r['input_layout']=='full_grid_3d'

@pytest.mark.parametrize('error,match',[
 ('time','缺时刻'),('grid','缺格/缺层'),('layer','缺格/缺层'),('column','缺列'),('unit','浓度需'),
 ('marker','缺测标记'),('negative','负浓度'),('nan','缺测'),('duplicate','重复'),('reference','required_species'),
 ('timezone','UTC偏移'),('interval','3600'),('perimeter','不将外圈'),('small','至少3')])
def test_errors_require_missing_information(error,match):
    frame,cfg=case();s=cfg['species'][0]['column'];fields=cfg['fields'];times=frame[fields['time']].unique()
    if error=='time':frame=frame[frame[fields['time']]==times[0]]
    elif error=='grid':frame=frame.iloc[1:]
    elif error=='layer':frame=frame[frame[fields['layer']]!=0]
    elif error=='column':frame=frame.drop(columns=s)
    elif error=='unit':cfg['species'][0]['unit']='mg/m^3'
    elif error=='marker':frame.loc[0,s]=-9999
    elif error=='negative':frame.loc[0,s]=-1
    elif error=='nan':frame.loc[0,s]=np.nan
    elif error=='duplicate':frame=pd.concat([frame,frame.iloc[:1]])
    elif error=='reference':cfg['required_species'].append('NO2')
    elif error=='timezone':frame[fields['time']]=frame[fields['time']].str.replace('+00:00','',regex=False)
    elif error=='interval':cfg['interval_seconds']=86400
    elif error=='perimeter':cfg['input_layout']='perimeter'
    elif error=='small':cfg['grid']['ncols']=2
    with pytest.raises(SchemaError,match=match):export(frame,cfg)
