import io,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.io import netcdf_file
from ecodata.models.adapters.camx_initial import export_initial
from ecodata.core.exceptions import SchemaError


def case(alternate=False):
    nx,ny,nl=(3,2,2) if not alternate else (2,3,3)
    g=dict(projection='LATLON',ncols=nx,nrows=ny,xorig=110,yorig=30,xcell=.1,ycell=.1,coordinate_units='degree',xcent=0,ycent=0,p_alp=0,p_bet=0,itzon=0)
    if alternate:g.update(projection='LCC',xorig=0,yorig=0,xcell=12,ycell=12,coordinate_units='km',xcent=-100,ycent=40,p_alp=30,p_bet=60,itzon=6)
    rows=[];expected=np.arange(nl*ny*nx).reshape(nl,ny,nx)+10
    for k,j,i in np.ndindex(nl,ny,nx):rows.append(dict(date='2021-01-01T00:00:00Z',k=k,j=j,i=i,ozone=expected[k,j,i],sulfate=expected[k,j,i]/10))
    df=pd.DataFrame(rows)
    cfg=dict(mode='initial_concentrations',version='CAMx-7.40',mechanism='CB6r5_CF3_SIMPLE_NoEvap',grid=g,
        fields=dict(time='date',layer='k',row='j',col='i'),species=[dict(name='O3',column='ozone',phase='gas',unit='ppbV'),dict(name='PSO4',column='sulfate',phase='aerosol',unit='ug/m^3')],
        index_base=0,layer_count=nl,layer_definition='Synthetic layer labels only; real met height fields must be supplied separately',
        required_species=['O3','PSO4'],valid_until='2021-01-01T01:00:00Z',missing_values=[-9999],source=dict(kind='synthetic',description='Synthetic hand-verifiable grid, not measured concentrations',license='CC0 synthetic test values'))
    if alternate:
        df=df.rename(columns={'date':'instant','k':'level','j':'row','i':'column','ozone':'o3mix','sulfate':'mass'}).iloc[::-1].copy()
        df[['level','row','column']]+=1;df['o3mix']/=1000;df['mass']/=1000
        cfg['fields']=dict(time='instant',layer='level',row='row',col='column');cfg['index_base']=1
        cfg['species'][0].update(column='o3mix',unit='ppmV');cfg['species'][1].update(column='mass',unit='mg/m^3')
    return df,cfg,expected


@pytest.mark.parametrize('alternate',[False,True])
def test_independent_netcdf_initial_values(alternate):
    df,cfg,expected=case(alternate);files,report=export_initial(df,cfg)
    with netcdf_file(io.BytesIO(files['initial.CAMx7.40.nc']),mmap=False) as nc:
        assert nc.CAMx_NAME==b'INITIAL' and nc.NLAYS==cfg['layer_count']
        assert nc.variables['O3'].dimensions==('TSTEP','LAY','ROW','COL')
        assert nc.variables['O3'].units==b'ppm' and nc.variables['PSO4'].units==b'ug/m3'
        np.testing.assert_allclose(nc.variables['O3'][0],expected*.001,rtol=1e-7,atol=1e-10)
        np.testing.assert_allclose(nc.variables['PSO4'][0],expected/10,rtol=1e-7,atol=1e-10)
        np.testing.assert_array_equal(nc.variables['TFLAG'][0,0],[2020366,180000] if alternate else [2021001,0])
        np.testing.assert_array_equal(nc.variables['ETFLAG'][0,0],[2020366,190000] if alternate else [2021001,10000])
        assert nc.XCELL==(12000 if alternate else .1) and nc.ITZON==(6 if alternate else 0)
    assert report['model_run_verified'] is False


def test_repeated_initial_exports_are_byte_identical():
    df,cfg,_=case(True)
    first=export_initial(df,cfg)[0]['initial.CAMx7.40.nc']
    for _ in range(5):
        assert export_initial(df,cfg)[0]['initial.CAMx7.40.nc']==first


@pytest.mark.parametrize('error',['missing','gap','layer','units','time','validity','species','marker','negative','duplicate','required'])
def test_rejection(error):
    df,cfg,_=case()
    if error=='missing':df=df.drop(columns='ozone')
    if error=='gap':df=df.iloc[:-1]
    if error=='layer':cfg['layer_count']+=1
    if error=='units':cfg['species'][0]['unit']='ug/m^3'
    if error=='time':df.loc[0,'date']='2021-01-01T01:00:00Z'
    if error=='validity':cfg['valid_until']='2021-01-01T00:00:00Z'
    if error=='species':cfg['species'][0]['name']='NOx'
    if error=='marker':df.loc[0,'ozone']=-9999
    if error=='negative':df.loc[0,'sulfate']=-1
    if error=='duplicate':df=pd.concat([df,df.iloc[:1]])
    if error=='required':cfg['required_species'].append('NO2')
    with pytest.raises(SchemaError):export_initial(df,cfg)


def save_examples(path):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    for alternate in (False,True):
        df,cfg,_=case(alternate);name='synthetic_LCC' if alternate else 'synthetic_LATLON'
        df.to_csv(path/f'A_{name}.csv',index=False)
        (path/f'config_{name}.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf8')
