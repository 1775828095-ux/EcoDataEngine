"""Small synthetic invariants; this is not a real inventory validation."""
import copy
from io import BytesIO
import numpy as np
import pandas as pd
import pytest
from scipy.io import netcdf_file
from pathlib import Path
import ecodata.models.adapters
stage_adapters=Path(__file__).resolve().parents[1]/'src/ecodata/models/adapters'
if stage_adapters.exists():ecodata.models.adapters.__path__.insert(0,str(stage_adapters))
from ecodata.models.adapters.camx_emissions import export_gridded_emissions
from ecodata.core.exceptions import SchemaError


def case():
    config={'mode':'gridded_emissions','version':'CAMx-7.40','mechanism':'CB6r5_CF3_SIMPLE_NoEvap',
            'fields':{'time':'timestamp','row':'iy','col':'ix'},'row_col_base':1,
            'species':[{'name':'NO','column':'nitric_oxide','phase':'gas','unit':'mol/s'},
                       {'name':'PEC','column':'carbon','phase':'aerosol','unit':'kg/hour'}],
            'grid':{'projection':'LATLON','ncols':3,'nrows':2,'xorig':110.,'yorig':30.,'xcell':.1,'ycell':.2,
                    'coordinate_units':'degree','xcent':0,'ycent':0,'p_alp':0,'p_bet':0,'itzon':-8},
            'missing_values':[-9999],'time_basis':'interval_start_hourly',
            'source':{'kind':'synthetic_test','url':'urn:ecodata:synthetic:camx-emissions','license':'CC0'}}
    rows=[]
    for ti,time in enumerate(['2021-12-31T15:00:00Z','2021-12-31T16:00:00Z']):
        for row in [1,2]:
            for col in [1,2,3]:rows.append([time,row,col,ti*6+(row-1)*3+col,.25*col])
    return pd.DataFrame(rows,columns=['timestamp','iy','ix','nitric_oxide','carbon']).sample(frac=1,random_state=5),config


def test_independent_read_structure_numbers_and_year_boundary():
    frame,cfg=case();files,report=export_gridded_emissions(frame,cfg)
    with netcdf_file(BytesIO(files['emissions.CAMx7.40.nc']),mmap=False) as f:
        assert f.CAMx_NAME==b'EMISSIONS' and f.CPROJ==0 and f.ITZON==-8
        assert f.variables['NO'].dimensions==('TSTEP','LAY','ROW','COL')
        np.testing.assert_array_equal(f.variables['TFLAG'][:,0],[[2021365,230000],[2022001,0]])
        np.testing.assert_array_equal(f.variables['ETFLAG'][:,0],[[2022001,0],[2022001,10000]])
        np.testing.assert_array_equal(f.variables['NO'][:].reshape(-1),np.arange(1,13)*3600)
        np.testing.assert_array_equal(f.variables['PEC'][:].reshape(2,6),[[250,500,750,250,500,750]]*2)
        np.testing.assert_allclose(f.variables['longitude'][0],[110.05,110.15,110.25],atol=1e-12)
        np.testing.assert_allclose(f.variables['latitude'][:,0],[30.1,30.3],atol=1e-12)
        assert f.variables['NO'].units==b'mol hr-1'
        assert f.variables['PEC'].units==b'g hr-1'
        assert f.dimensions['LAY']==1 and f.NSTEPS==2
    assert report['camx_run_verified'] is False and report['complete_model_input'] is False


def test_lcc_metre_globals_kilometre_coordinates():
    frame,cfg=case();cfg['grid'].update(projection='LCC',coordinate_units='km',xorig=-3.,yorig=-2.,xcell=2.,ycell=2.,xcent=-97.,ycent=40.,p_alp=30.,p_bet=60.)
    files,_=export_gridded_emissions(frame,cfg)
    with netcdf_file(BytesIO(files['emissions.CAMx7.40.nc']),mmap=False) as f:
        assert f.CPROJ==2 and f.GDTYP==2 and f.XORIG==-3000. and f.XCELL==2000.
        np.testing.assert_array_equal(f.variables['X'][:],[-2,0,2])
        assert f.variables['X'].units==b'km'
        assert abs(f.variables['longitude'][0,1]+97)<1e-10
        assert abs(f.variables['latitude'][0,1]-40)<.02


@pytest.mark.parametrize('fault', ['missing_cell','duplicate_cell','gap_hour','naive_time','half_hour','missing_marker','negative','nan','extra_column','gas_mass','wrong_phase','unknown_species','missing_unit','wrong_version','bad_grid','bad_base','unknown_projection','missing_config'])
def test_reject_bad_inputs(fault):
    f,c=case()
    if fault=='missing_cell':f=f.iloc[1:]
    elif fault=='duplicate_cell':f=pd.concat([f,f.iloc[:1]])
    elif fault=='gap_hour':f.loc[f.timestamp.str.contains('16:00'),'timestamp']='2021-12-31T18:00:00Z'
    elif fault=='naive_time':f['timestamp']=f.timestamp.str.removesuffix('Z')
    elif fault=='half_hour':f['timestamp']=f.timestamp.str.replace(':00:00',':30:00')
    elif fault=='missing_marker':f.iloc[0,f.columns.get_loc('carbon')]=-9999
    elif fault=='negative':f.iloc[0,f.columns.get_loc('carbon')]=-.1
    elif fault=='nan':f.iloc[0,f.columns.get_loc('carbon')]=np.nan
    elif fault=='extra_column':f['discard_me']=1
    elif fault=='gas_mass':c['species'][0]['unit']='kg/hour'
    elif fault=='wrong_phase':c['species'][0]['phase']='aerosol'
    elif fault=='unknown_species':c['species'][0]['name']='NOX'
    elif fault=='missing_unit':del c['species'][0]['unit']
    elif fault=='wrong_version':c['version']='CAMx-7.00'
    elif fault=='bad_grid':c['grid']['xcell']=-1
    elif fault=='bad_base':c['row_col_base']=True
    elif fault=='unknown_projection':c['grid']['projection']='UTM'
    elif fault=='missing_config':del c['time_basis']
    with pytest.raises(SchemaError):export_gridded_emissions(f,c)


def test_custom_columns_one_cell_snapshot_explicit_zero():
    f,c=case();f=f.iloc[:1].copy();f['iy']=1;f['ix']=1;f['nitric_oxide']=0
    c['grid'].update(ncols=1,nrows=1);f=f.rename(columns={'timestamp':'clock','iy':'rowid','ix':'colid','carbon':'soot','nitric_oxide':'NO_rate'})
    c['fields']={'time':'clock','row':'rowid','col':'colid'};c['species'][0]['column']='NO_rate';c['species'][1]['column']='soot'
    files,r=export_gridded_emissions(f,c);assert r['shape']==[1,1,1,1]
    with netcdf_file(BytesIO(files['emissions.CAMx7.40.nc']),mmap=False) as out:assert float(out.variables['NO'][:].item())==0


def test_float_rounding_with_engineering_limit():
    f,c=case();f['nitric_oxide']=1.23456789;c['species'][0]['unit']='mol/hour'
    files,r=export_gridded_emissions(f,c)
    with netcdf_file(BytesIO(files['emissions.CAMx7.40.nc']),mmap=False) as out:
        np.testing.assert_allclose(out.variables['NO'][:],1.23456789,rtol=2e-7,atol=1e-12)
    assert 0<r['float32_max_absolute_rounding_error']['NO']<1e-7
