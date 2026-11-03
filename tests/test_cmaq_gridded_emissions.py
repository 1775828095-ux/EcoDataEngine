import copy,io,json,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.io import netcdf_file
from ecodata.core.exceptions import SchemaError
from ecodata.io.structures import read_structure
from ecodata.models.adapters.cmaq_gridded_emissions import export_gridded_emissions

BASE=Path(__file__).parents[1]/'examples/ai_round1/cmaq/emissions'


@pytest.mark.parametrize('resolution',['s','ms','us','ns'])
def test_hourly_check_independent_of_datetime_storage_resolution(resolution):
    from ecodata.models.adapters.cmaq_gridded_emissions import _time
    import xarray as xr
    data=xr.Dataset(coords={'t':np.array(['2024-02-29T23','2024-03-01T00'],dtype=f'datetime64[{resolution}]')})
    _,flags,_,_,_=_time({'time_mode':'hourly','source_timezone':'UTC'},data,{'time':'t'})
    np.testing.assert_array_equal(flags,[[2024060,230000],[2024061,0]])

def load(real=False):
    a,c=('A_EDGAR_CO_2014_Paris.nc','config_EDGAR.json') if real else ('A_synthetic_LCC_two_hours.nc','config_synthetic.json')
    return read_structure('source.nc',(BASE/a).read_bytes()),json.loads((BASE/c).read_text())

def independent(payload):
    # SciPy has a separate pure-Python NetCDF3 reader, not the netCDF4 writer.
    return netcdf_file(io.BytesIO(payload),'r',mmap=False)


def test_real_annual_flux_area_molar_conversion():
    ds,cfg=load(True);files,report=export_gridded_emissions(ds,cfg)
    # Independent area calculation with GeographicLib (different implementation to pyproj).
    from geographiclib.geodesic import Geodesic
    expected=np.empty((3,3))
    for j in range(3):
        for i in range(3):
            south=cfg['grid']['YORIG']+j*.1;west=cfg['grid']['XORIG']+i*.1
            polygon=Geodesic.WGS84.Polygon()
            for lat,lon in [(south,west),(south,west+.1),(south+.1,west+.1),(south+.1,west)]:polygon.AddPoint(lat,lon)
            _,_,area=polygon.Compute();expected[j,i]=float(ds.fluxes.values[j,i])*1000*abs(area)/28.01
    with independent(files['emissions.CMAQ55.nc']) as n:
        assert n.FTYPE==1 and n.NVARS==1 and n.NLAYS==1
        assert n.SDATE==n.STIME==n.TSTEP==0 and not n.variables['TFLAG'][:].any()
        assert n.variables['CO'].dimensions==('TSTEP','LAY','ROW','COL')
        assert n.variables['CO'].units.strip()==b'moles/s'
        np.testing.assert_allclose(n.variables['CO'][0,0],expected,rtol=1e-7,atol=1e-10)
        assert n.XORIG==cfg['grid']['XORIG'] and n.YORIG==cfg['grid']['YORIG']
    assert report['time_interpolation']=='none' and report['source']['kind']=='real_inventory'
    assert 'file.cio' not in files


def test_hourly_projected_multi_species_order_and_units():
    ds,cfg=load();files,report=export_gridded_emissions(ds,cfg)
    with independent(files['emissions.CMAQ55.nc']) as n:
        assert n.NVARS==2 and n.GDTYP==2 and n.TSTEP==10000 and n.SDATE==2020366 and n.STIME==230000
        assert n.dimensions['TSTEP'] is None
        assert getattr(n,'VAR-LIST')==b'NO2             ASO4J           '
        np.testing.assert_array_equal(n.variables['TFLAG'][:],[[[2020366,230000],[2020366,230000]],[[2021001,0],[2021001,0]]])
        np.testing.assert_allclose(n.variables['NO2'][:,0],ds.nitrogen_dioxide.values[:,::-1,:]*1000/(3600*46),rtol=1e-7,atol=1e-12)
        np.testing.assert_allclose(n.variables['ASO4J'][:,0],ds.sulfate_J.values[:,::-1,:]/3600,rtol=1e-7,atol=1e-12)
        assert n.variables['ASO4J'].units.strip()==b'g/s'
        assert np.isfinite(n.variables['NO2'][:]).all()
    assert report['surface_layer_only']

@pytest.mark.parametrize('problem',['missing','nan','negative','marker','units','mw','mw_source','time_gap','time_duplicate','time_reverse','timezone','grid','origin','projection','crs','mode','surface','species','vertical','name','x_units','no_mapping','source','duplicate_source','extra_dim','static_assumption'])
def test_reject_incomplete_or_inconsistent(problem):
    ds,cfg=load()
    if problem=='missing':ds=ds.drop_vars('nitrogen_dioxide')
    elif problem=='nan':ds.nitrogen_dioxide.values[0,0,0]=np.nan
    elif problem=='negative':ds.nitrogen_dioxide.values[0,0,0]=-1
    elif problem=='marker':cfg['missing_values']=[1]
    elif problem=='units':cfg['fields']['NO2']['source_units']='kg/s'
    elif problem=='mw':cfg['fields']['NO2'].pop('molecular_weight_g_mol')
    elif problem=='mw_source':cfg['fields']['NO2'].pop('molecular_weight_source')
    elif problem=='time_gap':ds=ds.assign_coords(when=np.array(['2020-12-31T23','2021-01-01T01'],dtype='datetime64[h]'))
    elif problem=='time_duplicate':ds=ds.assign_coords(when=np.array(['2020-12-31T23','2020-12-31T23'],dtype='datetime64[h]'))
    elif problem=='time_reverse':ds=ds.isel(when=[1,0])
    elif problem=='timezone':cfg['source_timezone']='Asia/Shanghai'
    elif problem=='grid':cfg['grid']['NROWS']=3
    elif problem=='origin':cfg['grid']['YORIG']+=100
    elif problem=='projection':cfg['grid']['GDTYP']=6
    elif problem=='crs':cfg['source_crs']='EPSG:4326'
    elif problem=='mode':cfg['time_mode']='monthly'
    elif problem=='surface':cfg['layer_mode']='automatic'
    elif problem=='species':cfg['fields']['PM25']=cfg['fields'].pop('ASO4J')
    elif problem=='vertical':cfg['vertical']['VGLVLS']=[1,1]
    elif problem=='name':cfg['grid']['GDNAM']='unsafe name'
    elif problem=='x_units':ds.easting.attrs['units']='km'
    elif problem=='no_mapping':cfg['fields']={}
    elif problem=='source':cfg['source']={}
    elif problem=='duplicate_source':cfg['fields']['ASO4J']={**cfg['fields']['NO2'],'type':'aerosol_mass'}
    elif problem=='extra_dim':ds['nitrogen_dioxide']=ds.nitrogen_dioxide.expand_dims({'z':[1]})
    elif problem=='static_assumption':cfg['time_mode']='static_inventory';cfg['static_mean_description']=''
    with pytest.raises(SchemaError):export_gridded_emissions(ds,cfg)


def test_pseudonetcdf_independent_reader(tmp_path):
    # Optional third-party reader; production writer does not depend on it.
    pnc=pytest.importorskip('PseudoNetCDF')
    ds,cfg=load();files,_=export_gridded_emissions(ds,cfg)
    # netCDF4's Windows filename handling needs ASCII temp directory.
    with tempfile.NamedTemporaryFile(suffix='.nc',delete=False) as f:
        path=f.name;f.write(files['emissions.CMAQ55.nc'])
    try:
        from PseudoNetCDF.cmaqfiles import ioapi
        n=ioapi(path);assert n.NVARS==2 and n.NROWS==2 and n.NCOLS==3
        assert list(n.getVarlist(update=False,retval='list'))==['NO2','ASO4J']
        np.testing.assert_array_equal(n.variables['TFLAG'][:],[[[2020366,230000],[2020366,230000]],[[2021001,0],[2021001,0]]]);n.close()
    finally:Path(path).unlink()


@pytest.mark.parametrize('real',[True,False])
def test_griddesc_third_party_parser_matches_netcdf(real):
    from PseudoNetCDF.cmaqfiles import griddesc
    ds,cfg=load(real);files,_=export_gridded_emissions(ds,cfg)
    n=griddesc(io.StringIO(files['GRIDDESC'].decode()),GDNAM=cfg['grid']['GDNAM'],withcf=False)
    with independent(files['emissions.CMAQ55.nc']) as nc:
        for key in ('GDTYP','P_ALP','P_BET','P_GAM','XCENT','YCENT','XORIG','YORIG','XCELL','YCELL','NCOLS','NROWS','NTHIK'):
            assert getattr(n,key)==getattr(nc,key)


@pytest.mark.parametrize('kind',['bool','tiny'])
def test_invalid_dtype_and_underflow_rejected(kind):
    ds,cfg=load()
    if kind=='bool':
        attrs=ds.nitrogen_dioxide.attrs.copy();ds['nitrogen_dioxide']=ds.nitrogen_dioxide.astype(bool);ds.nitrogen_dioxide.attrs=attrs
    else:ds.nitrogen_dioxide[:]=1e-60
    with pytest.raises(SchemaError):export_gridded_emissions(ds,cfg)
