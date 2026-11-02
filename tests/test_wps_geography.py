import io,json,struct,zipfile
from pathlib import Path
import numpy as np
import pytest
import xarray as xr
from ecodata.models.adapters.wps_geography import export_geography
from ecodata.core.exceptions import SchemaError


def case(category=False,alternate=False):
    nx,ny=(3,2) if not alternate else (2,3)
    expected=np.arange(nx*ny).reshape(ny,nx)+100
    if category:expected=np.arange(nx*ny).reshape(ny,nx)%4+1
    ds=xr.Dataset({'value':(('lat','lon'),expected.astype(float),{'units':'1' if category else 'm'})},
        coords={'lat':30+np.arange(ny)*.1,'lon':110+np.arange(nx)*.1})
    ds.lat.attrs['units']='degrees_north';ds.lon.attrs['units']='degrees_east'
    cfg=dict(mode='geogrid_tiles',version='WPS-4.6-geogrid-tile',crs='EPSG:4326',field='LANDUSEF' if category else 'HGT_M',elevation_vertical_reference='MSL',
        coordinates=dict(latitude='lat',longitude='lon'),variable='value',wordsize=2,endian='little' if alternate else 'big',
        scale_factor=1.,max_absolute_quantization_error=0.,missing_values=[-9999],
        source=dict(kind='synthetic',description='Synthetic static grid, not real topography or land classification',license='CC0 synthetic values'))
    if category:cfg['landuse']=dict(category_min=1,category_max=4,mminlu='SYNTH_TEST',iswater=1,islake=2,isice=3,isurban=4)
    if alternate:
        if not category:ds['value']=ds.value/1000;ds.value.attrs['units']='km'
        ds=ds.isel(lat=slice(None,None,-1),lon=slice(None,None,-1)).rename({'lat':'north','lon':'east','value':'raster'}).transpose('east','north')
        cfg['coordinates']=dict(latitude='north',longitude='east');cfg['variable']='raster'
    return ds,cfg,expected


@pytest.mark.parametrize('category',[False,True])
@pytest.mark.parametrize('alternate',[False,True])
def test_independent_index_and_raw_binary(category,alternate):
    ds,cfg,expected=case(category,alternate);files,report=export_geography(ds,cfg)
    with zipfile.ZipFile(io.BytesIO(files[f'geogrid_{cfg["field"]}.zip'])) as z:
        index=dict(line.split('=',1) for line in z.read('index').decode().splitlines())
        assert index['projection']=='regular_ll' and index['row_order']=='bottom_top'
        nx,ny=int(index['tile_x']),int(index['tile_y']);assert (ny,nx)==expected.shape
        assert float(index['known_x'])==float(index['known_y'])==1
        assert float(index['known_lon'])==110 and float(index['known_lat'])==30
        assert abs(float(index['dx'])-.1)<1e-12 and abs(float(index['dy'])-.1)<1e-12
        raw=z.read(f'00001-{nx:05d}.00001-{ny:05d}')
        # Stdlib signed integer unpacking independent of NumPy byte writer.
        code='>' if index['endian']=='big' else '<'
        integers=struct.unpack(code+str(nx*ny)+'h',raw)
        actual=np.asarray(integers).reshape(ny,nx)*float(index['scale_factor'])
        np.testing.assert_array_equal(actual,expected)
        assert len(z.namelist())==2 and len(raw)==nx*ny*2
    assert report['model_run_verified'] is False


@pytest.mark.parametrize('error',['variable','missing','marker','units','time','grid','precision','overflow','land_codes','projection'])
def test_geo_tile_rejection(error):
    ds,cfg,_=case(category=error=='land_codes')
    if error=='variable':cfg['variable']='absent'
    if error=='missing':ds.value.values[0,0]=np.nan
    if error=='marker':ds.value.values[0,0]=-9999
    if error=='units':cfg['source_units']='km'
    if error=='time':ds['value']=ds.value.expand_dims(time=[0])
    if error=='grid':ds=ds.assign_coords(lon=[110,110.1,110.3]);ds.lon.attrs['units']='degree'
    if error=='precision':ds.value.values[0,0]=100.01
    if error=='overflow':ds.value.values[0,0]=40000
    if error=='land_codes':ds.value.values[0,0]=5
    if error=='projection':cfg['crs']='EPSG:3857'
    with pytest.raises(SchemaError):export_geography(ds,cfg)


def save_examples(path):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    for category in (False,True):
        for alternate in (False,True):
            ds,cfg,_=case(category,alternate);name=('landuse' if category else 'terrain')+('_mapped' if alternate else '_standard')
            ds.to_netcdf(path/f'A_synthetic_{name}.nc',engine='scipy')
            (path/f'config_synthetic_{name}.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf8')


@pytest.mark.parametrize('fault',[None,'sum','order','missing_layer','unit'])
def test_explicit_fraction_layers_preserve_mixed_categories(fault):
    ds,cfg,_=case(category=True)
    values=np.zeros((4,2,3),dtype=np.int16);values[0]=30;values[1]=70
    ds['value']=(('category','lat','lon'),values,{'units':'%'})
    ds=ds.assign_coords(category=[1,2,3,4]);cfg.update(representation='fraction_layers',category_coordinate='category')
    if fault=='sum':ds.value.values[0,0,0]=29
    if fault=='order':ds=ds.assign_coords(category=[2,1,3,4])
    if fault=='missing_layer':ds=ds.isel(category=slice(0,3))
    if fault=='unit':ds.value.attrs['units']='1'
    if fault:
        with pytest.raises(SchemaError):export_geography(ds,cfg)
        return
    files,report=export_geography(ds,cfg)
    with zipfile.ZipFile(io.BytesIO(files['geogrid_LANDUSEF.zip'])) as z:
        index=dict(line.split('=',1) for line in z.read('index').decode().splitlines())
        assert index['type']=='continuous' and index['tile_z']=='4'
        raw=z.read(report['native_tile'])
        actual=np.array(struct.unpack('>'+str(values.size)+'h',raw)).reshape(values.shape)
        np.testing.assert_array_equal(actual,values)
