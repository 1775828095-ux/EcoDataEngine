"""Small synthetic contract tests, not real scientific validation evidence."""
import io,zipfile
import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from ecodata.interface.calmet_raw_geography import from_rasters

def sample(missing=False,unmapped=False):
    files={}
    def raster(name,a,x):
        with MemoryFile() as m:
            with m.open(driver='GTiff',height=a.shape[0],width=a.shape[1],count=1,dtype=a.dtype,crs='EPSG:32631',transform=from_origin(x,45,10,10)) as d:d.write(a,1)
            files[name]=m.read()
    raster('dem1.tif',np.zeros((6,3),dtype='float32'),499995)
    if not missing:raster('dem2.tif',np.zeros((6,3),dtype='float32'),500025)
    raster('land.tif',np.full((6,6),2 if unmapped else 1,dtype='uint8'),499995)
    files['makegeo.inp']=b'! X = 11, .5, .18, 1, .20, 0, 1, 10 !'
    cfg=dict(schema='ecodata.calmet.raw-geography.v1',dem_files=['dem1.tif'] if missing else ['dem1.tif','dem2.tif'],landcover_files=['land.tif'],parameter_table_file='makegeo.inp',dem_resampling='average',landcover_resampling='mode',elevation_units='m',vertical_datum='EGM2008',class_crosswalk={'1':11},crosswalk_description='Synthetic explicit class mapping for contract test',water_category_range=[50,55],source={'description':'Synthetic software fixture, not measured terrain'})
    control='\n'.join('! '+k+' = '+str(v)+' !' for k,v in dict(PMAP='UTM',DATUM='WGS-84',IUTMZN=31,UTMHEM='N',NX=4,NY=4,DGRIDKM=.01,XORIGKM=500,YORIGKM=0).items()).encode('ascii')
    b=io.BytesIO()
    with zipfile.ZipFile(b,'w') as z:
        for n,a in files.items():z.writestr(n,a)
    return b.getvalue(),cfg,control

def test_source_half_pixel_alignment_and_valid_zero_elevation_preserved():
    files,report=from_rasters(*sample())
    assert report['shape_y_x']==[4,4] and report['input_classes']==[1]
    assert not report['source_bytes_modified'] and not report['missing_values_filled']
    assert report['vertical_datum']=='EGM2008' and b'GEO.DAT' in files['GEO.DAT']
    import xarray as xr
    with xr.open_dataset(io.BytesIO(files['geography.nc']),engine='scipy') as d:
        np.testing.assert_array_equal(d.elevation,np.zeros((4,4)))

def test_missing_tile_rejected_without_fill():
    with pytest.raises(ValueError,match='覆盖缺口|未完全覆盖'):from_rasters(*sample(missing=True))

def test_raw_upload_base_does_not_require_prepared_geography():
    from pathlib import Path
    from ecodata.interface.calpuff_project import unpack
    reference=Path(__file__).resolve().parents[3]/'双任务推进_20261030/Apex_WRF_grid_parameter_build.zip'
    base=io.BytesIO()
    with zipfile.ZipFile(reference) as src,zipfile.ZipFile(base,'w') as dst:
        for n in src.namelist():
            if n not in {'geography.nc','geography_config.json'}:dst.writestr(n,src.read(n))
    with pytest.raises(ValueError,match='工程ZIP'):unpack(base.getvalue(),'rebuild')
    data,_=unpack(base.getvalue(),'rebuild',raw_geography_inputs=True)
    assert 'WRF.nc' in data and 'geography.nc' not in data

def test_unknown_land_category_never_uses_fallback_parameters():
    with pytest.raises(ValueError,match='未明确匹配'):from_rasters(*sample(unmapped=True))

def test_ellipsoid_height_not_silently_treated_as_orthometric():
    data,cfg,control=sample();cfg['vertical_datum']='ellipsoid'
    with pytest.raises(ValueError,match='正高'):from_rasters(data,cfg,control)
