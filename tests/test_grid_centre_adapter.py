import numpy as np
import pytest
import xarray as xr
from ecodata.algorithms.platform_search import platform_registry,make_context


@pytest.mark.parametrize('identifier',['fusion.idw','station.metpy_linear','station.pykrige_ordinary'])
def test_grid_centre_adapter_uses_only_visible_source(identifier):
    ds=xr.Dataset({'temperature':(('lat','lon'),280+np.arange(36).reshape(6,6)/10)},
        coords={'lat':np.arange(30.,36.),'lon':np.arange(110.,116.)},attrs={'crs':'EPSG:4326'})
    ds.temperature.attrs.update(units='K',standard_name='air_temperature',spatial_semantics='cell_mean')
    req=dict(task='spatial',variable='temperature',unit='K',semantic_type='continuous',nonnegative=True,
        target_lat=np.linspace(29.9,35.1,12).tolist(),target_lon=np.linspace(109.9,115.1,12).tolist(),parameters={})
    ctx,_,_=make_context([('primary',ds)],req)
    adapter=platform_registry().get('raster_task.'+identifier)
    assert adapter.compatibility(ctx).eligible
    observed=adapter.station_context(ctx).observations
    np.testing.assert_array_equal(observed.value,ds.temperature.values.ravel())
    assert observed.split.eq('train').all()
    product=adapter.run(ctx).data.dataset
    assert product.temperature.shape==(12,12) and np.isfinite(product.temperature).all()
    assert product.temperature.attrs['units']=='K'
