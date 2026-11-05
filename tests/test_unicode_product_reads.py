"""Unicode product IO regressions; these fixtures do not establish accuracy."""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecodata.io.netcdf import load_dataset


@pytest.mark.parametrize('engine,format', [('h5netcdf', None), ('scipy', 'NETCDF3_CLASSIC'),
                                          ('scipy', 'NETCDF3_64BIT')])
def test_product_formats_preserve_decoding_and_close_handles(tmp_path, engine, format, monkeypatch):
    path = tmp_path / '中文目录 with spaces' / '气温.product'
    path.parent.mkdir()
    data = xr.Dataset({'t': ('time', [280., np.nan, 290.], {'units': 'K'})},
                      coords={'time': pd.date_range('2023-03-31', periods=3)}, attrs={'source': 'fixture'})
    kwargs = {'engine': engine}
    if format:
        kwargs['format'] = format
    data.to_netcdf(path, encoding={'t': {'dtype': 'int16', '_FillValue': -32768,
                                        'scale_factor': .01, 'add_offset': 273.15}}, **kwargs)
    raw = path.read_bytes()
    open_dataset = xr.open_dataset
    with path.open('rb') as stream, open_dataset(stream, engine=engine) as expected:
        expected = expected.load()

    def guarded(source, **kwargs):
        assert not isinstance(source, (str, Path)), 'Backend must not receive a filename'
        assert kwargs['engine'] == engine
        return open_dataset(source, **kwargs)

    monkeypatch.setattr(xr, 'open_dataset', guarded)
    actual = load_dataset(path)
    xr.testing.assert_identical(actual, expected)
    assert path.read_bytes() == raw
    # Windows refuses this rename if a backend still owns the open file.
    path.rename(path.with_suffix('.closed'))
    assert actual.t.values[-1] == pytest.approx(290.)


@pytest.mark.parametrize('engine,format', [('h5netcdf', None), ('scipy', 'NETCDF3_CLASSIC')])
def test_raw_decode_options_and_native_writeback_under_unicode_path(tmp_path, engine, format):
    from existing_air_converters.native_netcdf_update import update_variable

    folder = tmp_path / '原生字段 with spaces'
    folder.mkdir()
    path, output = folder / '原始.nc', folder / '结果.nc'
    data = xr.Dataset({'t': ('x', np.array([280., 281.], dtype='f4'), {'units': 'K'}),
                       'keep': ('x', np.array([1, 2], dtype='i2'))}, attrs={'source': 'fixture'})
    kwargs = {'engine': engine}
    if format:
        kwargs['format'] = format
    data.to_netcdf(path, **kwargs)
    raw = path.read_bytes()
    result = update_variable(path, output, 't', np.array([285., 281.], dtype='f4'),
                             allowed_mask=np.array([True, False]))
    assert result['all_metadata_preserved'] and result['all_non_target_variables_preserved']
    assert result['outside_allowed_mask_preserved'] and result['changed_values'] == 1
    actual = load_dataset(output, decode_times=False, mask_and_scale=False)
    np.testing.assert_array_equal(actual.t.values, [285., 281.])
    xr.testing.assert_identical(actual.keep, data.keep)
    assert actual.attrs == data.attrs and path.read_bytes() == raw
    with pytest.raises(ValueError, match='already exists'):
        update_variable(path, output, 't', np.array([290., 281.], dtype='f4'))


@pytest.mark.parametrize('payload', [b'not NetCDF', b'CDF\x05unsupported', b'CDF\x01broken',
                                    b'\x89HDF\r\n\x1a\nbroken'])
def test_bad_products_are_not_repaired_or_replaced(tmp_path, payload):
    path = tmp_path / '损坏.nc'
    path.write_bytes(payload)
    with pytest.raises((ValueError, OSError, TypeError)):
        load_dataset(path)
    assert path.read_bytes() == payload
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / '不存在.nc')


def test_stability_reads_baseline_and_all_deployments_under_unicode_path(tmp_path, monkeypatch):
    from ecodata.algorithms.platform_search import SearchInput
    from ecodata.algorithms.stability import assess

    root = tmp_path / '诊断产品'
    root.mkdir()
    ds = xr.Dataset({'v': ('x', [1., 2., 3.], {'units': '1'})}, attrs={'source_id': 'primary'})
    baseline = root / '基线.nc'
    ds.to_netcdf(baseline, engine='h5netcdf')
    raw = baseline.read_bytes()
    items = [SearchInput('source.nc', 'primary', ds.to_netcdf(engine='scipy'))]
    calls = []

    def execute(inputs, request, graph, folder):
        folder.mkdir()
        with xr.open_dataset(io.BytesIO(inputs[0].content)) as opened:
            product = opened.load()
        path = folder / '扰动.nc'
        product.to_netcdf(path, engine='h5netcdf')
        calls.append(path)
        return {'product': str(path)}

    original_open = xr.open_dataset

    def no_backend_filename(source, **kwargs):
        assert not isinstance(source, (str, Path))
        return original_open(source, **kwargs)

    monkeypatch.setattr(xr, 'open_dataset', no_backend_filename)
    result = assess(items, {'variable': 'v', 'semantic_type': 'continuous'},
                    {'product': str(baseline), 'selected_graph': {'key': 'fixture'}},
                    root / '诊断', execute=execute)
    assert result['status'] == 'DIAGNOSTIC_COMPLETE'
    assert result['new_executions'] == len(calls) == 4 and result['failures'] == 0
    assert baseline.read_bytes() == raw
    assert not result['accuracy_assessed'] and not result['used_for_selection']


def test_temperature_adapter_consumes_unicode_product_without_default_backend(tmp_path, monkeypatch):
    from ecodata.algorithms import platform_search
    from ecodata.models.native_meteorology import fuse_temperature

    root = tmp_path / '气温写回'
    root.mkdir()
    clock = pd.DatetimeIndex(['2023-03-31'])
    lon, lat = np.meshgrid([0., 1.], [0., 1.])
    obs = pd.DataFrame({'station': ['a', 'b', 'c', 'd'], 'longitude': lon.ravel(),
                        'latitude': lat.ravel(), 'time': [clock[0]] * 4, 'value': [280., 281., 282., 283.]})
    product = root / '生成产品.nc'
    values = np.array([[280., 281., 282., 283.]])
    xr.Dataset({'air_temperature': (('time', 'point'), values)}, coords={'time': clock}).to_netcdf(product, engine='h5netcdf')
    original = product.read_bytes()
    monkeypatch.setattr(platform_search, 'process_current_input',
                        lambda *a, **k: {'status': 'PASS', 'product': str(product)})
    original_open = xr.open_dataset

    def no_backend_filename(source, **kwargs):
        assert not isinstance(source, (str, Path))
        return original_open(source, **kwargs)

    monkeypatch.setattr(xr, 'open_dataset', no_backend_filename)
    updated, times, cells, _ = fuse_temperature(obs, {}, np.full((1, 2, 2), 275.), clock,
                                               lon, lat, 'fixture', root, None)
    np.testing.assert_array_equal(updated, values)
    np.testing.assert_array_equal(times, [0])
    np.testing.assert_array_equal(cells, [0, 1, 2, 3])
    assert product.read_bytes() == original


def test_actual_frozen_executor_readback_in_unicode_directory(tmp_path, monkeypatch):
    from ecodata.algorithms import platform_search as api
    from test_advanced_complements import field
    from test_raster_gap_application import registry, contract
    root = tmp_path / '中文执行器 with spaces'
    root.mkdir()
    registry(monkeypatch, root)
    data = field(n=8)
    declaration = contract()
    declaration.update(field='ndvi', valid_range=[-1., 1.])
    request = dict(task='multisource', variable='ndvi', unit='1', semantic_type='continuous',
                   target_lat=data.lat.values.tolist(), target_lon=data.lon.values.tolist(),
                   target_times=[str(t) for t in data.time.values], application_contract=declaration,
                   scoring=dict(registry_version='grid-test-v1'))
    sources = [api.SearchInput('fine.nc', 'primary', data.to_netcdf(engine='scipy'))]
    graph = dict(records=dict(s0=dict(operation='source', parents=[], parameters=dict(source='source0'))), key='s0')
    result = api.execute_frozen_graph(sources, request, graph, root / 'deploy')
    assert Path(result['product']).is_file() and not result['reference_used']
    np.testing.assert_array_equal(load_dataset(result['product']).ndvi.values, data.ndvi.values)


def test_actual_spatial_executor_readback_in_unicode_directory(tmp_path):
    from ecodata.algorithms import platform_search as api
    from test_spatial_relationship_fusion import fixture
    items, request, _ = fixture('stations')
    graph = dict(key='idw', records=dict(idw=dict(operation='spatial.idw', parents=[], parameters=dict(power=2.))))
    result = api.execute_frozen_graph(items, request, graph, tmp_path / '中文空间执行器')
    assert result['status'] == 'PASS' and not result['reference_used']
    assert np.isfinite(load_dataset(result['product']).temperature.values).all()
