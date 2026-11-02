from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from ecodata.core.exceptions import SchemaError
from ecodata.fusion.neural_inputs import read_experiment_grid, read_experiment_stations
from ecodata.fusion.neural_reconstruction import observation_reliability, NeuralReconstructionConfig, default_target_coordinates, run_neural_reconstruction
from ecodata.fusion.neural_experiment import evaluate_reconstruction, run_reconstruction_comparison, save_comparison


def test_reliability_uses_qa_completeness_and_error():
    times = pd.date_range('2024-01-01', periods=2, freq='h')
    train = pd.DataFrame({'station_id':['a','a','b'], 'time':[times[0],times[1],times[0]],
                          'quality_weight':[1., .5, 1.], 'observation_error':[1., 1., 2.]})
    result, report = observation_reliability(train, times)
    assert report['completeness'] == {'a':1., 'b':.5}
    assert result.loss_weight.iloc[0] == pytest.approx(result.loss_weight.iloc[1]*2)
    assert result.loss_weight.iloc[0] == pytest.approx(result.loss_weight.iloc[2]*8)
    assert result.loss_weight.mean() == pytest.approx(1)
    assert 0 < result.quality_weight.min() <= result.quality_weight.max() <= 1
    with pytest.raises(SchemaError, match='标准差'):
        observation_reliability(train.assign(observation_error=-1), times)
    with pytest.raises(SchemaError, match='重复'):
        observation_reliability(pd.concat([train, train.iloc[:1]]), times)


def _tiff(crs='EPSG:4326', unit='ug/m3'):
    with MemoryFile() as memory:
        with memory.open(driver='GTiff', width=4, height=3, count=1, dtype='float32', crs=crs,
                         transform=from_origin(110, 31, .1, .1) if crs=='EPSG:4326' else from_origin(0,3000,1000,1000)) as source:
            source.write(np.arange(12, dtype='float32').reshape(1,3,4))
            if unit:
                source.update_tags(1, units=unit)
        return memory.read()


def test_geotiff_reader_unit_time_direction_and_projection():
    ds = read_experiment_grid('background.tif', _tiff(), timestamp='2024-01-01')
    assert ds.value.dims == ('time','lat','lon')
    assert np.all(np.diff(ds.lat)>0)
    assert ds.value.values[0,0,0] == 8
    assert ds.value.attrs['units'] == 'ug/m3'
    with pytest.raises(SchemaError, match='单位'):
        read_experiment_grid('unknown.tif', _tiff(unit=None))
    projected = read_experiment_grid('dem.tif', _tiff('EPSG:3857', 'm'))
    assert projected.attrs['crs'] == 'EPSG:4326'
    assert abs(projected.lon.values).max() < 1
    assert projected.value.attrs['units'] == 'm'


def test_multi_variable_requires_explicit_selection(tmp_path):
    ds = xr.Dataset({'temperature':(('lat','lon'),np.ones((2,2))),
                     'humidity':(('lat','lon'),np.ones((2,2)))},coords={'lat':[1,2],'lon':[3,4]}, attrs={'crs':'EPSG:4326'})
    ds.temperature.attrs['units'] = 'K'
    path = tmp_path/'weather.nc'
    ds.to_netcdf(path, engine='h5netcdf')
    with pytest.raises(SchemaError, match='选择'):
        read_experiment_grid(path.name, path.read_bytes())
    read = read_experiment_grid(path.name, path.read_bytes(), variable='temperature')
    assert list(read.data_vars) == ['temperature']


def test_projected_time_bounds_preserved(tmp_path):
    times = pd.date_range('2024-01-01', periods=2, freq='h')
    ds = xr.Dataset({'PM25':(('time','y','x'), np.ones((2,3,4))),
                     'time_bnds':(('time','bnds'), np.stack([times.values, (times+pd.Timedelta(hours=1)).values],axis=1))},
                    coords={'time':times,'x':[500,1500,2500,3500],'y':[500,1500,2500]}, attrs={'crs':'EPSG:3857'})
    ds.PM25.attrs['units'] = 'ug/m3'
    ds.time.attrs['bounds'] = 'time_bnds'
    path = tmp_path/'projected.nc'
    ds.to_netcdf(path, engine='h5netcdf')
    result = read_experiment_grid(path.name, path.read_bytes())
    xr.testing.assert_equal(result.time_bnds, ds.time_bnds)
    assert result.time.attrs['bounds'] == 'time_bnds'


def test_default_grid_preserves_coarse_pixel_footprint():
    source = xr.DataArray(np.ones((3,4)), dims=('lat','lon'), coords={'lat':[1,2,3],'lon':[4,5,6,7]})
    lat, lon = default_target_coordinates(source)
    assert len(lat) == 12 and len(lon) == 16
    assert lat[0]-(lat[1]-lat[0])/2 == pytest.approx(.5)
    assert lon[-1]+(lon[1]-lon[0])/2 == pytest.approx(7.5)


def test_station_column_mapping_and_source_metadata():
    frame = read_experiment_stations('airnow.csv', b'station_id,PM25,unit\n001,12.5,ug/m3\n', value_column='PM25')
    assert frame.station_id.iloc[0] == '001'
    assert frame.value.iloc[0] == 12.5
    assert frame.attrs['value_column_mapping'] == {'PM25':'value'}


def _sample(hours=2):
    root = Path(__file__).resolve().parents[1]/'examples'/'datasets'
    bg = xr.load_dataset(root/'p3_pm25_coarse_10km_48h.nc').isel(time=slice(0,hours))
    truth = xr.load_dataset(root/'p3_pm25_truth_2p5km_48h.nc').isel(time=slice(0,hours))
    stations = pd.concat([pd.read_csv(root/'p3_pm25_train_stations.csv'),pd.read_csv(root/'p3_pm25_validation_stations.csv')])
    stations = stations[pd.to_datetime(stations.time).isin(bg.time.values)]
    return bg, truth, stations


def test_all_baselines_same_split_grid_and_export(tmp_path):
    pytest.importorskip('neuralop')
    pytest.importorskip('focal_frequency_loss')
    bg, reference, stations = _sample()
    result = run_reconstruction_comparison(bg, stations, evaluation_truth=reference,
        config=NeuralReconstructionConfig(epochs=1, hidden_channels=4, n_layers=1))
    assert result['comparison'].status.isin(['PASS','WARN']).all(), result['comparison'].to_string()
    assert len(result['fields']) == 6
    assert result['comparison'].validation_record_count.nunique() == 1
    for ds in result['fields'].values():
        assert ds.value.shape == (2,24,24)
    save_comparison(result, tmp_path/'experiment')
    assert len(list((tmp_path/'experiment').glob('*.nc'))) == 6
    with pytest.raises(FileExistsError):
        save_comparison(result, tmp_path/'experiment')


def test_no_truth_means_no_full_field_accuracy_or_spatial_correlation():
    bg, reference, stations = _sample(1)
    field = bg.interp(lat=reference.lat, lon=reference.lon, kwargs={'fill_value':'extrapolate'})
    held = stations[stations.split=='validation']
    predictions, metrics = evaluate_reconstruction(field, held)
    assert metrics['full_field_rmse'] is None
    assert metrics['spatial_correlation'] is None
    assert metrics['peak_amplitude_mae'] is None
    assert metrics['validation_peak_error'] >= 0
    assert len(predictions) == 10


def test_auxiliary_categories_and_incompatible_time():
    pytest.importorskip('neuralop')
    pytest.importorskip('focal_frequency_loss')
    bg, reference, stations = _sample()
    auxiliary = bg.isel(time=0, drop=True).rename({'value':'land_use'})
    auxiliary.land_use.values = np.indices(auxiliary.land_use.shape)[1] % 2
    auxiliary.land_use.attrs.update(units='1', data_kind='categorical')
    config = NeuralReconstructionConfig(backend='observation_guided_uno',epochs=1,hidden_channels=4)
    result = run_neural_reconstruction(bg,stations,auxiliaries=[auxiliary],evaluation_truth=reference,config=config)
    assert result.report['auxiliary_variables'] == ['land_use=0', 'land_use=1']
    assert result.report['auxiliary_alignment'][0]['method'] == 'nearest'
    with pytest.raises(SchemaError, match='时间对齐'):
        run_neural_reconstruction(bg,stations,auxiliaries=[bg.isel(time=slice(0,1))],evaluation_truth=reference,config=config)


def test_observation_residual_path_gradients_and_missing_rows():
    torch = pytest.importorskip('torch')
    from ecodata.fusion.observation_encoder import AttentionObservationEncoder
    encoder = AttentionObservationEncoder(4,2,residual_gate=.5)
    points = torch.tensor([[[0.,0.,1.,1.],[1.,1.,2.,.5],[99.,99.,99.,1.]]])
    valid = torch.tensor([[True,True,False]])
    queries = torch.tensor([[0.,0.],[.5,.5]])
    grid = torch.zeros(1,2,1,2)
    encoded = encoder(points,valid,queries,grid)
    permuted = encoder(points[:,[1,0,2]],valid[:,[1,0,2]],queries,grid)
    assert torch.allclose(encoded,permuted,atol=1e-6)
    missing = encoder(points,torch.zeros_like(valid),queries,grid)
    assert torch.equal(missing,torch.zeros_like(missing))
    encoded.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in encoder.parameters() if p.grad is not None)
