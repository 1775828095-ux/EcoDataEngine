import pandas as pd
import pytest
import xarray as xr

from ecodata.core.exceptions import SchemaError
from ecodata.fusion.neural_reconstruction import (
    NeuralReconstructionConfig,
    resolve_variable_policy,
    run_neural_reconstruction,
    split_stations,
)


def test_station_split_is_by_station_and_has_no_leakage():
    rows = []
    for station in range(25):
        for hour in range(2):
            rows.append({"station_id": f"S{station:02}", "time": f"2026-01-01 {hour:02}:00",
                         "latitude": station, "longitude": station, "value": station + hour})
    train, validation = split_stations(pd.DataFrame(rows), 20, 7)
    assert validation.station_id.nunique() == 20
    assert set(train.station_id).isdisjoint(set(validation.station_id))


def test_station_split_rejects_too_few_stations():
    frame = pd.DataFrame({"station_id": ["A", "B"], "time": ["2026-01-01"] * 2,
                          "latitude": [1, 2], "longitude": [1, 2], "value": [1, 2]})
    with pytest.raises(SchemaError, match="无法留出"):
        split_stations(frame, 2, 1)


@pytest.mark.parametrize('backend', ['fno', 'multiscale_uno', 'observation_guided_uno'])
def test_fno_sample_runs_without_validation_leakage(backend):
    pytest.importorskip("torch")
    pytest.importorskip("neuralop")
    if backend != 'fno':
        pytest.importorskip('focal_frequency_loss')
    root = __import__("pathlib").Path(__file__).resolve().parents[1] / "examples" / "datasets"
    background = xr.load_dataset(root / "p3_pm25_coarse_10km_48h.nc", engine="h5netcdf").isel(time=slice(0, 4))
    truth = xr.load_dataset(root / "p3_pm25_truth_2p5km_48h.nc", engine="h5netcdf").isel(time=slice(0, 4))
    train = pd.read_csv(root / "p3_pm25_train_stations.csv")
    validation = pd.read_csv(root / "p3_pm25_validation_stations.csv")
    keep = set(pd.to_datetime(background.time.values))
    stations = pd.concat([train, validation], ignore_index=True)
    stations = stations[pd.to_datetime(stations.time).isin(keep)]
    result = run_neural_reconstruction(
        background, stations, evaluation_truth=truth,
        auxiliaries=(xr.Dataset({"temperature": truth.value * 0 + 280,
                                "humidity": truth.value * 0 + 60}, attrs={"crs": "EPSG:4326"}),),
        config=NeuralReconstructionConfig(epochs=1, hidden_channels=4, n_layers=1, backend=backend),
    )
    assert result.report["station_overlap"] == []
    assert result.report["metrics"]["validation_station_count"] == 10
    assert result.field.value.shape == (4, 24, 24)
    assert result.report["metrics"]["rmse"] >= 0
    assert result.report["auxiliary_variables"] == ["temperature", "humidity"]
    altered = stations.copy()
    altered.loc[altered.split == "validation", "value"] += 100
    if backend == 'observation_guided_uno':
        altered['observation_error'] = 1.
        altered.loc[altered.split == 'validation', 'observation_error'] = 100.
        altered['quality_weight'] = 1.
        altered.loc[altered.split == 'validation', 'quality_weight'] = .1
    repeated = run_neural_reconstruction(
        background, altered, evaluation_truth=truth,
        auxiliaries=(xr.Dataset({"temperature": truth.value * 0 + 280,
                                "humidity": truth.value * 0 + 60}, attrs={"crs": "EPSG:4326"}),),
        config=NeuralReconstructionConfig(epochs=1, hidden_channels=4, n_layers=1, backend=backend),
    )
    xr.testing.assert_equal(result.field, repeated.field)
    reference_changed = run_neural_reconstruction(
        background, stations, evaluation_truth=truth + 100,
        auxiliaries=(xr.Dataset({"temperature": truth.value * 0 + 280,
                                "humidity": truth.value * 0 + 60}, attrs={"crs": "EPSG:4326"}),),
        config=NeuralReconstructionConfig(epochs=1, hidden_channels=4, n_layers=1, backend=backend))
    xr.testing.assert_equal(result.field, reference_changed.field)


def test_encoder_ignores_padding_and_is_permutation_invariant():
    torch = pytest.importorskip("torch")
    from ecodata.fusion.observation_encoder import SparseObservationEncoder
    encoder = SparseObservationEncoder(4)
    points = torch.tensor([[[0., 0., 1., 1.], [1., 1., 2., .5], [99., 99., 99., 1.]]])
    valid = torch.tensor([[True, True, False]])
    queries = torch.tensor([[.1, .2], [.5, .6]])
    actual = encoder(points, valid, queries)
    assert torch.allclose(actual, encoder(points[:, [1, 0, 2]], valid[:, [1, 0, 2]], queries))
    points[:, 2] = -999
    assert torch.allclose(actual, encoder(points, valid, queries))
    assert torch.isfinite(encoder(points, torch.zeros_like(valid), queries)).all()


def test_unknown_qa_and_spatial_overlap_are_rejected():
    frame = pd.DataFrame({"station_id": ["A", "B"], "time": ["2026-01-01"] * 2,
                          "latitude": [1, 1], "longitude": [1, 1], "value": [1, 2],
                          "split": ["train", "validation"], "qc_flag": ["VALID", "VALID"]})
    with pytest.raises(SchemaError, match="位置重叠"):
        split_stations(frame, 1, 1)
    frame.loc[0, "qc_flag"] = "UNDEFINED"
    with pytest.raises(SchemaError, match="未知 QA"):
        split_stations(frame, 1, 1)


def test_variable_policy_is_physical_metadata_driven():
    temperature = xr.DataArray([280.0], attrs={"units": "K", "standard_name": "air_temperature"})
    policy = resolve_variable_policy("t2m", temperature, NeuralReconstructionConfig())
    assert policy.family == "temperature"
    assert policy.nonnegative is False
    unknown = xr.DataArray([1.0], attrs={"units": "1"})
    with pytest.raises(SchemaError, match="variable_family"):
        resolve_variable_policy("value", unknown, NeuralReconstructionConfig())


def test_checkpoint_inference_and_incompatible_variable_rejection(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("neuralop")
    root = __import__("pathlib").Path(__file__).resolve().parents[1] / "examples" / "datasets"
    background = xr.load_dataset(root / "p3_pm25_coarse_10km_48h.nc", engine="h5netcdf").isel(time=slice(0, 2))
    train = pd.read_csv(root / "p3_pm25_train_stations.csv")
    validation = pd.read_csv(root / "p3_pm25_validation_stations.csv")
    stations = pd.concat([train, validation], ignore_index=True)
    stations = stations[pd.to_datetime(stations.time).isin(background.time.values)]
    checkpoint = tmp_path / "fno.pt"
    trained = run_neural_reconstruction(background, stations, config=NeuralReconstructionConfig(
        epochs=1, hidden_channels=4, n_layers=1, backend="fno", save_checkpoint_path=str(checkpoint)))
    inferred = run_neural_reconstruction(background, stations, config=NeuralReconstructionConfig(
        hidden_channels=4, n_layers=1, backend="fno", training_mode="inference",
        load_checkpoint_path=str(checkpoint)))
    xr.testing.assert_allclose(trained.field, inferred.field)
    assert inferred.report["training_control"]["epochs_completed"] == 0
    incompatible = background.rename({"value": "temperature"})
    incompatible.temperature.attrs.update(units="K", standard_name="air_temperature")
    with pytest.raises(SchemaError, match="checkpoint"):
        run_neural_reconstruction(incompatible, stations.assign(unit="K"), config=NeuralReconstructionConfig(
            hidden_channels=4, n_layers=1, backend="fno", training_mode="inference",
            load_checkpoint_path=str(checkpoint)))


def test_official_attention_padding_and_grid_features():
    torch = pytest.importorskip('torch')
    from ecodata.fusion.observation_encoder import AttentionObservationEncoder
    encoder = AttentionObservationEncoder(4, 2)
    points = torch.tensor([[[0., 0., 1., 1.], [1., 1., 2., .5], [99., 99., 99., 1.]]])
    valid = torch.tensor([[True, True, False]])
    queries = torch.tensor([[.1, .2], [.5, .6]])
    grid = torch.zeros(1, 2, 1, 2)
    actual = encoder(points, valid, queries, grid)
    assert torch.allclose(actual, encoder(points[:, [1, 0, 2]], valid[:, [1, 0, 2]], queries, grid), atol=1e-6)
    assert torch.equal(encoder(points, torch.zeros_like(valid), queries, grid), torch.zeros_like(actual))
    assert not torch.allclose(actual, encoder(points, valid, queries, grid + 1))
