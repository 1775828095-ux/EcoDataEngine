"""Small real ERA5 experiment using unmodified ClimateLearn ResNet/LitModule."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import xarray as xr
from climatelearn_runtime import official_classes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/algorithm_library_expansion/v4"
COMMIT = "b48fb0242acc47e365af86bfbd9dd86e9dcbd6d2"


def batch(items):
    x, y = zip(*items)
    return torch.stack(x), torch.stack(y), ["2m_temperature"], ["2m_temperature"]


def identity(value):
    return value


def main():
    torch.set_num_threads(2)
    pl.seed_everything(42, workers=True)
    with xr.open_dataset(OUT / "era5_bth_t2m_reference.nc", engine="h5netcdf") as source:
        high = source.load()
    low = high.coarsen(lat=4, lon=4, boundary="exact").mean(keep_attrs=True)
    low.to_netcdf(OUT / "era5_bth_t2m_coarse.nc", engine="h5netcdf")
    y = torch.tensor(high.t2m.values[:, None], dtype=torch.float32)
    x = F.interpolate(torch.tensor(low.t2m.values[:, None], dtype=torch.float32), size=(24, 24), mode="bilinear", align_corners=False)
    mean, std = float(y[:240].mean()), float(y[:240].std())
    train = TensorDataset((x[:240]-mean)/std, (y[:240]-mean)/std)
    val = TensorDataset((x[240:288]-mean)/std, (y[240:288]-mean)/std)
    ResNet, LitModule = official_classes()
    config = dict(in_channels=1, out_channels=1, history=1, hidden_channels=16, n_blocks=2, dropout=0.)
    net = ResNet(**config)
    initial = {k: v.detach().clone() for k, v in net.state_dict().items()}
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    module = LitModule(net, optimizer, None, torch.nn.MSELoss(), [torch.nn.MSELoss()], [torch.nn.MSELoss()],
                       val_target_transforms=[identity], test_target_transforms=[identity])
    callback = ModelCheckpoint(dirpath=OUT / "climatelearn_training", monitor="val/loss_0:agggregate", mode="min", save_top_k=1, filename="best")
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=5, logger=False, enable_progress_bar=False,
                         callbacks=[callback], num_sanity_val_steps=0, deterministic=True, enable_model_summary=False)
    trainer.fit(module, DataLoader(train, batch_size=16, shuffle=True, collate_fn=batch, num_workers=0),
                DataLoader(val, batch_size=16, collate_fn=batch, num_workers=0))
    trained = torch.load(callback.best_model_path, map_location="cpu", weights_only=False)["state_dict"]
    module.load_state_dict(trained)
    net.eval()
    changed = sum(not torch.equal(initial[k], v) for k, v in net.state_dict().items())
    assert changed and trainer.global_step > 0
    checkpoint = {"state_dict": net.state_dict(), "network_config": config, "mean": mean, "std": std, "commit": COMMIT,
        "variable": "air_temperature", "units": "K", "scale_factor": 4, "target_lat": high.lat.values.tolist(),
        "target_lon": high.lon.values.tolist(), "input_lat": low.lat.values.tolist(), "input_lon": low.lon.values.tolist(),
        "crs": "EPSG:4326", "weight_origin": "LOCAL_TRAINING; NOT official pretrained", "epochs": 5,
        "train_hours": 240, "validation_hours": 48, "test_hours": 48, "seed": 42,
        "reference_sha256": hashlib.sha256((OUT / "era5_bth_t2m_reference.nc").read_bytes()).hexdigest()}
    torch.save(checkpoint, OUT / "climatelearn_resnet_local.pt")
    (OUT / "climatelearn_model_contract.json").write_text(json.dumps({k: v for k, v in checkpoint.items() if k != "state_dict"}, indent=2), encoding="utf-8")
    with torch.no_grad():
        predicted = net((x[288:]-mean)/std)*std+mean
    output = high.isel(time=slice(288, None)).copy()
    output["t2m"] = output.t2m.copy(data=predicted[:, 0].numpy())
    output["bilinear"] = (("time", "lat", "lon"), x[288:, 0].numpy())
    output.bilinear.attrs["units"] = "K"
    output.to_netcdf(OUT / "climatelearn_heldout_predictions.nc", engine="h5netcdf")
    metrics = {}
    for name, values in (("climatelearn", predicted), ("bilinear", x[288:])):
        error = values-y[288:]
        metrics[name] = {"heldout_reference_rmse_K": float((error**2).mean().sqrt()), "heldout_reference_mae_K": float(error.abs().mean())}
    record = {"status": "TRAINED_AND_INFERRED", "official_architecture": "ResNet", "official_training_module": "LitModule",
        "commit": COMMIT, "parameters_updated": changed, "optimizer_steps": trainer.global_step, "config": config,
        "checkpoint": "climatelearn_resnet_local.pt", "checkpoint_sha256": hashlib.sha256((OUT / "climatelearn_resnet_local.pt").read_bytes()).hexdigest(),
        "weight_origin": "LOCAL_TRAINING", "epochs": 5, "split": "first240 train / next48 validation / last48 test",
        "normalization_fit": "training reference only", "metrics": metrics,
        "independent_measurement_truth_available": False,
        "metric_scope": "held-out ERA5 native-grid degrade-and-reconstruct reference; not independent real-world high-resolution accuracy",
        "arbitrary_resolution_verified": False}
    (OUT / "climatelearn_training.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
