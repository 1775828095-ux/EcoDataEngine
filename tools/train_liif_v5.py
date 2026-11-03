"""Small real ERA5 experiment using official LIIF, never image pretrained weights."""
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

from liif_runtime import ROOT, SOURCE, load, model_spec

OUT = ROOT / "output/algorithm_library_expansion/v5"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    upstream = ROOT / "output/algorithm_library_expansion/upstream/yinboc__liif"
    evidence = json.loads((upstream / "source_evidence.json").read_text(encoding="utf-8"))
    for entry in evidence["files"]:
        target = SOURCE / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(upstream / entry["path"], target)
    shutil.copy2(upstream / "source_evidence.json", SOURCE / "source_evidence.json")
    torch.manual_seed(20261017)
    torch.set_num_threads(2)
    models, utils = load()
    model = models.make(model_spec()).cpu()
    with xr.open_dataset(ROOT / "output/algorithm_library_expansion/v4/era5_bth_t2m_reference.nc", engine="h5netcdf") as ds:
        reference = ds.load()
    truth = torch.tensor(reference.t2m.values[:, None], dtype=torch.float32)
    mean, std = float(truth[:240].mean()), float(truth[:240].std())
    normalized = (truth-mean)/std
    coarse = F.avg_pool2d(normalized, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    best, best_state, logs, steps = float("inf"), None, [], 0
    started = time.perf_counter()
    for epoch in range(5):
        model.train()
        losses = []
        for batch, indices in enumerate(torch.randperm(240).split(8)):
            size = (9, 12, 18, 24)[(epoch+batch) % 4]
            coords = utils.make_coord((size, size)).unsqueeze(0).repeat(len(indices), 1, 1)
            cells = torch.ones_like(coords)*(2/size)
            target = F.interpolate(normalized[indices], size=(size, size), mode="area").flatten(2).transpose(1, 2)
            prediction = model(coarse[indices].repeat(1, 3, 1, 1), coords, cells)
            loss = F.l1_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            c = utils.make_coord((24, 24)).unsqueeze(0).repeat(48, 1, 1)
            pred = model(coarse[240:288].repeat(1, 3, 1, 1), c, torch.ones_like(c)/12)
            val = float(torch.mean((pred-normalized[240:288].flatten(2).transpose(1, 2))**2))
        logs.append({"epoch": epoch+1, "training_L1": float(np.mean(losses)), "validation_MSE": val})
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
    weights = OUT / "liif_era5_local.pt"
    torch.save({"state_dict": best_state, "spec": model_spec(), "mean": mean, "std": std}, weights)
    contract = {"commit": evidence["commit"], "license": "BSD-3-Clause", "weight_origin": "LOCAL_TRAINING",
        "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "optimizer_steps": steps,
        "updated_parameters": sum(not torch.equal(initial[k], v) for k, v in best_state.items()),
        "epochs": 5, "random_seed": 20261017, "training_seconds": time.perf_counter()-started,
        "train_hours": 240, "validation_hours": 48, "heldout_hours": 48,
        "normalization_fit": "training only", "variable": "air_temperature", "units": "K", "crs": "EPSG:4326",
        "input_shape": [6, 6], "reference_shape": [24, 24], "tested_scales": [1.5, 2, 3, 4],
        "bounds": [float(reference.lat[0]-.125), float(reference.lat[-1]+.125), float(reference.lon[0]-.125), float(reference.lon[-1]+.125)],
        "source": "v4/era5_bth_t2m_reference.nc", "source_sha256": hashlib.sha256((ROOT / "output/algorithm_library_expansion/v4/era5_bth_t2m_reference.nc").read_bytes()).hexdigest(),
        "input_representation": "single normalized temperature copied into three official EDSR input channels; scalar LIIF output",
        "device_adaptation": "one .cuda() replaced at module loading by .to(feat.device); original source retained unchanged",
        "independent_truth": False, "scope": "regional ERA5 degrade/reconstruct, not arbitrary variables or guaranteed fine-scale truth", "logs": logs}
    (OUT / "liif_training.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(contract))


if __name__ == "__main__":
    main()
