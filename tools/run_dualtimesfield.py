"""Fit and query the original upstream model; never fit held-out targets."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch


def main():
    folder = Path(sys.argv[1])
    spec = importlib.util.spec_from_file_location("official_dualtime_models", Path(__file__).resolve().parents[1] / "external/dualtimesfield/interpolation/models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    p = json.loads((folder / "options.json").read_text(encoding="utf-8"))
    torch.set_num_threads(2)
    torch.manual_seed(p["seed"])
    with np.load(folder / "input.npz", allow_pickle=False) as data:
        mask = np.isfinite(data["values"])
        mean, std = data["values"][mask].mean(), data["values"][mask].std()
        std = max(float(std), 1e-6)
        values = torch.tensor(np.where(mask, (data["values"]-mean)/std, 0), dtype=torch.float32).reshape(-1, 1)
        times = torch.tensor(data["times"], dtype=torch.float32)
        queries = torch.tensor(data["queries"], dtype=torch.float32)
    observed = torch.tensor(mask).reshape(-1, 1)
    model = module.DualTimesFieldInterpolator(num_variables=1)
    model.train()
    loss = model.fit(times, values, observed, target_mask=observed, num_epochs=p["epochs"])
    model.eval()
    output = model.predict(queries).detach().cpu().numpy().reshape(-1)*std+mean
    np.save(folder / "prediction.npy", output, allow_pickle=False)
    (folder / "report.json").write_text(json.dumps({"official_class": "DualTimesFieldInterpolator", **p,
        "training_observations": int(mask.sum()), "masked_input_count": int((~mask).sum()), "final_training_loss": loss,
        "normalization_fit": "finite observed training values only", "normalization_mean": float(mean), "normalization_std": std,
        "hidden_values_used_for_training": False}), encoding="utf-8")


if __name__ == "__main__":
    main()
