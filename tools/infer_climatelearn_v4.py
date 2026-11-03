"""Inference using the locally trained official ClimateLearn architecture."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from climatelearn_runtime import official_classes


def main():
    folder = Path(sys.argv[1])
    root = Path(__file__).resolve().parents[1]
    checkpoint = torch.load(root / "output/algorithm_library_expansion/v4/climatelearn_resnet_local.pt", map_location="cpu", weights_only=True)
    torch.set_num_threads(2)
    ResNet, _ = official_classes()
    model = ResNet(**checkpoint["network_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with np.load(folder / "input.npz", allow_pickle=False) as data:
        field = torch.tensor(data["values"][:, None], dtype=torch.float32)
    results = []
    with torch.no_grad():
        for part in field.split(16):
            high = F.interpolate(part, size=(24, 24), mode="bilinear", align_corners=False)
            results.append((model((high-checkpoint["mean"])/checkpoint["std"])*checkpoint["std"]+checkpoint["mean"])[:, 0].numpy())
    np.save(folder / "output.npy", np.concatenate(results), allow_pickle=False)


if __name__ == "__main__":
    main()
