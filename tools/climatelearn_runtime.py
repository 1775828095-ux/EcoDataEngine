"""Load original ClimateLearn files without unrelated package-wide imports."""
import importlib
from pathlib import Path
import sys
import types


def official_classes():
    root = Path(__file__).resolve().parents[1] / "external/climatelearn/src/climate_learn"
    package = types.ModuleType("official_climatelearn")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    net = importlib.import_module("official_climatelearn.models.hub.resnet").ResNet
    module = importlib.import_module("official_climatelearn.models.module").LitModule
    return net, module
