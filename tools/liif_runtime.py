"""Load pinned official LIIF; the sole source adaptation is CUDA -> input device."""
import importlib.util
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external/liif"


def load():
    # Official modules use absolute models/utils imports. This runs only in an
    # isolated subprocess, not in the web process's module namespace.
    spec = importlib.util.spec_from_file_location("utils", SOURCE / "utils.py")
    utils = importlib.util.module_from_spec(spec)
    sys.modules["utils"] = utils
    spec.loader.exec_module(utils)
    package = types.ModuleType("models")
    package.__path__ = [str(SOURCE / "models")]
    sys.modules["models"] = package
    for name in ("models", "edsr", "mlp", "liif"):
        path = SOURCE / "models" / (name + ".py")
        module = types.ModuleType("models." + name)
        module.__file__ = str(path)
        sys.modules[module.__name__] = module
        text = path.read_text(encoding="utf-8")
        if name == "liif":
            assert text.count(".cuda()") == 1
            text = text.replace(".cuda()", ".to(feat.device)")
        exec(compile(text, str(path), "exec"), module.__dict__)
        if name == "models":
            package.make, package.register = module.make, module.register
    return package, utils


def model_spec():
    return {"name": "liif", "args": {
        "encoder_spec": {"name": "edsr-baseline", "args": {"n_resblocks": 2, "n_feats": 16, "no_upsampling": True}},
        "imnet_spec": {"name": "mlp", "args": {"out_dim": 1, "hidden_list": [32, 32]}}}}
