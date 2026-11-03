"""Record checked upstream constraints, not placeholder executable algorithms."""
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/algorithm_library_expansion"


def main():
    notebook = OUTPUT / "upstream/IBM__granite-wxc/examples/merra2_downscaling/notebooks/merra2_downscaling_inference.ipynb"
    metadata_path = OUTPUT / "upstream/IBM__granite-wxc/source_evidence.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if notebook.exists():
        relative = notebook.relative_to(metadata_path.parent).as_posix()
        if not any(item["path"] == relative for item in metadata["files"]):
            metadata["files"].append({"path": relative,
                "url": f"https://raw.githubusercontent.com/IBM/granite-wxc/{metadata['commit']}/{relative}",
                "bytes": notebook.stat().st_size, "sha256": hashlib.sha256(notebook.read_bytes()).hexdigest()})
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    endpoint = "https://huggingface.co/api/models/ibm-granite/granite-geospatial-wxc-downscaling/tree/main"
    response = requests.get(endpoint, timeout=30)
    response.raise_for_status()
    weight = next(item for item in response.json() if item.get("path") == "pytorch_model.bin")
    syntax = {"status": "PASS"}
    try:
        ast.parse((OUTPUT / "upstream/kgavahi__PrecipitationFusion/train.py").read_text(encoding="utf-8"))
    except SyntaxError as exc:
        syntax = {"status": "FAIL", "type": type(exc).__name__, "line": exc.lineno, "message": exc.msg}
    families = [
        ("xESMF", "BLOCKED", "ESMF/ESMPy absent; no Conda/WSL runtime; strict bounds guard added, no kernel execution"),
        ("xarray-regrid", "PARTIAL", "Most-common/fractions/statistics verified on actual inputs; cubic actual output exists but downloaded official suite does not test cubic directly"),
        ("PyKrige", "PARTIAL", "Ordinary and Universal verified; regression/3D adapters lack complete official+real-data evidence"),
        ("MetPy", "VERIFIED", "Official 43 tests plus actual station interpolation and NOAA IGRA vertical-profile output; inverse-distance API means Barnes/Cressman, not generic power-IDW"),
        ("xsdba", "PARTIAL", "EQM official example and actual output verified; DQM/QDM real output and adapter tests passed, direct official method-specific tests pending; ExtremeValues not integrated"),
        ("TopoPyScale", "BLOCKED", "Checked revision requires Python>=3.13; main environment is 3.12.14; pressure-level climate and matching terrain workflow not prepared"),
        ("Thera", "BLOCKED", "GPU_REQUIRED: official Linux/Python3.10/NVIDIA environment not available; no weights downloaded"),
        ("Granite/Prithvi WxC", "BLOCKED", "Official notebook includes CPU path, but needs an 8.23GB checkpoint, multi-level MERRA2 variables and climatology; large downloads not attempted"),
        ("Sentinel-5P SR", "BLOCKED", "Explicit repository license not found in checked tree; no installation or checkpoint execution"),
        ("PRIMER", "BLOCKED", "Explicit repository license not found; CUDA/custom heavy dependencies and no verified checkpoint run; README's two A100 GPUs describe training, not an established inference minimum"),
        ("PrecipitationFusion", "BLOCKED", "Official train.py syntax error at line 55; TensorFlow pipeline has no supplied pretrained checkpoint/inference example in checked repository"),
    ]
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(),
        "granite_checkpoint": {"metadata_endpoint": endpoint, "path": weight["path"], "bytes": weight["size"], "downloaded": False},
        "precipitation_fusion_original_syntax_check": syntax,
        "families": [{"name": name, "status": status, "reason": reason} for name, status, reason in families],
        "paper_sources": {"TopoPyScale": "https://doi.org/10.21105/joss.05059", "TopoSCALE": "https://doi.org/10.5194/gmd-7-387-2014",
            "Thera": "https://arxiv.org/abs/2311.17643", "Prithvi WxC": "https://arxiv.org/abs/2409.13598",
            "Sentinel5P_supervised": "https://arxiv.org/abs/2501.17210", "Sentinel5P_self_supervised": "https://arxiv.org/abs/2604.17652",
            "PRIMER": "https://doi.org/10.1038/s41467-025-67987-9", "PrecipitationFusion": "https://doi.org/10.1016/j.rse.2023.113723"}}
    (OUTPUT / "upstream_readiness.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Recorded readiness of", len(families), "families; checkpoint bytes", weight["size"])


if __name__ == "__main__":
    main()
