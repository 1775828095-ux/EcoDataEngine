"""Read-only registry audit; write the inventory before installing new candidates."""
import json
from pathlib import Path

from ecodata.algorithms.capabilities import capability_catalog
from ecodata.algorithms.registry import create_default_registry
from ecodata.engines.grid import GridEngine


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "output" / "algorithm_library_expansion"
    output.mkdir(parents=True, exist_ok=True)
    registry = create_default_registry()
    report = {"registry": [spec.as_dict() for spec in registry.specs()],
              "capabilities": [record.as_dict() for record in capability_catalog(registry)],
              "xesmf": GridEngine.availability(),
              "status_note": "available is an execution declaration, not the ten-step VERIFIED evidence status"}
    path = output / "initial_registry_audit.json"
    if path.exists():
        raise SystemExit("Initial audit already exists; preserving the pre-expansion inventory")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
