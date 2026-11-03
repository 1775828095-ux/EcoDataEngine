"""Non-executable research candidates: never counted as covered capabilities."""
import json
from importlib.resources import files
from ecodata.algorithms.contracts import AlgorithmSpec


def candidate_specs():
    records = json.loads(files("ecodata.algorithms").joinpath("expansion_v2_catalog.json").read_text(encoding="utf-8"))
    records = [r for r in records if r["id"] != "research.dualtimesfield"]
    records += json.loads(files("ecodata.algorithms").joinpath("expansion_v3_catalog.json").read_text(encoding="utf-8"))
    for record in records:
        if record["id"] == "research.phydl_nwp":
            record["commit"] = "4af6dcd93d34367469fc5a9b93f97ddfd81bd86e"
            record["reason"] = "Official main.py actually attempted on CPU; missing data/ningxia_real_train.npy. Both official Zenodo archives inspected separately; see v3/source_manifest.json. No fabricated data or successful PDE/forecast run claimed."
            record["capabilities"] = ["terrain_auxiliary", "terrain_aware_weather_downscaling", "meteorological_downscaling"]
    return tuple(AlgorithmSpec(
        r["id"], r["name"], r["commit"], "capability_research", "research_only" if r.get("verification_status") == "RESEARCH_CANDIDATE" else "blocked", ("task_specific",), ("continuous",),
        requires_training=r["requires_training"], requires_weights=r["requires_weights"], requires_auxiliary=r["requires_auxiliary"],
        scale_range=(1.0, 1.0),
        license=r["license"], backend="official implementation; NOT executed", source_url=r["repository"],
        capabilities=tuple(r["capabilities"]), preconditions=(r["inputs"],), input_contract={"requirements": r["inputs"]},
        output_contract={"description": r["output"], "arbitrary_resolution_verified": False}, runtime_requirements=(r["reason"],),
        verification_status=r.get("verification_status", "BLOCKED"), verification_evidence=("output/algorithm_library_expansion/upstream/",),
        official_entry_point=r["entry"],
        notes=r["reason"]) for r in records)
