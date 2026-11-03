"""Real official daily-summary fallback, not a claim of raw-hour readiness."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT / ".venv-golem/Lib/site-packages"))
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.spatial import ConvexHull, Delaunay
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from ecodata.algorithms.dag import Artifact
from ecodata.algorithms.epa_task import EPATask, UNIT
from ecodata.algorithms.registry import create_default_registry
from golem.core.optimisers.graph import OptNode, OptGraph


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("D:/codex/数据/公开数据三层预审_20261018/05_实验落地_20261019"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--region", choices=("four_counties", "central_valley", "northern_california"), default="four_counties")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = args.data
    counties = json.loads((data / "original/four_counties.geojson").read_text())
    region = unary_union([shape(f["geometry"]) for f in counties["features"]])
    if args.region in ("central_valley", "northern_california"):
        basins = json.loads((data / "original/carb_basins.geojson").read_text())
        names = {"SAN JOAQUIN VALLEY", "SACRAMENTO VALLEY"}
        if args.region == "northern_california":
            names.add("SAN FRANCISCO BAY")
        features = [f for f in basins["features"] if str(f["properties"].get("NAME", "")).upper() in names]
        if len(features) != len(names):
            raise ValueError("Official Central Valley basin names not found")
        region = unary_union([shape(f["geometry"]) for f in features])
    rows = []
    with zipfile.ZipFile(data / "original/daily_88101_2022.zip") as archive:
        for chunk in pd.read_csv(archive.open(archive.namelist()[0]), chunksize=100000, low_memory=False):
            keep = chunk["State Code"].eq(6)
            if args.region == "four_counties":
                keep &= chunk["County Code"].isin([37, 59, 65, 71])
            rows.append(chunk.loc[keep])
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(args.output / "regional_raw_daily_summary_2022.csv", index=False)
    table["station_id"] = table["State Code"].astype(str).str.zfill(2)+"-"+table["County Code"].astype(str).str.zfill(3)+"-"+table["Site Num"].astype(str).str.zfill(4)
    # Datum conversion is explicit; no four-county bounding rectangle shortcut.
    table["longitude"], table["latitude"] = np.nan, np.nan
    for datum, subset in table.groupby("Datum"):
        if datum in ("WGS84", "NAD83"):
            tx = Transformer.from_crs("EPSG:4326" if datum == "WGS84" else "EPSG:4269", "EPSG:4326", always_xy=True)
            xx, yy = tx.transform(subset.Longitude.to_numpy(), subset.Latitude.to_numpy())
            table.loc[subset.index, "longitude"], table.loc[subset.index, "latitude"] = xx, yy
    table["in_official_counties"] = [bool(np.isfinite(x) and np.isfinite(y) and region.covers(Point(x, y)))
                                      for x, y in zip(table.longitude, table.latitude)]
    audit = {"raw_county_rows": len(table), "durations": table["Sample Duration"].value_counts().to_dict(),
        "units": table["Units of Measure"].value_counts().to_dict(), "methods": table["Method Code"].value_counts().to_dict(),
        "event_types": table["Event Type"].fillna("NONE").value_counts().to_dict(),
        "negative_rows": int((table["Arithmetic Mean"] < 0).sum()),
        "limitation": "EPA daily summaries, not reaggregation of acquired raw hours. Per-hour qualifiers/revision history unavailable."}
    allowed_events = table["Event Type"].isna() | table["Event Type"].isin(["No Events", "Events Included", "Included"])
    valid = (table["in_official_counties"] & table["Units of Measure"].eq("Micrograms/cubic meter (LC)")
        & table["Sample Duration"].isin(["1 HOUR", "24-HR BLK AVG"])
        & table["Observation Count"].ge(18) & table["Observation Percent"].ge(75)
        & table["Arithmetic Mean"].ge(0) & np.isfinite(table["Arithmetic Mean"])
        & ~pd.to_numeric(table["Method Code"], errors="coerce").isin([236, 238]) & allowed_events)
    accepted = table.loc[valid].copy()
    keys = ["station_id", "POC", "Date Local"]
    conflict = accepted.groupby(keys)["Arithmetic Mean"].agg(lambda x: np.ptp(x.to_numpy(float)))
    bad = conflict[conflict > 1e-8].index
    accepted = accepted.loc[~pd.MultiIndex.from_frame(accepted[keys]).isin(bad)]
    accepted = accepted.sort_values(keys+["Observation Count"], ascending=[True, True, True, False]).drop_duplicates(keys)
    coverage = accepted.groupby(["station_id", "POC"]).agg(valid_days=("Date Local", "nunique"),
        valid_hour_count_sum=("Observation Count", "sum")).reset_index()
    coverage.to_csv(args.output / "poc_daily_coverage_2022.csv", index=False)
    primary = coverage.sort_values(["station_id", "valid_days", "valid_hour_count_sum", "POC"],
        ascending=[True, False, False, True]).drop_duplicates("station_id")
    accepted = accepted.merge(primary[["station_id", "POC"]], on=["station_id", "POC"], how="inner")
    qualified = primary.loc[primary.valid_days.ge(270)]
    accepted = accepted[accepted.station_id.isin(qualified.station_id)].copy()
    accepted["unit"] = UNIT
    accepted["time"], accepted["value"] = accepted["Date Local"], accepted["Arithmetic Mean"]
    accepted.to_csv(args.output / "qualified_continuous_daily_2022.csv", index=False)
    accepted.groupby("time").station_id.nunique().rename("common_sites").to_csv(args.output / "common_dates_2022.csv")
    qualified.to_csv(args.output / "qualified_sites_2022.csv", index=False)
    audit.update(duplicate_conflicting_site_poc_days=len(bad), qualified_sites=len(qualified),
                 accepted_site_days=len(accepted), selected_pocs=primary.to_dict("records"))
    dump(args.output / "daily_quality_audit.json", audit)
    sites = accepted.sort_values("time").drop_duplicates("station_id").sort_values("station_id")
    if args.prepare_only:
        sites[["station_id", "longitude", "latitude", "Datum", "POC", "Local Site Name"]].to_csv(
            args.output / "station_metadata.csv", index=False)
        print(json.dumps({"qualified_sites": len(sites), "site_days": len(accepted), "algorithms_executed": False}))
        return
    if len(sites) < 8:
        raise ValueError("Too few qualified stations even for a labelled smoke test")
    points = sites[["longitude", "latitude"]].to_numpy(float)
    hull = set(ConvexHull(points).vertices)
    interior = [i for i in range(len(sites)) if i not in hull]
    interior.sort(key=lambda i: hashlib.sha256(("11:"+sites.iloc[i].station_id).encode()).hexdigest())
    if len(interior) < 2:
        raise ValueError("Insufficient interior validation stations without extrapolation")
    validation_ids = set(sites.iloc[interior[:2]].station_id)
    # Reserve a test station without using any target values; validation must
    # remain supported by the remaining training hull. Test is not evaluated.
    possible_test = sorted([i for i in range(len(sites)) if i not in interior[:2]],
        key=lambda i: hashlib.sha256(("11:test:"+sites.iloc[i].station_id).encode()).hexdigest())
    test_index = next((i for i in possible_test if (Delaunay(np.delete(points, interior[:2]+[i], axis=0))
                       .find_simplex(points[interior[:2]]) >= 0).all()), None)
    if test_index is None:
        raise ValueError("No geometry-only split supports both fixed validation stations")
    test_ids = {sites.iloc[test_index].station_id}
    sites["split"] = ["validation" if s in validation_ids else "test" if s in test_ids else "train" for s in sites.station_id]
    sites[["station_id", "longitude", "latitude", "Datum", "POC", "Local Site Name", "split"]].to_csv(
        args.output / "station_split.csv", index=False)
    accepted["split"] = accepted.station_id.map(dict(zip(sites.station_id, sites.split)))
    required = set(sites.loc[~sites.split.eq("test"), "station_id"])
    chosen = next((date for date, group in accepted.sort_values("time").groupby("time")
                   if date >= "2022-01-15" and required.issubset(set(group.station_id))), None)
    if chosen is None:
        raise ValueError("No common date supports fixed split")
    # Test targets never enter external inputs or Objective; no test metrics.
    sample = accepted[accepted.time.eq(chosen) & ~accepted.split.eq("test")].copy()
    train, validation = sample[sample.split.eq("train")], sample[sample.split.eq("validation")]
    lat = np.linspace(sites.latitude.min()-.01, sites.latitude.max()+.01, 55)
    lon = np.linspace(sites.longitude.min()-.01, sites.longitude.max()+.01, 65)
    template = xr.Dataset({"pm25": (("lat", "lon"), np.full((len(lat), len(lon)), np.nan))},
                          coords={"lat": lat, "lon": lon}, attrs={"crs": "EPSG:4326", "purpose": "target geometry only, no background values"})
    template.pm25.attrs.update(units=UNIT, data_kind="continuous")
    task = EPATask(create_default_registry(), template, sealed_test_stations=tuple(test_ids))
    external = {"stations": Artifact("epa_train", train)}
    results = []
    for method in ("fusion.idw", "station.pykrige_ordinary"):
        nodes = task.graph(method)
        by_id = {}
        for node in nodes:
            params = dict(node.parameters)
            parents, bindings = [], {}
            for port, (parent, output) in node.inputs.items():
                if by_id[parent] not in parents:
                    parents.append(by_id[parent])
                bindings[port] = [parents.index(by_id[parent]), output]
            params["_port_bindings"] = bindings
            by_id[node.id] = OptNode({"name": node.operation, "params": params}, nodes_from=parents)
        try:
            graph = OptGraph([by_id["quality"]])
            result = task.objective(method, external, validation, args.output / method, golem_graph=graph)
            result.update(method=method, date_local_standard=chosen, true_golem_graph_executed=True)
        except Exception as exc:
            import traceback
            result = {"method": method, "status": "FAIL", "error": str(exc), "traceback": traceback.format_exc()}
        results.append(result)
        dump(args.output / "results.json", results)
    dump(args.output / "experiment_status.json", {"qualified_station_count": len(qualified), "target": 20,
        "data_ready": False, "pipeline_smoke_passed": all(r["status"] == "PASS" for r in results),
        "date": chosen, "split": sites.groupby("split").station_id.apply(list).to_dict(),
        "formal_2018_2020_train_2021_val_2022_test": False,
        "scope": "2022 real daily-summary spatial holdout smoke only, not multi-year formal validation",
        "test_target_values_used_for_fitting_or_evaluation": False, "objective_counters": task.counters})
    print(json.dumps([{k: v for k, v in r.items() if k != "trace"} for r in results], ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
