"""Audit preserved actual predictions and source evidence without retraining."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.spatial import Delaunay

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/taiwan_2017_zenodo4420335"
OUT = ROOT / "output/taiwan_2017_preparation_20261019"
RUN = OUT / "golem_dag_smoke"


def main():
    source = DATA / "provisional_members/zhang_data/pm25_taiwan"
    raw = pd.read_csv(source / "Taiwan_PM_2017.csv")
    raw["DT"] = pd.to_datetime(raw.DT)
    coverage = raw.groupby("stn_eng").agg(records=("DT", "size"), valid_pm25=("PM2.5", "count"), first=("DT", "min"), last=("DT", "max"), distinct_hours=("DT", "nunique"))
    coverage["missing_rows_against_8760"] = 8760-coverage.distinct_hours
    coverage["missing_pm25"] = coverage.records-coverage.valid_pm25
    coverage.to_csv(OUT / "station_annual_coverage.csv")
    predictions = pd.read_csv(RUN / "predictions.csv")
    executions = pd.read_csv(RUN / "executions.csv")
    split = pd.read_csv(RUN / "station_split.csv")
    assert not set(predictions.station_id) & set(split.loc[split.split.eq("test"), "station_id"])
    validation_sites = set(split.loc[split.split.eq("validation"), "station_id"])
    matched = pd.read_csv(RUN / "matched_train_validation_only.csv")
    denominator_rows = []
    for experiment in ["experiment2", "experiment3"]:
        stamps = pd.to_datetime(executions.loc[executions.experiment.eq(experiment), "time"].unique())
        local_stamps = stamps if experiment == "experiment2" else stamps + pd.Timedelta(hours=8)
        available = raw.loc[raw.stn_eng.isin(validation_sites) & raw.DT.isin(local_stamps) & raw["PM2.5"].ge(0)]
        successful = predictions.loc[predictions.experiment.eq(experiment)]
        qualified = successful[["time", "station_id"]].drop_duplicates()
        aod_available = len(matched.loc[matched.station_id.isin(validation_sites) & pd.to_datetime(matched.utc_hour).isin(stamps)]) if experiment == "experiment3" else None
        denominator_rows.append(dict(experiment=experiment, possible_validation_site_hours=len(stamps)*len(validation_sites), actual_valid_ground=len(available), aod_matched_validation=aod_available,
            in_hull_qualified=len(qualified), common_successes=len(qualified), coverage_of_valid_ground=len(qualified)/len(available) if len(available) else None))
    pd.DataFrame(denominator_rows).to_csv(RUN / "coverage_denominators.csv", index=False)
    qualification = []
    station_metadata = split.set_index("station_id")
    projector = Transformer.from_crs(4326, 3826, always_xy=True)
    for experiment in ["experiment2", "experiment3"]:
        stamps = sorted(executions.loc[executions.experiment.eq(experiment), "time"].unique())
        for stamp in stamps:
            time = pd.Timestamp(stamp)
            if experiment == "experiment2":
                selected = raw.loc[raw.DT.eq(time) & raw["PM2.5"].ge(0)].rename(columns={"stn_eng": "station_id"})
                selected = selected.merge(split, on="station_id")
            else:
                selected = matched.loc[pd.to_datetime(matched.utc_hour).eq(time)]
            train = selected.loc[selected.split.eq("train")]
            hull = Delaunay(np.column_stack(projector.transform(train.longitude, train.latitude))) if len(train) >= 3 else None
            available_ids = set(selected.station_id)
            for station in sorted(validation_sites):
                row = station_metadata.loc[station]
                inside = bool(hull.find_simplex(np.column_stack(projector.transform([row.longitude], [row.latitude])))[0] >= 0) if hull is not None and station in available_ids else False
                reason = "QUALIFIED" if inside else "OUTSIDE_TRAINING_HULL" if station in available_ids else "MISSING_VALID_GROUND_OR_AOD"
                qualification.append(dict(experiment=experiment, time=stamp, station_id=station, train_stations=len(train), reason=reason, qualified=inside))
    pd.DataFrame(qualification).to_csv(RUN / "sample_qualification.csv", index=False)
    summaries = []
    for (experiment, method, site), g in predictions.groupby(["experiment", "method", "station_id"]):
        e = g.predicted-g.observed
        summaries.append(dict(experiment=experiment, method=method, station_id=site, n=len(g), RMSE=float(np.sqrt(np.mean(e**2))), MAE=float(np.mean(abs(e))), Bias=float(e.mean())))
    pd.DataFrame(summaries).to_csv(RUN / "station_metrics.csv", index=False)
    cohort = []
    for experiment, methods in [("experiment2", ["idw", "ok"]), ("experiment3", ["linear", "gwr", "uk", "rf"])]:
        g = predictions.loc[predictions.experiment.eq(experiment)]
        sets = [set(zip(g.loc[g.method.eq(m), "time"], g.loc[g.method.eq(m), "station_id"])) for m in methods]
        assert all(s == sets[0] for s in sets)
        cohort.append(dict(experiment=experiment, common_samples=len(sets[0]), methods=methods, identical_keys=True))
    readbacks = []
    for nc in RUN.rglob("predictions.nc"):
        with xr.open_dataset(nc) as ds:
            assert np.isfinite(ds.pm25).all() and (ds.pm25 >= 0).all()
            assert ds.pm25.attrs["units"] == "ug m-3"
            readbacks.append(dict(file=str(nc.relative_to(ROOT)), sha256=hashlib.sha256(nc.read_bytes()).hexdigest(), valid=True))
    checksums = []
    for file in [source / "Taiwan_PM_2017.csv", source / "StationInfo.csv", DATA / "official_record.json", ROOT / "output/capability_final_acceptance/frozen_generic_pool.json"]:
        checksums.append(dict(file=str(file.relative_to(ROOT)), bytes=file.stat().st_size, sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    report = dict(cohort=cohort, coverage_denominators=denominator_rows, readbacks=readbacks, source_and_frozen_pool=checksums,
        executed=int(executions.status.eq("EXECUTED_PROVISIONAL").sum()), failed=int(executions.status.eq("FAIL").sum()),
        test_labels_used_for_fit_or_selection=False, formal_ready=False,
        blockers=["Official full-archive MD5 not verified", "Ground DT timezone/hour-window and AHI filename time convention need direct evidence", "Coordinate datum not specified in archive station table"],
        environment={p:version(p) for p in ["numpy", "pandas", "xarray", "pykrige", "scikit-learn", "mgwr"]})
    report["code_sha256"] = {str(file.relative_to(ROOT)):hashlib.sha256(file.read_bytes()).hexdigest()
        for file in [ROOT / "tools/run_taiwan_preparation.py", ROOT / "tools/audit_taiwan_outputs.py", ROOT / "src/ecodata/algorithms/dag.py", ROOT / "tools/run_three_source_ready_smoke.py"]}
    (OUT / "actual_output_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(cohort=cohort, executed=report["executed"], failed=report["failed"], readbacks=len(readbacks))))


if __name__ == "__main__":
    main()
