"""Small real-data Taiwan baselines; provisional provenance never means formal readiness."""
import json
import sys
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.spatial import cKDTree, Delaunay
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.plugins import IDWPlugin
from ecodata.algorithms.dag import Artifact, Port, Node, NodeAdapter, NumericalDAGExecutor
from ecodata.core.schemas import EcoDataBundle
from ecodata.evaluation.quality_gates import RawInputQualityGate, ResultQualityGate

DATA = ROOT / "data/taiwan_2017_zenodo4420335"
OUT = ROOT / "output/taiwan_2017_preparation_20261019"
UNIT = "ug m-3"
TRANSFORM = Transformer.from_crs(4326, 3826, always_xy=True)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def xy(table):
    return np.column_stack(TRANSFORM.transform(table.longitude.to_numpy(), table.latitude.to_numpy()))


def field(table, values):
    ds = xr.Dataset({"pm25": ("sample", np.asarray(values, float))},
        coords={"longitude": ("sample", table.longitude.to_numpy()), "latitude": ("sample", table.latitude.to_numpy())},
        attrs={"crs": "EPSG:4326", "crs_evidence": "UNVERIFIED: original geographic coordinates used approximately as WGS84"})
    ds.pm25.attrs.update(units=UNIT, data_kind="continuous")
    return ds


def idw(train, query, values=None):
    # Existing frozen Registry implementation, not a newly implemented IDW kernel.
    table = train.copy()
    table["value"] = table.value if values is None else values
    lat, lon = np.unique(query.latitude), np.unique(query.longitude)
    template = xr.Dataset({"pm25": (("lat", "lon"), np.zeros((2, 2)))},
        coords={"lat": [21., 27.], "lon": [118., 123.]}, attrs={"crs": "EPSG:4326"})
    template.pm25.attrs["units"] = UNIT
    context = AlgorithmContext((EcoDataBundle(template),), "pm25", "continuous", target_lat=tuple(lat), target_lon=tuple(lon),
        observations=table.assign(split="train", unit=UNIT), parameters={"observation_crs": "EPSG:4326"})
    result = IDWPlugin().run(context).data.dataset.pm25
    return result.sel(lat=xr.DataArray(query.latitude.to_numpy(), dims="point"), lon=xr.DataArray(query.longitude.to_numpy(), dims="point")).values


def spatial(train, query, method):
    a, b = xy(train), xy(query)
    if method == "idw":
        return idw(train, query)
    kwargs = dict(variogram_model="linear", pseudo_inv=True, verbose=False)
    if method == "ok":
        model = OrdinaryKriging(a[:, 0], a[:, 1], train.value.to_numpy(), **kwargs)
        values, _ = model.execute("points", b[:, 0], b[:, 1])
    elif method == "uk":
        model = UniversalKriging(a[:, 0], a[:, 1], train.value.to_numpy(), drift_terms=["specified"], specified_drift=[train.aod.to_numpy()], **kwargs)
        values, _ = model.execute("points", b[:, 0], b[:, 1], specified_drift_arrays=[query.aod.to_numpy()])
    else:
        raise ValueError(method)
    return np.asarray(values, float)


def rf_fit(train):
    if not train.split.eq("train").all():
        raise ValueError("Nontraining station reached RF fit")
    model = RandomForestRegressor(n_estimators=50, min_samples_leaf=3, max_depth=6, random_state=11, n_jobs=1)
    x, y = train[["aod"]].to_numpy(), train.value.to_numpy()
    oof = cross_val_predict(model, x, y, groups=train.station_id, cv=GroupKFold(5), n_jobs=1)
    model.fit(x, y)
    return dict(model=model, residual=y-oof, fit_stations=sorted(train.station_id.tolist()), crossfit="5-fold station-disjoint")


def graph_executor():
    def source(inputs, params, external):
        artifact = external[params["name"]]
        if params["name"] == "train":
            table = artifact.value
            if not table.split.eq("train").all() or not table.unit.eq(UNIT).all():
                raise ValueError("Training role/unit contract failed")
            if (table.value < 0).any() or not np.isfinite(table[["longitude", "latitude", "value"]]).all().all():
                raise ValueError("Invalid raw training values")
        return {"table": artifact}

    def fit(inputs, params, external):
        return {"state": Artifact("rf_state", rf_fit(inputs["train"].value))}

    def predict(inputs, params, external):
        train, query = inputs["train"].value, inputs["query"].value
        if not train.split.eq("train").all():
            raise ValueError("Only training stations may enter prediction fit")
        if train.time.nunique() != 1 or query.time.nunique() != 1 or train.time.iloc[0] != query.time.iloc[0]:
            raise ValueError("Mismatched observation time windows")
        if train.duplicated(["station_id", "time"]).any():
            raise ValueError("Duplicate fitting observations")
        if "value" in query:
            raise ValueError("Prediction query must not contain validation/test labels")
        if set(train.station_id) & set(query.station_id):
            raise ValueError("Station leakage")
        method = params["method"]
        if method in {"idw", "ok", "uk"}:
            pred = spatial(train, query, method)
        elif method == "linear":
            pred = LinearRegression().fit(train[["aod"]], train.value).predict(query[["aod"]])
        elif method == "gwr":
            from mgwr.gwr import GWR
            model = GWR(xy(train), train.value.to_numpy()[:, None], train[["aod"]].to_numpy(),
                bw=min(25, len(train)-1), kernel="bisquare", fixed=False, n_jobs=1)
            fitted = model.fit()
            pred = model.predict(xy(query), query[["aod"]].to_numpy(), exog_scale=fitted.scale, exog_resid=fitted.resid_response).predictions.ravel()
        elif method == "rf":
            state = inputs["state"].value
            pred = state["model"].predict(query[["aod"]].to_numpy()) + idw(train, query, state["residual"])
        else:
            raise ValueError(method)
        return {"prediction": Artifact("prediction", np.asarray(pred))}

    def quality(inputs, params, external):
        train, query = inputs["train"].value, inputs["query"].value
        raw_values = inputs["prediction"].value
        if not np.isfinite(raw_values).all() or len(raw_values) != len(query):
            raise ValueError("Incomplete prediction coverage")
        raw, output = field(train, train.value), field(query, np.maximum(raw_values, 0))
        raw_quality = RawInputQualityGate().validate(raw, variable="pm25", intended_operation="spatial", expected_unit=UNIT)
        result_quality = ResultQualityGate().validate(raw, output, variable="pm25", nonnegative=True)
        if "BLOCK" in {raw_quality["status"], result_quality["status"]}:
            raise ValueError("Quality gate BLOCK")
        result_quality.update(status="WARN", provenance_status="Archive checksum, exact time and coordinate provenance pending", projection_count=int((raw_values < 0).sum()))
        return {"prediction": Artifact("prediction", np.maximum(raw_values, 0)),
            "raw_prediction": Artifact("raw_prediction", raw_values),
            "quality": Artifact("quality", dict(raw=raw_quality, result=result_quality))}

    adapters = {
        ("source_train", "run"): NodeAdapter({}, {"table": Port("training")}, source),
        ("source_query", "run"): NodeAdapter({}, {"table": Port("query")}, source),
        ("rf_fit", "run"): NodeAdapter({"train": Port("training")}, {"state": Port("rf_state")}, fit),
        ("quality", "run"): NodeAdapter({"train": Port("training"), "query": Port("query"), "prediction": Port("prediction")},
            {"prediction": Port("prediction"), "raw_prediction": Port("raw_prediction"), "quality": Port("quality")}, quality),
    }
    for method in ["idw", "ok", "uk", "linear", "gwr", "rf"]:
        ports = {"train": Port("training"), "query": Port("query")}
        if method == "rf":
            ports["state"] = Port("rf_state")
        adapters[(method, "run")] = NodeAdapter(ports, {"prediction": Port("prediction")}, predict)
    return NumericalDAGExecutor(adapters, version="taiwan-smoke-1")


def execute(method, train, query):
    edges = {"train": ("training", "table"), "query": ("query", "table")}
    nodes = [Node("training", "source_train", parameters={"name": "train"}), Node("query", "source_query", parameters={"name": "query"})]
    if method == "rf":
        nodes.append(Node("fit", "rf_fit", {"train": ("training", "table")}))
        edges["state"] = ("fit", "state")
    nodes.append(Node("prediction", method, edges, {"method": method}))
    nodes.append(Node("quality", "quality", {"train": ("training", "table"), "query": ("query", "table"), "prediction": ("prediction", "prediction")}))
    # Existing bridge constructs genuine official OptGraph parent edges/port bindings.
    from run_three_source_ready_smoke import golem_graph
    result = graph_executor().run_golem(golem_graph(nodes), external={"train": Artifact("training", train), "query": Artifact("query", query)})
    return result


def validation_metrics(prediction, validation):
    if not validation.split.eq("validation").all():
        raise ValueError("Numerical Objective accepts validation labels only")
    if len(prediction) != len(validation) or not np.isfinite(prediction).all():
        raise ValueError("Objective requires complete predictions on frozen targets")
    error = np.asarray(prediction) - validation.value.to_numpy()
    return dict(RMSE=float(np.sqrt(np.mean(error**2))), MAE=float(np.mean(abs(error))), Bias=float(error.mean()), R2=float(r2_score(validation.value, prediction)))


def satellite_samples(files, sites):
    rows, qa = [], []
    coords = xy(sites)
    for file in files:
        stamp = pd.to_datetime(file.stem.replace("results_Taiwan_", ""), format="%Y%m%d_%H%M")
        with xr.open_dataset(file) as ds:
            lon, lat, aod, quality = [ds[v].values.ravel() for v in ["Longitude", "Latitude", "AOD550", "QCAll"]]
            valid = np.isfinite(lon) & np.isfinite(lat) & (abs(lon) <= 180) & (abs(lat) <= 90) & np.isfinite(aod) & np.isin(quality, [0, 1]) & (aod >= -.05) & (aod <= 5)
            a, b = TRANSFORM.transform(lon[valid], lat[valid])
            tree = cKDTree(np.column_stack([a, b]))
            neighborhoods = tree.query_ball_point(coords, 27500)
            qa.append(dict(file=str(file.relative_to(ROOT)), time_utc_assumption=str(stamp), pixels=len(aod), qa_valid=int(valid.sum()), fraction=float(valid.mean())))
            values = aod[valid]
            for site, neighbors in zip(sites.station_id, neighborhoods):
                if neighbors:
                    rows.append(dict(station_id=site, utc_hour=stamp.floor("h"), aod=float(values[neighbors].mean()), valid_pixels=len(neighbors)))
    pd.DataFrame(qa).to_csv(OUT / "satellite_qa.csv", index=False)
    if not rows:
        raise ValueError("No actual QA-passing station AOD")
    return pd.DataFrame(rows).groupby(["station_id", "utc_hour"], as_index=False).agg(aod=("aod", "mean"), scans=("aod", "size"))


def evaluate_case(name, time, table, methods):
    train = table.loc[table.split.eq("train")].dropna(subset=["value"])
    validation = table.loc[table.split.eq("validation")].dropna(subset=["value"])
    if len(train) < 15 or len(validation) < 3:
        return [], [dict(experiment=name, time=str(time), status="INSUFFICIENT_INPUT", train=len(train), validation=len(validation))]
    inside = Delaunay(xy(train)).find_simplex(xy(validation)) >= 0
    validation = validation.loc[inside].copy()
    if len(validation) < 3:
        return [], [dict(experiment=name, time=str(time), status="INSUFFICIENT_IN_HULL_VALIDATION")]
    query = validation.drop(columns=["value", "split"])
    raw = field(train, train.value)
    raw_quality = RawInputQualityGate().validate(raw, variable="pm25", intended_operation="spatial", expected_unit=UNIT)
    predictions, runs = [], []
    for method in methods:
        start = perf_counter()
        folder = OUT / name / str(time).replace(":", "").replace(" ", "_") / method
        try:
            if raw_quality["status"] == "BLOCK":
                raise ValueError("Raw quality gate BLOCK")
            result = execute(method, train, query)
            unprojected = result["outputs"]["raw_prediction"].value
            if not np.isfinite(unprojected).all() or len(unprojected) != len(validation):
                raise ValueError("Incomplete prediction coverage")
            pred = result["outputs"]["prediction"].value
            ds = field(query, pred)
            quality = result["outputs"]["quality"].value["result"]
            if quality["status"] == "BLOCK":
                raise ValueError(str(quality))
            folder.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(folder / "predictions.nc", engine="h5netcdf")
            with xr.open_dataset(folder / "predictions.nc") as reopened:
                np.testing.assert_allclose(reopened.pm25.values, pred)
            save(folder / "execution_trace.json", result["trace"])
            save(folder / "quality.json", dict(raw=raw_quality, result=quality))
            save(folder / "objective.json", dict(metrics=validation_metrics(pred, validation), executed_golem_graph=True,
                quality_status="WARN", seconds=perf_counter()-start, formal_eligible=False, failure_reason=None))
            for (_, row), value, before in zip(validation.iterrows(), pred, unprojected):
                predictions.append(dict(experiment=name, method=method, time=str(time), station_id=row.station_id, observed=row.value, predicted=float(value), before_projection=float(before)))
            runs.append(dict(experiment=name, method=method, time=str(time), status="EXECUTED_PROVISIONAL", expected=len(validation), predicted=len(pred), seconds=perf_counter()-start, projection_count=int((unprojected < 0).sum())))
        except Exception as error:
            save(folder / "failure.json", dict(error=repr(error), trace=getattr(error, "trace", [])))
            runs.append(dict(experiment=name, method=method, time=str(time), status="FAIL", expected=len(validation), predicted=0, seconds=perf_counter()-start, error=repr(error)))
    return predictions, runs


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output
    if (OUT / "executions.csv").exists():
        raise ValueError("Output already exists; use a fresh --output to preserve evidence")
    OUT.mkdir(parents=True, exist_ok=True)
    base = DATA / "provisional_members/zhang_data"
    sites = pd.read_csv(base / "pm25_taiwan/StationInfo.csv").rename(columns={"SiteEngName": "station_id", "Lon": "longitude", "Lat": "latitude"})
    # Freeze before inspecting target values; no site is selected by model performance.
    ordered = sites.sort_values(["latitude", "longitude"]).reset_index(drop=True)
    order = np.random.default_rng(11).permutation(len(ordered))
    roles = {i: "train" if rank < 50 else "validation" if rank < 63 else "test" for rank, i in enumerate(order)}
    ordered["split"] = [roles[i] for i in range(len(ordered))]
    ordered.to_csv(OUT / "station_split.csv", index=False)
    plan = dict(seed=11, training_stations=50, validation_stations=13, sealed_test_stations=14,
        evaluation="same-hour input-station observations predict held-out validation stations; no future forecast claim",
        nonnegative_projection="same explicit max(0,prediction) for all methods, raw predictions retained",
        coordinate_status="UNVERIFIED original geographic coordinates approximately WGS84; projected EPSG:3826 only exploratory",
        time_status="UNVERIFIED file timestamp convention: exploratory assumption ground UTC+8, AHI filenames UTC; not formal data",
        aod="QCAll 0 or 1; valid range [-0.05,5]; spatial mean radius 27.5km; temporal mean by hour; actual available scans only",
        gwr="official mgwr 2.2.1, adaptive bisquare 25 neighbors, fixed a priori, not exact paper reproduction",
        rf="50 trees, depth6, minleaf3, seed11; 5-fold held-station OOF residual IDW; no validation fit",
        archive_verified=json.loads((DATA / "download_verification.json").read_text()).get("verified", False) if (DATA / "download_verification.json").exists() else False)
    save(OUT / "analysis_plan.json", plan)
    ground = pd.read_csv(base / "pm25_taiwan/Taiwan_PM_2017.csv", usecols=["stn_eng", "DT", "PM2.5"]).rename(columns={"stn_eng": "station_id", "DT": "time", "PM2.5": "value"})
    audit = dict(rows=len(ground), sites=ground.station_id.nunique(), missing=int(ground.value.isna().sum()), duplicates=int(ground.duplicated(["station_id", "time"]).sum()), negative=int((ground.value < 0).sum()), first=ground.time.min(), last=ground.time.max())
    save(OUT / "ground_audit.json", audit)
    if audit["duplicates"]:
        raise ValueError("Unresolved ground duplicates")
    # Test labels never enter the fitting, covariate match or objective tables.
    ground = ground.loc[ground.station_id.isin(ordered.loc[~ordered.split.eq("test"), "station_id"])].merge(ordered, on="station_id", validate="many_to_one")
    ground["time"] = pd.to_datetime(ground.time)
    ground = ground.loc[ground.value.ge(0)].copy()
    ground["unit"] = UNIT
    ground["utc_hour"] = ground.time - pd.Timedelta(hours=8)
    predictions, executions = [], []
    # Fixed season dates, not selected after examining prediction errors.
    for stamp in pd.to_datetime(["2017-01-15 12:00", "2017-04-15 12:00", "2017-07-15 12:00", "2017-10-15 12:00"]):
        p, r = evaluate_case("experiment2", stamp, ground.loc[ground.time.eq(stamp)], ["idw", "ok"])
        predictions.extend(p); executions.extend(r)
    files = sorted((base / "ahi_aod_taiwan").rglob("*.nc"))
    if files:
        matches = satellite_samples(files, ordered)
        matched = ground.merge(matches, on=["station_id", "utc_hour"], validate="one_to_one")
        matched.to_csv(OUT / "matched_train_validation_only.csv", index=False)
        for stamp in sorted(matched.utc_hour.unique())[:6]:
            p, r = evaluate_case("experiment3", stamp, matched.loc[matched.utc_hour.eq(stamp)], ["linear", "gwr", "uk", "rf"])
            predictions.extend(p); executions.extend(r)
    p, r = pd.DataFrame(predictions), pd.DataFrame(executions)
    p.to_csv(OUT / "predictions.csv", index=False)
    r.to_csv(OUT / "executions.csv", index=False)
    summaries = []
    for experiment, group in p.groupby("experiment"):
        methods = ["idw", "ok"] if experiment == "experiment2" else ["linear", "gwr", "uk", "rf"]
        sets = [set(zip(part.time, part.station_id)) for method in methods for part in [group.loc[group.method.eq(method)]]]
        common = set.intersection(*sets)
        for method in methods:
            g = group.loc[group.method.eq(method)].copy()
            g = g.loc[[(t, s) in common for t, s in zip(g.time, g.station_id)]]
            run = r.loc[r.experiment.eq(experiment) & r.method.eq(method)]
            if len(g):
                err = g.predicted-g.observed
                summaries.append(dict(experiment=experiment, method=method, common_samples=len(g), stations=g.station_id.nunique(), times=g.time.nunique(), RMSE=float(np.sqrt(np.mean(err**2))), MAE=float(np.mean(abs(err))), Bias=float(err.mean()), R2=float(r2_score(g.observed,g.predicted)), coverage=float(run.predicted.sum()/run.expected.sum()), failures=int(run.status.eq("FAIL").sum()), seconds=float(run.seconds.sum())))
    pd.DataFrame(summaries).to_csv(OUT / "comparison.csv", index=False)
    save(OUT / "readiness.json", dict(experiment2="REAL_EXECUTION_PROVISIONAL", experiment3="SEE_EXECUTIONS_NOT_FORMALLY_READY", formal_abc_ready=False, reason="Full archive checksum and timestamp/coordinate provenance pending; 4 configurations enumerable; no GOLEM advantage claim"))
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
