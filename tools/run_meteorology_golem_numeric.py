"""Real numerical GOLEM experiment adapter. No third-party optimiser edits.

Unspecified workbook units/datums stay unspecified. The experiment operates on
training-standardised values and an explicitly non-geodetic coordinate plane.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / ".venv-golem/Lib/site-packages"))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import Delaunay
from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.registry import create_default_registry
from ecodata.core.schemas import EcoDataBundle
from ecodata.evaluation.quality_gates import RawInputQualityGate, ResultQualityGate
from golem.core.adapter import IdentityAdapter
from golem.core.dag.verification_rules import has_no_cycle, has_no_self_cycled_nodes
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators import base_mutations
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.objective import Objective, ObjectiveEvaluate
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams

PLANE = 'LOCAL_CS["Training-standardised source coordinate plane - NOT geodetic",LOCAL_DATUM["Unknown source datum",0],UNIT["dimensionless coordinate standard deviation",1],AXIS["X",EAST],AXIS["Y",NORTH]]'
STATIONS = ("station.metpy_barnes", "station.metpy_cressman", "station.metpy_linear",
            "station.metpy_natural_neighbor", "station.pykrige_ordinary", "station.pykrige_universal")
REGRIDS = ("spatial.nearest", "spatial.bilinear")
SEEDS = (11, 22, 33, 44, 55)
DATES = ("01-15", "02-15", "04-15", "05-15", "07-15", "08-15", "10-15", "11-15")


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")


def chain(graph):
    if not graph.nodes or len(graph.nodes) > 2:
        raise ValueError("Only a station estimator and optional genuine coarse-to-target resampling are meaningful")
    roots = [n for n in graph.nodes if not n.nodes_from]
    if len(roots) != 1 or roots[0].name not in STATIONS:
        raise ValueError("Exactly one station-input estimator required")
    ordered = [roots[0]]
    if len(graph.nodes) == 2:
        child = next(n for n in graph.nodes if n is not roots[0])
        if child.name not in REGRIDS or child.nodes_from != [roots[0]]:
            raise ValueError("Grid resampling must follow the station estimator; branching/repeated fits are incompatible")
        ordered.append(child)
    return tuple(n.name for n in ordered)


def graph_of(plan):
    node = OptNode(plan[0])
    if len(plan) == 2:
        node = OptNode(plan[1], nodes_from=[node])
    return OptGraph([node])


def graph_record(graph):
    nodes = list(graph.nodes)
    return [{"algorithm": n.name, "parents": [nodes.index(p) for p in n.nodes_from]} for n in nodes]


def prepare(out):
    if (out / "experiment_config.json").exists():
        raise FileExistsError("Frozen experiment configuration already exists; use a new output directory")
    data = out / "data"
    table = pd.read_parquet(data / "regional_six_years.parquet")
    table["time"] = pd.to_datetime(table.time)
    if table.duplicated(["station", "time"]).any():
        raise ValueError("Duplicate station/time rows require explicit resolution")
    history = table[table.time.dt.year <= 2020]
    positions = history.groupby("station")[["x_as_stored", "y_as_stored"]].median()
    stations = []
    for sid, row in positions.iterrows():
        ix = min(3, int((row.x_as_stored - 114) / 1.25))
        iy = min(3, int((row.y_as_stored - 36) / 1.25))
        stations.append((ix, iy, hashlib.sha256(("meteo_split_20261018:" + sid).encode()).hexdigest(), sid))
    split = {}
    for cell in sorted({row[:2] for row in stations}):
        items = sorted([row for row in stations if row[:2] == cell])
        for i, row in enumerate(items[:6]):
            split[row[3]] = "validation" if i == 4 else ("test" if i == 5 else "train")
    coordinates = positions.loc[[sid for sid, role in split.items() if role == "train"]]
    center, scale = coordinates.mean().values, coordinates.std(ddof=0).values
    table = table[table.station.isin(split)].copy()
    table["split"] = table.station.map(split)
    table["x"] = (table.x_as_stored - center[0]) / scale[0]
    table["y"] = (table.y_as_stored - center[1]) / scale[1]
    canonical = (coordinates.values - center) / scale
    rect = np.array([np.quantile(canonical[:, 0], [.3, .7]), np.quantile(canonical[:, 1], [.3, .7])])
    # Fix an inscribed target rectangle using training locations, never target values.
    hull = Delaunay(canonical)
    for _ in range(12):
        xx, yy = np.meshgrid(rect[0], rect[1])
        if (hull.find_simplex(np.c_[xx.ravel(), yy.ravel()]) >= 0).all():
            break
        rect *= .9
    panel = table[table.time.dt.year.isin([2021, 2022]) & table.time.dt.strftime("%m-%d").isin(DATES)].copy()
    # Evaluation stations inside the target rectangle; exclusion is geometry-only.
    eligible = panel.groupby("station").apply(lambda f: bool(f.x.between(*rect[0]).all() and f.y.between(*rect[1]).all()), include_groups=False)
    roles = {sid: role for sid, role in split.items() if role == "train" or eligible.get(sid, False)}
    panel = panel[panel.station.isin(roles)]
    if not all(sum(v == role for v in roles.values()) >= 2 for role in ("validation", "test")):
        raise ValueError(f"Insufficient geometry-only evaluation stations: {Counter(roles.values())}")
    norms = {}
    train = table[(table.time.dt.year <= 2020) & table.split.eq("train")]
    for variable in ("rh", "pressure"):
        norms[variable] = {"mean": float(train[variable].mean()), "std": float(train[variable].std(ddof=0)),
                           "fit_rows": int(train[variable].count()), "fit_years": sorted(train.time.dt.year.unique().tolist()),
                           "fit_station_ids": sorted(train.station.unique().tolist())}
    panel[(panel.time.dt.year == 2021) & panel.split.ne("test")].to_parquet(data / "validation.parquet", index=False)
    panel[(panel.time.dt.year == 2022) & panel.split.ne("validation")].to_parquet(data / "sealed_test.parquet", index=False)
    pd.DataFrame([{"station": sid, "role": role, "x_source_median": positions.loc[sid].x_as_stored,
                   "y_source_median": positions.loc[sid].y_as_stored} for sid, role in sorted(roles.items())]).to_csv(data / "station_split.csv", index=False)
    changes = table.groupby(["station", table.time.dt.year])[["x_as_stored", "y_as_stored"]].agg(["min", "max"])
    changes.to_csv(data / "coordinate_versions.csv")
    config = {"scope": "Real-data numerical interpolation benchmark, not verified absolute-unit/geodetic accuracy",
        "variables": {"rh": {"source_header": "相对湿度24小时均值", "source_unit": "UNCONFIRMED", "percent_bounds_used": False},
                      "pressure": {"source_header": "海平面气压24小时均值", "source_unit": "UNCONFIRMED", "uniform_pressure_bounds_used": False}},
        "time": "daily source labels; timezone and exact 24h cut-off unconfirmed; never converted to UTC or made hourly",
        "excluded_2018": "Coordinates stored as 12231/5258-like numbers, not decimal degrees; no undocumented DMS conversion. Sea-level pressure absent.",
        "coordinate_mode": "Training-standardised raw numerical coordinate plane; no WGS84 claim or metre distances",
        "coordinate_center": center.tolist(), "coordinate_scale": scale.tolist(), "crs": PLANE,
        "target_rect": rect.tolist(), "target_size": 17, "intermediate_size": 9,
        "station_counts": dict(Counter(roles.values())), "normalisation": norms,
        "validation_dates": [f"2021-{day}" for day in DATES], "test_dates": [f"2022-{day}" for day in DATES],
        "seeds": list(SEEDS), "pilot_budget": 20, "main_budget": 40,
        "objective": "validation_RMSE / training_std + 0.01 * declared_cost + 0.002 * extra_grid_step; invalid=1000000",
        "good_error_threshold": "RMSE/training_std <= 0.75, fixed before outcomes",
        "cost": {"low": .25, "medium": 1., "high": 4., "gpu": 8.},
        "cache": "fresh graph cache per run; identical graphs reuse that run's actual results; no cross-mode numerical cache",
        "task_graphs": "6 station estimators, each direct 17x17 or 9x9 followed by nearest/bilinear to 17x17 = 18 legal graphs",
        "claim_limit": "Physical semantic effect limited to input/output dependencies, no extrapolation, finite values and metadata; not humidity-percent/pressure-range validation"}
    dump(out / "experiment_config.json", config)
    print("PREPARED", config["station_counts"], flush=True)


class Evaluator:
    def __init__(self, out, variable, directory, phase="validation", pilot=False):
        self.out, self.variable, self.directory = Path(out), variable, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = json.loads((self.out / "experiment_config.json").read_text(encoding="utf-8"))
        self.table = pd.read_parquet(self.out / "data" / ("validation.parquet" if phase == "validation" else "sealed_test.parquet"))
        if pilot:
            dates = sorted(self.table.time.unique())
            self.table = self.table[self.table.time.isin([dates[0], dates[4]])]
        self.phase = phase
        self.registry = create_default_registry()
        self.norm = self.config["normalisation"][variable]
        self.cache, self.executions, self.records = {}, 0, []
        self.metadata_cost = {s.id: self.config["cost"][s.compute] for s in self.registry.specs()}

    def execute(self, plan):
        started = time.perf_counter()
        predictions, datasets, quality_records = [], [], []
        xmin, xmax = self.config["target_rect"][0]
        ymin, ymax = self.config["target_rect"][1]
        fine_x, fine_y = np.linspace(xmin, xmax, 17), np.linspace(ymin, ymax, 17)
        role = "validation" if self.phase == "validation" else "test"
        for date, frame in self.table.groupby("time", sort=True):
            obs = frame[frame.split.eq("train")].dropna(subset=[self.variable]).copy()
            target = frame[frame.split.eq(role)].copy()
            if len(obs) < 5 or target.empty or target[self.variable].isna().any():
                raise ValueError("Insufficient valid input/evaluation observations; no imputation")
            if (Delaunay(obs[["x", "y"]]).find_simplex(target[["x", "y"]]) < 0).any():
                raise ValueError("Unsupported extrapolation outside contemporary input-station hull")
            grid_n = 9 if len(plan) == 2 else 17
            xx, yy = np.linspace(xmin, xmax, grid_n), np.linspace(ymin, ymax, grid_n)
            ds = xr.Dataset({self.variable: (("lat", "lon"), np.zeros((2, 2)))},
                coords={"lat": [ymin, ymax], "lon": [xmin, xmax]},
                attrs={"crs": PLANE, "data_kind": "continuous", "source": "destination template only; not a fabricated observation field"})
            ds[self.variable].attrs.update(units="1", data_kind="continuous", spatial_semantics="point_estimate")
            observations = pd.DataFrame({"longitude": obs.x, "latitude": obs.y,
                "value": (obs[self.variable] - self.norm["mean"]) / self.norm["std"], "split": "train", "unit": "1"})
            raw = xr.Dataset({self.variable: ("observation", observations.value.to_numpy())},
                coords={"x": ("observation", obs.x.to_numpy()), "y": ("observation", obs.y.to_numpy())}, attrs={"crs": PLANE})
            raw[self.variable].attrs.update(units="1", data_kind="continuous")
            input_quality = RawInputQualityGate().validate(raw, variable=self.variable, intended_operation="reconstruction", requires_spatial=True)
            if input_quality["status"] == "BLOCK":
                raise ValueError("Raw quality gate rejected numerical inputs")
            context = AlgorithmContext((EcoDataBundle(ds),), self.variable, "continuous",
                tuple(yy), tuple(xx), observations=observations,
                parameters={"observation_crs": PLANE, "variogram_model": "linear", "minimum_neighbors": 3})
            plugin = self.registry.get(plan[0])
            result = plugin.run(context)
            if len(plan) == 2:
                context = AlgorithmContext((result.data,), self.variable, "continuous", tuple(fine_y), tuple(fine_x))
                result = self.registry.get(plan[1]).run(context)
            output = result.data.dataset
            quality = ResultQualityGate().validate(ds, output, variable=self.variable, expected_output_unit="1")
            if quality["status"] == "BLOCK" or not np.isfinite(output[self.variable]).all():
                raise ValueError("Result gate or finite-domain requirement rejected the output")
            qx, qy = xr.DataArray(target.x.to_numpy(), dims="site"), xr.DataArray(target.y.to_numpy(), dims="site")
            prediction = output[self.variable].interp(lon=qx, lat=qy).values * self.norm["std"] + self.norm["mean"]
            if not np.isfinite(prediction).all():
                raise ValueError("Nonfinite held-out predictions")
            for sid, truth, estimate in zip(target.station, target[self.variable], prediction):
                predictions.append({"station": sid, "time_label": str(date), "role": role,
                                    "observed_source_value": float(truth), "prediction_source_value": float(estimate)})
            output = output.rename({"lat": "y", "lon": "x"})
            output[self.variable] = output[self.variable] * self.norm["std"] + self.norm["mean"]
            output[self.variable].attrs.update(units="unknown", source_unit_status="UNCONFIRMED",
                long_name=self.config["variables"][self.variable]["source_header"])
            output.attrs.update(crs=PLANE, coordinate_warning="Numerical standardised coordinate plane, not geodetic or metres",
                                source_time_semantics="daily source label, timezone/cutoff unconfirmed")
            datasets.append(output.expand_dims(time_label=[date]))
            quality_records.append({"time": str(date), "raw": input_quality, "output": quality})
        frame = pd.DataFrame(predictions)
        errors = frame.prediction_source_value - frame.observed_source_value
        result = {"rmse": float(np.sqrt(np.mean(errors ** 2))), "mae": float(np.mean(abs(errors))),
                  "bias": float(errors.mean()), "n": len(frame), "quality_status": "PASS_NUMERICAL_ONLY",
                  "physical_absolute_units_verified": False, "graph": list(plan), "physical_violations": 0,
                  "seconds": time.perf_counter() - started, "test_used_for_fit": False}
        key = hashlib.sha256("|".join(plan).encode()).hexdigest()[:12]
        destination = self.directory / f"{key}.nc"
        combined = xr.concat(datasets, dim="time_label")
        combined.to_netcdf(destination, engine="h5netcdf")
        with xr.open_dataset(destination, engine="h5netcdf") as back:
            xr.testing.assert_allclose(back, combined)
        frame.to_csv(self.directory / f"{key}_predictions.csv", index=False)
        dump(self.directory / f"{key}_quality.json", quality_records)
        result["netcdf"] = str(destination)
        result["score"] = result["rmse"] / self.norm["std"] + .01 * sum(self.metadata_cost[a] for a in plan) + .002 * (len(plan) - 1)
        return result

    def evaluate(self, graph):
        try:
            plan = chain(graph)
            key = "|".join(plan)
        except ValueError as exc:
            return {"valid": False, "score": 1e6, "reason": str(exc), "cache_hit": False}
        if key in self.cache:
            return {**self.cache[key], "cache_hit": True}
        self.executions += 1
        try:
            result = {"valid": True, **self.execute(plan)}
        except Exception as exc:
            result = {"valid": False, "score": 1e6, "reason": f"{type(exc).__name__}: {exc}", "graph": list(plan)}
        self.cache[key] = result
        return {**result, "cache_hit": False}


def search(out, variable, mode, seed, budget, pilot=False, ablation="full"):
    random.seed(seed)
    np.random.seed(seed)
    label = f"{variable}_{mode}_{seed}_{ablation}" + ("_pilot" if pilot else "")
    evaluator = Evaluator(out, variable, Path(out) / "runs" / label, pilot=pilot)
    frozen = json.loads((ROOT / "output/capability_final_acceptance/frozen_generic_pool.json").read_text(encoding="utf-8"))
    available = [a["id"] for a in frozen["algorithms"]]
    assert set(STATIONS + REGRIDS).issubset(available)
    counts, trace, proposals = Counter(), [], []
    beginning = time.perf_counter()

    def legal(graph, *args, **kwargs):
        counts["posterior_checks"] += 1
        try:
            chain(graph)
            return True
        except ValueError:
            counts["posterior_rejections"] += 1
            return False

    def objective(graph):
        counts["objective_calls"] += 1
        if len(trace) >= budget:
            counts["budget_deferred_calls"] += 1
            return 1e6
        result = evaluator.evaluate(graph)
        trace.append({"evaluation": len(trace) + 1, "elapsed": time.perf_counter() - beginning,
                      "structure": graph_record(graph), **result})
        return result["score"]

    def counted(fn):
        def mutate(graph, **kwargs):
            counts["mutation_proposals"] += 1
            result = fn(graph, kwargs["requirements"], kwargs["graph_gen_params"], kwargs["parameters"])
            try:
                chain(result)
                valid = True
            except ValueError:
                valid = False
                counts["invalid_proposals"] += 1
            proposals.append({"operator": fn.__name__, "valid_contract": valid, "graph": graph_record(result)})
            return result
        mutate.__name__ = "counted_" + fn.__name__
        return mutate

    def guided(graph, **kwargs):
        counts["mutation_proposals"] += 1
        before = graph_record(graph)
        try:
            current = chain(graph)
        except ValueError:
            current = ()
        # Existing EcoData score: alpha*match + beta*capability_gap - gamma*cost.
        # Typed candidates adapt that rule to a real station -> optional grid chain.
        options = STATIONS + REGRIDS
        weights = []
        for candidate in options:
            match = candidate in STATIONS
            gap = match and not current
            alpha = 0 if ablation == "hard_only" else 1.
            beta = 0 if ablation in {"no_gap", "hard_only"} else 2.
            gamma = 0 if ablation in {"no_cost", "hard_only"} else .25
            weights.append(math.exp(alpha * match + beta * gap - gamma * evaluator.metadata_cost[candidate]))
        chosen = random.choices(options, weights=weights)[0]
        if chosen in STATIONS:
            plan = (chosen,) + (current[1:] if current and random.random() < .5 else ())
        else:
            plan = ((current[0] if current else random.choice(STATIONS)), chosen)
        result = graph_of(plan)
        proposals.append({"operator": "typed_semantic_mutation", "ablation": ablation,
                          "chosen": chosen, "before": before, "graph": graph_record(result), "valid_contract": True})
        return result

    official = [base_mutations.tree_growth, base_mutations.single_add_mutation, base_mutations.single_change_mutation,
                base_mutations.single_drop_mutation, base_mutations.single_edge_mutation]
    mutations = [counted(fn) for fn in official] if mode in {"A", "B"} else [guided]
    rules = [has_no_cycle, has_no_self_cycled_nodes] + ([legal] if mode in {"B", "C"} else [])
    requirements = GraphRequirements(num_of_generations=80, timeout=timedelta(seconds=180),
                                    max_depth=3, n_jobs=1, show_progress=False, early_stopping_iterations=30)
    params = GPAlgorithmParameters(pop_size=8, max_pop_size=8, mutation_prob=1., crossover_prob=0.,
        mutation_types=mutations, crossover_types=[CrossoverTypesEnum.none], max_num_of_operator_attempts=20)
    generation = GraphGenerationParams(adapter=IdentityAdapter(), rules_for_constraint=rules, available_node_types=available)
    obj = Objective({"real_validation_error_and_declared_cost": objective})
    optimizer = EvoGraphOptimizer(obj, [graph_of(("station.pykrige_ordinary",))], requirements, generation, params)
    optimizer.stop_optimization.add_condition(lambda: len(trace) >= budget, "Shared Objective budget")
    optimizer.optimise(ObjectiveEvaluate(obj))
    valid = [item for item in trace if item["valid"]]
    best = min(valid, key=lambda item: item["score"]) if valid else None
    good = [item for item in valid if item["rmse"] / evaluator.norm["std"] <= .75]
    result = {"variable": variable, "mode": mode, "seed": seed, "ablation": ablation, "pilot": pilot,
        "budget": budget, "scored_objective_calls": len(trace), "counts": dict(counts),
        "numerical_executions": evaluator.executions, "cache_hits": sum(item.get("cache_hit", False) for item in trace),
        "valid_evaluations": len(valid), "invalid_evaluations": len(trace) - len(valid),
        "legal_proposal_ratio": 1 - counts["invalid_proposals"] / max(1, counts["mutation_proposals"]),
        "qualified_graph_found": best is not None,
        "first_qualified_evaluation": valid[0]["evaluation"] if valid else None,
        "first_threshold_evaluation": good[0]["evaluation"] if good else None,
        "search_seconds": time.perf_counter() - beginning,
        "algorithm_seconds": sum(value.get("seconds", 0) for value in evaluator.cache.values()),
        "best": best, "objective_trace": trace, "mutation_trace": proposals,
        "test_opened": False, "official_golem": "0.5.0", "default_mutation_family": [fn.__name__ for fn in official]}
    dump(evaluator.directory / "search_result.json", result)
    print("SEARCH", label, "eval", len(trace), "valid", len(valid), "best", best["rmse"] if best else None, flush=True)
    return result


def run(out, variables, phase):
    if phase == "prepare":
        prepare(out)
        return
    if phase == "pilot":
        if (out / "pilot_results.json").exists():
            raise FileExistsError("Pilot evidence already exists; use a new experiment directory")
        results = [search(out, var, mode, 11, 20, pilot=True) for var in variables for mode in ("A", "B", "C")]
        dump(out / "pilot_results.json", results)
        if any(not row["qualified_graph_found"] for row in results):
            raise ValueError("Pilot failed; main search not permitted")
        return
    if phase == "main":
        pilots = json.loads((out / "pilot_results.json").read_text(encoding="utf-8"))
        assert all(row["qualified_graph_found"] for row in pilots)
        results = []
        for var in variables:
            for seed in SEEDS:
                for mode in ("A", "B", "C"):
                    path = out / "runs" / f"{var}_{mode}_{seed}_full" / "search_result.json"
                    results.append(json.loads(path.read_text(encoding="utf-8")) if path.exists() else search(out, var, mode, seed, 40))
                    dump(out / "main_results.json", results)
        # All selections frozen before opening any test targets.
        dump(out / "frozen_main_selections.json", [{"variable": r["variable"], "mode": r["mode"], "seed": r["seed"],
              "plan": r["best"]["graph"] if r["best"] else None} for r in results])
        return
    if phase == "baselines":
        results = []
        for var in variables:
            evaluator = Evaluator(out, var, out / "baselines" / var)
            for method in STATIONS:
                for suffix in ((), ("spatial.nearest",), ("spatial.bilinear",)):
                    result = evaluator.evaluate(graph_of((method,) + suffix))
                    results.append({"variable": var, **result})
        dump(out / "exhaustive_baselines.json", results)
        return
    if phase == "ablation":
        results = []
        for var in variables:
            for seed in SEEDS:
                for ablation in ("no_gap", "no_cost", "hard_only"):
                    path = out / "runs" / f"{var}_C_{seed}_{ablation}" / "search_result.json"
                    results.append(json.loads(path.read_text(encoding="utf-8")) if path.exists() else search(out, var, "C", seed, 40, ablation=ablation))
        dump(out / "ablation_results.json", results)
        return
    if phase == "test":
        if (out / "final_test_results.json").exists():
            raise FileExistsError("Final test already evaluated; do not silently repeat or overwrite it")
        selected = json.loads((out / "frozen_main_selections.json").read_text(encoding="utf-8"))
        results = []
        cache = {}
        for row in selected:
            if row["plan"] is None:
                results.append({**row, "status": "NO_VALID_SELECTION"})
                continue
            key = (row["variable"], tuple(row["plan"]))
            if key not in cache:
                evaluator = Evaluator(out, row["variable"], out / "final_test" / (row["variable"] + "_" + hashlib.sha256(str(key).encode()).hexdigest()[:10]), phase="test")
                cache[key] = evaluator.evaluate(graph_of(row["plan"]))
            results.append({**row, "test_result": cache[key], "shared_identical_plan_prediction": True})
        dump(out / "final_test_results.json", results)
        return
    raise ValueError(phase)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "output/golem_real_meteorology_round1")
    p.add_argument("--variables", nargs="+", default=["rh", "pressure"])
    p.add_argument("--phase", choices=["prepare", "pilot", "main", "baselines", "ablation", "test"], required=True)
    args = p.parse_args()
    run(args.output, args.variables, args.phase)
