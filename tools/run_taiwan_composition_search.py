"""Real GOLEM search over the existing frozen components' typed execution modes."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import timedelta, datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT / ".venv-golem/Lib/site-packages"))
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import r2_score
import taiwan_annual_adapter as legacy
from taiwan_working_contract import CONFIG, CONTRACT, OUT as ANNUAL, sha, save
from ecodata.algorithms.dag import Artifact, NumericalDAGExecutor
from ecodata.algorithms.station_fusion_dag import StationFusionDAG, StationGraphFactory, SPATIAL
from ecodata.algorithms.frozen_station_dag import FrozenStationFusionDAG, FrozenStationGraphFactory
from ecodata.algorithms.golem_generation import typed_tree_growth
from ecodata.algorithms.registry import create_default_registry
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from ecodata.algorithms.semantic_requirements import SemanticFeatureExtractor, RequirementMapper, EvidenceValue
from ecodata.evaluation.quality_gates import RawInputQualityGate
from golem.core.optimisers.graph import OptGraph
from golem.core.adapter import IdentityAdapter
from golem.core.dag.verification_rules import has_no_cycle, has_no_self_cycled_nodes
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators import base_mutations
from golem.core.optimisers.genetic.operators.crossover import CrossoverTypesEnum
from golem.core.optimisers.objective import Objective, ObjectiveEvaluate
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams

POOL = ROOT / "output/capability_final_acceptance/frozen_generic_pool.json"
OUT = ROOT / "output/taiwan_composition_search_v1"


def setup():
    legacy.configure()
    backend = FrozenStationFusionDAG(xy=legacy.task.xy,
        quality=legacy.working_factory().adapters[("quality", "run")].execute,
        input_quality=lambda table:RawInputQualityGate().validate(legacy.task.field(table,table.value),
            variable="pm25",intended_operation="spatial",expected_unit="ug m-3"))
    frozen = json.loads(POOL.read_text(encoding="utf-8"))["algorithms"]
    available=[a["id"] for a in frozen]
    registry=create_default_registry()
    registry.register_task_adapters("taiwan_pm25_station_aod",backend.adapters(),algorithm_ids=available,
        boundary_ids=("source.train","source.query","quality.station"))
    factory = FrozenStationGraphFactory(registry.task_adapters("taiwan_pm25_station_aod"), available, max_nodes=24, max_depth=12)
    sample=pd.read_parquet(ANNUAL/"data/ground_non_test.parquet",columns=["longitude","latitude","value","split","reason","time"])
    sample=sample.loc[sample.split.eq("train") & sample.reason.eq("VALID")]
    sample=sample.loc[sample.time.eq(sample.time.iloc[0])]
    semantics=SemanticFeatureExtractor().extract(legacy.task.field(sample,sample.value),"pm25",confirmed=dict(
        variable_class="continuous",unit="ug m-3",nonnegative=True,crs="EPSG:4326",
        spatial_semantics="station_point_support",temporal_semantics="interval_mean",temporal_resolution="1h"))
    # Recognition of a working convention is not proof of the author's datum.
    from dataclasses import replace
    semantics=replace(semantics,spatial=EvidenceValue(semantics.spatial.value,("frozen working contract; author datum unverified",),"WORKING_ASSUMPTION"),
        temporal=EvidenceValue(semantics.temporal.value,("frozen [t,t+1h) working convention",),"WORKING_ASSUMPTION"))
    factory.requirements=RequirementMapper().map(semantics,dict(operation="station_to_points",source_count=1))
    factory.semantics=semantics
    factory.task_registry=registry
    return backend, factory, frozen


def record(graph):
    nodes = list(graph.nodes)
    return [{"operation": n.name, "parameters": dict(n.parameters),
             "parents": [nodes.index(p) for p in n.nodes_from]} for n in nodes]


def restore(rows):
    from golem.core.optimisers.graph import OptNode
    nodes = [OptNode({"name": r["operation"], "params": r["parameters"]}) for r in rows]
    for node, row in zip(nodes, rows):
        node.nodes_from = [nodes[i] for i in row["parents"]]
    return OptGraph(nodes)


def fingerprint(graph, factory):
    return hashlib.sha256(factory.inspect(graph)["key"].encode()).hexdigest()


def cases(role="validation", selected=None):
    # Preserve original source-row order: RF bootstrapping is order-sensitive.
    frame = pd.read_parquet(ANNUAL / "data/ground_non_test.parquet")
    if role == "test":
        from taiwan_working_contract import ground
        frame = pd.concat([frame.loc[frame.split.eq("train")], ground(include_test=True)], ignore_index=True)
    cov = pd.read_parquet(ANNUAL / "data/aod_hourly_all_sites.parquet")
    frame = frame.loc[frame.reason.eq("VALID")].merge(cov, on=["station_id", "time"], validate="one_to_one")
    columns = ["station_id", "longitude", "latitude", "value", "time", "split", "unit", "aod"]
    frame = frame.loc[frame.aod.notna(), columns]
    if selected is not None:
        frame = frame.loc[frame.time.isin(pd.to_datetime(selected, utc=True))]
    out = []
    for stamp, table in frame.groupby("time", sort=True):
        train, target, status = legacy.qualified(table, role)
        if status == "ELIGIBLE":
            out.append((stamp, train.reset_index(drop=True), target.reset_index(drop=True)))
    return out


def freeze_plan():
    if (OUT / "protocol.json").exists():
        path=OUT/"protocol.json"
        previous=json.loads(path.read_text())
        current={name:sha(ROOT/name) for name in previous["code_hashes"]}
        if current != previous["code_hashes"]:
            assert not (OUT/"smoke_summary.json").exists(), "Preserve run evidence; changed code requires a new output version"
            save(OUT/"protocol_history"/(sha(path)+".json"),previous)
            previous.update(code_hashes=current,pre_execution_amendment="Source changed during pre-execution integration checks; previous protocol preserved; no numerical evaluation has started")
            save(path,previous)
        return
    backend, factory, frozen = setup()
    all_cases = cases()
    times = pd.Series([t for t, _, _ in all_cases])
    selected = []
    # Fixed eight evenly spaced eligible hours/month; eligibility uses only
    # observed availability, geometry and frozen QA, never prediction errors.
    for _, group in times.groupby(times.dt.tz_convert("Asia/Taipei").dt.month):
        selected.extend(group.iloc[np.unique(np.linspace(0, len(group)-1, 8).astype(int))].tolist())
    recipes = factory.recipes("validated_field")
    graphs = {}
    for _, recipe in recipes:
        g = OptGraph([factory.instantiate(recipe)])
        graphs[fingerprint(g, factory)] = record(g)
    save(OUT / "catalog.json", graphs)
    contracts = []
    for spec in frozen:
        adapters = [(key, a) for key, a in backend.adapters().items() if key[0] == spec["id"]]
        contracts.append(dict(spec, task_eligible=bool(adapters), modes=[dict(mode=k[1],
            inputs={n: {"kind": p.kind, "required_metadata": p.required_metadata} for n,p in a.inputs.items()},
            outputs={n: {"kind": p.kind, "required_metadata": p.required_metadata} for n,p in a.outputs.items()}) for k,a in adapters],
            task_reason="Executable station/OOF-residual port adapter" if adapters else
            "This task has station PM2.5 + dimensionless AOD, no required grids/profiles/historical simulated pairs/weights; no compatible task execution mode",
            task_rules={"unit":"ug m-3", "covariate":"QA-passed AOD550 dimensionless", "metric_crs":"EPSG:32651",
                "time":"same frozen [t,t+1h) UTC interval", "support":"training convex hull", "fit":"train stations only",
                "parameters":"PyKrige ordinary/universal, linear variogram; 5 station-disjoint OOF folds with hull masks; MetPy official defaults; NO RF/IDW aliases",
                "errors":["missing field", "unit/CRS/time mismatch", "nonfinite", "state fingerprint mismatch", "outside hull", "label leakage"]}))
    save(OUT / "contracts_34.json", contracts)
    save(OUT / "semantic_requirements.json",dict(semantics=factory.semantics.as_dict(),
        requirements=factory.requirements.as_dict(),AOD_role="dimensionless predictor, NOT a second PM2.5 source to average"))
    save(OUT / "protocol.json", dict(frozen_at=datetime.now(timezone.utc).isoformat(),
        contract_sha256=sha(CONFIG), pool_sha256=sha(POOL), seeds=[11,22,33,44,55],
        evaluation_budget=20, proposal_budget=160, wall_seconds=120, catalog_size=len(graphs),
        input_hashes={p.name:sha(p) for p in (ANNUAL/"data").glob("*.parquet")},
        code_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__), ROOT/"src/ecodata/algorithms/station_fusion_dag.py",ROOT/"src/ecodata/algorithms/frozen_station_dag.py",ROOT/"src/ecodata/fusion/residual.py",ROOT/"src/ecodata/algorithms/dag.py",ROOT/"src/ecodata/algorithms/golem_generation.py",ROOT/"src/ecodata/algorithms/registry.py",ROOT/"src/ecodata/algorithms/pipeline.py",ROOT/"src/ecodata/algorithms/semantic_requirements.py"]},
        times=[str(t) for t in selected], search_cohort="96 deterministic eligible hours stratified across 12 months, not full validation; final selected graphs additionally evaluated on full frozen validation",
        objective="RMSE on entire fixed common subset; quality BLOCK -> reject (1e6); partial graph ->1000+missing ports; no cost mixed into RMSE",
        cache="cold node/objective caches for every arm and seed; repeated keys count budget and explicit cache hits; no cross-arm warm advantage",
        guidance="C uses same official operators, up to 4 proposals, weights missing-port reduction and fewer unneeded nodes; every proposal charged",
        final_test="14 unchanged stations PREVIOUSLY EVALUATED in chapter29: reused holdout, not fresh independent confirmation; never loaded during search",
        inference="exploratory paired seed descriptives only; no significance/independent station-hour assumption"))
    print("FROZEN", len(graphs), "graphs", len(selected), "hours", flush=True)


def metrics(values, truth):
    error = np.asarray(values)-np.asarray(truth)
    return dict(RMSE=float(np.sqrt(np.mean(error**2))), MAE=float(np.mean(abs(error))),
        Bias=float(error.mean()), R2=float(r2_score(truth, values)), count=len(error))


class Evaluator:
    def __init__(self, selected, folder, role="validation"):
        self.backend, self.factory, _ = setup()
        self.pipeline=UnifiedAlgorithmPipeline(self.factory.task_registry)
        self.execution_version=json.dumps(json.loads((OUT/"protocol.json").read_text())["code_hashes"],sort_keys=True)
        self.cases, self.folder, self.role = selected, folder, role
        self.folder.mkdir(parents=True, exist_ok=True)

    def execute(self, graph, key):
        before = perf_counter()
        rows, traces, failures, clips = [], [], [], 0
        executed = cached = 0
        for stamp, train, target in self.cases:
            meta = dict(unit="ug m-3", metric_crs="EPSG:32651", time=stamp, contract=CONTRACT["version"])
            external = {"train":Artifact("training", train, meta),
                "query":Artifact("query", target.drop(columns=["value","split"]), meta)}
            try:
                result = self.pipeline.run_dag(graph,task_id="taiwan_pm25_station_aod",external=external,version=self.execution_version)
                output = result["outputs"]["field"].value
                pred, raw = output["prediction"], output["raw_prediction"]
                usage=np.asarray(output["residual_used_mask"],dtype=int) if "residual_used_mask" in output else np.full(len(target),-1)
                assert np.isfinite(pred).all() and (pred >= 0).all()
                clips += int((raw < 0).sum())
                executed += sum(not t["cache_hit"] for t in result["trace"])
                cached += sum(t["cache_hit"] for t in result["trace"])
                traces.append(dict(time=str(stamp), trace=result["trace"], quality=output["quality"]))
            except Exception as exc:
                pred, raw = np.full(len(target),np.nan), np.full(len(target),np.nan)
                usage=np.full(len(target),-2)
                failure_trace=getattr(exc,"trace",[])
                failures.append(dict(time=str(stamp),error=str(exc),trace=failure_trace,
                    failure_node=next((t.get("node") for t in reversed(failure_trace) if t.get("status")=="FAIL"),None),
                    category="QUALITY_OR_DATA_CONTRACT",repair="Choose another compatible graph; do not fill or drop held-out targets"))
            for i, row in target.iterrows():
                rows.append(dict(time=stamp,station_id=row.station_id,observed=row.value,prediction=pred[i],
                    raw_prediction=raw[i],longitude=row.longitude,latitude=row.latitude,residual_usage=usage[i]))
        frame = pd.DataFrame(rows)
        folder = self.folder/key
        folder.mkdir(exist_ok=True)
        frame.to_parquet(folder/"predictions.parquet",index=False)
        with (folder/"execution_trace.jsonl").open("w",encoding="utf-8") as f:
            for row in traces:
                f.write(json.dumps(row,default=str)+"\n")
        valid = np.isfinite(frame.prediction)
        # Never rank a method on only its easy/successful cases.
        summary = dict(status="WARN_WORKING_CONTRACT" if valid.all() else "BLOCK_INCOMPLETE_COVERAGE",
            coverage=float(valid.mean()),failure_hours=len(failures), expected=len(frame), clips=clips,
            executed_nodes=executed,cached_nodes=cached,seconds=perf_counter()-before,
            background_only_points=int(frame.residual_usage.eq(0).sum()),
            failures=failures,score=1e6)
        if valid.all():
            summary.update(metrics(frame.prediction,frame.observed))
            summary["score"] = summary["RMSE"]
        ds = xr.Dataset({"pm25":("sample",frame.prediction.to_numpy()),
            "observed":("sample",frame.observed.to_numpy()), "valid_mask":("sample",valid.to_numpy().astype("i1"))},
            coords={"longitude":("sample",frame.longitude.to_numpy()),"latitude":("sample",frame.latitude.to_numpy()),
                "time":("sample",pd.to_datetime(frame.time,utc=True).dt.tz_localize(None).to_numpy()),
                "station_id":("sample",frame.station_id.to_numpy().astype(str))},
            attrs={"contract":CONTRACT["version"],"crs":"EPSG:4326 working assumption", "metric_crs":"EPSG:32651",
                "time_window":"[t,t+1h) UTC", "split":self.role,"grid_coverage":"NOT_APPLICABLE_POINT_OUTPUT"})
        ds.pm25.attrs["units"] = "ug m-3"
        ds.observed.attrs["units"]="ug m-3"
        ds.longitude.attrs.update(units="degrees_east",standard_name="longitude")
        ds.latitude.attrs.update(units="degrees_north",standard_name="latitude")
        ds["residual_usage"]=("sample",frame.residual_usage.to_numpy().astype("i1"))
        ds.residual_usage.attrs.update(flag_values=np.array([-2,-1,0,1],dtype="i1"),
            flag_meanings="execution_failed not_applicable background_only background_plus_residual")
        ds.to_netcdf(folder/"predictions.nc")
        with xr.open_dataset(folder/"predictions.nc") as check:
            np.testing.assert_allclose(check.pm25,frame.prediction,equal_nan=True)
        summary["netcdf_readback"] = True
        save(folder/"result.json",summary)
        return summary


class BudgetStop(RuntimeError):
    pass


def real_graph_feedback(evaluator, graph, key, cache):
    """One real execution per content key, including terminal failure feedback."""
    if key in cache:
        return dict(cache[key],real_execution=False,cache_hit=True)
    result=dict(evaluator.execute(graph,key),real_execution=True,cache_hit=False,
                key=key,graph=record(graph))
    cache[key]=dict(result)
    return result


def write_search_event(path, event, **payload):
    # Line-by-line durable records preserve partial attempts after interruption.
    with path.open("a",encoding="utf-8") as stream:
        stream.write(json.dumps(dict(event=event,**payload),ensure_ascii=False,default=str)+"\n")


def search_job(mode, seed, stage="search", budget=20, *, output=None, case_times=None,
               evaluator_factory=None, case_provider=None):
    global OUT
    if output is not None: OUT=Path(output)
    protocol = json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    folder = OUT/stage/f"{mode}_{seed}"
    if (folder/"summary.json").exists():
        return json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    random.seed(seed); np.random.seed(seed)
    selected_times=protocol["times"] if case_times is None else case_times
    evaluator = (evaluator_factory or Evaluator)((case_provider or cases)(selected=selected_times),folder/"executions")
    factory = evaluator.factory
    initial = factory.initial()
    counts, records, proposals, cache = Counter(), [], [], {}
    begun = perf_counter()
    journal=folder/"search_events.jsonl"
    if journal.exists():
        raise FileExistsError("Interrupted attempt journal exists; preserve it and use a fresh output version")
    write_search_event(journal,"run_start",mode=mode,seed=seed,budget=budget,
        protocol_sha256=sha(OUT/"protocol.json"),evaluation_times=selected_times)
    operators = [typed_tree_growth,base_mutations.single_add_mutation,base_mutations.single_change_mutation,
                 base_mutations.single_drop_mutation,base_mutations.single_edge_mutation]

    def limited():
        return len(records)>=budget or counts["proposals"]>=protocol["proposal_budget"] or perf_counter()-begun>=protocol["wall_seconds"]

    def legal(g,*args,**kwargs):
        counts["posterior_checks"] += 1
        try:
            factory.inspect(g)
            return True
        except Exception as exc:
            counts["posterior_rejections"] += 1
            proposals.append(dict(event="posterior_reject",error=str(exc)))
            return False

    def objective(g):
        counts["objective_callbacks"] += 1
        if len(records)>=budget:
            counts["deferred_callbacks"] += 1
            return 1e6
        row = dict(index=len(records)+1,real_execution=False,cache_hit=False,graph=record(g))
        write_search_event(journal,"objective_start",index=row["index"],elapsed=perf_counter()-begun,graph=row["graph"])
        try:
            status = factory.inspect(g)
            key = fingerprint(g,factory)
            row.update(key=key,gap=status["gap"])
            if not status["complete"]:
                row.update(status="INCOMPLETE",score=1000+status["gap"],missing=status["missing"])
            elif key in cache:
                row.update(real_graph_feedback(evaluator,g,key,cache))
                counts["cache_hits"] += 1
            else:
                counts["real_graph_evaluations"] += 1
                row.update(real_graph_feedback(evaluator,g,key,cache))
                if row["score"]<1e6:
                    counts["successful_graph_evaluations"] += 1
                else:
                    counts["numeric_failures"] += 1
        except Exception as exc:
            row.update(status="CONTRACT_REJECT",score=1e6,error=str(exc),call_chain=traceback.format_exc())
            counts["contract_rejections"] += 1
        row["elapsed"] = perf_counter()-begun
        records.append(row)
        write_search_event(journal,"objective_result",record=row,counts=dict(counts))
        return row["score"]

    def proposal(fn,g,kwargs):
        if limited():
            raise BudgetStop("Shared bounded proposal/objective/wall limit")
        counts["proposals"] += 1
        write_search_event(journal,"proposal_start",operator=fn.__name__,proposal=counts["proposals"],elapsed=perf_counter()-begun)
        before = factory.inspect(g)["gap"]
        try:
            c = fn(deepcopy(g),kwargs["requirements"],kwargs["graph_gen_params"],kwargs["parameters"])
            status = factory.inspect(c)
            proposals.append(dict(operator=fn.__name__,valid=True,gap_before=before,gap_after=status["gap"],
                key=fingerprint(c,factory),graph=record(c)))
            write_search_event(journal,"proposal_result",record=proposals[-1])
            return c,status
        except Exception as exc:
            proposals.append(dict(operator=fn.__name__,valid=False,error=str(exc),call_chain=traceback.format_exc()))
            counts["proposal_rejections"] += 1
            write_search_event(journal,"proposal_result",record=proposals[-1])
            return deepcopy(g),None

    def counted(fn):
        def mutation(g,**kw):
            return proposal(fn,g,kw)[0]
        mutation.__name__ = "counted_"+fn.__name__
        return mutation

    def guided(g,**kw):
        choices, weights = [], []
        before = factory.inspect(g)["gap"]
        seen = set()
        for _ in range(4):
            if limited():
                break
            c,status = proposal(random.choice(operators),g,kw)
            if status is not None and status["key"] not in seen:
                previous=cache.get(fingerprint(c,factory))
                if previous is not None and previous.get("score",1e6)>=1e6:
                    counts["failure_feedback_avoided"]+=1
                    continue
                seen.add(status["key"])
                choices.append(c)
                weights.append(np.exp(np.clip(2*(before-status["gap"])-.1*status["algorithm_nodes"],-20,20)))
        return random.choices(choices,weights=weights)[0] if choices else deepcopy(g)

    requirements = GraphRequirements(num_of_generations=80,timeout=timedelta(seconds=protocol["wall_seconds"]),
        start_depth=12,max_depth=12,n_jobs=1,show_progress=False,early_stopping_iterations=15)
    parameters = GPAlgorithmParameters(pop_size=4,max_pop_size=4,mutation_prob=1.,crossover_prob=0.,
        mutation_types=[guided] if mode=="C" else [counted(f) for f in operators],
        crossover_types=[CrossoverTypesEnum.none],max_num_of_operator_attempts=3)
    generation = GraphGenerationParams(adapter=IdentityAdapter(),node_factory=factory,random_graph_factory=factory.random_graph,
        rules_for_constraint=[has_no_cycle,has_no_self_cycled_nodes]+([legal] if mode in {"B","C"} else []))
    obj = Objective({"validation_rmse":objective})
    optimizer = EvoGraphOptimizer(obj,[initial],requirements,generation,parameters)
    optimizer.stop_optimization.add_condition(limited,"Shared budgets")
    error = None
    try:
        if mode == "R":
            objective(initial)
            while not limited():
                counts["proposals"] += 1
                objective(factory.random_graph(requirements))
        else:
            optimizer.optimise(ObjectiveEvaluate(obj))
    except BudgetStop:
        counts["bounded_resource_stop"] += 1
    except Exception as exc:
        error = dict(error=str(exc),call_chain=traceback.format_exc())
    successes = [v for v in cache.values() if v["score"]<1e6]
    summary = dict(mode=mode,seed=seed,counts=dict(counts),objective_records=records,proposals=proposals,
        factory_trace=factory.trace,wall_seconds=perf_counter()-begun,optimizer_error=error,
        unique_legal_graphs=len(successes),best=min(successes,key=lambda v:v["score"]) if successes else None,
        gap_completions=sum(p.get("gap_before",0)>0 and p.get("gap_after",1)==0 for p in proposals),
        evaluation_times=selected_times,
        stop_reason="BUDGET" if limited() else "OFFICIAL_OPTIMIZER_STOP",test_labels_read=False)
    save(folder/"summary.json",summary)
    write_search_event(journal,"run_complete",counts=dict(counts),stop_reason=summary["stop_reason"])
    print(stage,mode,seed,"evaluations",len(records),"real",counts["real_graph_evaluations"],"legal",len(successes),"error",error,flush=True)
    return {k:v for k,v in summary.items() if k not in {"objective_records","proposals","factory_trace"}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["freeze","smoke","pilot","search","enumerate"],default="freeze")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    global OUT
    if args.output is not None: OUT=args.output.resolve()
    freeze_plan()
    if args.stage=="freeze": return
    if args.stage=="enumerate":
        evaluator=Evaluator(cases(selected=json.loads((OUT/"protocol.json").read_text())["times"]),OUT/"enumeration")
        for key, rows in json.loads((OUT/"catalog.json").read_text()).items():
            if not (OUT/"enumeration"/key/"result.json").exists():
                print("ENUM",key[:8],evaluator.execute(restore(rows),key)["score"],flush=True)
        return
    if args.stage=="smoke": jobs=[(m,11,"smoke",5) for m in "ABC"]
    elif args.stage=="pilot": jobs=[(m,11,"pilot",20) for m in "ABC"]
    else: jobs=[(m,s,"search",20) for s in [11,22,33,44,55] for m in "ABC"]
    if args.stage=="search":
        smoke=json.loads((OUT/"pilot_summary.json").read_text())
        assert all(r["best"] and not r["optimizer_error"] for r in smoke), "Smoke must actually execute all arms"
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pilot_times=json.loads((OUT/"protocol.json").read_text())["times"][::8] if args.stage=="pilot" else None
        futures=[pool.submit(search_job,*j,output=OUT,case_times=pilot_times) for j in jobs]
        for f in as_completed(futures):
            rows.append(f.result())
            save(OUT/f"{args.stage}_summary.json",rows)


if __name__=="__main__": main()
