"""Real full-grammar random-search reference under the same cold-cache ceilings."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import run_taiwan_composition_search as run
from golem.core.optimisers.optimization_parameters import GraphRequirements


def worker(seed,output):
    run.OUT=Path(output)
    folder=run.OUT/"random_reference"/str(seed)
    receipt=folder/"summary.json"
    if receipt.exists(): return json.loads(receipt.read_text(encoding="utf-8"))
    protocol=json.loads((run.OUT/"protocol.json").read_text(encoding="utf-8"))
    random.seed(seed); np.random.seed(seed)
    evaluator=run.Evaluator(run.cases(selected=protocol["times"]),folder/"executions")
    factory=evaluator.factory
    initial=factory.initial()
    records=[dict(index=1,status="INCOMPLETE",score=1000+factory.inspect(initial)["gap"],
        real_execution=False,cache_hit=False,elapsed=0.,graph=run.record(initial))]
    cache={}; proposals=0
    begun=perf_counter()
    requirements=GraphRequirements(max_depth=12)
    journal=folder/"search_events.jsonl"
    run.write_search_event(journal,"run_start",seed=seed,budget=20,policy="initial same incomplete graph; then shared typed random_graph")
    while len(records)<20 and proposals<160 and perf_counter()-begun<120:
        proposals+=1
        graph=factory.random_graph(requirements)
        key=run.fingerprint(graph,factory)
        run.write_search_event(journal,"objective_start",index=len(records)+1,key=key)
        row=run.real_graph_feedback(evaluator,graph,key,cache)
        row.update(index=len(records)+1,elapsed=perf_counter()-begun)
        records.append(row)
        run.write_search_event(journal,"objective_result",record=row)
    good=[r for r in records if r.get("RMSE") is not None and r["score"]<1e6]
    result=dict(seed=seed,policy="R_full_typed_grammar_random_search",records=records,
        best=min(good,key=lambda r:r["RMSE"]) if good else None,proposals=proposals,
        evaluations=len(records),real=sum(r["real_execution"] for r in records),cache=sum(r["cache_hit"] for r in records),
        wall_seconds=perf_counter()-begun,test_labels_read=False,
        source_sha256=run.sha(Path(__file__)),protocol_sha256=run.sha(run.OUT/"protocol.json"),
        caveat="Uniform complete-grammar proposals, not incremental mutation; same frozen pool/grammar/data/starting incomplete graph and numerical budget")
    run.save(receipt,result)
    print("R",seed,"evaluations",len(records),"real",result["real"],"best",result["best"]["RMSE"] if good else None,flush=True)
    return {k:v for k,v in result.items() if k!="records"}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=run.ROOT/"output/taiwan_frozen_pool_search_final")
    parser.add_argument("--workers",type=int,default=3)
    args=parser.parse_args()
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(worker,seed,args.output.resolve()) for seed in [11,22,33,44,55]]
        for f in as_completed(futures):
            rows.append(f.result())
            run.save(args.output/"random_reference_summary.json",rows)
