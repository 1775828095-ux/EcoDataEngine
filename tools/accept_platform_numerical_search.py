"""Bounded service acceptance ONLY: three predeclared 2015 public cases, two clients.

Does not call experiment runners, enumeration, or any 2016/2017 evaluation.
"""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"tools"))
from ecodata.algorithms.temporal_search import ReferenceEvaluation
from ecodata.algorithms.numerical_search import save
from ecodata.algorithms.mixed_gap_dag import Batch,SealedArtifact,readonly,TASK
from ecodata.interface.temporal_search_page import process_temporal_request,parse_upload
from run_beijing_mixed_gap import search_request
import joblib,numpy as np,pandas as pd

OUT=ROOT/"output/platform_search_service_20261021/bounded_final"


def main():
    source=ROOT/"output/beijing_mixed_gap_20261020/unified_platform_v4"
    config=dict(objective_limit=5,proposal_limit=50,callback_limit=100,wall_seconds=60)
    request=dict(operation="temporal_missing_recovery",require_complete_coverage=True,prefer_observation_support=True,time_access="both_sides")
    save(OUT/"protocol.json",dict(cases=[0,8,1],seed=11,mode="C",budget=config,request=request,
         scope="engineering acceptance, no ranking, no final test",evaluation="pooled RMSE over original legitimate validation positions only"))
    data=joblib.load(source/"data/2015/public_inputs.joblib")
    reference=pd.read_parquet(source/"data/2015/private_targets.parquet")
    states=joblib.load(source/"training_states.joblib")
    selected=[0,8,1];cases=[];lengths=[];offsets=[];query=[];truth=[];offset=0
    for index in selected:
        case=data["batch"].cases[index];start,end=data["batch"].offsets[index]
        for array in (case.time,case.values,case.missing):array.flags.writeable=False
        cases.append(case);lengths.extend(data["batch"].lengths[start:end]);offsets.append((offset,offset+end-start))
        hit=(data["query"]>=start)&(data["query"]<end)
        query.extend(data["query"][hit]-start+offset);truth.extend(reference.truth.to_numpy()[hit]);offset+=end-start
    batch=Batch(tuple(cases),readonly(np.array(lengths)),tuple(offsets));meta=dict(unit="ug m-3",contract=TASK)
    external=dict(batch=SealedArtifact("masked_batch",batch,meta),history=SealedArtifact("fitted_history",states,meta),
                  query=SealedArtifact("query_positions",readonly(np.array(query)),meta))
    results=[]
    for name,client in [("page",process_temporal_request),("research",search_request)]:
        result=client(external,request,ReferenceEvaluation(truth),config,OUT/name,seed=11)
        results.append(result)
    def stable(result):
        return dict(rows=[{k:r.get(k) for k in ("key","gap","status","objective","fitness","prediction_fingerprint")} for r in result["rows"]],
                    weights=[{k:p.get(k) for k in ("keys","weights","probabilities","chosen_index")} for p in result["proposals"] if p.get("status")=="SHARED_DECISION"],
                    graph=result["best"]["graph"] if result["best"] else None)
    a,b=results
    decision_a=json.loads((OUT/"page/decision.json").read_text(encoding="utf-8"));decision_b=json.loads((OUT/"research/decision.json").read_text(encoding="utf-8"))
    decision_a.pop("seconds",None);decision_b.pop("seconds",None)
    checks=dict(same_semantics=decision_a==decision_b,same_domain=a["candidate_domain"]==b["candidate_domain"],
                same_trace_and_selection=stable(a)==stable(b),real_execution_each=all(r["counts"].get("real_executions",0)>0 for r in results),
                incomplete_initial=all(any(row["status"]=="INCOMPLETE" for row in r["rows"]) for r in results),
                weights_consumed=all(any(p.get("status")=="SHARED_DECISION" and "chosen_index" in p for p in r["proposals"]) for r in results),
                optimizer_feedback=all(r["counts"].get("callbacks",0)>0 and r["best"] for r in results))
    # No scientific success claim: these are deterministic integration checks.
    checks={k:bool(v) for k,v in checks.items()}
    save(OUT/"acceptance.json",dict(checks=checks,status="PASS" if all(checks.values()) else "FAIL",targets=len(truth),
         runs=[dict(client=name,counts=r["counts"],best_objective=r["best"]["objective"] if r["best"] else None,
                    seconds=r["platform_total_seconds"],optimizer_error=r["optimizer_error"]) for name,r in zip(["page","research"],results)],
         public_input_sha256=hashlib.sha256((source/"data/2015/public_inputs.joblib").read_bytes()).hexdigest()))
    print(checks)
    if not all(checks.values()):raise RuntimeError("Bounded acceptance failed; inspect saved evidence")


def accept_csv(name="csv_upload_delivery"):
    """Same UI CSV parser with an original, predeclared 168-hour public case."""
    source=ROOT/"output/beijing_mixed_gap_20261020/unified_platform_v4"
    data=joblib.load(source/"data/2015/public_inputs.joblib")
    truth=pd.read_parquet(source/"data/2015/private_targets.parquet")
    case=data["batch"].cases[8];start,end=data["batch"].offsets[8]
    refs=truth.loc[(data["query"]>=start)&(data["query"]<end),["station","time","truth"]]
    frame=pd.DataFrame(dict(station=case.station,time=case.time,pm25=case.values))
    destination=OUT/name;destination.mkdir(parents=True,exist_ok=True)
    frame.to_csv(destination/"input.csv",index=False);refs.to_csv(destination/"reference.csv",index=False)
    states=joblib.load(source/"training_states.joblib")
    state=states[case.station]
    history={case.station:dict(medians=state["medians"].tolist(),training_end=state["training_end"])}
    save(destination/"history.json",history)
    budget=dict(objective_limit=4,proposal_limit=30,callback_limit=80,wall_seconds=30)
    request=dict(operation="temporal_missing_recovery",require_complete_coverage=True,prefer_observation_support=True,time_access="both_sides")
    save(destination/"protocol.json",dict(source_case_index=8,seed=11,budget=budget,request=request,reference_role="2015 validation only"))
    external,evaluation=parse_upload((destination/"input.csv").read_bytes(),(destination/"reference.csv").read_bytes(),history,variable="pm25",unit="ug m-3")
    result=process_temporal_request(external,request,evaluation,budget,destination/"run",seed=11)
    save(destination/"acceptance.json",dict(status="PASS" if result["best"] and result["product"]["readback"]=="PASS" else "FAIL",
         counts=result["counts"],reference_samples=len(refs),csv_parser="actual UI handler",manual_browser="NOT_RUN"))
    if not result["best"]:raise RuntimeError("CSV end-to-end search did not find a complete result")


if __name__=="__main__":main()
