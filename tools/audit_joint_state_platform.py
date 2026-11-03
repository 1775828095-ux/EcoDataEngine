"""Independent score arithmetic/product readback; does not call the scorer."""
import hashlib
import json
from pathlib import Path
import numpy as np
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output/joint_state_20261024"
ACCEPT=OUT/"acceptance_v3"


def main():
    checked=0;decisions=0;mutations=0;failures=[]
    for path in ACCEPT.glob("*/summary.json"):
        report=json.loads(path.read_text(encoding="utf-8"));decisions+=1
        scenarios=report["portraits"]["model"]["state_policy"]["scenario"]["issues"]
        issues={i["attribute"]:i for i in scenarios};total=sum(i["importance"] for i in scenarios)
        for row in report["rows"]:
            try:
                assert row["status"]=="PASS" and row["file_readback"]=="PASS"
                product=Path(row["product"])
                assert hashlib.sha256(product.read_bytes()).hexdigest()==row["product_sha256"]
                with xr.open_dataset(product) as ds:
                    array=np.asarray(ds[report["evaluation_protocol"]["variable"]],float)
                    assert np.isfinite(array).all()
                states=[a["metadata"]["joint_state"] for n in row["node_trace"] for a in n.get("outputs",{}).values()
                        if a.get("kind")=="validated_field" and "joint_state" in a.get("metadata",{})]
                assert states and abs(states[-1]["metrics"]["missing_fraction"]-(1-np.isfinite(array).mean()))<1e-10
                lower=upper=0.
                for term in row["matching"]["terms"]:
                    issue=issues[term["dimension"]]
                    assert abs(term["weight"]-issue["importance"]/total)<1e-12
                    before=term["data_evidence"]["before"];after=term["algorithm_evidence"]["after"]
                    if before is not None and after is not None:
                        r0=min(1.,max(0.,(before-issue["acceptable"])/(issue["bad"]-issue["acceptable"])))
                        r1=min(1.,max(0.,(after-issue["acceptable"])/(issue["bad"]-issue["acceptable"])))
                        assert abs(term["utility"]-(r0-r1))<1e-10
                    else:assert term["utility"] is None
                    lower+=term["interval"][0];upper+=term["interval"][1]
                assert np.allclose([lower,upper],row["matching"]["score_interval"])
                assert row["matching"]["score"] is None or np.isclose(row["matching"]["score"],lower)
                checked+=1
            except Exception as exc:failures.append(dict(summary=str(path),key=row.get("key"),error=str(exc)))
        evolution=report.get("state_evolution")
        if evolution:
            assert evolution["optimizer_error"] is None
            mutations+=sum(p["status"]=="ACCEPTED" and p["parent"]!=p["child"] for p in evolution["proposals"])
            bykey={r["key"]:r for r in report["rows"]}
            expected=min(bykey,key=lambda k:(-bykey[k]["matching"]["score_interval"][0],
                sum(a["declared_cost"] for a in report["portraits"]["algorithms"] for identifier in bykey[k]["steps"] if a["id"]==identifier),
                len(bykey[k]["steps"]),k))
            assert expected==report["recommended_key"]==evolution["selected"]
            assert evolution["true_effect_global_optimum"]=="NOT_PROVEN"
    assert decisions==7 and checked==35 and mutations>0 and not failures
    result=dict(status="PASS",decisions=decisions,numerical_products=checked,accepted_distinct_native_mutations=mutations,
                failures=failures,scope="Independent arithmetic and artifact checks, not native simulator or scientific efficacy validation")
    (OUT/"independent_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
