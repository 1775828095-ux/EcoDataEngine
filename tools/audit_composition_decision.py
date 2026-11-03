"""Independent artifact/arithmetic audit; does not invoke the planner or scorer."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output/composition_20261024"
ACCEPT=OUT/"acceptance_v5"


def load(p):return json.loads(p.read_text(encoding="utf-8"))


def numeric_key(ds):
    digest=hashlib.sha256()
    for name in sorted(ds.variables):
        a=ds[name];digest.update(json.dumps([name,list(a.dims),list(a.shape)]).encode())
        values=a.values
        if values.dtype.kind in "fiu":values=values.astype("float64")
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def main():
    failures=[];products=0;decisions=0;mutations=0;counts={}
    for folder in sorted(p.parent for p in ACCEPT.glob("*/summary.json")):
        try:
            report=load(folder/"summary.json");protocol=load(folder/"request.json")
            assert math.prod(a["alternatives"] for a in protocol["axes"])==protocol["grammar_count"]
            assert report["counts"]["grammar_combinations"]==protocol["grammar_count"]
            sources={}
            for item in protocol["visible_input_snapshots"]:
                path=Path(item["path"]);assert hashlib.sha256(path.read_bytes()).hexdigest()==item["sha256"]
                with xr.open_dataset(path,engine="h5netcdf") as ds:sources[item["source_id"]]=ds.load()
            # A path/provenance-only edit is not an independent scientific input.
            numeric=set();valid=[]
            for row in report["rows"]:
                if row["status"]!="PASS":
                    assert row.get("error");continue
                path=Path(row["product"]);assert hashlib.sha256(path.read_bytes()).hexdigest()==row["product_sha256"]
                with xr.open_dataset(path,engine="h5netcdf") as ds:output=ds.load()
                numeric.add(numeric_key(output));products+=1
                overall=np.zeros(2);total=0.
                for portrait,matching in zip(protocol["portraits"],row["matching"]["fields"]):
                    model=portrait["model"];issues={i["attribute"]:i for i in model["state_policy"]["scenario"]["issues"]}
                    importance=portrait.get("importance",1);bounds=np.zeros(2)
                    var=portrait["variable"]
                    for term in matching["terms"]:
                        issue=issues[term["dimension"]];before=term["data_evidence"]["before"];after=term["algorithm_evidence"]["after"]
                        metric=issue["metric"]
                        if metric=="negative_fraction":
                            expected=float((output[var].values<0).mean());assert after is not None and np.isclose(after,expected)
                        if metric=="missing_fraction":assert np.isclose(after,1-np.isfinite(output[var]).mean())
                        if metric in {"negative_fraction","missing_fraction"}:
                            initial=np.mean([float((s[var].values<0).mean()) if metric=="negative_fraction" else float(1-np.isfinite(s[var]).mean()) for s in sources.values()])
                            assert np.isclose(before,initial)
                        def risk(q):return float(np.clip((q-issue["acceptable"])/(issue["bad"]-issue["acceptable"]),0,1))
                        raw=[risk(before)-risk(after)]*2 if before is not None and after is not None else [risk(before)-1,risk(before)] if before is not None else [-risk(after),1-risk(after)] if after is not None else [-1,1]
                        weight=issue["importance"]/sum(i["importance"] for i in issues.values())
                        setting=model.get("scoring_parameters",{}).get(term["dimension"],{})
                        weight*=setting.get("multiplier",1)
                        expected=np.array(raw)*weight
                        if not term["definition"]["scoring_enabled"] or term["raw_activation"]<setting.get("activation_threshold",0):expected*=0
                        np.testing.assert_allclose(expected,term["interval"])
                        bounds+=expected
                    np.testing.assert_allclose(bounds,matching["score_interval"])
                    overall+=importance*bounds;total+=importance
                np.testing.assert_allclose(overall/total,row["matching"]["score_interval"])
                valid.append(row)
            assert len(valid)==report["counts"]["usable_observed"]
            assert len(numeric)==report["counts"]["distinct_numeric_products_observed"]
            if report["counts"]["exact_count_complete"]:
                assert len(report["rows"])==protocol["grammar_count"]
                assert report["counts"]["exact_usable_count"]==len(valid)
            if valid:
                expected=min(valid,key=lambda r:(-r["matching"]["score_interval"][0],r["scientific_steps"],r["index"]))
                assert expected["key"]==report["recommended_key"]
                initial=[r for r in valid if r["origin"]=="three_portrait_screen"]
                mutated=[r for r in valid if r["origin"]!="exact_audit"]
                order=lambda r:(-r["matching"]["score_interval"][0],r["scientific_steps"],r["index"])
                initial_score=min(initial,key=order)["matching"]["score_interval"][0] if initial else None
                mutation_best=min(mutated,key=order) if mutated else None
                assert initial_score==report["initial_best_score"]
                assert (mutation_best["key"] if mutation_best else None)==report["mutation_recommended_key"]
                if initial_score is not None:
                    assert np.isclose(report["mutation_improvement"],mutation_best["matching"]["score_interval"][0]-initial_score)
            assert report["exact_audit_additional_executions"]==sum(r["origin"]=="exact_audit" for r in report["rows"])
            assert len({r["index"] for r in report["rows"]})==len(report["rows"])
            for proposal in report["proposals"]:
                assert proposal["parent"]!=proposal["child"] and proposal["old_choice"]!=proposal["new_choice"]
                def decode(index):
                    digits=[]
                    for axis in reversed(protocol["axes"]):
                        digits.append(index%axis["alternatives"]);index//=axis["alternatives"]
                    return list(reversed(digits))
                parent=decode(proposal["parent"]);child=decode(proposal["child"])
                changed={i for i,(a,b) in enumerate(zip(parent,child)) if a!=b}
                recorded=set()
                for edit in proposal["edits"]:
                    index=next(i for i,a in enumerate(protocol["axes"]) if [a["field"],a["window"],a["branch"]]==edit["axis"])
                    assert parent[index]==edit["old_choice"] and child[index]==edit["new_choice"]
                    recorded.add(index)
                assert changed==recorded
                mutations+=1
            counts[folder.name]=report["counts"];decisions+=1
        except Exception as exc:
            failures.append(dict(folder=str(folder),error=str(exc)))
    assert decisions==7 and mutations>0 and not failures,failures
    result=dict(status="PASS",decisions=decisions,products=products,changed_genome_proposals=mutations,counts=counts,failures=failures,
        scope="Independent counts, input/product hashes, direct sign/missing state checks and score arithmetic; no native simulator or scientific superiority claim")
    (OUT/"independent_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(dict(status="PASS",decisions=decisions,products=products,changed_genome_proposals=mutations)))


if __name__=="__main__":main()
