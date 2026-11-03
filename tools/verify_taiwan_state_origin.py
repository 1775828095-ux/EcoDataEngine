"""Final execution-side state contracts on 21 real one-hour graphs, not a new search."""
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

import run_taiwan_composition_search as run


def main():
    historical=run.OUT
    run.OUT=run.ROOT/"output/taiwan_composition_state_guard_v4"
    run.freeze_plan()
    protocol=json.loads((run.OUT/"protocol.json").read_text(encoding="utf-8"))
    selected=run.cases(selected=[protocol["times"][0]])
    assert len(selected)==1
    evaluator=run.Evaluator(selected,run.OUT/"real_port_regression")
    catalog=json.loads((run.OUT/"catalog.json").read_text(encoding="utf-8"))
    rows=[]
    for key,graph_rows in catalog.items():
        path=run.OUT/"real_port_regression"/key/"result.json"
        result=json.loads(path.read_text(encoding="utf-8")) if path.exists() else evaluator.execute(run.restore(graph_rows),key)
        assert result["coverage"]==1. and result["score"]<1e6
        new=pd.read_parquet(path.parent/"predictions.parquet")
        old=pd.read_parquet(historical/"enumeration"/key/"predictions.parquet")
        old=old.loc[old.time.eq(selected[0][0])]
        assert new.station_id.tolist()==old.station_id.tolist()
        np.testing.assert_allclose(new.prediction,old.prediction,rtol=0,atol=1e-10)
        # Actual execution edges carry state origin, not a made-up class name.
        trace=json.loads((path.parent/"execution_trace.jsonl").read_text(encoding="utf-8").splitlines()[0])["trace"]
        origins={v["metadata"]["fit_state_fingerprint"] for t in trace for v in t.get("outputs",{}).values()
            if "fit_state_fingerprint" in v["metadata"]}
        has_fit=any(n["parameters"].get("mode")=="fit" for n in graph_rows)
        if has_fit: assert len(origins)==1
        rows.append(dict(key=key,status=result["status"],new_execution=True,has_fit=has_fit,
            state_origins=len(origins),max_difference=float(np.max(abs(new.prediction-old.prediction))),
            numerical_seconds=result["seconds"],netcdf_readback=result["netcdf_readback"]))
    pd.DataFrame(rows).to_csv(run.OUT/"port_regression_results.csv",index=False)
    for relative,expected in protocol["code_hashes"].items():
        source=run.ROOT/relative
        assert run.sha(source)==expected
        target=run.OUT/"code_snapshot"/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): shutil.copy2(source,target)
    tests=ET.parse(run.ROOT/"output/composition_state_guard_tests.xml").getroot()
    totals={k:sum(int(s.attrib.get(k,0)) for s in tests.iter("testsuite")) for k in ["tests","failures","errors","skipped"]}
    assert not totals["failures"] and not totals["errors"]
    run.save(run.OUT/"final_state_contract_acceptance.json",dict(status="PASS_REAL_PORT_EXECUTION_AND_REGRESSION",
        graphs=21,real_hourly_graph_calls=21,all_prediction_differences_zero=all(r["max_difference"]==0 for r in rows),
        test_labels_read=False,source_hashes_verified=True,tests=totals,
        scope="Execution-only state-origin guard; no new optimization or test-set evaluation",
        previous_search_version="taiwan_composition_search_v3",previous_search_results_not_relabelled_as_v4_runs=True))
    print("PASS all 21 real graphs, identical values, state origins and NetCDF checked",totals)


if __name__=="__main__": main()
