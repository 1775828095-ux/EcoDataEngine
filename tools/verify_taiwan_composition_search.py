"""Audit real-search receipts without rerunning searches or reading raw data."""
import json
from importlib.metadata import version
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from run_taiwan_composition_search import OUT, ROOT, POOL, CONFIG, save, sha, setup, restore, fingerprint


def main():
    protocol=json.loads((OUT/"protocol.json").read_text())
    final=json.loads((OUT/"final_checks.json").read_text())
    _,factory,_=setup()
    results=[]
    sample_keys=None
    initial_keys=set()
    for mode in "ABC":
        for seed in protocol["seeds"]:
            folder=OUT/"search"/f"{mode}_{seed}"
            result=json.loads((folder/"summary.json").read_text())
            records=result["objective_records"]
            assert 0<len(records)<=20 and result["counts"]["proposals"]<=160
            assert result["optimizer_error"] is None and result["gap_completions"]>0
            assert result["test_labels_read"] is False
            assert result["best"]["score"]==result["best"]["RMSE"]
            initial_keys.add(fingerprint(restore(records[0]["graph"]),factory))
            for row in records:
                if not row["real_execution"]: continue
                path=folder/"executions"/row["key"]
                values=pd.read_parquet(path/"predictions.parquet")
                keys=tuple(zip(values.time.astype(str),values.station_id))
                if sample_keys is None: sample_keys=keys
                assert keys==sample_keys,"Different numerical validation cohort/order"
                assert row["netcdf_readback"] and row["coverage"]==1.
                assert row["score"]==row["RMSE"]
                # Recheck actual errors, not just the reported objective.
                rmse=float(np.sqrt(np.mean((values.prediction-values.observed)**2)))
                assert np.isclose(rmse,row["RMSE"],rtol=0,atol=1e-12)
            results.append(dict(mode=mode,seed=seed,callbacks=result["counts"]["objective_callbacks"],
                scored=len(records),real=sum(r["real_execution"] for r in records),
                cache=sum(r["cache_hit"] for r in records),partial=sum(r["status"]=="INCOMPLETE" for r in records),
                gap_completions=result["gap_completions"],proposals=result["counts"]["proposals"]))
    assert len(initial_keys)==1
    # The historical fixed baselines must use the identical target labels,
    # not merely have the same number of rows.
    annual=ROOT/"output/taiwan_annual_working_v1/runs/validation/experiment3"
    historical=pd.concat([pd.read_parquet(annual/f"month_{m:02d}"/"predictions.parquet") for m in range(1,13)],ignore_index=True)
    historical=historical.loc[historical.time.isin(pd.to_datetime(protocol["times"],utc=True))]
    historical["time"]=historical.time.astype("datetime64[ns, UTC]")
    example=next((OUT/"enumeration").glob("*/predictions.parquet"))
    expected_table=pd.read_parquet(example)
    expected_table["time"]=expected_table.time.astype("datetime64[ns, UTC]")
    expected=expected_table.set_index(["time","station_id"])["observed"].sort_index()
    for _,group in historical.groupby("method"):
        actual=group.set_index(["time","station_id"])["observed"].sort_index()
        pd.testing.assert_series_equal(actual,expected)
    assert final["receipt_matches"] and final["frozen_pool_unchanged"] and final["contract_unchanged"]
    full=pd.read_csv(OUT/"full_validation_and_reused_test.csv")
    selected=json.loads((OUT/"search_selection_frozen.json").read_text())["selection"]
    pd.DataFrame(selected)[["mode","seed","key"]].merge(full,on="key",validate="many_to_many").to_csv(
        OUT/"seed_full_validation_and_reused_test.csv",index=False)
    for role in ["validation","test"]:
        reference=None
        for key in full.loc[full.split.eq(role),"key"]:
            values=pd.read_parquet(OUT/"final_frozen"/f"{role}_{key}.parquet")
            cohort=values[["time","station_id","observed"]].reset_index(drop=True)
            if reference is None: reference=cohort
            else: pd.testing.assert_frame_equal(reference,cohort)
    # Audit the source that actually ran this historical experiment. A later
    # integration revision must not erase or silently re-sign old evidence.
    for name,expected in protocol["code_hashes"].items():
        recorded=OUT/"code_snapshot"/name
        assert recorded.exists() and sha(recorded)==expected
    for name,expected in protocol["input_hashes"].items(): assert sha(ROOT/"output/taiwan_annual_working_v1/data"/name)==expected
    tests=ET.parse(ROOT/"output/composition_final_directed_tests.xml").getroot()
    totals={key:sum(int(s.attrib.get(key,0)) for s in tests.iter("testsuite")) for key in ["tests","failures","errors","skipped"]}
    assert totals["failures"]==totals["errors"]==0
    source_files=[ROOT/name for name in protocol["code_hashes"]]+[
        ROOT/"tools/finish_taiwan_composition_search.py",Path(__file__),ROOT/"tests/test_station_fusion_dag.py",
        ROOT/"tools/taiwan_annual_adapter.py",ROOT/"tools/taiwan_working_contract.py",
        ROOT/"tools/run_taiwan_preparation.py"]
    snapshot=OUT/"code_snapshot"
    for path in source_files:
        target=snapshot/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha(target)!=sha(path):
            target=OUT/"code_snapshot_revisions"/sha(path)/path.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): shutil.copy2(path,target)
    pd.DataFrame(results).to_csv(OUT/"execution_accounting.csv",index=False)
    save(OUT/"acceptance_audit.json",dict(status="PASS_ENGINEERING_WITH_RESEARCH_LIMITATIONS",
        same_initial_graph=True,same_ordered_validation_cohort=len(sample_keys),shared_contract_and_pool=True,
        complete_numerical_calls=sum(r["real"] for r in results),scored_callbacks=sum(r["scored"] for r in results),
        cache_hits=sum(r["cache"] for r in results),partial_graph_scores=sum(r["partial"] for r in results),
        optimizer_callbacks_including_deferred=sum(r["callbacks"] for r in results),directed_tests=totals,
        executed_source_sha256=protocol["code_hashes"],
        current_source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_files},
        environment={p:version(p) for p in ["numpy","pandas","xarray","scikit-learn","PyKrige","metpy","h5py"]},
        limitations=["96-hour exploratory search subset, not all full-year validation samples during optimization",
            "14 test stations already evaluated in chapter29; not a fresh blind confirmatory test",
            "21 meaningful graphs: small enough for exhaustive comparison",
            "A/B share typed feasibility; posterior filter rejects zero additional graphs",
            "C uses more proposals/wall time; validation advantage is not equal-compute superiority",
            "Barnes has genuine full-year coverage failures; failed methods not ranked on a smaller cohort",
            "Source historical clock and datum remain working assumptions"]))
    print(json.dumps(dict(tests=totals,real=sum(r["real"] for r in results),scored=sum(r["scored"] for r in results))))


if __name__=="__main__": main()
