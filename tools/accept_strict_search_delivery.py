"""Read back final artifacts without fitting, changing selection, or rerunning search."""
import json
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import run_taiwan_composition_search as run
import xarray as xr


if __name__=="__main__":
    out=run.ROOT/"output/taiwan_frozen_pool_search_final"
    protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    assert run.sha(run.POOL)==protocol["pool_sha256"]
    assert run.sha(run.CONFIG)==protocol["contract_sha256"]
    for relative,expected in protocol["code_hashes"].items():
        assert run.sha(run.ROOT/relative)==expected,relative
    for relative,expected in protocol["input_hashes"].items():
        assert run.sha(run.ANNUAL/"data"/relative)==expected,relative
    selection=out/"strict_validation_selection_frozen.json"
    chosen=json.loads(selection.read_text(encoding="utf-8"))["selection"]
    keys={r["key"] for r in chosen}
    folders=list((out/"enumeration").glob("*/predictions.parquet"))
    folders+=list((out/"postfreeze_reused_holdout").glob("*/predictions.parquet"))
    folders+=list((out/"postfreeze_full_validation").glob("month_*/*/predictions.parquet"))
    assert len(folders)==18+len(keys)+12*len(keys)
    cohorts={}; blocked=0
    for path in folders:
        frame=pd.read_parquet(path)
        assert not frame.duplicated(["time","station_id"]).any()
        result=json.loads(path.with_name("result.json").read_text(encoding="utf-8"))
        good=np.isfinite(frame.prediction)
        assert not np.isinf(frame.prediction).any()
        assert (frame.loc[good,"prediction"]>=0).all()
        assert float(good.mean())==result["coverage"]
        assert ("BLOCK" in result["status"])==bool((~good).any())
        blocked+=int("BLOCK" in result["status"])
        assert set(frame.residual_usage.unique()) <= {-2,-1,0,1}
        with h5py.File(path.with_name("predictions.nc")) as ds:
            np.testing.assert_allclose(ds["pm25"][:],frame.prediction,equal_nan=True)
            np.testing.assert_allclose(ds["observed"][:],frame.observed,equal_nan=True)
            np.testing.assert_array_equal(ds["residual_usage"][:],frame.residual_usage)
            np.testing.assert_array_equal(ds["valid_mask"][:],good)
            np.testing.assert_allclose(ds["longitude"][:],frame.longitude)
            np.testing.assert_allclose(ds["latitude"][:],frame.latitude)
            unit=ds["pm25"].attrs["units"]
            assert (unit.decode() if isinstance(unit,bytes) else unit)=="ug m-3"
        with xr.open_dataset(path.with_name("predictions.nc")) as ds:
            np.testing.assert_array_equal(pd.to_datetime(ds.time.values,utc=True),frame.time)
            np.testing.assert_array_equal(ds.station_id.values.astype(str),frame.station_id.astype(str))
            assert ds.attrs["contract"]==run.CONTRACT["version"]
            assert ds.attrs["metric_crs"]=="EPSG:32651"
            assert ds.attrs["grid_coverage"]=="NOT_APPLICABLE_POINT_OUTPUT"
        identity=tuple(zip(frame.time.astype(str),frame.station_id.astype(str),frame.observed))
        group=path.parent.parent
        previous=cohorts.setdefault(str(group),identity)
        assert previous==identity,"Methods evaluated on different target cohorts"
    annual=pd.read_csv(out/"postfreeze_full_validation_metrics.csv")
    assert len(annual)==len(keys) and annual.expected.nunique()==1
    assert not annual.used_for_reselection.any() and not annual.test_labels_read.any()
    holdout=json.loads((out/"holdout_receipts.json").read_text(encoding="utf-8"))
    assert {r["key"] for r in holdout}==keys
    assert all(r["selection_sha256"]==run.sha(selection) for r in holdout)
    evidence=["strict_final_acceptance.json","strict_validation_selection_frozen.json",
        "holdout_plan.json","holdout_receipts.json","postfreeze_full_validation_receipt.json",
        "seed_results.csv","common_budget_comparison.csv","enumeration_results.csv"]
    scripts=["run_strict_random_reference.py","audit_strict_frozen_search.py",
        "run_strict_frozen_holdout.py","verify_strict_full_validation.py",Path(__file__).name]
    run.save(out/"delivery_acceptance.json",dict(
        status="PASS_ENGINEERING_WITH_DISCLOSED_BLOCKED_OUTPUTS_AND_RESEARCH_LIMITS",
        checked_netcdf_files=len(folders),blocked_file_results_preserved=blocked,
        common_cohorts_verified=True,code_and_data_fingerprints_unchanged=True,
        no_test_based_reselection=True,fresh_blind_confirmation=False,
        search_superiority_proven=False,
        versions={p:version(p) for p in ["numpy","pandas","scipy","pykrige","metpy","scikit-learn"]},
        evidence_sha256={p:run.sha(out/p) for p in evidence},
        auxiliary_script_sha256={p:run.sha(run.ROOT/"tools"/p) for p in scripts}))
    print("PASS artifact integrity and equal cohorts; blocked outputs remain blocked",len(folders))
