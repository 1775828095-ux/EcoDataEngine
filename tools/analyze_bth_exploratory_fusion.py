"""Summarize frozen real predictions, not another fitting or search stage."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from importlib.metadata import version

import numpy as np
import pandas as pd
import xarray as xr
from run_bth_exploratory_fusion import score,save,sha,ROOT,Data,setup,POOL
from ecodata.evaluation.quality_gates import ResultQualityGate


def verified_source_paths(protocol,original):
    """Recover only path labels by unique pre-fix SHA identity, never alter frozen bytes."""
    recovered=[]
    for recorded,digest in protocol["input_hashes"].items():
        matches=[p for p,s in original["input_hashes"].items() if s==digest]
        if len(matches)!=1:raise ValueError("Ambiguous pre-fix source fingerprint")
        actual=matches[0]
        if sha(actual)!=digest:raise ValueError("Input data changed after freeze")
        recovered.append(dict(recorded_path=recorded,verified_path=actual,sha256=digest,
            path_label_recovered=recorded!=actual))
    return recovered


def selected_exclusions(exclusions,times):
    return exclusions.loc[pd.to_datetime(exclusions.time).isin(pd.to_datetime(times))].copy()


def annotate_grid(grid,diagnostic=False):
    """New delivery file only: make frozen values' time/mask semantics explicit."""
    grid=grid.copy(deep=True)
    if diagnostic:grid=grid.rename({"pm25_estimate":"pm25_rh35_diagnostic"})
    variable="pm25_rh35_diagnostic" if diagnostic else "pm25_estimate"
    times=pd.DatetimeIndex(grid.time.values)
    grid["time_bounds"]=(('time','bounds'),np.column_stack([
        (times-pd.Timedelta(minutes=30)).values,(times+pd.Timedelta(minutes=30)).values]))
    grid.time.attrs.update(standard_name="time",timezone="UTC",bounds="time_bounds")
    for name in ["time","time_bounds"]:
        grid[name].encoding.update(units="seconds since 1970-01-01 00:00:00",calendar="proleptic_gregorian")
    grid.lat.attrs.update(standard_name="latitude",units="degrees_north")
    grid.lon.attrs.update(standard_name="longitude",units="degrees_east")
    grid[variable].attrs.update(units="ug m-3",cell_methods="time: mean",
        physical_definition="GEOS RH35 original lev72" if diagnostic else "statistical estimate of station hourly mean")
    grid.valid_domain.attrs.update(units="1",flag_values=np.array([0,1],dtype="uint8"),flag_meanings="invalid valid")
    grid["time_evaluated"]=("time",grid.valid_domain.values.astype(bool).any(axis=(1,2)).astype("uint8"))
    grid.time_evaluated.attrs.update(units="1",flag_meanings="not_evaluated evaluated")
    grid.attrs.update(Conventions="CF-1.8",crs="EPSG:4326",crs_evidence="USER_WORKING_ASSUMPTION_NOT_GEODETICALLY_VERIFIED",
        time_window="[center UTC -30min, center UTC +30min); station label ending hour",
        mask_note="NaN outside contemporaneous training support or at an unevaluated hour; no fabricated coverage")
    return grid


def main(out):
    protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    selection=json.loads((out/"selection_frozen.json").read_text(encoding="utf-8"))
    assert selection["protocol_sha256"]==sha(out/"protocol.json") and not selection["test_labels_used"]
    archive=out/"pre_utf8_fix/protocol.json"
    original=json.loads(archive.read_text(encoding="utf-8")) if archive.exists() else protocol
    recovered=verified_source_paths(protocol,original)
    save(out/"source_path_encoding_recovery.json",dict(frozen_protocol_unchanged=True,sources=recovered,
        reason="Console transport damaged Chinese path labels during pre-test UTF8 amendment; source SHA identities unchanged"))
    lineage=json.loads((ROOT/"output/bth_pm25_fusion_20261020_final/run_result.json").read_text(encoding="utf-8"))
    assert all(sha(a["path"])==a["sha256"] for a in lineage["source_manifest"])
    _,factory=setup()
    save(out/"algorithm_task_coverage.json",[dict(id=a["id"],
        task_modes=[key[1] for key in factory.adapters if key[0]==a["id"]],
        reason="Executed task mode, see catalog and traces" if any(key[0]==a["id"] for key in factory.adapters) else
        "No compatible task port for this station/GEOS-feature/ERA5 problem; not forced into search")
        for a in json.loads(POOL.read_text(encoding="utf-8"))["algorithms"]])
    assert all(sha(ROOT/p)==s for p,s in protocol["code_hashes"].items())
    rows=[];by_station=[];by_day=[];qualities={};frames={};cohorts=[];grid_stats=[]
    for stage in ["validation","enumeration","sensitivity_nearest","retrospective_test"]:
        comparison=pd.read_csv(out/stage/"comparison.csv");identities=[]
        exclusions=pd.read_csv(out/stage/"geometry_exclusions.csv")
        first_frame=pd.read_parquet(out/stage/comparison.key.iloc[0]/"predictions.parquet")
        exclusions=selected_exclusions(exclusions,first_frame.time.unique())
        exclusions.to_csv(out/stage/"evaluation_window_geometry_exclusions.csv",index=False)
        for row in comparison.itertuples():
            path=out/stage/row.key
            frame=pd.read_parquet(path/"predictions.parquet")
            keys=frame[["station_id","time","truth"]].to_csv(index=False)
            identities.append(hashlib.sha256(keys.encode()).hexdigest())
            calculated=score(frame)
            assert calculated["expected"]==row.expected and np.isclose(calculated["RMSE"],row.RMSE)
            frames[stage,row.method]=frame
            ds=xr.Dataset({"estimate":("sample",frame.prediction.to_numpy())})
            ds.estimate.attrs["units"]="ug m-3"
            q=ResultQualityGate().validate(ds,ds,variable="estimate",nonnegative=True)
            assert q["status"]!="BLOCK"
            q["checks"].append(dict(id="working_coordinate_convention",category="空间",status="WARN",
                message="WGS84 is user-authorized working assumption, not verified datum",value=None,threshold=None,
                locations=[],decision="exploratory_only"))
            q["summary"]["WARN"]+=1
            q.update(status="WARN",decision="exploratory_only",experiment_status="EXPLORATORY_RETROSPECTIVE",
                     coordinate_status="USER_WGS84_ASSUMPTION_NOT_VERIFIED",
                     geos_layer="original lev72 feature, not asserted surface",accuracy_assessed=False,
                     accuracy_note="This gate checks numerical output only; held-out station metrics are in all_method_metrics.csv. No full-field truth.")
            qualities[stage+"/"+row.method]=q
            rows.append(dict(stage=stage,method=row.method,**calculated,seconds=row.seconds,
                point_nonnegative_projections=int(frame.raw_prediction.lt(0).sum()),
                point_and_grid_projections=row.nonnegative_projection_count,
                full_roster_coverage=len(frame)/(len(frame)+len(exclusions)),quality="WARN_EXPLORATORY"))
            frame=frame.assign(date=pd.to_datetime(frame.time).dt.strftime("%Y-%m-%d"))
            for group,dest in [("station_id",by_station),("date",by_day)]:
                for label,part in frame.groupby(group):
                    dest.append(dict(stage=stage,method=row.method,group=label,**score(part)))
            if (path/"grid.nc").exists():
                with xr.open_dataset(path/"grid.nc",engine="h5netcdf") as opened:
                    grid=opened.load();mask=grid.valid_domain.values.astype(bool)
                    assert np.isfinite(grid.pm25_estimate.values[mask]).all()
                    assert np.isnan(grid.pm25_estimate.values[~mask]).all()
                    assert (grid.pm25_estimate.values[mask]>=0).all()
                    annotated=annotate_grid(grid,row.method=="GEOS_RH35_diagnostic")
                    annotated.to_netcdf(path/"grid_cf.nc",engine="h5netcdf")
                    with xr.open_dataset(path/"grid_cf.nc",engine="h5netcdf") as check:
                        xr.testing.assert_allclose(annotated,check.load())
                    grid_stats.append(dict(stage=stage,method=row.method,valid_cells=int(mask.sum()),
                        cells=int(mask.size),native_grid_fraction=float(mask.mean()),
                        evaluated_hours=int(mask.any(axis=(1,2)).sum()),
                        native_fraction_during_evaluated_hours=float(mask.sum()/max(1,mask.any(axis=(1,2)).sum()*mask.shape[1]*mask.shape[2])),
                        readback="PASS",annotated_file=str(path/"grid_cf.nc"),annotated_sha256=sha(path/"grid_cf.nc")))
        assert len(set(identities))==1,"Different method evaluation samples/labels"
        cohorts.append(dict(stage=stage,common_samples=int(comparison.expected.iloc[0]),
            common_sample_sha256=identities[0],geometry_excluded=len(exclusions),
            station_count=frames[stage,comparison.method.iloc[0]].station_id.nunique(),
            time_count=frames[stage,comparison.method.iloc[0]].time.nunique(),
            available_targets=int(comparison.expected.iloc[0])+len(exclusions)))
    table=pd.DataFrame(rows);table.to_csv(out/"all_method_metrics.csv",index=False)
    pd.DataFrame(by_station).to_csv(out/"metrics_by_station.csv",index=False)
    pd.DataFrame(by_day).to_csv(out/"metrics_by_utc_date.csv",index=False)
    pd.DataFrame(cohorts).to_csv(out/"cohort_coverage.csv",index=False)
    pd.DataFrame(grid_stats).to_csv(out/"grid_readback_and_coverage.csv",index=False)
    save(out/"all_result_quality.json",qualities)
    # Prespecified source ablations and strong-baseline differences, not test-driven model selection.
    contrasts=[];leaveout=[];concentration=[]
    pairs=[("UK_both","UK_geos"),("UK_both","UK_era5"),("UK_both","Ordinary_Kriging"),
           ("UK_both","IDW"),("GEOS_affine_OOF_residual_metpy_linear","IDW")]
    for stage in ["validation","retrospective_test"]:
        for candidate,reference in pairs:
            a=frames[stage,candidate];b=frames[stage,reference]
            d=pd.DataFrame(dict(station_id=a.station_id,time=a.time,
                se_a=(a.prediction-a.truth)**2,se_b=(b.prediction-b.truth)**2))
            d["date"]=pd.to_datetime(d.time).dt.strftime("%Y-%m-%d")
            day=d.groupby("date").agg(a=("se_a","sum"),b=("se_b","sum"),n=("se_a","size"))
            rng=np.random.default_rng(20261020);indices=rng.integers(0,len(day),size=(500,len(day)))
            va=day.a.to_numpy()[indices].sum(1);vb=day.b.to_numpy()[indices].sum(1);n=day.n.to_numpy()[indices].sum(1)
            delta=np.sqrt(va/n)-np.sqrt(vb/n)
            rmse_a=float(np.sqrt(d.se_a.mean()));rmse_b=float(np.sqrt(d.se_b.mean()))
            contrasts.append(dict(stage=stage,candidate=candidate,reference=reference,
                RMSE_difference=rmse_a-rmse_b,relative_improvement_percent=100*(rmse_b-rmse_a)/rmse_b,
                day_block_CI_low=float(np.quantile(delta,.025)),day_block_CI_high=float(np.quantile(delta,.975)),
                days=len(day),bootstrap=500,exploratory=True))
            for sid in d.station_id.unique():
                part=d[d.station_id.ne(sid)]
                leaveout.append(dict(stage=stage,candidate=candidate,reference=reference,removed_station=sid,
                    RMSE_difference=float(np.sqrt(part.se_a.mean())-np.sqrt(part.se_b.mean()))))
            for grouping in ["station_id","date"]:
                shares=d.groupby(grouping).se_a.sum()/d.se_a.sum()
                concentration.append(dict(stage=stage,candidate=candidate,grouping=grouping,
                    largest_error_share=float(shares.max()),largest_group=str(shares.idxmax())))
    pd.DataFrame(contrasts).to_csv(out/"source_ablation_and_day_block_CI.csv",index=False)
    pd.DataFrame(leaveout).to_csv(out/"leave_one_station_out_error_diagnostics.csv",index=False)
    pd.DataFrame(concentration).drop_duplicates().to_csv(out/"error_concentration.csv",index=False)
    search=[];events=[]
    test_lookup=table[table.stage.eq("retrospective_test")].set_index("method")
    catalog=json.loads((out/"catalog.json").read_text(encoding="utf-8"))
    for folder in sorted((out/"search").iterdir()):
        result=json.loads((folder/"summary.json").read_text(encoding="utf-8"));c=result["counts"]
        assert len(result["objective_records"])==8 and result["optimizer_error"] is None
        best=result["best"];label=catalog[best["key"]]["label"] if best else None
        search.append(dict(mode=result["mode"],seed=result["seed"],evaluations=len(result["objective_records"]),
            raw_objective_callbacks=c.get("objective_callbacks",0),deferred_callbacks=c.get("deferred_callbacks",0),
            real_executions=c.get("real_graph_evaluations",0),cache_hits=c.get("cache_hits",0),
            proposals=c.get("proposals",0),proposal_rejections=c.get("proposal_rejections",0),
            posterior_rejections=c.get("posterior_rejections",0),numeric_failures=c.get("numeric_failures",0),
            unique_successful_graphs=result["unique_legal_graphs"],gap_completions=result["gap_completions"],
            best_validation_RMSE=best["RMSE"] if best else None,selected_method=label,
            retrospective_test_RMSE=test_lookup.loc[label,"RMSE"] if label else None,wall_seconds=result["wall_seconds"],
            complete_candidate_fraction=sum(r.get("gap",1)==0 for r in result["objective_records"])/8))
        for row in result["objective_records"]:
            events.append(dict(mode=result["mode"],seed=result["seed"],
                **{k:v for k,v in row.items() if not isinstance(v,(list,dict))}))
    pd.DataFrame(search).to_csv(out/"search_comparison.csv",index=False)
    pd.DataFrame(events).to_csv(out/"search_runs.csv",index=False)
    # Behavioral duplicates are measured from real predictions, not node names.
    behavior={}
    for item in catalog.values():
        pred=frames["enumeration",item["label"]].prediction.to_numpy()
        digest=hashlib.sha256(np.round(pred,8).tobytes()).hexdigest()
        behavior.setdefault(digest,[]).append(item["label"])
    save(out/"observed_behavior_groups.json",behavior)
    sensitivity=table[table.stage.isin(["enumeration","sensitivity_nearest"])].pivot(index="method",columns="stage",values="RMSE")
    sensitivity["difference_nearest_minus_linear"]=sensitivity.sensitivity_nearest-sensitivity.enumeration
    sensitivity.to_csv(out/"spatial_matching_sensitivity.csv")
    test_xml=ROOT/"output/bth_exploratory_tests_final.xml"
    suites=list(ET.parse(test_xml).getroot().iter("testsuite"))
    tests={k:sum(int(s.attrib[k]) for s in suites) for k in ["tests","failures","errors","skipped"]}
    assert tests["failures"]==tests["errors"]==0
    data=Data(out);case_list=data.cases()
    save(out/"reproducibility.json",dict(status="EXPLORATORY_REAL_EXECUTION_COMPLETE",tests=tests,
        sources_and_numerical_code_fingerprints_unchanged=True,selection_frozen_before_current_test=True,
        historical_test_previously_evaluated=True,formal_geodetic_accuracy_verified=False,
        data_assumptions=protocol["assumptions"],native_grid_resolution="0.25 degree, no proven finer information",
        registry_pool_sha256=protocol["input_hashes"][str(ROOT/"output/capability_final_acceptance/frozen_generic_pool.json")],
        catalog_graphs=len(catalog),observed_behavior_groups=len(behavior),search_runs=len(search),
        search_scored_objective_calls=sum(r["evaluations"] for r in search),
        search_objective_callbacks=sum(r["raw_objective_callbacks"] for r in search),
        deferred_callbacks=sum(r["deferred_callbacks"] for r in search),
        new_graph_executions=sum(r["real_executions"] for r in search),cache_hits=sum(r["cache_hits"] for r in search),
        contemporaneous_train_station_count=[min(len(c["train"]) for c in case_list),max(len(c["train"]) for c in case_list)],
        protocol_sha256=sha(out/"protocol.json"),selection_sha256=sha(out/"selection_frozen.json"),
        analysis_code_sha256=sha(__file__),
        packages={n:version(n) for n in ["numpy","pandas","scipy","pykrige","xarray","metpy"]},
        test_xml_sha256=sha(test_xml)))
    print(pd.DataFrame(search).to_string(index=False));print("PASS frozen comparisons, masks/readbacks, shared cohorts and source fingerprints")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    main(parser.parse_args().output)
