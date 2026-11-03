"""Paired, shared-day uncertainty and complete-cohort sparse-CV delivery."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from run_bth_sparse_fusion import METHODS, SEEDS, ROOT, ALIGNED, save, sha, metrics


def aggregate(frame, keys):
    return frame.groupby(keys, observed=True).agg(n=("error", "size"),
        squared_error=("squared_error", "sum"), absolute_error=("absolute_error", "sum"),
        error=("error", "sum"), truth=("truth", "sum"), truth2=("truth2", "sum")).reset_index()


def finish(table):
    table=table.copy()
    table["RMSE"]=np.sqrt(table.squared_error/table.n)
    table["MAE"]=table.absolute_error/table.n
    table["Bias"]=table.error/table.n
    denominator=table.truth2-table.truth**2/table.n
    table["R2"]=np.where(denominator>0,1-table.squared_error/denominator,np.nan)
    return table


def paired_uncertainty(daily, comparator="IDW", reps=1000):
    """Same sampled UTC days across all overlapping seed groups; paired methods."""
    rows=[];rng=np.random.default_rng(20261020)
    for (count,cohort),part in daily.groupby(["count","cohort"],sort=True):
        dates=sorted(part.day.unique());seeds=sorted(part.seed.unique())
        if comparator not in set(part.method):continue
        arrays={}
        index=pd.MultiIndex.from_product([seeds,dates],names=["seed","day"])
        for method in sorted(part.method.unique()):
            pivot=part[part.method.eq(method)].set_index(["seed","day"]).reindex(index).fillna(0)
            arrays[method]=pivot[["n","squared_error","absolute_error"]].to_numpy().reshape(len(seeds),len(dates),3)
        day_draw=rng.integers(0,len(dates),(reps,len(dates)))
        seed_draw=rng.integers(0,len(seeds),(reps,len(seeds)))
        baseline=arrays[comparator][:,day_draw,:].sum(2).transpose(1,0,2)
        for method,array in arrays.items():
            if method==comparator:continue
            trial=array[:,day_draw,:].sum(2).transpose(1,0,2)
            row=dict(count=int(count),cohort=cohort,method=method,comparator=comparator,
                resamples=reps,seed_groups=len(seeds),days=len(dates))
            for name,pos,root in [("RMSE",1,True),("MAE",2,False)]:
                base=baseline[:,:,pos]/np.maximum(baseline[:,:,0],1)
                candidate=trial[:,:,pos]/np.maximum(trial[:,:,0],1)
                if root:base=np.sqrt(base);candidate=np.sqrt(candidate)
                improvement=100*(base-candidate)/base
                boot=improvement[np.arange(reps)[:,None],seed_draw].mean(1)
                actual_base=arrays[comparator].sum(1);actual_trial=array.sum(1)
                b=actual_base[:,pos]/actual_base[:,0];c=actual_trial[:,pos]/actual_trial[:,0]
                if root:b=np.sqrt(b);c=np.sqrt(c)
                delta=100*(b-c)/b
                row.update({name+"_improvement_pct_mean":float(delta.mean()),
                    name+"_CI_low":float(np.quantile(boot,.025)),name+"_CI_high":float(np.quantile(boot,.975)),
                    name+"_winning_groups":int((delta>0).sum())})
            row["conditional_stable_advantage"]=row["RMSE_CI_low"]>0 and row["RMSE_winning_groups"]>=8
            rows.append(row)
    return pd.DataFrame(rows)


def analyze(out):
    out=Path(out);config=json.loads((out/"config.json").read_text(encoding="utf-8"))
    assert all(sha(p)==s for p,s in config["input_hashes"].items())
    assert all(sha(p)==s for p,s in config["code_hashes"].items())
    prepared_manifest=json.loads((out/"prepared_integrity.json").read_text(encoding="utf-8"))
    assert sha(out/"train40_features.parquet")==prepared_manifest["features_sha256"]
    combinations=json.loads((out/"station_combinations.json").read_text(encoding="utf-8"))
    tables=[];daily=[];sites=[];distances=[];coverages=[];audits=[];common_hash={};netcdf_count=0
    for count in [5,10,20]:
        for seed in SEEDS:
            folder=out/"run"/f"n{count}_seed{seed}"
            summary=json.loads((folder/"result.json").read_text(encoding="utf-8"))
            assert sha(folder/"predictions.parquet")==summary["predictions_sha256"]
            pred=pd.read_parquet(folder/"predictions.parquet")
            eligible=pd.read_parquet(folder/"eligibility.parquet")
            audit=json.loads((folder/"fit_audit.json").read_text(encoding="utf-8"))
            gates=json.loads((folder/"quality.json").read_text(encoding="utf-8"))
            assert len(gates)==16 and all(q["numeric_gate"]["status"]!="BLOCK" for q in gates)
            assert not set(audit["available"])&set(audit["heldout"])
            for record in audit["records"].values():
                assert set(record["fit_stations"])==set(audit["available"])
                assert all(not set(f["fit_stations"])&set(f["held_stations"]) for f in record["folds"])
            assert not pred.duplicated(["station_id","time"]).any()
            assert set(pred.station_id)<=set(audit["heldout"])
            assert pred[METHODS].notna().all().all(),"Incomplete predictions must not enter paired ranking"
            assert np.isfinite(pred[METHODS].values).all() and pred[METHODS].min().min()>=0
            assert np.isfinite(pred.truth).all()
            combo=next(c for c in combinations if c["count"]==count and c["seed"]==seed)
            control=pred.loc[pred.common_density,["time","station_id","truth"]].sort_values(["time","station_id"])
            control_sha=__import__("hashlib").sha256(control.to_csv(index=False).encode()).hexdigest()
            if seed in common_hash:assert common_hash[seed]==control_sha,"Density target cohort changed"
            common_hash[seed]=control_sha
            coverages.append(dict(count=count,seed=seed,heldout_sites=len(audit["heldout"]),
                all_real_targets=len(eligible),supported_targets=len(pred),geometry_coverage=len(pred)/len(eligible),
                supported_sites=pred.station_id.nunique(),supported_hours=pred.time.nunique(),
                common_density_targets=len(control),hull_area_fraction=combo["hull_area_fraction"],
                group_seconds=summary["seconds"],failures=summary["failures"]))
            table=pd.read_csv(folder/"metrics.csv");assert table.coverage.eq(1).all()
            table["ranking_eligible"]=True;tables.append(table)
            for cohort in ["all_supported","common_density"]:
                subset=pred if cohort=="all_supported" else pred[pred.common_density]
                for method in METHODS:
                    frame=subset[["station_id","time","truth","nearest_available_km",method]].copy()
                    frame["error"]=frame[method]-frame.truth
                    frame["squared_error"]=frame.error**2;frame["absolute_error"]=frame.error.abs();frame["truth2"]=frame.truth**2
                    frame["day"]=frame.time.dt.strftime("%Y-%m-%d")
                    frame["distance_bin"]=pd.cut(frame.nearest_available_km,[0,20,50,np.inf],right=False,labels=["0-20km","20-50km",">=50km"])
                    for target,keys in [(daily,["day"]),(sites,["station_id"]),(distances,["distance_bin"])]:
                        a=aggregate(frame,keys);a["count"]=count;a["seed"]=seed;a["cohort"]=cohort;a["method"]=method;target.append(a)
            if seed==SEEDS[0]:
                nc=pred.set_index(["time","station_id"])[METHODS+["truth","common_density"]].to_xarray()
                for method in METHODS+["truth"]:nc[method].attrs["units"]="ug m-3"
                nc.attrs.update(experiment="Exploratory original-train40 spatial CV, not blind validation",
                    coordinate_basis="WGS84 working assumption, original datum unverified",GEOS="RH35 lev72 feature, surface definition unverified",
                    spatial_domain="only available-station convex-hull support; NaN outside evaluated support")
                nc.to_netcdf(folder/"heldout_predictions.nc",engine="h5netcdf")
                with xr.open_dataset(folder/"heldout_predictions.nc",engine="h5netcdf") as read:
                    for method in METHODS:np.testing.assert_allclose(read[method],nc[method],equal_nan=True)
                netcdf_count+=1
            audits.append(dict(count=count,seed=seed,fit_isolation=True,OOF_site_isolation=True,
                prediction_readback=True,quality_nonnegative_finite=True,density_cohort_sha256=control_sha))
    results=pd.concat(tables,ignore_index=True);results.to_csv(out/"method_metrics.csv",index=False)
    coverage=pd.DataFrame(coverages);coverage.to_csv(out/"coverage_by_group.csv",index=False)
    daily=pd.concat(daily,ignore_index=True);finish(daily).to_csv(out/"metrics_by_UTC_day.csv",index=False)
    finish(pd.concat(sites,ignore_index=True)).to_csv(out/"metrics_by_station.csv",index=False)
    finish(pd.concat(distances,ignore_index=True)).to_csv(out/"metrics_by_distance.csv",index=False)
    summary=results.groupby(["count","cohort","method"])[["RMSE","MAE","Bias","R2","coverage","failure_rate","seconds","negative_projection_count"]].agg(["mean","std"])
    summary.columns=["_".join(c) for c in summary.columns];summary=summary.reset_index()
    summary.to_csv(out/"summary_by_station_count.csv",index=False)
    relative=results.merge(results[results.method.eq("IDW")][["count","seed","cohort","RMSE","MAE"]],on=["count","seed","cohort"],suffixes=("","_IDW"))
    for metric in ["RMSE","MAE"]:relative[metric+"_improvement_pct"]=100*(relative[metric+"_IDW"]-relative[metric])/relative[metric+"_IDW"]
    relative.to_csv(out/"relative_to_IDW.csv",index=False)
    uncertainty=paired_uncertainty(daily);uncertainty.to_csv(out/"paired_uncertainty.csv",index=False)
    source_ablation=[]
    for comparator in ["RF_GEOS","RF_ERA5","RF_GEOS_OOF_mean_residual","RF_ERA5_OOF_mean_residual"]:
        target="RF_BOTH_OOF_mean_residual" if comparator.endswith("residual") else "RF_BOTH"
        compared=paired_uncertainty(daily[daily.method.isin([comparator,target])],comparator)
        source_ablation.append(compared)
    pd.concat(source_ablation,ignore_index=True).to_csv(out/"source_ablation.csv",index=False)
    availability=[dict(request="A",method="IDW",status="EXECUTED",meaning="contemporaneous selected-site observations")]
    availability.append(dict(request="B",method="GEOS_raw",status="NOT_COMPARABLE_NOT_SCORED",
        meaning="RH35 lev72 source definition is not verified equivalent to ground concentration; no direct concentration metrics"))
    for request,method in [("C","GEOS_affine"),("C","RF_GEOS"),("D","RF_ERA5"),("E","RF_BOTH")]:
        availability.append(dict(request=request,method=method,status="EXECUTED",meaning="selected-site history only, retrospective spatial CV"))
    for method in METHODS:
        if method.endswith("residual"):
            availability.append(dict(request="OOF_residual_sensitivity",method=method,status="EXECUTED",meaning="static station-group OOF mean residual IDW"))
    pd.DataFrame(availability).to_csv(out/"method_availability.csv",index=False)
    with xr.open_dataset(ALIGNED,engine="h5netcdf") as ds:
        metadata={k:dict(units=v.attrs.get("units"),attrs=v.attrs,shape=v.shape) for k,v in ds.data_vars.items()}
        time_info=dict(first=str(ds.time.values[0]),last=str(ds.time.values[-1]),hours=ds.sizes["time"])
        prepared=pd.read_parquet(out/"train40_features.parquet")
        sampling_checks=[]
        for _,point in prepared.iloc[np.linspace(0,len(prepared)-1,15).astype(int)].iterrows():
            for feature,variable in {"geos":"pm25_rh35_gcc",**{k:k for k in ["t2m","d2m","u10","v10","sp","tp"]}}.items():
                field=ds[variable].squeeze(drop=True).sel(time=point.time).sortby("lat").sortby("lon")
                lat,lon=field.lat.values,field.lon.values
                i=np.searchsorted(lat,point.latitude)-1;j=np.searchsorted(lon,point.longitude)-1
                assert 0<=i<len(lat)-1 and 0<=j<len(lon)-1
                a=(point.latitude-lat[i])/(lat[i+1]-lat[i]);b=(point.longitude-lon[j])/(lon[j+1]-lon[j])
                v=field.transpose("lat","lon").values
                independent=(1-a)*(1-b)*v[i,j]+a*(1-b)*v[i+1,j]+(1-a)*b*v[i,j+1]+a*b*v[i+1,j+1]
                np.testing.assert_allclose(point[feature],independent,rtol=1e-9,atol=1e-9)
                sampling_checks.append(dict(station=point.station_id,time=str(point.time),feature=feature,
                    sampled=float(point[feature]),independent_four_cell_value=float(independent)))
        save(out/"independent_feature_sampling_checks.json",sampling_checks)
        prepared.describe(include="number").to_csv(out/"input_distribution.csv")
    save(out/"source_variable_metadata.json",metadata)
    save(out/"delivery_quality.json",dict(status="WARN_EXPLORATORY",numeric_status="PASS",
        original_datum_verified=False,original_GE0S_lev72_surface_definition_verified=False,
        raw_GE0S_direct_concentration_comparison="NOT_COMPARABLE_NOT_SCORED",
        gate_scope="sample-array finite/unit/nonnegative checks only; original gate CRS-null PASS is not geodetic validation",
        accuracy_scope="remaining original training stations only; no original validation/test targets accessed",
        geometry_rule="No unsupported extrapolation; excluded genuine targets retained in eligibility denominators",
        method_runtime_scope="RF plain/corrected share state; recorded time is shared fit plus paired predict/correction block, not additive independent runtime"))
    save(out/"integrity_and_isolation.json",dict(groups=30,methods=METHODS,RF_fitted_states=90,RF_forests_including_OOF=540,affine_fits=30,
        seeded_groups=SEEDS,audits=audits,original_validation_and_test_targets_parsed=False,
        raw_inputs_unchanged=True,source_hashes=config["input_hashes"],time=time_info,netcdf_readbacks=netcdf_count,
        prediction_coverage_complete_on_declared_cohort=True,failed_groups=int(coverage.failures.sum()),
        search_runs=0,uncertainty_scope="Conditional paired seed+shared-day bootstrap, not independent external replication"))
    import importlib.metadata
    save(out/"reproducibility.json",dict(input_hashes=config["input_hashes"],
        code_hashes={**config["code_hashes"],str(Path(__file__)):sha(__file__),
            str(ROOT/"tests/test_bth_sparse_fusion.py"):sha(ROOT/"tests/test_bth_sparse_fusion.py")},
        versions={package:importlib.metadata.version(package) for package in ["numpy","pandas","scikit-learn","xarray","scipy","pyproj"]},
        artifacts={str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.is_file() and p.name!="reproducibility.json"},
        contract_config_sha256=sha(out/"config.json"),station_combinations_sha256=sha(out/"station_combinations.json")))
    selected=summary[summary.cohort.eq("all_supported")&summary.method.isin(["IDW","RF_ERA5","RF_BOTH","RF_BOTH_OOF_mean_residual"])]
    print(selected[["count","method","RMSE_mean","RMSE_std","MAE_mean"]].to_string(index=False))
    print(coverage.groupby("count")[["supported_targets","geometry_coverage","common_density_targets","group_seconds"]].mean().to_string())


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    analyze(parser.parse_args().output)
