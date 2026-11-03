"""Source verification, training-only states and outcome-blind mask generation."""
import hashlib,json,sys,os
from pathlib import Path
from time import perf_counter
import joblib,numpy as np,pandas as pd
from sklearn.neighbors import KNeighborsRegressor
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.append(str(ROOT/".venv-golem/Lib/site-packages"))
from ecodata.algorithms.mixed_gap_dag import (SeriesCase,Batch,SealedArtifact,missing_runs,readonly,calendar_features,TASK)
OUT=Path(os.environ.get("ECODATA_BEIJING_MIXED_OUTPUT",str(ROOT/"output/beijing_mixed_gap_20261020")))
DATA=Path("D:/codex/数据/公开数据三层预审_20261018/01_UCI_Beijing")
LENGTHS=[1,3,6,12,24,48,72];MASK_SEEDS=[101,202,303]


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding="utf-8")


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_sources():
    manifest=json.loads((DATA/"source_manifest.json").read_text(encoding="utf-8"));result={};audit=[]
    for item in manifest["files"]:
        file=DATA/item["path"]
        if file.suffix!=".csv":continue
        assert sha(file)==item["sha256"]
        d=pd.read_csv(file);index=pd.to_datetime(d[["year","month","day","hour"]]);station=str(d.station.iloc[0])
        assert not index.duplicated().any() and index.is_monotonic_increasing
        values=pd.to_numeric(d["PM2.5"],errors="coerce").to_numpy(float,copy=True)
        bad=(values<0)|np.isinf(values);values[bad]=np.nan
        y=pd.Series(values,index=pd.DatetimeIndex(index).as_unit("ns"));assert (np.diff(y.index.asi8)==3600*10**9).all()
        result[station]=y
        audit.append(dict(station=station,file=str(file),sha256=sha(file),rows=len(d),natural_missing=int(d["PM2.5"].isna().sum()),
            invalid_negative_or_inf=int(bad.sum()),minimum=float(y.min()),maximum=float(y.max()),duplicates=0,
            unit="ug m-3",unit_evidence="official_original/uci_501_dataset_page.html line PM2.5: concentration (ug/m^3)",
            clock="timezone unspecified; original hourly labels preserved",start=str(y.index.min()),end=str(y.index.max())))
    assert len(result)==12
    pd.DataFrame(audit).to_csv(OUT/"source_audit.csv",index=False,encoding="utf-8-sig")
    return result


def fit(sources):
    if (OUT/"training_states.joblib").exists():return joblib.load(OUT/"training_states.joblib")
    states={};rows=[];start=perf_counter()
    for station,y in sources.items():
        train=y.loc[:"2014-12-31 23:00"].dropna();overall=float(train.median())
        grouped=train.groupby(train.index.hour);count=grouped.count().reindex(range(24),fill_value=0)
        med=grouped.median().reindex(range(24));med[count<30]=overall
        model=KNeighborsRegressor(n_neighbors=24,weights="distance").fit(calendar_features(train.index),train.to_numpy())
        states[station]=dict(medians=readonly(med.to_numpy()),knn=model,peak_threshold=float(train.quantile(.9)),training_end=str(train.index.max()))
        rows.append(dict(station=station,n=len(train),overall_median=overall,minimum_hour_count=int(count.min()),high_threshold=states[station]["peak_threshold"]))
    joblib.dump(states,OUT/"training_states.joblib");pd.DataFrame(rows).to_csv(OUT/"training_statistics.csv",index=False)
    save(OUT/"training_fit_cost.json",dict(seconds=perf_counter()-start,fit_years=[2013,2014],knn_neighbors=24,weights="distance",minimum_hour_samples=30))
    return states


def build(sources,year,states):
    """Private reference table is returned separately from the public batch."""
    cases=[];lengths=[];query=[];targets=[];offsets=[];mask_rows=[];shortages=[];offset=0
    for number,(station,series) in enumerate(sorted(sources.items())):
        y=series.loc[str(year)];values=y.to_numpy();times=y.index
        for replica,seed in enumerate(MASK_SEEDS):
            rng=np.random.default_rng(seed+year*1000+number*100000)
            designs=[]
            valid=np.flatnonzero(np.isfinite(values));positions=np.sort(rng.choice(valid,int(.2*len(valid)),replace=False))
            designs.append(("random","random",values.copy(),times,positions,{}))
            for length in LENGTHS:
                selected=[];blocked=np.zeros(len(y),bool);blocks={}
                for quarter in range(1,5):
                    possible=[i for i in range(1,len(y)-length) if times[i].quarter==quarter and times[i+length-1].quarter==quarter
                        and np.isfinite(values[i-1:i+length+1]).all()]
                    accepted=[]
                    for begin in rng.permutation(possible):
                        if blocked[max(0,begin-24):min(len(y),begin+length+24)].any():continue
                        selected.extend(range(begin,begin+length));accepted.append(int(begin));blocks.update({i:f"{quarter}_{begin}" for i in range(begin,begin+length)})
                        blocked[max(0,begin-24):min(len(y),begin+length+24)]=True
                        if len(accepted)==5:break
                    shortages.append(dict(station=station,year=year,replica=replica,scenario=f"internal_{length}",quarter=quarter,
                        requested=5,accepted=len(accepted),reason="availability only; 24h isolation; absent seasons remain absent"))
                designs.append(("internal",str(length),values.copy(),times,np.array(sorted(selected),int),blocks))
            # Fixed disjoint 168h calendar windows, separate copies per side/length.
            for month in [1,4,7,10]:
                start=pd.Timestamp(year,month,1)+pd.Timedelta(days=replica*14)
                window=pd.date_range(start,periods=168,freq="h")
                if not window.isin(times).all():continue
                block=y.reindex(window).to_numpy()
                for direction in ["start","end"]:
                    for length in [6,24,72]:
                        pos=np.arange(length) if direction=="start" else np.arange(168-length,168)
                        adjacent=length if direction=="start" else 168-length-1
                        accepted=np.isfinite(block[pos]).all() and np.isfinite(block[adjacent])
                        shortages.append(dict(station=station,year=year,replica=replica,scenario=f"boundary_{direction}_{length}",quarter=(month+2)//3,
                            requested=1,accepted=int(accepted),reason="fixed calendar window; raw target and adjacent support must be valid"))
                        if accepted:designs.append(("boundary",f"{direction}_{length}",block.copy(),window,pos,{i:str(start) for i in pos}))
            for big,sub,masked,index,pos,blocks in designs:
                if not len(pos):continue
                truth=masked[pos].copy();masked[pos]=np.nan;missing=np.flatnonzero(~np.isfinite(masked));all_lengths=missing_runs(masked);run_lengths=all_lengths[missing]
                case_id=f"{year}_{station}_{replica}_{big}_{sub}_{len(cases)}"
                assert np.isnan(masked[pos]).all() and np.isfinite(truth).all()
                cases.append(SeriesCase(station,readonly(index.to_numpy()),readonly(masked),readonly(missing)))
                offsets.append((offset,offset+len(missing)));query.extend((offset+np.searchsorted(missing,pos)).tolist());lengths.extend(run_lengths.tolist())
                for p,target in zip(pos,truth):
                    targets.append(dict(case_id=case_id,station=station,replica=replica,big=big,sub=sub,time=index[p],truth=target,
                        block=blocks.get(int(p),str(index[p].floor("7D"))),high=bool(target>=states[station]["peak_threshold"])))
                    mask_rows.append(dict(case_id=case_id,station=station,replica=replica,big=big,sub=sub,time=index[p],
                        actual_visible_gap_length=int(all_lengths[p]),seed=seed))
                offset+=len(missing)
    batch=Batch(tuple(cases),readonly(np.array(lengths)),tuple(offsets))
    frame=pd.DataFrame(targets)
    # Fix all group weights before evaluating any method.
    group_columns=["station","replica","big","sub"]
    group_id,groups=pd.factorize(pd.MultiIndex.from_frame(frame[group_columns]),sort=True)
    group_table=pd.DataFrame(list(groups),columns=group_columns)
    counts=group_table.groupby(["station","replica","big"])["sub"].transform("count")
    weights=1/(len(sources)*3*3*counts.to_numpy(float))
    assert np.isclose(weights.sum(),1),"Every qualified station/replica must have all three major scenarios"
    frame["group_id"]=group_id
    folder=OUT/"data"/str(year);folder.mkdir(parents=True,exist_ok=True)
    frame.to_parquet(folder/"private_targets.parquet",index=False);pd.DataFrame(mask_rows).to_parquet(folder/"masks.parquet",index=False)
    pd.DataFrame(shortages).to_csv(folder/"eligibility.csv",index=False);group_table.to_csv(folder/"groups.csv",index=False)
    joblib.dump(dict(batch=batch,query=readonly(np.array(query)),weights=weights,groups=group_table),folder/"public_inputs.joblib")
    save(folder/"summary.json",dict(cases=len(cases),targets=len(frame),all_missing_positions=len(lengths),groups=len(groups),
        group_weights_sum=float(weights.sum()),year=year,mask_seeds=MASK_SEEDS,truth_in_public_input=False))
    return batch,np.array(query),frame,weights


def load(year,states):
    folder=OUT/"data"/str(year);data=joblib.load(folder/"public_inputs.joblib")
    # Read-only flags are reinstated after deserialization.
    batch=data["batch"]
    for case in batch.cases:
        for a in [case.time,case.values,case.missing]:a.flags.writeable=False
    batch.lengths.flags.writeable=False
    meta=dict(unit="ug m-3",contract=TASK)
    external={"batch":SealedArtifact("masked_batch",batch,meta),"history":SealedArtifact("fitted_history",states,meta),
        "query":SealedArtifact("query_positions",readonly(data["query"]),meta)}
    truth=pd.read_parquet(folder/"private_targets.parquet")
    return external,truth,data["weights"]


if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    sources=read_sources();states=fit(sources)
    for year in [2015,2016]:
        if not (OUT/"data"/str(year)/"summary.json").exists():build(sources,year,states)
    print("PREPARED",flush=True)
