"""Real-data BTH exploratory fusion. No downloads, no implicit datum verification."""
import os
for _name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_name]="1"
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.append(str(ROOT/".venv-golem/Lib/site-packages"))
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import Delaunay
from ecodata.algorithms.bth_exploratory_dag import BTHFusionDAG,BTHGraphFactory,FEATURES,xy
from ecodata.algorithms.dag import Artifact,DAGError
from ecodata.algorithms.registry import create_default_registry
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from ecodata.algorithms.pm25_alignment import observation_centers
from ecodata.evaluation.quality_gates import RawInputQualityGate
from run_taiwan_composition_search import record,restore,fingerprint,search_job
from golem.core.optimisers.graph import OptGraph
from run_pm25_spatiotemporal_peak_diagnostics import idw_weights

POOL=ROOT/"output/capability_final_acceptance/frozen_generic_pool.json"
ALIGNED=ROOT/"output/bth_pm25_fusion_20261020_final/native_grid_covariates_NOT_FUSION.nc"
GROUND=Path("D:/codex/数据/京津冀_PM25融合测试_201908/01_地面监测站/ground_pm25_201908_60stations.csv")
METADATA=GROUND.parent/"ground_metadata_status.json"
TASK="bth_exploratory_pm25"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding="utf-8")


def setup():
    backend=BTHFusionDAG();registry=create_default_registry()
    frozen=[a["id"] for a in json.loads(POOL.read_text(encoding="utf-8"))["algorithms"]]
    registry.register_task_adapters(TASK,backend.adapters(),algorithm_ids=frozen,
        boundary_ids=("source.train","source.query","prepare.calibrated","quality.station"))
    factory=BTHGraphFactory(registry.task_adapters(TASK),frozen,max_depth=10,max_nodes=24)
    return registry,factory


def score(frame):
    valid=np.isfinite(frame.prediction);err=frame.loc[valid,"prediction"].to_numpy()-frame.loc[valid,"truth"].to_numpy()
    truth=frame.loc[valid,"truth"].to_numpy();ss=float(np.sum((truth-truth.mean())**2)) if len(truth) else 0
    return dict(expected=len(frame),predicted=int(valid.sum()),coverage=float(valid.mean()) if len(frame) else 0.,
        failure_rate=float((~valid).mean()) if len(frame) else 1.,
        RMSE=float(np.sqrt(np.mean(err**2))) if len(err) else None,MAE=float(np.mean(np.abs(err))) if len(err) else None,
        Bias=float(err.mean()) if len(err) else None,R2=float(1-np.sum(err**2)/ss) if ss else None)


class Data:
    def __init__(self,out,sampling="linear",test=False):
        self.out=Path(out);self.sampling=sampling
        self.contract=json.loads((self.out/"protocol.json").read_text(encoding="utf-8"))
        if test and not (self.out/"selection_frozen.json").exists():raise DAGError("Test labels sealed until selection freezes")
        self.station=pd.read_csv(GROUND,usecols=["station_id","longitude_as_stored","latitude_as_stored","split"]).drop_duplicates().rename(
            columns={"longitude_as_stored":"longitude","latitude_as_stored":"latitude"})
        old=json.loads((ROOT/"output/golem_multisource_round2_preparation/preparation_audit.json").read_text(encoding="utf-8"))
        assert dict(zip(self.station.station_id,self.station.split))=={r["station_id"]:r["splits"][0] for r in old["station"]["per_station"]}
        self.labels=[]
        with GROUND.open(encoding="utf-8-sig",newline="") as stream:
            for row in csv.DictReader(stream):
                if row["split"]=="test" and not test:continue
                self.labels.append((row["station_id"],row["time_as_stored"],row["split"],pd.to_numeric(row["pm25_as_stored"],errors="coerce")))
        self.labels=pd.DataFrame(self.labels,columns=["station_id","stored_time","split","value"])
        self.labels["time"]=observation_centers(self.labels.stored_time,json.loads(METADATA.read_text(encoding="utf-8")))
        self.labels=self.labels[np.isfinite(self.labels.value)&self.labels.value.ge(0)]
        with xr.open_dataset(ALIGNED,engine="h5netcdf") as opened:self.grid=opened.load()
        self.times=pd.DatetimeIndex(self.grid.time.values)
        self.names={"geos":"pm25_rh35_gcc",**{n:n for n in FEATURES["era5"]}}
        self.values={}
        for name,var in self.names.items():
            field=self.grid[var].squeeze(drop=True).transpose("time","lat","lon")
            self.values[name]=field.interp(lat=xr.DataArray(self.station.latitude.to_numpy(),dims="station"),
                lon=xr.DataArray(self.station.longitude.to_numpy(),dims="station"),method=sampling).transpose("time","station").values
        lon,lat=np.meshgrid(self.grid.lon,self.grid.lat)
        self.grid_points=pd.DataFrame(dict(station_id=["grid_"+str(i) for i in range(lon.size)],longitude=lon.ravel(),latitude=lat.ravel()))
        self.excluded=[];self._cases={}

    def cases(self,role="validation",selected=None,include_grid=False):
        cache_key=(role,include_grid)
        if cache_key not in self._cases:
            found=[]
            labels=self.labels.set_index(["time","station_id"])
            for ti,time in enumerate(self.times):
                points=self.station.copy()
                for name in self.names:points[name]=self.values[name][ti]
                points["time"]=time;points["unit"]="ug m-3"
                try:values=labels.loc[time,"value"]
                except KeyError:continue
                points["value"]=points.station_id.map(values)
                train=points[points.split.eq("train")&np.isfinite(points.value)].copy()
                if len(train)<15:continue
                hull=Delaunay(xy(train));target=points[points.split.eq(role)&np.isfinite(points.value)].copy()
                supported=hull.find_simplex(xy(target))>=0
                self.excluded.extend(dict(time=str(time),station_id=s,role=role,reason="outside contemporaneous training hull") for s in target.loc[~supported,"station_id"])
                target=target.loc[supported].reset_index(drop=True)
                if target.empty:continue
                query=target.drop(columns=["value","split"]).copy()
                grid_mask=hull.find_simplex(xy(self.grid_points))>=0
                if include_grid:
                    q=self.grid_points.loc[grid_mask].copy()
                    q["time"],q["unit"]=time,"ug m-3"
                    for name,var in self.names.items():q[name]=self.grid[var].isel(time=ti).values.reshape(-1)[grid_mask]
                    query=pd.concat([query,q],ignore_index=True)
                found.append(dict(time=time,train=train.reset_index(drop=True),query=query,target=target,
                                  target_count=len(target),grid_mask=grid_mask,grid_index=ti))
            self._cases[cache_key]=found
        found=self._cases[cache_key]
        if selected is not None:
            wanted=set(pd.to_datetime(selected));found=[c for c in found if c["time"] in wanted]
        return found


class Evaluator:
    def __init__(self,cases,folder,data):
        self.cases=cases;self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True);self.data=data
        registry,self.factory=setup();self.pipeline=UnifiedAlgorithmPipeline(registry)

    def execute(self,graph,key):
        folder=self.folder/key;folder.mkdir(parents=True,exist_ok=True)
        started=perf_counter();frames=[];failures=[];projection=0
        grids=np.full((len(self.data.times),len(self.data.grid.lat),len(self.data.grid.lon)),np.nan,dtype=np.float32)
        masks=np.zeros_like(grids,dtype="uint8")
        with (folder/"trace.jsonl").open("w",encoding="utf-8") as trace_file:
            for case in self.cases:
                train,query=case["train"],case["query"]
                quality=None;trace=[]
                try:
                    if isinstance(graph,str) and graph=="IDW":
                        raw=idw_weights(xy(train)/1000.,xy(query)/1000.)@train.value.to_numpy()
                        values=np.maximum(raw,0)
                    elif isinstance(graph,str) and graph=="GEOS_RH35_diagnostic":
                        raw=query.geos.to_numpy();values=raw.copy()
                    else:
                        meta=dict(unit="ug m-3",metric_crs="EPSG:32650",time=case["time"],
                            contract=self.data.contract["contract_version"],coordinate_status="WGS84_USER_WORKING_ASSUMPTION")
                        external={"train":Artifact("training",train,meta),"query":Artifact("query",query,meta)}
                        result=self.pipeline.run_dag(graph,task_id=TASK,external=external,version=self.data.contract["code_fingerprint"])
                        field=result["outputs"]["field"].value;values=field["prediction"];raw=field["raw_prediction"]
                        quality=field["quality"];trace=result["trace"]
                    if not np.isfinite(values).all():raise DAGError("Nonfinite prediction for fixed common query cohort")
                    projection+=int((raw<0).sum())
                    if len(query)>case["target_count"]:
                        grid=grids[case["grid_index"]].reshape(-1);grid[case["grid_mask"]]=values[case["target_count"]:]
                        masks[case["grid_index"]]=case["grid_mask"].reshape(grids.shape[1:])
                except Exception as exc:
                    failures.append(dict(time=str(case["time"]),error=str(exc)))
                    values=np.full(len(query),np.nan);raw=values
                    trace=getattr(exc,"trace",[])
                frame=case["target"][["station_id","time","value"]].rename(columns={"value":"truth"}).copy()
                frame["prediction"]=values[:case["target_count"]];frame["raw_prediction"]=raw[:case["target_count"]]
                frames.append(frame)
                trace_file.write(json.dumps(dict(time=str(case["time"]),trace=trace,quality=quality),default=str)+"\n")
        frame=pd.concat(frames,ignore_index=True);frame.to_parquet(folder/"predictions.parquet",index=False)
        np.testing.assert_allclose(pd.read_parquet(folder/"predictions.parquet").prediction,frame.prediction,equal_nan=True)
        if any(len(c["query"])>c["target_count"] for c in self.cases):
            ds=xr.Dataset({"pm25_estimate":(("time","lat","lon"),grids),"valid_domain":(("time","lat","lon"),masks)},
                coords={"time":self.data.times,"lat":self.data.grid.lat,"lon":self.data.grid.lon},
                attrs={"experiment":"EXPLORATORY_RETROSPECTIVE","coordinate_assumption":"WGS84, not verified",
                       "GEOS_layer":"lev72 retained as feature; NOT asserted surface","method":key,
                       "role":"RH35 cross-definition diagnostic" if isinstance(graph,str) and graph=="GEOS_RH35_diagnostic" else "station-definition statistical estimate",
                       "grid":"native GEOS 0.25 degree; no verified downscaling"})
            ds.pm25_estimate.attrs.update(units="ug m-3")
            ds.to_netcdf(folder/"grid.nc",engine="h5netcdf")
            with xr.open_dataset(folder/"grid.nc",engine="h5netcdf") as reopened:xr.testing.assert_allclose(ds,reopened.load())
        result=score(frame);result.update(key=key,score=result["RMSE"] if result["coverage"]==1. else 1e6,
            seconds=perf_counter()-started,failed_cases=len(failures),failures=failures,nonnegative_projection_count=projection,
            status="WARN_EXPLORATORY" if result["coverage"]==1. else "BLOCK_INCOMPLETE_COVERAGE",
            graph=record(graph) if not isinstance(graph,str) else graph,netcdf_readback=any(len(c["query"])>c["target_count"] for c in self.cases))
        save(folder/"result.json",result);return result


def graph_label(graph):
    if isinstance(graph,str):return graph
    for node in graph.nodes:
        if node.name=="station.pykrige_universal":return "UK_"+node.parameters["features"]
    if any(n.name=="fusion.quality_aware" for n in graph.nodes):
        correction=[n.name for n in graph.nodes if n.parameters.get("mode")=="residual_interpolate"]
        return "GEOS_affine"+("_OOF_residual_"+correction[0].removeprefix("station.") if correction else "")
    return "Ordinary_Kriging"


def freeze(out):
    out.mkdir(parents=True,exist_ok=True)
    code=[Path(__file__),ROOT/"src/ecodata/algorithms/bth_exploratory_dag.py",ROOT/"tools/run_taiwan_composition_search.py"]
    signature=hashlib.sha256("".join(sha(p) for p in code).encode()).hexdigest()
    if (out/"protocol.json").exists():
        assert json.loads((out/"protocol.json").read_text(encoding="utf-8"))["code_fingerprint"]==signature,"Changed code requires new evidence output"
        return
    save(out/"protocol.json",dict(contract_version="bth_exploratory_v1",created=datetime.now(timezone.utc).isoformat(),
        code_fingerprint=signature,code_hashes={str(p.relative_to(ROOT)):sha(p) for p in code},
        input_hashes={str(p):sha(p) for p in [ALIGNED,GROUND,METADATA,POOL]},
        assumptions=["user-authorized WGS84 interpretation, not geodetic proof","lev72 feature only, original layer metadata retained"],
        retrospective=True,training="40 fixed contemporaneous input stations; station-disjoint 5-fold affine residuals",
        validation="10 fixed validation stations, geometry support only; no target labels in query ports",
        test="10 historical test stations, opened once after selection freeze; NOT a new blind test",
        search_seeds=[11,22,33],evaluation_budget=8,proposal_budget=120,wall_seconds=120,
        primary_sampling="linear",sensitivity_sampling="nearest",search_time_rule="24 evenly spaced eligible windows, availability/geometry only",
        future_prediction_claim=False,full_grid_truth=False))
    data=Data(out);cases=data.cases()
    times=[str(c["time"]) for c in cases];selected=[times[i] for i in np.unique(np.linspace(0,len(times)-1,24).astype(int))]
    protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"));protocol.update(times=selected,full_validation_times=len(times))
    save(out/"protocol.json",protocol)
    _,factory=setup();catalog={}
    for _,recipe in factory.recipes("validated_field"):
        graph=OptGraph([factory.instantiate(recipe)]);catalog[fingerprint(graph,factory)]={"label":graph_label(graph),"graph":record(graph)}
    save(out/"catalog.json",catalog)
    data.station.to_csv(out/"fixed_station_split.csv",index=False)


def evaluate_set(data,folder,role="validation",times=None,include_grid=True):
    catalog=json.loads((data.out/"catalog.json").read_text(encoding="utf-8"));cases=data.cases(role,times,include_grid)
    evaluator=Evaluator(cases,folder,data);rows=[]
    for key,item in list(catalog.items())+[("IDW",{"label":"IDW","graph":"IDW"}),("GEOS_RH35_diagnostic",{"label":"GEOS_RH35_diagnostic","graph":"GEOS_RH35_diagnostic"})]:
        path=Path(folder)/key/"result.json"
        graph=item["graph"] if isinstance(item["graph"],str) else restore(item["graph"])
        result=json.loads(path.read_text(encoding="utf-8")) if path.exists() else evaluator.execute(graph,key)
        rows.append(dict(method=item["label"],**{k:v for k,v in result.items() if not isinstance(v,(list,dict))}))
        print(Path(folder).name,item["label"],result["RMSE"],result["coverage"],flush=True)
    pd.DataFrame(rows).to_csv(Path(folder)/"comparison.csv",index=False)
    pd.DataFrame(data.excluded).to_csv(Path(folder)/"geometry_exclusions.csv",index=False)
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage",choices=["freeze","smoke","validation","search","test","sensitivity"],required=True)
    args=parser.parse_args();out=args.output;freeze(out)
    protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"));data=Data(out)
    if args.stage=="freeze":return
    if args.stage=="smoke":evaluate_set(data,out/"smoke",times=protocol["times"][:2]);return
    if args.stage=="validation":evaluate_set(data,out/"validation");return
    if args.stage=="sensitivity":
        evaluate_set(Data(out,sampling="nearest"),out/"sensitivity_nearest",times=protocol["times"])
        return
    if args.stage=="search":
        evaluate_set(data,out/"enumeration",times=protocol["times"],include_grid=False)
        rows=[]
        for seed in protocol["search_seeds"]:
            for mode in ["A","R","C"]:
                rows.append(search_job(mode,seed,budget=8,output=out,
                    evaluator_factory=lambda cases,folder:Evaluator(cases,folder,data),
                    case_provider=lambda selected:data.cases(selected=selected)))
        save(out/"search_summary.json",rows)
        chosen=[]
        for row in rows:
            summary=json.loads((out/"search"/f"{row['mode']}_{row['seed']}"/"summary.json").read_text(encoding="utf-8"))
            if summary["best"]:chosen.append(dict(mode=row["mode"],seed=row["seed"],key=summary["best"]["key"],graph=summary["best"]["graph"]))
        fixed=pd.read_csv(out/"enumeration/comparison.csv")
        fixed=fixed[fixed.coverage.eq(1)&~fixed.method.eq("GEOS_RH35_diagnostic")]
        save(out/"selection_frozen.json",dict(at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(out/"protocol.json"),
            selections=chosen,best_fixed=fixed.loc[fixed.RMSE.idxmin(),["method","key"]].to_dict(),test_labels_used=False))
        return
    if args.stage=="test":
        if not (out/"selection_frozen.json").exists():raise DAGError("Search/fixed selection not frozen")
        save(out/"retrospective_test_opened.json",dict(selection_sha256=sha(out/"selection_frozen.json"),status="RETROSPECTIVE_NOT_NEW_BLIND_TEST"))
        evaluate_set(Data(out,test=True),out/"retrospective_test",role="test")


if __name__=="__main__":main()
