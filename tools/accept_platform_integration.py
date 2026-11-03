"""Fixed small real-data integration checks; never launches a research batch."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
import numpy as np,pandas as pd,xarray as xr
from scipy.spatial import Delaunay
from pyproj import Transformer
from ecodata.algorithms.platform_search import SearchInput,DatasetReference,EngineeringMetric,submit_search
from ecodata.algorithms.numerical_search import save
from ecodata.interface.search_page import process_search_upload

OUT=ROOT/"output/platform_integration_20261021"
BUDGET=dict(objective_limit=8,proposal_limit=80,callback_limit=150,wall_seconds=30)


def nc_bytes(ds):return ds.to_netcdf(engine="h5netcdf")


def prepare():
    if (OUT/"protocol.json").exists():
        required=["spatial.nc","temporal.nc","observations.csv","query.csv","covariates.csv","validation_reference.nc"]
        if not all((OUT/"inputs"/name).is_file() for name in required):raise ValueError("Frozen acceptance inputs incomplete; preserve evidence and investigate")
        return
    folder=OUT/"inputs";folder.mkdir(parents=True,exist_ok=True)
    tile=ROOT/"output/lghap_reconstruction_20261020_v3/tiles/train/BTH_20190101.nc"
    with xr.open_dataset(tile,engine="h5netcdf") as d:ds=d.isel(lat=slice(0,16),lon=slice(0,16)).load()
    spatial=ds.isel(lat=slice(None,None,2),lon=slice(None,None,2))
    spatial.to_netcdf(folder/"spatial.nc",engine="h5netcdf")
    lat=np.linspace(float(spatial.lat.min()),float(spatial.lat.max()),6);lon=np.linspace(float(spatial.lon.min()),float(spatial.lon.max()),6)
    # User goal is interpolation onto an explicit grid; no HR-truth accuracy claim.
    request=dict(task="spatial",variable="aod",unit="1",semantic_type="continuous",spatial_semantics="cell_mean",nonnegative=True,target_lat=lat.tolist(),target_lon=lon.tolist())
    save(folder/"spatial_request.json",request)
    era=next(Path("D:/codex").glob("*/*/meteorology_era5/anonymous_public/era5_201908_bth.nc"))
    with xr.open_dataset(era,engine="h5netcdf") as d:
        a=d.tp.isel(valid_time=slice(1,25),latitude=10,longitude=10).load().rename(valid_time="time")
    a.attrs.update(data_kind="continuous",temporal_semantics="accumulation")
    temporal=a.to_dataset(name="tp")
    bounds=np.column_stack([a.time.values-np.timedelta64(1,"h"),a.time.values]).astype("datetime64[ns]")
    temporal["time_bounds"]=(("time","bounds"),bounds);temporal.time.attrs["bounds"]="time_bounds"
    temporal.attrs.update(source_file=str(era),interval_evidence="Existing tp.accumulation_comment: 1h ending at valid_time")
    temporal.to_netcdf(folder/"temporal.nc",engine="h5netcdf")
    save(folder/"temporal_request.json",dict(task="temporal",variable="tp",unit="m",semantic_type="continuous",temporal_semantics="accumulation",nonnegative=True,parameters=dict(target_time_frequency="6h")))
    path=ROOT/"output/taiwan_annual_working_v1/data/matched_non_test.parquet"
    frame=pd.read_parquet(path)
    frame=frame.loc[frame.reason.eq("VALID") & frame.aod.notna()].copy()
    # Date and cohort picked ONLY from coverage and geometric support, not PM values.
    tx=Transformer.from_crs("EPSG:4326","EPSG:32651",always_xy=True)
    selected=None
    for stamp,table in frame.groupby("time",sort=True):
        train=table.loc[table.split.eq("train")].sort_values("station_id")
        query=table.loc[table.split.eq("validation")].sort_values("station_id")
        if len(train)<15 or not len(query):continue
        xy=lambda t:np.column_stack(tx.transform(t.longitude.values,t.latitude.values))
        supported=Delaunay(xy(train)).find_simplex(xy(query))>=0
        query=query.loc[supported]
        if len(query)<2:continue
        selected=(stamp,train,query);break
    if selected is None:raise ValueError("No actual station/AOD cohort satisfies original contracts")
    stamp,train,query=selected
    cols=["station_id","longitude","latitude","time","unit"]
    train[cols+["value","split"]].to_csv(folder/"observations.csv",index=False)
    query[cols].to_csv(folder/"query.csv",index=False)
    pd.concat([train,query])[["station_id","time","aod"]].to_csv(folder/"covariates.csv",index=False)
    reference=xr.Dataset({"pm25":("point",query.value.to_numpy())},coords={"station_id":("point",query.station_id.astype(str).to_numpy()),"longitude":("point",query.longitude.to_numpy()),"latitude":("point",query.latitude.to_numpy())})
    reference.pm25.attrs["units"]="ug m-3";reference.to_netcdf(folder/"validation_reference.nc",engine="h5netcdf")
    save(folder/"multisource_request.json",dict(task="multisource",adapter="station_covariates",variable="pm25",unit="ug m-3",covariate_unit="1",semantic_type="continuous",nonnegative=True,nonnegative_action="project",crs="EPSG:4326",metric_crs="EPSG:32651",coordinate_evidence="Frozen Taiwan working convention; author datum unverified",time_semantics="Frozen Asia/Taipei ground [t,t+1h); actual QA AHI scans in same interval; retrospective exploratory",selected_time=str(stamp)))
    save(OUT/"protocol.json",dict(seed=11,budget=BUDGET,selection="first frozen non-test hour with >=15 train and >=2 validation inside hull; no value-based date selection",train_stations=train.station_id.tolist(),validation_stations=query.station_id.tolist(),final_test_read=False,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [tile,path,era]},scope="engineering only, no task-one/formal A/B/C experiment"))


def run_one(task,client=submit_search,suffix="toolkit"):
    folder=OUT/"inputs";request=json.loads((folder/f"{task}_request.json").read_text(encoding="utf-8"))
    if task=="multisource":
        inputs=[SearchInput(name+".csv",name,(folder/(name+".csv")).read_bytes()) for name in ["observations","query","covariates"]]
        with xr.open_dataset(folder/"validation_reference.nc",engine="h5netcdf") as d:evaluation=DatasetReference(d.pm25.load(),"pm25")
    else:
        inputs=[SearchInput(task+".nc","primary",(folder/(task+".nc")).read_bytes())]
        evaluation=EngineeringMetric(request["variable"])
    result=client(inputs,request,evaluation,OUT/"runs"/(task+"_"+suffix),budget=BUDGET,seed=11)
    print(task,suffix,"executions",result["counts"].get("real_executions",0),"best",bool(result["best"]),flush=True)
    if not result["best"]:raise ValueError("No accepted real execution; inspect saved failure evidence")
    return result


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--suffix",default="toolkit")
    parser.add_argument("--client",choices=["toolkit","page"],default="toolkit")
    parser.add_argument("--task",choices=["spatial","temporal","multisource"])
    options=parser.parse_args()
    prepare()
    for task in [options.task] if options.task else ["spatial","temporal","multisource"]:
        run_one(task,process_search_upload if options.client=="page" else submit_search,options.suffix)
