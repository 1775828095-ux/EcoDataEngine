"""Frozen LGHAP degradation benchmark through the existing Registry and Pipeline."""
import os
os.environ["OMP_NUM_THREADS"]="2"
import argparse
from dataclasses import replace,asdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
import numpy as np
import pandas as pd
import xarray as xr
import torch
from ecodata.algorithms.registry import create_default_registry
from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from ecodata.algorithms.router import AlgorithmSelector
from ecodata.algorithms.raster_reconstruction_task import (
    degradation,RasterBaselineAdapter,NonnegativeRaster,CoarseObservationProjection,LocalLIIFRaster,reconstruction_plans,select_raster_validation)
from ecodata.core.schemas import EcoDataBundle
from liif_runtime import load,model_spec

DATA=ROOT/"data/lghap2019_official_subset"
REGIONS={"train":[("BTH",116.3,39.8),("Sichuan",104.2,30.7),("Henan",113.3,34.7)],
    "validation":[("Yangtze",119.8,32.1),("Guanzhong",108.8,34.2),("Hunan",112.7,28.2)],
    "test":[("Pearl",113.3,23.3),("Wuhan",114.2,30.6),("Shandong",117.1,36.6)]}
DATES={"train":["20190101","20190415"],"validation":["20190715","20190815"],"test":["20191015","20191215"]}
SIZE=96;COARSE=[64,48,32]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding="utf-8")


def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    cfg=dict(dataset="LGHAP 2019 AOD v1, derived product reference NOT independent measured truth",
        regions=REGIONS,dates=DATES,tile_size=SIZE,coarse_sizes=COARSE,scales=[1.5,2.,3.],
        degradation="spherical-area overlap weighted cell means, common outer edges, no strict AOD total-mass law",
        target="original 0.01 degree LGHAP centers; nominal 1 km, not constant metric 1 km",
        reference_unit="1",crs="EPSG:4326 per official Python/R demo",missing_value=65535,scale_factor=.01,
        metrics_support="common interior of coarse-center envelope for ALL methods; full output nearest edge prior declared",
        fixed_baseline="raster_task.spatial.bilinear+raster_task.nonnegative",
        validation_selection="per scale pooled equal-tile MSE, all legal plans exhaustive, no test access; tie runtime then ID",
        train=dict(seed=20261020,steps=300,loss="L1",optimizer="Adam",learning_rate=.001,query_samples=1024,
            spec=model_spec(),input="single normalized AOD repeated to 3 channels, official EDSR interface",cpu_threads=2),
        optional_composition="base estimator -> nonnegative declared-coarse-operator QP; not mandatory conservation",
        costs="selection cost + inference reported separately; small candidate space exhausted, no GOLEM advantage claimed",
        source_hashes={str(p):sha(p) for p in DATA.glob("*.nc")},
        numerical_code_hashes={str(p):sha(p) for p in [Path(__file__),ROOT/"src/ecodata/algorithms/raster_reconstruction_task.py",ROOT/"tools/liif_runtime.py"]})
    cfg=json.loads(json.dumps(cfg))  # Compare the persisted JSON types, not tuple versus list.
    if (out/"config.json").exists():
        assert json.loads((out/"config.json").read_text(encoding="utf-8"))==cfg,"Changed frozen configuration requires new output directory"
        return
    save(out/"config.json",cfg)
    entries=[]
    for split in ["train","validation","test"]:
        folder=out/"tiles"/split;folder.mkdir(parents=True,exist_ok=True)
        for date in DATES[split]:
            with xr.open_dataset(DATA/f"LGHAP.AOD.D001.A{date}.nc") as raw:
                for name,lon,lat in REGIONS[split]:
                    j=int(np.argmin(abs(raw.lon.values-lon)));i=int(np.argmin(abs(raw.lat.values-lat)))
                    tile=raw.isel(lon=slice(j-48,j+48),lat=slice(i-48,i+48)).transpose("lat","lon").load().rename(AOD="aod")
                    if tile.aod.shape!=(SIZE,SIZE) or not np.isfinite(tile.aod).all():raise ValueError(f"Incomplete tile {split}/{name}/{date}; no hidden truth filling")
                    tile.attrs.update(crs="EPSG:4326",reference_kind="LGHAP derived fused product, not independent observations",
                        product_date=date,split=split,region=name,source_file=f"LGHAP.AOD.D001.A{date}.nc")
                    tile.aod.attrs.update(units="1",data_kind="continuous",physical_variable="aerosol_optical_depth",spatial_semantics="cell_mean")
                    tile.lat.attrs["units"]="degrees_north";tile.lon.attrs["units"]="degrees_east"
                    path=folder/f"{name}_{date}.nc";tile.to_netcdf(path,engine="h5netcdf")
                    gy,gx=np.gradient(tile.aod.values)
                    entries.append(dict(split=split,region=name,date=date,path=str(path),sha256=sha(path),
                        min=float(tile.aod.min()),max=float(tile.aod.max()),mean=float(tile.aod.mean()),
                        gradient_rms=float(np.sqrt(np.mean(gy**2+gx**2))),finite=tile.aod.size,
                        south=float(tile.lat.min()),north=float(tile.lat.max()),west=float(tile.lon.min()),east=float(tile.lon.max())))
    pd.DataFrame(entries).to_csv(out/"dataset_manifest.csv",index=False)
    # Unit/scale-factor check against raw integers at a fixed original location, without rescaling twice.
    p=DATA/"LGHAP.AOD.D001.A20190101.nc"
    with xr.open_dataset(p,decode_cf=False) as raw,xr.open_dataset(p) as decoded:
        block=raw.AOD.isel(lon=slice(4500,4510),lat=slice(2400,2410)).values
        expected=np.where(block==65535,np.nan,block*.01)
        np.testing.assert_allclose(decoded.AOD.isel(lon=slice(4500,4510),lat=slice(2400,2410)),expected,equal_nan=True)
    save(out/"source_decode_check.json",dict(fill_value=65535,scale_factor=.01,offset=0,decoded_once=True,
        variable_order="raw lon,lat -> standard lat,lon",dimensionless=True))


def train(out):
    path=out/"liif_aod_local.pt"
    if path.exists():return
    evidence=json.loads((ROOT/"external/liif/source_evidence.json").read_text(encoding="utf-8"))
    assert all(sha(ROOT/"external/liif"/f["path"])==f["sha256"] for f in evidence["files"])
    torch.set_num_threads(2);torch.manual_seed(20261020);rng=np.random.default_rng(20261020)
    samples=[xr.load_dataset(p,engine="h5netcdf") for p in sorted((out/"tiles/train").glob("*.nc"))]
    mean=float(np.mean([s.aod.mean() for s in samples]));std=float(np.std(np.stack([s.aod.values for s in samples])))
    if std<=0:raise ValueError("Constant training data")
    prepared=[]
    for tile in samples:
        for n in COARSE:
            coarse,_=degradation(tile,n)
            prepared.append((torch.tensor((coarse.aod.values-mean)/std,dtype=torch.float32)[None,None].repeat(1,3,1,1),
                torch.tensor((tile.aod.values.reshape(-1)-mean)/std,dtype=torch.float32)[None,:,None]))
    models,utils=load();model=models.make(model_spec()).cpu();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    coordinate=utils.make_coord((SIZE,SIZE))[None];cell=torch.ones_like(coordinate)*2/SIZE;log=[];clock=perf_counter()
    for step in range(300):
        inp,target=prepared[int(rng.integers(len(prepared)))];idx=rng.choice(SIZE*SIZE,1024,replace=False)
        optimizer.zero_grad();prediction=model(inp,coordinate[:,idx],cell[:,idx]);loss=torch.nn.functional.l1_loss(prediction,target[:,idx])
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        log.append(dict(step=step+1,loss=float(loss.detach()),seconds=perf_counter()-clock))
    torch.save(dict(spec=model_spec(),state_dict=model.state_dict(),mean=mean,std=std,variable="aod",unit="1"),path)
    pd.DataFrame(log).to_csv(out/"training_log.csv",index=False)
    values=np.concatenate([s.aod.values.ravel() for s in samples]);threshold=float(np.quantile(values,.9))
    save(out/"training_contract.json",dict(official_commit=evidence["commit"],license="BSD-3-Clause",steps=300,
        training_tiles=[str(p) for p in sorted((out/"tiles/train").glob("*.nc"))],source_integrity_checked=True,
        normalization_train_only=dict(mean=mean,std=std),peak_threshold_train_p90=threshold,
        checkpoint_sha256=sha(path),seconds=perf_counter()-clock,origin="LOCAL_AOD_TRAINING_NOT_OFFICIAL_PRETRAINED",
        source_adapter="existing CUDA -> input device replacement only; official neural layers retained"))


def context_for(tile,n):
    coarse,operator=degradation(tile,n)
    return AlgorithmContext((EcoDataBundle(coarse),),"aod","continuous",tuple(tile.lat.values),tuple(tile.lon.values),
        model_constraints={"nonnegative":True,"unit":"1","output_unit":"1"},
        parameters={"coarse_reference":coarse,"degradation_operator":operator}),operator


def registry_for(out):
    registry=create_default_registry()
    for upstream in ["spatial.nearest","spatial.bilinear","spatial.xarray_cubic"]:
        registry.register(RasterBaselineAdapter(registry.get(upstream)))
    for plugin in [NonnegativeRaster(),CoarseObservationProjection(),LocalLIIFRaster(out/"liif_aod_local.pt")]:registry.register(plugin)
    return registry


def evaluate(reference,prediction,context,operator,threshold):
    truth=reference.aod.values;values=prediction.aod.values
    yy,xx=np.meshgrid(reference.lat.values,reference.lon.values,indexing="ij");coarse=context.sources[0].dataset
    mask=(yy>=coarse.lat.min().item())&(yy<=coarse.lat.max().item())&(xx>=coarse.lon.min().item())&(xx<=coarse.lon.max().item())
    if not np.isfinite(values[mask]).all():raise ValueError("Incomplete common reference cohort")
    error=values-truth;e=error[mask]
    gy,gx=np.gradient(truth);py,px=np.gradient(values)
    interior=mask.copy();interior[[0,-1],:]=False;interior[:,[0,-1]]=False
    high=mask&(truth>=threshold)
    exact=np.unravel_index(np.argmax(np.where(mask,truth,-np.inf)),truth.shape)
    peak=np.unravel_index(np.argmax(np.where(mask,values,-np.inf)),values.shape)
    std=truth[mask].std();sst=float(np.sum((truth[mask]-truth[mask].mean())**2))
    return dict(RMSE=float(np.sqrt(np.mean(e**2))),MAE=float(np.mean(abs(e))),Bias=float(e.mean()),
        R2=float(1-np.sum(e**2)/sst) if sst else None,
        spatial_correlation=float(np.corrcoef(truth[mask],values[mask])[0,1]) if std and values[mask].std() else None,
        gradient_RMSE=float(np.sqrt(np.mean(((py-gy)**2+(px-gx)**2)[interior]))),
        high_value_RMSE=float(np.sqrt(np.mean(error[high]**2))) if high.any() else None,high_pixels=int(high.sum()),
        peak_position_error_pixels=float(np.linalg.norm(np.array(exact)-np.array(peak))),
        coarse_consistency_RMSE=float(np.sqrt(np.mean((operator@values.ravel()-coarse.aod.values.ravel())**2))),
        evaluated_pixels=int(mask.sum()),coverage=1.,negative_count=int((values<0).sum()))


def run_stage(out,stage):
    torch.set_num_threads(2);registry=registry_for(out);pipeline=UnifiedAlgorithmPipeline(registry)
    threshold=json.loads((out/"training_contract.json").read_text())["peak_threshold_train_p90"]
    paths=sorted((out/"tiles"/("validation" if stage in {"smoke","validation"} else "test")).glob("*.nc"))
    if stage=="smoke":paths=paths[:1]
    folder=out/stage;folder.mkdir(exist_ok=True);records=[];failures=[];selections={};screen_rows=[]
    for path in paths:
        tile=xr.load_dataset(path,engine="h5netcdf")
        for n in COARSE:
            context,operator=context_for(tile,n);plans,screen=reconstruction_plans(registry,context)
            if stage=="test":
                frozen=json.loads((out/"selected_plans.json").read_text(encoding="utf-8"))
                ids=set(frozen[str(n)].values())|{json.loads((out/"config.json").read_text())["fixed_baseline"]}
                # Test ALL predeclared plans once for transparent comparison, never choose on test.
                assert ids<=set(p.id for p in plans)
            for name,compatibility in screen.items():
                screen_rows.append(dict(tile=path.stem,coarse_size=n,algorithm=name,eligible=compatibility.eligible,
                    reasons="; ".join(compatibility.reasons),category=registry.get(name).spec.category))
            for plan in plans:
                clock=perf_counter();identifier=f"{path.stem}_n{n}_{plan.id.replace('+','__')}"
                try:
                    result=pipeline.run(context,plan)
                    prediction=result.data.dataset
                    score=evaluate(tile,prediction,context,operator,threshold)
                    output=folder/(identifier+".nc");prediction.to_netcdf(output,engine="h5netcdf")
                    with xr.open_dataset(output,engine="h5netcdf") as reread:np.testing.assert_allclose(reread.aod,prediction.aod)
                    save(folder/(identifier+".quality.json"),dict(**result.report,reference_kind="LGHAP product NOT independent ground truth"))
                    records.append(dict(tile=path.stem,region=tile.attrs["region"],date=tile.attrs["product_date"],
                        stage=stage,coarse_size=n,scale=SIZE/n,plan=plan.id,composite=plan.steps[-1].endswith("consistency"),
                        **score,seconds=perf_counter()-clock,quality=result.report["status"],failure=False))
                except Exception as exc:
                    failures.append(dict(tile=path.stem,coarse_size=n,plan=plan.id,error=str(exc),seconds=perf_counter()-clock))
            print(stage,path.stem,n,"executed",len(records),"failures",len(failures),flush=True)
    pd.DataFrame(records).to_csv(folder/"metrics.csv",index=False);save(folder/"failures.json",failures)
    pd.DataFrame(screen_rows).to_csv(folder/"screening.csv",index=False)
    if stage=="smoke":
        assert len(records)==24 and not failures,"Smoke must validate all four bases and optional constraint branches"
        save(out/"smoke_pass.json",dict(plans_executed=len(records),true_arrays=True,readbacks=True))
    if stage=="validation":
        selections=select_raster_validation(records,len(paths))
        save(out/"selected_plans.json",selections)
        save(out/"selection_evidence.json",dict(basis="Validation-only exhaustive actual Pipeline execution, equal tile-weight MSE",
            candidates=8,validation_tiles=len(paths),test_targets_used=False,selection=selections))
    return records


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    p.add_argument("--stage",choices=["prepare","train","smoke","validation","test"],required=True)
    args=p.parse_args();prepare(args.output)
    if args.stage=="train":train(args.output)
    if args.stage in {"smoke","validation","test"}:
        if args.stage=="validation":assert (args.output/"smoke_pass.json").exists()
        if args.stage=="test":assert (args.output/"selected_plans.json").exists()
        run_stage(args.output,args.stage)
