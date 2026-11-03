"""Controlled generator. Only this module/evaluator knows family or reference fields."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import xarray as xr
from ecodata.algorithms.raster_reconstruction_task import degradation


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def field(seed,family,index):
    rng=np.random.default_rng(seed);x=(np.arange(100)+.5)/100;yy,xx=np.meshgrid(x,x,indexing="ij")
    metadata={}
    if family=="A":
        bx=np.r_[0,np.sort(rng.choice(np.arange(2,9),size=3,replace=False))/10,1]
        by=np.r_[0,np.sort(rng.choice(np.arange(2,9),size=2,replace=False))/10,1]
        levels=rng.uniform(.1,1.9,(len(by)-1,len(bx)-1))
        values=levels[np.searchsorted(by[1:],yy),np.searchsorted(bx[1:],xx)]
        metadata=dict(block_x=bx.tolist(),block_y=by.tolist(),levels=levels.tolist(),identifiability="aligned_blocks")
    else:
        values=.8+rng.uniform(-.25,.25)*(xx-.5)+rng.uniform(-.25,.25)*(yy-.5)
        for frequency,amp in [(1.,.15),(2.,.08),(4.,.04)]:
            values+=rng.uniform(.4,1.2)*amp*np.sin(2*np.pi*(frequency*xx+rng.random()))*np.cos(2*np.pi*(frequency*yy+rng.random()))
        values-=min(0,values.min()-.05)
        if family=="C":
            peaks=[]
            for sigma in [rng.uniform(.035,.08),rng.uniform(.02,.045)]:
                cx,cy=rng.uniform(.15,.85,2);amplitude=rng.uniform(.7,2.)
                values+=amplitude*np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*sigma*sigma))
                peaks.append(dict(x=float(cx),y=float(cy),sigma=float(sigma),amplitude=float(amplitude)))
            metadata=dict(peaks=peaks,identifiability="resolved_broad")
            if index%2:
                # Two adjacent fine cells inside BOTH the 50-grid and 40-grid
                # source-cell intersection. Longitude widths are equal: moving
                # this spike preserves both area-observation operators exactly.
                row=int(rng.integers(15,85));col=int(rng.integers(2,8))*10
                amplitude=float(rng.uniform(3.,5.));values[row,col]+=amplitude
                twin=values.copy();twin[row,col]-=amplitude;twin[row,col+1]+=amplitude
                metadata.update(identifiability="unresolved_subcell_spike",spike=[row,col],spike_amplitude=amplitude,twin=twin)
    ds=xr.Dataset({"field":(("lat","lon"),values)},coords={"lat":30+x,"lon":110+x},attrs={"crs":"EPSG:4326"})
    ds.field.attrs.update(units="1",data_kind="continuous",spatial_semantics="cell_mean",physical_variable="normalized_environmental_scalar")
    return ds,metadata


def generate(root):
    if (root/"protocol.json").exists():raise FileExistsError("Preserve generated protocol; use a new root")
    protocol=dict(reference_shape=[100,100],coarse_shapes=[[50,50],[40,40]],
        counts_per_family=dict(development=12,validation=6,test=12),families=["A","B","C"],seed=20261021,
        aggregation="Declared spherical-area cell overlap; 0..1 degree common footprint; not universal physical conservation",
        mask="all 100x100 finite reference pixels for every candidate",candidates="nearest/bilinear/cubic/PCHIP, each nonnegative or coarse-QP: <=8",
        router="KNN regression of six loss components, k=5, StandardScaler fitted on development only; no family ID or filename",
        indifference_tolerance=.002,main_metric="normalized task score; RMSE/MAE/Bias/gradient/peak/consistency also separate",
        components=["RMSE/std(coarse)","gradient_RMSE/std(coarse)","boundary_band_RMSE/std(coarse)",
            "peak_amplitude_error/std(coarse)","peak_position_error/diagonal","coarse_RMSE/std(coarse)"],
        primary_statistic="paired field-cluster bootstrap, keep both scales together, 2000 draws, percentile 95% intervals; pointwise not multiplicity corrected",
        practical_margin="Report absolute/relative differences; no significance-only innovation claim",test_may_not_tune=True)
    save(root/"protocol.json",protocol)
    records=[];counter=0
    for family_number,family in enumerate(["A","B","C"]):
        for split,count in protocol["counts_per_family"].items():
            for index in range(count):
                seed=20261021+family_number*100000+{"development":0,"validation":10000,"test":20000}[split]+index
                reference,meta=field(seed,family,index)
                token=hashlib.sha256(str(seed).encode()).hexdigest()[:16]
                private=root/"sealed_reference"/f"{token}.nc";private.parent.mkdir(exist_ok=True);reference.to_netcdf(private,engine="h5netcdf")
                twin=meta.pop("twin",None);witness=[]
                if twin is not None:
                    alternate=reference.copy(deep=True);alternate.field.values[:]=twin
                    alternate.to_netcdf(private.with_name(token+"_alternative.nc"),engine="h5netcdf")
                for n in [50,40]:
                    coarse,operator=degradation(reference,n,"field")
                    if twin is not None:
                        error=float(abs(operator@(reference.field.values-twin).ravel()).max())
                        assert error<1e-10;witness.append(dict(coarse_size=n,max_same_input_error=error))
                    public=root/"uploads"/split/f"{token}_{n}.nc";public.parent.mkdir(parents=True,exist_ok=True)
                    coarse.to_netcdf(public,engine="h5netcdf")
                    with xr.open_dataset(public,engine="h5netcdf") as read:np.testing.assert_allclose(read.field,coarse.field)
                    description={"A":"重建空间场并保留区域边界","B":"重建平滑连续场并尊重粗尺度观测","C":"关注局部高值和峰位，尊重观测信息限制"}[family]
                    request=dict(variable="field",target_size=100,description=description,
                        boundary_priority=1. if family=="A" else 0.,gradient_priority=.3,
                        peak_priority=1. if family=="C" else 0.,consistency_priority=1. if family=="B" else .3)
                    request_path=public.with_suffix(".json");save(request_path,request)
                    records.append(dict(field_id=token,family=family,split=split,index=index,seed=seed,coarse_size=n,
                        input=str(public),request=str(request_path),reference=str(private),metadata=meta,witness=witness.copy(),
                        input_sha256=hashlib.sha256(public.read_bytes()).hexdigest()))
                    counter+=1
    save(root/"sealed_reference/manifest.json",records)
    print("Generated",counter,"uploads from 90 independent fields, two scales each")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);generate(p.parse_args().output)
