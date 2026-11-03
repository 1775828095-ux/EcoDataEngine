"""Official PseudoNetCDF static profile APIs; no fabricated chemical values."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import PseudoNetCDF as pnc
from PseudoNetCDF.cmaqfiles.profile import icon_profile, bcon_profile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("kind",choices=["icon","bcon"])
    parser.add_argument("profile")
    parser.add_argument("griddesc")
    parser.add_argument("output")
    parser.add_argument("config")
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding="utf-8"))
    profile=(icon_profile if args.kind=="icon" else bcon_profile)(args.profile)
    assert cfg["profile_kind"]=="static_CB05_AERO5_reference"
    # The legacy profile reader deliberately omits its commented date parser.
    # Supply explicit project time metadata before its official interpSigma API.
    profile.SDATE=cfg["SDATE"];profile.STIME=cfg["STIME"];profile.TSTEP=0
    profile.createDimension("TSTEP",1)
    profile.createDimension("DATE-TIME",2)
    sigma=np.asarray(profile.VGLVLS)
    grid=pnc.pncopen(args.griddesc,format="griddesc",GDNAM=cfg["grid_name"],FTYPE=1 if args.kind=="icon" else 2,
        VGLVLS=sigma,VGTYP=cfg["VGTYP"],VGTOP=cfg["VGTOP"],SDATE=cfg["SDATE"],STIME=cfg["STIME"],TSTEP=0,nsteps=1,withcf=False)
    if args.kind=="icon":
        out=profile.toGriddedFile(grid)
    else:
        # Same vertical levels: no interpolation is needed. PNC3.4.1 toPerimFile
        # first creates TFLAG during interpSigma, then wrongly treats it as a
        # concentration. Use its public toPerim with explicit metadata supplied
        # to the otherwise-empty output constructor; no boundary math is changed.
        import PseudoNetCDF.cmaqfiles.profile as module
        original=module.ioapi_base
        def configured_empty():
            result=original()
            result.SDATE=cfg["SDATE"];result.STIME=cfg["STIME"];result.TSTEP=0
            return result
        module.ioapi_base=configured_empty
        try:
            out=profile.toPerim(grid.NCOLS,grid.NROWS,1)
        finally:
            module.ioapi_base=original
        for key in grid.ncattrs():
            if key not in {"VAR-LIST","FILEDESC","HISTORY"}:setattr(out,key,getattr(grid,key))
        out.updatemeta()
    out.save(args.output,verbose=0,format="NETCDF3_CLASSIC")
    # Independent netCDF4 reader, not PseudoNetCDF's writer/parser.
    from netCDF4 import Dataset
    with Dataset(args.output) as ds:
        species=[key for key in ds.variables if key not in {"TFLAG","sigma","LAY"}]
        expected=("TSTEP","LAY","ROW","COL") if args.kind=="icon" else ("TSTEP","LAY","PERIM")
        for key in species:
            a=ds.variables[key]
            assert a.dimensions==expected and np.isfinite(a[:]).all() and (a[:]>=0).all()
            assert a.units.strip()==profile.variables[key].units.strip()
        assert len(ds.dimensions["LAY"])==len(profile.dimensions["LAY"])
        assert ds.NROWS==grid.NROWS and ds.NCOLS==grid.NCOLS
        if args.kind=="bcon":assert len(ds.dimensions["PERIM"])==2*(ds.NROWS+ds.NCOLS)+4
        for key in ("GDTYP","P_ALP","P_BET","P_GAM","XORIG","YORIG","XCELL","YCELL","VGTYP","VGTOP","SDATE","STIME","TSTEP"):
            assert np.allclose(ds.getncattr(key),getattr(grid,key))
        record={"kind":args.kind,"backend":"PseudoNetCDF "+pnc.__version__,"dimensions":{k:len(v) for k,v in ds.dimensions.items()},
            "species":species,"units":{k:ds.variables[k].units.strip() for k in species},
            "profile_kind":cfg["profile_kind"],"grid_name":cfg["grid_name"],"config":cfg,
            "profile_sha256":hashlib.sha256(Path(args.profile).read_bytes()).hexdigest(),
            "output_sha256":hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
            "independent_reader":"netCDF4.Dataset", "MODEL_NATIVE_VERIFIED":False,
            "time_semantics":"time-independent static profile, not dynamic meteorological boundary",
            "vertical_metadata_basis":"Explicit library/example project configuration, not measured MCIP metadata"}
    # Check horizontal repetition against the actual official input profile.
    with Dataset(args.output) as ds:
        for key in species:
            if args.kind=="icon":
                expected=np.asarray(profile.variables[key])
                np.testing.assert_allclose(ds.variables[key][0],np.broadcast_to(expected[:,None,None],ds.variables[key].shape[1:]),rtol=1e-6)
            else:
                values=np.asarray(profile.variables[key])
                lengths=[grid.NCOLS+1,grid.NROWS+1,grid.NCOLS+1,grid.NROWS+1]
                start=0
                for side,length in enumerate(lengths):
                    np.testing.assert_allclose(ds.variables[key][0,:,start:start+length],np.broadcast_to(values[:,side,None],(len(sigma)-1,length)),rtol=1e-6)
                    start+=length
    import xarray as xr
    src=xr.Dataset({key:(tuple("d"+str(i) for i in range(np.asarray(profile.variables[key]).ndim)),np.asarray(profile.variables[key])) for key in species})
    for key in species:src[key].attrs["units"]=profile.variables[key].units.strip()
    src.to_netcdf(args.output+".source.nc",engine="h5netcdf")
    Path(args.output+".json").write_text(json.dumps(record,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()),encoding="utf-8")
    print(json.dumps(record,default=str))


if __name__=="__main__":main()
