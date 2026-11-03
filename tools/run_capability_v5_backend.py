"""Official numerical backends in the optional V5 process."""
import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kind")
    p.add_argument("source")
    p.add_argument("target")
    p.add_argument("output")
    p.add_argument("options")
    args = p.parse_args()
    options = json.loads(Path(args.options).read_text(encoding="utf-8"))
    source = xr.load_dataset(args.source, engine="h5netcdf")
    target = xr.load_dataset(args.target, engine="h5netcdf")
    var = options["variable"]
    a = source[var]
    report = {"backend": args.kind, "independent_accuracy_proven": False}
    if args.kind == "liif":
        import torch
        from liif_runtime import ROOT, load
        torch.set_num_threads(2)
        models, utils = load()
        state = torch.load(ROOT / "output/algorithm_library_expansion/v5/liif_era5_local.pt", map_location="cpu", weights_only=True)
        model = models.make(state["spec"]).cpu()
        model.load_state_dict(state["state_dict"])
        model.eval()
        h, w = target.sizes["lat"], target.sizes["lon"]
        coord = utils.make_coord((h, w)).unsqueeze(0)
        cell = torch.ones_like(coord)
        cell[:, :, 0] *= 2/h
        cell[:, :, 1] *= 2/w
        chunks = []
        with torch.no_grad():
            for batch in torch.tensor(a.values, dtype=torch.float32).split(8):
                x = ((batch[:, None]-state["mean"])/state["std"]).repeat(1, 3, 1, 1)
                y = model(x, coord.repeat(len(batch), 1, 1), cell.repeat(len(batch), 1, 1))
                chunks.append((y.reshape(-1, h, w)*state["std"]+state["mean"]).numpy())
        output = xr.Dataset({var: (("time", "lat", "lon"), np.concatenate(chunks))}, coords={"time": source.time, "lat": target.lat, "lon": target.lon})
        report.update(official_entry="LIIF.forward/query_rgb", weight_origin="LOCAL_TRAINING", arbitrary_query=True)
    elif args.kind.startswith("pyresample"):
        from pyresample import geometry, kd_tree
        from pyresample.bilinear import NumpyBilinearResampler
        src = geometry.SwathDefinition(lons=source.lon.values, lats=source.lat.values)
        lat, lon = target.lat.values, target.lon.values
        dy, dx = float(np.diff(lat)[0]), float(np.diff(lon)[0])
        area = geometry.AreaDefinition("target", "target", "latlon", {"proj": "longlat", "datum": "WGS84"},
            len(lon), len(lat), (lon[0]-dx/2, lat[0]-dy/2, lon[-1]+dx/2, lat[-1]+dy/2))
        values = a.values if a.ndim == 3 else a.values[None]
        result = []
        for field in values:
            if args.kind.endswith("nearest"):
                y = kd_tree.resample_nearest(src, field, area, radius_of_influence=options["radius_m"], fill_value=np.nan)
            else:
                y = NumpyBilinearResampler(src, area, options["radius_m"], neighbours=16, reduce_data=False).resample(field, fill_value=np.nan)
            result.append(np.asarray(y)[::-1])
        dims = ("time", "lat", "lon") if a.ndim == 3 else ("lat", "lon")
        output = xr.Dataset({var: (dims, np.stack(result) if a.ndim == 3 else result[0])}, coords={"lat": lat, "lon": lon})
        if a.ndim == 3:
            output = output.assign_coords(time=source.time)
        report.update(conservative=False, coverage_fraction=float(np.isfinite(output[var]).mean()))
    elif args.kind == "emiproc_spatial":
        from shapely.geometry import box
        from emiproc.regrid import calculate_weights_mapping, weights_remap
        def polygons(ds):
            dy, dx = float(np.diff(ds.y)[0]), float(np.diff(ds.x)[0])
            return [box(float(x-dx/2), float(y-dy/2), float(x+dx/2), float(y+dy/2)) for y in ds.y.values for x in ds.x.values]
        # Geometry is explicit projected/equal-area grid coordinates; no degree-area mass claim.
        mapping = calculate_weights_mapping(polygons(source), polygons(target))
        values = weights_remap(mapping, a.values.ravel(), (target.sizes["y"], target.sizes["x"]))
        output = xr.Dataset({var: (("y", "x"), values)}, coords={"y": target.y, "x": target.x})
        report.update(input_total=float(a.sum()), output_total=float(values.sum()),
            conservation_relative_error=float(abs(values.sum()-a.values.sum())/max(abs(a.values.sum()), 1e-12)))
    elif args.kind == "emiproc_temporal":
        import pandas as pd
        from emiproc.profiles.temporal.profiles import DailyProfile
        from emiproc.profiles.temporal.operators import create_scaling_factors_time_serie
        times = pd.DatetimeIndex(target.time.values).tz_localize("UTC")
        factors = create_scaling_factors_time_serie(times[0], times[-1], [DailyProfile(np.array(options["ratios"]))])
        # A daily total is split into hourly amounts, not multiplied into hourly rates without normalization.
        ratios = factors.values/24
        output = xr.Dataset({var: (("time",)+a.dims, ratios.reshape((-1,)+(1,)*a.ndim)*a.values)},
            coords={"time": target.time, **{k: source.coords[k] for k in a.dims}})
        report.update(profile_ratio_sum=float(ratios.sum()), input_total=float(a.sum()), output_total=float(output[var].sum()),
            conservation_relative_error=float(abs(output[var].sum()-a.sum())/max(abs(float(a.sum())), 1e-12)))
    else:
        raise ValueError(args.kind)
    output.attrs.update(source.attrs)
    output[var].attrs.update(a.attrs)
    if args.kind == "emiproc_temporal":
        output[var].attrs.update(temporal_semantics="accumulation", time_label="interval_start", interval_duration="1 hour")
        output["time_bounds"] = (("time", "bounds"), np.column_stack([output.time.values, output.time.values+np.timedelta64(1,"h")]))
        output.time.attrs.update(bounds="time_bounds", timezone="UTC")
        output.attrs["daily_total_date_UTC"] = options["daily_total_date"]
    if args.kind.startswith("emiproc"):
        from pyproj import CRS
        unit = CRS.from_user_input(source.attrs["crs"]).axis_info[0].unit_name
        for axis in ("x","y"):
            output[axis].attrs["units"] = unit
    output.to_netcdf(args.output, engine="h5netcdf")
    Path(args.output+".json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
