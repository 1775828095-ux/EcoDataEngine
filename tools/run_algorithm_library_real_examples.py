"""Small reproducible runs using existing real files; no downloads or training."""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.registry import create_default_registry
from ecodata.core.schemas import EcoDataBundle
from ecodata.evaluation.quality_gates import ResultQualityGate

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/algorithm_library_expansion/real_examples"


def save_result(name, context, result):
    path = OUTPUT / (name + ".nc")
    quality = ResultQualityGate().validate(context.sources[0].dataset, result.data.dataset, variable=context.variable,
        semantic_type=context.semantic_type, nonnegative=bool(context.model_constraints.get("nonnegative")))
    assert quality["status"] != "BLOCK", quality
    result.data.dataset.to_netcdf(path, engine="h5netcdf")
    with xr.open_dataset(path, engine="h5netcdf") as reopened:
        quality = ResultQualityGate().validate(context.sources[0].dataset, reopened, variable=context.variable,
                                               semantic_type=context.semantic_type,
                                               nonnegative=bool(context.model_constraints.get("nonnegative")))
        assert quality["status"] != "BLOCK", quality
    report = {"output": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "report": result.report, "quality": quality}
    (OUTPUT / (name + ".json")).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["regrid", "landcover", "pykrige", "metpy", "bias"])
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    registry = create_default_registry()
    path = ROOT / "examples/engineering_case_air_quality/background/pm25_background.nc"
    with xr.open_dataset(path, engine="h5netcdf") as opened:
        source = opened.isel(time=0, drop=True).drop_vars("time_bnds", errors="ignore").load()
    source.PM25.attrs.update(units="ug/m3", data_kind="continuous", spatial_semantics="cell_mean")
    context = AlgorithmContext((EcoDataBundle(source),), "PM25", "continuous",
                               target_lat=tuple(np.linspace(float(source.lat.min()), float(source.lat.max()), 5)),
                               target_lon=tuple(np.linspace(float(source.lon.min()), float(source.lon.max()), 5)))
    if args.stage == "landcover":
        run_landcover(registry)
    elif args.stage == "regrid":
        for name in ["spatial.bilinear", "spatial.nearest", "spatial.xarray_cubic", "spatial.xarray_mean",
                     "spatial.xarray_median", "spatial.xarray_min", "spatial.xarray_max"]:
            if name.endswith("cubic"):
                path = ROOT.parent / "数据/京津冀_PM25融合测试_201908/02_GEOS_CF模拟背景场/geos_cf_pm25_201908_bth.nc"
                with xr.open_dataset(path, engine="h5netcdf") as opened:
                    large = opened.isel(time=0, lev=0, lat=slice(0, 6), lon=slice(0, 6), drop=True).rename({"pm25_rh35_gcc": "PM25"}).load()
                large.attrs["crs"] = "EPSG:4326"
                ctx = replace(context, sources=(EcoDataBundle(large),),
                    target_lat=tuple(np.linspace(float(large.lat.min()), float(large.lat.max()), 11)),
                    target_lon=tuple(np.linspace(float(large.lon.min()), float(large.lon.max()), 11)))
                save_result(name, ctx, registry.get(name).run(ctx))
                continue
            ctx = context
            if name.startswith("spatial.xarray_"):
                ctx = replace(context, target_lat=(34.125, 34.375), target_lon=(-118.125, -117.875))
            save_result(name, ctx, registry.get(name).run(ctx))
    else:
        run_other_stage(args.stage, registry, context)
    print("Completed", args.stage, OUTPUT)


def run_other_stage(stage, registry, context):
    data_root = ROOT.parent / "数据/京津冀_PM25融合测试_201908"
    path = data_root / "01_地面监测站/ground_pm25_201908_60stations.csv"
    table = pd.read_csv(path)
    first_time = table.time_as_stored.min()
    table = table.loc[table.time_as_stored.eq(first_time)].rename(columns={
        "longitude_as_stored": "longitude", "latitude_as_stored": "latitude", "pm25_as_stored": "value"})
    table["unit"] = "ug/m3"
    table = table.dropna(subset=["value"])
    # The existing real GEOS file supplies only target-grid metadata, not interpolation values.
    with xr.open_dataset(data_root / "02_GEOS_CF模拟背景场/geos_cf_pm25_201908_bth.nc", engine="h5netcdf") as opened:
        source = opened.isel(time=0, lev=0, drop=True).rename({"pm25_rh35_gcc": "PM25"}).load()
    source.attrs["crs"] = "EPSG:4326"
    source.PM25.attrs.update(units="ug/m3", data_kind="continuous")
    context = replace(context, sources=(EcoDataBundle(source),), observations=table,
                      target_lat=tuple(np.linspace(float(table.latitude.min()), float(table.latitude.max()), 9)),
                      target_lon=tuple(np.linspace(float(table.longitude.min()), float(table.longitude.max()), 9)),
                      parameters={"observation_crs": "EPSG:4326"})
    if stage in {"pykrige", "metpy"}:
        prefix = "station.pykrige_" if stage == "pykrige" else "station.metpy_"
        for spec in registry.specs():
            if not spec.id.startswith(prefix) or spec.id.endswith(("3d", "regression", "vertical")):
                continue
            result = registry.get(spec.id).run(context)
            result.report.update(real_input=str(path), time_as_stored=first_time,
                                 coordinate_assumption="original coordinates approximately WGS84; datum remains unverified",
                                 purpose="execution test; not algorithm accuracy validation")
            save_result(spec.id, context, result)
        if stage == "metpy":
            run_vertical_example(registry)
    else:
        run_bias_example(registry, data_root)


def run_bias_example(registry, data_root):
    path = data_root / "01_地面监测站/ground_pm25_201908_60stations.csv"
    table = pd.read_csv(path)
    table = table.loc[table.split.eq("train")].copy()
    table["time"] = pd.to_datetime(table.time_as_stored) - pd.Timedelta(hours=8, minutes=30)
    stations = table.drop_duplicates("station_id")
    reference = table.groupby("time").pm25_as_stored.mean().sort_index()
    background_path = data_root / "02_GEOS_CF模拟背景场/geos_cf_pm25_201908_bth.nc"
    with xr.open_dataset(background_path, engine="h5netcdf") as opened:
        field = opened.pm25_rh35_gcc.isel(lev=0, drop=True).interp(
            lat=xr.DataArray(stations.latitude_as_stored.values, dims="station"),
            lon=xr.DataArray(stations.longitude_as_stored.values, dims="station")).mean("station").load()
    common = np.intersect1d(field.time.values, reference.index.values)
    ref = xr.DataArray(reference.reindex(common).values, dims="time", coords={"time": common}, attrs={"units": "ug m-3", "split": "train"})
    hist = field.sel(time=common).assign_attrs(units="ug m-3", split="train")
    before = common < np.datetime64("2019-08-15")
    sim = hist.isel(time=~before).assign_attrs(units="ug m-3")
    ds = sim.to_dataset(name="PM25")
    context = AlgorithmContext((EcoDataBundle(ds),), "PM25", "continuous", model_constraints={"nonnegative": True}, parameters={
        "training_reference": ref.isel(time=before), "historical_simulation": hist.isel(time=before),
        "nquantiles": 20, "adjustment_kind": "*"})
    for method in ("eqm", "dqm", "qdm"):
        result = registry.get("bias.xsdba_" + method).run(context)
        result.report.update(real_input=[str(path), str(background_path)],
            role="40 training-station mean distribution only; no heldout accuracy claim",
            time_assumption="DATATIME UTC+8; preceding-hour mean matched to GEOS-CF HH:30",
            remaining_limitation="original station datum unconfirmed; GEOS RH35 and monitor humidity basis differ")
        save_result("bias.xsdba_" + method, context, result)


def run_vertical_example(registry):
    path = ROOT / "examples/ai_round1/calpuff/upper_air/A_NOAA_IGRA_72357_20260101.csv"
    table = pd.read_csv(path)
    table = table.loc[table.time.eq(table.time.min())].sort_values("pressure_pa")
    table = table.loc[(table.temperature_c > -100) & (table.pressure_pa > 0)]
    ds = xr.Dataset({"temperature": ("pressure", table.temperature_c.values, {"units": "degC"})},
                    coords={"pressure": ("pressure", table.pressure_pa.values, {"units": "Pa"})})
    context = AlgorithmContext((EcoDataBundle(ds),), "temperature", "continuous", parameters={
        "vertical_coordinate": "pressure", "target_levels": [55000., 70000., 85000., 90000.], "log_vertical": True})
    save_result("vertical.metpy", context, registry.get("vertical.metpy").run(context))


def run_landcover(registry):
    import rasterio
    import rasterio.windows
    import os
    os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
    os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif"
    url = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N39E114_Map.tif"
    window = rasterio.windows.Window(24000, 24000, 240, 240)
    with rasterio.open(url) as remote:
        values = remote.read(1, window=window)
        transform = remote.window_transform(window)
        profile = remote.profile.copy()
        profile.update(width=240, height=240, transform=transform, tiled=False)
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
    path = OUTPUT / "ESA_WorldCover_2021_v200_subset.tif"
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values, 1)
    lon = transform.c + (np.arange(240) + .5) * transform.a
    lat = transform.f + (np.arange(240) + .5) * transform.e
    ds = xr.Dataset({"landuse": (("lat", "lon"), values, {"units": "1", "data_kind": "categorical", "spatial_semantics": "category"})},
                    coords={"lat": lat, "lon": lon}, attrs={"crs": "EPSG:4326", "source": url})
    target = xr.Dataset(coords={"lat": [float(np.mean(lat[:120])), float(np.mean(lat[120:]))],
                               "lon": [float(np.mean(lon[:120])), float(np.mean(lon[120:]))]})
    context = AlgorithmContext((EcoDataBundle(ds),), "landuse", "categorical", parameters={"target_grid": target})
    for method in ("most_common", "fractions"):
        result = registry.get("spatial.xarray_" + method).run(context)
        save_result("spatial.xarray_" + method, replace(context, semantic_type="continuous") if method == "fractions" else context, result)
    manifest = {"url": url, "source": "ESA WorldCover 2021 v200 official AWS bucket", "license": "CC-BY-4.0",
                "download": "HTTP Range window only; no full tile", "window": list(window.flatten()),
                "source_crs": "EPSG:4326", "pixel_size_degrees": [transform.a, transform.e],
                "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "scope": "10m nominal WorldCover to coarser geographic cells; exact 30m-to-1km tested separately on synthetic UTM grid"}
    (OUTPUT / "worldcover_source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
