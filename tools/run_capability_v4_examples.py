"""Existing Pipeline, actual local trained weights, and real ERA5 plus USGS DEM."""
import hashlib
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
import xarray as xr
from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.algorithms.registry import create_default_registry
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from ecodata.algorithms.router import AlgorithmPlan
from ecodata.core.schemas import EcoDataBundle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/algorithm_library_expansion/v4"


def project_dem(lat, lon):
    path = ROOT / "examples/engineering_case_air_quality/terrain/dem.tif"
    with rasterio.open(path) as src:
        metadata = json.loads(src.tags()["ecodata_metadata"])
        assert metadata["variables"]["elevation"]["units"] == "m"
        target = np.full((len(lat), len(lon)), np.nan, dtype=np.float64)
        dy, dx = float(np.diff(lat).mean()), float(np.diff(lon).mean())
        transform = from_origin(lon[0]-dx/2, lat[-1]+dy/2, dx, dy)
        reproject(rasterio.band(src, 1), target, src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs="EPSG:4326", dst_nodata=np.nan, resampling=Resampling.average)
    assert np.isfinite(target).all()
    return target[::-1].copy()


def execute(context, algorithm, name):
    result = UnifiedAlgorithmPipeline(create_default_registry()).run(context, AlgorithmPlan(name, (algorithm,), 0., ("V4 explicit verification; no GOLEM search",)))
    path = OUT / (name + ".nc")
    result.data.dataset.to_netcdf(path, engine="h5netcdf")
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        assert np.isfinite(ds[context.variable]).all()
        assert ds[context.variable].attrs["units"] == "K"
        assert ds.attrs["crs"] == "EPSG:4326"
        record = {"algorithm": algorithm, "path": path.name, "status": result.report["status"], "report": result.report,
                  "dimensions": dict(ds.sizes), "units": "K", "nan_count": int(ds[context.variable].isnull().sum()),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return result, record


def main():
    records = []
    with xr.open_dataset(OUT / "era5_bth_t2m_coarse.nc", engine="h5netcdf") as ds:
        test = ds.isel(time=slice(288, None)).load()
    context = AlgorithmContext((EcoDataBundle(test),), "t2m", "continuous", model_constraints={"nonnegative": True})
    _, record = execute(context, "downscale.climatelearn_t2m", "climatelearn_pipeline_output")
    records.append(record)
    with xr.open_dataset(OUT / "era5_california_2hours.nc", engine="h5netcdf") as ds:
        source = ds.load()
    with xr.open_dataset(OUT / "era5_california_surface_geopotential.nc", engine="h5netcdf") as z:
        xr.align(source.isel(time=0, drop=True), z.isel(time=0, drop=True), join="exact")
        height = z.geopotential_at_surface.isel(time=0, drop=True).values / 9.80665
    source["elevation_coarse"] = (("lat", "lon"), height)
    source.elevation_coarse.attrs.update(units="m", source="Actual ERA5 model surface geopotential / standard gravity 9.80665 m s-2",
        vertical_reference="ERA5 geopotential height; comparison with USGS orthometric DEM is an approximation, not exact vertical-datum harmonization")
    target_lat, target_lon = np.linspace(34.05, 34.45, 21), np.linspace(-118.2, -117.8, 21)
    target = xr.Dataset({"elevation": (("lat", "lon"), project_dem(target_lat, target_lon))},
                        coords={"lat": target_lat, "lon": target_lon}, attrs={"crs": "EPSG:4326"})
    target.elevation.attrs.update(units="m", source="USGS 3DEP actual DEM; coarsened to about 2km")
    source.to_netcdf(OUT / "fords_real_coarse_input.nc", engine="h5netcdf")
    target.to_netcdf(OUT / "fords_real_target_dem.nc", engine="h5netcdf")
    context = AlgorithmContext((EcoDataBundle(source),), "t2m", "continuous", auxiliaries=(target,),
        parameters={"target_grid": target}, model_constraints={"nonnegative": True})
    original, record = execute(context, "terrain.fords_temperature", "fords_temperature_pipeline_output")
    records.append(record)
    changed = target.copy(deep=True)
    changed.elevation.values[:] += 100
    changed.elevation.attrs["sensitivity_only"] = "Artificial +100m perturbation; not a real input dataset or independent truth"
    perturbed = AlgorithmContext((EcoDataBundle(source),), "t2m", "continuous", auxiliaries=(changed,),
        parameters={"target_grid": changed}, model_constraints={"nonnegative": True})
    result, record = execute(perturbed, "terrain.fords_temperature", "fords_dem_sensitivity_only")
    records.append(record)
    delta = result.data.dataset.t2m.values-original.data.dataset.t2m.values
    assert np.max(np.abs(delta)) > 0, "DEM was not used"
    (OUT / "terrain_sensitivity.json").write_text(json.dumps({"test_only_DEM_offset_m": 100,
        "temperature_response_min_K": float(delta.min()), "temperature_response_max_K": float(delta.max()),
        "explanation": "Official locally estimated lapse rate times elevation change; sign can include local inversion. Not proof of accuracy.",
        "original_DEM_modified": False}, indent=2), encoding="utf-8")
    (OUT / "execution_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{k:r[k] for k in ("algorithm", "status", "dimensions")} for r in records]))


if __name__ == "__main__":
    main()
