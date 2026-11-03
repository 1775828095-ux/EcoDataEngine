"""Real ERA5/DEM and upstream emission test fixtures through existing quality gates."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
import xarray as xr

from ecodata.algorithms.registry import create_default_registry
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from ecodata.algorithms.router import AlgorithmPlan
from ecodata.algorithms.contracts import AlgorithmContext
from ecodata.core.schemas import EcoDataBundle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/algorithm_library_expansion/v5"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    def run(kind, source, target, parameters=None, semantic="continuous", label=None):
        algorithm = "v5."+kind
        context = AlgorithmContext((EcoDataBundle(source),), "value", semantic,
            parameters={"target_grid": target, **(parameters or {})})
        result = UnifiedAlgorithmPipeline(create_default_registry()).run(context,
            AlgorithmPlan(label or algorithm, (algorithm,), 0., ("V5 explicit official-backend execution",)))
        name = (label or kind)+".nc"
        result.data.dataset.to_netcdf(OUT/name, engine="h5netcdf")
        with xr.open_dataset(OUT/name, engine="h5netcdf") as output:
            assert np.isfinite(output.value).any()
            assert output.value.attrs["units"] == source.value.attrs["units"]
            record = {"algorithm": algorithm, "file": name, "status": result.report["status"],
                "sha256": hashlib.sha256((OUT/name).read_bytes()).hexdigest(), "dimensions": dict(output.sizes),
                "quality": result.report, "data_kind": source.attrs["data_kind"]}
        records.append(record)
        return result.data.dataset
    with xr.open_dataset(ROOT/"output/algorithm_library_expansion/v4/era5_bth_t2m_coarse.nc", engine="h5netcdf") as d:
        source = d.isel(time=slice(288, None)).load().rename(t2m="value")
    source.attrs["data_kind"] = "real_ERA5_derived_coarse"
    contract = json.loads((OUT/"liif_training.json").read_text())
    comparison = []
    with xr.open_dataset(ROOT/"output/algorithm_library_expansion/v4/era5_bth_t2m_reference.nc", engine="h5netcdf") as d:
        reference = d.t2m.isel(time=slice(288, None)).load()
    import torch
    import torch.nn.functional as F
    for scale in (1.5, 2, 3, 4):
        n = int(6*scale)
        axes = {}
        for axis, (lo, hi) in zip(("lat", "lon"), (contract["bounds"][:2], contract["bounds"][2:])):
            axes[axis] = np.linspace(lo+(hi-lo)/n/2, hi-(hi-lo)/n/2, n)
        target = xr.Dataset(coords=axes, attrs={"crs": "EPSG:4326"})
        output = run("liif", source, target, label="liif_"+str(scale).replace(".", "_")+"x")
        truth = F.interpolate(torch.tensor(reference.values[:, None]), size=(n, n), mode="area").numpy()[:, 0]
        baseline = F.interpolate(torch.tensor(source.value.values[:, None]), size=(n, n), mode="bilinear", align_corners=False).numpy()[:, 0]
        comparison.append({"scale": scale, "liif_reference_rmse_K": float(np.sqrt(np.mean((output.value.values-truth)**2))),
            "bilinear_reference_rmse_K": float(np.sqrt(np.mean((baseline-truth)**2))), "independent_truth": False})
    (OUT/"liif_reference_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    dem_path = ROOT/"examples/engineering_case_air_quality/terrain/dem.tif"
    with rasterio.open(dem_path) as dem:
        values = dem.read(1).astype(float)
        rows, cols = np.indices(values.shape)
        x, y = rasterio.transform.xy(dem.transform, rows, cols)
        lon, lat = Transformer.from_crs(dem.crs, 4326, always_xy=True).transform(np.array(x).reshape(values.shape), np.array(y).reshape(values.shape))
    swath = xr.Dataset({"value": (("row", "col"), values)}, coords={"lon": (("row", "col"), lon), "lat": (("row", "col"), lat)},
        attrs={"crs": "EPSG:4326", "data_kind": "real_USGS_DEM_pixel_geolocation", "source_sha256": hashlib.sha256(dem_path.read_bytes()).hexdigest()})
    swath.value.attrs.update(units="m", standard_name="surface_altitude")
    swath.to_netcdf(OUT/"real_dem_2d_geolocation.nc", engine="h5netcdf")
    target = xr.Dataset(coords={"lat": np.linspace(34.05, 34.45, 20), "lon": np.linspace(-118.2, -117.8, 20)}, attrs={"crs": "EPSG:4326"})
    for kind in ("pyresample_nearest", "pyresample_bilinear"):
        run(kind, swath, target, {"radius_m": 5000})
    yy, xx = np.mgrid[0:21, 0:21]
    curved_lat = 30+yy*.1+.01*np.sin(xx)
    curved_lon = 100+xx*.1+.01*np.cos(yy)
    curved = xr.Dataset({"value": (("row", "col"), curved_lat+curved_lon)},
        coords={"lat": (("row", "col"), curved_lat), "lon": (("row", "col"), curved_lon)},
        attrs={"crs": "EPSG:4326", "data_kind": "synthetic_nonrectilinear_geometry_test_not_environmental_truth"})
    curved.value.attrs["units"] = "K"
    grid = xr.Dataset(coords={"lat": np.linspace(30.3,31.7,10), "lon": np.linspace(100.3,101.7,10)}, attrs={"crs":"EPSG:4326"})
    for kind in ("pyresample_nearest", "pyresample_bilinear"):
        result = run(kind, curved, grid, {"radius_m":50000}, label=kind+"_curved_geometry")
        assert np.isfinite(result.value).all()
        assert float(abs(result.value-(grid.lat+grid.lon)).max()) < .25
    # Exact official emiproc tests_utils fixture; these are format/numerical tests,
    # not a measured emissions inventory. Metadata says so explicitly.
    source = xr.Dataset({"value": (("y", "x"), [[3., 5.], [4., 6.]])}, coords={"y": [.5, 1.5], "x": [.5, 1.5]},
        attrs={"crs": "EPSG:6933", "data_kind": "synthetic_official_emiproc_test_fixture_subset", "source": "emiproc tests_utils/test_inventories.py inv.gdf adf/CH4 first four cells; CRS/units are explicit numerical-test configuration, not real inventory metadata"})
    source.value.attrs.update(units="kg", spatial_semantics="cell_total", variable_role="emissions")
    target = xr.Dataset(coords={"y": [.25, .75, 1.25, 1.75], "x": [.25, .75, 1.25, 1.75]}, attrs={"crs": "EPSG:6933"})
    run("emiproc_spatial", source, target, {"model_preprocessing": True}, "cell_total")
    hours = xr.Dataset(coords={"time": pd.date_range("2019-08-01", periods=24, freq="h")}, attrs={"crs": "EPSG:6933"})
    ratios = (np.arange(1, 25)/300).tolist()
    run("emiproc_temporal", source, hours, {"model_preprocessing": True, "ratios": ratios, "time_basis": "UTC", "daily_total_date": "2019-08-01"}, "cell_total")
    from ecodata.algorithms.soil_spline import SoilSplinePlugin
    profile = pd.read_csv(OUT/"real_honcut_profile.csv")
    soil = xr.Dataset({"value": ("horizon", profile.field_ph), "upper": ("horizon", profile.top), "lower": ("horizon", profile.bottom)},
        attrs={"data_kind": "real_Honcut_Creek_field_measured_pH", "source": "aqp::sp2; USGS Bulletin 1590-G (1989)"})
    soil.value.attrs["units"] = "pH"
    soil.upper.attrs["units"] = soil.lower.attrs["units"] = "cm"
    context = AlgorithmContext((EcoDataBundle(soil),), "value", "continuous", parameters={"depth_boundaries_cm": [0,5,15,30,60], "physical_range": [0,14], "allow_missing_horizons": True})
    algorithm = "soil.mpspline2"
    result = UnifiedAlgorithmPipeline(create_default_registry()).run(context, AlgorithmPlan(algorithm, (algorithm,), 0., ("Real Honcut field pH, confirmed missing horizons",)))
    name = "soil_honcut_pH.nc"
    result.data.dataset.to_netcdf(OUT/name, engine="h5netcdf")
    with xr.open_dataset(OUT/name, engine="h5netcdf") as ds:
        assert np.isfinite(ds.value).all() and ds.value.attrs["units"] == "pH"
    records.append({"algorithm": algorithm, "file": name, "status": result.report["status"],
        "sha256": hashlib.sha256((OUT/name).read_bytes()).hexdigest(), "quality": result.report, "data_kind": soil.attrs["data_kind"]})
    (OUT/"execution_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps([{k:r[k] for k in ("algorithm", "file", "status", "data_kind")} for r in records]))


if __name__ == "__main__":
    main()
