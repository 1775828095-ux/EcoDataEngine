"""Prepare bounded real inputs and preserve the licensed ClimateLearn source."""
import hashlib
import json
import shutil
from pathlib import Path
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/algorithm_library_expansion/v4"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    upstream = ROOT / "output/algorithm_library_expansion/upstream/aditya-grover__climate-learn"
    target = ROOT / "external/climatelearn"
    names = ["LICENSE", "README.md", "source_evidence.json", "src/climate_learn/models/hub/resnet.py",
        "src/climate_learn/models/hub/utils.py", "src/climate_learn/models/hub/components/cnn_blocks.py",
        "src/climate_learn/models/module.py", "src/climate_learn/data/processing/era5_constants.py"]
    for name in names:
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(upstream / name, target / name)
    path = ROOT.parent / "数据/京津冀_PM25融合测试_201908/03_ERA5气象数据/era5_201908_bth.nc"
    with xr.open_dataset(path, engine="h5netcdf") as source:
        ds = source[["t2m"]].isel(valid_time=slice(0, 336), latitude=slice(0, 24), longitude=slice(0, 24)).load()
        ds = ds.rename({"valid_time": "time", "latitude": "lat", "longitude": "lon"}).sortby("lat").sortby("lon")
    ds.attrs.update(crs="EPSG:4326", source_original_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        experiment="Real ERA5 native 0.25deg; derived coarse 1deg input. Not 1deg -> newly validated 30m detail")
    ds.t2m.attrs.update(standard_name="air_temperature", units="K", data_kind="continuous", temporal_semantics="instantaneous")
    ds.to_netcdf(OUT / "era5_bth_t2m_reference.nc", engine="h5netcdf")
    ca = OUT / "era5_california_2hours.nc"
    if not ca.exists():
        url = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
        source = xr.open_zarr(url, chunks=None, storage_options={"token": "anon"})
        # Only two hourly single-variable global chunks are transferred; only the
        # 3x3 California subset is saved, matching the actual USGS DEM footprint.
        small = source[["2m_temperature"]].sel(time=slice("2019-08-01T00:00", "2019-08-01T01:00"),
            latitude=slice(34.5, 34.0), longitude=slice(241.75, 242.25)).load()
        small = small.assign_coords(longitude=small.longitude-360).rename({"2m_temperature": "t2m", "latitude": "lat", "longitude": "lon"}).sortby("lat")
        small.attrs.update(crs="EPSG:4326", source=url, transfer_scope="two hourly global surface-temperature chunks; saved 2x3x3 subset")
        small.t2m.attrs.update(units="K", standard_name="air_temperature", data_kind="continuous")
        small.to_netcdf(ca, engine="h5netcdf")
    geopotential_path = OUT / "era5_california_surface_geopotential.nc"
    if not geopotential_path.exists():
        url = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
        store = xr.open_zarr(url, chunks=None, storage_options={"token": "anon"})
        z = store[["geopotential_at_surface"]].sel(time=slice("2019-08-01T00:00", "2019-08-01T00:00"),
            latitude=slice(34.5, 34.0), longitude=slice(241.75, 242.25)).load()
        assert z.geopotential_at_surface.attrs["units"] == "m**2 s**-2"
        z = z.assign_coords(longitude=z.longitude-360).rename({"latitude": "lat", "longitude": "lon"}).sortby("lat")
        z.attrs.update(crs="EPSG:4326", source=url, transfer_scope="one surface-geopotential hourly chunk")
        z.to_netcdf(geopotential_path, engine="h5netcdf")
    records = []
    for p in (OUT / "era5_bth_t2m_reference.nc", ca, geopotential_path):
        with xr.open_dataset(p, engine="h5netcdf") as d:
            records.append({"path": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                            "dimensions": dict(d.sizes), "attrs": dict(d.attrs)})
    (OUT / "source_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(records, default=str))


if __name__ == "__main__":
    main()
