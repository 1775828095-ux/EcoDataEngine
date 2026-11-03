"""Actual EDGAR CO inventory: official emiproc spatial and time operations."""
import hashlib
import json
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod
from shapely.geometry import box
import xarray as xr
from emiproc.inventories import Inventory
from emiproc.regrid import remap_inventory
from emiproc.profiles.temporal.profiles import DailyProfile
from emiproc.profiles.temporal.operators import create_scaling_factors_time_serie

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output/capability_final_acceptance"


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument("source",nargs="?",default=str(ROOT/"examples/ai_round1/cmaq/emissions/A_EDGAR_CO_2014_Paris.nc"))
    parser.add_argument("destination",nargs="?",default=str(OUT))
    parser.add_argument("config",nargs="?")
    args=parser.parse_args()
    if args.config:
        config=json.loads(Path(args.config).read_text(encoding="utf-8"))
        assert config.get("temporal_assumption")=="uniform_annual_mean_equivalent_day" and config.get("confirm_assumption") is True
    OUT=Path(args.destination);OUT.mkdir(parents=True,exist_ok=True)
    path=Path(args.source)
    d=xr.load_dataset(path,engine="netcdf4")
    assert d.fluxes.attrs["units"]=="kg m-2 s-1" and d.fluxes.attrs["substance"]=="CO" and d.fluxes.attrs["year"]=="2014"
    np.testing.assert_allclose(d.lat,[48.75,48.85,48.95])
    np.testing.assert_allclose(d.lon,[2.25,2.35,2.45])
    def polygons(lat,lon,dy,dx):return [box(x-dx/2,y-dy/2,x+dx/2,y+dy/2) for y in lat for x in lon]
    cells=gpd.GeoSeries(polygons(d.lat.values,d.lon.values,.1,.1),crs=4326)
    geod=Geod(ellps="WGS84")
    area=np.array([abs(geod.geometry_area_perimeter(p)[0]) for p in cells])
    # Explicit equivalent mean day, not an observed 2014-01-01 emissions record.
    daily_mass=d.fluxes.values.ravel()*area*86400
    inv=Inventory.from_gdf(gpd.GeoDataFrame({("TOTALS","CO"):daily_mass},geometry=cells.to_crs(6933)))
    lat=np.linspace(48.725,48.975,6);lon=np.linspace(2.225,2.475,6)
    target=gpd.GeoSeries(polygons(lat,lon,.05,.05),crs=4326).to_crs(6933)
    result=remap_inventory(inv,target)
    mapped=result.gdf[("TOTALS","CO")].values.reshape(6,6)
    relative=float(abs(mapped.sum()-daily_mass.sum())/daily_mass.sum())
    assert relative<1e-8
    ds=xr.Dataset({"mass":(("lat","lon"),mapped)},coords={"lat":lat,"lon":lon},attrs={"crs":"EPSG:4326",
        "source":d.attrs["source"],"substance":"CO","inventory_year":2014,
        "time_support":"Equivalent mean day derived from annual mean flux; not actual daily emissions",
        "area_method":"WGS84 geodesic source cell area; projected EPSG6933 intersection fractions"})
    ds.mass.attrs.update(units="kg",spatial_semantics="cell_total")
    ds.to_netcdf(OUT/"edgar_spatial.nc",engine="h5netcdf")
    source=xr.Dataset({"mass":(("lat","lon"),daily_mass.reshape(3,3))},coords={"lat":d.lat,"lon":d.lon},attrs={"crs":"EPSG:4326"})
    source.mass.attrs.update(units="kg",spatial_semantics="cell_total")
    source.to_netcdf(OUT/"edgar_daily_mass_source.nc",engine="h5netcdf")
    # The official uniform DailyProfile is an explicit low-information modelling
    # assumption. Do not apply an unrelated country/sector's measured profile.
    times=pd.date_range("2014-01-01",periods=24,freq="h",tz="UTC")
    factors=create_scaling_factors_time_serie(times[0],times[-1],[DailyProfile()])
    fractions=factors.to_numpy()/24
    hourly=ds.expand_dims(time=times.tz_localize(None).to_numpy()).copy(deep=True)
    hourly["mass"]=(("time","lat","lon"),fractions[:,None,None]*mapped)
    hourly.mass.attrs.update(units="kg",spatial_semantics="cell_total",temporal_semantics="accumulation",time_label="interval_start")
    hourly["time_bounds"]=(("time","bounds"),np.column_stack([hourly.time.values,hourly.time.values+np.timedelta64(1,'h')]))
    hourly.time.attrs.update(timezone="UTC",bounds="time_bounds")
    hourly.time.encoding.update(units="hours since 2014-01-01 00:00:00",calendar="proleptic_gregorian")
    hourly.time_bounds.encoding.update(units="hours since 2014-01-01 00:00:00",calendar="proleptic_gregorian")
    hourly.attrs["temporal_profile"]="Official DailyProfile uniform default, explicitly assumed; not empirically validated for Paris TOTALS"
    np.testing.assert_allclose(hourly.mass.sum('time'),mapped,rtol=1e-12)
    hourly.to_netcdf(OUT/"edgar_hourly.nc",engine="h5netcdf")
    (OUT/"real_emissions_execution.json").write_text(json.dumps({"source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "source":"European Commission JRC EDGAR v8.1 2014 CO TOTALS actual 3x3 Paris subset","source_is_actual_inventory":True,
        "spatial_entry":"emiproc.regrid.remap_inventory","temporal_entry":"DailyProfile/create_scaling_factors_time_serie",
        "input_daily_equivalent_kg":float(daily_mass.sum()),"spatial_total_kg":float(mapped.sum()),"spatial_relative_error":relative,
        "hourly_total_kg":float(hourly.mass.sum()),"hourly_ratio_sum":float(fractions.sum()),
        "temporal_assumption":"Annual mean equivalent day with explicit uniform hourly distribution; NOT observed hourly truth",
        "MODEL_NATIVE_VERIFIED":False},indent=2),encoding="utf-8")


if __name__=="__main__":main()
