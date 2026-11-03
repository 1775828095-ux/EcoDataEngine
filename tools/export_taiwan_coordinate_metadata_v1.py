"""Complete coordinate metadata in new exports, preserving all prior numerical files."""
import numpy as np
import xarray as xr
import h5py
import json
from pyproj import CRS, Geod

from taiwan_working_contract import OUT, save, sha, sites


def main():
    folder = OUT/"exports_coordinate_metadata_v1"
    folder.mkdir(exist_ok=False)
    records = []
    for role in ["validation","test"]:
        receipt = json.loads((OUT/f"{role}_evaluation_receipt.json").read_text())
        source = OUT/receipt["output"]
        for file in sorted(source.glob("*.nc")):
            before = sha(file)
            with xr.open_dataset(file) as opened:
                ds = opened.load()
            expected = ds.pm25.values.copy()
            ds.longitude.attrs.update(units="degrees_east",standard_name="longitude")
            ds.latitude.attrs.update(units="degrees_north",standard_name="latitude")
            ds.time.attrs.update(standard_name="time",timezone="UTC")
            ds.attrs.update(coordinate_metadata_revision="v1",coordinate_basis_status="working assumption; author datum not confirmed")
            output = folder/role/file.name
            output.parent.mkdir(exist_ok=True)
            for name in ["time","time_bounds"]:
                ds[name].encoding.update(units="hours since 2016-12-31 16:00:00",calendar="proleptic_gregorian")
            ds.to_netcdf(output,engine="h5netcdf")
            with h5py.File(output) as check:
                np.testing.assert_allclose(check["pm25"][...],expected,equal_nan=True)
                assert check["longitude"].attrs["units"] in ["degrees_east",b"degrees_east"]
                assert check["latitude"].attrs["units"] in ["degrees_north",b"degrees_north"]
            assert sha(file) == before
            records.append(dict(source=str(file.relative_to(OUT)),source_sha256=before,
                output=str(output.relative_to(OUT)),sha256=sha(output),numerical_values_unchanged=True))
    station = sites()
    area = CRS(32651).area_of_use
    outside = station.loc[~station.longitude.between(area.west,area.east),["station_id","longitude","latitude","split"]]
    i,j = np.triu_indices(len(station),1)
    lon,lat = station.longitude.to_numpy(),station.latitude.to_numpy()
    _,_,geodesic = Geod(ellps="WGS84").inv(lon[i],lat[i],lon[j],lat[j])
    positions = station[["easting","northing"]].to_numpy()
    planar = np.linalg.norm(positions[i]-positions[j],axis=1)
    save(folder/"manifest.json",dict(exports=records,coordinate_attrs_complete=True,numerical_retraining=False,
        projection_status="WARN_OUTSIDE_NOMINAL_ZONE_FOR_THREE_OFFSHORE_STATIONS",
        stations_outside_nominal_UTM_zone=outside.to_dict("records"),
        maximum_station_pair_relative_distance_difference=float(np.max(abs(planar-geodesic)/geodesic)),
        no_coordinate_or_model_change_after_test=True,
        limitation="This is a numerical check under WGS84 working coordinates, not proof of author datum or an error bound for every satellite pixel"))
    print("Eight metadata-completed numerical-identical exports; three offshore projection-area warnings retained",flush=True)


if __name__ == "__main__":
    main()
