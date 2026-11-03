"""Official ForDs temperature function; replace only its CDO file-I/O hook."""
from pathlib import Path
import sys
import numpy as np
import xarray as xr


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "external/fords/offline"))
    import topo_factory
    folder = Path(sys.argv[1])
    with np.load(folder / "input.npz", allow_pickle=False) as a:
        lat, lon, out_lat, out_lon = a["lat"], a["lon"], a["target_lat"], a["target_lon"]
        def bilinear_io_adapter(name, input_lat, input_lon, field, year, month, day):
            # The original hook writes NetCDF and calls CDO's bilinear weights.
            # On Windows, delegate this interpolation/I-O step to mature xarray;
            # do not rewrite ForDs lapse-rate estimation or temperature formula.
            ds = xr.DataArray(field, dims=("time", "lat", "lon"), coords={"lat": input_lat, "lon": input_lon})
            return ds.interp(lat=out_lat, lon=out_lon).values
        topo_factory.bilInterp = bilinear_io_adapter
        coarse = a["temperature"]
        elevation = a["coarse_elevation"]
        fine_t = xr.DataArray(coarse, dims=("time", "lat", "lon"), coords={"lat": lat, "lon": lon}).interp(lat=out_lat, lon=out_lon).values
        fine_elevation_reference = xr.DataArray(elevation, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}).interp(lat=out_lat, lon=out_lon).values
        values = topo_factory.downscale_air_temperature(coarse, fine_t, elevation,
            fine_elevation_reference, a["fine_elevation"], lat, lon, 2019, 8, 0)
        lapse = topo_factory.calc_lapse_rate(coarse.transpose(1, 2, 0), elevation)
    np.savez(folder / "output.npz", values=values, baseline=fine_t, coarse_lapse=lapse)


if __name__ == "__main__":
    main()
