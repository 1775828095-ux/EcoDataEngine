"""Isolate the official xsdba unit-registry settings from the EcoData process."""
import json
import sys
from pathlib import Path

import xarray as xr
from xsdba import adjustment


def main():
    directory = Path(sys.argv[1])
    options = json.loads((directory / "options.json").read_text(encoding="utf-8"))
    arrays = []
    for name in ("ref", "hist", "sim"):
        with xr.open_dataarray(directory / (name + ".nc"), engine="h5netcdf") as opened:
            arrays.append(opened.load())
    method = options["method"]
    if method=='Scaling':
        estimator=adjustment.Scaling.train(arrays[0],arrays[1],kind=options['kind'],group='time')
        corrected=estimator.adjust(arrays[2])
    else:
        estimator = getattr(adjustment, "EmpiricalQuantileMapping" if method == "ExtremeValues" else method).train(
            arrays[0], arrays[1], nquantiles=options["nquantiles"], kind=options["kind"], group="time")
        corrected = estimator.adjust(arrays[2], interp="linear", extrapolation="constant")
    if method == "ExtremeValues":
        estimator = adjustment.ExtremeValues.train(
            arrays[0], arrays[1], cluster_thresh=options["cluster_thresh"], q_thresh=options["q_thresh"])
        corrected = estimator.adjust(arrays[2], corrected, frac=options["frac"], power=options["power"])
    corrected.to_netcdf(directory / "corrected.nc", engine="h5netcdf")
    (directory / "parameters.json").write_text(json.dumps(list(estimator.ds.data_vars)), encoding="utf-8")


if __name__ == "__main__":
    main()
