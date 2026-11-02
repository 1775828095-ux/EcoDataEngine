"""Isolated worker; no EcoData/numpy-2 dependency imported here."""
import contextlib
import json
import sys
from importlib.metadata import version
from pathlib import Path


def inspect(path, format):
    import PseudoNetCDF as pnc
    # netCDF4's Windows C backend cannot open some non-ASCII filenames;
    # a read-only memory dataset preserves the bytes and avoids that boundary.
    file = pnc.pncopen("EcoData memory inspection", format=format, memory=Path(path).read_bytes()) if format == "ioapi" else pnc.pncopen(path, format=format)
    required = ["FTYPE", "GDTYP", "P_ALP", "P_BET", "P_GAM", "XCENT", "YCENT", "XORIG", "YORIG",
                "XCELL", "YCELL", "NCOLS", "NROWS", "NLAYS", "NVARS", "SDATE", "STIME", "TSTEP", "VAR-LIST", "VGLVLS"]
    attrs = {name: getattr(file, name) for name in file.ncattrs()}
    variables = {name: {"dimensions": list(value.dimensions), "shape": list(value.shape),
                        "units": getattr(value, "units", "").strip()} for name, value in file.variables.items()}
    times = [str(value) for value in file.getTimes()] if ("TFLAG" in file.variables or "time" in file.variables) else []
    report = {"backend": "PseudoNetCDF", "version": version("PseudoNetCDF"), "input_format": format,
              "attributes": attrs, "variables": variables, "times": times,
              "missing_ioapi_attributes": [name for name in required if name not in attrs],
              "tflag_present": "TFLAG" in file.variables}
    if hasattr(file, "close"):
        file.close()
    return report


if __name__ == "__main__":
    with contextlib.redirect_stdout(sys.stderr):
        report = inspect(sys.argv[1], sys.argv[2])
    print(json.dumps(report, ensure_ascii=False, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)))
