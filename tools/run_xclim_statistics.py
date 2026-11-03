"""Keep xclim's Pint registry settings isolated from the platform process."""
import json
import sys
from pathlib import Path
import xarray as xr
from xclim.indices import generic
from xclim.core.units import convert_units_to


def main():
    folder = Path(sys.argv[1])
    p = json.loads((folder / "options.json").read_text(encoding="utf-8"))
    with xr.open_dataarray(folder / "input.nc", engine="h5netcdf") as opened:
        a = opened.load()
    op, freq = p["statistic"], p["target_frequency"]
    if op in {"duration", "frequency"}:
        threshold = convert_units_to(p["threshold"], a)
        if op == "frequency":
            out = generic.threshold_count(a, p.get("comparison", ">="), threshold, freq).assign_attrs(units="1")
        else:
            out = generic.spell_length_statistics(a, p["threshold"], window=1, win_reducer="min",
                op=p.get("comparison", ">="), spell_reducer="max", freq=freq).assign_attrs(units="d")
    else:
        method = "integral" if op == "accumulation" and p["temporal_semantics"] == "rate" else ("sum" if op == "accumulation" else op)
        out = generic.select_resample_op(a, method, freq=freq)
    out.attrs.update(statistic=op, temporal_semantics="period_statistic")
    ds = out.to_dataset(name="result")
    ds["contributing_sample_count"] = a.resample(time=freq).count()
    ds.contributing_sample_count.attrs["units"] = "1"
    ds.to_netcdf(folder / "output.nc", engine="h5netcdf")


if __name__ == "__main__":
    main()
