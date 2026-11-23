"""Local CF precipitation products for interval or UTC-day targets."""
from datetime import timedelta
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .scan import local_member


def precipitation_bounds(dataset, clock, variable, daily):
    from netCDF4 import num2date, date2num
    from ecodata.engines.units import UnitEngine

    time = dataset[clock]
    units = time.getncattr("units")
    calendar = getattr(time, "calendar", "standard")
    if daily and calendar not in {"standard", "gregorian", "proleptic_gregorian"}:
        raise ValueError("目标要求 UTC 日，原时间坐标没有标准公历日")
    time_basis = str(getattr(dataset, "time_basis", getattr(time, "time_basis", ""))).upper()
    if daily and time_basis not in {"", "UTC"}:
        raise ValueError("声明的时间基准无法作为 UTC 日")
    bounds = getattr(time, "bounds", None)
    if bounds not in dataset.variables:
        raise ValueError("缺少明确的 CF 时间区间")
    array = np.ma.asarray(dataset[bounds][:], dtype=float).filled(np.nan)
    if (array.shape != (time.size, 2) or not time.size or not np.isfinite(array).all()
            or (array[:, 1] <= array[:, 0]).any() or (array[1:, 0] < array[:-1, 1]).any()):
        raise ValueError("时间区间无效或重叠")
    times = np.ma.asarray(time[:], dtype=float).filled(np.nan)
    if (not np.isfinite(times).all() or (np.diff(times) <= 0).any()
            or ((times < array[:, 0]) | (times > array[:, 1])).any()):
        raise ValueError("时间坐标存在重复、倒序或超出声明区间")
    from re import escape, search
    expected = str(getattr(variable, "temporal_semantics", "")).lower()
    if expected not in {"mean_rate", "interval_mean", "mean"} and not search(
            r"\b" + escape(clock) + r"\s*:\s*mean\b", str(getattr(variable, "cell_methods", "")), flags=2):
        raise ValueError("需明确该变量是区间平均降水率")
    engine = UnitEngine()
    source = getattr(variable, "units", "")
    from ecodata.core.exceptions import SchemaError
    for target in ("mm/s", "kg/m^2/s"):
        try:
            factor, offset = engine.affine(source, target)
            if offset == 0:
                break
        except SchemaError:
            pass
    else:
        raise ValueError("变量单位不能换算为降水率")
    dates = num2date(array, units, calendar=calendar)
    seconds = np.asarray([(end - start).total_seconds() for start, end in dates], dtype=float)
    if not daily:
        return [(i, i + 1) for i in range(time.size)], array, seconds * factor
    groups, normalized = [], []
    start_index = 0
    while start_index < len(dates):
        start, _ = dates[start_index]
        midnight = start.replace(hour=0, minute=0, second=0, microsecond=0)
        if start != midnight:
            raise ValueError("UTC 日缺少日初覆盖")
        day_end, position, previous = start + timedelta(days=1), start_index, start
        while position < len(dates) and previous < day_end:
            first, last = dates[position]
            if first != previous or last > day_end:
                raise ValueError("UTC 日存在缺段或跨日区间，无法可靠分配日累计量")
            previous = last
            position += 1
        if previous != day_end:
            raise ValueError("UTC 日缺少日末覆盖")
        groups.append((start_index, position))
        normalized.append((start, day_end))
        start_index = position
    return groups, np.asarray(date2num(normalized, units, calendar=calendar)), seconds * factor


def export(row, field, destination, *, daily=False):
    from netCDF4 import Dataset

    destination = Path(destination)
    with local_member(row) as path, Dataset(path) as source, TemporaryDirectory(dir=destination.parent) as scratch:
        temporary = Path(scratch) / "precipitation.nc"
        original = source[field["name"]]
        axes = [name for name in original.dimensions if name.casefold() in {"time", "tstep"}
                or (name in source.variables and getattr(source[name], "axis", "") == "T")]
        if len(axes) != 1:
            raise ValueError("不能唯一定位降水时间坐标")
        clock = axes[0]
        extra = tuple(dim for dim in original.dimensions if dim != clock)
        groups, bounds, multiplier = precipitation_bounds(source, clock, original, daily)
        mapping = getattr(original, "grid_mapping", None)
        coordinates = set(extra) | set(str(getattr(original, "coordinates", "")).split())
        if mapping in source.variables:
            coordinates.add(mapping)
        pending = list(coordinates)
        while pending:
            name = pending.pop()
            if name not in source.variables:
                continue
            bound = getattr(source[name], "bounds", None)
            if bound in source.variables and bound not in coordinates:
                coordinates.add(bound)
                pending.append(bound)
        with Dataset(temporary, "w", format="NETCDF4") as out:
            for dim, size in source.dimensions.items():
                out.createDimension(dim, len(groups) if dim == clock else len(size))
            interval_dim = "ecodata_interval_edge"
            while interval_dim in out.dimensions:
                interval_dim += "_"
            out.createDimension(interval_dim, 2)
            for name in sorted(coordinates - {clock, field["name"]}):
                if name not in source.variables:
                    continue
                var = source[name]
                if clock in var.dimensions or var.dtype.kind not in "iufS":
                    continue
                copied = out.createVariable(name, var.dtype, var.dimensions,
                                            fill_value=getattr(var, "_FillValue", None))
                var.set_auto_maskandscale(False)
                copied[:] = var[:]
                copied.setncatts({key: var.getncattr(key) for key in var.ncattrs() if key != "_FillValue"})
            time = out.createVariable(clock, "f8", (clock,))
            time[:] = bounds[:, 0]
            time.setncatts(dict(units=source[clock].units, calendar=getattr(source[clock], "calendar", "standard"),
                              standard_name="time", axis="T", bounds="ecodata_time_bounds"))
            out.createVariable("ecodata_time_bounds", "f8", (clock, interval_dim))[:] = bounds
            result = out.createVariable(field["name"], "f8", (clock, *extra), zlib=True, fill_value=np.nan)
            result.setncatts(dict(units="mm", standard_name="lwe_thickness_of_precipitation_amount",
                                 cell_methods=f"{clock}: sum"))
            if mapping in out.variables:
                result.grid_mapping = mapping
            present = [name for name in str(getattr(original, "coordinates", "")).split() if name in out.variables]
            if present:
                result.coordinates = " ".join(present)
            out.setncatts(dict(Conventions="CF-1.8", aggregation="UTC_day" if daily else "preserved_source_intervals"))
            shape = [len(source.dimensions[dim]) for dim in extra]
            widths = [1] * len(shape)
            budget = 4096
            for index in reversed(range(len(shape))):
                widths[index] = max(1, min(64, shape[index], budget))
                budget = max(1, budget // widths[index])
            if any(not size for size in shape):
                raise ValueError("原始空间网格为空")
            for starts in product(*(range(0, n, width) for n, width in zip(shape, widths))):
                selections = {dim: slice(start, min(start + width, size))
                              for dim, start, width, size in zip(extra, starts, widths, shape)}
                output_key = tuple(selections[dim] for dim in extra)
                for index, (start, stop) in enumerate(groups):
                    amount = None
                    for time_index in range(start, stop):
                        piece = np.ma.asarray(original[tuple(time_index if dim == clock else selections[dim]
                                                           for dim in original.dimensions)], dtype=float).filled(np.nan)
                        if (piece[np.isfinite(piece)] < 0).any():
                            raise ValueError("原始降水率存在负值")
                        piece *= multiplier[time_index]
                        amount = piece if amount is None else amount + piece
                    result[(index, *output_key)] = amount
        with Dataset(temporary) as written:
            saved_time = written[clock]
            saved_amount = written[field["name"]]
            if (saved_amount.shape != (len(groups), *shape) or saved_amount.units != "mm"
                    or not np.array_equal(saved_time[:], bounds[:, 0])
                    or not np.array_equal(written[saved_time.bounds][:], bounds)):
                raise ValueError("回读校验未通过，未替换降水结果")
        temporary.replace(destination)
    return dict(path=str(destination), groups=len(groups), mode="UTC_day" if daily else "source_intervals")
