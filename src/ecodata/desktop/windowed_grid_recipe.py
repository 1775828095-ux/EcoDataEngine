"""Windowed fixed recipes for regular CF grids and GeoTIFF fields."""
from contextlib import ExitStack
from dataclasses import dataclass
from math import ceil, floor

import numpy as np
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, bounds, from_bounds, transform

from ecodata.engines.units import UnitEngine

from . import recipe_trials
from .scan import local_member

FORMATS = {"GeoTIFF", "NetCDF", "HDF5"}
CONSERVATIVE = "spatial.regrid.conservative"
TEMPORAL = "temporal.semantic"
ALIGN = {"raster_task.spatial.nearest": Resampling.nearest, "raster_task.spatial.bilinear": Resampling.bilinear,
         CONSERVATIVE: "conservative"}
BLOCK = 32


@dataclass
class Grid:
    crs: CRS
    transform: object
    width: int
    height: int
    read: object
    held: Window
    times: np.ndarray | None = None
    holdout_time: int = 0
    support: str = "point"
    temporal: str = "static"
    interval: np.ndarray | None = None

    @property
    def whole(self):
        return Window(0, 0, self.width, self.height)


def held_window(width, height, axis, reversed_axis=False):
    length = height if axis == "y" else width
    held = max(1, length // 5)
    start = 0 if reversed_axis else length - held
    return Window(0, start, width, held) if axis == "y" else Window(start, 0, held, height)


def source(stack, row, field, unit):
    path = stack.enter_context(local_member(row))
    attrs = field
    if row["format"] == "GeoTIFF":
        import rasterio

        ds = stack.enter_context(rasterio.open(path))
        band = field["band"]
        attrs = {**field, **ds.tags(), **ds.tags(band)}
        if not ds.crs or ds.transform.b or ds.transform.d or ds.transform.a <= 0 or ds.transform.e >= 0:
            raise ValueError("栅格需有明确投影及规则像元轴")
        scale, offset = UnitEngine().affine(attrs.get("unit") or ds.units[band - 1], unit)
        if attrs.get("flag_values") or attrs.get("flag_meanings") or attrs.get("data_kind") == "categorical":
            raise ValueError("分类像元需专用评分方案")

        def read(window, _):
            data = ds.read(band, window=window, masked=True).astype(float).filled(np.nan)
            return (data * ds.scales[band - 1] + ds.offsets[band - 1]) * scale + offset

        support = attrs.get("spatial_semantics") or "grid_estimate"
        return Grid(ds.crs, ds.transform, ds.width, ds.height, read,
                    held_window(ds.width, ds.height, "y" if ds.height > 1 else "x"), support=support)

    ds = stack.enter_context(xr.open_dataset(path, cache=False))
    array = ds[field["name"]]
    attrs = {**field, **ds.attrs, **array.attrs}
    axes = [next((name for name in names if name in array.dims), None)
            for names in (("lat", "latitude", "y"), ("lon", "longitude", "x"))]
    y, x = axes
    if not y or not x or set(array.dims) - {y, x, "time"}:
        raise ValueError("该固定方案需要可定位的规则二维网格或时空网格")
    crs_value = attrs.get("crs")
    mapping = array.attrs.get("grid_mapping")
    if not crs_value and mapping in ds:
        crs_value = ds[mapping].attrs.get("spatial_ref") or ds[mapping].attrs.get("crs_wkt")
    if not crs_value:
        raise ValueError("缺少明确网格坐标基准")
    if attrs.get("flag_values") is not None or attrs.get("data_kind") == "categorical":
        raise ValueError("分类像元需专用评分方案")
    scale, offset = UnitEngine().affine(attrs.get("units") or attrs.get("unit"), unit)
    xx, yy = (np.asarray(ds[axis], dtype=float) for axis in (x, y))
    if any(values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() for values in (xx, yy)):
        raise ValueError("网格坐标须为有限的一维轴")
    dx, dy = np.diff(xx), np.diff(yy)
    if not all(diff[0] != 0 and np.allclose(diff, diff[0]) for diff in (dx, dy)):
        raise ValueError("网格坐标间隔不一致")
    held_axis = next(name for name in array.dims if name != "time")
    held = held_window(len(xx), len(yy), "y" if held_axis == y else "x",
                       dy[0] > 0 if held_axis == y else dx[0] < 0)
    if dx[0] < 0:
        array = array.isel({x: slice(None, None, -1)})
    if dy[0] > 0:
        array = array.isel({y: slice(None, None, -1)})
    affine = from_origin(xx.min() - abs(dx[0])/2, yy.max() + abs(dy[0])/2, abs(dx[0]), abs(dy[0]))
    times, intervals, holdout_time = None, None, 0
    temporal = "static"
    if "time" in array.dims:
        times = array.time.values
        temporal = attrs.get("temporal_semantics")
        if (temporal not in {"instantaneous", "mean", "accumulation"} or not len(times)
                or not np.issubdtype(times.dtype, np.datetime64) or np.isnat(times).any()
                or (np.diff(times) <= np.timedelta64(0, "ns")).any()):
            raise ValueError("须有明确、递增且无重复的时间统计坐标")
        if temporal != "instantaneous":
            edges = ds.get(array.time.attrs.get("bounds", "time_bounds"))
            if edges is None:
                raise ValueError("累计或平均网格缺少时间区间")
            intervals = edges.transpose("time", ...).values
            if (intervals.shape != (len(times), 2) or not np.issubdtype(intervals.dtype, np.datetime64)
                    or np.isnat(intervals).any() or (intervals[:, 1] <= intervals[:, 0]).any()
                    or (intervals[1:, 0] < intervals[:-1, 1]).any()):
                raise ValueError("时间区间无效或重叠")
        holdout_time = len(times) - max(1, len(times)//5)

    def read(window, time):
        selection = {x: slice(int(window.col_off), int(window.col_off + window.width)),
                     y: slice(int(window.row_off), int(window.row_off + window.height))}
        if time is not None:
            selection["time"] = time
        return np.asarray(array.isel(selection).transpose(y, x), dtype=float) * scale + offset

    return Grid(CRS.from_user_input(crs_value), affine, len(xx), len(yy), read, held, times, holdout_time,
                attrs.get("spatial_semantics") or "grid_estimate", temporal, intervals)


def intersection(base, other, base_window, other_window, *, contained):
    extent = transform_bounds(other.crs, base.crs, *bounds(other_window, other.transform))
    result = from_bounds(*extent, transform=base.transform)
    lower, upper = (ceil, floor) if contained else (floor, ceil)
    left = max(int(base_window.col_off), lower(result.col_off + (-1e-7 if contained else 1e-7)))
    top = max(int(base_window.row_off), lower(result.row_off + (-1e-7 if contained else 1e-7)))
    right = min(int(base_window.col_off + base_window.width), upper(result.col_off + result.width + (1e-7 if contained else -1e-7)))
    bottom = min(int(base_window.row_off + base_window.height), upper(result.row_off + result.height + (1e-7 if contained else -1e-7)))
    return Window(left, top, max(0, right-left), max(0, bottom-top))


def clock_pairs(a, b):
    if a.times is None:
        return [(None, None)]
    if a.interval is not None:
        _, left, right = np.intersect1d(a.times, b.times, assume_unique=True, return_indices=True)
        return [(int(i), int(j)) for i, j in zip(left, right, strict=True)
                if np.array_equal(a.interval[i], b.interval[j])]
    # Exact matches remain untouched. Interpolation uses two actual adjacent
    # observations and does not bridge a longer break or extrapolate endpoints.
    cadence = np.min(np.diff(b.times)) if len(b.times) > 1 else np.timedelta64(0, "ns")
    out = []
    for i, (time, right) in enumerate(zip(a.times, np.searchsorted(b.times, a.times), strict=True)):
        if right < len(b.times) and b.times[right] == time:
            out.append((i, int(right)))
        elif 0 < right < len(b.times) and b.times[right]-b.times[right-1] == cadence:
            weight = float((time-b.times[right-1])/cadence)
            out.append((i, (int(right-1), int(right), weight)))
    return out


def pair(stack, scan, task):
    rows = [scan["rows"][index] for index, _ in task["fields"]]
    if rows[0]["ref"] == rows[1]["ref"] and rows[0].get("containers") == rows[1].get("containers"):
        raise ValueError("同一输入不能组成两个独立来源")
    a, b = [source(stack, row, field, task["unit"]) for row, field in zip(rows, task["field_metadata"], strict=True)]
    geog = a.crs == b.crs == CRS.from_epsg(4326)
    continuous = {a.support, b.support} <= {"grid_estimate", "point"}
    conservation = a.support == b.support and a.support in {"cell_mean", "flux"} and geog
    if a.temporal != b.temporal or not (continuous or conservation):
        raise ValueError("两源需相同时间统计口径和连续空间支持；单元均值或通量需同 CRS 守恒适配")
    if a.crs == b.crs and abs(a.transform.a * a.transform.e) < abs(b.transform.a * b.transform.e):
        a, b = b, a
    shared = intersection(a, b, a.whole, b.whole, contained=True)
    if not shared.width or not shared.height:
        raise ValueError("两源没有共同空间范围")
    times = clock_pairs(a, b)
    if not times:
        raise ValueError("两源缺少一致区间或有据可插值的时间范围")
    return a, b, shared, times


def available(scan, task):
    with ExitStack() as stack:
        grid, _, _, times = pair(stack, scan, task)
    temporal = [TEMPORAL] if any(isinstance(right, tuple) for _, right in times) else []
    space = [CONSERVATIVE] if grid.support in {"cell_mean", "flux"} else [step for step in ALIGN if step != CONSERVATIVE]
    return [[*temporal, step] for step in space]


def remap(values, source_affine, source_crs, shape, target_affine, target_crs, method):
    if method == "conservative":
        from .grid_preparation import remap as remap_cf

        if min(values.shape + shape) < 2:
            return np.full(shape, np.nan)

        def coordinates(affine, shape):
            return dict(lat=affine.f+(np.arange(shape[0])+.5)*affine.e,
                        lon=affine.c+(np.arange(shape[1])+.5)*affine.a)

        field = xr.Dataset({"value": (("lat", "lon"), values, dict(data_kind="continuous", spatial_semantics="cell_mean"))},
                           coords=coordinates(source_affine, values.shape), attrs=dict(crs=source_crs.to_string()))
        field = field.sortby("lat")
        target = xr.Dataset(coords=coordinates(target_affine, shape)).sortby("lat")
        return remap_cf(field, target, "value", "conservative").value.values[::-1]
    result = np.full(shape, np.nan)
    reproject(values, result, src_transform=source_affine, src_crs=source_crs, src_nodata=np.nan,
              dst_transform=target_affine, dst_crs=target_crs, dst_nodata=np.nan, resampling=method)
    return result


def read_pair(a, b, window, time, method, *, limit=None):
    i, j = time
    limit = limit or b.whole
    source_window = intersection(b, a, limit, window, contained=False)
    if method != Resampling.nearest and source_window.width and source_window.height:
        left = max(int(limit.col_off), int(source_window.col_off)-1)
        top = max(int(limit.row_off), int(source_window.row_off)-1)
        right = min(int(limit.col_off + limit.width), int(source_window.col_off + source_window.width)+1)
        bottom = min(int(limit.row_off + limit.height), int(source_window.row_off + source_window.height)+1)
        source_window = Window(left, top, right-left, bottom-top)
    left = a.read(window, i)
    aligned = np.full(left.shape, np.nan)
    target_affine = transform(window, a.transform)
    if source_window.width and source_window.height:
        if isinstance(j, tuple):
            from ecodata.algorithms.spatial_fusion_task import blend_fields
            first, last, weight = j
            values = blend_fields(b.read(source_window, last), b.read(source_window, first), weight)
        else:
            values = b.read(source_window, j)
        aligned = remap(values, transform(source_window, b.transform), b.crs,
                        left.shape, target_affine, a.crs, method)
    return left, aligned, target_affine


def combine(left, right, task, steps, temporal):
    variable = task["quantity"]
    attrs = dict(physical_variable=variable, units=task["unit"], data_kind="continuous",
                 spatial_semantics="grid_estimate", temporal_semantics=temporal)
    frames = [xr.Dataset({variable: (("y", "x"), values, attrs)}) for values in (left, right)]
    return recipe_trials.combine(frames, variable, steps)[0][variable].values


def review(scan, task, steps, *, independent=False):
    with ExitStack() as stack:
        a, b, shared, times = pair(stack, scan, task)
        method = next(ALIGN[step] for step in steps if step in ALIGN)
        if not independent and a.times is None:
            shared = intersection(a, b, a.held, b.held, contained=True)
        elif not independent:
            times = [(i, j) for i, j in times if i >= a.holdout_time and (j[0] if isinstance(j, tuple) else j) >= b.holdout_time]
        cols, rows = (max(1, ceil(size/BLOCK) - int(0 < size % BLOCK < 4 and size > BLOCK)) if size else 0
                      for size in (shared.width, shared.height))
        if a.crs != b.crs and a.times is None and not independent:
            # A projected bounding box is insufficient proof of disjointness.
            return recipe_trials.audit(0, "withheld_spatial_block", None)

        def trial(index):
            time, cell = divmod(index, rows * cols)
            y, x = divmod(cell, cols)
            w = Window(shared.col_off + x*BLOCK, shared.row_off + y*BLOCK,
                       shared.width-x*BLOCK if x == cols-1 else BLOCK,
                       shared.height-y*BLOCK if y == rows-1 else BLOCK)
            left, right, affine = read_pair(a, b, w, times[time], method,
                                            limit=b.held if a.times is None and not independent else b.whole)
            group = f"{time}/{y}/{x}"
            if min(left.shape) < 3 or not np.isfinite(left).any() or not np.isfinite(right).any():
                return group, None, "insufficient_score_support"
            candidate_output = combine(left, right, task, steps, a.temporal)
            if ((np.isfinite(left) | np.isfinite(right)) & ~np.isfinite(candidate_output)).any():
                return group, None, "recipe_output_missing_on_reference"
            from affine import Affine
            coarse_shape = tuple(ceil(size/2) for size in left.shape)
            coarse_affine = affine @ Affine.scale(left.shape[1]/coarse_shape[1], left.shape[0]/coarse_shape[0])
            predictions = []
            for values in (left, right):
                reduced = remap(values, affine, a.crs, coarse_shape, coarse_affine, a.crs,
                                method if method == "conservative" else Resampling.average)
                predictions.append(remap(reduced, coarse_affine, a.crs, left.shape, affine, a.crs, method))
            mask = np.isfinite(predictions[0]) & np.isfinite(predictions[1])
            truth = np.where(mask, left, np.nan)
            candidate = combine(*predictions, task, steps, a.temporal)
            baselines = sorted(predictions, key=lambda values: np.nansum((values - truth)**2))
            metrics, failure = recipe_trials.score(candidate, baselines[0], truth, "reduced_scale_grid_proxy",
                                                   retained=np.stack(predictions), request=task.get("scoring_request"))
            return group, metrics, failure

        unit = "withheld_spatial_block" if a.times is None else "withheld_time_and_spatial_block"
        return recipe_trials.audit(len(times) * rows * cols, unit, trial)


def run(scan, task_id, task, folder, model_id, progress):
    from .pair_processing import artifact

    search = None
    execution = dict(task=task_id, methods=task["recipes"][0], mutation_status="此网格尚无适配的原版变异评分入口")
    with ExitStack() as stack:
        a, b, window, times = pair(stack, scan, task)
        variable = task["quantity"]
        steps, = task["recipes"]
        method = next(ALIGN[step] for step in steps if step in ALIGN)
        seeds = (a.temporal == "instantaneous" and a.crs == CRS.from_epsg(4326) and len(times) >= 3
                 and recipe_trials.BLEND in steps)
        results, shared_times, shared_fields = [], [], []
        for time in times:
            left, right, affine = read_pair(a, b, window, time, method)
            values = combine(left, right, task, steps, a.temporal)
            results.append(values)
            if seeds and np.isfinite(left).all() and np.isfinite(right).all():
                shared_times.append(a.times[time[0]])
                shared_fields.append((left, right))
        axes = dict(x=affine.c + (np.arange(int(window.width))+.5)*affine.a,
                    y=affine.f + (np.arange(int(window.height))+.5)*affine.e)
        if a.times is None:
            dims, values = ("y", "x"), results[0]
        else:
            dims, values = ("time", "y", "x"), np.stack(results)
            axes["time"] = a.times[[i for i, _ in times]]
        ds = xr.Dataset({variable: (dims, values, dict(units=task["unit"], temporal_semantics=a.temporal,
                        physical_variable=variable, spatial_semantics=a.support))}, coords=axes,
                        attrs=dict(crs=a.crs.to_string(), spatial_x="x", spatial_y="y"))
        if a.interval is not None:
            ds["time_bounds"] = (("time", "bounds"), a.interval[[i for i, _ in times]])
            ds.time.attrs["bounds"] = "time_bounds"
            ds.time.encoding["units"] = ds.time_bounds.encoding["units"] = "seconds since 1970-01-01"
        if len(shared_fields) >= 3:
            updated, search, decision = mutate(task_id, task, ds, shared_times, shared_fields, folder, model_id, progress)
            ds[variable].loc[dict(time=shared_times)] = updated
            execution["mutation_status"] = decision
        file, details = artifact(ds, "aligned-filled-fusion.nc", variable)
    return dict(files=[file], source_formats=[scan["rows"][index]["format"] for index, _ in task["fields"]],
                target_format="NetCDF", original_report=[execution], grid_searches={task_id: search} if search else {},
                local=dict(artifacts=[details], reason="已执行验收通过的网格对齐、备源补位及融合方案。"))


def mutate(task_id, task, template, times, pairs, folder, model_id, progress):
    from ecodata.algorithms.platform_search import SearchInput, with_recommendations
    from ecodata.core.exceptions import EcoDataError
    from .local_search import pareto_comparison, submit
    from .result_summary import ranked_candidates
    from .service import fusion_evidence

    variable = task["quantity"]
    inputs = []
    for index, arrays in enumerate(zip(*pairs)):
        frame = template[[variable]].sel(time=times).rename({"x": "lon", "y": "lat"}).copy(deep=True)
        frame[variable].values[:] = np.stack(arrays)
        frame[variable].attrs["data_kind"] = "continuous"
        frame.attrs.update(source_id=f"accepted-source-{index}", spatial_x="lon", spatial_y="lat")
        frame.lat.attrs["units"], frame.lon.attrs["units"] = "degrees_north", "degrees_east"
        inputs.append(SearchInput(f"source-{index}.nc", "primary" if not index else "background",
                                  bytes(frame.to_netcdf(engine="h5netcdf"))))
    request = dict(task="multisource", variable=variable, physical_variable=variable, unit=task["unit"],
                   semantic_type="continuous", temporal_semantics="instantaneous", processing_goal="spatial_fusion",
                   initial_algorithms=[recipe_trials.BLEND], observation_crs="EPSG:4326", mutation_control="random")
    request = with_recommendations(request, [dict(id=task_id, task=task_id, steps=[recipe_trials.BLEND],
                                                 scope=dict(variable=variable))])
    path = folder/"spatial-search"
    try:
        result = submit(inputs, request, path, model_id=model_id, progress=progress)
        if result.get("status") != "PASS" or not result.get("product"):
            raise ValueError("原版搜索未提供可用产物")
        with xr.open_dataset(result["product"]) as product:
            aligned = product[variable].rename({"lat": "y", "lon": "x"}).transpose("time", "y", "x")
            aligned = aligned.sel(y=template.y, x=template.x, time=times).load()
            if not np.isfinite(aligned).all():
                raise ValueError("寻优产物存在未恢复的观测缺口")
    except (EcoDataError, OSError, ValueError, RuntimeError, KeyError) as error:
        return template[variable].sel(time=times), None, "保留验收方案，原版搜索未生成可替换产物："+str(error)
    rankings = ranked_candidates(path, result)
    evidence = dict(best=result["best"], selected_graph=result["selected_graph"], evidence=fusion_evidence(result),
                    confirmed_initial=result["confirmed_initial"], rankings=rankings, pareto_comparison=pareto_comparison(rankings))
    return aligned, evidence, "双源完整时段使用原版变异寻优结果"
