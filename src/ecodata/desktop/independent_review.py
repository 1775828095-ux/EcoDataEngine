"""Revalidate a fallback candidate using newly supplied local observations."""
from pathlib import Path

from .scan import assert_current, local_member
from .release_gate import assess, resolve_field, task_fields


def start(recommendation, result):
    from copy import deepcopy

    options = {option["id"]: option for option in recommendation.get("local_recipe_options", ())}
    if result["received"] != result["total"] or result.get("candidate") not in options:
        return None
    option = options[result["candidate"]]
    candidate = dict(recommendation, **{key: option[key] for key in ("id", "name", "description", "steps")},
                     recommendation_origin="deepseek_fallback",
                     local_holdout_review=[], independent_required=True)
    candidate["previous_recommendation_review"] = deepcopy({
        key: recommendation[key] for key in ("id", "steps", "recommendation_origin",
            "local_holdout_review", "execution_release", "previous_recommendation_review") if key in recommendation})
    reason = result.get("reason") or ""
    candidate["reason"] = (f"{reason} " if reason else "") + (
        f"该方案获 {result['votes'][option['id']]}/{result['total']} 个分片支持，等待独立数据本地复验。")
    candidate["execution_release"] = dict(status="pending",
        checks=[], blockers=[dict(reason="new_independent_evidence_required")])
    candidate.pop("final_verified_recommendation", None)
    candidate.pop("final_recommendation_origin", None)
    return candidate


def fields(scan, candidate):
    return [(index, fid, field) for index, name, fid in task_fields({"fields": candidate.get("execution_fields", ())})
            if (field := resolve_field(scan["rows"][index], index, name, fid))]


def review(scan, candidate, task, fid, path):
    from . import file_tools, sampling_plan
    from .fallback_evidence import digest_file
    from .raster_reference import METHODS, compare

    assert_current(scan)
    source_fields = fields(scan, candidate)
    found = next((entry for entry in source_fields if entry[1] == fid), None)
    if found is None:
        raise ValueError("本次候选没有需要复验的对应字段")
    path = Path(path).resolve(strict=True)
    validation_digest = digest_file(path)
    if validation_digest in candidate.get("previous_independent_hashes", ()):
        raise ValueError("这个复验文件已用于上一轮调整，请提供新的独立数据")
    for index in {index for index, _, _ in source_fields}:
        with local_member(scan["rows"][index]) as original:
            if path.samefile(original) or digest_file(original) == validation_digest:
                raise ValueError("复验文件与先前发送范围内的原始资料重复")
    index, _, field = found
    source = scan["rows"][index]
    independent = file_tools.inspect(path, "read_header", {"path": ""})["row"]
    if independent["format"] != source["format"]:
        raise ValueError("请提供同格式的独立数据，以复现当前算法和字段约定")
    if source["format"] == "GeoTIFF":
        method = next((METHODS[step] for step in candidate["steps"] if step in METHODS), None)
        matching = [option for option in independent["fields"] if option.get("band") == field.get("band")]
        if not method or len(matching) != 1:
            raise ValueError("独立栅格缺少对应波段或可复验的空间算法")
        local = sampling_plan.sample(source, field, holdout=True)
        metrics = compare(source, field, local, dict(path=path, band=matching[0]["band"], baseline="nearest"), method)
        accepted = bool(metrics and all(metrics["candidate"][key] <= metrics["baseline"][key] + 1e-9
                                        for key in ("rmse", "mae")))
        report = dict(status="checked" if accepted else "failed", basis="new_independent_raster_reference",
                      field=fid, metrics=metrics, count=(metrics or {}).get("n", 0))
    else:
        matching = [option for option in independent["fields"] if option["name"] == field["name"]
                    and (not field.get("sheet") or option.get("sheet") == field["sheet"])]
        if len(matching) != 1 or not matching[0].get("unit") or matching[0]["unit"] != field.get("unit"):
            raise ValueError("独立文件需提供相同字段和明确一致的单位")
        from .sampling_risk import attach
        validation = {"rows": [independent]}
        reviewed = sampling_plan.recommendation_review(validation, candidate, task, [(0, fid, matching[0])])
        report = reviewed[0]
        attach(validation, 0, matching[0], report, {})
    if digest_file(path) != validation_digest:
        raise ValueError("复验文件在检查期间已变化")
    assert_current(scan)
    report.update(independent=True, independent_file=path.name, independent_sha256=validation_digest)
    return report


def apply(scan, candidate, task, report):
    apply_many(scan, candidate, task, (report,))


def apply_many(scan, candidate, task, reports):
    existing = {item["field"]: item for item in candidate.get("local_holdout_review", ())}
    existing.update((item["field"], item) for item in reports)
    candidate["local_holdout_review"] = list(existing.values())
    candidate["execution_release"] = assess(scan, candidate, task, fields(scan, candidate))


def review_pair(scan, candidate, task, paths, *, model=None):
    """Test a complete recipe on a new pair with disjoint observation dates."""
    import numpy as np
    import xarray as xr
    from ecodata.spatial.regridding import normalize_grid_axes
    from . import file_tools, independent_alignment
    from .fallback_evidence import digest_file

    assert_current(scan)
    source_fields = ordered_fields(scan, candidate, task)
    if len(source_fields) != 2 or len(paths) != 2:
        raise ValueError("整套方案需要分别为两个来源提供独立数据")
    if task.get("route") != "independent_grid_search":
        return review_generic_pair(scan, candidate, task, source_fields, paths, model=model)

    def support(dataset):
        dataset, _ = normalize_grid_axes(dataset)
        key = dataset.time.attrs.get("bounds", "time_bounds")
        if key not in dataset:
            raise ValueError("无法确认复验时段，需提供带明确时间边界的数据")
        values = dataset[key].transpose("time", ...).values
        if (not values.size or not np.issubdtype(values.dtype, np.datetime64) or np.isnat(values).any()):
            raise ValueError("无法确认复验时段，需提供带明确时间边界的数据")
        return (values.min(), values.max()), tuple(dataset[axis].values for axis in ("lat", "lon"))

    previous_periods = [tuple(np.datetime64(bound) for bound in period)
                        for period in candidate.get("previous_independent_periods", ())]
    original_grids = []
    if task.get("cf_daily"):
        for index, _, _ in source_fields:
            with local_member(scan["rows"][index]) as original, xr.open_dataset(original, cache=False) as dataset:
                period, grid = support(dataset)
                previous_periods.append(period)
                original_grids.append(grid)
    else:
        from ecodata.core.temporary import TemporaryDirectory
        # Product adapters establish the original UTC bounds using their receipts.
        with TemporaryDirectory() as folder:
            adapted, *_ = independent_alignment.prepare(scan, candidate["task"], task, Path(folder) / "adapted")
            for item in adapted:
                period, grid = support(item["dataset"])
                previous_periods.append(period)
                original_grids.append(grid)

    rows, checks = [], []
    for path, (index, fid, field), original_grid in zip(paths, source_fields, original_grids, strict=True):
        path = Path(path).resolve(strict=True)
        with xr.open_dataset(path, cache=False) as dataset:
            (begin, end), grid = support(dataset)
        if any(begin < old_end and old_begin < end for old_begin, old_end in previous_periods):
            raise ValueError("新增复验文件的时段与原数据重叠，请选从未用于推荐和外发的时段")
        if any(not np.array_equal(before, after) for before, after in zip(original_grid, grid)):
            raise ValueError("新增独立时段需要沿用对应原始数据的空间网格，才能复验同一套对齐方案")
        checksum = digest_file(path)
        if checksum in candidate.get("previous_independent_hashes", ()) or any(checksum == item[2] for item in checks):
            raise ValueError("请提供两份不同的独立观测，且未用于上一轮验收")
        row = file_tools.inspect(path, "read_header", {"path": ""})["row"]
        if row["format"] != "NetCDF" or not any(new["name"] == field["name"] for new in row["fields"]):
            raise ValueError(f"独立文件需要包含来源 {scan['rows'][index]['name']} 对应的字段 {field['name']}")
        rows.append(row)
        checks.append((path, fid, checksum, (str(begin), str(end))))

    validation = dict(task, cf_daily=True, fields=[(index, field["name"])
                      for index, (_, _, field) in enumerate(source_fields)])
    from .recipe_trials import review as trial
    report = trial({"rows": rows}, candidate["task"], validation, candidate["steps"], model=model, independent=True)
    if any(digest_file(path) != checksum for path, _, checksum, _ in checks):
        raise ValueError("复验文件在检查期间已变化")
    assert_current(scan)
    return [dict(report, field=fid, independent=True, independent_file=path.name,
                 independent_sha256=checksum, independent_period=period)
            for path, fid, checksum, period in checks]


def ordered_fields(scan, candidate, task):
    values = {(index, field["name"]): (index, fid, field) for index, fid, field in fields(scan, candidate)}
    order = [(index, name) for index, name, _ in task_fields(task)]
    return [values[key] for key in order if key in values] if order else list(values.values())


def review_generic_pair(scan, candidate, task, originals, paths, *, model):
    from contextlib import ExitStack
    from hashlib import sha256
    import numpy as np
    from rasterio.warp import transform_bounds
    from rasterio.windows import bounds

    from . import file_tools, indexed_table_recipe, recipe_trials, windowed_grid_recipe
    from .fallback_evidence import digest_file

    route = task.get("route")
    if route not in recipe_trials.ARRAY_ROUTES | {"station_grid"}:
        raise ValueError("此方案还缺少独立组合复验入口")

    def evidence(stack, row, field):
        if indexed_table_recipe.supports(row, field):
            source = indexed_table_recipe.table(stack, row, field, field["unit"])
            hashes = {sha256(f"{source.category}:{key}".encode()).hexdigest() for key in source.groups}
            return dict(kind="table", category=source.category, groups=sorted(hashes))
        grid = windowed_grid_recipe.source(stack, row, field, field["unit"])
        if grid.times is not None:
            start, end = ((grid.times[0], grid.times[-1]+np.timedelta64(1, "ns")) if grid.interval is None else
                          (grid.interval[0, 0], grid.interval[-1, 1]))
            return dict(kind="time", start=str(start), end=str(end))
        return dict(kind="space", crs=grid.crs.to_string(), extent=list(bounds(grid.whole, grid.transform)))

    def overlaps(old, new):
        if old["kind"] != new["kind"]:
            return False
        if new["kind"] == "time":
            return np.datetime64(new["start"]) < np.datetime64(old["end"]) and np.datetime64(old["start"]) < np.datetime64(new["end"])
        if new["kind"] == "table":
            return old["category"] != new["category"] or not set(old["groups"]).isdisjoint(new["groups"])
        a, b, c, d = transform_bounds(old["crs"], new["crs"], *old["extent"])
        x, y, xx, yy = new["extent"]
        return a < xx and x < c and b < yy and y < d

    with ExitStack() as stack:
        previous = list(candidate.get("previous_independent_supports", ()))
        previous.extend(evidence(stack, scan["rows"][index], field) for index, _, field in originals)
        rows, fields_meta, records = [], [], []
        for original, path in zip(originals, paths, strict=True):
            index, fid, field = original
            path = Path(path).resolve(strict=True)
            checksum = digest_file(path)
            if checksum in candidate.get("previous_independent_hashes", ()) or any(checksum == item[2] for item in records):
                raise ValueError("复验文件重复或已用于上一轮调整")
            row = file_tools.inspect(path, "read_header", {"path": ""})["row"]
            if row["format"] != scan["rows"][index]["format"]:
                raise ValueError("复验需保持对应来源的文件格式")
            matching = [item for item in row["fields"] if item["name"] == field["name"]
                        and (not field.get("sheet") or item.get("sheet") == field["sheet"])]
            if len(matching) != 1 or not matching[0].get("unit"):
                raise ValueError("独立文件缺少对应字段或单位")
            proof = evidence(stack, row, matching[0])
            if any(overlaps(old, proof) for old in previous):
                raise ValueError("新增数据和原数据或旧复验的观测范围重叠，请提供独立分组")
            rows.append(row)
            fields_meta.append(matching[0])
            records.append((path, fid, checksum, proof))
        validation = dict(task, fields=[(index, field["name"]) for index, field in enumerate(fields_meta)],
                          field_metadata=fields_meta)
        if route == "station_grid":
            indexes = {original[0]: i for i, original in enumerate(originals)}
            validation.update(grid_index=indexes[task["grid_index"]], table_index=indexes[task["table_index"]],
                              table_field=fields_meta[indexes[task["table_index"]]])
        report = recipe_trials.review({"rows": rows}, candidate["task"], validation, candidate["steps"],
                                      model=model, independent=True)
    if any(digest_file(path) != checksum for path, _, checksum, _ in records):
        raise ValueError("复验文件在检查期间已变化")
    assert_current(scan)
    return [dict(report, field=fid, independent=True, independent_file=path.name,
                 independent_sha256=checksum, independent_support=proof) for path, fid, checksum, proof in records]


def invalidate(scan, candidate, task):
    from copy import deepcopy

    selected = fields(scan, candidate)
    if candidate.get("local_holdout_review"):
        candidate["previous_recommendation_review"] = deepcopy({
            key: candidate[key] for key in ("id", "steps", "recommendation_origin",
                "local_holdout_review", "execution_release", "previous_recommendation_review") if key in candidate})
    candidate.pop("final_verified_recommendation", None)
    candidate.pop("final_recommendation_origin", None)
    candidate["independent_required"] = True
    candidate["local_holdout_review"] = []
    candidate["execution_release"] = assess(scan, candidate, task, selected)
    if not selected:
        candidate["execution_release"].update(status="pending",
            blockers=[dict(reason="new_independent_evidence_required")])
