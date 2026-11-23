"""Revalidate a fallback candidate using newly supplied local observations."""
from pathlib import Path

from .scan import assert_current, local_member
from .release_gate import assess, resolve_field, task_fields


def start(recommendation, result):
    options = {option["id"]: option for option in recommendation.get("local_recipe_options", ())}
    if result["received"] != result["total"] or result.get("candidate") not in options:
        return None
    option = options[result["candidate"]]
    candidate = dict(recommendation, **{key: option[key] for key in ("id", "name", "description", "steps")},
                     recommendation_origin="deepseek_fallback",
                     local_holdout_review=[], independent_required=True)
    candidate["reason"] = (f"已审核的候选在 {result['votes'][option['id']]}/{result['total']} 个分片得到支持。"
                           "需要新增独立数据本地复验。")
    candidate["execution_release"] = dict(status="pending",
        checks=[], blockers=[dict(reason="new_independent_evidence_required")])
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


def review_pair(scan, candidate, task, paths):
    """Test a complete recipe on a new pair with disjoint observation dates."""
    import numpy as np
    import xarray as xr
    from ecodata.spatial.regridding import normalize_grid_axes
    from . import file_tools, independent_alignment
    from .fallback_evidence import digest_file

    assert_current(scan)
    source_fields = fields(scan, candidate)
    if len(source_fields) != 2 or len(paths) != 2:
        raise ValueError("整套方案需要分别为两个来源提供独立 NetCDF 数据")

    def support(path):
        with xr.open_dataset(path, cache=False) as dataset:
            dataset, _ = normalize_grid_axes(dataset)
            time = dataset.time
            bounds = dataset[time.attrs.get("bounds", "time_bounds")]
            values = bounds.transpose("time", ...).values
            if (not values.size or not np.issubdtype(values.dtype, np.datetime64) or np.isnat(values).any()):
                raise ValueError("无法确认复验时段，需提供带明确时间边界的数据")
            return (values.min(), values.max()), tuple(dataset[axis].values for axis in ("lat", "lon"))

    previous_periods = [tuple(np.datetime64(bound) for bound in period)
                        for period in candidate.get("previous_independent_periods", ())]
    original_grids = []
    for index, _, _ in source_fields:
        with local_member(scan["rows"][index]) as original:
            period, grid = support(original)
            previous_periods.append(period)
            original_grids.append(grid)

    rows, checks = [], []
    for path, (index, fid, field), original_grid in zip(paths, source_fields, original_grids, strict=True):
        path = Path(path).resolve(strict=True)
        (begin, end), grid = support(path)
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
    report = independent_alignment.review({"rows": rows}, candidate["task"], validation,
                                          candidate["steps"], independent=True)
    if any(digest_file(path) != checksum for path, _, checksum, _ in checks):
        raise ValueError("复验文件在检查期间已变化")
    assert_current(scan)
    return [dict(report, field=fid, independent=True, independent_file=path.name,
                 independent_sha256=checksum, independent_period=period)
            for path, fid, checksum, period in checks]


def invalidate(scan, candidate, task):
    selected = fields(scan, candidate)
    candidate.pop("final_verified_recommendation", None)
    candidate.pop("final_recommendation_origin", None)
    candidate["independent_required"] = True
    candidate["local_holdout_review"] = []
    candidate["execution_release"] = assess(scan, candidate, task, selected)
    if not selected:
        candidate["execution_release"].update(status="pending",
            blockers=[dict(reason="new_independent_evidence_required")])
