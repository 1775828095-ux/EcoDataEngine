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
    existing = {item["field"]: item for item in candidate.get("local_holdout_review", ())}
    existing[report["field"]] = report
    candidate["local_holdout_review"] = list(existing.values())
    candidate["execution_release"] = assess(scan, candidate, task, fields(scan, candidate))


def invalidate(scan, candidate, task):
    selected = fields(scan, candidate)
    candidate["independent_required"] = True
    candidate["local_holdout_review"] = []
    candidate["execution_release"] = assess(scan, candidate, task, selected)
    if not selected:
        candidate["execution_release"].update(status="pending",
            blockers=[dict(reason="new_independent_evidence_required")])
