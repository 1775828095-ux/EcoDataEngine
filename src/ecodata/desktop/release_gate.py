"""Local authorization for execution, separate from externally supplied advice."""
from pathlib import Path

from .scan import assert_current, local_member


def assess(scan, recommendation, task, fields):
    reviews = {item["field"]: item for item in recommendation.get("local_holdout_review", ())}
    steps = tuple(recommendation.get("steps", ()))
    checks = []
    for index, fid, field in fields:
        source = scan["rows"][index]
        review = reviews.get(fid, {})
        status, reason = review.get("status", "pending"), review.get("basis", "holdout_not_reviewed")
        supported = source["format"] in {"NetCDF", "GeoTIFF", "CSV", "TSV", "Excel"}
        if not supported:
            continue
        if recommendation.get("independent_required") and not review.get("independent"):
            status, reason = "pending", "new_independent_evidence_required"
        elif status == "failed":
            pass
        elif source["format"] in {"CSV", "TSV", "Excel"} and not field.get("unit"):
            status, reason = "pending", "selected_table_unit_missing"
        elif source["format"] == "GeoTIFF" and any(step.startswith(("spatial.", "raster_task.spatial")) for step in steps):
            if review.get("grid_trial") or review.get("metrics"):
                status, reason = "checked", "spatial_holdout_review_complete"
            elif task and (task.get("target_grid") or task.get("target_resolution")) and not review.get("count", 0):
                status, reason = "pending", "insufficient_sample_coverage"
            else:
                status, reason = "pending", "target_grid_or_holdout_evidence_missing"
        elif source["format"] == "NetCDF" and "engineering.precipitation_adaptation" in steps:
            if review.get("precipitation_trial"):
                status, reason = "checked", "precipitation_holdout_integration_complete"
            else:
                status, reason = "pending", "precipitation_time_bounds_or_unit_missing"
        elif (source["format"] == "NetCDF" and status != "checked" and review.get("count", 0)
              and not recommendation.get("independent_required")
              and task and task.get("route") in {
                "pair_processing", "station_grid", "observation_gap_search", "independent_alignment",
                "independent_grid_search", "native_reference_search", "statistical_preprocessing"}):
            # The existing registered executor supplies the operation-specific gate
            # and quality evaluation. An empty holdout is never accepted here.
            from .discovery import field_identity
            quantity, _ = field_identity(source, field)
            if quantity and field.get("unit") and review.get("basis") != "heldout_worksheet_field_semantics_need_declaration":
                status, reason = "checked", "heldout_inputs_and_registered_execution_contract"
        elif status == "pending" and not review.get("count", 0) and reason == "algorithm_or_independent_reference_not_verified":
            reason = "insufficient_sample_coverage"
        checks.append(dict(field=fid, source=source["name"], format=source["format"], status=status, reason=reason))

    engineering = any(step.startswith("engineering.") for step in steps)
    if engineering:
        for check in scan.get("document_evidence_checks", ()):
            if check["status"] != "checked":
                checks.append(dict(source=check["document"], status=check["status"],
                                   reason=check.get("basis", "missing_engineering_reference")))
    blocked = [entry for entry in checks if entry["status"] != "checked"]
    if recommendation.get("advisory_only") or not steps:
        status = "pending"
    elif any(entry["status"] in {"failed", "conflict"} for entry in blocked):
        status = "failed"
    else:
        status = "pending" if blocked else "released"
    return dict(status=status, checks=checks, blockers=blocked,
                members=package_members(scan, checks, engineering))


def package_members(scan, checks, engineering):
    from .document_evidence import relative_reference

    combined = {}
    ranks = {"checked": 0, "pending": 1, "failed": 2, "conflict": 2}

    def add(name, item_status, reason):
        if not name or "!/" not in name:
            return
        member = combined.setdefault(name, dict(member=name, status="checked", checks=[]))
        if ranks.get(item_status, 1) > ranks[member["status"]]:
            member["status"] = item_status if item_status in ranks else "pending"
        member["checks"].append(reason)
        member["outbound"] = scan.get("member_outbound", {}).get(name, "local_only")

    for item in checks:
        add(item.get("source"), item["status"], item["reason"])
    if engineering:
        for item in scan.get("document_evidence_checks", ()):
            if item.get("check") == "member_reference" and item.get("reference"):
                name, _ = relative_reference(scan, item["document"], item["reference"])
                add(name, item["status"], "required_engineering_reference")
            elif item.get("data"):
                add(item["data"], item["status"], item.get("basis", "document_contract"))
    return list(combined.values())


def allowed(recommendation):
    return (not recommendation.get("advisory_only")
            and recommendation.get("execution_release", {}).get("status", "released") == "released")


def require(recommendations):
    blocked = [item["task"] for item in recommendations if not allowed(item)]
    if blocked:
        raise ValueError("留出回检未放行：" + "、".join(dict.fromkeys(blocked)))


def require_whole_scope(recommendations):
    require([item for item in recommendations if item.get("execution_release")
             and not item.get("advisory_only") and item.get("steps")])


def explanation(release):
    labels = {
        "precipitation_time_bounds_or_unit_missing": "降水单位或统计时间区间缺少可核实依据",
        "target_grid_or_holdout_evidence_missing": "目标网格、覆盖或留出块验证不足",
        "missing_engineering_reference": "工程引用尚未核对",
        "invalid_netcdf_time_coordinates": "时间轴存在重复、无效坐标或重叠区间",
        "heldout_worksheet_field_semantics_need_declaration": "需要补充留出工作表的字段定义",
        "algorithm_or_independent_reference_not_verified": "留出数据尚未验证该算法",
        "selected_table_unit_missing": "表格数值列的单位尚未确认",
        "new_independent_evidence_required": "全量发送后需要未发送的独立数据复验",
        "formal_acceptance_sample_failed": "正式抽检发现不合格的留出单位",
    }
    reasons = [labels.get(item["reason"], "请补充字段定义、工程证据或留出回检")
               for item in release.get("blockers", ())]
    return "待放行 · " + "；".join(dict.fromkeys(reasons or ["请查看本地回检记录"]))


def runtime(scan, task, folder):
    """Read original inputs in bounded windows immediately before the real executor."""
    from .sampling_plan import netcdf_time_review

    assert_current(scan)
    reports = []
    for index, name, fid in task_fields(task):
        row = scan["rows"][index]
        field = resolve_field(row, index, name, fid)
        if field is None:
            raise ValueError("执行字段已变化")
        if row["format"] == "NetCDF":
            evidence = netcdf_time_review(row, field, allow_materialization=True)
            if evidence and any(evidence[key] for key in (
                    "nonincreasing", "invalid_coordinates", "invalid_bounds", "overlapping_bounds")):
                raise ValueError("正式读取发现 NetCDF 时间戳重复、无效或时间区间重叠")
            record = dict(source=row["name"], time=evidence)
            if evidence:
                from .segment_review import review_netcdf
                changed = bool(evidence["interval_changes"] or evidence["interval_width_changes"])
                segmented = review_netcdf(row, field, evidence,
                    precipitation=bool(changed and
                                       "engineering.precipitation_adaptation" in task.get("steps", ())))
                record["segmented_recheck"] = segmented
                if changed and segmented["status"] != "checked":
                    failing = [f"{item['start']}–{item['stop']}" for item in segmented.get("segments", ())
                               if item["status"] != "checked"]
                    raise ValueError("NetCDF 频率分段复验未通过：" + ("、".join(failing) or segmented.get("reason", "待核实")))
            reports.append(record)
        elif row["format"] == "GeoTIFF":
            reports.append(dict(source=row["name"], **raster_review(row, field)))
        elif row["format"] in {"CSV", "TSV", "Excel"} and not row.get("original_reader"):
            reports.append(dict(source=row["name"], **table_review(row, field, folder)))
    assert_current(scan)
    return reports


def task_fields(task):
    for item in task.get("fields", ()):
        if isinstance(item, dict):
            yield item["index"], item["name"], item.get("field_id")
        elif isinstance(item, (list, tuple)) and len(item) in {2, 3} and isinstance(item[1], str):
            yield (*item, None) if len(item) == 2 else tuple(item)


def resolve_field(row, index, name, fid):
    matches = [field for offset, field in enumerate(row["fields"], 1)
               if field["name"] == name and (fid is None or fid == f"D{index+1}.F{offset}")]
    return matches[0] if len(matches) == 1 else None


def precipitation_trial(held, field):
    import numpy as np
    from netCDF4 import num2date
    from ecodata.core.exceptions import EcoDataError
    from ecodata.engines.units import UnitEngine

    unit = held.get("unit")
    if not unit:
        return None
    engine = UnitEngine()
    operation, factor = None, None
    for target in ("mm/s", "kg/m^2/s"):
        try:
            factor, offset = engine.affine(unit, target)
            if offset == 0:
                operation = "rate_to_interval_amount"
                break
        except (ValueError, EcoDataError):
            pass
    if not operation:
        methods = str(field.get("time_cell_methods", "")).casefold()
        if "sum" not in methods and field.get("temporal_semantics") not in {"accumulated", "accumulation", "interval_sum"}:
            return None
        try:
            factor, offset = engine.affine(unit, "mm")
            if offset != 0:
                return None
            operation = "confirmed_interval_amount"
        except (ValueError, EcoDataError):
            return None
    count = 0
    for window in held.get("windows", ()):
        clock = window.get("time", {})
        if not clock.get("intervals_valid") or not clock.get("bounds") or not clock.get("units"):
            return None
        dates = num2date(clock["bounds"], clock["units"], calendar=clock.get("calendar") or "standard")
        durations = np.asarray([(stop - start).total_seconds() for start, stop in dates])
        original = np.asarray(window.get("local_values", ()), dtype=float)
        if not len(original) or len(original) % len(durations):
            return None
        values = original.reshape(len(durations), -1) * factor
        if operation == "rate_to_interval_amount":
            values *= durations[:, None]
        finite = values[np.isfinite(values)]
        if (finite < 0).any():
            return None
        count += int(finite.size)
    return dict(operation=operation, unit="mm", valid_samples=count) if count else None


def raster_review(row, field):
    import numpy as np
    import rasterio
    from .sampling_plan import raster_windows

    valid = missing = 0
    with local_member(row) as path, rasterio.open(path) as ds:
        if not ds.crs or any(not np.isfinite(v) or v <= 0 for v in ds.res):
            raise ValueError("正式读取发现栅格坐标系或分辨率缺失")
        band = field["band"]
        for window in raster_windows(ds, band):
            array = ds.read(band, window=window, masked=True)
            count = int(np.isfinite(array.compressed()).sum())
            valid += count
            missing += array.size - count
    if not valid:
        raise ValueError("正式读取发现栅格没有有效像元")
    return dict(valid_cells=valid, missing_cells=missing)


def table_review(row, field, folder):
    from contextlib import closing
    import sqlite3
    from tempfile import TemporaryDirectory
    import pandas as pd
    from itertools import islice
    from .sampling_plan import IDENTIFIERS, TIME, normalized, records
    from .sample_values import numeric

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with records(row, field, encoding) as (rows, _, __), TemporaryDirectory(dir=folder) as temporary:
                header = list(next(rows, ()))
                label = field.get("source_column") or field.get("original_header") or field["name"]
                col = header.index(label)
                names = [normalized(item) for item in header]
                clock = next((i for i, name in enumerate(names) if name in TIME), None)
                station = next((i for i, name in enumerate(names) if name in IDENTIFIERS - {"id"}), None)
                units = next((i for i, name in enumerate(names) if name in {"unit", "units", "单位"}), None)
                declared = field.get("unit")
                total = missing = 0
                unit_ranges = []
                distinct = set()
                current_range = None
                segment_count = 0
                with closing(sqlite3.connect(Path(temporary) / "time.sqlite")) as seen:
                    seen.execute("CREATE TABLE records (station TEXT, instant TEXT, PRIMARY KEY (station, instant)) WITHOUT ROWID")
                    while batch := list(islice(rows, 4096)):
                        total += len(batch)
                        missing += sum(value is None for value in numeric(
                            record[col] if len(record) > col else None for record in batch))
                        if units is not None:
                            observed = {str(record[units]).strip() for record in batch if len(record) > units and record[units]}
                            for position, record in enumerate(batch, total - len(batch) + 1):
                                unit = (str(record[units]).strip() if len(record) > units and record[units] else declared)
                                distinct.add(unit)
                                if current_range is None or current_range["unit"] != unit:
                                    current_range = dict(start=position, stop=position, unit=unit)
                                    segment_count += 1
                                    if len(unit_ranges) < 64:
                                        unit_ranges.append(current_range)
                                else:
                                    current_range["stop"] = position
                        if clock is not None:
                            dates = pd.to_datetime([str(record[clock]).replace("_", "T", 1)
                                                    if len(record) > clock and record[clock] is not None else None for record in batch],
                                                   format="mixed", utc=True, errors="coerce")
                            if dates.isna().any():
                                raise ValueError("正式读取发现日期缺失或无法解析")
                            entries = [(str(record[station]) if station is not None and len(record) > station else "",
                                        stamp.isoformat()) for record, stamp in zip(batch, dates)]
                            try:
                                seen.executemany("INSERT INTO records VALUES (?, ?)", entries)
                            except sqlite3.IntegrityError as error:
                                raise ValueError("正式读取发现同一站点的重复时刻") from error
                if not total or missing == total:
                    raise ValueError("正式读取发现表格没有有效数值")
                if len(distinct) > 1 or (declared and distinct and distinct != {declared}):
                    from ecodata.core.exceptions import EcoDataError
                    from ecodata.engines.units import UnitEngine

                    statuses = {}
                    for unit in distinct:
                        try:
                            UnitEngine().affine(unit, declared or unit_ranges[0]["unit"])
                        except (EcoDataError, ValueError, TypeError):
                            statuses[unit] = "failed"
                        else:
                            statuses[unit] = "checked"
                    display = "、".join(f"第 {item['start']}–{item['stop']} 条 {item['unit'] or '单位未知'} {statuses[item['unit']]}"
                                      for item in unit_ranges[:10])
                    raise ValueError(f"表格已分成 {segment_count} 段并复查单位。当前整表方案不能安全覆盖这些分段：" + display)
                return dict(records=total, missing=missing, unit=declared or next(iter(distinct), None),
                            unit_segments=unit_ranges)
        except UnicodeDecodeError:
            if encoding == "gb18030":
                raise
