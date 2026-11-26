"""Preplanned finite-population acceptance sampling on reserved units."""
from fractions import Fraction
from math import ceil, floor
from random import Random
from secrets import randbits


def settings():
    from os import environ
    return dict(defect_share=float(environ.get("ECODATA_ACCEPTANCE_DEFECT_SHARE", "0.05")),
                detection=float(environ.get("ECODATA_ACCEPTANCE_DETECTION", "0.95")))


def attach(scan, index, field, report, cache):
    basis, partition = report.get("holdout_basis"), report.get("partition") or {}
    key = ("acceptance_sampling", index, report["field"], basis, tuple(sorted(partition.items())))
    if key not in cache:
        cache[key] = review_field(scan["rows"][index], field, basis=basis, partition=partition, **settings())
    result = cache[key]
    if result is not None:
        report["acceptance_sampling"] = result
        if result["status"] == "failed":
            report.update(status="failed", basis="formal_acceptance_sample_failed")
    return report


def plan(population, *, defect_share=0.05, detection=0.95):
    from scipy.stats import hypergeom

    if population < 1 or not 0 < defect_share <= 1 or not 0 < detection < 1:
        raise ValueError("抽检总体、目标问题占比或检出概率无效")
    defects = max(1, ceil(population * Fraction(str(defect_share))))
    low, high = 1, population
    while low < high:
        middle = (low + high) // 2
        if hypergeom.pmf(0, population, defects, middle) <= 1 - detection:
            high = middle
        else:
            low = middle + 1
    return dict(population=population, sample_size=low, rounds=1, defect_share=defect_share,
                detection=detection, assumed_defects=defects)


def evaluate(population, unit, test, *, seed=0, defect_share=0.05, detection=0.95, offset=0):
    """Select all sample positions before evaluating the fixed acceptance rule."""
    design = plan(population, defect_share=defect_share, detection=detection)
    selected = sorted(Random(seed).sample(range(population), design["sample_size"]))
    result = [test(offset + index) for index in selected]
    failures = [item for item in result if item]
    return dict(design, unit=unit, inspected=len(result), failed=len(failures),
                status="failed" if failures else "checked", failures=failures[:20])


def recipe_plan(population, *, acceptable=0.05, rejectable=0.10, false_reject=0.05,
                false_accept=0.05, budget=512, strict_boundary=False, zero_failures=False):
    """Find the smallest finite-population plan satisfying both risk bounds."""
    from scipy.stats import hypergeom

    if (population < 1 or budget < 1 or not 0 <= acceptable <= rejectable <= 1
            or not 0 < false_reject < 1 or not 0 < false_accept < 1
            or false_reject + false_accept >= 1):
        raise ValueError("验收总体或风险设置无效")
    good = floor(population * Fraction(str(acceptable)))
    boundary = population * Fraction(str(rejectable))
    bad = max(good + 1, floor(boundary) + 1 if strict_boundary else ceil(boundary))
    if zero_failures:
        bad = max(1, ceil(boundary))
    if bad > population:
        raise ValueError("可接受质量与应驳回质量没有可区分的总体")
    limit = min(population, budget)
    design = dict(population=population, design_good_share=acceptable, acceptable_share=rejectable if strict_boundary else acceptable,
                  rejectable_share=rejectable, strict_boundary=strict_boundary,
                  good_failures=good, bad_failures=bad, false_reject_limit=false_reject,
                  false_accept_limit=false_accept, budget=limit, rounds=1)
    if zero_failures:
        design.update(acceptance_rule="zero_failures", acceptable_share=rejectable,
                      design_good_share=None, good_failures=None, false_reject_limit=None)
        n = plan(population, defect_share=rejectable, detection=1 - false_accept)["sample_size"]
        if n <= limit:
            return dict(design, sample_size=n, allowed_failures=0,
                        false_accept_risk=float(hypergeom.pmf(0, population, bad, n)))
        return dict(design, sample_size=None, allowed_failures=None)
    for size in range(1, limit + 1):
        allowed = int(hypergeom.ppf(1 - false_reject, population, good, size))
        miss = float(hypergeom.cdf(allowed, population, bad, size))
        if miss <= false_accept:
            return dict(design, sample_size=size, allowed_failures=allowed, false_accept_risk=miss,
                        false_reject_risk=float(hypergeom.sf(allowed, population, good, size)))
    return dict(design, sample_size=None, allowed_failures=None)


def recipe_settings():
    from os import environ

    good_quality = environ.get("ECODATA_RECIPE_ACCEPTABLE_SHARE")
    return dict(acceptable=float(good_quality if good_quality is not None else "0.01"),
                rejectable=float(environ.get("ECODATA_RECIPE_REJECTABLE_SHARE", "0.05")),
                false_reject=float(environ.get("ECODATA_RECIPE_FALSE_REJECT", "0.05")),
                false_accept=float(environ.get("ECODATA_RECIPE_FALSE_ACCEPT", "0.05")),
                budget=int(environ.get("ECODATA_RECIPE_TRIAL_BUDGET", "512")),
                strict_boundary=good_quality is not None, zero_failures=good_quality is None)


def evaluate_recipe(population, unit, test, *, seed=None, **policy):
    design = recipe_plan(population, **policy)
    size, allowed = design["sample_size"], design["allowed_failures"]
    if size is None:
        return dict(design, status="pending", unit=unit, inspected=0, failed=0, failures=[], unscored=0)
    # Draw only after the recipe is frozen. Store the seed for local replay.
    if seed is None:
        seed = randbits(64)
    failures = []
    unscored = []
    def result(status, inspected):
        return dict(design, status=status, unit=unit, inspected=inspected, sampling_seed=seed,
                    failed=len(failures), failures=failures[:20], unscored=len(unscored),
                    unscored_units=unscored[:20])
    for inspected, index in enumerate(Random(seed).sample(range(population), size), 1):
        if failure := test(index):
            if failure.get("unscored"):
                unscored.append(failure)
                continue
            failures.append(failure)
            if len(failures) > allowed:
                return result("failed", inspected)
    return result("pending" if unscored else "checked", size)


def recipe_summary(audit):
    units = {"withheld_spatial_block": "空间块", "withheld_time_and_spatial_block": "时空块",
             "withheld_station": "个站点", "withheld_month": "个月", "withheld_utc_day": "个 UTC 日",
             "withheld_day": "天"}
    unit = units.get(audit["unit"], "组")
    if audit["sample_size"] is None:
        return f"可抽 {audit['population']} {unit}。当前试跑预算不足以满足设定的误判风险。"
    return (f"可抽 {audit['population']} {unit}，计划试跑 {audit['sample_size']} 组，"
            f"最多允许 {audit['allowed_failures']} 组不合格。已查 {audit['inspected']} 组，"
            f"其中 {audit['failed']} 组不合格，{audit.get('unscored', 0)} 组缺少评分依据。")


def review_field(row, field, *, basis=None, partition=None, defect_share=0.05, detection=0.95):
    """Check uniformly chosen withheld raster blocks or time windows."""
    import numpy as np
    from .scan import local_member
    from .sampling_plan import member_available, open_netcdf_for_sampling, time_dimension

    if not member_available(row) or row.get("original_reader"):
        return None
    options = dict(defect_share=defect_share, detection=detection)
    partition = partition or {}
    if row["format"] == "GeoTIFF":
        import rasterio
        from rasterio.windows import Window
        with local_member(row) as path, rasterio.open(path) as source:
            along_y = source.height >= 2
            size = source.height if along_y else source.width
            if size < 2:
                return None
            boundary = size - max(1, size // 5)
            x0, y0 = (0, boundary) if along_y else (boundary, 0)
            width, height = source.width - x0, source.height - y0
            cols, rows = ceil(width / 4), ceil(height / 4)
            def test(index):
                i, j = divmod(index, cols)
                block = source.read(field["band"], window=Window(
                    x0 + 4*j, y0 + 4*i, min(4, width - 4*j), min(4, height - 4*i)), masked=True)
                if not np.isfinite(block.compressed()).any():
                    return dict(unit=index, rule="heldout_block_has_no_valid_values")
                return None
            return evaluate(cols * rows, "withheld_spatial_block", test, **options)
    if row["format"] == "NetCDF":
        with local_member(row) as path, open_netcdf_for_sampling(path) as dataset:
            array = dataset[field["name"]]
            clock = time_dimension(array, dataset)
            axis = partition.get("axis", clock)
            if axis not in array.dims:
                return None
            size = array.sizes[axis]
            if size < 2:
                return None
            start = partition.get("holdout_start", size - max(1, size // 5))
            temporal = axis == clock
            numeric_clock = temporal and clock in dataset.coords and np.issubdtype(dataset[clock].dtype, np.number)
            selectors = {dim: (array.sizes[dim] - 1) // 2 for dim in array.dims if dim != axis}
            def test(index):
                bounds = slice(start + index*8, min(start + (index+1)*8, size))
                if numeric_clock:
                    clock_values = dataset[clock].isel({clock: bounds}).values.astype(float)
                    if not np.isfinite(clock_values).all() or (np.diff(clock_values) <= 0).any():
                        return dict(unit=index, rule="time_window_coordinates_invalid")
                values = array.isel({axis: bounds, **selectors}).values
                if not np.isfinite(values).any():
                    return dict(unit=index, rule="holdout_window_has_no_valid_values")
                return None
            return evaluate(ceil((size - start) / 8),
                            "withheld_time_window" if temporal else "withheld_spatial_window", test, **options)
    if row["format"] in {"CSV", "TSV", "Excel"}:
        return table_review_field(row, field, basis=basis, partition=partition, **options)
    return None


def table_review_field(row, field, *, basis, partition, **options):
    from .csv_index import month
    from .sample_values import field_values
    from .sampling_plan import IDENTIFIERS, PROBE_LIMIT, TIME, normalized, records, station_reserved
    from datetime import datetime
    from functools import lru_cache
    from itertools import chain, islice

    @lru_cache(maxsize=4096)
    def valid_date(stamp):
        try:
            datetime.fromisoformat(stamp.replace("_", "T", 1))
            return True
        except ValueError:
            return False

    for encoding in ("utf-8-sig", "gb18030"):
        try:
            held_sheet = partition.get("held_sheet")
            with records(row, field, encoding, held_sheet) as (rows, _, __):
                header = list(next(rows, ()))
                name = field.get("source_column") or field.get("original_header") or field["name"]
                if name not in header:
                    return None
                column = header.index(name)
                names = [normalized(name) for name in header]
                site = next((index for index, name in enumerate(names) if name in IDENTIFIERS - {"id"}), None)
                clock = next((index for index, name in enumerate(names) if name in TIME), None)
                prefix = list(islice(rows, PROBE_LIMIT))
                known_months = sorted({date for record in prefix if clock is not None and len(record) > clock
                                       if (date := month(record[clock]))})
                reserved = (partition.get("held_month") if basis == "calendar_months"
                            else known_months[-1] if len(known_months) > 1 else None)
                use_station = site is not None and basis != "calendar_months" and not held_sheet
                use_month = clock is not None and not use_station and not held_sheet
                summary = {}
                for index, record in enumerate(chain(prefix, rows)):
                    if len(record) <= column:
                        continue
                    stamp = record[clock] if clock is not None and len(record) > clock else None
                    if held_sheet:
                        key = held_sheet
                    elif use_station:
                        key = str(record[site]).strip() if len(record) > site else ""
                        if not key or not station_reserved(key):
                            continue
                    elif use_month:
                        key = month(stamp)
                        if reserved is None or key != reserved:
                            continue
                    else:
                        if (index // 8) % 4 != 3:
                            continue
                        key = str(index // 8)
                    previous = summary.get(key, (False, True))
                    # A station with invalid dates is a defective unit.
                    valid = next(field_values([record[column]],field)) is not None
                    dated = bool(stamp and valid_date(str(stamp))) if clock is not None else True
                    summary[key] = (previous[0] or valid, previous[1] and dated)
                if not summary:
                    return None
                units = list(summary.values())
                unit = ("withheld_worksheet" if held_sheet else "withheld_station" if use_station
                        else "withheld_month" if use_month else "withheld_record_block")
                return evaluate(len(units), unit,
                                lambda i: None if all(units[i]) else dict(unit=i, rule="tabular_unit_incomplete"),
                                **options)
        except UnicodeDecodeError:
            if encoding == "gb18030":
                raise
