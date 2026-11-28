"""Preplanned finite-population acceptance sampling on reserved units."""
from fractions import Fraction
from math import ceil, floor
from random import Random
from secrets import randbits

BAYESIAN_RULE = "finite_population_uniform_prior"
BAYESIAN_TARGET = 0.95
BAYESIAN_CHECKPOINT = 126


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
    unit = units.get(audit.get("unit"), "组")
    if audit.get("acceptance_rule") == BAYESIAN_RULE:
        if audit.get("census"):
            tally = (f"本批共 {audit['population']} {unit}，已逐组复跑，不适配 {audit['failed']}，"
                     f"无法评分 {audit['unscored']}。")
            if audit["posterior_probability"] is None:
                return tally + "缺少评分依据，暂不放行。"
            return (tally + f"本批不适配率 {audit['failed'] / audit['population']:.2%}，要求低于 5%。"
                    "结论仅适用于本批回检总体。")
        if strata := audit.get("strata"):
            tally = "，".join(
                f"{item['label']} {item['population']} {unit}，已抽 {item['inspected']}，"
                f"不适配 {item['failed']}，无法评分 {item['unscored']}"
                for item in strata)
            probability = audit["posterior_probability"]
            if probability is None:
                return f"{tally}。同量回检的评分证据不足，未计算整批放行概率。"
            note = "已调组结果反映本批复跑，泛化表现看未调组。"
            best_possible = audit.get("best_possible_probability")
            if best_possible is not None and best_possible < BAYESIAN_TARGET:
                note += "按两类现有组数，即使等量抽满且全部通过，后验也达不到阈值。"
            if audit.get("single_reference_per_unit"):
                note += "每个站点时刻只有一项实测，已调组使用相同实测复跑。"
            return (f"{tally}。按两类实际规模计算，整体不适配率低于 5% 的后验概率为 "
                    f"{probability:.2%}，要求至少 {BAYESIAN_TARGET:.0%}（各类均匀先验）。"
                    f"{note}")
        note = (f"此前用于修改的 {audit['excluded_as_failures']} 个单元按最坏情况计算。"
                if audit.get("excluded_as_failures") else "")
        tally = (f"留出总体 {audit['population']} {unit}，抽检 {audit['inspected']}，其中 {audit['failed']} 不适配，"
                 f"{audit['unscored']} 无法评分。")
        if audit['posterior_probability'] is None:
            return f"{tally}评分证据不足，未给出整批放行概率。{note}"
        if audit.get('revision_required') and audit['inspected'] < audit['sample_size']:
            note += (f"即使剩余单元全部通过，抽满 {audit['sample_size']} 个的后验概率也最多为 "
                     f"{audit['best_possible_probability']:.2%}，已停止本方案抽检。")
        return (f"{tally}整体不适配率低于 5% 的后验概率为 "
                f"{audit['posterior_probability']:.2%}，要求至少 {BAYESIAN_TARGET:.0%}（均匀先验）。{note}")
    if audit["sample_size"] is None:
        return f"可抽 {audit['population']} {unit}。当前试跑预算不足以满足设定的误判风险。"
    return (f"可抽 {audit['population']} {unit}，计划试跑 {audit['sample_size']} 组，"
            f"最多允许 {audit['allowed_failures']} 组不合格。已查 {audit['inspected']} 组，"
            f"其中 {audit['failed']} 组不合格，{audit.get('unscored', 0)} 组缺少评分依据。")


def posterior_below_five_percent(population, scored, failed, *, excluded_as_failures=0):
    """Uniform prior on D, conditioned on a simple random sample without replacement."""
    from scipy.stats import hypergeom

    remaining = population - excluded_as_failures
    if not 0 <= failed <= scored <= remaining or population < 1 or excluded_as_failures < 0:
        raise ValueError("贝叶斯验收要求 0 ≤ 不适配数 ≤ 已评分数 ≤ 总体数")
    # The posterior of D+1 is the (failed+1)th order statistic among
    # scored+1 draws from 1..population+1. This evaluates the exact sum
    # without enumerating each possible number of failures in a large lot.
    below = (population + 19) // 20 - excluded_as_failures
    if below <= 0:
        return 0.0
    return float(hypergeom.sf(failed, remaining + 1, below, scored + 1))


def stratified_posterior(sizes, scored, failed):
    """Posterior of the sum of the two finite failure counts under independent uniform priors."""
    import numpy as np
    from scipy.stats import betabinom, hypergeom

    old, new = sizes
    n_old, n_new = scored
    d_old, d_new = failed
    if old + new < 1 or any(not 0 <= d <= n <= size for size, n, d in zip(sizes, scored, failed)):
        raise ValueError("回检分层计数不一致")
    hidden = np.arange(old - n_old + 1)
    cutoff = (old + new + 19) // 20
    remaining = cutoff - d_old - hidden
    masses = betabinom.pmf(hidden, old - n_old, d_old + 1, n_old - d_old + 1)
    chances = hypergeom.sf(d_new, new + 1, np.clip(remaining, 0, new + 1), n_new + 1)
    return float(np.clip(masses @ chances, 0, 1))


def sample_other_units(population, groups, count, rng):
    """Sample ranks from the complement without allocating the whole population."""
    excluded = sorted(groups)
    selected = []
    for rank in rng.sample(range(population - len(excluded)), count):
        for index in excluded:
            if index > rank:
                break
            rank += 1
        selected.append(rank)
    return selected


def evaluate_revision_sample(population, unit, test, scope):
    """Evaluate development cohorts or an equal-sized, frozen two-stratum sample."""
    tuning = scope["mode"] == "development"
    census = scope["mode"] == "census"
    old = scope["previous"]
    new = scope["development_new"]
    participating = set(old) | set(new)
    if (len(participating) != len(old) + len(new) or population != scope["population"]
            or any(not 0 <= index < population for index in participating)):
        raise ValueError("回检单元与锁定总体不一致")
    if tuning:
        sizes = len(old), len(new)
        queues = old, new
        labels = "前轮已调组", "本轮新增调参组"
    elif census:
        sizes = (population,)
        queues = (range(population),)
        labels = ("本批全部组",)
    else:
        sizes = len(participating), population - len(participating)
        quota = min(*sizes, BAYESIAN_CHECKPOINT // 2)
        rng = Random(scope["seed"])
        queues = (rng.sample(sorted(participating), quota),
                  sample_other_units(population, participating, quota, rng))
        labels = "已调组", "未调组"
    strata = [dict(label=label, population=size, inspected=0, scored=0, failed=0, unscored=0)
              for label, size in zip(labels, sizes)]
    failures, unscored, checked = [], [], []

    def inspect(index, stratum):
        entry = strata[stratum]
        entry["inspected"] += 1
        checked.append(index)
        outcome = test(index)
        if outcome and outcome.get("unscored"):
            unscored.append(outcome)
            entry["unscored"] += 1
        else:
            entry["scored"] += 1
            if outcome:
                failures.append(outcome)
                entry["failed"] += 1

    probability = None
    best_possible = None
    if tuning or census:
        for stratum, indices in enumerate(queues):
            for index in indices:
                inspect(index, stratum)
        if census and not unscored:
            probability = float(20 * len(failures) < population)
    else:
        best_possible = stratified_posterior(sizes, (quota, quota), (0, 0)) if quota else None
        for pair in zip(*queues):
            for stratum, index in enumerate(pair):
                inspect(index, stratum)
            if not unscored:
                probability = stratified_posterior(sizes, [item["scored"] for item in strata],
                                                   [item["failed"] for item in strata])
                if probability >= BAYESIAN_TARGET:
                    break
                if failures:
                    optimistic = stratified_posterior(sizes, (quota, quota), [item["failed"] for item in strata])
                    if optimistic < BAYESIAN_TARGET:
                        break
    passed = probability is not None and not unscored and probability >= BAYESIAN_TARGET
    insufficient = best_possible is not None and best_possible < BAYESIAN_TARGET
    return dict(status="checked" if passed else "pending" if tuning or unscored or not checked or insufficient else "failed",
                acceptance_rule=BAYESIAN_RULE, prior="complete_population" if census else
                "independent_uniform_on_stratum_failure_counts",
                unit=unit, population=population, available_population=population, excluded_as_failures=0,
                sample_size=sum(map(len, queues)), allowed_failures=None, inspected=len(checked),
                scored=len(checked)-len(unscored), failed=len(failures), failures=failures,
                unscored=len(unscored), unscored_units=unscored, inspected_indices=checked,
                posterior_probability=None if unscored else probability, posterior_target=BAYESIAN_TARGET,
                rejectable_share=0.05, revision_required=False, strata=strata,
                development_only=tuning, census=census, best_possible_probability=best_possible)


def evaluate_bayesian_recipe(population, unit, test, *, excluded=(), seed=None):
    excluded = set(excluded)
    if any(not 0 <= index < population for index in excluded):
        raise ValueError("上一轮抽检单元与本轮数据总体不一致")
    remaining = population - len(excluded)
    if remaining < 1:
        raise ValueError("无未参与前轮反馈的验收单元")
    limit = min(remaining, BAYESIAN_CHECKPOINT)
    if seed is None:
        seed = randbits(64)
    selected = (i for i in Random(seed).sample(range(population), limit + len(excluded)) if i not in excluded)
    failures, unscored, checked = [], [], []
    scored = 0
    def posterior(n, d):
        return posterior_below_five_percent(population, n, d, excluded_as_failures=len(excluded))

    probability = posterior(0, 0)
    best_possible = posterior(limit, 0)
    for index in selected:
        checked.append(index)
        outcome = test(index)
        if outcome and outcome.get("unscored"):
            unscored.append(outcome)
        else:
            scored += 1
            if outcome:
                failures.append(outcome)
                best_possible = posterior(limit, len(failures))
            probability = posterior(scored, len(failures))
        if probability >= BAYESIAN_TARGET and not unscored:
            break
        # With the full retry budget available, stop as soon as even an
        # all-passing remainder cannot change the recommendation decision.
        if limit == BAYESIAN_CHECKPOINT and not unscored and best_possible < BAYESIAN_TARGET:
            break
    passed = probability >= BAYESIAN_TARGET and not unscored
    revise = limit == BAYESIAN_CHECKPOINT and not unscored and best_possible < BAYESIAN_TARGET
    return dict(status="checked" if passed else "failed" if not unscored else "pending",
                acceptance_rule=BAYESIAN_RULE, prior="uniform_on_finite_failure_count", unit=unit,
                population=population, available_population=remaining, excluded_as_failures=len(excluded),
                sample_size=limit, allowed_failures=None,
                inspected=len(checked), scored=scored, failed=len(failures), failures=failures,
                unscored=len(unscored), unscored_units=unscored, sampling_seed=seed,
                inspected_indices=checked, posterior_probability=None if unscored else probability,
                posterior_target=BAYESIAN_TARGET, best_possible_probability=best_possible,
                rejectable_share=0.05, revision_required=revise,
                remaining_after_review=remaining - len(checked))


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
