"""Local trial and formal screen for DeepSeek's station/grid proposals."""
from contextlib import ExitStack
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ecodata.algorithms.paired_station_search import correction_coefficients
from ecodata.core.exceptions import EcoDataError

from . import indexed_table_recipe, recipe_trials, station_grid
from .local_search import model_request
from .sample_values import RECOMMENDATION_SAMPLE_LIMIT, export_sample
from .scan import assert_current

MIN_FORMAL = 6
MIN_TRAIN = 3
MIN_DEVELOPMENT = 3
MAX_DEVELOPMENT_BLOCKS = 3
DEVELOPMENT_TIMESTAMPS = 2 * MAX_DEVELOPMENT_BLOCKS
RATIO = "bias.prior_series_ratio"
UNCALIBRATED = "original_grid"
RETAIN = "retain_grid"
TIME_METHODS = (station_grid.EXACT, station_grid.NEAREST)
CORRECTIONS = {
    UNCALIBRATED: "保留已有实测，缺测处用同一格点的原值补齐",
    station_grid.BIAS: "训练段内逐站计算平均残差，对原网格作加性偏差订正",
    RATIO: "训练段内逐站计算实测总和与原网格总和之比，对原网格作比例订正",
}
READ_ERRORS = (EcoDataError, OSError, RuntimeError, ValueError, KeyError, TypeError, IndexError, ImportError)


def components(execution):
    corrections = dict(CORRECTIONS)
    if execution["unit"].strip().casefold() not in {"k", "kelvin"} and execution["quantity"] not in {
            "precipitation_amount", "precipitation_flux", "precipitation_rate"}:
        corrections.pop(RATIO)
    return dict(time_matching={method: station_grid.NAMES[method] for method in TIME_METHODS},
                correction=corrections,
                fixed_processing="统一单位和数据结构，匹配原生最近格点，保留已有实测，预测值只用于补缺")


def proposal_task(task, execution, methods):
    available = components(execution)
    available["time_matching"] = {method: available["time_matching"][method] for method in methods}
    return {**{key: task[key] for key in ("id", "context", "verified_checks", "engineering_note")},
            "executable_components": available,
            "source_sample_counts": task["pretrial_evidence"]["source_sample_counts"]}


def training_evidence(task, execution, trial):
    from copy import deepcopy

    from .sampling_plan import distributed

    task = deepcopy(task)
    training = trial.training.iloc[distributed(range(len(trial.training)), RECOMMENDATION_SAMPLE_LIMIT)]
    for field in task["context"].get("pair_details", ()):
        fid = field["id"]
        source = fid.split(".", 1)[0]
        if source == f"D{execution['grid_index'] + 1}":
            values = training.background
            if execution.get("wind_components"):
                field["sample_quantity"] = "wind_speed"
                field["sample_derivation"] = "CAMx 10 m wind speed from both components"
        elif source == f"D{execution['table_index'] + 1}":
            values = training.observed
            if execution.get("wind_components"):
                field["sample_quantity"] = "wind_speed"
        else:
            continue
        samples = export_sample(values, limit=RECOMMENDATION_SAMPLE_LIMIT)["perturbed_samples"]
        field.update(perturbed_samples=samples, recommendation_sample_basis="station_training_partition",
                     sample_unit=execution["unit"])
        task["source_sample_counts"][fid] = len(samples)
    return task


def proposals(reply, tasks, *, mock=False):
    """Translate the model's component choices to executable, canonical recipes."""
    choices = reply.get("proposals")
    if not isinstance(choices, list):
        choices = []
    parsed = {task["id"]: [] for task in tasks}
    for task in tasks:
        available = task["executable_components"]
        task_choices = choices
        if mock:
            task_choices = [dict(task=task["id"], time_matching=next(iter(available["time_matching"])),
                                 correction=correction, reason="模拟提议，用于验证本地试跑流程。")
                            for correction in available["correction"]]
        seen = set()
        for choice in task_choices:
            if not isinstance(choice, dict) or choice.get("task") != task["id"]:
                continue
            time, correction = choice.get("time_matching"), choice.get("correction")
            if (not isinstance(time, str) or time not in available["time_matching"]
                    or not isinstance(correction, str) or correction not in available["correction"]
                    or (time, correction) in seen):
                continue
            seen.add((time, correction))
            steps = [station_grid.UNIT_STEP, station_grid.FORMAT_STEP, time, station_grid.MATCH, recipe_trials.FILL]
            if correction != UNCALIBRATED:
                steps.append(correction)
            description = f"{available['time_matching'][time]}，{available['correction'][correction]}"
            parsed[task["id"]].append(dict(id=f"DS{len(seen)}", name=description, steps=steps,
                                          description=description, time_matching=time, correction=correction,
                                          proposal_reason=str(choice.get("reason", ""))[:600]))
            if len(seen) == 4:
                break
    return parsed


@dataclass
class Trial:
    training: pd.DataFrame
    development: pd.DataFrame
    formal: pd.DataFrame
    method_counts: dict

    def score(self, option, points, request):
        baseline = points.background.to_numpy(float)
        correction = option["correction"]
        candidate = baseline
        if correction != UNCALIBRATED:
            coefficients = {station: correction_coefficients(group)
                            for station, group in self.training.groupby("station")}
            if correction == station_grid.BIAS:
                candidate = baseline + points.station.map({k: v[0] for k, v in coefficients.items()}).to_numpy(float)
            else:
                if (self.training[["observed", "background"]].lt(0).any().any() or (baseline < 0).any()
                        or any(not np.isfinite(ratio) for _, ratio in coefficients.values())):
                    return None, "ratio_training_unavailable"
                candidate = baseline * points.station.map({k: v[1] for k, v in coefficients.items()}).to_numpy(float)
        return recipe_trials.score(candidate, baseline, points.observed.to_numpy(float),
                                   "heldout_station_observations",
                                   retained=self.training.background.to_numpy(float), request=request)


def sample_development_times(evaluation, times):
    if len(times) <= DEVELOPMENT_TIMESTAMPS:
        return times
    station_sets = evaluation.loc[evaluation.grid_time.isin(times)].groupby("grid_time").station.agg(set)
    covered = set()
    sampled = []
    blocks = np.array_split(np.arange(len(times)), MAX_DEVELOPMENT_BLOCKS)
    for number, block in enumerate(blocks):
        ordered = times[block]
        # Keep the start and end of the window represented. Within each block,
        # give the other slot to stations not yet sampled, then favour time spread.
        if number == len(blocks) - 1:
            ordered = ordered[::-1]
        anchor = ordered[0]
        covered.update(station_sets[anchor])
        companion = max(reversed(ordered[1:]), key=lambda time: len(station_sets[time] - covered))
        covered.update(station_sets[companion])
        sampled.extend((anchor, companion))
    return sampled


def partition(rows):
    # Require the exact same station/time labels and original grid on each trial.
    finite = {method: frame.loc[np.isfinite(frame[["observed", "background"]]).all(axis=1)
                                & frame.matched_observations.eq(1)]
              for method, frame in rows.items()}
    counts = {method: len(frame) for method, frame in finite.items()}
    try:
        common, _ = station_grid.common_queue(finite)
    except ValueError as error:
        if str(error) != "No common actual evaluation queue":
            raise
        common = {method: frame.iloc[:0] for method, frame in finite.items()}
    reference = next(iter(common.values()))
    if any(not np.array_equal(method.background.to_numpy(), reference.background.to_numpy())
           for method in common.values()):
        raise ValueError("不同方案在同一站点和时刻匹配到了不同格点原值，无法使用共同基线。")
    frame = reference.sort_values(["grid_time", "station"])
    clocks = pd.DatetimeIndex(frame.grid_time.unique())
    training_length = max(MIN_TRAIN, min(len(clocks) // 3, len(clocks) - MIN_DEVELOPMENT - MIN_FORMAL))
    training = frame[frame.grid_time.isin(clocks[:training_length])]
    trained = training.groupby("station").size()
    evaluation = frame.loc[~frame.index.isin(training.index) & frame.station.isin(trained[trained >= MIN_TRAIN].index)]
    # Put whole timestamps on the same side of each boundary.
    counts_by_time = evaluation.groupby("grid_time", sort=True).size().cumsum()
    cuts = counts_by_time[counts_by_time.ge(MIN_DEVELOPMENT) & counts_by_time.le(len(evaluation) - MIN_FORMAL)]
    window_length = max(DEVELOPMENT_TIMESTAMPS, len(counts_by_time) // 3)
    within_window = cuts[cuts.index.isin(counts_by_time.index[:window_length])]
    cut = within_window.index[-1] if len(within_window) else cuts.index[0] if len(cuts) else None
    if cut is None:
        return Trial(training, evaluation.iloc[:0], evaluation, counts)
    # Lock the later suffix first. Sample whole timestamps across the earlier
    # window so every eligible station at a selected time is scored together.
    formal = evaluation.loc[evaluation.grid_time > cut]
    trial_times = counts_by_time.index[counts_by_time.index <= cut]
    development = evaluation.loc[evaluation.grid_time.isin(sample_development_times(evaluation, trial_times))]
    return Trial(training, development, formal, counts)


def matched_rows(scan, execution):
    assert_current(scan)
    definition = execution.get("locked_holdout")
    if definition is None:
        raise ValueError(execution.get("holdout_lock_error") or "无法锁定未外发的实测留出范围。")
    row = scan["rows"][execution["table_index"]]
    field = execution.get("table_field") or next(item for item in row["fields"] if item["name"] == execution["table_name"])
    with ExitStack() as stack:
        source = indexed_table_recipe.table(stack, row, field, execution["obs_unit"])
        if station_grid.station_holdout(scan, execution, source, field) != definition:
            raise ValueError("锁定的站点留出范围与当前外发记录不一致。")
        table = pd.DataFrame((record for group in definition["groups"] for record in source.read(group)),
                             columns=source.header)
    if table.empty:
        frame = pd.DataFrame(columns=["station", "grid_time", "observed", "background", "matched_observations"], dtype=float)
        return dict.fromkeys(TIME_METHODS, frame)
    rows = station_grid.load(scan, execution, table_data=table, minimum_grid_index=definition["minimum_grid_index"])
    assert_current(scan)
    return rows


def prepare(scan, execution, methods):
    rows = matched_rows(scan, execution)
    return partition({method: rows[method] for method in methods})


def reserve_trial(scan, execution):
    """Freeze a comparable scoring queue before any algorithm is proposed."""
    rows = matched_rows(scan, execution)
    common = partition(rows)
    if len(common.formal) >= MIN_FORMAL and not common.development.empty:
        return common, TIME_METHODS
    individual = {method: partition({method: rows[method]}) for method in TIME_METHODS}
    viable = [method for method, trial in individual.items()
              if len(trial.formal) >= MIN_FORMAL and not trial.development.empty]
    if viable:
        method = max(viable, key=lambda method: len(individual[method].formal))
        trial = individual[method]
        trial.method_counts = common.method_counts
        return trial, (method,)
    best = max(individual.values(), key=lambda trial: len(trial.formal))
    best.method_counts = common.method_counts
    return best, ()


def score_request(task, target):
    return model_request(dict(task="multisource", variable=task["quantity"], unit=task["unit"],
                              semantic_type="continuous"), target)


def bias_direction(bias):
    if np.isclose(bias, 0., rtol=0, atol=1e-12):
        return "near_zero"
    return "too_high" if bias > 0 else "too_low"


def development_blocks(points):
    times = pd.DatetimeIndex(points.grid_time.unique()).sort_values()
    if times.empty:
        return []
    count = min(len(times), MAX_DEVELOPMENT_BLOCKS, max(2, (len(times) + 1) // 2))
    blocks = []
    first = times[0].to_datetime64()
    hour = np.timedelta64(1, "h")
    for number, block_times in enumerate(np.array_split(times.to_numpy(), count), 1):
        block = points.loc[points.grid_time.isin(block_times)]
        blocks.append((dict(block=f"B{number}", timestamps=len(block_times),
                            start_hour_offset=float((block_times[0] - first) / hour),
                            end_hour_offset=float((block_times[-1] - first) / hour),
                            points=len(block), stations=int(block.station.nunique())), block))
    return blocks


def block_score(trial, option, descriptor, points, request):
    score, issue = trial.score(option, points, request)
    result = dict(descriptor, scored_points=score["count"] if score else 0, failure_reason=issue)
    if score:
        candidate, baseline = score["candidate"], score["baseline"]
        result.update(candidate_bias=candidate["Bias"], baseline_bias=baseline["Bias"],
                      candidate_bias_direction=bias_direction(candidate["Bias"]),
                      baseline_bias_direction=bias_direction(baseline["Bias"]),
                      candidate_rmse=candidate["RMSE"], baseline_rmse=baseline["RMSE"],
                      rmse_change=candidate["RMSE"] - baseline["RMSE"],
                      mae_change=candidate["MAE"] - baseline["MAE"],
                      loss_change=score["candidate_loss"] - score["baseline_loss"],
                      loss_tolerance=score["loss_tolerance"])
    return result


def trial_scores(trial, options, request):
    summaries = []
    blocks = development_blocks(trial.development)
    for option in options:
        score, issue = trial.score(option, trial.development, request)
        if score:
            metrics = {key: score[key] for key in ("candidate", "baseline", "candidate_loss", "baseline_loss",
                                                    "loss_tolerance", "count")}
        else:
            metrics = {"count": 0}
        summaries.append(dict(recipe=option["id"], status="failed" if issue else "scored",
                              failure_reason=issue,
                              time_blocks=[block_score(trial, option, descriptor, points, request)
                                           for descriptor, points in blocks], **metrics))
    return summaries


def revised_option(row, task):
    revision = row.get("revision")
    if not isinstance(revision, dict):
        return None
    task_id = task["id"]
    options = proposals({"proposals": [dict(revision, task=task_id)]},
                        [{"id": task_id, "executable_components": task["revision_components"]}])[task_id]
    if not options:
        return None
    option, = options
    if any(item["steps"] == option["steps"] for item in task["allowed_recipes"]):
        return None
    return dict(option, id="DS-revision")


def formal_review(trial, option, request, *, scope=None):
    available = len(trial.formal)
    if available < MIN_FORMAL:
        return dict(status="pending", basis="station_formal_minimum", count=0, checks=[],
                    reason=f"正式初筛需要至少 {MIN_FORMAL} 个可评分的留出实测点，当前只有 {available} 个。")

    def evaluate_point(index):
        score, issue = trial.score(option, trial.formal.iloc[[index]], request)
        return index, score, issue

    report = recipe_trials.audit(len(trial.formal), "withheld_station_times", evaluate_point, scope=scope)
    audit = report["acceptance_sampling"]
    report["unit_scores"] = report["sampled_scores"]
    inspected = trial.formal.iloc[audit["inspected_indices"]]
    score, _ = trial.score(option, inspected, request)
    report["sampled_scores"] = [dict(score, unit="formal")] if score else []
    if scope and scope["mode"] == "stratified":
        audit["single_reference_per_unit"] = True
    if audit["scored"] < MIN_FORMAL:
        report.update(status="pending", basis="station_formal_minimum", count=0,
                      reason=f"正式初筛需要至少 {MIN_FORMAL} 个实测点成功评分。")
    elif report["status"] == "failed":
        report["reason"] = (f"已评分 {audit['scored']} 个留出实测点，{audit['failed']} 个不适配，"
                            f"整批不适配率低于 5% 的后验概率为 {audit['posterior_probability']:.2%}。")
    return report


def pending(reason, basis, **details):
    return dict(status="pending", basis=basis, count=0, checks=[], reason=reason, **details)


def advisory(task, report):
    return dict(task=task["id"], id=RETAIN, name="站点融合待核验", description=report["reason"], steps=[],
                advisory_only=True, reason=report["reason"], station_two_stage=True, station_report=report)


def revise_formal_selection(row, report, trial, request, available, send, progress):
    if row["id"] == RETAIN or not recipe_trials.needs_revision(report):
        return row, report, []
    task_id = row["task"]
    options = []
    for time in available["time_matching"]:
        for correction in available["correction"]:
            choice = dict(task=task_id, time_matching=time, correction=correction)
            option, = proposals({"proposals": [choice]}, [{"id": task_id, "executable_components": available}])[task_id]
            options.append(dict(option, id=f"DS-Bayes-{len(options)+1}"))
    from itertools import count

    rounds = count(1)
    row, report, prior, status = recipe_trials.tune_recipe(
        row, report, options, lambda option, scope: formal_review(trial, option, request, scope=scope),
        lambda payload: send(f"bayesian_revision_{next(rounds)}", payload), progress, READ_ERRORS)
    row["bayesian_revision_status"] = status
    return row, report, prior


def review_revisions(rows, selection_tasks, trials, requests, target, send, progress):
    revised_tasks = []
    for row in rows:
        task_id = row["task"]
        task = selection_tasks[task_id]
        option = revised_option(row, task)
        if option is None:
            continue
        score, = trial_scores(trials[task_id], [option], requests[task_id])
        task["allowed_recipes"].append(option)
        task["development_trials"].append(score)
        updated = {key: value for key, value in task.items() if key != "revision_components"}
        updated["previous_recommendation"] = dict(recipe=row["id"], reason=row["reason"])
        revised_tasks.append(updated)
    if not revised_tasks:
        return rows
    if progress:
        progress("已在相同时间块复跑修改方案，DeepSeek 正在比较并确认最终推荐…")
    try:
        reply = send("revision_selection", dict(stage="recommendation", station_trial_selection=True,
                     station_revision_review=True, target_model=target, tasks=revised_tasks))
    except READ_ERRORS:
        if progress:
            progress("修改方案的复评回复未完成，沿用此前已核验的推荐。")
        return rows
    replacements = {row["task"]: row for row in reply["recommendations"]}
    return [replacements.get(row["task"], row) for row in rows]


def recommend(scan, plan, executions, *, mock, api_key, progress, on_request):
    from . import recommendations, service

    receipts, digests, outcomes = [], {}, {}
    tasks = {task["id"]: task for task in plan["tasks"]}

    def send(phase, summary):
        request = service.preview(summary)
        digests[phase] = request.digest
        if on_request is not None:
            on_request(request)
        return service.send(request, request.digest, mock, api_key=api_key,
                            on_progress=progress, on_receipt=receipts.append)

    proposal_tasks, trials = [], {}
    for task_id, task in tasks.items():
        try:
            trial, methods = reserve_trial(scan, executions[task_id])
        except READ_ERRORS as error:
            outcomes[task_id] = pending(str(error), "station_holdout_unavailable")
            continue
        if methods:
            trials[task_id] = trial
            proposal = proposal_task(task, executions[task_id], methods)
            proposal_tasks.append(training_evidence(proposal, executions[task_id], trial))
        else:
            reason = (f"共同覆盖内最多有 {max(trial.method_counts.values())} 个可配对实测点。"
                      f"需要分别留出至少 {MIN_TRAIN} 个拟合点、{MIN_DEVELOPMENT} 个小样本试跑点"
                      f"和 {MIN_FORMAL} 个正式初筛点。当前未能划出互不重叠的三批点。")
            outcomes[task_id] = pending(reason, "station_formal_minimum", matched_counts=trial.method_counts,
                                        training_count=len(trial.training), available_formal_count=len(trial.formal))
    if not proposal_tasks:
        return [advisory(tasks[key], value) for key, value in outcomes.items()], dict(
            status="STATION_PENDING", api_receipts=receipts, station_request_sha256=digests)
    outbound = dict(plan, stage="station_proposal", tasks=proposal_tasks, can_request_recommendation_evidence=True)
    outbound.pop("shared_algorithm_catalogue", None)
    selected = []
    requests, selection_tasks = {}, {}
    try:
        if progress:
            progress("DeepSeek 正在根据资料组合站点与网格的试跑方案…")
        reply = send("proposals", outbound)
        if isinstance(reply.get("evidence_requests"), list) and reply["evidence_requests"]:
            evidence = recommendations.requested_evidence(scan, plan, reply["evidence_requests"])
            outbound.pop("can_request_recommendation_evidence")
            reply = send("proposal_evidence", dict(outbound, recommendation_evidence=evidence))
        options = proposals(reply, proposal_tasks, mock=mock)
        revision_components = {task["id"]: task["executable_components"] for task in proposal_tasks}
        if progress:
            progress("正在用相同的时间块和可配对站点试跑 DeepSeek 提出的方案…")
        for task_id, trial in trials.items():
            recipes = options[task_id]
            if len(recipes) < 2:
                outcomes[task_id] = pending("需要 DeepSeek 提出至少两套不同且可执行的方案才能比较。",
                                            "station_proposals_incomplete")
                continue
            execution = executions[task_id]
            request = score_request(execution, plan["target_model"]["id"])
            scores = trial_scores(trial, recipes, request)
            requests[task_id] = request
            selection_tasks[task_id] = dict(id=task_id, operation="fusion", context={}, unit=execution["unit"],
                                            allowed_recipes=recipes, development_trials=scores,
                                            revision_components=revision_components[task_id])
        if selection_tasks:
            if progress:
                progress("正在把各时间块的试跑分数发给 DeepSeek，比较收益、退化与修改方案…")
            choice = send("recommendation", dict(stage="recommendation", station_trial_selection=True,
                          target_model=plan["target_model"], tasks=list(selection_tasks.values())))
            final_choices = review_revisions(choice["recommendations"], selection_tasks, trials,
                                             requests, plan["target_model"], send, progress)
            # Formal points are scored only after initial selection and any revision are finished.
            assert_current(scan)
            for row in final_choices:
                task_id = row["task"]
                trial = trials[task_id]
                if row["id"] == RETAIN:
                    available = revision_components[task_id]
                    time_matching = next(method for method in TIME_METHODS if method in available["time_matching"])
                    choice = dict(task=task_id, time_matching=time_matching, correction=UNCALIBRATED)
                    baseline, = proposals({"proposals": [choice]},
                                          [{"id": task_id, "executable_components": available}])[task_id]
                    row = {**baseline, **row, "steps": baseline["steps"]}
                report = formal_review(trial, row, requests[task_id])
                row, report, history = revise_formal_selection(row, report, trial, requests[task_id],
                                                               revision_components[task_id], send, progress)
                if history:
                    row["bayesian_revision_history"] = history
                if row["correction"] == UNCALIBRATED:
                    row.update(station_baseline=True, name="保留原网格并补齐站点缺测",
                               description="已有站点实测优先保留，缺测处使用已配对的原网格值。")
                    if tasks[task_id].get("engineering_requires_fusion"):
                        row.update(advisory_only=True, baseline_block_reason="工程明确要求完成融合，仍需有通过初筛的融合方案。")
                if report["status"] == "checked" and not row.get("advisory_only"):
                    execution = executions[task_id]
                    if row["steps"] not in execution["recipe_steps"]:
                        execution["recipe_steps"] = [*execution["recipe_steps"], row["steps"]]
                row.update(station_two_stage=True, station_report=report,
                           explanation=recommendations.explanation(tasks[task_id], row["reason"]),
                           development_trials=selection_tasks[task_id]["development_trials"],
                           proposal_options=selection_tasks[task_id]["allowed_recipes"],
                           recommendation_origin="simulated" if mock else "deepseek")
                selected.append(row)
    except READ_ERRORS as error:
        if progress:
            progress("站点融合的 DeepSeek 推荐未完成：" + str(error))
        for task_id in tasks:
            outcomes.setdefault(task_id, pending("本轮未完成 DeepSeek 站点融合推荐。", "station_recommendation_incomplete"))

    chosen = {row["task"] for row in selected}
    selected.extend(advisory(tasks[task_id], report) for task_id, report in outcomes.items() if task_id not in chosen)
    return selected, dict(status="SIMULATED" if mock else "COMPLETED" if chosen else "STATION_PENDING",
                          api_receipts=receipts, station_request_sha256=digests)
