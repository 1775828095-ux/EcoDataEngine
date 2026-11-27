import csv
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ecodata.desktop import catalog_decision, deepseek_transport, discovery, fallback_evidence
from ecodata.desktop import recommendations, release_gate, station_grid, station_recommendation
from ecodata.desktop.holdout_plan import export_field, exposures
from ecodata.desktop.sampling_plan import station_reserved
from ecodata.desktop.scan import Scanner


def station_input(folder, *, withheld, changed_after_trial=False, offset_times=False, missing_record=False):
    dates = pd.date_range("2024-01-01", periods=96, freq="h")
    variable = "air_temperature"
    station = next(f"s{i}" for i in range(100) if station_reserved(f"s{i}"))
    xr.Dataset({variable: (("time", "lat", "lon"), np.full((96, 4, 4), 280.0),
                          {"units": "K", "standard_name": variable})},
               coords=dict(time=dates, lat=np.arange(4.), lon=np.arange(4.)),
               attrs={"crs": "EPSG:4326"}).to_netcdf(folder / "grid.nc", engine="scipy")
    with (folder / "sites.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("station_id", "time_utc", "longitude", "latitude", variable))
        if missing_record:
            writer.writerow((station, dates[-withheld-1].strftime("%Y-%m-%dT%H:%M:%SZ"), 1., 2., ""))
        for i, date in enumerate(dates[-withheld:]):
            value = 278 if changed_after_trial and i >= withheld - 6 else 282
            if offset_times and i < withheld - 1:
                date += pd.Timedelta(minutes=10)
            writer.writerow((station, date.strftime("%Y-%m-%dT%H:%M:%SZ"), 1., 2., value))
        # An unscorable raw row at a distinct timestamp does not count toward six.
        if withheld == 8:
            writer.writerow((station, dates[-9].strftime("%Y-%m-%dT%H:%M:%SZ"), 1., 2., ""))
    scan, _ = Scanner().scan(folder)
    scan["on_demand"] = True
    summary, _ = discovery.discover(scan, "data_only")
    fields, detail = catalog_decision.local_fields(scan, summary)
    grid, observation = [next(item for item in fields.values() if scan["rows"][item[0]]["format"] == fmt
                              and item[2]["name"] == variable) for fmt in ("NetCDF", "CSV")]
    observation[2]["unit"] = "K"
    pair, local = discovery.describe_pair(scan, "C1", grid, observation, "data_only", detail)
    approved = dict(summary, candidates=[pair])
    decisions = [dict(candidate="C1", action="fusion", physical_quantity=variable)]
    checked, gated = discovery.check(scan, {"C1": local}, {"decisions": decisions}, approved, sampled_workflow=True)
    for index, _, field in (grid, observation):
        assert export_field(scan, index, field, phase="field_probe")["basis"] == "station_review_reserved"
        assert not exposures(scan, index, field)
    plan = recommendations.make_request(scan, approved, checked, gated, sampled_workflow=True)
    assert gated["C1", "fusion"]["route"] == "station_grid"
    assert gated["C1", "fusion"]["locked_holdout"]["minimum_grid_index"] == 0
    return scan, approved, decisions, plan, gated


@pytest.mark.parametrize("choice,mandatory", [("retain_grid", False), ("DS1", False), ("retain_grid", True)])
def test_verified_station_baseline_is_selectable_and_fills_only_missing(tmp_path, monkeypatch, choice, mandatory):
    from io import BytesIO
    from ecodata.desktop import issues, selected_processing

    scan, approved, decisions, plan, gated = station_input(tmp_path, withheld=25, missing_record=True)
    plan["tasks"][0]["engineering_requires_fusion"] = mandatory

    def fetch(request, opener, stage, digest, progress):
        payload = json.loads(json.loads(request.data)["messages"][1]["content"])
        task, = payload["tasks"]
        if stage == "station_proposal":
            method = next(iter(task["executable_components"]["time_matching"]))
            reply = dict(proposals=[dict(task=task["id"], time_matching=method, correction=correction)
                                    for correction in (station_recommendation.UNCALIBRATED, station_grid.BIAS)])
        else:
            assert stage == "recommendation"
            reply = dict(recommendations=[dict(task=task["id"], recipe=choice, reason="保留原网格")])
        return dict(id="baseline-fixture", choices=[dict(message=dict(content=json.dumps(reply)))])

    monkeypatch.setattr(deepseek_transport, "fetch", fetch)
    rows, _ = recommendations.recommend_plan(plan, decisions, scan=scan, gated=gated, api_key="test-only")
    recommendations.review_holdouts(scan, approved, plan, rows, gated)
    row, = rows
    assert row["station_baseline"] and not recommendations.is_private_fusion_candidate(row)
    audit = row["station_report"]["acceptance_sampling"]
    assert audit["inspected"] == 11 and audit["failed"] == 0 and audit["posterior_probability"] == 1
    if mandatory:
        assert not release_gate.selectable(row)
        assert "工程明确要求完成融合" in issues.recommendation_checks(rows)[0]["problem"]
        return

    assert release_gate.selectable(row) and row["final_verified_recommendation"]
    task = dict(gated["C1", "fusion"], recipes=[row["steps"]], runtime_fields=row["execution_fields"])
    result = selected_processing.run(scan, [(row["task"], task)], tmp_path / "output", model_id="data_only")
    assert result["processing_completed"] and not result["grid_searches"], result["failed_tasks"]
    report, = result["original_report"][0]["report"]
    assert report["delivery_status"] == "BASELINE_RETAINED" and report["gap_filled"] == 1
    _, data = next((name, data) for name, data in result["files"] if name.endswith(".parquet"))
    table = pd.read_parquet(BytesIO(data))
    assert table.loc[table.observed.isna(), "selected"].tolist() == [280.]
    assert table.loc[table.observed.notna(), "selected"].tolist() == [282.] * 25


def test_camx_wind_magnitude_is_scored_on_reserved_station_points(tmp_path):
    from netCDF4 import Dataset

    dates = pd.date_range("2024-01-01", periods=18, freq="h")
    tflag = np.repeat(np.column_stack((np.full(len(dates), 2024001), dates.hour * 10000))[:, None, :], 2, axis=1)
    path = tmp_path / "camx.nc"
    xr.Dataset(
        {"u10": (("time", "lat", "lon"), np.full((18, 3, 3), 3.),
                 {"units": "m/s", "long_name": "longitudinal wind at 10 m"}),
         "v10": (("time", "lat", "lon"), np.full((18, 3, 3), -4.),
                 {"units": "m/s", "long_name": "latitudinal wind at 10 m"}),
         "TFLAG": (("time", "VAR", "DATE-TIME"), tflag)},
        coords=dict(time=dates, lat=np.arange(3.), lon=np.arange(3.)),
        attrs={"CAMx_NAME": "2D METEOROLOGY", "ITZON": 0, "VAR-LIST": "u10 v10"},
    ).to_netcdf(path)
    with (tmp_path / "sites.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("station_id", "time_utc", "longitude", "latitude", "wind_speed"))
        writer.writerows(("s0", date.strftime("%Y-%m-%dT%H:%M:%SZ"), 1., 1., 7.) for date in dates)

    scan, _ = Scanner().scan(tmp_path)
    scan["on_demand"] = True
    summary, _ = discovery.discover(scan, "camx")
    fields, details = catalog_decision.local_fields(scan, summary)
    by_name = {field["name"]: (index, fid, field) for index, fid, field in fields.values()}
    grid, observation = by_name["v10"], by_name["wind_speed"]
    observation[2].update(unit="m/s", standard_name="wind_speed")
    pair, local = discovery.describe_pair(scan, "C1", grid, observation, "camx", details)
    approved = dict(summary, candidates=[pair])
    decisions = [{"candidate": "C1", "action": "fusion"}]
    checked, gated = discovery.check(scan, {"C1": local}, {"decisions": decisions}, approved, sampled_workflow=True)
    recommendations.make_request(scan, approved, checked, gated, sampled_workflow=True)
    task = gated["C1", "fusion"]
    assert task["wind_components"] == ("u10", "v10")
    for index, _, field in (by_name["u10"], grid, observation):
        assert export_field(scan, index, field, phase="recommendation")["basis"] == "station_review_reserved"
    trial, methods = station_recommendation.reserve_trial(scan, task)
    assert methods and len(trial.development) >= 3 and len(trial.formal) >= 6
    assert trial.formal.background.eq(5.).all()
    report = station_recommendation.formal_review(trial, {"correction": station_grid.BIAS}, {})
    score, = report["sampled_scores"]
    assert (score["candidate"]["RMSE"], score["baseline"]["RMSE"]) == (0., 2.)

    with Dataset(path, "r+") as dataset:
        dataset["TFLAG"][0, 1, 1] = 10000
        with pytest.raises(ValueError, match="风分量的时间标记不一致"):
            station_grid.wind_variables(dataset, task["wind_components"])


@pytest.mark.parametrize("withheld, expected", [(12, "released"), (12, "failed"), (8, "pending")],
                         ids=["six-formal-pass", "six-formal-reject", "insufficient"])
def test_deepseek_proposes_then_selects_before_independent_station_screen(tmp_path, monkeypatch, withheld, expected):
    source = tmp_path / "inputs"
    source.mkdir()
    scan, approved, decisions, plan, gated = station_input(
        source, withheld=withheld, changed_after_trial=expected == "failed", offset_times=expected == "released")
    calls = []
    report_calls = []
    formal_review = station_recommendation.formal_review

    def check_formal(trial, option, request, *, scope=None):
        assert calls[-1] == "recommendation"
        assert not set(trial.training.grid_time) & set(trial.development.grid_time)
        assert not set(trial.training.grid_time) & set(trial.formal.grid_time)
        assert not set(trial.development.grid_time) & set(trial.formal.grid_time)
        report_calls.append(len(trial.formal))
        return formal_review(trial, option, request, scope=scope)

    def fetch(request, opener, stage, digest, progress):
        calls.append(stage)
        messages = json.loads(request.data)["messages"]
        payload = json.loads(messages[1]["content"])
        if stage == "bayesian_revision":
            assert expected == "failed"
            assert payload["bias_evidence"]["failed"] == 6
            assert payload["bias_evidence"]["candidate_rmse_median"] == 4
            assert payload["bias_evidence"]["baseline_rmse_median"] == 2
            return dict(id=f"fixture-{len(calls)}", choices=[dict(message=dict(content='{"recipe": null}'))])
        task, = payload["tasks"]
        assert "formal" not in json.dumps(payload)
        if stage == "station_proposal":
            assert "allowed_recipes" not in task
            assert set(task["source_sample_counts"].values()) == {3}
            components = task["executable_components"]
            assert station_recommendation.RATIO in components["correction"]
            if expected == "released":
                assert list(components["time_matching"]) == [station_grid.NEAREST]
            if "can_request_recommendation_evidence" in payload:
                output = dict(evidence_requests=[dict(task=task["id"], kind="field_metadata",
                                                     target=task["context"]["fields"][0])])
            else:
                assert payload["recommendation_evidence"][0]["status"] == "available"
                corrections = [station_recommendation.UNCALIBRATED, station_grid.BIAS, "unregistered"]
                if expected == "failed":
                    corrections.append(station_recommendation.RATIO)
                output = dict(proposals=[dict(task=task["id"], time_matching=station_grid.NEAREST,
                                             correction=method, reason="试跑不同的订正思路")
                                         for method in corrections])
        else:
            assert stage == "recommendation" and not report_calls
            assert "候选的本地试跑分数" not in messages[0]["content"]
            assert "在哪些退化" in messages[0]["content"]
            baseline, additive, *extra = task["development_trials"]
            assert baseline["count"] == additive["count"] == 3
            assert baseline["candidate"]["RMSE"] == additive["baseline"]["RMSE"] == 2
            assert additive["candidate"]["RMSE"] == 0
            assert [block["scored_points"] for block in additive["time_blocks"]] == [2, 1]
            assert all(block["rmse_change"] == -2 and block["baseline_bias_direction"] == "too_low"
                       and block["candidate_bias_direction"] == "near_zero" for block in additive["time_blocks"])
            selection = dict(task=task["id"], recipe=additive["recipe"], reason="按时间块选择订正")
            if payload.get("station_revision_review"):
                assert expected == "released" and task["previous_recommendation"]["recipe"] == additive["recipe"]
                assert "revision_components" not in task
                revised, = extra
                assert revised["recipe"] == "DS-revision" and revised["time_blocks"] == additive["time_blocks"]
                selection["recipe"] = revised["recipe"]
            elif expected == "released":
                assert not extra
                selection["revision"] = dict(time_matching=station_grid.NEAREST, correction=station_recommendation.RATIO,
                                             reason="用比例订正复测两个时间块")
            output = dict(recommendations=[selection])
        return dict(id=f"fixture-{len(calls)}", choices=[dict(message=dict(content=json.dumps(output)))])

    monkeypatch.setattr(station_recommendation, "formal_review", check_formal)
    monkeypatch.setattr(deepseek_transport, "fetch", fetch)
    rows, receipt = recommendations.recommend_plan(plan, decisions, scan=scan, gated=gated, api_key="test-only")
    recommendations.review_holdouts(scan, approved, plan, rows, gated)
    assert recommendations.finalize_recipes(plan, rows, mock=False, api_key="test-only", progress=None) is None
    row, = rows
    assert row["station_two_stage"] and row["execution_release"]["status"] == expected, row
    assert not fallback_evidence.eligible(row)
    assert release_gate.allowed(row) == (expected == "released")
    if expected == "pending":
        assert report_calls == []
        assert calls == []
        assert row["station_report"]["basis"] == "station_formal_minimum"
        assert row["station_report"]["count"] == 0 and row["station_report"]["available_formal_count"] == 5
    else:
        assert report_calls == ([6] if expected == "released" else [6, 6])
        assert calls == ["station_proposal", "station_proposal", "recommendation"] + (
            ["recommendation"] if expected == "released" else ["bayesian_revision", "bayesian_revision"])
        assert row["station_report"]["count"] == 6
        assert row["station_report"]["sampled_scores"][0]["candidate"]["RMSE"] == (0 if expected == "released" else 4)
    if expected == "released":
        from ecodata.desktop import selected_processing

        assert row["steps"][-1] == station_recommendation.RATIO
        task = dict(gated["C1", "fusion"], recipes=[row["steps"]], recommendation_origin=row["recommendation_origin"],
                    runtime_fields=row["execution_fields"])
        result = selected_processing.run(scan, [(row["task"], task)], tmp_path / "output", model_id="data_only")
        assert result["processing_completed"], result["failed_tasks"]
        run = result["original_report"][0]["report"][0]["mutation_search"]
        assert run["recommendation_validation"][0]["status"] == "PASS", run["recommendation_validation"]
        assert run["delivery_status"] == "BASELINE_RETAINED"
    assert len(receipt["api_receipts"]) == len(calls)


def test_formal_review_does_not_count_missing_observations():
    frame = pd.DataFrame(dict(station="test", grid_time=pd.date_range("2024-01-01", periods=6, freq="h"),
                              background=280., observed=[np.nan, 282, 282, 282, 282, 282]))
    trial = station_recommendation.Trial(frame, frame, frame, {})
    report = station_recommendation.formal_review(trial, {"correction": station_recommendation.UNCALIBRATED}, {})
    assert report["basis"] == "station_formal_minimum" and report["count"] == 0


def test_development_returns_failure_and_cannot_select_a_worse_recipe():
    training = pd.DataFrame(dict(station="test", background=[1., 2., 3.], observed=[3., 4., 5.]))
    development = pd.DataFrame(dict(station="test", grid_time=pd.date_range("2024-01-01", periods=3, freq="h"),
                                    background=[10., 11., 12.], observed=[12., 13., 14.]))
    trial = station_recommendation.Trial(training, development, pd.DataFrame(), {})
    options = [dict(id="DS1", correction=station_recommendation.RATIO)]
    scores = station_recommendation.trial_scores(trial, options, {})
    assert scores[0]["failure_reason"] == "recipe_score_below_single_source"
    assert scores[0]["count"] == 3
    summary = dict(station_trial_selection=True, tasks=[dict(id="C1:fusion", allowed_recipes=options,
                                                             development_trials=scores)])
    with pytest.raises(ValueError, match="未收到有效推荐"):
        recommendations.validate(dict(recommendations=[dict(task="C1:fusion", recipe="DS1")]), summary)


def test_trial_samples_across_times_and_stations_before_formal_holdout():
    clocks = pd.date_range("2024-01-01", periods=90, freq="h")
    times = clocks.repeat(3)
    frame = pd.DataFrame(dict(station=np.tile(["a", "b", "c"], len(clocks)), grid_time=times,
                              background=20., observed=22., matched_observations=1))
    sparse = pd.DataFrame(dict(station="sparse", grid_time=clocks[[0, 1, 2, 34, 75]],
                               background=20., observed=22., matched_observations=1))
    frame = pd.concat([frame, sparse], ignore_index=True)
    trial = station_recommendation.partition({station_grid.NEAREST: frame})
    sampled = trial.development.grid_time.drop_duplicates().sort_values()
    assert len(sampled) == 6 and sampled.iloc[0] == clocks[30] and sampled.iloc[-1] == clocks[49]
    assert trial.training.grid_time.max() < sampled.iloc[0] < sampled.iloc[-1] < trial.formal.grid_time.min()
    blocks = station_recommendation.development_blocks(trial.development)
    assert clocks[34] in set(sampled)
    assert [(item["points"], item["stations"]) for item, _ in blocks] == [(7, 4), (6, 3), (6, 3)]
