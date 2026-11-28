from ecodata.desktop import service, wrf_station_gap as gap


def test_unverified_local_limit_does_not_block_improving_candidate(monkeypatch):
    def uncited(*args, **kwargs):
        return {"max_block_mae_increase_k": .1, "source_url": "https://example.gov/rule"}

    monkeypatch.setattr(service, "send", uncited)
    policy = gap.resolve_policy(None, "fixture", False, None, [])
    assert policy["max_block_mae_increase_k"] is None
    assert policy["origin"] == "no_applicable_local_limit"

    def measured(mae, blocks):
        def metrics(value):
            return {"MAE": value, "RMSE": value, "Bias": 0., "count": 2}
        return {"metrics": metrics(mae), "blocks": {k: metrics(v) for k, v in blocks.items()}}

    candidate = measured(2., {"a": 1., "b": 3.})
    baseline = measured(2.5, {"a": 3., "b": 2.})
    result = gap.compare(candidate, baseline, policy)
    assert gap.qualified(result)
    assert not result["local_limit_applied"] and result["worst_block_mae_increase_k"] == 1.


def test_recommendation_catalogue_binds_complete_terrain_residual_recipe():
    available = (4, 8, 16)
    payload = [
        {"method": "wrf_height_residual", "parameters": {
            "neighbors": [4, 8], "power": 2, "lapse_k_per_km": 3.25, "strength": .75}},
        {"method": "wrf_height_residual", "parameters": {
            "neighbors": 40, "power": 2, "lapse_k_per_km": 3.25, "strength": .75}},
    ]
    offered = list(gap.proposals({"proposals": payload}, available))
    assert len(offered) == 2
    assert {dict(spec[1])["neighbors"] for spec, _ in offered} == {4, 8}
    assert set(entry["method"] for entry in gap.offered_catalogue(available)) == set(gap.CATALOG)


def test_station_gap_output_review_uses_task_scope(monkeypatch, tmp_path):
    from ecodata.desktop import issues, output_review, research_dispatch
    from ecodata.models.requirements import RequirementStore

    monkeypatch.setattr(research_dispatch, "assert_current", lambda scan: None)
    monkeypatch.setattr(research_dispatch, "Resources", lambda scan, file: None)

    def fill(scan, request, stage, *args, **kwargs):
        if request.get("missing"):
            raise ValueError("站点表缺字段：elevation_m")
        (stage / "holdout_predictions.csv").write_text("station,error\nA,1", encoding="utf-8")
        return {"status": "PUBLISHED_REPLAY_ONLY"}

    def unexpected_review(*args, **kwargs):
        raise AssertionError("站点补缺不应发送完整工程的线上复核请求")

    monkeypatch.setattr(gap, "run", fill)
    monkeypatch.setattr(service, "send", unexpected_review)
    task = dict(tool="wrf_station_gap", label="WRF 站点气温补缺", file=0, request={})
    result = research_dispatch.run({}, [task], tmp_path / "replay")
    profile = RequirementStore().get("wrf")
    stale_review = dict(verdict="needs_verification", missing_data=[],
                        missing_information=["需要核实静态地理数据 static_geography"])
    check = output_review.describe_output(result, profile)
    assert check["model"] == "wrf"
    assert check["task_scope"] == "wrf_station_gap"
    assert check["target_format"] == "CSV / JSON"
    assert check["requirements"] == []
    assert "原生" not in output_review.availability_text(check, stale_review)
    assert issues.output(result, stale_review) == []
    assert service.review_model_output(result, profile, mock=True) == (None, None)

    task["request"] = {"missing": True}
    blocked = research_dispatch.run({}, [task], tmp_path / "blocked")
    check = output_review.describe_output(blocked, profile)
    assert check["target_format"] == "JSON"
    assert "站点表缺字段：elevation_m" in output_review.availability_text(check)
    assert [issue["problem"] for issue in issues.output(blocked, stale_review)] == [
        "站点表缺字段：elevation_m"
    ]

    full = output_review.describe_output({"manifest": {"native_ready": False}}, profile)
    assert full["requirements"]
    assert output_review.review_request(full, [])["task_scope"] == "model_project"
    assert "原生文件、输入绑定和完整工程配置仍需核验" in output_review.availability_text(full)



def test_wrf_result_names_the_delivered_fusion_and_its_validation_scope():
    from ecodata.desktop import output_review

    fusion = dict(method="wrf_height_residual",
                  parameters=dict(neighbors=16, power=1, lapse_k_per_km=3.25, strength=1))
    baseline = dict(method="station_idw", parameters=dict(neighbors=8, power=2, lapse_k_per_km=6.5))
    data = dict(status="BATCH_RECOMMENDED", recommended=True, candidate=fusion,
                delivered_recipe=fusion, best_single=baseline,
                final_controls=[dict(**fusion, metrics=dict(MAE=2.012, RMSE=2.987, count=426)),
                                dict(**baseline, metrics=dict(MAE=2.265, RMSE=3.443, count=426))],
                final_review=dict(mae_gain_k=.253, worst_block_mae_increase_k=3.29),
                recommendation_rounds=[dict(proposed_candidates=[fusion])],
                recommendation_status="ONLINE_ATTEMPTED", policy=dict(max_block_mae_increase_k=None))
    row = dict(tool="wrf_station_gap", label="WRF 多站同时段气温补缺", outcome=data)
    card = output_review.station_gap_card(row, "CSV / JSON")
    summary, = output_review.selection_summaries([card], [], [])
    assert summary["selection_label"] == "本批推荐"
    assert "海拔订正＋邻站偏差订正" in summary["title"]
    assert "邻站数 16" in summary["parameters"] and "3.25 K/km" in summary["parameters"]
    assert summary["metrics"]["count"] == 426
    assert "DeepSeek 提名" in summary["reason"]
    assert "2.265 K" in summary["reason"] and "+3.290 K" in summary["reason"]

    data.update(status="HOLDOUT_REJECTED", recommended=False, delivered_recipe=baseline)
    rejected = output_review.station_gap_card(row, "CSV / JSON")
    assert not rejected["selected"] and not rejected["metrics"]
    assert "留出评分未达标" in rejected["reason"] and "实际输出采用站点反距离插值" in rejected["reason"]



def test_recommendation_requires_a_scored_nomination_and_revises_on_development_evidence(monkeypatch, tmp_path):
    from contextlib import nullcontext
    import json

    import numpy as np
    import pandas as pd

    from ecodata.desktop import file_tools, scan

    rows = pd.DataFrame(dict(station_id=list("ABCDEFGHI"), latitude=1., longitude=2., x=1., y=2.,
                             temperature_K=280., grid_temperature_k=282., elevation_m=100.,
                             grid_elevation_m=90., qc_usable=[True] * 8 + [False],
                             time_offset_seconds=0))
    roles = np.array(["training"] * 4 + ["development"] * 2 + ["final"] * 2 + ["gap"])
    blocks = np.array(["0", "0", "1", "1", "2", "3", "4", "5", "6"])
    path = tmp_path / "input"
    path.write_bytes(b"a different, synthetic input")
    monkeypatch.setattr(file_tools, "selected", lambda *a, **k: nullcontext((path,)))
    monkeypatch.setattr(scan, "assert_current", lambda _: None)
    monkeypatch.setattr(gap, "align", lambda *a: (rows, {"total_reports": len(rows), "matching_window_seconds": 900}))
    monkeypatch.setattr(gap, "partition", lambda _: (roles, blocks))
    monkeypatch.setattr(gap, "neighbors_for", lambda *a: (
        np.zeros((9, 4), dtype=int), np.full((9, 4), 30_000.), np.full(9, 4)))
    bad = dict(method="wrf_residual", parameters=dict(neighbors=4, power=1, strength=.5))
    good = dict(method="wrf_residual", parameters=dict(neighbors=4, power=1, strength=.75))

    def run(mode):
        requests, opened_final = [], []

        def send(approved, *args, **kwargs):
            assert not opened_final
            payload = json.loads(approved.summary)
            assert "final_scores" not in payload and "training_spatial_block_summary" in payload
            requests.append(payload)
            if mode == "no nomination":
                return {"proposals": []}
            if len(requests) == 1:
                return {"proposals": [bad]}
            assert any(item["nominated"] and not item["overall_ok"] and item["range_comparison"]["blocks"]
                       for item in payload["development_scores"])
            return {"proposals": [good]}

        def predict(spec, rows, ids, distance, targets):
            if roles[targets[0]] == "final":
                opened_final.append(spec)
            method, params = spec
            if method in gap.SINGLE:
                error = 2.
            elif dict(params).get("strength") == .5:
                error = 3.
            elif dict(params).get("strength") == .75:
                error = 3. if mode == "final failure" and roles[targets[0]] == "final" else 1.
            else:
                error = .5  # A stronger fixed control still cannot bypass nomination.
            return np.full(len(targets), 280. + error)

        monkeypatch.setattr(service, "send", send)
        monkeypatch.setattr(gap, "predict", predict)
        stage = tmp_path / mode
        stage.mkdir()
        report = gap.run({"path": path}, {"grid": "a", "stations": "b"}, stage, mock=True)
        return report, requests, stage

    report, requests, stage = run("revision")
    assert len(requests) == 2 and report["status"] == "BATCH_RECOMMENDED"
    assert report["delivered_recipe"] == good and not report["independent_acceptance"]
    assert pd.read_csv(stage / "station_gap_predictions.csv").temperature_fill_k.tolist() == [281.]

    rejected, requests, _ = run("final failure")
    assert len(requests) == 2 and rejected["status"] == "HOLDOUT_REJECTED"
    assert rejected["delivered_recipe"] == rejected["best_single"]

    empty, _, _ = run("no nomination")
    assert empty["status"] == "NO_FUSION_NOMINATION" and not empty["recommended"]
    assert empty["candidate"] is None and empty["delivered_recipe"] == empty["best_single"]
