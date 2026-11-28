import csv

import numpy as np
import pytest
import xarray as xr

from ecodata.desktop import fallback_evidence, recipe_trials, sampling_risk
from ecodata.desktop.scan import inventory


def test_cmorph_daily_shards_share_a_holdout_with_power(tmp_path):
    import pandas as pd
    from ecodata.desktop import independent_alignment, precipitation_partitions, sampling_plan
    from ecodata.desktop.scan import Scanner
    from test_precipitation_product_admission import product

    dates = pd.date_range("2020-01-01", periods=7)
    axis = np.arange(6.)
    power = product("power").rename(arbitrary="PRECTOTCORR").reindex(time=dates, lat=axis, lon=axis)
    power.PRECTOTCORR.values[:] = 12.
    power.attrs.update(time_basis="UTC", crs="EPSG:4326")
    power.time.attrs["delta"] = "0000-00-01 00:00:00"
    power.to_netcdf(tmp_path / "power.nc")
    for date in dates[2:5]:
        shard = product("cmorph").assign_coords(time=[date]).reindex(lat=axis, lon=axis)
        shard.arbitrary.values[:] = 12.
        shard.time.attrs["bounds"] = "edges"
        shard["edges"] = (("time", "nv"), [[date.to_datetime64(),
                          (date + pd.Timedelta(seconds=86399)).to_datetime64()]])
        shard.time.encoding["units"] = shard.edges.encoding["units"] = "seconds since 1970-01-01"
        shard.attrs.update(time_coverage_start=date.strftime("%Y-%m-%dT00:00:00Z"),
                           time_coverage_end=date.strftime("%Y-%m-%dT23:59:59Z"), crs="EPSG:4326")
        shard.to_netcdf(tmp_path / date.strftime("CMORPH_daily_%Y%m%d.nc"))
    scan, _ = Scanner().scan(tmp_path)
    precipitation_partitions.prepare(scan)
    assert len(scan["rows"]) == 2
    fields = []
    for index, row in enumerate(scan["rows"]):
        assert row["precipitation_holdout_days"] == ["2020-01-05"]
        field = next(item for item in row["fields"] if item["name"] in {"arbitrary", "PRECTOTCORR"})
        exported = sampling_plan.sample(row, field)
        assert exported["values"]
        assert max(item["stop_index"] for item in exported["windows"]) == (2 if field["name"] == "arbitrary" else 4)
        fields.append((index, field["name"]))
    task = dict(action="fusion", route="independent_grid_search", fields=fields)
    from ecodata.desktop.holdout_plan import export_field, freeze

    scan["on_demand"] = True
    for idx, name in fields:
        field = next(item for item in scan["rows"][idx]["fields"] if item["name"] == name)
        assert export_field(scan, idx, field, phase="field_probe")["perturbed_samples"]
    freeze(scan, {("C1", "fusion"): task}, [dict(candidate="C1", operation="fusion")])
    assert task["locked_holdout"] == {"days": ["2020-01-05"]}
    result = independent_alignment.review(scan, "C1:fusion", task,
        ["spatial.regrid.conservative", recipe_trials.FILL, recipe_trials.BLEND])
    assert result["basis"] == "joint_recipe_acceptance_passed", result
    assert result["sampled_scores"][0]["day"] == "2020-01-05"


def test_default_hypergeometric_boundary_and_no_early_acceptance():
    from scipy.stats import hypergeom

    policy = sampling_risk.recipe_settings()
    result = sampling_risk.evaluate_recipe(10_000_000, "time_window", lambda _: None, **policy)
    assert (result["status"], result["sample_size"], result["allowed_failures"], result["inspected"]) == ("checked", 59, 0, 59)
    assert result["bad_failures"] == 500_000 and result["false_accept_risk"] <= .05
    assert hypergeom.pmf(0, result["population"], result["bad_failures"], 58) > .05
    failed = sampling_risk.evaluate_recipe(1000, "time", lambda _: {"rule": "failed_score"}, **policy)
    assert failed["inspected"] == 1 and failed["status"] == "failed"


def test_bayesian_posterior_matches_the_finite_population_sum():
    from math import comb

    population, n, d = 103, 60, 2
    weights = [comb(k, d) * comb(population - k, n - d) for k in range(population + 1)]
    expected = sum(weights[:6]) / sum(weights)
    observed = sampling_risk.posterior_below_five_percent(population, n, d)
    assert observed == pytest.approx(expected)
    assert sampling_risk.posterior_below_five_percent(100, 100, 5) == 0
    assert sampling_risk.posterior_below_five_percent(100, 100, 4) == 1


@pytest.mark.parametrize('failures, expected', [(0, 58), (1, 92), (2, 123), (3, 17)])
def test_bayesian_acceptance_revises_at_the_first_unreachable_outcome(failures, expected):
    from itertools import count

    inspected = count(1)
    failure_positions = {3, 8, 17} if failures == 3 else set(range(1, failures + 1))
    report = sampling_risk.evaluate_bayesian_recipe(10_000_000, 'block',
        lambda index: {'rule': 'recipe_score_below_single_source'} if next(inspected) in failure_positions else None, seed=7)
    assert report['inspected'] == expected and report['failed'] == failures
    assert report['status'] == ('failed' if failures == 3 else 'checked')
    assert report['revision_required'] == (failures == 3)


def test_bayesian_acceptance_only_revises_when_exact_population_cannot_pass():
    from itertools import count

    inspected = count(1)
    report = sampling_risk.evaluate_bayesian_recipe(126, 'block',
        lambda index: {'rule': 'recipe_score_below_single_source'} if next(inspected) <= 3 else None, seed=7)
    assert report['failed'] == 3 and report['status'] == 'checked'
    assert not report['revision_required']


def test_bayesian_retry_continues_after_failed_formal_review(monkeypatch):
    from ecodata.desktop import recommendations, service

    population = 100_000
    requests, development, final = [], [], []

    initial = sampling_risk.evaluate_bayesian_recipe(population, 'block',
        lambda index: {'rule': 'recipe_score_below_single_source'}, seed=7)
    row = dict(task='C1:fusion', id='R1', steps=['initial'], execution_release={'status': 'failed'},
               local_holdout_review=[dict(acceptance_sampling=initial, sampled_scores=[])])
    plan = dict(target_model={'id': 'test'}, tasks=[
        dict(id='C1:fusion', candidate='C1', operation='fusion', bayesian_recipe_gate=True, allowed_recipes=[
            dict(id='R1', steps=['initial']), dict(id='R2', steps=['revision']), dict(id='R3', steps=['better'])])])

    def send(preview, *args, **kwargs):
        import json
        request = json.loads(preview.summary)
        assert request['current_development']['development_only']
        assert request['current_development']['posterior_probability'] is None
        assert not request['bias_evidence']['development_only']
        assert request['bias_evidence']['failed'] > 0
        assert 'inspected_indices' not in preview.summary
        assert 'development_new' not in preview.summary
        requests.append(request)
        if len(requests) == 1:
            return {'recipe': 'R1'}
        if len(requests) == 2:
            assert 'reply_issue' in request
            return {'recipe': 'R2'}
        assert [choice['id'] for choice in request['allowed_recipes']] == ['R3']
        return {'recipe': 'R3'}

    def run_trial(execution, steps):
        scope = execution['bayesian_review_scope']
        bad = steps == ['initial'] or (steps == ['revision'] and scope['mode'] != 'development')
        audit = sampling_risk.evaluate_revision_sample(population, 'block',
            lambda index: {'rule': 'recipe_score_below_single_source'} if bad else None, scope)
        (development if scope['mode'] == 'development' else final).append((scope, steps, audit))
        return dict(acceptance_sampling=audit, sampled_scores=[])

    def review(scan, approved, plan, rows, gated):
        report = run_trial(gated['C1', 'fusion'], rows[0]['steps'])
        status = 'released' if report['acceptance_sampling']['status'] == 'checked' else 'failed'
        rows[0].update(local_holdout_review=[report], execution_release={'status': status})

    monkeypatch.setattr(service, 'send', send)
    monkeypatch.setattr(recommendations, 'review_holdouts', review)
    monkeypatch.setattr(recipe_trials, 'review', lambda scan, task_id, task, steps, **kwargs: run_trial(task, steps))
    rows = [row]
    recommendations.retry_bayesian_reviews({}, {}, plan, rows, {('C1', 'fusion'): {}},
                                           mock=False, api_key='fixture', progress=None)
    assert rows[0]['id'] == 'R3' and rows[0]['execution_release']['status'] == 'released'
    assert rows[0]['final_verified_recommendation']
    assert recommendations.finalize_recipes(plan, rows, mock=False, api_key='fixture', progress=None) is None
    assert [scope['mask_phase'] for scope, _, _ in development] == [1, 1, 2, 2]
    assert requests[2]['previous_development_trials'][0]['recipe'] == 'R2'
    (failed_scope, _, failed), (scope, steps, result) = final
    assert failed['status'] == 'failed'
    assert scope['mask_phase'] == 3 and steps == ['better']
    participants = set(scope['previous']) | set(scope['development_new'])
    assert set(initial['inspected_indices']) <= participants
    assert set(failed['inspected_indices']) <= set(scope['previous'])
    assert set(failed_scope['development_new']) <= set(scope['previous'])
    assert all(old in participants and new not in participants
               for old, new in zip(result['inspected_indices'][::2], result['inspected_indices'][1::2]))
    assert result['strata'][0]['inspected'] == result['strata'][1]['inspected']
    assert result['excluded_as_failures'] == 0


def test_single_group_rejection_recommends_again_and_scores_entire_batch():
    first = dict(id='R1', task='C1:fusion', steps=['bad'])
    second = dict(id='R2', steps=['good'])

    def evaluate(option, scope):
        def trial(index):
            truth = np.array([10.])
            candidate = truth if option['id'] == 'R2' else truth + 5
            metrics, failure = recipe_trials.score(candidate, truth, truth, 'fixture', retained=truth)
            return index, metrics, failure
        return recipe_trials.audit(1, 'day', trial, scope=scope)

    initial = evaluate(first, None)
    assert not initial['acceptance_sampling']['revision_required']
    requests = []

    def send(payload):
        requests.append(payload)
        return {'recipe': 'R2'}

    selected, final, history, status = recipe_trials.tune_recipe(
        first, initial, [first, second], evaluate, send, None, (ValueError,))
    assert status == 'passed' and selected['id'] == 'R2' and len(history) == 1
    assert requests[0]['bias_evidence']['candidate_rmse_median'] == 5
    assert final['acceptance_sampling']['census'] and final['acceptance_sampling']['posterior_probability'] == 1


def test_stratified_posterior_matches_explicit_finite_hypergeometric_sum():
    from math import comb

    sizes, inspected, failed = (12, 89), (8, 8), (1, 0)
    def weights(size, n, d):
        return [comb(k, d) * comb(size-k, n-d) for k in range(size+1)]

    old, new = [weights(*args) for args in zip(sizes, inspected, failed)]
    expected = sum(a*b for i, a in enumerate(old) for j, b in enumerate(new) if 20*(i+j) < sum(sizes))
    assert sampling_risk.stratified_posterior(sizes, inspected, failed) == pytest.approx(
        expected / (sum(old)*sum(new)))
    assert sampling_risk.stratified_posterior((20, 80), (20, 80), (1, 4)) == 0
    assert sampling_risk.stratified_posterior((20, 80), (20, 80), (1, 3)) == 1
    assert sampling_risk.stratified_posterior((63, 99_937), (63, 63), (1, 0)) >= .95


def test_gapfill_retries_hide_different_reference_cells_in_the_same_group(monkeypatch):
    from ecodata.desktop.grid_preparation import trial_recipe

    values = np.arange(1., 65.).reshape(1, 8, 8)
    ds = xr.Dataset({"precipitation": (("time", "lat", "lon"), values)},
                    coords=dict(time=[np.datetime64("2025-01-01")], lat=range(8), lon=range(8)))
    ds.precipitation.attrs.update(units="mm", physical_variable="precipitation",
                                  spatial_semantics="cell_mean", temporal_semantics="accumulation")
    seen = []

    def score(candidate, baseline, truth, *args, **kwargs):
        seen.append(set(truth))
        return None, None

    monkeypatch.setattr(recipe_trials, "score", score)
    for phase in range(4):
        trial_recipe([ds, ds.copy()], 0, [recipe_trials.FILL], mask_phase=phase, mask_seed=7)
    assert all(len(points) == 16 for points in seen)
    assert len(set.union(*seen)) == 64


def row(path, kind, field):
    return dict(name=path.name, ref=(path, "", None, None), format=kind, fields=[field], bytes=path.stat().st_size)


def test_no_shared_groups_can_request_more_coverage_but_missing_scores_cannot_pass():
    empty = recipe_trials.audit(0, "withheld_station", None)
    assert empty["population"] == 0 and empty["status"] == "pending"
    release = {"status": "pending", "blockers": [dict(field="D1.F1", reason=empty["basis"])]}
    assert fallback_evidence.eligible({"execution_release": release})
    report = recipe_trials.audit(3, "withheld_station", lambda index: (index, None, None))
    assert report["status"] == "pending" and report["acceptance_sampling"]["unscored"] == 3


def task_for(route, fields, align):
    steps = ["units.harmonize", "format.normalize", align, recipe_trials.FILL, recipe_trials.BLEND]
    return dict(route=route, action="fusion", quantity="air_temperature", unit="degC", field_metadata=fields,
                fields=[(i, field["name"]) for i, field in enumerate(fields)], recipes=[steps]), steps


@pytest.mark.parametrize("kind", ["GeoTIFF", "NetCDF"])
def test_raster_and_mixed_grid_recipe_score_on_original_windows(tmp_path, kind):
    import rasterio
    from rasterio.transform import from_origin
    from ecodata.desktop import release_gate, windowed_grid_recipe

    def write_tiff(path, amount):
        with rasterio.open(path, "w", driver="GTiff", width=64, height=80, count=1,
                           dtype="float32", crs="EPSG:4326", transform=from_origin(0, 8, .1, .1)) as data:
            data.write(np.full((80, 64), amount, dtype="float32"), 1)
            data.set_band_unit(1, "degC")

    def write_secondary(path, amount):
        if kind == "GeoTIFF":
            write_tiff(path, amount)
            return
        xr.Dataset({"temperature": (("lat", "lon"), np.full((80, 64), amount), {"units": "degC"})},
                   coords=dict(lat=8-(np.arange(80)+.5)*.1, lon=(np.arange(64)+.5)*.1),
                   attrs={"crs": "EPSG:4326"}).to_netcdf(path, engine="scipy")

    left, right = tmp_path/"left.tif", tmp_path/("right.tif" if kind == "GeoTIFF" else "right.nc")
    write_tiff(left, 10.)
    write_secondary(right, 10.)
    fields = [dict(name="band_1", band=1, unit="degC"), dict(name="band_1" if kind == "GeoTIFF" else "temperature",
              band=1, unit="degC")]
    scan = dict(rows=[row(left, "GeoTIFF", fields[0]), row(right, kind, fields[1])])
    task, steps = task_for("windowed_grid_recipe", fields, "raster_task.spatial.bilinear")
    report = recipe_trials.review(scan, "C1:fusion", task, steps)
    assert report["status"] == "checked" and report["acceptance_sampling"]["unit"] == "withheld_spatial_block"
    assert report["sampled_scores"][0]["scoring_rule"].startswith("reliable_selection_v2")
    assert windowed_grid_recipe.run(scan, "C1:fusion", task, tmp_path, "data_only", None)["files"]

    write_secondary(right, 30.)
    report = recipe_trials.review(scan, "C1:fusion", task, steps)
    binding = [(i, f"D{i+1}.F1", field) for i, field in enumerate(fields)]
    candidate = dict(task="C1:fusion", steps=steps,
                     local_holdout_review=[dict(report, field=fid) for _, fid, _ in binding])
    candidate["execution_release"] = release_gate.assess(scan, candidate, task, binding)
    assert report["basis"] == "recipe_numeric_rejected"
    audit = report["acceptance_sampling"]
    assert audit["inspected"] == audit["failed"] == 2 and audit["posterior_probability"] == 0
    assert not fallback_evidence.eligible(candidate)


@pytest.mark.parametrize("kind", ["CSV", "Excel"])
def test_station_groups_are_scored_without_a_prebuilt_index(tmp_path, kind):
    from ecodata.desktop import csv_index, indexed_table_recipe
    from ecodata.desktop.sampling_plan import station_reserved

    names = [f"station-{i}" for i in range(32)]
    fields = [dict(name="temp [degC]", unit="degC", temporal_semantics="instantaneous") for _ in range(2)]
    sources = []
    for index, field in enumerate(fields):
        extension = "csv" if kind == "CSV" else "xlsx"
        path = tmp_path/f"table-{index}.{extension}"
        values = [["station_id", "time_utc", field["name"]],
                  *[(name, f"2024-10-01T{hour:02d}:00:00Z", 12.+hour) for name in names for hour in range(8)]]
        if kind == "CSV":
            with path.open("w", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerows(values)
        else:
            from openpyxl import Workbook
            book = Workbook()
            for record in values:
                book.active.append(record)
            book.save(path)
            book.close()
        sources.append(row(path, kind, field))
    scan = dict(rows=sources, path=tmp_path, signature=tuple(inventory(tmp_path)))
    task, steps = task_for("indexed_table_recipe", fields, indexed_table_recipe.LINEAR)
    report = recipe_trials.review(scan, "C1:fusion", task, steps)
    assert report["status"] == "checked", report
    assert report["acceptance_sampling"]["population"] == sum(map(station_reserved, names))
    if kind == "CSV":
        assert not list(tmp_path.glob(f"*{csv_index.SUFFIX}"))
        for source in sources:
            csv_index.build(source["ref"][0])
        assert recipe_trials.review(scan, "C1:fusion", task, steps)["status"] == "checked"
    assert indexed_table_recipe.run(scan, "C1:fusion", task, tmp_path, "data_only", None)["files"]

    from ecodata.desktop.holdout_plan import export_field, exposures, freeze

    scan["on_demand"] = True
    for index, field in enumerate(fields):
        assert export_field(scan, index, field, phase="field_probe")["perturbed_samples"]
    freeze(scan, {("C1", "fusion"): task}, [dict(candidate="C1", operation="fusion")])
    locked = task["locked_holdout"]["groups"]
    assert locked and all(station_reserved(key) for key in locked)
    report = recipe_trials.review(scan, "C1:fusion", task, steps)
    assert report["acceptance_sampling"]["population"] == len(locked)
    event = exposures(scan, 0, fields[0])["field_probe"]
    event["groups"] = (*event["groups"], locked[0])
    blocked = recipe_trials.review(scan, "C1:fusion", task, steps)
    assert blocked["status"] == "pending" and "重叠" in blocked["reason"]


def test_full_recipe_gate_does_not_accept_individual_column_checks():
    from ecodata.desktop.release_gate import assess

    field = {"name": "value", "unit": "degC"}
    scan = {"rows": [{"name": "source.csv", "format": "CSV"}]}
    task = {"action": "fusion", "route": "indexed_table_recipe"}
    recommendation = dict(task="C1:fusion", steps=[recipe_trials.BLEND],
                          local_holdout_review=[dict(field="D1.F1", basis="heldout_inputs", status="checked")])
    assert assess(scan, recommendation, task, [(0, "D1.F1", field)])["status"] == "pending"


def test_table_fusion_is_scored_even_when_gap_fill_passes(monkeypatch):
    from types import SimpleNamespace
    import pandas as pd
    from ecodata.desktop import indexed_table_recipe, recommendations

    index = pd.MultiIndex.from_product([["station"], pd.date_range("2024-01-01", periods=8, freq="h")],
                                       names=["station_id", "time"])
    primary = pd.Series(np.arange(8.) + 12, index=index)
    backup = primary + 8
    left = SimpleNamespace(groups={"station"}, held={"station"}, category="station", data=primary)
    right = SimpleNamespace(groups={"station"}, held={"station"}, category="station", data=backup)
    monkeypatch.setattr(indexed_table_recipe, "pair", lambda *_: (left, right))
    monkeypatch.setattr(indexed_table_recipe, "rows_for", lambda source, group: source.data)
    fields = [dict(name="temperature"), dict(name="temperature")]
    task = dict(quantity="air_temperature", unit="degC", fields=[(0, "temperature"), (1, "temperature")],
                field_metadata=fields)
    report = indexed_table_recipe.review({}, task, [indexed_table_recipe.LINEAR, recipe_trials.FILL, recipe_trials.BLEND])
    gap, fusion = report["sampled_scores"]
    assert (gap["assessment"], gap["candidate"]["RMSE"], gap["baseline"]["RMSE"]) == ("gap_fill", 0, 8)
    assert (fusion["assessment"], fusion["candidate"]["RMSE"], fusion["baseline"]["RMSE"]) == ("fusion", 4, 0)
    assert report["basis"] == "recipe_numeric_rejected" and report["acceptance_sampling"]["failed"] == 1
    detail = recommendations.explanation_text({"local_holdout_review": [report]})
    assert "补齐评分" in detail and "融合评分" in detail

    right.data = primary.copy()
    no_reference = indexed_table_recipe.review({}, task, [recipe_trials.FILL, recipe_trials.BLEND])
    assert no_reference["basis"] == "recipe_scoring_reference_insufficient"
    assert no_reference["acceptance_sampling"]["unscored_units"][0]["rule"] == "insufficient_fusion_score_support"
    assert "双源同时有效" in no_reference["reason"]


def test_missing_engineering_trial_reports_the_route():
    from ecodata.desktop.release_gate import assess, explanation

    scan = {"rows": [{"name": "source.nc", "format": "NetCDF"}]}
    task = {"action": "fusion", "route": "original_engineering"}
    report = recipe_trials.review(scan, "C1:fusion", task, [recipe_trials.BLEND])
    row = dict(task="C1:fusion", steps=[recipe_trials.BLEND], local_holdout_review=[dict(report, field="D1.F1")])
    release = assess(scan, row, task, [(0, "D1.F1", {})])
    assert release["status"] == "pending" and "原生工程端口" in explanation(release)


def test_calendar_population_includes_unseen_months_and_scores_cf_csv(tmp_path):
    from contextlib import ExitStack
    from ecodata.desktop import indexed_table_recipe as tables
    from ecodata.desktop.sampling_plan import PROBE_LIMIT

    # January fills the outbound prefix; March falls wholly in the CF holdout.
    dates = np.arange("2024-01-01", "2024-03-02", dtype="datetime64[m]")[::5]
    cf = tmp_path / "observations.nc"
    text = tmp_path / "observations.csv"
    values = np.full(len(dates), 10.)
    field = dict(name="temperature", unit="K", dimensions=["time"], temporal_semantics="instantaneous")
    dataset = xr.Dataset({"temperature": ("time", values, dict(units="K", cell_methods="time: point"))},
                         coords={"time": dates.astype("datetime64[ns]")})
    dataset.to_netcdf(cf, engine="scipy")
    assert dates[PROBE_LIMIT].astype("datetime64[M]") == np.datetime64("2024-01")
    with text.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_utc", "temperature"])
        writer.writerows((str(t), 10.) for t in dates)
    sources = [row(cf, "NetCDF", field), row(text, "CSV", field)]
    assert all(tables.supports(source, field) for source in sources)
    scan = dict(rows=sources)
    task, steps = task_for("indexed_table_recipe", [field, field], tables.LINEAR)
    task["unit"] = "K"
    with ExitStack() as stack:
        primary, backup = tables.pair(stack, scan, task)
        assert primary.held == {"2024-03"}
        assert backup.held == {"2024-02", "2024-03"}
    review = recipe_trials.review(scan, "C1:fusion", task, steps)
    assert review["status"] == "checked", review
    assert review["acceptance_sampling"]["population"] == 1
    assert tables.run(scan, "C1:fusion", task, tmp_path, "data_only", None)["files"]


def test_new_draw_can_be_replayed_and_missing_evidence_cannot_pass():
    groups = []
    def inspect(index):
        groups.append(index)
        if len(groups) == 1:
            return dict(unit=index, unscored=True)
    report = sampling_risk.evaluate_recipe(500, "withheld_station", inspect, **sampling_risk.recipe_settings())
    assert report["status"] == "pending" and report["unscored"] == 1 and report["failed"] == 0
    replay = []
    sampling_risk.evaluate_recipe(500, "withheld_station", replay.append,
                                  seed=report["sampling_seed"], **sampling_risk.recipe_settings())
    assert replay == groups and len(set(groups)) == report["sample_size"]
    assert "1 组缺少评分依据" in sampling_risk.recipe_summary(report)


def test_cf_time_only_fields_keep_self_reference_as_diagnostic_by_day(tmp_path):
    from ecodata.desktop import catalog_decision, discovery, recommendations
    from ecodata.desktop.scan import Scanner

    dates = np.arange("2024-01-01", "2024-01-21", dtype="datetime64[h]").astype("datetime64[ns]")
    attributes = dict(units="K", standard_name="air_temperature", cell_methods="time: point")
    for name in ("first", "second"):
        xr.Dataset({"tas": ("time", np.full(len(dates), 280.), attributes)},
                   coords={"time": dates}).to_netcdf(tmp_path / (name + ".nc"), engine="scipy")
    scan, _ = Scanner().scan(tmp_path)
    summary, _ = discovery.discover(scan, "data_only")
    originals, details = catalog_decision.local_fields(scan, summary)
    sources = [entry for entry in originals.values() if entry[2]["name"] == "tas"]
    pair, local = discovery.describe_pair(scan, "C1", *sources, "data_only", details)
    approved = dict(summary, candidates=[pair])
    checked, gated = discovery.check(scan, {"C1": local},
                                     {"decisions": [{"candidate": "C1", "action": "fusion"}]}, approved)
    assert gated["C1", "fusion"]["route"] == "indexed_table_recipe"
    plan = recommendations.make_request(scan, approved, checked, gated,prepared_graph_service=False)
    recipes, _ = recommendations.recommend_plan(plan, [], mock=True)
    recommendations.review_holdouts(scan, approved, plan, recipes, gated)
    report = recipes[0]["local_holdout_review"][0]
    audit = report["diagnostic_sampling"]
    assert (audit["unit"], audit["population"], audit["inspected"]) == ("withheld_day", 4, 4)
    assert audit["failed"] == 0 and "acceptance_sampling" not in report
    assert report["basis"] == "external_accuracy_reference_required"
    assert recipes[0]["execution_release"]["status"] == "pending"
    assert recommendations.finalize_recipes(plan, recipes, mock=True, api_key=None, progress=None) is None


def test_station_grid_preserves_true_mixed_format_holdout(tmp_path):
    from ecodata.desktop import station_grid
    from ecodata.desktop.sampling_plan import station_reserved

    dates = np.arange("2024-01-01T00", "2024-01-01T08", dtype="datetime64[h]")
    variable = "air_temperature"
    station = next(f"s{i}" for i in range(100) if station_reserved(f"s{i}"))
    p, q = tmp_path/"grid.nc", tmp_path/"sites.csv"
    xr.Dataset({variable: (("time", "lat", "lon"), np.full((8, 4, 4), 12.0), {"units": "degC"})},
               coords=dict(time=dates, lat=np.arange(4.), lon=np.arange(4.)),
               attrs={"crs": "EPSG:4326"}).to_netcdf(p, engine="scipy")
    with q.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows([["station_id", "time_utc", "longitude", "latitude", variable],
                                      *[(station, str(date)+":00:00Z", 1., 2., 14.) for date in dates]])
    field = dict(name=variable, unit="degC", dimensions=["time", "lat", "lon"])
    scan = dict(rows=[row(p, "NetCDF", field), row(q, "CSV", field)])
    task = dict(action="fusion", route="station_grid", quantity=variable, unit="degC", obs_unit="degC",
                grid_index=0, table_index=1, grid_name=variable, table_name=variable, table_header=variable)
    steps = [station_grid.UNIT_STEP, station_grid.FORMAT_STEP, station_grid.EXACT,
             station_grid.MATCH, recipe_trials.FILL, station_grid.BIAS]
    result = recipe_trials.review(scan, "C1:fusion", task, steps)
    assert result["status"] == "checked", result
    assert result["sampled_scores"][0]["candidate"]["RMSE"] == 0
    assert result["sampled_scores"][0]["baseline"]["RMSE"] == 2


def test_first_recommendation_checks_a_real_grid_pair_without_deep_read(tmp_path, monkeypatch):
    import json
    import rasterio
    from rasterio.transform import from_origin
    from ecodata.desktop import catalog_decision, discovery, recommendations
    from ecodata.desktop.scan import Scanner

    variable = "air_temperature"
    for prefix in ("source-a", "source-b"):
        with rasterio.open(tmp_path/f"{prefix}.tif", "w", driver="GTiff", width=64, height=80, count=1,
                           dtype="float32", crs="EPSG:4326", transform=from_origin(0, 8, .1, .1)) as data:
            data.write(np.full((80, 64), 13.0, dtype="float32"), 1)
            data.update_tags(ecodata_metadata=json.dumps(dict(export_variable=variable,
                variables={variable: dict(units="degC", physical_variable=variable, data_kind="continuous")})))
    scan, _ = Scanner().scan(tmp_path)
    summary, _ = discovery.discover(scan, "data_only")
    originals, details = catalog_decision.local_fields(scan, summary)
    pair, local = discovery.describe_pair(scan, "C1", *originals.values(), "data_only", details)
    approved = dict(summary, candidates=[pair])
    monkeypatch.setattr("ecodata.desktop.scan.read_selected", lambda *_: pytest.fail("preflight read whole raster"))
    # Reproduce the desktop entry where the file agent stops at alignment.
    alignment = [{"candidate": "C1", "action": "align_space"}]
    assert discovery.workflow_proposals(alignment, dict(approved, task_goal="grid_comparison")) == alignment
    gate = discovery.workflow_proposals(alignment, approved)
    checked, gated = discovery.check(scan, {"C1": local}, {"decisions": gate}, approved)
    assert gated["C1", "fusion"]["route"] == "windowed_grid_recipe"
    scan["on_demand"] = True
    # This test checks the retained legacy fixed-recipe approval, not search admission.
    plan = recommendations.make_request(scan, approved, checked, gated, sampled_workflow=True,prepared_graph_service=False)
    execution = gated["C1", "fusion"]
    assert execution["locked_holdout"]["window"] == [0, 64, 64, 16]
    from ecodata.desktop.holdout_plan import exposures
    fields = [(index, field) for index, _, field in originals.values()]
    assert all(item["perturbed_samples"] for item in fallback_evidence.supplement(scan, fields))
    assert all("supplement" in exposures(scan, index, field) for index, field in fields)
    task, = plan["tasks"]
    assert task["sample_complete"]
    assert all(len(f["perturbed_samples"]) == 48 for f in task["context"]["pair_details"])
    evidence = task["pretrial_evidence"]
    assert evidence["source_sample_counts"] == {
        field["id"]: len(field["perturbed_samples"]) for field in task["context"]["pair_details"]}
    assert evidence["sample_comparison"] == "unpaired_perturbed_examples"
    assert set(evidence["distinguishing_steps"]) == {item["id"] for item in task["allowed_recipes"]}
    assert "locked_holdout" not in json.dumps(evidence)
    candidate = task["allowed_recipes"][0]
    pretrial, = recommendations.validate(dict(recommendations=[dict(task=task["id"], recipe=candidate["id"],
        reason="需要同一批参考检查", pretrial_assessment="uncertain")]), plan)
    assert pretrial["pretrial_assessment"]["outlook"] == "uncertain"
    assert not pretrial.get("final_verified_recommendation")
    from ecodata.desktop import deepseek_transport
    field_id = task["context"]["fields"][0]
    calls = []

    def fetch(request, opener, stage, digest, progress):
        messages = json.loads(request.data)["messages"]
        body = json.loads(messages[1]["content"])
        calls.append(body)
        if len(calls) == 1:
            assert body["can_request_recommendation_evidence"]
            assert all(len(field["perturbed_samples"]) == 48
                       for field in body["tasks"][0]["context"]["pair_details"])
            output = dict(evidence_requests=[dict(task=task["id"], kind=kind, target=target) for kind,target in (
                ("field_metadata", field_id), ("numeric_samples", field_id),
                ("document_excerpt", "E1"), ("numeric_samples", "D99.F1"))])
        else:
            assert len(calls) == 2 and "can_request_recommendation_evidence" not in body
            metadata, samples, document, denied = body["recommendation_evidence"]
            assert metadata["metadata"]["unit"] == "degC"
            assert len(samples["perturbed_samples"]) == 16
            assert document["excerpt"] == "已批准说明"
            assert denied["status"] == "outside_scope"
            output = dict(recommendations=[dict(task=task["id"], recipe=candidate["id"], reason="补问后建议试跑",
                                                pretrial_assessment="uncertain")])
        return dict(id=f"mock-{len(calls)}", choices=[dict(message=dict(content=json.dumps(output)))])

    with monkeypatch.context() as patch:
        patch.setattr(deepseek_transport, "fetch", fetch)
        followup_plan = dict(plan, engineering_documents=[dict(id="E1", text="已批准说明")])
        supplemented, receipt = recommendations.recommend_plan(followup_plan, [], scan=scan, api_key="test-only")
        assert len(receipt["api_receipts"]) == 2 and receipt["evidence_followup_sha256"]
        assert all(entry["context_budget"]["conservative_total"] <= 1_000_000 for entry in receipt["api_receipts"])
        assert supplemented[0]["pretrial_assessment"]["outlook"] == "uncertain"
        assert "execution_release" not in supplemented[0]
        assert execution["locked_holdout"]["window"] == [0, 64, 64, 16]
    with pytest.raises(ValueError, match="未收到有效推荐"):
        recommendations.validate(dict(recommendations=[dict(task=task["id"], recipe="registry_advice",
                                 algorithms=["spatial.blend"])]), plan)
    recipes, _ = recommendations.recommend_plan(plan, [], mock=True)
    assert len(recipes) == 1 and recipe_trials.FILL in recipes[0]["steps"]
    recommendations.review_holdouts(scan, approved, plan, recipes, gated)
    from ecodata.desktop import release_gate
    assert recipes[0]["execution_release"]["status"] == "pending" and not release_gate.allowed(recipes[0])
    assert recipes[0]["local_holdout_review"][0]["basis"] == "external_accuracy_reference_required"
    assert recommendations.finalize_recipes(plan, recipes, mock=True, api_key=None, progress=None) is None
    index, field = fields[0]
    supplement = exposures(scan, index, field)["supplement"]
    supplement["windows"] = (*supplement["windows"], dict(row=64, column=0, width=4, height=4))
    blocked = recipe_trials.review(scan, "C1:fusion", execution, recipes[0]["steps"])
    assert blocked["status"] == "pending" and "重叠" in blocked["reason"]
