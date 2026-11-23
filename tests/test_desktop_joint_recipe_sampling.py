import csv

import numpy as np
import pytest
import xarray as xr

from ecodata.desktop import fallback_evidence, recipe_trials, sampling_risk
from ecodata.desktop.scan import inventory


def test_default_hypergeometric_boundary_and_no_early_acceptance():
    from scipy.stats import hypergeom

    policy = sampling_risk.recipe_settings()
    result = sampling_risk.evaluate_recipe(10_000_000, "time_window", lambda _: None, **policy)
    assert (result["status"], result["sample_size"], result["allowed_failures"], result["inspected"]) == ("checked", 59, 0, 59)
    assert result["bad_failures"] == 500_000 and result["false_accept_risk"] <= .05
    assert hypergeom.pmf(0, result["population"], result["bad_failures"], 58) > .05
    failed = sampling_risk.evaluate_recipe(1000, "time", lambda _: {"rule": "failed_score"}, **policy)
    assert failed["inspected"] == 1 and failed["status"] == "failed"


def row(path, kind, field):
    return dict(name=path.name, ref=(path, "", None, None), format=kind, fields=[field], bytes=path.stat().st_size)


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
    assert report["basis"] == "recipe_numeric_rejected" and report["acceptance_sampling"]["inspected"] == 1
    assert fallback_evidence.eligible(candidate)


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


def test_full_recipe_gate_does_not_accept_individual_column_checks():
    from ecodata.desktop.release_gate import assess

    field = {"name": "value", "unit": "degC"}
    scan = {"rows": [{"name": "source.csv", "format": "CSV"}]}
    task = {"action": "fusion", "route": "indexed_table_recipe"}
    recommendation = dict(task="C1:fusion", steps=[recipe_trials.BLEND],
                          local_holdout_review=[dict(field="D1.F1", basis="heldout_inputs", status="checked")])
    assert assess(scan, recommendation, task, [(0, "D1.F1", field)])["status"] == "pending"


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
    checked, gated = discovery.check(scan, {"C1": local}, {"decisions": [{"candidate": "C1", "action": "fusion"}]}, approved)
    assert gated["C1", "fusion"]["route"] == "windowed_grid_recipe"
    plan = recommendations.make_request(scan, approved, checked, gated, include_samples=False)
    recipes, _ = recommendations.recommend_plan(plan, [], mock=True)
    assert len(recipes) == 1 and recipe_trials.FILL in recipes[0]["steps"]
    recommendations.review_holdouts(scan, approved, plan, recipes, gated)
    recommendations.finalize_recipes(plan, recipes, mock=True, api_key=None, progress=None)
    assert recipes[0]["final_verified_recommendation"]
