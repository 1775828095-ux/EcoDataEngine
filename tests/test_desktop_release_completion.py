import io
import json
from zipfile import ZipFile

import numpy as np
import pytest

from ecodata.desktop import fallback_evidence, independent_review, sampling_risk
from ecodata.desktop.scan import inventory


def source(path, kind, fields):
    return dict(name=path.name, format=kind, ref=(path, "", None, None), bytes=path.stat().st_size, fields=fields)


def test_risk_calculation_and_one_round_failure():
    assert sampling_risk.plan(10_000_000)["sample_size"] == 59
    assert sampling_risk.plan(10_000_000, defect_share=0.01)["sample_size"] == 299
    result = sampling_risk.evaluate(1000, "station", lambda i: {"station": i})
    assert result["rounds"] == 1 and result["failed"] == result["inspected"]
    assert result["status"] == "failed"


def test_ruptures_detects_sustained_change():
    from ecodata.desktop.segment_review import change_candidates
    assert change_candidates([0] * 32 + [10] * 32) == [32]


def test_runtime_splits_and_rechecks_evidenced_netcdf_intervals(tmp_path):
    import xarray as xr
    from ecodata.desktop.release_gate import runtime
    path = tmp_path / "rain.nc"
    widths = np.array([1] * 24 + [2] * 24)
    edges = np.r_[0, widths.cumsum()]
    xr.Dataset({"rain": ("time", np.ones(48), {"units": "mm/h"}),
                "time_bounds": (("time", "bounds"), np.column_stack([edges[:-1], edges[1:]]))},
               coords={"time": ("time", edges[:-1], {"units": "hours since 2000-01-01",
                                                      "bounds": "time_bounds"})}).to_netcdf(path)
    field = dict(name="rain", unit="mm/h")
    inputs = dict(path=path, signature=tuple(inventory(path)), rows=[source(path, "NetCDF", [field])])
    report = runtime(inputs, dict(fields=[(0, "rain")], steps=["engineering.precipitation_adaptation"]), tmp_path)[0]
    segments = report["segmented_recheck"]["segments"]
    assert [(item["start"], item["stop"]) for item in segments] == [(0, 24), (24, 48)]
    assert all(item["precipitation_trial"]["operation"] == "rate_to_interval_amount" for item in segments)


def test_native_raster_copy_preserves_approved_codes(tmp_path):
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    path = tmp_path / "source.tif"
    original = np.array([[1, 6], [9, 0]], dtype=np.uint16)
    with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=2,
                       transform=from_origin(0, 1, .01, .01), crs="EPSG:4326",
                       nodata=0, dtype="uint16") as dataset:
        dataset.write(original * 10, 1)
        dataset.write(original, 2)
    numeric = dict(name="rain", band=1, unit="mm")
    code = dict(name="land_cover", band=2, data_kind="categorical")
    inputs = dict(path=path, signature=tuple(inventory(path)), rows=[source(path, "GeoTIFF", [numeric, code])])
    selected = [(0, numeric)]
    archive = tmp_path / "sanitized.zip"
    manifest = fallback_evidence.build(inputs, selected + fallback_evidence.classification_fields(inputs, selected), archive)
    with ZipFile(archive) as package:
        with MemoryFile(package.read(manifest[1]["native_member"])) as memory, memory.open() as dataset:
            assert dataset.crs.to_string() == "EPSG:4326"
            pixels = dataset.read(1, masked=True)
            assert pixels.compressed().tolist() == [1, 6, 9] and pixels.mask[-1, -1]
        with MemoryFile(package.read(manifest[0]["native_member"])) as memory, memory.open() as dataset:
            assert (dataset.read(1, masked=True).compressed() > [10, 60, 90]).all()


def test_fallback_consensus_and_new_independent_review(tmp_path):
    original = tmp_path / "original.csv"
    independent = tmp_path / "new.csv"
    original.write_text("rain (mm)\n" + "\n".join(["1"] * 32), encoding="utf-8")
    independent.write_text("rain (mm)\n" + "\n".join(["2"] * 32), encoding="utf-8")
    field = dict(name="rain", original_header="rain (mm)", unit="mm")
    inputs = dict(path=original, signature=tuple(inventory(original)), rows=[source(original, "CSV", [field])])
    archive = tmp_path / "sanitized.zip"
    manifest = fallback_evidence.build(inputs, [(0, field)], archive)
    option = dict(id="R1", name="Parquet", steps=["format.parquet"], description="原版格式转换")
    initial = dict(task="C1:convert_format", local_recipe_options=[option],
                   execution_fields=[(0, "rain", "D1.F1")])
    results = fallback_evidence.transmit(archive, manifest, fallback_evidence.digest_file(archive),
        api_key="not-live", recipes=[option],
        send=lambda *args: ({"content": json.dumps({"recipe": "R1"})}, {"id": "mock"}))
    candidate = independent_review.start(initial, results)
    assert candidate["execution_release"]["status"] == "pending"
    with pytest.raises(ValueError, match="重复"):
        independent_review.review(inputs, candidate, {}, "D1.F1", original)
    report = independent_review.review(inputs, candidate, {}, "D1.F1", independent)
    independent_review.apply(inputs, candidate, {}, report)
    assert report["independent"] and candidate["execution_release"]["status"] == "released"
    with pytest.raises(ValueError, match="上一轮"):
        independent_review.review(inputs, dict(candidate, previous_independent_hashes=[report["independent_sha256"]]),
                                  {}, "D1.F1", independent)


def test_targeted_supplement_rejects_reserved_station(tmp_path):
    from ecodata.desktop.sampling_plan import station_reserved

    held = next(str(i) for i in range(100) if station_reserved(str(i)))
    recommended = next(str(i) for i in range(100) if not station_reserved(str(i)))
    path = tmp_path / "stations.csv"
    path.write_text(f"station_id,rain\n{held},99\n{recommended},12\n", encoding="utf-8")
    field = dict(name="rain", unit="mm")
    row = source(path, "CSV", [field])
    assert fallback_evidence.targeted_table(row, field, {}, ("station", recommended)) == [12]
    with pytest.raises(ValueError, match="留作独立"):
        fallback_evidence.targeted_table(row, field, {}, ("station", held))
def test_formal_acceptance_checks_the_same_reserved_axis(tmp_path):
    import numpy as np
    import xarray as xr
    from ecodata.desktop.sampling_risk import attach

    path = tmp_path / "grid.nc"
    values = np.ones((20, 20))
    values[:, 16:] = np.nan
    xr.Dataset({"rain": (("time", "x"), values)}, coords={"time": np.arange(20)}).to_netcdf(path)
    field = {"name": "rain"}
    row = {"format": "NetCDF", "name": path.name, "bytes": path.stat().st_size,
           "ref": (path, "", None, None)}
    scan, cache = {"rows": [row]}, {}
    for axis, basis, unit, status in [
        ("time", "time_windows", "withheld_time_window", "checked"),
        ("x", "spatial_blocks", "withheld_spatial_window", "failed"),
    ]:
        report = dict(field="D1.F1", status="checked", holdout_basis=basis,
                      partition=dict(axis=axis, holdout_start=16, size=20))
        result = attach(scan, 0, field, report, cache)["acceptance_sampling"]
        assert (result["unit"], result["status"]) == (unit, status)


def test_formal_acceptance_uses_reserved_month_even_with_station_column(tmp_path):
    from ecodata.desktop.sampling_risk import attach

    path = tmp_path / "stations.csv"
    path.write_text("station_id,time,rain\nA,2021-01-01,1\nA,2021-02-01,\n", encoding="utf-8")
    row = {"format": "CSV", "name": path.name, "bytes": path.stat().st_size, "ref": (path, "", None, None)}
    report = dict(field="D1.F1", status="checked", holdout_basis="calendar_months",
                  partition=dict(held_month="2021-02"))
    result = attach({"rows": [row]}, 0, {"name": "rain"}, report, {})["acceptance_sampling"]
    assert (result["unit"], result["status"]) == ("withheld_month", "failed")


def test_formal_acceptance_checks_the_reserved_worksheet(tmp_path):
    from openpyxl import Workbook

    path = tmp_path / "sheets.xlsx"
    workbook = Workbook()
    workbook.active.title = "observations"
    workbook.active.append(["rain"])
    workbook.active.append([10])
    held = workbook.create_sheet("reserved")
    held.append(["rain"])
    held.append(["missing"])
    workbook.save(path)
    field = dict(name="rain", sheet="observations")
    report = dict(field="D1.F1", status="checked", holdout_basis="worksheet_groups",
                  partition=dict(held_sheet="reserved"))
    scan = {"rows": [source(path, "Excel", [field])]}
    result = sampling_risk.attach(scan, 0, field, report, {})["acceptance_sampling"]
    assert (result["unit"], result["status"]) == ("withheld_worksheet", "failed")
def test_recipe_acceptance_finds_smallest_design_and_stops_only_on_excess_failures():
    from scipy.stats import hypergeom
    from ecodata.desktop.sampling_risk import evaluate_recipe, recipe_plan

    design = recipe_plan(10_000)
    n, c = design["sample_size"], design["allowed_failures"]
    assert n > c > 0
    assert hypergeom.sf(c, 10_000, 500, n) <= .05
    assert hypergeom.cdf(c, 10_000, 1000, n) <= .05
    for size in range(1, n):
        threshold = hypergeom.ppf(.95, 10_000, 500, size)
        assert hypergeom.cdf(threshold, 10_000, 1000, size) > .05
    rejected = evaluate_recipe(10_000, "day", lambda index: dict(unit=index, rule="low_score"))
    assert rejected["status"] == "failed" and rejected["inspected"] == c + 1
    verified = evaluate_recipe(10_000, "day", lambda _: None)
    assert verified["status"] == "checked" and verified["inspected"] == n
    pending = evaluate_recipe(10_000, "day", lambda _: {"unscored": True})
    assert pending["status"] == "pending" and pending["failed"] == 0 and pending["inspected"] == n
    assert recipe_plan(10_000, acceptable=.05, rejectable=.05, budget=512)["sample_size"] is None
