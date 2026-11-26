from copy import deepcopy
import pytest
from ecodata.desktop.recommendations import continue_rejected_search, is_private_fusion_candidate
from ecodata.desktop.release_gate import allowed, selectable, require


def fixture():
    row = dict(task="C1:fusion", id="recipe", steps=["spatial.blend"],
               execution_release=dict(status="failed"), local_holdout_review=[dict(
                   basis="recipe_numeric_rejected", acceptance_sampling=dict(failures=[dict(
                       rule="recipe_score_below_single_source", unscored=False)]))])
    return [row], {("C1", "fusion"): dict(route="independent_grid_search")}


def test_numeric_failure_allows_search_not_recipe():
    rows, gated = fixture()
    original = deepcopy(rows[0])
    continue_rejected_search(rows, gated)
    assert rows[0] == original
    assert not allowed(rows[0]) and not selectable(rows[0])
    assert selectable(rows[1]) and not allowed(rows[1])
    assert not is_private_fusion_candidate(rows[1])
    require([rows[1]], allow_search=True)
    with pytest.raises(ValueError): require([rows[1]])
    assert rows[1]["steps"] == ["spatial.source"]
    assert gated[("C1", "fusion")]["search_after_rejection"]["rejected_ids"] == ["recipe"]
    continue_rejected_search(rows, gated)
    assert len(rows) == 2


@pytest.mark.parametrize("change", ["pending", "unscored", "coverage", "unsupported", "accepted"])
def test_unsafe_or_unnecessary_continuation_not_admitted(change):
    rows, gated = fixture()
    report = rows[0]["local_holdout_review"][0]
    if change == "pending": rows[0]["execution_release"]["status"] = "pending"
    if change == "unscored": report["acceptance_sampling"]["failures"][0]["unscored"] = True
    if change == "coverage": report["acceptance_sampling"]["failures"][0]["rule"] = "recipe_output_missing_on_reference"
    if change == "unsupported": gated[("C1", "fusion")]["route"] = "unknown_adapter"
    if change == "accepted": rows.append(dict(task="C1:fusion", id="accepted", execution_release=dict(status="released")))
    count = len(rows)
    continue_rejected_search(rows, gated)
    assert len(rows) == count
    assert "search_after_rejection" not in gated[("C1", "fusion")]


def test_untrusted_admission_cannot_bypass_gate():
    rows, _ = fixture()
    rows[0]["search_admission"] = dict(status="SEARCH_ONLY")
    assert not selectable(rows[0])
    with pytest.raises(ValueError): require(rows)


def test_insufficient_scoring_support_only_admits_bounded_exploration():
    rows,gated=fixture();rows[0]['execution_release']['status']='pending'
    rows[0]['local_holdout_review']=[dict(basis='recipe_scoring_reference_insufficient',
        acceptance_sampling=dict(unscored_units=[dict(rule='insufficient_score_support')]))]
    continue_rejected_search(rows,gated)
    assert len(rows)==2 and selectable(rows[1]) and not allowed(rows[0]) and not allowed(rows[1])
    assert rows[1]['search_admission']['reason']=='insufficient_reference_exploration_only'


def test_page_shows_failed_recipe_separately_from_search_permission(monkeypatch):
    from PySide6.QtWidgets import QApplication, QCheckBox
    from ecodata.desktop.__main__ import Window
    app = QApplication.instance() or QApplication([])
    rows, gated = fixture()
    rows[0].update(name="failed", description="failed", reason="numeric failure")
    continue_rejected_search(rows, gated)
    window = Window()
    monkeypatch.setattr(window, "original_background_ready", lambda: False)
    window.render_recommendations(dict(mode="功能测试", recommendations=rows))
    boxes = [box for box in window.findChildren(QCheckBox) if box.text().startswith("C1:fusion")]
    assert len(boxes) == 1
    assert boxes[0].isEnabled() and boxes[0].isChecked()
    assert "复核" in window.recommendation_status.text()
    assert window.recommend_cards.count() == 1 and window.recommendation_detail_cards.count() == 1
    assert not allowed(rows[0]) and not allowed(rows[1])
    window.batch_timer.stop()
    window.close()
    app.processEvents()


@pytest.mark.parametrize("route,steps", [
    ("station_grid", ["station.time.exact", "station.grid.nearest", "station.bias.holdout"]),
    ("windowed_grid_recipe", ["raster_task.spatial.bilinear", "fusion.series_missing_fallback", "spatial.blend"]),
    ("indexed_table_recipe", ["temporal.wide_table_interpolation", "fusion.series_missing_fallback", "spatial.blend"]),
])
def test_shared_admission_keeps_preparation_not_rejected_recipe(route,steps):
    rows,gated=fixture();rows[0]["steps"]=steps;gated[("C1","fusion")]["route"]=route
    continue_rejected_search(rows,gated)
    assert len(rows)==2 and selectable(rows[1]) and not allowed(rows[1])
    assert "spatial.blend" not in rows[1]["steps"] and "station.bias.holdout" not in rows[1]["steps"]
    assert rows[0]["steps"]==steps


def test_unlocated_series_never_invents_coordinates():
    import pandas as pd
    from ecodata.algorithms.paired_station_search import prepare
    frame=pd.DataFrame(dict(station="series",grid_time=pd.date_range("2020-01-01",periods=40),
                            observed=range(40),background=range(1,41),unit="K"))
    data,state,validation,held,protocol=prepare(frame,"temperature","K",unlocated_series=True)
    assert set(data.temperature.dims)=={"time"}
    assert not {"lat","lon","latitude","longitude"}.intersection(data.coords)
    assert "crs" not in data.attrs
    changed=frame.copy();changed.loc[20:,"observed"]+=1000
    _,other,_,_,_=prepare(changed,"temperature","K",unlocated_series=True)
    assert state.equals(other)


def test_no_reference_isolation_cannot_promote_even_a_better_candidate(tmp_path):
    import pandas as pd
    from ecodata.algorithms.platform_search import SearchInput, submit_search
    data=pd.DataFrame(dict(station="series",grid_time=pd.date_range("2000-01-01",periods=48,freq="h"),
                           observed=280.+2+pd.Series(range(48)),background=280.+pd.Series(range(48)),unit="K"))
    request=dict(task="temporal",adapter="paired_station_calibration",variable="temperature",unit="K",
                 unlocated_series=True,recommended_recipes=[],reference_basis="PRIMARY_SERIES_PROXY",
                 recommendation_isolation=dict(status="INSUFFICIENT",reason="Used for recommendation"))
    result=submit_search([SearchInput("pairs.csv","paired_observations",data.to_csv(index=False).encode())],
                        request,None,tmp_path/"search",seed=11,
                        budget=dict(objective_limit=80,total_call_limit=120,proposal_limit=256,wall_seconds=30))
    station=result["stations"][0]
    assert station["status"]=="PASS", station
    assert station["evolution_used"] and station["mutation_actual_new_executions"]>0
    assert not station["recommendation_validation"]
    assert station["delivery_status"]=="BASELINE_RETAINED"
    assert not station["post_mutation_acceptance"]["recommended"]
    assert station["recommendation_review"]["status"]=="TRIAL_AVAILABLE"
    assert station["recommendation_review"]["evidence_status"]=="INSUFFICIENT"
    import xarray as xr
    with xr.open_dataset(station["product"]) as product:
        assert set(product.temperature.dims)=={"time"}
        assert not {"lat","lon"}.intersection(product.coords)


def test_ready_continuation_does_not_skip_supplementary_recommendation(monkeypatch):
    from PySide6.QtWidgets import QApplication
    from ecodata.desktop.__main__ import Window
    app=QApplication.instance() or QApplication([])
    rows,gated=fixture();continue_rejected_search(rows,gated)
    rows[0]["execution_release"]["blockers"]=[dict(field="D1.F1",reason="recipe_numeric_rejected")]
    window=Window();window.scan={"data":"functional-test"};window.recommendation_rows=rows
    monkeypatch.setattr(window,"effective_api_key",lambda:"not-used")
    from ecodata.desktop import fallback_evidence
    monkeypatch.setattr(fallback_evidence,"approved_fields",lambda *a:[(0,dict(name="rain",unit="mm"))])
    calls=[]
    monkeypatch.setattr(window,"review_full_fallback",lambda row,**k:calls.append((row,k)))
    window.mock.setChecked(False)
    window.send_next_automatic_fallback()
    window.send_next_automatic_fallback()
    assert calls==[(rows[0],dict(automatic=True))]
    assert rows[0]["automatic_fallback_started"]
    assert selectable(rows[1]) and not allowed(rows[0])
    window.batch_timer.stop();window.close();app.processEvents()


def test_windowed_search_failure_never_returns_rejected_template(tmp_path,monkeypatch):
    import numpy as np
    import pandas as pd
    import xarray as xr
    from ecodata.desktop import local_search,windowed_grid_recipe
    template=xr.Dataset({"temperature":(("time","y","x"),np.ones((3,3,3)),dict(
        units="K",physical_variable="air_temperature",physical_definition="temperature",temporal_semantics="instantaneous"))},
        coords=dict(time=pd.date_range("2020-01-01",periods=3),y=[3.,2.,1.],x=[1.,2.,3.]),attrs=dict(crs="EPSG:4326"))
    pairs=[(np.ones((3,3)),np.ones((3,3))*2) for _ in range(3)]
    monkeypatch.setattr(local_search,"submit",lambda *a,**k:(_ for _ in ()).throw(ValueError("actual execution failure")))
    task=dict(quantity="temperature",unit="K",search_after_rejection={"status":"SEARCH_ONLY"})
    with pytest.raises(ValueError,match="actual execution failure"):
        windowed_grid_recipe.mutate("C:fusion",task,template,list(template.time.values),pairs,tmp_path,"data_only",None)


def test_station_raw_group_uses_same_codec_and_rejects_conflicting_duplicates(tmp_path):
    import numpy as np
    import pandas as pd
    import xarray as xr
    from ecodata.desktop.station_grid import load,EXACT
    grid=tmp_path/"model.nc"
    clock=pd.date_range("2020-01-01",periods=12,freq="h")
    xr.Dataset({"temperature":(("time","lat","lon"),np.full((12,2,2),280.))},
               coords=dict(time=clock,lat=[30.,31.],lon=[110.,111.])).to_netcdf(grid,engine="scipy")
    columns=["STATION","DATE","LATITUDE","LONGITUDE","ELEVATION","REPORT_TYPE","REPORT_TYPE","TMP"]
    rows=[["S",str(t)+"Z",30.5,110.5,20.,"FM-15","FM-15","+0100,1"] for t in clock]
    data=pd.DataFrame(rows,columns=columns)
    scan=dict(rows=[dict(ref=(grid,"",None,None),containers=[],name="model.nc",format="NetCDF",bytes=grid.stat().st_size),{}])
    task=dict(grid_index=0,table_index=1,grid_name="temperature",table_name="TMP.air_temperature",
              table_header="TMP.air_temperature",quantity="air_temperature",unit="K",obs_unit="degC")
    result=load(scan,task,table_data=data)[EXACT]
    assert len(result)==12 and np.allclose(result.observed,283.15)
    assert result.attrs['matching_audit']['identical_duplicate_columns']==['REPORT_TYPE']
    data.iloc[0,6]="conflict"
    with pytest.raises(ValueError,match="内容冲突"):
        load(scan,task,table_data=data)
