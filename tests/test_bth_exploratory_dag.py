"""Small deterministic interface fixtures, not environmental accuracy claims."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"));sys.path.append(str(ROOT/".venv-golem/Lib/site-packages"))
from run_bth_exploratory_fusion import setup,score,FEATURES
from ecodata.algorithms.dag import Artifact,DAGError
from ecodata.algorithms.bth_exploratory_dag import BTHFusionDAG
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from golem.core.optimisers.graph import OptGraph


def inputs():
    lon,lat=np.meshgrid(np.linspace(115.,117.,5),np.linspace(37.,40.,4))
    train=pd.DataFrame(dict(station_id=[f"s{i}" for i in range(20)],longitude=lon.ravel(),latitude=lat.ravel()))
    query=pd.DataFrame(dict(station_id=["q1","q2"],longitude=[116.,116.2],latitude=[38.,38.5]))
    for table in (train,query):
        table["time"]=pd.Timestamp("2019-08-15 00:30");table["unit"]="ug m-3"
        table["geos"]=table.longitude*2+table.latitude
        for i,n in enumerate(FEATURES["era5"]):table[n]=table.longitude**(1+i%2)+table.latitude*(i+1)
    train["value"]=train.geos*.4+3+np.sin(train.latitude);train["split"]="train"
    meta=dict(unit="ug m-3",metric_crs="EPSG:32650",time=train.time.iloc[0],contract="fixture",
              coordinate_status="WGS84_USER_WORKING_ASSUMPTION")
    return train,query,meta


def test_same_registry_generates_seven_actual_routes():
    registry,factory=setup();train,query,meta=inputs()
    pipeline=UnifiedAlgorithmPipeline(registry)
    recipes=factory.recipes("validated_field")
    assert len(recipes)==7
    for _,recipe in recipes:
        graph=OptGraph([factory.instantiate(recipe)])
        result=pipeline.run_dag(graph,task_id="bth_exploratory_pm25",external={
            "train":Artifact("training",train,meta),"query":Artifact("query",query,meta)},version="fixture")
        field=result["outputs"]["field"].value
        assert np.isfinite(field["prediction"]).all() and (field["prediction"]>=0).all()
        assert field["quality"]["status"]=="WARN"
        assert not field["quality"]["accuracy_assessed"]


@pytest.mark.parametrize("fault",["query_label","not_train","no_assumption","bad_unit","missing_covariate"])
def test_source_rejects_invalid_contracts(fault):
    train,query,meta=inputs();key="train";table=train.copy()
    if fault=="query_label":key="query";table=query.assign(value=2)
    if fault=="not_train":table.loc[0,"split"]="test"
    if fault=="no_assumption":meta.pop("coordinate_status")
    if fault=="bad_unit":table["unit"]="mg m-3"
    if fault=="missing_covariate":table=table.drop(columns="geos")
    with pytest.raises(DAGError):BTHFusionDAG().source({},dict(source=key),{key:Artifact("training",table,meta)})


def test_five_fold_residuals_are_station_disjoint_and_state_owned():
    train,query,meta=inputs();backend=BTHFusionDAG();a=Artifact("training",train,meta)
    state=backend.fit_calibration({"train":a},{},{})["state"]
    for fold in state.value["folds"]:assert not set(fold["fit"])&set(fold["held"])
    changed=train.copy();changed.loc[0,"value"]+=1
    with pytest.raises(DAGError,match="owner"):
        backend.calibrated({"train":Artifact("training",changed,meta),"query":Artifact("query",query,meta),"state":state},{},{})


def test_failed_predictions_keep_common_denominator():
    result=score(pd.DataFrame(dict(truth=[10.,20.],prediction=[10.,np.nan])))
    assert result["expected"]==2 and result["coverage"]==.5


def test_gap_can_be_completed_by_official_mutation():
    from golem.core.optimisers.genetic.operators import base_mutations
    _,factory=setup();initial=factory.initial()
    assert factory.inspect(initial)["gap"]>0
    completed=base_mutations.add_separate_parent_node(initial,factory)
    assert factory.inspect(completed,complete=True)["gap"]==0


def test_shared_search_accepts_utf8_protocol_and_real_task_callback(tmp_path,monkeypatch):
    import json
    import run_taiwan_composition_search as search
    train,query,meta=inputs();called=[]
    protocol=dict(times=[str(meta["time"])],proposal_budget=10,wall_seconds=20,note="中文路径契约")
    (tmp_path/"protocol.json").write_text(json.dumps(protocol,ensure_ascii=False),encoding="utf-8")
    class FixtureEvaluator:
        def __init__(self,cases,folder):
            assert cases==["fixture"]
            folder.mkdir(parents=True,exist_ok=True)
            registry,self.factory=setup();self.pipeline=UnifiedAlgorithmPipeline(registry)
        def execute(self,graph,key):
            called.append(key)
            result=self.pipeline.run_dag(graph,task_id="bth_exploratory_pm25",external={
                "train":Artifact("training",train,meta),"query":Artifact("query",query,meta)},version="search-fixture")
            values=result["outputs"]["field"].value["prediction"]
            error=float(np.sqrt(np.mean((values-20.)**2)))
            return dict(RMSE=error,score=error,status="FIXTURE_NUMERICAL_EXECUTION")
    monkeypatch.setattr(search,"OUT",tmp_path)
    summary=search.search_job("R",11,budget=2,output=tmp_path,evaluator_factory=FixtureEvaluator,
        case_provider=lambda selected:["fixture"])
    assert summary["optimizer_error"] is None
    assert len(called)==1 and summary["counts"]["real_graph_evaluations"]==1


def test_source_path_recovery_uses_unique_sha_and_rejects_changed_bytes(tmp_path):
    from analyze_bth_exploratory_fusion import verified_source_paths,sha
    file=tmp_path/"中文数据.csv";file.write_bytes(b"real fixture bytes")
    digest=sha(file)
    original={"input_hashes":{str(file):digest}}
    frozen={"input_hashes":{"damaged_path_label":digest}}
    result=verified_source_paths(frozen,original)
    assert result[0]["verified_path"]==str(file) and result[0]["path_label_recovered"]
    file.write_bytes(b"changed")
    with pytest.raises(ValueError,match="changed"):
        verified_source_paths(frozen,original)


def test_subsample_coverage_does_not_include_other_dates_exclusions():
    from analyze_bth_exploratory_fusion import selected_exclusions
    rows=pd.DataFrame(dict(time=["2019-08-01","2019-08-02"],station_id=["a","b"]))
    filtered=selected_exclusions(rows,[pd.Timestamp("2019-08-02")])
    assert filtered.station_id.tolist()==["b"]


def test_grid_annotation_keeps_values_and_exposes_window_and_unevaluated_time(tmp_path):
    import xarray as xr
    from analyze_bth_exploratory_fusion import annotate_grid
    grid=xr.Dataset(dict(pm25_estimate=(("time","lat","lon"),np.array([[[3.]],[[np.nan]]])),
        valid_domain=(("time","lat","lon"),np.array([[[1]],[[0]]],dtype="uint8"))),
        coords=dict(time=pd.date_range("2019-08-01 00:30",periods=2,freq="h"),lat=[38.],lon=[116.]))
    output=annotate_grid(grid,True)
    np.testing.assert_array_equal(output.pm25_rh35_diagnostic,grid.pm25_estimate)
    assert output.time_evaluated.values.tolist()==[1,0]
    assert output.time_bounds.values[0,0]==np.datetime64("2019-08-01 00:00")
    assert output.attrs["crs_evidence"].startswith("USER_WORKING")
    output.to_netcdf(tmp_path/"grid.nc",engine="h5netcdf")
    with xr.open_dataset(tmp_path/"grid.nc",engine="h5netcdf") as reopened:xr.testing.assert_allclose(output,reopened.load())
