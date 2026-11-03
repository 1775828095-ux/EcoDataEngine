from pathlib import Path
import sys,json,inspect
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"tools"))
import numpy as np,pandas as pd,pytest
from ecodata.algorithms.temporal_search import (TemporalEvaluator,ReferenceEvaluation,CoverageEvaluation,
    submit_temporal_search,validate_external)
from ecodata.algorithms.numerical_search import run_numerical_search,key_for
from ecodata.interface.temporal_search_page import process_temporal_request,parse_upload
from test_beijing_mixed_gap import fixture
from test_decision_boundaries import graph

REQUEST=dict(operation="temporal_missing_recovery",require_complete_coverage=True,prefer_observation_support=True,time_access="both_sides")
BUDGET=dict(objective_limit=4,proposal_limit=20,callback_limit=100,wall_seconds=10)


def evaluator(tmp_path):
    original,external=fixture()
    refs=original[external["batch"].value.cases[0].missing]
    return TemporalEvaluator(external,ReferenceEvaluation(refs),tmp_path,REQUEST)


def test_clients_share_service_without_local_policy(monkeypatch,tmp_path):
    import ecodata.algorithms.temporal_search as service
    from run_beijing_mixed_gap import search_request
    calls=[]
    monkeypatch.setattr(service,"submit_temporal_search",lambda *a,**k:calls.append((a,k)) or {"shared":True})
    # Research alias must resolve the same public entry; no copied loop/weights.
    import run_beijing_mixed_gap as runner
    monkeypatch.setattr(runner,"submit_temporal_search",service.submit_temporal_search)
    for client in (process_temporal_request,search_request):
        assert client({},REQUEST,None,BUDGET,tmp_path)=={"shared":True}
        assert "semantic_proposal_weights" not in inspect.getsource(client)
    assert len(calls)==2 and calls[0]==calls[1]
    original,external=fixture()
    frame=pd.DataFrame(dict(truth=original[external["batch"].value.cases[0].missing],group_id=0))
    monkeypatch.setattr(runner,"OUT",tmp_path)
    monkeypatch.setattr(runner,"load",lambda *args:(external,frame,np.ones(1)))
    assert runner.search("C",11,{},BUDGET)=={"shared":True}
    assert len(calls)==3
    import ecodata.algorithms.numerical_search as common
    assert "from tools" not in inspect.getsource(common)


@pytest.mark.parametrize("kind",["missing","test","unknown_metric","unsupported_task"])
def test_evaluation_conditions_rejected_before_search(kind,tmp_path):
    _,external=fixture();request=dict(REQUEST);ev=None
    if kind=="test":
        with pytest.raises(ValueError,match="not final test"):ReferenceEvaluation([1.],role="test")
        return
    if kind=="unknown_metric":ev=lambda p:{"objective":0.}
    if kind=="unsupported_task":request["operation"]="spatial_reconstruction"
    with pytest.raises(ValueError):submit_temporal_search(external,request,ev,BUDGET,tmp_path)
    assert not (tmp_path/"summary.json").exists()
    assert (tmp_path/"failure.json").exists()


def test_reference_never_enters_decisions_or_prediction_inputs(tmp_path):
    a=evaluator(tmp_path/"a");b=evaluator(tmp_path/"b")
    b.evaluate=ReferenceEvaluation(np.ones(len(b.external["query"].value))*9999)
    x=a.decision().as_dict();y=b.decision().as_dict();x.pop("seconds");y.pop("seconds")
    assert x==y and set(a.external)=={"batch","history","query"}
    aa=a.execute(graph("nearest"),"a",persist=False);bb=b.execute(graph("nearest"),"b",persist=False)
    assert aa["prediction_fingerprint"]==bb["prediction_fingerprint"] and aa["objective"]!=bb["objective"]


def test_missing_or_mismatched_history_excluded_not_silently_reused(tmp_path):
    e=evaluator(tmp_path)
    assert "baseline.temporal_calendar_knn" in e.excluded
    assert not any(k[0]=="baseline.temporal_calendar_knn" for k in e.factory.adapters)
    from ecodata.algorithms.mixed_gap_dag import SealedArtifact
    ext=dict(e.external);ext["history"]=SealedArtifact("fitted_history",{"different_station":{}},ext["history"].metadata)
    other=TemporalEvaluator(ext,e.evaluate,tmp_path/"b",REQUEST)
    assert set(other.excluded)=={"baseline.temporal_calendar_knn","baseline.temporal_hour_median"}
    assert not other.decision().matches["baseline.temporal_hour_median"]["eligible"]


def test_json_history_adaptation_uses_existing_predictor_without_mutating_input(tmp_path):
    from ecodata.algorithms.mixed_gap_dag import SealedArtifact
    e=evaluator(tmp_path);external=dict(e.external)
    original={"s":dict(medians=[50.]*24,training_end="2014-12-31 23:00")}
    external["history"]=SealedArtifact("fitted_history",original,external["history"].metadata)
    adapted=TemporalEvaluator(external,e.evaluate,tmp_path/"json",REQUEST)
    result=adapted.execute(graph("hour_median"),"json-state",persist=False)
    assert result["status"]=="PASS" and isinstance(original["s"]["medians"],list)
    assert isinstance(adapted.external["history"].value["s"]["medians"],np.ndarray)


def test_csv_schema_reference_isolation_and_storage_units():
    data=b"station,time,pm25\ns,2015-01-01 00:00,2\ns,2015-01-01 01:00,\ns,2015-01-01 02:00,4\n"
    ref=b"station,time,truth\ns,2015-01-01 01:00,3\n"
    ext,ev=parse_upload(data,ref,{},variable="pm25",unit="ug m-3")
    validate_external(ext)
    assert np.isnan(ext["batch"].value.cases[0].values[1]) and ev.truth.tolist()==[3.]
    with pytest.raises(ValueError,match="移除"):
        parse_upload(data,ref.replace(b"01:00",b"00:00"),{},variable="pm25",unit="ug m-3")
    with pytest.raises(ValueError,match="单位"):
        parse_upload(data,ref,{},variable="pm25",unit="UNKNOWN")
    for altered in [data.replace(b"01:00",b"00:30"),data.replace(b"02:00",b"03:00")]:
        bad,_=parse_upload(altered,None,{},variable="pm25",unit="ug m-3")
        with pytest.raises(ValueError,match="regular 1-hour"):validate_external(bad)


def test_unknown_and_missing_capability_refused_by_common_execution(tmp_path):
    from dataclasses import replace
    e=evaluator(tmp_path)
    d=e.decision();d.requirements=replace(d.requirements,required=tuple(d.requirements.required)+("required_marker",))
    row=e.execute(graph("nearest"),"missing",persist=False)
    assert row["failure_stage"]=="task_requirements" and not row["numerical_execution_attempted"]
    assert row["task_acceptance"]["acceptance_status"]=="REJECT"
    e=evaluator(tmp_path);e.decision().matches.pop("baseline.temporal_nearest")
    row=e.execute(graph("nearest"),"unknown",persist=False)
    assert row["task_acceptance"]["acceptance_status"]=="UNKNOWN" and not row["numerical_execution_attempted"]


def test_budget_cache_incomplete_acceptance_and_execution_failure_separate(monkeypatch,tmp_path):
    import ecodata.algorithms.numerical_search as common
    from types import SimpleNamespace
    e=evaluator(tmp_path/"executions");execute=e.execute
    def fail_selected(g,k):
        if k==key_for(graph("hour_median"),e.factory):
            original=e.pipeline.run_dag
            e.pipeline.run_dag=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError("controlled I/O failure"))
            try:return execute(g,k)
            finally:e.pipeline.run_dag=original
        return execute(g,k)
    e.execute=fail_selected
    class Probe:
        def __init__(self,obj,*args):self.obj=obj;self.stop_optimization=SimpleNamespace(add_condition=lambda *a:None)
        def optimise(self,callback):
            for g in [graph("nearest"),graph("nearest"),graph("linear"),e.factory.initial(),graph("hour_median"),graph("nearest")]:self.obj(g)
    monkeypatch.setattr(common,"EvoGraphOptimizer",Probe)
    r=run_numerical_search("A",11,e,BUDGET,tmp_path)
    c=r["counts"]
    assert c["graph_cache_hits"]==1 and c["real_executions"]==2 and c["quality_failures"]==1
    assert c["task_acceptance_failures"]==1 and c["incomplete_feedback"]==1 and c["deferred_callbacks"]==1
    assert r["stop_reason"]=="budget"
    with pytest.raises(ValueError,match="already contains"):run_numerical_search("A",11,e,BUDGET,tmp_path)


def test_coverage_mode_does_not_claim_accuracy():
    metric=CoverageEvaluation();result=metric(np.array([1.,2.]))
    assert result["objective"]==-1 and not result["accuracy_assessed"] and "RMSE" not in result


def test_proposal_ceiling_does_not_discard_pending_numeric_batch(monkeypatch,tmp_path):
    import ecodata.algorithms.numerical_search as common
    from types import SimpleNamespace
    e=evaluator(tmp_path/"executions")
    class Probe:
        def __init__(self,obj,initial,requirements,generation,parameters):
            self.obj=obj;self.initial=initial;self.requirements=requirements;self.generation=generation;self.parameters=parameters
            self.stop_optimization=SimpleNamespace(add_condition=lambda *a:None)
        def optimise(self,callback):
            self.obj(self.initial[0])
            self.parameters.mutation_types[0](self.initial[0],requirements=self.requirements,graph_gen_params=self.generation,parameters=self.parameters)
            self.obj(graph("nearest")) # pending graph must still use its numeric budget
    monkeypatch.setattr(common,"EvoGraphOptimizer",Probe)
    r=run_numerical_search("A",11,e,dict(BUDGET,proposal_limit=1),tmp_path)
    assert r["counts"]["proposals"]==1 and r["counts"]["real_executions"]==1 and r["best"]


def test_page_renders_saved_real_result_without_starting_search(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import ecodata.algorithms.temporal_search as service
    monkeypatch.setattr(service,"submit_temporal_search",lambda *a,**kw:pytest.fail("Rerender must not start a search"))
    folder=ROOT/"output/platform_search_service_20261021/bounded_final/page"
    result=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    app=AppTest.from_string("from ecodata.interface.temporal_search_page import render_temporal_search_page\nrender_temporal_search_page()")
    app.session_state["ts_result"]=(result,str(folder));app.run(timeout=30)
    assert not app.exception and len(app.success)==1 and len(app.json)>=2
    app.run(timeout=30)
    assert not app.exception and len(app.get("download_button"))==1
    from ecodata.interface.temporal_search_page import download_search_result
    import io,zipfile
    with zipfile.ZipFile(io.BytesIO(download_search_result(folder))) as archive:
        assert {"processed.csv","summary.json","selected_graph.json","request.json","decision.json"}<=set(archive.namelist())
        assert any(name.endswith("prediction.npz") for name in archive.namelist())
        product=pd.read_csv(archive.open("processed.csv"))
        assert np.isfinite(product.processed_value).all()


def test_existing_app_navigation_reaches_public_search_page():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/"src/ecodata/interface/app.py"))
    app.session_state["main_entry"]="时间恢复图搜索"
    app.run(timeout=30)
    assert not app.exception
    assert any("GOLEM真实数值搜索" in item.value for item in app.subheader)
    assert len(app.get("file_uploader"))==3
