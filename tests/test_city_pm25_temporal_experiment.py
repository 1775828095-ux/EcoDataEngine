"""Synthetic fixtures check leakage/contracts, not environmental accuracy."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"));sys.path.append(str(ROOT/".venv-golem/Lib/site-packages"))
from city_pm25_experiment import masks,series_from,metrics,execute_graph
from ecodata.algorithms.temporal_reconstruction_dag import reconstruct,temporal_setup
from ecodata.algorithms.pipeline import UnifiedAlgorithmPipeline
from ecodata.algorithms.dag import DAGError
from golem.core.optimisers.graph import OptGraph
from golem.core.optimisers.genetic.operators import base_mutations


def y():return pd.Series(np.arange(720,dtype=float),index=pd.date_range("2022-01-01 03:00",periods=720,freq="h"))


def test_masks_are_deterministic_and_never_score_natural_missing():
    series=y();series.iloc[10:20]=np.nan
    a=list(masks(series));b=list(masks(series))
    assert len(a)==9
    for (task,seed,idx),(other,other_seed,again) in zip(a,b):
        assert (task,seed)==(other,other_seed)
        np.testing.assert_array_equal(idx,again)
        assert series.iloc[idx].notna().all()
        if task=="random20":assert len(idx)==int(series.notna().sum()*.2)


def test_parity_is_timestamp_axis_not_valid_row_number():
    series=y();series.iloc[1]=np.nan
    parts={task:idx for task,_,idx in masks(series)}
    assert np.all(parts["alternate_phase0"]%2==0)
    assert np.all(parts["alternate_phase1"]%2==1)
    assert 1 not in parts["alternate_phase1"]


def test_contiguous_blocks_have_real_truth_and_declared_length():
    for task,_,idx in masks(y()):
        if task.startswith("block"):
            assert len(idx)==int(task[5:])
            assert np.all(np.diff(idx)==1)


def test_entities_remain_separate_and_conflicts_not_averaged():
    t=pd.date_range("2022-01-01",periods=72,freq="h")
    frames=[]
    for entity,value in [("a",1.),("b",100.)]:
        frames.append(pd.DataFrame(dict(entity_id=entity,time=t,value=value,source_row=np.arange(72),reason="VALID")))
    data=pd.concat(frames,ignore_index=True)
    parts=list(series_from(data));assert len(parts)==2
    assert parts[0][1].mean()==1. and parts[1][1].mean()==100.
    data=pd.concat([data,data.iloc[[0]]],ignore_index=True)
    assert np.isnan(list(series_from(data))[0][1].iloc[0])


@pytest.mark.parametrize("method",["linear","nearest","ffill","hour_median"])
def test_hidden_truth_cannot_change_prediction(method):
    series=y();idx=np.array([10,11,12]);masked=series.copy();masked.iloc[idx]=np.nan
    first=reconstruct(masked,method)
    series.iloc[idx]=1e9
    np.testing.assert_array_equal(first,reconstruct(masked,method))


def test_ffill_is_past_only_and_nearest_boundary_extension_explicit():
    series=y();series.iloc[0]=np.nan;series.iloc[5:10]=np.nan
    assert np.isnan(reconstruct(series,"ffill").iloc[0])
    assert reconstruct(series,"ffill").iloc[5]==4.
    assert reconstruct(series,"nearest").iloc[0]==1.
    assert np.isnan(reconstruct(series,"linear").iloc[0])


def test_task_registry_has_only_four_minimal_baselines_and_eight_graphs():
    registry,factory=temporal_setup()
    assert len(registry.specs())==4
    assert len(factory.recipes("validated_field"))==8
    initial=factory.initial();assert factory.inspect(initial)["gap"]==1
    result=base_mutations.add_separate_parent_node(initial,factory)
    assert factory.inspect(result,complete=True)["gap"]==0


def test_real_dag_executes_and_rejects_unmasked_targets():
    registry,factory=temporal_setup();pipeline=UnifiedAlgorithmPipeline(registry)
    series=y();masked=series.copy();masked.iloc[10:20]=np.nan
    case=dict(file_id="fixture",entity_id="a",y=series,masked=masked)
    for _,recipe in factory.recipes("validated_field"):
        graph=OptGraph([factory.instantiate(recipe)])
        values,trace=execute_graph(graph,case,np.arange(10,20),pipeline)
        assert np.isfinite(values).all() and trace
    values,_=execute_graph(graph,case,np.arange(10,12),pipeline)
    assert len(values)==2  # Target ports participate in cache identity.
    case["masked"]=series
    with pytest.raises(DAGError,match="leaked"):
        execute_graph(graph,case,np.arange(10,20),pipeline)


def test_metrics_keep_failed_denominator_and_no_fake_perfect_r2():
    r=metrics([1,2,3],[1,np.nan,4],2.)
    assert r["expected"]==3 and r["predicted"]==2 and r["coverage"]==2/3
    assert metrics([1,1],[1,1],2.)["R2"] is None


def test_validation_executor_cannot_access_final_period_even_for_fitting(tmp_path,monkeypatch):
    import city_pm25_experiment as experiment
    from run_city_temporal_search import TemporalEvaluator
    series=y();masked=series.copy();positions=np.array([20,40,600]);masked.iloc[positions]=np.nan
    cutoff=series.index[479];seen=[]
    def capture(graph,case,idx,pipeline):
        seen.append(case["masked"].index.max())
        assert case["masked"].index.max()<=cutoff
        return np.zeros(len(idx)),[]
    monkeypatch.setattr(experiment,"execute_graph",capture)
    case=dict(file_id="fixture",entity_id="a",case_id="sealed",y=series,masked=masked,positions=positions)
    evaluator=TemporalEvaluator([case],dict(validation_end=str(cutoff)),tmp_path)
    result=evaluator.execute(evaluator.factory.initial(),"fixture")
    assert result["expected"]==2 and seen==[cutoff]


def test_combined_mask_manifest_preserves_columns_when_optional_field_absent(tmp_path,monkeypatch):
    import finish_city_pm25_experiment as finish
    monkeypatch.setattr(finish,"OUT",tmp_path)
    paths=[]
    for name in ["pilot","pilot_final_test"]:
        folder=tmp_path/name;folder.mkdir()
        row=dict(file_id="fixture",entity_id="a",task="random20",seed=11,case_id="sample",
            time="2022-01-01",split="validation" if name=="pilot" else "test",source_row=2)
        if name=="pilot":row["peak_threshold"]=10.
        pd.DataFrame([row]).to_csv(folder/"masks.csv",index=False)
        paths.append(folder/"metrics.csv")
    finish.combine_masks(paths)
    frame=pd.read_csv(tmp_path/"mask_manifest.csv")
    assert frame.stage.tolist()==["pilot_validation","pilot_test"]
    assert frame.peak_threshold.iloc[0]==10. and pd.isna(frame.peak_threshold.iloc[1])
