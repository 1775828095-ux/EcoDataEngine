"""Bounded orchestration checks; no external API or experimental reranking."""
from copy import deepcopy

from ecodata.desktop.independent_review import start
from ecodata.desktop.recommendations import explanation_text
from ecodata.desktop.release_gate import allowed


def test_supplementary_candidate_keeps_rejection_without_inheriting_release():
    original = dict(id="R1", steps=["spatial.blend"], task="C1:fusion",
        automatic_fallback_started=True, final_verified_recommendation=True,
        execution_release=dict(status="failed"),
        local_holdout_review=[dict(basis="recipe_numeric_rejected", sampled_scores=[dict(reference="proxy")])],
        local_recipe_options=[dict(id="R1", name="blend", description="registered recipe", steps=["spatial.blend"])])
    before = deepcopy(original)
    candidate = start(original, dict(received=1,total=1,candidate="R1",votes={"R1":1}))
    assert original == before
    assert candidate["previous_recommendation_review"]["execution_release"]["status"] == "failed"
    assert candidate["previous_recommendation_review"]["local_holdout_review"] == original["local_holdout_review"]
    assert candidate["independent_required"] and not allowed(candidate)
    assert candidate["automatic_fallback_started"]
    assert not candidate.get("final_verified_recommendation")
    candidate["previous_recommendation_review"]["local_holdout_review"].clear()
    assert original == before


def test_incomplete_supplementary_response_is_not_a_recommendation():
    assert start(dict(local_recipe_options=[]),dict(received=0,total=1,candidate=None)) is None


def test_proxy_trial_and_final_confirmation_are_not_conflated():
    row = dict(local_holdout_review=[dict(acceptance_sampling=dict(population=1,sample_size=1,
        allowed_failures=0,inspected=1,failed=1),sampled_scores=[dict(
            reference="REDUCED_SCALE_PRODUCT_RECONSTRUCTION_PROXY",
            candidate=dict(RMSE=23.69),baseline=dict(RMSE=9.89))])])
    text = explanation_text(row)
    assert "产品重建代理初筛" in text and "不是独立实测精度检验" in text
    assert "不代替变异后冻结方案的独立分块复核" in text


def test_single_option_is_not_presented_as_algorithm_optimization():
    assert "只有一项可推荐配方" in explanation_text(dict(local_recipe_options=[dict(id="R1")]))
    assert "只有一项可推荐配方" not in explanation_text(dict(local_recipe_options=[dict(id="R1"),dict(id="R2")]))


def test_exposure_invalidation_preserves_measured_failure(monkeypatch):
    from ecodata.desktop import independent_review
    candidate = dict(task="C1:fusion",id="R1",steps=["spatial.blend"],
        local_holdout_review=[dict(basis="recipe_numeric_rejected")],
        execution_release=dict(status="failed"),final_verified_recommendation=True)
    monkeypatch.setattr(independent_review,"fields",lambda *args:[])
    monkeypatch.setattr(independent_review,"assess",lambda *args:dict(status="pending"))
    independent_review.invalidate({},candidate,{})
    assert candidate["previous_recommendation_review"]["execution_release"]["status"] == "failed"
    assert candidate["local_holdout_review"] == [] and not allowed(candidate)
    prior = deepcopy(candidate["previous_recommendation_review"])
    independent_review.invalidate({},candidate,{})
    assert candidate["previous_recommendation_review"] == prior


def grid_pair():
    import numpy as np
    import xarray as xr
    def grid(size,step,identity):
        axis=(np.arange(size)+.5)*step
        return xr.Dataset({"precipitation":(("time","lat","lon"),
            (2+axis[:,None]+axis[None,:])[None],dict(units="mm",spatial_semantics="cell_mean",data_kind="continuous",
                physical_variable="precipitation",temporal_semantics="accumulation"))},
            coords=dict(time=[np.datetime64("2020-01-01")],lat=axis,lon=axis),
            attrs=dict(source_id=identity,crs="EPSG:4326"))
    return [grid(12,1,"fine"),grid(6,2,"coarse")]


def test_recipe_uses_exact_search_protocol_without_primary_reference_bias():
    from ecodata.evaluation.fusion_protocol import fixed_recipe_cohort, prepare
    from types import SimpleNamespace
    import pandas as pd
    import numpy as np
    native=grid_pair()
    protocol,cohort=fixed_recipe_cohort(native,native[1],"precipitation")
    expected,_=prepare(SimpleNamespace(native_fields=native,self_reference=True,template=native[1],
                         request=dict(variable="precipitation")),pd.DatetimeIndex(native[0].time.values),0)
    assert protocol["version"]==expected["version"]
    assert protocol["fixed_support_sha256"]==expected["fixed_support_sha256"]
    assert protocol["reference_source"]=="fine"
    swapped,other=fixed_recipe_cohort(native[::-1],native[1],"precipitation")
    assert swapped["reference_source"]=="fine"
    np.testing.assert_allclose(cohort["reference"],other["reference"])
    np.testing.assert_allclose(cohort["inputs"][0],other["inputs"][1])
    assert not protocol["accuracy_established"]


def test_unknown_cell_support_cannot_silently_become_reconstruction_truth():
    from ecodata.evaluation.fusion_protocol import fixed_recipe_cohort
    native=grid_pair()
    native[0].precipitation.attrs.pop("spatial_semantics")
    protocol,cohort=fixed_recipe_cohort(native,native[1],"precipitation")
    assert cohort is None and protocol["missing_evidence"]
    assert "spatial_semantics" not in native[0].precipitation.attrs


def test_actual_recipe_trial_receives_native_inputs_and_reports_shared_protocol():
    from ecodata.desktop.grid_preparation import remap, trial_recipe
    from ecodata.desktop.recipe_trials import BLEND, FILL
    native=grid_pair()
    aligned=[remap(ds,native[1],"precipitation","conservative") for ds in native]
    _,_,metrics,failure=trial_recipe(aligned,1,[FILL,BLEND],native=native)
    assert metrics["evaluation_protocol"]["reference_source"]=="fine"
    assert metrics["count"]==metrics["evaluation_protocol"]["selection_count"]
    assert failure in {None,"recipe_score_below_single_source"}
