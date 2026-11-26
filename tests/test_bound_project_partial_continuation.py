"""UI admission is not scientific acceptance; public backend remains required."""
from test_desktop_simple_flow import window
from test_desktop_continuation_gate import setup_recipe
import ecodata.desktop.__main__ as desktop


def test_bound_project_can_reach_public_execution_despite_unrelated_bad_proposal(window,monkeypatch):
    monkeypatch.setattr(desktop,'assert_current',lambda scan:None)
    row=setup_recipe(window)
    window.decision_model='camx'
    window.scan={'native_model_probes':{'camx':{'status':'project'}}}
    window.candidates={'C1':object(),'C2':object()}
    window.update_task_checks([dict(candidate='C2',action='fusion',status='条件不足',block_code='unit_unknown',reason='unrelated unresolved pair')])
    calls=[]
    monkeypatch.setattr(window,'execute_now',lambda:calls.append(window.selected_recommendations))
    window.continue_recommendations()
    assert calls==[[row]]
    assert window.original_background_ready()
    assert window.result is None  # Admission never invents a processed product.


def test_skipped_advice_is_visible_but_not_counted_as_missing_data(window):
    from ecodata.desktop.issues import checks
    window.present_issues(checks([dict(candidate='C1',status='条件不足',block_code='coordinate_field',
        disposition='SKIPPED_NOT_PROCESSING_VARIABLE',reason='coordinate is not an observation')]))
    assert '0 项待核实' in window.problem_title.text()
    assert '1 项' in window.problem_title.text()
    assert window.current_issues[0]['status']=='已自动跳过'
