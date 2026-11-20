"""Button/state regression only; do not rerun fusion or native simulations."""
import pytest
from PySide6.QtWidgets import QCheckBox
from test_desktop_simple_flow import window
import ecodata.desktop.__main__ as desktop


def setup_recipe(w):
    row=dict(task='C1:fusion',steps=['engineering.port_binding'])
    check=QCheckBox();check.setChecked(True)
    w.recommendation_rows=[row];w.recommendation_checks=[check];w.scan={}
    return row


def test_both_buttons_consume_same_checks_even_when_cached_flag_is_false(window,monkeypatch):
    monkeypatch.setattr(desktop,'assert_current',lambda scan:None)
    row=setup_recipe(window);calls=[]
    monkeypatch.setattr(window,'execute_now',lambda:calls.append((window.engineering_continuation,list(window.selected_recommendations))))
    checks=[dict(candidate='C1',action='fusion',status='待工程核验',next_step='original_workbench',reason='public port checks')]
    window.continue_automatically(dict(failures=[]),checks)
    assert calls==[] and window.tabs.currentIndex()==1
    window.original_recommendation_allowed=False
    window.search_button.setEnabled(True);window.search_button.click()
    assert calls==[(True,[row])]


def test_manual_button_reports_actual_blocker(window,monkeypatch):
    monkeypatch.setattr(desktop,'assert_current',lambda scan:None);setup_recipe(window)
    window.update_task_checks([dict(candidate='C2',action='fusion',status='条件不足',reason='缺少明确的时间区间')])
    errors=[];monkeypatch.setattr(window,'error',lambda exc:errors.append(str(exc)))
    monkeypatch.setattr(window,'execute_now',lambda:pytest.fail('blocked work must not execute'))
    window.search_button.setEnabled(True);window.search_button.click()
    assert len(errors)==1 and 'C2' in errors[0] and '时间区间' in errors[0]


def test_confirmation_uses_user_selection_not_all_recommendations(window,monkeypatch):
    monkeypatch.setattr(desktop,'assert_current',lambda scan:None)
    first=setup_recipe(window)
    second=dict(task='C2:fusion',steps=['engineering.port_binding'])
    other=QCheckBox()
    window.recommendation_rows.append(second);window.recommendation_checks.append(other)
    checks=[dict(candidate=c,action='fusion',status='待工程核验',next_step='original_workbench') for c in ('C1','C2')]
    calls=[]
    monkeypatch.setattr(window,'execute_now',lambda:calls.append(list(window.selected_recommendations)))
    window.continue_automatically(dict(failures=[]),checks)
    assert calls==[] and other.isChecked()
    window.recommendation_checks[0].setChecked(False)
    window.search_button.setEnabled(True);window.search_button.click()
    assert calls==[[second]] and first not in window.selected_recommendations


def test_input_reset_invalidates_checks_and_empty_selection_cannot_bypass(window,monkeypatch):
    window.update_task_checks([dict(status='通过')]);window.reset_target()
    assert not window.checked_decisions and not window.original_recommendation_allowed
    monkeypatch.setattr(window,'original_background_ready',lambda:True)
    monkeypatch.setattr(window,'execute_original',lambda:pytest.fail('no bypass'))
    with pytest.raises(ValueError,match='有效核验记录'):window.continue_recommendations()
