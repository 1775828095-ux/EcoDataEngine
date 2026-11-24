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


def test_confirmed_table_recipe_can_continue_without_grid_alignment_record(window, monkeypatch):
    from ecodata.desktop import selected_processing
    row = setup_recipe(window)
    row.update(steps=["temporal.linear", "fusion.series_missing_fallback", "spatial.blend"],
               requires_final_confirmation=True, final_verified_recommendation=True,
               execution_release={"status": "released"},
               local_holdout_review=[{"basis": "joint_recipe_acceptance_passed"}])
    window.selected_recommendations = [row]
    window.gated = {("C1", "fusion"): dict(route="indexed_table_recipe")}
    tasks = []
    monkeypatch.setattr(desktop, "assert_current", lambda scan: None)
    monkeypatch.setattr(selected_processing, "run", lambda scan, selected, *args, **kwargs: tasks.extend(selected))
    monkeypatch.setattr(window, "work", lambda run, finish: run(lambda message: None))
    window.execute_selected()
    assert tasks[0][1]["recipes"] == [row["steps"]]


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


def test_numeric_rejection_sends_full_copy_without_user_confirmation(window, monkeypatch):
    import json
    from ecodata.desktop import fallback_evidence

    field = dict(name="rain", unit="mm")
    window.scan = dict(rows=[dict(name="inputs.zip!/a.csv", format="CSV", fields=[field]),
                            dict(name="inputs.zip!/b.csv", format="CSV", fields=[field])])
    sources = [(0, "rain", "D1.F1"), (1, "rain", "D2.F1")]
    rejected = dict(task="C1:fusion", steps=["spatial.blend"], execution_fields=sources,
                    execution_release=dict(status="failed", blockers=[dict(field="D1.F1", reason="recipe_numeric_rejected")]),
                    local_recipe_options=[dict(id="R1", name="融合", steps=["spatial.blend"], description="融合")])
    pending = dict(task="C2:fusion", steps=["spatial.blend"], execution_fields=[],
                   execution_release=dict(status="pending", blockers=[dict(field="D1.F1", reason="no_disjoint_partition")]))
    window.recommendation_rows = [pending, rejected]
    window.mock.setChecked(False)
    monkeypatch.setattr(window, "effective_api_key", lambda: "synthetic-key")
    monkeypatch.setattr(window, "render_recommendations",
                        lambda reply: setattr(window, "recommendation_rows", reply["recommendations"]))
    monkeypatch.setattr(window, "present_issues", lambda *_: None)
    monkeypatch.setattr(window, "work", lambda run, finish: finish(run(lambda _: None)))
    monkeypatch.setattr(desktop, "assert_current", lambda _: None)
    monkeypatch.setattr(desktop.QMessageBox, "question", lambda *args: pytest.fail("automatic branch prompted"))
    monkeypatch.setattr(desktop.QInputDialog, "getItem", lambda *args: pytest.fail("automatic branch requested a scope"))

    def build(scan, fields, archive, progress, *, include_native):
        assert [index for index, _ in fields] == [0, 1]
        assert include_native is False
        archive.write_bytes(b"synthetic")
        return [dict(source_member=row["name"], bytes=9, records=3, parts=["part"]) for row in scan["rows"]]

    sent = []
    def transmit(archive, entries, digest, **options):
        sent.append(archive)
        assert options["native_approved"] is False
        return dict(received=2, total=2, votes={"R1": 2}, candidate="R1", receipts=[], native_receipts=[])

    monkeypatch.setattr(fallback_evidence, "build", build)
    monkeypatch.setattr(fallback_evidence, "transmit", transmit)
    window.send_next_automatic_fallback()
    window.send_next_automatic_fallback()
    assert len(sent) == 1
    audit = json.loads(sent[0].with_suffix(".approval.json").read_text(encoding="utf8"))
    assert audit["authorization"] == "automatic_numeric_rejection"
    candidate = next(row for row in window.recommendation_rows if row["task"] == "C1:fusion")
    assert candidate["independent_required"] and candidate["execution_release"]["status"] == "pending"


def test_desktop_hides_fusion_recipe_until_final_confirmation(window, monkeypatch):
    from PySide6.QtWidgets import QLabel
    from ecodata.desktop.recommendations import display_record

    monkeypatch.setattr(window, "original_background_ready", lambda: False)
    row = dict(task="C1:fusion", id="R1", name="private-plan", description="private-steps",
               reason="private-reason", steps=["spatial.blend"], requires_final_confirmation=True,
               execution_release={"status": "released"}, local_recipe_options=["private-alternatives"])
    window.render_recommendations(dict(mode="DeepSeek API", recommendations=[row]))
    card = window.recommend_cards.itemAt(0).widget()
    assert "融合方案待核验" in [label.text() for label in card.findChildren(QLabel)]
    assert not any("private-" in label.text() for label in card.findChildren(QLabel))
    assert not window.recommendation_checks[0].isEnabled()
    assert display_record(row) == {"task": "C1:fusion", "status": "released"}

    window.show_decision_feedback(dict(phase=4, notice="正在验收", rows=["private-reply"]))
    assert window.recommend_json.toPlainText() == "正在验收"
    row.update(final_verified_recommendation=True, final_recommendation_origin="deepseek", reason="最终验收理由")
    window.render_recommendations(dict(mode="DeepSeek API", recommendations=[row]))
    assert window.recommendation_checks[0].isEnabled()
    assert "最终验收理由" in window.recommend_json.toPlainText()
    assert "private-alternatives" not in window.recommend_json.toPlainText()
