import pytest
from PySide6.QtWidgets import QLabel
from test_desktop_simple_flow import window
from ecodata.desktop.release_gate import allowed, selectable


def rows():
    return [dict(task='C1:fusion',id='failed',name='未通过的混合',description='失败方案',reason='代理误差比基线大',
        steps=['spatial.blend'],execution_release=dict(status='failed',blockers=[dict(reason='recipe_numeric_rejected')])),
        dict(task='C2:fusion',id='pending',name='待核实',description='缺少单位',reason='单位尚未确认',
        steps=['spatial.blend'],execution_release=dict(status='pending',blockers=[dict(reason='selected_table_unit_missing')])),
        dict(task='C1:fusion',id='legal',name='合法候选搜索',description='真实比较其他合法算法，结果仍需复核。',reason='已确认存在合法执行路径；未确认增强收益。',
        steps=['spatial.source'],search_admission=dict(protocol='rejected-recipe-search-v1',status='SEARCH_ONLY',route='independent_grid_search'),
        execution_release=dict(status='search_only',blockers=[]))]


def test_only_legal_choices_on_main_and_failed_rows_retained_in_details(window,monkeypatch):
    monkeypatch.setattr(window,'original_background_ready',lambda:False)
    original=rows();window.tabs.setCurrentIndex(1)
    window.render_recommendations(dict(mode='test',recommendations=original))
    assert window.recommend_cards.count()==1 and window.recommendation_detail_cards.count()==2
    main=window.recommend_cards.itemAt(0).widget()
    labels=' '.join(x.text() for x in main.findChildren(QLabel))
    assert '自动比较合法处理方案' in labels and '保留单源' not in labels
    assert '待放行' not in labels and '缺少单位' not in labels
    assert {r['id'] for r in window.recommendation_rows}=={'failed','pending','legal'}
    checks={r['id']:c for r,c in zip(window.recommendation_rows,window.recommendation_checks)}
    assert checks['legal'].isEnabled() and checks['legal'].isChecked()
    assert not checks['failed'].isEnabled() and not checks['pending'].isEnabled()
    assert window.search_button.isEnabled() and window.search_button.text()=='继续处理'
    assert not allowed(original[-1]) and selectable(original[-1])


def test_busy_state_blocks_click_and_direct_submission_without_changing_gate(window,monkeypatch):
    monkeypatch.setattr(window,'original_background_ready',lambda:False)
    window.recommendation_review_busy=True
    window.render_recommendations(dict(mode='test',recommendations=rows()))
    assert not window.search_button.isEnabled() and window.recommendation_cards_view.isHidden()
    assert '自动完成' in window.recommendation_status.text()
    with pytest.raises(ValueError,match='尚未完成'):
        window.continue_recommendations()
    window.recommendation_review_busy=False;window.refresh_recommendation_action()
    assert window.search_button.isEnabled() and not window.recommendation_cards_view.isHidden()


def test_blocked_state_names_actual_reason_and_has_no_disabled_main_choices(window,monkeypatch):
    monkeypatch.setattr(window,'original_background_ready',lambda:False)
    window.render_recommendations(dict(mode='test',recommendations=[rows()[1]]))
    assert window.recommend_cards.count()==0 and window.recommendation_detail_cards.count()==1
    assert not window.search_button.isEnabled()
    assert '单位' in window.recommendation_status.text() and '暂不能继续' in window.recommendation_status.text()
    window.open_recommendation_details()
    assert window.drawer.pages.currentWidget() is window.recommendation_details_view


def test_one_click_uses_only_visible_admitted_plan(window,monkeypatch):
    monkeypatch.setattr(window,'original_background_ready',lambda:False)
    window.render_recommendations(dict(mode='test',recommendations=rows()))
    import ecodata.desktop.__main__ as desktop
    monkeypatch.setattr(desktop,'assert_current',lambda scan:None)
    window.scan={};window.gated={('C1','fusion'):dict(route='independent_grid_search')}
    calls=[];monkeypatch.setattr(window,'execute_now',lambda:calls.append([r['id'] for r in window.selected_recommendations]))
    window.search_button.click()
    assert calls==[['legal']]


def test_independent_engineering_remains_a_distinct_action(window,monkeypatch):
    monkeypatch.setattr(window,'original_background_ready',lambda:True)
    window.render_recommendations(dict(mode='test',recommendations=[rows()[1]]))
    assert window.search_button.isEnabled() and window.search_button.text()=='继续工程核验'
    assert '不代表融合已通过' in window.recommendation_status.text()


def test_details_cannot_make_failed_recipe_executable(window,monkeypatch):
    monkeypatch.setattr(window,'original_background_ready',lambda:False)
    window.render_recommendations(dict(mode='test',recommendations=rows()))
    checks={r['id']:c for r,c in zip(window.recommendation_rows,window.recommendation_checks)}
    checks['failed'].setChecked(True)
    import ecodata.desktop.__main__ as desktop
    monkeypatch.setattr(desktop,'assert_current',lambda scan:None)
    monkeypatch.setattr(window,'execute_now',lambda:pytest.fail('Failed recipe bypassed release gate'))
    with pytest.raises(ValueError,match='未放行'):
        window.continue_recommendations()


def test_issue_without_a_recipe_still_has_readable_details(window,monkeypatch):
    monkeypatch.setattr(window,'original_background_ready',lambda:False)
    window.render_recommendations(dict(mode='test',recommendations=[]))
    window.tabs.setCurrentIndex(1)
    window.present_issues([dict(stage='输入检查',target='降水',status='待处理',problem='缺少统计时间区间',action='提供时间边界')])
    assert '缺少统计时间区间' in window.recommendation_status.text()
    assert not window.recommendation_details_button.isHidden()
    window.recommendation_details_button.click()
    assert window.drawer.pages.currentWidget() is window.issue_details


def test_prior_failed_numbers_remain_available_after_exposure(window,monkeypatch):
    from PySide6.QtWidgets import QPlainTextEdit
    monkeypatch.setattr(window,'original_background_ready',lambda:False)
    pending=rows()[1]
    pending['previous_recommendation_review']=dict(reason='先前数值检查失败',local_holdout_review=[dict(
        acceptance_sampling=dict(population=1,sample_size=1,allowed_failures=0,inspected=1,failed=1),
        sampled_scores=[dict(reference='REDUCED_SCALE_PRODUCT_RECONSTRUCTION_PROXY',
            candidate=dict(RMSE=15.42156),baseline=dict(RMSE=7.14486))])])
    window.render_recommendations(dict(mode='test',recommendations=[pending]))
    card=window.recommendation_detail_cards.itemAt(0).widget()
    text=' '.join(editor.toPlainText() for editor in card.findChildren(QPlainTextEdit))
    assert '此前方案检查' in text and '15.42' in text and '7.145' in text
    assert not window.search_button.isEnabled()
