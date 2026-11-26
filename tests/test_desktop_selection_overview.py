from PySide6.QtWidgets import QLabel

from ecodata.desktop import output_review, service
from test_desktop_simple_flow import window


def test_alignment_fallback_and_measured_selection_have_distinct_labels(window):
    alignment = dict(selection_basis="GEOMETRY_BASELINE", candidates=[], selected_source_grid=2)
    result = dict(mode="selected_operations", source_formats=["NetCDF"], target_format="NetCDF",
                  original_report=[dict(task="C1:align_space", backend="independent_alignment",
                                        report=[dict(alignment=alignment)])])
    check = output_review.describe_output(result, None)
    def render():
        items = output_review.cards(result, check)
        summaries = output_review.selection_summaries(items, [], [], result=result)
        window.result_cards.render(items, check, summaries=summaries)
        return summaries, [widget.text() for widget in window.result_cards.findChildren(QLabel)]
    summaries, labels = render()
    assert summaries[0]["performed"] and not summaries[0].get("metrics")
    assert "保守重网格 · 已执行" in labels and not any("本次最优" in text for text in labels)
    alignment["candidates"] = [
        dict(source_grid=i, latitude_step=.25*i, longitude_step=.25*i, rmse=loss, mae=loss, n=20)
        for i, loss in [(1, 2.), (2, 1.)]]
    summaries, labels = render()
    assert summaries[0]["metrics"]["RMSE"] == 1.
    assert any("第 1 名" in text and "本次最优" in text for text in labels)


def test_overview_uses_recorded_winner_and_keeps_failures_unscored(window):
    check = dict(source_formats=['NetCDF'], target_format='NetCDF', conversion_reason='',
                 model=None, native_ready=False, requirements=[], notes=[])
    search = dict(evidence={'reference_basis': 'PRIMARY_GRID_RECONSTRUCTION_PROXY'},
                  rankings=[
                      dict(rank=1, selected=True, family='SINGLE_SOURCE', algorithm='spatial.source',
                           metrics=dict(RMSE=1.25, MAE=0.7, count=52)),
                      dict(rank=2, selected=False, family='MULTISOURCE', algorithm='spatial.blend',
                           metrics=dict(RMSE=1.7, MAE=1, count=52)),
                  ])
    result = dict(grid_searches={'C1:fusion': search})
    items = output_review.cards(result, check)
    summaries = output_review.selection_summaries(
        items, [{'task': 'C2:fusion', 'reason': "'time_bounds'"}],
        [{'task': 'C2:fusion', 'steps': ['spatial.regrid.conservative', 'spatial.blend']}],
    )
    winner, failure = summaries
    assert winner['title'] == '保留来源场'
    assert winner['metrics'] == {'RMSE': 1.25, 'MAE': 0.7, 'count': 52}
    assert 'RMSE' in winner['reason'] and '代理评价' in winner['reference']
    assert 'metrics' not in failure and '未选出算法' in failure['title']
    assert '加权融合' in failure['proposed'] and 'time_bounds' in failure['reason']

    window.result_cards.render(items, check, summaries=summaries)
    overview = window.result_cards.cards
    assert overview.itemAt(0).widget().text() == '最终方案与评分'
    assert overview.itemAt(1).widget().findChild(QLabel, 'selectionMetrics').text() == (
        'RMSE 1.25   ·   MAE 0.7   ·   样本 52'
    )
    assert overview.itemAt(2).widget().findChild(QLabel, 'selectionMetrics') is None


def test_conversion_without_candidates_displays_no_score():
    summaries = output_review.selection_summaries([], [], [])
    assert summaries[0]['title'] == '未产生算法评分'
    assert 'metrics' not in summaries[0]


def test_engineering_handoff_reports_completed_adaptation_without_fabricating_scores():
    result = dict(manifest=dict(status='ENGINEERING_INSPECTED', preparation=dict(source_admission=dict(
        adaptation=[dict(status='ADAPTED_AWAITING_ADMISSION', product='daily.nc', source_id='rain',
                         time_records=7, units='mm', operations=[dict(operation='rate_interval_integral')]),
                    dict(status='BLOCKED_PRODUCT_CONTRACT', source_id='unverified')],
        weather=dict(blocked_search=dict(reason='缺少可验证的端口坐标')),
    ))))
    rows = output_review.selection_summaries([], [], [], result=result)
    assert [row['title'] for row in rows] == ['降水率按时间区间积分', '工程盘点']
    assert '7 条' in rows[0]['reason'] and '端口坐标' in rows[1]['reason']
    assert all(row['performed'] and 'metrics' not in row for row in rows)


def test_nonspatial_search_shows_deployed_steps_and_objective():
    from ecodata.desktop.result_summary import nonspatial_winner

    recorded = nonspatial_winner(
        dict(service='PlatformSearch/PortraitMutation', product='product.nc',
             best=dict(key='out', metrics=dict(objective=0.3)),
             mutation_new_executions=2, final_validation_best_delivered=True),
        dict(raw=dict(operation='source'), out=dict(operation='temporal.semantic', parents=['raw'])),
    )
    rows = output_review.selection_summaries([], [], [], result=dict(other_searches={'test': recorded}))
    assert rows[0]['metrics'] == {'objective': 0.3}
    assert '变异执行' in rows[0]['search_note']


def test_deepseek_benefits_use_only_public_winner_evidence_and_update_its_card(window):
    result=dict(mode='independent_grid_search',source_formats=['NetCDF'],target_format='NetCDF',
                grid_searches={'private/file.nc': dict(
                    evidence=dict(reference_basis='REDUCED_SCALE_PRODUCT_RECONSTRUCTION'),
                    rankings=[dict(rank=1,selected=True,family='SINGLE_SOURCE',algorithm='spatial.source',
                                   metrics=dict(RMSE=3.13,MAE=2.4,count=26)),
                              dict(rank=2,selected=False,family='MULTISOURCE',algorithm='spatial.blend',
                                   metrics=dict(RMSE=3.276,MAE=2.6,count=26))])})
    check=output_review.describe_output(result,None)
    items=output_review.cards(result,check)
    summaries=output_review.selection_summaries(items,[],[],result=result)
    request=output_review.review_request(check,items,summaries=summaries)
    plan,=request['selected_plans']
    assert plan['steps']==['保守重网格','保留来源场']
    assert plan['scores']=={'RMSE':3.13,'MAE':2.4,'count':26}
    assert plan['alternative_RMSE']==3.276 and plan['grid_proxy']
    assert 'private/file.nc' not in service.dump(plan)

    window.result_cards.render(items,check,summaries=summaries)
    window.result_cards.update_check(check,pending=[plan['id']])
    note=window.result_cards.plan_notes[plan['id']]
    assert '正在生成' in note.text() and not note.isHidden()

    reply,error=service.review_model_output(result,None,mock=True,summaries=summaries)
    assert error is None and plan['id'] in reply['plan_explanations']
    window.result_cards.update_check(check,reply,simulated=True)
    assert '模拟处理收益' in note.text()

    validated=output_review.validate_review(dict(verdict='ready',conversion_benefit='输出用途',
        missing_data=[],missing_information=[],plan_explanations=[
            dict(id='P999',benefit='无关解释',limits='无关说明'),
            dict(id=plan['id'],benefit='按覆盖面积对齐共同网格，方便逐格核对。',limits='代理评分不能代表独立实测精度。'),
        ]),plan_ids=[plan['id']])
    assert list(validated['plan_explanations'])==[plan['id']]
    window.result_cards.update_check(check,validated)
    assert 'DeepSeek 处理收益' in note.text() and '覆盖面积' in note.text() and '适用边界' in note.text()
