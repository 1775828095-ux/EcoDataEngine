"""Presentation and real tiny numerical callback checks, not environment accuracy."""
from test_desktop_simple_flow import window
from test_spatial_relationship_fusion import fixture,BUDGET
from ecodata.algorithms import platform_search as api
from ecodata.algorithms.spatial_fusion_task import SpatialTask,SpatialSearch


def test_result_page_receives_actual_progress_and_keeps_it_after_render(window):
    window.tabs.setCurrentIndex(2)
    window.show_progress('搜索停止：已评价2图，暂无独立精度结论')
    assert '已评价2图' in window.result_cards.progress_log.toPlainText()
    window.result_cards.render([],dict(model=None,native_ready=False,requirements=[],notes=[]))
    assert '已评价2图' in window.result_cards.progress_log.toPlainText()
    window.result_cards.reset()
    assert not window.result_cards.progress_log.toPlainText()


def test_fresh_numeric_candidate_emits_real_metrics_and_cached_call_does_not(tmp_path,window):
    items,q,_=fixture();parsed,audit=api.parse_inputs(items)
    window.tabs.setCurrentIndex(2);events=[]
    def receive(message):
        events.append(message);window.show_progress(message)
    engine=SpatialSearch(SpatialTask.build(parsed,q,audit),tmp_path,BUDGET,11,progress=receive)
    engine.execute(engine.leaves[0])
    assert len(events)==1 and engine.rows[0]['status']=='PASS'
    assert f"RMSE {engine.rows[0]['metrics']['RMSE']:.5g}" in events[0]
    assert 'spatial.source' in window.result_cards.progress_log.toPlainText()
    engine.execute(engine.leaves[0]);assert len(events)==1
