from threading import Event

from PySide6.QtCore import QAbstractAnimation
from PySide6.QtTest import QTest

from ecodata.desktop import service
from ecodata.desktop.empty_state import EmptyDataIllustration, KEYFRAMES, pose_at
from ecodata.desktop.theme import WORKSPACE_BACKGROUND
from test_desktop_simple_flow import window


def test_pose_waits_before_opening_then_lifts_turns_and_returns_to_rest():
    start, peak, end = pose_at(0), pose_at(0.57), pose_at(1)
    assert pose_at(0.10) == start
    assert pose_at(0.30).guide == 1 and pose_at(0.30).top_lift == 0
    assert start.top_lift == end.top_lift == 0
    assert start.breath == end.breath == 0
    assert peak.top_lift > peak.middle_lift > 0
    assert peak.middle_shift > 0 and start.middle_shift == end.middle_shift == 0
    assert 0 < pose_at(0.43).turn < peak.turn < end.turn == 1


def test_motion_speed_does_not_jump_between_keyframes():
    step = 0.000001
    for time, _ in KEYFRAMES[1:-1]:
        before, at, after = (pose_at(t) for t in (time - step, time, time + step))
        assert all(abs((b - a) / step - (c - b) / step) < 0.1
                   for a, b, c in zip(before, at, after))


def test_animation_starts_only_when_user_starts_processing(window, monkeypatch):
    window.show()
    illustration = window.empty_state.findChild(EmptyDataIllustration)
    assert illustration.isVisible()
    assert illustration.clock.state() == QAbstractAnimation.State.Stopped
    preview = illustration.grab()
    scale = preview.devicePixelRatio()
    assert preview.toImage().pixelColor(round(133 * scale), round(94 * scale)).name() == WORKSPACE_BACKGROUND

    window.path.setText('fixture.csv')
    window.set_decision_model('swat_plus')
    release = Event()
    monkeypatch.setattr(window, 'scan_now', lambda: window.work(
        lambda _: release.wait(5), lambda _: window.continue_automatically(dict(failures=[]), [])))
    window.start_button.click()
    art = window.processing_illustration
    try:
        assert art.isVisible() and art.clock.state() == QAbstractAnimation.State.Running
        assert art.clock.loopCount() == -1
        art.clock.setCurrentTime(1150)
        assert art.pose.guide > 0 and art.pose.top_lift == 0
        pixmap = art.grab()
        image, ratio = pixmap.toImage(), pixmap.devicePixelRatio()
        # The front-right dotted edge falls within the top face at this pose.
        assert any(image.pixelColor(round(x * ratio), round(y * ratio)).red() < 160
                   for x in range(145, 153) for y in range(79, 88))
    finally:
        release.set()
        for _ in range(100):
            if window.thread is None:
                break
            QTest.qWait(20)
    assert window.thread is None and art.clock.state() == QAbstractAnimation.State.Stopped


def test_results_page_shows_shared_art_until_the_job_returns(window, monkeypatch):
    window.show()
    window.tabs.setCurrentIndex(2)
    cards = window.result_cards
    assert cards.illustration.isVisible()
    assert cards.empty_title.text() == '等待处理结果'
    assert cards.illustration.clock.state() == QAbstractAnimation.State.Stopped

    release = Event()
    result = dict(source_formats=['NetCDF'], target_format='NetCDF', files=[])

    def generate(*args, **kwargs):
        release.wait(5)
        return result

    window.scan = {'rows': []}
    window.decision_model = 'swat_plus'
    monkeypatch.setattr('ecodata.desktop.__main__.assert_current', lambda _: None)
    monkeypatch.setattr(window, 'effective_api_key', lambda: '')
    monkeypatch.setattr(service, 'run_original_workbench', generate)
    window.execute_original()
    try:
        assert cards.illustration.isVisible()
        assert cards.illustration.clock.state() == QAbstractAnimation.State.Running
        assert '正在生成原版工程输出包' in cards.status.text()
        assert not window.export_button.isEnabled()
    finally:
        release.set()
        for _ in range(100):
            if window.thread is None:
                break
            QTest.qWait(20)
    assert window.result is result
    assert cards.pages.currentWidget() is cards.results_page
    assert not cards.illustration.isVisible()
    assert cards.illustration.clock.state() == QAbstractAnimation.State.Stopped
    assert window.export_button.isEnabled()
