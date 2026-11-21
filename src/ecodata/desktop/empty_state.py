"""Transparent isometric artwork for the native desktop empty state."""

from bisect import bisect_right
from math import cos, radians, sin, sqrt
from typing import NamedTuple

from PySide6.QtCore import QPointF, QRectF, Qt, QVariantAnimation
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from .theme import WORKSPACE_BACKGROUND


class Pose(NamedTuple):
    top_lift: float
    middle_lift: float
    middle_shift: float
    turn: float
    breath: float
    guide: float
    impact: float


KEYFRAMES = (
    (0.00, Pose(0, 0, 0, 0, 0, 0, 0)),
    (0.13, Pose(0, 0, 0, 0, 0, 0, 0)),
    (0.25, Pose(0, 0, 0, 0.08, 0.12, 1, 0)),
    (0.38, Pose(0, 0, 0, 0.19, 0.25, 1, 0)),
    (0.48, Pose(17, 7, 0, 0.34, 0.70, 1, 0)),
    (0.57, Pose(27, 8, 22, 0.60, 1, 1, 0)),
    (0.66, Pose(18, 6, 0, 0.83, 0.57, 1, 0)),
    (0.76, Pose(16, 5, 0, 0.90, 0.43, 0.96, 0)),
    (0.83, Pose(0, 0, 0, 1, 0.15, 0, 1)),
    (0.91, Pose(0, 0, 0, 1, 0, 0, 0)),
    (1.00, Pose(0, 0, 0, 1, 0, 0, 0)),
)
POSITIONS = tuple(position for position, _ in KEYFRAMES)
GRAPHITE = '#383838'
ISO_COS = sqrt(3) / 2


def shared_tangents():
    """Use one velocity on either side of a keyframe, without overshoot."""
    width = [b - a for a, b in zip(POSITIONS, POSITIONS[1:])]
    slopes = [Pose(*((b - a) / span for a, b in zip(start, end)))
              for (_, start), (_, end), span in zip(KEYFRAMES, KEYFRAMES[1:], width)]
    stationary = Pose(0, 0, 0, 0, 0, 0, 0)
    tangents = [stationary]
    for i, (left, right) in enumerate(zip(slopes, slopes[1:])):
        left_weight, right_weight = 2 * width[i + 1] + width[i], width[i + 1] + 2 * width[i]
        tangents.append(Pose(*((left_weight + right_weight) / (left_weight / a + right_weight / b)
                               if a * b > 0 else 0 for a, b in zip(left, right))))
    return [*tangents, stationary]


TANGENTS = shared_tangents()


def pose_at(progress):
    if progress <= 0:
        return KEYFRAMES[0][1]
    if progress >= 1:
        return KEYFRAMES[-1][1]
    end = bisect_right(POSITIONS, progress)
    left, start = KEYFRAMES[end - 1]
    right, finish = KEYFRAMES[end]
    span = right - left
    t = (progress - left) / span
    eased = t * t * (3 - 2 * t)
    return Pose(*(a + (b - a) * eased + span * t * (1 - t) * ((1 - t) * m - t * n)
                  for a, b, m, n in zip(start, finish, TANGENTS[end - 1], TANGENTS[end])))


def project(x, y, z):
    """Project a level data layer onto two axes angled at ±30 degrees."""
    return QPointF((x - y) * ISO_COS, (x + y) / 2 - z)


def plate_corners(z, size=40, shift=0):
    return [project(x + shift, y, z) for x, y in (
        (-size, -size), (size, -size), (size, size), (-size, size),
    )]


def outline(width=2.15, color=GRAPHITE):
    pen = QPen(QColor(color), width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


class EmptyDataIllustration(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(230, 202)
        self.setObjectName('emptyDataIllustration')
        self.setAccessibleName('灰色等距数据层示意')
        self.pose = pose_at(0)
        self.running = False
        self.clock = QVariantAnimation(self)
        self.clock.setStartValue(0.0)
        self.clock.setEndValue(1.0)
        self.clock.setDuration(3450)
        self.clock.setLoopCount(-1)
        self.clock.valueChanged.connect(self.update_pose)

    def update_pose(self, progress):
        self.pose = pose_at(progress)
        self.update()

    def set_running(self, running):
        if self.running == running:
            return
        self.running = running
        if running and self.isVisible():
            self.clock.start()
        elif not running:
            self.clock.stop()
            self.update_pose(0)

    def showEvent(self, event):
        super().showEvent(event)
        if self.running:
            self.clock.start()

    def hideEvent(self, event):
        self.clock.stop()
        super().hideEvent(event)

    def paintEvent(self, event):
        pose = self.pose
        with QPainter(self) as painter:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.translate(self.width() / 2 - 5, 127)
            painter.scale(1.22, 1.22)
            self.draw_plate(painter, 8)
            self.draw_plate(painter, 16 + pose.middle_lift, shift=pose.middle_shift)
            top = 24 + pose.top_lift
            self.draw_plate(painter, top)
            self.draw_cross(painter, top, pose.turn)
            self.draw_capsule(painter, -49, 45, 30, pose.breath)
            self.draw_capsule(painter, 53, 41, -30, pose.breath)
            if pose.impact:
                self.draw_impact(painter, pose.impact)
            if pose.guide:
                # The guide is a diagram overlay. Its front edges must remain
                # visible even when they project over a solid plate face.
                color = QColor(GRAPHITE)
                color.setAlphaF(pose.guide * 0.88)
                pen = outline(1.55, color)
                pen.setDashPattern([2, 3])
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPolygon(QPolygonF(plate_corners(24 + 36 * pose.guide, size=41)))

    @staticmethod
    def draw_plate(painter, z, shift=0):
        top = plate_corners(z, shift=shift)
        bottom = plate_corners(z - 8, shift=shift)
        painter.setPen(outline())
        # Use the page color to mask the rear edges of overlapping plates.
        painter.setBrush(QColor(WORKSPACE_BACKGROUND))
        for start, end in ((1, 2), (2, 3)):
            painter.drawPolygon(QPolygonF([top[start], top[end], bottom[end], bottom[start]]))
        painter.drawPolygon(QPolygonF(top))

    @staticmethod
    def draw_cross(painter, z, turn):
        angle = radians(90 * turn)
        painter.setPen(outline(2.5))
        for x, y in ((9 * cos(angle), 9 * sin(angle)), (-9 * sin(angle), 9 * cos(angle))):
            painter.drawLine(project(-x, -y, z), project(x, y, z))

    @staticmethod
    def draw_capsule(painter, x, y, angle, breath):
        painter.save()
        painter.translate(x, y + breath * 0.8)
        painter.rotate(angle)
        width, height = 24 + breath * 1.5, 6.3 + breath * 0.4
        capsule = QRectF(-width / 2, -height / 2, width, height)
        painter.setPen(outline(1.85))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(capsule, height / 2, height / 2)
        painter.restore()

    @staticmethod
    def draw_impact(painter, amount):
        color = QColor(GRAPHITE)
        color.setAlphaF(amount * 0.72)
        painter.setPen(outline(1.7, color))
        for direction in (-1, 1):
            painter.drawLine(QPointF(direction * 76, 17), QPointF(direction * (76 + 7 * amount), 17 - 3 * amount))
            painter.drawLine(QPointF(direction * 76, 25), QPointF(direction * (76 + 6 * amount), 25 + 3 * amount))
