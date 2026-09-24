"""Curve editor widget for Curves adjustment (port of CurvesCanvas.swift)."""
from __future__ import annotations

from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from ..core.model import CurvesSettings


class CurveEditor(QWidget):
    def __init__(self, settings: CurvesSettings, parent=None):
        super().__init__(parent)
        self.setFixedSize(240, 240)
        self.points = list(settings.rgb_points) or [(0.0, 0.0), (1.0, 1.0)]

    def mousePressEvent(self, ev):
        p = self._to_norm(ev.position())
        self.points.append(p)
        self.points.sort()
        self.update()

    def _to_norm(self, pos: QPointF):
        w, h = self.width(), self.height()
        x = min(1.0, max(0.0, pos.x() / w))
        y = min(1.0, max(0.0, 1.0 - pos.y() / h))
        return (round(x, 3), round(y, 3))

    def settings(self) -> CurvesSettings:
        pts = [p for p in self.points if not (p == (0, 0) and len(self.points) > 2
                                              and p == self.points[0]) ]
        interior = [p for p in self.points if p not in ((0.0, 0.0), (1.0, 1.0))]
        return CurvesSettings(rgb_points=self.points if interior else [])

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 30, 32))
        p.setPen(QPen(QColor(90, 90, 95), 1, Qt.PenStyle.DashLine))
        w, h = self.width(), self.height()
        p.drawLine(0, h, w, 0)  # identity diagonal
        p.setPen(QPen(QColor(255, 255, 255), 2))
        import numpy as np
        xs = [pt[0] for pt in self.points]
        ys = [pt[1] for pt in self.points]
        path_pts = []
        for t in range(0, 101):
            x = t / 100.0
            y = float(np.interp(x, xs, ys))
            path_pts.append(QPointF(x * w, (1 - y) * h))
        for a, b in zip(path_pts, path_pts[1:]):
            p.drawLine(a, b)
        p.setBrush(QColor(80, 170, 255))
        p.setPen(Qt.PenStyle.NoPen)
        for x, y in self.points:
            p.drawEllipse(QPointF(x * w, (1 - y) * h), 4, 4)
        p.end()
