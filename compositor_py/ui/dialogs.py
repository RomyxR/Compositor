"""Dialogs — Python ports of UI/NewDocumentSheet.swift, AdjustmentsPanel.swift,
LayerEffectsSheet.swift and the various inspector sheets (levels/curves/etc.).
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                             QFormLayout, QHBoxLayout, QLabel, QListWidget,
                             QPushButton, QSlider, QSpinBox, QTabWidget,
                             QVBoxLayout, QWidget)

from ..core.model import (
    AdjustmentKind, CanvasDocument, CurvesSettings, HueSaturationSettings,
    LayerEffects,
)


class NewDocumentDialog(QDialog):
    def __init__(self, parent=None, width=1024, height=768):
        super().__init__(parent)
        self.setWindowTitle("New Document")
        form = QFormLayout(self)
        self.w = QSpinBox(); self.w.setRange(1, 30000); self.w.setValue(width)
        self.h = QSpinBox(); self.h.setRange(1, 30000); self.h.setValue(height)
        self.res = QSpinBox(); self.res.setRange(1, 1200); self.res.setValue(72)
        form.addRow("Width (px):", self.w)
        form.addRow("Height (px):", self.h)
        form.addRow("Resolution (ppi):", self.res)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        form.addRow(bb)


class AdjustmentDialog(QDialog):
    """Editor for one adjustment layer; mirrors AdjustmentsPanel's sections."""

    def __init__(self, session, layer, parent=None):
        super().__init__(parent)
        self.session = session
        self.layer = layer
        adj = layer.adjustment
        self.setWindowTitle(f"{adj.kind.value} Settings")
        v = QVBoxLayout(self)
        kind = adj.kind
        if kind == AdjustmentKind.HSV:
            s = adj.hsv
            self.colorize = QCheckBox("Colorize"); self.colorize.setChecked(s.colorize)
            v.addWidget(self.colorize)
            self.hue = _slider_row(v, "Hue", -180, 180, s.hue)
            self.sat = _slider_row(v, "Saturation", -100, 100, s.saturation)
            self.light = _slider_row(v, "Lightness", -100, 100, s.lightness)
        elif kind == AdjustmentKind.LEVELS:
            s = adj.levels
            self.black = _slider_row(v, "Input Black", 0, 100, s.black * 100)
            self.white = _slider_row(v, "Input White", 0, 100, s.white * 100)
            self.gamma = _dspin_row(v, "Gamma", 0.01, 99.0, s.gamma, 0.05)
            self.out_min = _slider_row(v, "Output Min", 0, 100, s.output_min * 100)
            self.out_max = _slider_row(v, "Output Max", 0, 100, s.output_max * 100)
        elif kind == AdjustmentKind.CURVES:
            v.addWidget(QLabel("Click to add control points on the curve."))
            from .curve_widget import CurveEditor
            self.curve = CurveEditor(adj.curves)
            v.addWidget(self.curve)
        elif kind == AdjustmentKind.EXPOSURE:
            s = adj.exposure
            self.exp = _slider_row(v, "Exposure", -1000, 1000, s.exposure * 100)
            self.off = _slider_row(v, "Offset", -50, 50, s.offset * 100)
            self.gam = _slider_row(v, "Gamma", 1, 999, s.gamma * 100)
        elif kind == AdjustmentKind.GRADIENT_MAP:
            v.addWidget(QLabel("Gradient stops (position : colour):"))
            self.stops = list(adj.gradient_map.stops)
            self.listw = QListWidget()
            for pos, col in self.stops:
                self.listw.addItem(f"{pos:.2f} → rgb{col}")
            v.addWidget(self.listw)
            row = QHBoxLayout()
            addb = QPushButton("Add mid stop"); remb = QPushButton("Remove last")
            addb.clicked.connect(self._add_stop); remb.clicked.connect(self._rem_stop)
            row.addWidget(addb); row.addWidget(remb); v.addLayout(row)
        elif kind == AdjustmentKind.GRAIN:
            s = adj.grain
            self.intensity = _slider_row(v, "Intensity", 0, 100, s.intensity * 100)
            self.contrast = _slider_row(v, "Contrast", 0, 100, s.contrast * 100)
        elif kind == AdjustmentKind.BLACK_WHITE:
            s = adj.black_white
            self.reds = _slider_row(v, "Reds", 0, 100, s.reds * 100)
            self.yellows = _slider_row(v, "Yellows", 0, 100, s.yellows * 100)
            self.greens = _slider_row(v, "Greens", 0, 100, s.greens * 100)
            self.cyans = _slider_row(v, "Cyans", 0, 100, s.cyans * 100)
            self.blues = _slider_row(v, "Blues", 0, 100, s.blues * 100)
            self.magentas = _slider_row(v, "Magentas", 0, 100, s.magentas * 100)
        elif kind == AdjustmentKind.COLOR_BALANCE:
            s = adj.color_balance
            v.addWidget(QLabel("Shadows C/M/Y:"))
            self.sh = [_slider_row(v, n, -100, 100, v0 * 100)
                       for n, v0 in zip("CMY", s.shadows_cmy)]
            v.addWidget(QLabel("Midtones C/M/Y:"))
            self.mid = [_slider_row(v, n, -100, 100, v0 * 100)
                        for n, v0 in zip("CMY", s.midtones_cmy)]
            v.addWidget(QLabel("Highlights C/M/Y:"))
            self.hi = [_slider_row(v, n, -100, 100, v0 * 100)
                       for n, v0 in zip("CMY", s.highlights_cmy)]
        else:  # INVERT has no settings
            v.addWidget(QLabel("Invert has no parameters."))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._apply); bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _add_stop(self):
        self.stops.append((0.5, (128, 128, 128)))
        self.stops.sort()
        self._relist()

    def _rem_stop(self):
        if len(self.stops) > 2:
            self.stops.pop()
            self._relist()

    def _relist(self):
        self.listw.clear()
        for pos, col in self.stops:
            self.listw.addItem(f"{pos:.2f} → rgb{col}")

    def _apply(self):
        adj = self.layer.adjustment
        k = adj.kind
        if k == AdjustmentKind.HSV:
            adj.hsv.hue = self.hue.value()
            adj.hsv.saturation = self.sat.value()
            adj.hsv.lightness = self.light.value()
            adj.hsv.colorize = self.colorize.isChecked()
        elif k == AdjustmentKind.LEVELS:
            adj.levels.black = self.black.value() / 100
            adj.levels.white = self.white.value() / 100
            adj.levels.gamma = self.gamma.value()
            adj.levels.output_min = self.out_min.value() / 100
            adj.levels.output_max = self.out_max.value() / 100
        elif k == AdjustmentKind.CURVES:
            adj.curves = self.curve.settings()
        elif k == AdjustmentKind.EXPOSURE:
            adj.exposure.exposure = self.exp.value() / 100
            adj.exposure.offset = self.off.value() / 100
            adj.exposure.gamma = self.gam.value() / 100
        elif k == AdjustmentKind.GRADIENT_MAP:
            adj.gradient_map.stops = list(self.stops)
        elif k == AdjustmentKind.GRAIN:
            adj.grain.intensity = self.intensity.value() / 100
            adj.grain.contrast = self.contrast.value() / 100
        elif k == AdjustmentKind.BLACK_WHITE:
            adj.black_white.reds = self.reds.value() / 100
            adj.black_white.yellows = self.yellows.value() / 100
            adj.black_white.greens = self.greens.value() / 100
            adj.black_white.cyans = self.cyans.value() / 100
            adj.black_white.blues = self.blues.value() / 100
            adj.black_white.magentas = self.magentas.value() / 100
        elif k == AdjustmentKind.COLOR_BALANCE:
            adj.color_balance.shadows_cmy = tuple(w.value() / 100 for w in self.sh)
            adj.color_balance.midtones_cmy = tuple(w.value() / 100 for w in self.mid)
            adj.color_balance.highlights_cmy = tuple(w.value() / 100 for w in self.hi)
        self.session.begin_edit(f"Edit {k.value}")
        self.session.end_edit()
        self.accept()


def _slider_row(parent_layout, label, lo, hi, value) -> QSlider:
    row = QHBoxLayout()
    row.addWidget(QLabel(label))
    sl = QSlider(Qt.Orientation.Horizontal)
    sl.setRange(lo, hi)
    sl.setValue(int(value))
    val = QLabel(str(int(value)))
    val.setFixedWidth(36)
    sl.valueChanged.connect(lambda v: val.setText(str(v)))
    row.addWidget(sl, 1)
    row.addWidget(val)
    parent_layout.addLayout(row)
    return sl


def _dspin_row(parent_layout, label, lo, hi, value, step) -> QDoubleSpinBox:
    row = QHBoxLayout()
    row.addWidget(QLabel(label))
    sp = QDoubleSpinBox()
    sp.setRange(lo, hi); sp.setSingleStep(step); sp.setValue(value)
    row.addWidget(sp, 1)
    parent_layout.addLayout(row)
    return sp


class LayerEffectsDialog(QDialog):
    """Port of LayerEffectsSheet: stroke, drop shadow, color overlay."""

    def __init__(self, session, layer, parent=None):
        super().__init__(parent)
        self.session, self.layer = session, layer
        fx = layer.effects or LayerEffects()
        self.fx = LayerEffects(**{k: getattr(fx, k) for k in fx.__dataclass_fields__})
        self.setWindowTitle("Layer Effects")
        tabs = QTabWidget()
        v = QVBoxLayout(self); v.addWidget(tabs)

        # Stroke tab
        st = QWidget(); h = QVBoxLayout(st)
        self.stroke_on = QCheckBox("Stroke"); self.stroke_on.setChecked(self.fx.stroke_rgba is not None)
        h.addWidget(self.stroke_on)
        self.stroke_w = _slider_row(h, "Width", 1, 50, self.fx.stroke_width)
        tabs.addTab(st, "Stroke")

        # Shadow tab
        sh = QWidget(); hv = QVBoxLayout(sh)
        self.shadow_on = QCheckBox("Drop Shadow")
        self.shadow_on.setChecked(self.fx.drop_shadow_enabled)
        hv.addWidget(self.shadow_on)
        self.sdx = _slider_row(hv, "Offset X", -100, 100, self.fx.drop_shadow_dx)
        self.sdy = _slider_row(hv, "Offset Y", -100, 100, self.fx.drop_shadow_dy)
        self.sblur = _slider_row(hv, "Blur", 0, 100, self.fx.drop_shadow_blur)
        tabs.addTab(sh, "Drop Shadow")

        # Overlay tab
        ov = QWidget(); ho = QVBoxLayout(ov)
        self.overlay_btn = QPushButton("Set Color Overlay…")
        self.overlay_btn.clicked.connect(self._pick_overlay)
        self.overlay_clear = QPushButton("Clear")
        self.overlay_clear.clicked.connect(lambda: setattr(self.fx, "color_overlay", None))
        ho.addWidget(self.overlay_btn); ho.addWidget(self.overlay_clear)
        tabs.addTab(ov, "Color Overlay")

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._apply); bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _pick_overlay(self):
        from PyQt6.QtGui import QColor
        from PyQt6.QtWidgets import QColorDialog
        c = QColorDialog.getColor(parent=self)
        if c.isValid():
            self.fx.color_overlay = (c.redF(), c.greenF(), c.blueF(), c.alphaF())

    def _apply(self):
        fx = self.fx
        fx.drop_shadow_enabled = self.shadow_on.isChecked()
        fx.drop_shadow_dx = self.sdx.value()
        fx.drop_shadow_dy = self.sdy.value()
        fx.drop_shadow_blur = self.sblur.value()
        fx.stroke_width = self.stroke_w.value()
        if self.stroke_on.isChecked() and fx.stroke_rgba is None:
            fx.stroke_rgba = (0.0, 0.0, 0.0, 1.0)
        if not self.stroke_on.isChecked():
            fx.stroke_rgba = None
        self.session.set_effects(self.layer, fx)
        self.accept()
