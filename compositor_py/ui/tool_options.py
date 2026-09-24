"""Tool options bar — Python port of UI/ToolOptionsBar.swift.

Contextual strip under the toolbar: per-tool settings (brush size/hardness/
opacity/erase toggle, wand tolerance, shape kind & fill, text size, gradient
presets), colour swatches with swap, mask-editing toggle, zoom field and
tool buttons grid is in MainWindow's left rail.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel,
                             QPushButton, QSlider, QWidget)

from ..core.model import NavigationTool, ShapeKind


class ColorSwatch(QPushButton):
    color_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(28, 28)
        self.color_rgba = (0.0, 0.0, 0.0, 1.0)
        self.clicked.connect(self.color_clicked)

    def set_color(self, rgba):
        self.color_rgba = rgba
        c = QColor(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
        self.setStyleSheet(
            f"background:{c.name()}; border:1px solid #666; border-radius:4px;")


class ToolOptionsBar(QWidget):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._buttons = {}
        self._build()
        self.refresh()

    def _build(self):
        from PyQt6.QtWidgets import QStackedWidget
        self.stack = QStackedWidget(self)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(6, 2, 6, 2)
        outer.addWidget(self.stack)
        s = self.session

        # Shared brush controls page -----------------------------------
        self.brush_page = QWidget()
        row = QHBoxLayout(self.brush_page)
        row.addWidget(QLabel("Size"))
        self.brush_size = QSlider(Qt.Orientation.Horizontal)
        self.brush_size.setRange(1, 400)
        self.brush_size.valueChanged.connect(
            lambda v: setattr(s.defaults.brush, "size", float(v)))
        row.addWidget(self.brush_size)
        self.brush_size_lbl = QLabel()
        self.brush_size_lbl.setFixedWidth(40)
        row.addWidget(self.brush_size_lbl)
        row.addWidget(QLabel("Hardness"))
        self.brush_hard = QSlider(Qt.Orientation.Horizontal)
        self.brush_hard.setRange(0, 100)
        self.brush_hard.valueChanged.connect(
            lambda v: setattr(s.defaults.brush, "hardness", v / 100.0))
        row.addWidget(self.brush_hard)
        row.addWidget(QLabel("Opacity"))
        self.brush_op = QSlider(Qt.Orientation.Horizontal)
        self.brush_op.setRange(1, 100)
        self.brush_op.valueChanged.connect(
            lambda v: setattr(s.defaults.brush, "opacity", v / 100.0))
        row.addWidget(self.brush_op)
        self.erase_btn = QPushButton("Eraser")
        self.erase_btn.setCheckable(True)
        self.erase_btn.toggled.connect(
            lambda on: setattr(s.defaults.brush, "erase_mode", on))
        row.addWidget(self.erase_btn)
        self.mask_btn = QPushButton("Mask")
        self.mask_btn.setCheckable(True)
        self.mask_btn.setToolTip("Paint on the layer mask instead of pixels")
        self.mask_btn.toggled.connect(lambda on: setattr(s, "painting_on_mask", on))
        row.addWidget(self.mask_btn)
        self.stack.addWidget(self.brush_page)

        # Marquee / lasso / wand page ----------------------------------
        self.sel_page = QWidget()
        r2 = QHBoxLayout(self.sel_page)
        r2.addWidget(QLabel("Selection: drag on canvas · Shift adds · Alt subtracts"))
        r2.addStretch(1)
        self.stack.addWidget(self.sel_page)

        self.wand_page = QWidget()
        r3 = QHBoxLayout(self.wand_page)
        r3.addWidget(QLabel("Tolerance"))
        self.wand_tol = QSlider(Qt.Orientation.Horizontal)
        self.wand_tol.setRange(1, 90)
        self.wand_tol.valueChanged.connect(
            lambda v: setattr(s.defaults, "wand_tolerance", v / 100.0))
        r3.addWidget(self.wand_tol)
        r3.addWidget(QLabel("Click canvas to select · Shift-click adds"))
        r3.addStretch(1)
        self.stack.addWidget(self.wand_page)

        # Move page ------------------------------------------------------
        self.move_page = QWidget()
        r4 = QHBoxLayout(self.move_page)
        r4.addWidget(QLabel("Move: drag layers · corner handles scale · top handle rotates"))
        r4.addStretch(1)
        self.stack.addWidget(self.move_page)

        # Crop page --------------------------------------------------------
        self.crop_page = QWidget()
        r5 = QHBoxLayout(self.crop_page)
        r5.addWidget(QLabel("Crop: drag a rectangle; release applies"))
        r5.addStretch(1)
        self.stack.addWidget(self.crop_page)

        # Gradient page -----------------------------------------------------
        self.grad_page = QWidget()
        r6 = QHBoxLayout(self.grad_page)
        r6.addWidget(QLabel("Gradient preset"))
        self.grad_choice = QComboBox()
        self._presets = {
            "Black → White": [(0.0, (0, 0, 0, 255)), (1.0, (255, 255, 255, 255))],
            "White → Black": [(0.0, (255, 255, 255, 255)), (1.0, (0, 0, 0, 255))],
            "Foreground → Transparent": [
                (0.0, (0, 0, 0, 255)), (1.0, (0, 0, 0, 0))],
        }
        for name in self._presets:
            self.grad_choice.addItem(name)
        self.grad_choice.currentTextChanged.connect(self._grad_preset)
        r6.addWidget(self.grad_choice)
        r6.addWidget(QLabel("Drag across the active layer"))
        r6.addStretch(1)
        self.stack.addWidget(self.grad_page)

        # Shape page ---------------------------------------------------------
        self.shape_page = QWidget()
        r7 = QHBoxLayout(self.shape_page)
        r7.addWidget(QLabel("Shape"))
        self.shape_kind = QComboBox()
        for k in ShapeKind:
            self.shape_kind.addItem(k.value, userData=k)
        self.shape_kind.currentIndexChanged.connect(self._shape_kind)
        r7.addWidget(self.shape_kind)
        self.shape_fill = ColorSwatch()
        self.shape_fill.color_clicked.connect(self._pick_shape_fill)
        r7.addWidget(self.shape_fill)
        r7.addWidget(QLabel("Drag on canvas to place"))
        r7.addStretch(1)
        self.stack.addWidget(self.shape_page)

        # Type page ------------------------------------------------------------
        self.type_page = QWidget()
        r8 = QHBoxLayout(self.type_page)
        r8.addWidget(QLabel("Font size"))
        self.text_size = QDoubleSpinBox()
        self.text_size.setRange(6, 500)
        self.text_size.setValue(s.defaults.text_size)
        self.text_size.valueChanged.connect(
            lambda v: setattr(s.defaults, "text_size", float(v)))
        r8.addWidget(self.text_size)
        r8.addWidget(QLabel("Click canvas to place text"))
        r8.addStretch(1)
        self.stack.addWidget(self.type_page)

        # Generic page (hand/zoom/eyedropper/idle/heal/clone/blur) -------------
        self.info_page = QWidget()
        r9 = QHBoxLayout(self.info_page)
        self.info_lbl = QLabel("")
        r9.addWidget(self.info_lbl)
        r9.addStretch(1)
        self.stack.addWidget(self.info_page)

    # ------------------------------------------------------------------
    def _grad_preset(self, name):
        stops = list(self._presets[name])
        if name == "Foreground → Transparent":
            c = self.session.fg_color
            col = tuple(int(round(x * 255)) for x in c[:3])
            stops = [(0.0, col + (255,)), (1.0, col + (0,))]
        self.session.defaults.gradient_stops = stops

    def _shape_kind(self, _):
        k = self.shape_kind.currentData()
        if k:
            self.session.defaults.shape_kind = k.value

    def _pick_shape_fill(self):
        from PyQt6.QtWidgets import QColorDialog
        c = QColorDialog.getColor(initial=QColor(0, 0, 0), parent=self)
        if c.isValid():
            rgba = (c.redF(), c.greenF(), c.blueF(), c.alphaF())
            self.session.defaults.shape_fill = rgba
            self.shape_fill.set_color(rgba)

    def refresh(self):
        s = self.session
        t = s.tool
        b = s.defaults.brush
        pages = {
            NavigationTool.BRUSH: self.brush_page,
            NavigationTool.SPOT_HEALING: self.info_page,
            NavigationTool.CLONE_STAMP: self.info_page,
            NavigationTool.BLUR: self.brush_page,
            NavigationTool.MARQUEE: self.sel_page,
            NavigationTool.LASSO: self.sel_page,
            NavigationTool.WAND: self.wand_page,
            NavigationTool.CROP: self.crop_page,
            NavigationTool.GRADIENT: self.grad_page,
            NavigationTool.SHAPE: self.shape_page,
            NavigationTool.TYPE: self.type_page,
            NavigationTool.MOVE: self.move_page,
        }
        info = {
            NavigationTool.HAND: "Hand: drag to pan · wheel zooms",
            NavigationTool.ZOOM: "Zoom: click in / Shift-click out · wheel zooms",
            NavigationTool.EYEDROPPER: "Eyedropper: click to pick foreground colour",
            NavigationTool.IDLE: "No tool: use keyboard shortcuts (V/M/L/W/B…)",
            NavigationTool.SPOT_HEALING: "Spot Healing: paint over blemishes",
            NavigationTool.CLONE_STAMP: "Clone Stamp: Alt-click source, then paint",
        }
        page = pages.get(t, self.info_page)
        if page is self.info_page and t.value in ("hand", "zoom", "eyedropper",
                                                  "idle", "spotHealing", "cloneStamp"):
            self.info_lbl.setText(info.get(t, t.label))
        self.stack.setCurrentWidget(page)
        # Sync widgets
        self.brush_size.blockSignals(True); self.brush_size.setValue(int(b.size))
        self.brush_size.blockSignals(False)
        self.brush_size_lbl.setText(f"{int(b.size)}px")
        self.brush_hard.blockSignals(True); self.brush_hard.setValue(int(b.hardness * 100))
        self.brush_hard.blockSignals(False)
        self.brush_op.blockSignals(True); self.brush_op.setValue(int(b.opacity * 100))
        self.brush_op.blockSignals(False)
        self.erase_btn.setChecked(b.erase_mode)
        self.mask_btn.setChecked(s.painting_on_mask)
        self.mask_btn.setEnabled(s.active_layer is not None and
                                 getattr(s.active_layer, "mask", None) is not None)
        self.wand_tol.blockSignals(True)
        self.wand_tol.setValue(int(s.defaults.wand_tolerance * 100)); self.wand_tol.blockSignals(False)
        self.shape_fill.set_color(s.defaults.shape_fill)
