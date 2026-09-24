"""Layer panel — Python port of UI/LayerList.swift + LayerRowViews.swift
(LayerThumbnailView, LayerGroupDisclosureView, LayerOpacitySlider).

Top-most layer first, folder disclosure triangles with indentation, eye
toggles, live thumbnails (rendered from the layer's own raster), opacity
slider and blend-mode combo for the selected layer.
"""
from __future__ import annotations

from typing import List

import numpy as np
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QListWidget,
                             QListWidgetItem, QSlider, QVBoxLayout, QWidget)

from ..core.model import BLEND_MODE_GROUPS, ImageLayer, LayerBlendMode, hierarchy_entries


def _qpix_from_rgba(rgba: np.ndarray, size: int = 48) -> QPixmap:
    h, w = rgba.shape[:2]
    scale = min(size / max(w, 1), size / max(h, 1), 1.0)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    from PIL import Image
    img = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA")
    img = img.resize((nw, nh), Image.LANCZOS)
    arr = np.ascontiguousarray(np.asarray(img))
    qimg = QImage(arr.data, nw, nh, 4 * nw, QImage.Format.Format_RGBA8888).copy()
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.drawImage((size - nw) // 2, (size - nh) // 2, qimg)
    p.end()
    return pix


class LayerRow(QWidget):
    """One row in the stack: thumbnail | name (+ group caret)."""

    def __init__(self, layer: ImageLayer, depth: int, visible_inherited: bool, owner):
        super().__init__()
        self.layer = layer
        self.owner = owner
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4 + depth * 14, 2, 4, 2)
        # Visibility toggle
        self.eye = QLabel("👁" if layer.visible else "—")
        self.eye.setCursor(Qt.CursorShape.PointingHandCursor)
        self.eye.mousePressEvent = lambda ev: self._toggle()
        lay.addWidget(self.eye)
        # Thumbnail
        self.thumb = QLabel()
        self.thumb.setFixedSize(48, 48)
        self.refresh_thumb()
        lay.addWidget(self.thumb)
        # Name
        prefix = "▸ " if layer.is_group else ""
        text = f"{prefix}{layer.name}"
        if layer.adjustment is not None:
            text = f"◐ {layer.name}"
        elif layer.mask is not None:
            text = f"◻ {layer.name}"
        lbl = QLabel(text)
        lbl.setStyleSheet("color: %s;" % ("white" if visible_inherited else "#888"))
        lay.addWidget(lbl, 1)
        if layer.mask_source_id:
            clip = QLabel("↘")
            clip.setToolTip("Clipped to layer below")
            lay.addWidget(clip)

    def _toggle(self):
        self.owner.session.toggle_visible(self.layer)
        self.owner.refresh()

    def refresh_thumb(self):
        raster = self.layer.raster()
        if raster is None:
            self.thumb.setText("·")
            self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return
        self.thumb.setPixmap(_qpix_from_rgba(raster))


class LayersPanel(QWidget):
    selection_changed = pyqtSignal()

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        title = QLabel("Layers")
        title.setStyleSheet("font-weight: bold; color: white;")
        v.addWidget(title)
        self.list = QListWidget()
        self.list.setStyleSheet(
            "QListWidget { background:#26262a; border:1px solid #3a3a3e; }"
            "QListWidget::item:selected { background:#3d6fb0; }")
        self.list.currentRowChanged.connect(self._row_changed)
        v.addWidget(self.list, 1)
        # Opacity + blend controls (port of LayerOpacitySlider / LayerBlendMenu)
        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("Blend"))
        self.blend = QComboBox()
        last_group = None
        for gi, grp in enumerate(BLEND_MODE_GROUPS):
            for m in grp:
                self.blend.addItem(m.value, userData=m)
        self.blend.currentIndexChanged.connect(self._blend_changed)
        ctl.addWidget(self.blend, 1)
        v.addLayout(ctl)
        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Opacity"))
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.valueChanged.connect(self._opacity_changed)
        op_row.addWidget(self.opacity, 1)
        self.op_label = QLabel("100%")
        self.op_label.setFixedWidth(38)
        op_row.addWidget(self.op_label)
        v.addLayout(op_row)
        btns = QHBoxLayout()
        for text, slot in (("＋", self._add_layer), ("－", self._delete_layer),
                           ("⧉", self._dup_layer), ("⇧", lambda: self._move(1)),
                           ("⇩", lambda: self._move(-1)), ("ƒ", self._group)):
            b = QPushButton_small(text)
            b.clicked.connect(slot)
            btns.addWidget(b)
        v.addLayout(btns)
        self._rows: List[LayerRow] = []
        self._suspend = False

    # ------------------------------------------------------------------
    def refresh(self):
        self._suspend = True
        self.list.clear()
        self._rows = []
        if self.session.document is not None:
            entries = hierarchy_entries(self.session.document.layers, top_first=True,
                                        collapsed=self.session.collapsed_groups)
            for lyr, depth, vis in entries:
                item = QListWidgetItem()
                row = LayerRow(lyr, depth, vis, self)
                item.setSizeHint(QSize(0, 54))
                self.list.addItem(item)
                self.list.setItemWidget(item, row)
                self._rows.append(row)
                if lyr.id in self.session.selected_layer_ids:
                    self.list.setCurrentItem(item)
        # Sync blend/opacity widgets to active layer
        lyr = self.session.active_layer
        if lyr and not lyr.is_group:
            idx = self.blend.findData(lyr.blend_mode)
            self.blend.setCurrentIndex(max(0, idx))
            self.opacity.setValue(int(round(lyr.opacity * 100)))
        self._suspend = False

    def _current_layer(self):
        row = self.list.currentRow()
        if 0 <= row < len(self._rows):
            return self._rows[row].layer
        return None

    def _row_changed(self, row):
        if self._suspend:
            return
        lyr = self._current_layer()
        if lyr is not None:
            self.session.selected_layer_ids = [lyr.id]
            self.refresh()
        self.selection_changed.emit()

    def _blend_changed(self, _):
        if self._suspend:
            return
        lyr = self.session.active_layer
        mode = self.blend.currentData()
        if lyr and mode:
            self.session.set_blend_mode(lyr, mode)

    def _opacity_changed(self, v):
        if self._suspend:
            return
        self.op_label.setText(f"{v}%")
        lyr = self.session.active_layer
        if lyr:
            self.session.set_opacity(lyr, v / 100.0)

    def _add_layer(self):
        self.session.add_blank_layer()
        self.refresh()

    def _delete_layer(self):
        self.session.remove_selected()
        self.refresh()

    def _dup_layer(self):
        self.session.duplicate_selected()
        self.refresh()

    def _move(self, delta):
        lyr = self.session.active_layer
        if lyr:
            self.session.move_layer(lyr, delta)
            self.refresh()

    def _group(self):
        self.session.make_group_from_selected()
        self.refresh()


def QPushButton_small(text):
    from PyQt6.QtWidgets import QPushButton
    b = QPushButton(text)
    b.setFixedWidth(28)
    return b
