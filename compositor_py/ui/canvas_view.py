"""Qt canvas viewport — Python port of Rendering/CanvasSurface.swift,
Rendering/CanvasViewRepresentable.swift and the tool state machines from
UI/ToolOptionsBar.swift / UI/TransformControls.swift.

The viewport renders the composited document (with a checkerboard behind
transparency), supports pan/zoom (scroll wheel zooms like the Swift build;
Ctrl+wheel also), rulers, grid, guides, selection marching-ants preview,
the move-tool bounding box with rotate/scale handles, marquee/lasso/crop
drag previews, brush cursor ring, painting, eyedropper, gradient drag,
shape drag, text click-to-place and type-tool editing.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRectF, QTimer, pyqtSignal
from PyQt6.QtGui import (QColor, QImage, QPainter, QPainterPath, QPen, QPixmap,
                         QFont)
from PyQt6.QtWidgets import QWidget

from ..core.model import (
    LayerBlendMode, LayerEffects, LayerSampling, LayerShape, LayerText,
    LayerTransform, NavigationTool, ShapeKind,
)
from ..core import editing
from ..core.compositor import composite_document, checkerboard_preview


def qimage_from_rgba(rgba: np.ndarray) -> QImage:
    arr = np.ascontiguousarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8))
    h, w = arr.shape[:2]
    img = QImage(arr.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
    return img.copy()  # detach from numpy buffer


class CanvasView(QWidget):
    pointer_changed = pyqtSignal(float, float)   # doc coords for status bar
    zoom_changed = pyqtSignal(float)
    finished_edit = pyqtSignal(str)              # history label on gesture end

    MIN_ZOOM, MAX_ZOOM = 0.02, 64.0

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.cursor_override: Optional[str] = None
        # Gesture state
        self._drag_start_doc: Optional[Tuple[float, float]] = None
        self._drag_cur_doc: Optional[Tuple[float, float]] = None
        self._panning = False
        self._pan_last_px: Optional[Tuple[float, float]] = None
        self._painting = False
        self._paint_prev_doc: Optional[Tuple[float, float]] = None
        self._moving_layer = False
        self._move_origins: dict = {}
        self._rotating = False
        self._scaling = None  # handle index
        self._transform_before: Optional[LayerTransform] = None
        self._lasso_pts: List[Tuple[float, float]] = []
        self._pending_label: Optional[str] = None
        self._preview_cache: Optional[np.ndarray] = None
        self._preview_key = None
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._animate_ants)
        self._timer.start()
        self._ant_phase = 0

    # ------------------------------------------------------------------
    # Coordinate mapping (port of CanvasSurface's layout maths)
    # ------------------------------------------------------------------
    def _doc_size(self) -> Optional[Tuple[int, int]]:
        if self.session.document is None:
            return None
        return (self.session.document.width, self.session.document.height)

    def _origin_px(self) -> Tuple[float, float]:
        ds = self._doc_size()
        z = self.session.zoom
        if ds is None:
            return (0, 0)
        cw, ch = ds[0] * z, ds[1] * z
        ox = (self.width() - cw) / 2 + self.session.pan[0]
        oy = (self.height() - ch) / 2 + self.session.pan[1]
        return (ox, oy)

    def widget_to_doc(self, p: QPointF) -> Tuple[float, float]:
        ox, oy = self._origin_px()
        z = self.session.zoom
        return ((p.x() - ox) / z, (p.y() - oy) / z)

    def doc_to_widget(self, x: float, y: float) -> QPointF:
        ox, oy = self._origin_px()
        z = self.session.zoom
        return QPointF(ox + x * z, oy + y * z)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------
    def paintEvent(self, ev):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(37, 36, 38))
        ds = self._doc_size()
        if ds is None:
            painter.setPen(QColor(150, 150, 150))
            f = QFont(); f.setPointSize(14); painter.setFont(f)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No document\nFile ▸ New… or open an image")
            return
        w, h = ds
        z = self.session.zoom
        ox, oy = self._origin_px()
        # Composite (cached per edit generation)
        key = self._cache_key()
        if self._preview_cache is None or self._preview_key != key:
            rgba = composite_document(self.session.document)
            self._preview_cache = checkerboard_preview(rgba, cell=max(2, int(8 * min(z, 4))))
        img = qimage_from_rgba(self._preview_cache)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, z < 1.0)
        painter.drawImage(QRectF(ox, oy, w * z, h * z), img)
        # Canvas border
        painter.setPen(QPen(QColor(90, 90, 95), 1))
        painter.drawRect(QRectF(ox - 0.5, oy - 0.5, w * z + 1, h * z + 1))

        self._draw_grid(painter, ox, oy, w, h, z)
        self._draw_guides(painter, ox, oy, w, h, z)
        self._draw_selection(painter, ox, oy, w, h, z)
        self._draw_drag_preview(painter, ox, oy, z)
        if self.session.tool == NavigationTool.MOVE:
            self._draw_move_handles(painter, ox, oy, z)
        if self.session.tool.is_brush_tool:
            self._draw_brush_cursor(painter)
        if self.session.shows_rulers:
            self._draw_rulers(painter, ox, oy, w, h, z)
        painter.end()

    def _cache_key(self):
        d = self.session.document
        return (id(d), self.session.history.index, self._gesture_revision())

    _gesture_rev = 0

    def _gesture_revision(self):
        return self._gesture_rev

    def _bump_gesture(self):
        self._gesture_rev += 1

    def _animate_ants(self):
        if self.session.document and self.session.document.selection \
                and not self.session.document.selection.is_empty:
            self._ant_phase = (self._ant_phase + 1) % 1000
            self.update()

    def _draw_grid(self, painter, ox, oy, w, h, z):
        if not self.session.shows_grid:
            return
        step = 100
        painter.setPen(QPen(QColor(255, 255, 255, 40), 1))
        x = step
        while x < w:
            painter.drawLine(QPointF(ox + x * z, oy), QPointF(ox + x * z, oy + h * z))
            x += step
        y = step
        while y < h:
            painter.drawLine(QPointF(ox, oy + y * z), QPointF(ox + w * z, oy + y * z))
            y += step

    def _draw_guides(self, painter, ox, oy, w, h, z):
        for g in self.session.document.guides:
            painter.setPen(QPen(QColor(80, 170, 255, 200), 1,
                                Qt.PenStyle.DashLine))
            if g.vertical:
                x = ox + g.position * z
                painter.drawLine(QPointF(x, oy), QPointF(x, oy + h * z))
            else:
                y = oy + g.position * z
                painter.drawLine(QPointF(ox, y), QPointF(ox + w * z, y))

    def _draw_selection(self, painter, ox, oy, w, h, z):
        sel = self.session.document.selection
        if sel is None or sel.is_empty:
            return
        bbox = sel.bounding_box()
        if bbox is None:
            return
        pen = QPen(QColor(0, 0, 0), 1, Qt.PenStyle.CustomDashLine)
        pen.setDashPattern([4, 4]); pen.setDashOffset(self._ant_phase * 0.16)
        painter.setPen(pen)
        r = QRectF(ox + bbox[0] * z, oy + bbox[1] * z,
                   (bbox[2] - bbox[0]) * z, (bbox[3] - bbox[1]) * z)
        painter.drawRect(r)
        pen2 = QPen(QColor(255, 255, 255), 1, Qt.PenStyle.CustomDashLine)
        pen2.setDashPattern([4, 4]); pen2.setDashOffset(self._ant_phase * 0.16 + 4)
        painter.setPen(pen2)
        painter.drawRect(r)

    def _draw_drag_preview(self, painter, ox, oy, z):
        if self._drag_start_doc is None or self._drag_cur_doc is None:
            return
        s, c = self._drag_start_doc, self._drag_cur_doc
        tool = self.session.tool
        if tool in (NavigationTool.MARQUEE, NavigationTool.CROP):
            rect = QRectF(*[min(a, b) for a, b in zip(s, c)],
                          abs(c[0] - s[0]), abs(c[1] - s[1]))
            rect.translate(ox, oy)
            pen = QPen(QColor(255, 255, 255), 1, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(rect)
        elif tool == NavigationTool.LASSO and self._lasso_pts:
            path = QPainterPath()
            p0 = self.doc_to_widget(*self._lasso_pts[0])
            path.moveTo(p0)
            for pt in self._lasso_pts[1:]:
                path.lineTo(self.doc_to_widget(*pt))
            painter.setPen(QPen(QColor(255, 255, 255), 1, Qt.PenStyle.DashLine))
            painter.drawPath(path)
        elif tool == NavigationTool.GRADIENT:
            painter.setPen(QPen(QColor(255, 255, 255, 180), 1))
            painter.drawLine(self.doc_to_widget(*s), self.doc_to_widget(*c))
        elif tool == NavigationTool.SHAPE:
            rect = QRectF(min(s[0], c[0]), min(s[1], c[1]),
                          abs(c[0] - s[0]), abs(c[1] - s[1]))
            rect.translate(ox, oy)
            col = self.session.defaults.shape_fill
            painter.setPen(QPen(QColor(120, 120, 120), 1, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(int(col[0] * 255), int(col[1] * 255),
                                    int(col[2] * 255), int(col[3] * 200)))
            painter.drawRect(rect)

    # Move tool bounding box + handles -------------------------------
    def _active_transform_rects(self):
        rects = []
        for lid in self.session.selected_layer_ids:
            lyr = self.session.layer_by_id(lid)
            if lyr is None or lyr.is_group or not lyr.visible:
                continue
            rects.append(lyr.transform)
        return rects

    def _draw_move_handles(self, painter, ox, oy, z):
        if self._moving_layer or self._rotating or self._scaling is not None:
            pass
        for t in self._active_transform_rects():
            pts = [self.doc_to_widget(*p) for p in t.corners()]
            path = QPainterPath(pts[0])
            for p in pts[1:]:
                path.lineTo(p)
            path.closeSubpath()
            painter.setPen(QPen(QColor(255, 255, 255, 220), 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
            painter.setPen(QPen(QColor(120, 120, 120), 1))
            painter.setBrush(QColor(255, 255, 255))
            for p in pts:
                painter.drawRect(QRectF(p.x() - 4, p.y() - 4, 8, 8))
            c = self.doc_to_widget(*t.center)
            top = pts[0]
            painter.setPen(QPen(QColor(255, 255, 255, 160), 1, Qt.PenStyle.DotLine))
            painter.drawLine(QPointF(top.x(), top.y() - 18),
                             QPointF(top.x(), top.y()))
            painter.setBrush(QColor(255, 255, 255))
            painter.drawEllipse(QPointF(top.x(), top.y() - 18), 4, 4)

    def _handle_at(self, pos: QPointF) -> Optional[str]:
        for t in self._active_transform_rects():
            pts = [self.doc_to_widget(*p) for p in t.corners()]
            for i, p in enumerate(pts):
                if abs(pos.x() - p.x()) <= 6 and abs(pos.y() - p.y()) <= 6:
                    return ("scale", i, t)
            top = pts[0]
            if abs(pos.x() - top.x()) <= 6 and abs(pos.y() - (top.y() - 18)) <= 6:
                return ("rotate", -1, t)
        return None

    def _draw_brush_cursor(self, painter):
        pos = getattr(self, "_last_pos", None)
        if pos is None or not self.rect().contains(pos.toPoint() if hasattr(pos,"toPoint") else pos):
            return
        size = self.session.defaults.brush.size * self.session.zoom
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(pos, size / 2, size / 2)
        painter.setPen(QPen(QColor(0, 0, 0, 120), 1))
        painter.drawEllipse(pos, size / 2 + 1, size / 2 + 1)

    # Rulers ------------------------------------------------------------
    def _draw_rulers(self, painter, ox, oy, w, h, z):
        band = 22
        painter.fillRect(QRectF(0, 0, self.width(), band), QColor(48, 48, 50))
        painter.fillRect(QRectF(0, 0, band, self.height()), QColor(48, 48, 50))
        painter.setPen(QColor(170, 170, 170))
        f = QFont(); f.setPointSize(8); painter.setFont(f)
        step = 50
        while step * z < 40:
            step *= 2
        x = 0.0
        while x <= w:
            px = ox + x * z
            if band <= px <= self.width():
                painter.drawLine(QPointF(px, band - 5), QPointF(px, band))
                painter.drawText(QRectF(px + 2, 0, 60, band),
                                 Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                                 str(int(x)))
            x += step
        y = 0.0
        while y <= h:
            py = oy + y * z
            if band <= py <= self.height():
                painter.drawLine(QPointF(band - 5, py), QPointF(band, py))
                painter.save()
                painter.translate(band - 2, py - 2)
                painter.rotate(-90)
                painter.drawText(QRectF(2, -band, 80, band),
                                 Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                                 str(int(y)))
                painter.restore()
            y += step

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------
    def _modifier_mode(self, ev) -> str:
        mods = ev.modifiers()
        if mods & Qt.KeyboardModifier.AltModifier:
            return "subtract"
        if mods & Qt.KeyboardModifier.ShiftModifier:
            return "add"
        return "replace"

    def mousePressEvent(self, ev):
        if self.session.document is None:
            return
        pos = ev.position()
        doc_pt = self.widget_to_doc(pos)
        tool = self.session.tool
        self._last_pos = pos
        mods = ev.modifiers()

        # Middle button or space/hand: pan anywhere.
        if ev.button() == Qt.MouseButton.MiddleButton or tool == NavigationTool.HAND:
            self._panning = True
            self._pan_last_px = (pos.x(), pos.y())
            return
        if ev.button() == Qt.MouseButton.RightButton:
            # Right-drag pans like the SwiftUI build's fallback.
            self._panning = True
            self._pan_last_px = (pos.x(), pos.y())
            return
        if tool == NavigationTool.ZOOM:
            factor = 1.25 if not (mods & Qt.KeyboardModifier.ShiftModifier) else 0.8
            self._apply_zoom(factor, doc_pt)
            return
        if tool == NavigationTool.EYEDROPPER:
            col = self.session.sample_color_at(doc_pt)
            if col:
                self.session.fg_color = col
                self.finished_edit.emit("Pick Color")
            return

        # Brush-size shortcuts via [ ] handled in keyPress.
        if tool.is_brush_tool:
            self._start_paint(doc_pt)
            return
        if tool.is_selection_tool or tool == NavigationTool.CROP:
            self._drag_start_doc = doc_pt
            self._drag_cur_doc = doc_pt
            if tool == NavigationTool.LASSO:
                self._lasso_pts = [doc_pt]
            elif tool == NavigationTool.WAND:
                self.session.magic_wand(doc_pt,
                                        additive=bool(mods & Qt.KeyboardModifier.ShiftModifier))
                self._drag_start_doc = None
            return
        if tool == NavigationTool.TYPE:
            self._place_text(doc_pt)
            return
        if tool == NavigationTool.GRADIENT:
            self._drag_start_doc = doc_pt
            self._drag_cur_doc = doc_pt
            return
        if tool == NavigationTool.SHAPE:
            self._drag_start_doc = doc_pt
            self._drag_cur_doc = doc_pt
            return
        if tool == NavigationTool.MOVE:
            hit = self._handle_at(pos)
            if hit:
                kind, idx, t = hit
                lyr = self._layer_with_transform(t)
                self._transform_before = t.copy_with()
                if kind == "rotate":
                    self._rotating = True
                    self._drag_start_doc = doc_pt
                else:
                    self._scaling = (idx, lyr)
                    self._drag_start_doc = doc_pt
                self.session.selected_layer_ids = [lyr.id]
                self._pending_label = "Rotate Layer" if kind == "rotate" else "Scale Layer"
                self._bump_gesture()
                self.update()
                return
            # Hit-test layers top-most first.
            for lid in reversed([l.id for l in self._visible_layers()]):
                lyr = self.session.layer_by_id(lid)
                if lyr and lyr.transform.contains(doc_pt):
                    self.session.selected_layer_ids = [lyr.id]
                    self._moving_layer = True
                    self._drag_start_doc = doc_pt
                    self._move_origins = {
                        x.id: x.transform.origin
                        for x in self.session.document.layers
                        if x.id in self.session.selected_layer_ids}
                    self._pending_label = "Move Layer"
                    self.update()
                    return
            self.session.selected_layer_ids = []
            self.update()

    def _visible_layers(self):
        return [l for l, d, vis in hierarchy_entries_iter(self.session.document)
                if vis and not l.is_group]

    def _layer_with_transform(self, t: LayerTransform):
        for lid in self.session.selected_layer_ids:
            lyr = self.session.layer_by_id(lid)
            if lyr and lyr.transform is t:
                return lyr
        for l in self.session.document.layers:
            if l.transform is t:
                return l
        return self.session.active_layer

    def mouseMoveEvent(self, ev):
        if self.session.document is None:
            return
        pos = ev.position()
        self._last_pos = pos
        doc_pt = self.widget_to_doc(pos)
        self.pointer_changed.emit(doc_pt[0], doc_pt[1])

        if self._panning and self._pan_last_px:
            dx, dy = pos.x() - self._pan_last_px[0], pos.y() - self._pan_last_px[1]
            self.session.pan = (self.session.pan[0] + dx, self.session.pan[1] + dy)
            self._pan_last_px = (pos.x(), pos.y())
            self.update()
            return
        if self._painting:
            self._continue_paint(doc_pt)
            return
        if self._moving_layer and self._drag_start_doc:
            dx = doc_pt[0] - self._drag_start_doc[0]
            dy = doc_pt[1] - self._drag_start_doc[1]
            snap = self._snap_delta(dx, dy)
            dx, dy = snap
            for lid, org in self._move_origins.items():
                lyr = self.session.layer_by_id(lid)
                if lyr:
                    lyr.transform.origin = (org[0] + dx, org[1] + dy)
            self._bump_gesture(); self.update()
            return
        if self._rotating and self._drag_start_doc:
            lyr = self.session.active_layer
            if lyr:
                t = lyr.transform
                c = t.center
                a0 = math.degrees(math.atan2(self._drag_start_doc[1] - c[1],
                                             self._drag_start_doc[0] - c[0]))
                a1 = math.degrees(math.atan2(doc_pt[1] - c[1], doc_pt[0] - c[0]))
                base = self._transform_before.rotation
                rot = base + (a1 - a0)
                if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    rot = round(rot / 15.0) * 15
                lyr.transform = t.copy_with(rotation=rot)
            self._bump_gesture(); self.update()
            return
        if self._scaling is not None and self._drag_start_doc:
            idx, lyr = self._scaling
            if lyr:
                t = lyr.transform
                before = self._transform_before
                cx, cy = before.center
                ux, uy = before.transformed_point_from_document(doc_pt)
                nw = max(1.0, abs(ux - 0.5) * 2 * before.size[0])
                nh = max(1.0, abs(uy - 0.5) * 2 * before.size[1])
                if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    ratio = before.size[1] / before.size[0]
                    nh = nw * ratio
                origin = (cx - nw / 2, cy - nh / 2)
                lyr.transform = before.copy_with(origin=origin, size=(nw, nh))
            self._bump_gesture(); self.update()
            return
        if self._drag_start_doc is not None:
            self._drag_cur_doc = doc_pt
            tool = self.session.tool
            if tool == NavigationTool.LASSO:
                if not self._lasso_pts or \
                        math.hypot(doc_pt[0] - self._lasso_pts[-1][0],
                                   doc_pt[1] - self._lasso_pts[-1][1]) > 2:
                    self._lasso_pts.append(doc_pt)
            self.update()
            return
        # Hover cursor feedback
        if self.session.tool == NavigationTool.MOVE and self._handle_at(pos):
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif self.session.tool.is_brush_tool:
            self.update()
        else:
            self.unsetCursor()

    def _snap_delta(self, dx: float, dy: float) -> Tuple[float, float]:
        """Port of SnapGeometry: snap layer edges to guides / canvas centre."""
        s = self.session
        lyr = s.active_layer
        if lyr is None or not (s.snap_to_guides or s.snap_to_layers):
            return (dx, dy)
        t = lyr.transform.copy_with(origin=(lyr.transform.origin[0] + dx,
                                            lyr.transform.origin[1] + dy))
        tol = 6.0 / s.zoom
        corners = t.corners()
        targets_v = [g.position for g in s.document.guides if g.vertical] + \
                    [0, s.document.width / 2, s.document.width]
        targets_h = [g.position for g in s.document.guides if not g.vertical] + \
                    [0, s.document.height / 2, s.document.height]
        best_dx, best_dy = dx, dy
        bd = bh = tol
        for cx, cy in corners:
            for tv in targets_v:
                d = abs(cx - tv)
                if d < bd:
                    bd = d; best_dx = dx + (tv - cx)
            for th in targets_h:
                d = abs(cy - th)
                if d < bh:
                    bh = d; best_dy = dy + (th - cy)
        return (best_dx, best_dy)

    def mouseReleaseEvent(self, ev):
        if self.session.document is None:
            return
        doc_pt = self.widget_to_doc(ev.position())
        if self._panning:
            self._panning = False
            self._pan_last_px = None
            return
        if self._painting:
            self._end_paint(doc_pt)
            return
        if self._moving_layer:
            self._moving_layer = False
            self._finish("Move Layer")
            return
        if self._rotating:
            self._rotating = False
            self._finish("Rotate Layer")
            return
        if self._scaling is not None:
            self._scaling = None
            self._finish("Scale Layer")
            return
        if self._drag_start_doc is not None:
            start, cur = self._drag_start_doc, doc_pt
            tool = self.session.tool
            self._drag_start_doc = self._drag_cur_doc = None
            self._lasso_pts_done = None
            if tool == NavigationTool.SHAPE:
                self.commit_shape_drag(start, cur)
                self._bump_gesture(); self.update()
                return
            if tool == NavigationTool.GRADIENT:
                self.commit_gradient_drag(start, cur)
                return
            if tool == NavigationTool.CROP:
                rect = (min(start[0], cur[0]), min(start[1], cur[1]),
                        max(start[0], cur[0]), max(start[1], cur[1]))
                if rect[2] - rect[0] >= 4 and rect[3] - rect[1] >= 4:
                    self.session.crop(rect)
                    self._bump_gesture(); self.update()
                return
            if tool == NavigationTool.MARQUEE:
                rect = (min(start[0], cur[0]), min(start[1], cur[1]),
                        max(start[0], cur[0]), max(start[1], cur[1]))
                if abs(rect[2] - rect[0]) >= 2 and abs(rect[3] - rect[1]) >= 2:
                    mode = self._modifier_mode(ev)
                    if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
                        self.session.select_ellipse(rect, mode)
                    else:
                        self.session.select_rect(rect, mode)
            elif tool == NavigationTool.LASSO and len(self._lasso_pts) > 2:
                self.session.select_polygon(list(self._lasso_pts),
                                            self._modifier_mode(ev))
            self._lasso_pts = []
            self.update()
            return
        if self._drag_start_pending_shape:
            pass

    _drag_start_pending_shape = None

    def _finish(self, label):
        self._bump_gesture()
        self.update()
        doc = self.session.document
        if doc is not None:
            self.session.begin_edit(label)
            self.session.end_edit()
        self.finished_edit.emit(label)

    # Wheel: zoom (plain) / pan (shift) --------------------------------
    def wheelEvent(self, ev):
        if self.session.document is None:
            return
        delta = ev.angleDelta().y()
        mods = ev.modifiers()
        if mods & Qt.KeyboardModifier.ShiftModifier:
            self.session.pan = (self.session.pan[0] - delta * 0.5, self.session.pan[1])
            self.update()
            return
        factor = 1.1 if delta > 0 else 1 / 1.1
        doc_pt = self.widget_to_doc(ev.position())
        self._apply_zoom(factor, doc_pt)

    def _apply_zoom(self, factor, anchor_doc: Optional[Tuple[float, float]] = None):
        old = self.session.zoom
        new = min(self.MAX_ZOOM, max(self.MIN_ZOOM, old * factor))
        if new == old:
            return
        if anchor_doc is not None:
            # Keep the point under the cursor fixed.
            ox, oy = self._origin_px()
            wx = ox + anchor_doc[0] * old
            wy = oy + anchor_doc[1] * old
            self.session.zoom = new
            ox2, oy2 = self._origin_px()
            dx = wx - (ox2 + anchor_doc[0] * new)
            dy = wy - (oy2 + anchor_doc[1] * new)
            self.session.pan = (self.session.pan[0] + dx, self.session.pan[1] + dy)
        else:
            self.session.zoom = new
        self._bump_gesture()
        self.zoom_changed.emit(new)
        self.update()

    def zoom_fit(self):
        ds = self._doc_size()
        if ds is None:
            return
        z = min((self.width() - 60) / ds[0], (self.height() - 60) / ds[1])
        self.session.zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, z))
        self.session.pan = (0, 0)
        self._bump_gesture(); self.zoom_changed.emit(self.session.zoom); self.update()

    def zoom_actual(self):
        self.session.zoom = 1.0
        self.session.pan = (0, 0)
        self._bump_gesture(); self.zoom_changed.emit(1.0); self.update()

    # Keyboard -----------------------------------------------------------
    def keyPressEvent(self, ev):
        key = ev.key()
        mods = ev.modifiers()
        s = self.session
        if key == Qt.Key.Key_Space:
            # Temporary hand tool handled by press-and-hold elsewhere.
            return
        letters = {
            Qt.Key.Key_V: NavigationTool.MOVE, Qt.Key.Key_M: NavigationTool.MARQUEE,
            Qt.Key.Key_L: NavigationTool.LASSO, Qt.Key.Key_W: NavigationTool.WAND,
            Qt.Key.Key_C: NavigationTool.CROP, Qt.Key.Key_B: NavigationTool.BRUSH,
            Qt.Key.Key_J: NavigationTool.SPOT_HEALING, Qt.Key.Key_S: NavigationTool.CLONE_STAMP,
            Qt.Key.Key_R: NavigationTool.BLUR, Qt.Key.Key_G: NavigationTool.GRADIENT,
            Qt.Key.Key_U: NavigationTool.SHAPE, Qt.Key.Key_T: NavigationTool.TYPE,
            Qt.Key.Key_I: NavigationTool.EYEDROPPER, Qt.Key.Key_H: NavigationTool.HAND,
            Qt.Key.Key_Z: NavigationTool.ZOOM, Qt.Key.Key_A: NavigationTool.IDLE,
        }
        if key in letters and not (mods & (Qt.KeyboardModifier.ControlModifier |
                                           Qt.KeyboardModifier.MetaModifier)):
            s.tool = letters[key]
            self.update()
            return
        if key == Qt.Key.Key_E and not (mods & Qt.KeyboardModifier.ControlModifier):
            s.defaults.brush.erase_mode = True
            s.tool = NavigationTool.BRUSH
            self.update()
            return
        if key in (Qt.Key.Key_BracketLeft, Qt.Key.Key_BracketRight):
            b = s.defaults.brush
            step = max(1.0, b.size * 0.15)
            b.size = max(1.0, b.size + (step if key == Qt.Key.Key_BracketRight else -step))
            self.update()
            return
        if key == Qt.Key.Key_Comma:
            s.defaults.brush.hardness = max(0.0, s.defaults.brush.hardness - 0.1)
        if key == Qt.Key.Key_Period:
            s.defaults.brush.hardness = min(1.0, s.defaults.brush.hardness + 0.1)
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._apply_zoom(1.2)
        if key == Qt.Key.Key_Minus:
            self._apply_zoom(1 / 1.2)
        if key == Qt.Key.Key_0:
            self.zoom_actual()
        if key == Qt.Key.Key_1:
            self.zoom_fit()
        arrows = {Qt.Key.Key_Left: (-1, 0), Qt.Key.Key_Right: (1, 0),
                  Qt.Key.Key_Up: (0, -1), Qt.Key.Key_Down: (0, 1)}
        if key in arrows:
            dx, dy = arrows[key]
            n = 10 if (mods & Qt.KeyboardModifier.ShiftModifier) else 1
            s.nudge_active(dx * n, dy * n)
            self._bump_gesture(); self.update()
            return
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            s.delete_selection_contents()
            self._bump_gesture(); self.update()
            return
        super().keyPressEvent(ev)

    # ------------------------------------------------------------------
    # Painting gestures (brush / eraser / heal / clone / blur / mask)
    # ------------------------------------------------------------------
    def _target_paint_layer(self):
        s = self.session
        lyr = s.active_layer
        if lyr is None or lyr.is_group:
            if s.document is None:
                return None
            lyr = s.add_blank_layer("Painted Layer")
            s.selected_layer_ids = [lyr.id]
        return lyr

    def _start_paint(self, doc_pt):
        s = self.session
        tool = s.tool
        if tool == NavigationTool.CLONE_STAMP and (
                s.clone_source is None or
                bool(s.grabAlt if hasattr(s, "grabAlt") else False)):
            pass
        lyr = self._target_paint_layer()
        if lyr is None:
            return
        self._paint_target = lyr
        self._paint_prev_doc = doc_pt
        self._painting = True
        self._stroke_points = [doc_pt]
        self._apply_paint_stroke(doc_pt, doc_pt)

    def _continue_paint(self, doc_pt):
        if not self._painting:
            return
        prev = self._paint_prev_doc or doc_pt
        self._apply_paint_stroke(prev, doc_pt)
        self._paint_prev_doc = doc_pt
        self._bump_gesture()
        self.update()

    def _end_paint(self, doc_pt):
        self._painting = False
        self._paint_prev_doc = None
        self._finish("Paint")

    def _apply_paint_stroke(self, a, b):
        s = self.session
        tool = s.tool
        lyr = getattr(self, "_paint_target", None)
        if lyr is None:
            return
        br = s.defaults.brush
        if s.painting_on_mask:
            value = 0.0 if br.erase_mode else 1.0
            editing.apply_mask_paint(lyr, a, b, value, br.size, br.hardness, br.opacity)
            return
        if tool == NavigationTool.BRUSH:
            editing.stamp_brush(lyr, a, b, s.bg_color if br.erase_mode else s.fg_color,
                                br.size, br.hardness, br.opacity, erase=br.erase_mode)
        elif tool == NavigationTool.SPOT_HEALING:
            editing.spot_heal(lyr, b, br.size / 2)
        elif tool == NavigationTool.CLONE_STAMP:
            src = s.clone_source
            if src is None:
                # Alt-click sets the source point.
                if bool(self._alt_down()):
                    s.clone_source = (lyr.id, b)
                return
            src_layer = s.layer_by_id(src[0]) or lyr
            raster = src_layer.raster()
            if raster is not None:
                editing.clone_stamp_sample(lyr, b, src[1], raster, br.size, br.opacity)
        elif tool == NavigationTool.BLUR:
            editing.blur_smudge(lyr, a, b, br.size)

    def _alt_down(self):
        return bool(QApplication_instance().keyboardModifiers() &
                    Qt.KeyboardModifier.AltModifier)

    # Gradient / shape / text placement ---------------------------------
    def mouseDoubleClickEvent(self, ev):
        # Double-click with the type tool edits existing text near the click.
        pass

    def _place_text(self, doc_pt):
        from PyQt6.QtWidgets import QInputDialog, QApplication
        s = self.session
        txt, ok = QInputDialog.getText(self, "Type Tool", "Text:")
        if not ok or not txt:
            return
        t = LayerText(string=txt, font_size=s.defaults.text_size,
                      color_rgba=s.fg_color)
        w = max(40, len(txt) * s.defaults.text_size * 0.6)
        h = s.defaults.text_size * 1.5
        s.add_text_layer((doc_pt[0], doc_pt[1], doc_pt[0] + w, doc_pt[1] + h), t)
        self._bump_gesture(); self.update()

    def commit_shape_drag(self, start, cur):
        from ..core.model import LayerShape, ShapeKind
        s = self.session
        kind = next(k for k in ShapeKind if k.value == s.defaults.shape_kind)
        sh = LayerShape(kind=kind, fill_rgba=s.defaults.shape_fill)
        s.add_shape_layer((start[0], start[1], cur[0], cur[1]), sh)

    def commit_gradient_drag(self, start, cur):
        s = self.session
        lyr = self._target_paint_layer()
        if lyr is None:
            return
        editing.ensure_pixels(lyr)
        grad = editing.apply_gradient_tool(lyr.image, start, cur, s.defaults.gradient_stops)
        s.begin_edit("Gradient")
        lyr.image = grad
        s.end_edit()
        self._bump_gesture(); self.update()


def hierarchy_entries_iter(doc):
    from ..core.model import hierarchy_entries
    return hierarchy_entries(doc.layers, top_first=True)


def QApplication_instance():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance()
