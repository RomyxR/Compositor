"""Compositor — main application window.

Python/Qt port of CompositorApp.swift + ContentView.swift +
ProjectWindowBridge.swift: assembles the canvas viewport, tool options bar,
layers panel and status bar, wires up all menus / keyboard shortcuts and the
project open/save/export actions.

Run with:  python -m compositor_py   (or python compositor_py/main.py)
"""
from __future__ import annotations

import os
import sys
import traceback

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QKeySequence
from PyQt6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
                             QMainWindow, QMessageBox, QScrollArea, QSizePolicy,
                             QToolBar, QVBoxLayout, QWidget)

from .core.model import AdjustmentKind, NavigationTool
from .core.session import EditorSession
from .io_files.project_io import (ProjectStore, copy_merged_rgba, export_jpeg,
                                  export_png, import_image, read_psd)
from .ui.canvas_view import CanvasView
from .ui.dialogs import AdjustmentDialog, LayerEffectsDialog, NewDocumentDialog
from .ui.layers_panel import LayersPanel
from .ui.tool_options import ToolOptionsBar


class MainWindow(QMainWindow):
    def __init__(self, project_path: str | None = None):
        super().__init__()
        self.setWindowTitle("Compositor")
        self.resize(1400, 900)
        self.store = ProjectStore()

        # --- session & widgets ------------------------------------------
        self.session = EditorSession()
        self.canvas = CanvasView(self.session)
        self.options = ToolOptionsBar(self.session)
        self.layers = LayersPanel(self.session)

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(False)
        scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll.setStyleSheet("QScrollArea { background:#1e1e22; border:none; }")
        self.scroll = scroll
        self.canvas.setMinimumSize(800, 600)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(0)
        rv.addWidget(self.options)
        rv.addWidget(self.layers, 1)
        right.setFixedWidth(320)
        right.setStyleSheet("background:#232327;")

        central = QWidget()
        hb = QHBoxLayout(central)
        hb.setContentsMargins(0, 0, 0, 0)
        hb.setSpacing(0)
        hb.addWidget(scroll, 1)
        hb.addWidget(right)
        self.setCentralWidget(central)

        # --- status bar ---------------------------------------------------
        self.pos_label = QLabel("")
        self.zoom_label = QLabel("100%")
        self.mode_label = QLabel(NavigationTool.MOVE.value)
        sb = self.statusBar()
        sb.setStyleSheet("color:#bbb; background:#1b1b1f;")
        sb.addWidget(self.pos_label)
        sb.addPermanentWidget(self.mode_label)
        sb.addPermanentWidget(self.zoom_label)

        self.canvas.pointer_changed.connect(
            lambda x, y: self.pos_label.setText(f"{x:.0f}, {y:.0f}"))
        self.canvas.zoom_changed.connect(
            lambda z: self.zoom_label.setText(f"{z * 100:.0f}%"))
        self.canvas.finished_edit.connect(lambda _label: self.refresh_all())
        self.layers.selection_changed.connect(self.canvas.update)

        # Debounced repaint while painting gestures run.
        self._repaint_timer = QTimer(self)
        self._repaint_timer.setSingleShot(True)
        self._repaint_timer.setInterval(16)
        self._repaint_timer.timeout.connect(self.canvas.update)

        self.session.on_change = self._on_session_change
        self.options.refresh_requested = self.refresh_all  # type: ignore[attr-defined]

        self._build_menus()
        self._build_toolbar()

        if project_path:
            self._open_project(project_path)
        else:
            self.new_document()

    # ------------------------------------------------------------------
    # Session plumbing
    # ------------------------------------------------------------------
    def _on_session_change(self):
        """EditorSession.notify() -> repaint canvas + panels."""
        self._repaint_timer.start()
        self.layers.refresh()
        self.options.refresh()
        self._update_title()

    def refresh_all(self):
        self.canvas.update()
        self.layers.refresh()
        self.options.refresh()
        self._update_title()

    def _update_title(self):
        name = "Untitled"
        if self.session.project_url:
            name = os.path.basename(self.session.project_url)
        elif self.session.document:
            name = "Untitled"
        dirty = "*" if self.session.dirty else ""
        self.setWindowTitle(f"{name}{dirty} — Compositor")

    # ------------------------------------------------------------------
    # Menus / toolbar
    # ------------------------------------------------------------------
    def _act(self, menu, text, slot, shortcut=None):
        from PyQt6.QtGui import QAction
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.triggered.connect(slot)
        menu.addAction(a)
        return a

    def _build_menus(self):
        mb = self.menuBar()
        mb.setStyleSheet("QMenuBar { background:#1b1b1f; color:#ddd; }"
                         "QMenuBar::item:selected { background:#3d6fb0; }"
                         "QMenu { background:#26262a; color:#ddd; }"
                         "QMenu::item:selected { background:#3d6fb0; }")

        m_file = mb.addMenu("&File")
        self._act(m_file, "New…", self.new_document, "Ctrl+N")
        self._act(m_file, "Open…", self.open_project, "Ctrl+O")
        m_file.addSeparator()
        self._act(m_file, "Save", self.save_project, "Ctrl+S")
        self._act(m_file, "Save As…", self.save_project_as, "Ctrl+Shift+S")
        m_file.addSeparator()
        self._act(m_file, "Import Image…", self.import_image, "Ctrl+I")
        self._act(m_file, "Import PSD…", self.import_psd)
        self._act(m_file, "Export PNG…", self.export_png, "Ctrl+E")
        self._act(m_file, "Export JPEG…", self.export_jpeg, "Ctrl+Shift+E")
        m_file.addSeparator()
        self._act(m_file, "Quit", self.close, "Ctrl+Q")

        m_edit = mb.addMenu("&Edit")
        self.undo_act = self._act(m_edit, "Undo", self.session.undo, "Ctrl+Z")
        self.redo_act = self._act(m_edit, "Redo", self.session.redo,
                                  "Ctrl+Shift+Z")
        m_edit.addSeparator()
        self._act(m_edit, "Cut Merged", self.cut_merged, "Ctrl+X")
        self._act(m_edit, "Copy Merged", self.copy_merged, "Ctrl+C")
        self._act(m_edit, "Paste as Layer", self.paste_layer, "Ctrl+V")
        m_edit.addSeparator()
        self._act(m_edit, "Deselect", self.session.clear_selection, "Ctrl+D")
        self._act(m_edit, "Select All", self.session.select_all, "Ctrl+A")
        self._act(m_edit, "Invert Selection", self.invert_selection, "Ctrl+Shift+I")

        m_image = mb.addMenu("&Image")
        self._act(m_image, "Flip Canvas Horizontal",
                  lambda: self.session.flip_canvas_dir(True))
        self._act(m_image, "Flip Canvas Vertical",
                  lambda: self.session.flip_canvas_dir(False))
        m_image.addSeparator()
        self._act(m_image, "Crop to Selection", self.crop_to_selection)
        self._act(m_image, "Canvas Size…", self.canvas_size)
        self._act(m_image, "Image Size…", self.image_size)
        m_image.addSeparator()
        self._act(m_image, "Flatten Image", self.session.flatten)

        m_layer = mb.addMenu("&Layer")
        self._act(m_layer, "New Layer", lambda: self.session.add_blank_layer(),
                  "Ctrl+Shift+N")
        self._act(m_layer, "Duplicate Layer", self.session.duplicate_selected,
                  "Ctrl+J")
        self._act(m_layer, "Delete Layer", self.session.remove_selected,
                  "Delete")
        self._act(m_layer, "Merge Down", self.session.merge_down, "Ctrl+M")
        self._act(m_layer, "Group Selected", self.session.make_group_from_selected,
                  "Ctrl+G")
        m_layer.addSeparator()
        self._act(m_layer, "Add Layer Mask", lambda: self.session.add_mask(True))
        self._act(m_layer, "Create Clipping Mask", self.session.create_clipping_mask)
        self._act(m_layer, "Layer Effects…", self.layer_effects)
        m_layer.addSeparator()
        for kind in AdjustmentKind:
            self._act(m_layer, f"New {kind.value} Adjustment",
                      lambda checked=False, k=kind: self.add_adjustment(k))

        m_select = mb.addMenu("&Selection")
        self._act(m_select, "Expand…", lambda: self._selection_amount(
            self.session.expand_selection, "Expand"))
        self._act(m_select, "Contract…", lambda: self._selection_amount(
            self.session.contract_selection, "Contract"))
        self._act(m_select, "Feather…", lambda: self._selection_amount(
            self.session.feather_selection, "Feather"))
        m_select.addSeparator()
        self._act(m_select, "Content-Aware Fill", self.session.content_aware_fill)
        self._act(m_select, "Delete Contents",
                  self.session.delete_selection_contents)

        m_filter = mb.addMenu("F&ilters")
        self._act(m_filter, "Gaussian Blur…", lambda: self._filter_dialog(
            "Gaussian Blur", [("Radius", 1, 200, 10)],
            lambda v: self.session.filter_gaussian_blur(v[0])))
        self._act(m_filter, "Motion Blur…", lambda: self._filter_dialog(
            "Motion Blur", [("Radius", 1, 200, 20), ("Angle", 0, 360, 0)],
            lambda v: self.session.filter_motion_blur(v[0], v[1])))
        self._act(m_filter, "Add Noise…", lambda: self._filter_dialog(
            "Add Noise", [("Amount", 1, 100, 25)],
            lambda v: self.session.filter_noise(v[0] / 100)))
        self._act(m_filter, "Lens Correction…", lambda: self._filter_dialog(
            "Lens Correction", [("Distortion", -100, 100, 10)],
            lambda v: self.session.filter_lens(v[0] / 100)))
        self._act(m_filter, "Invert", self.session.filter_invert)

        m_view = mb.addMenu("&View")
        self._act(m_view, "Zoom In", lambda: self.canvas._apply_zoom(1.2), "Ctrl++")
        self._act(m_view, "Zoom Out", lambda: self.canvas._apply_zoom(1 / 1.2), "Ctrl+-")
        self._act(m_view, "Fit on Screen", self.canvas.zoom_fit, "Ctrl+0")
        self._act(m_view, "Actual Pixels", self.canvas.zoom_actual, "Ctrl+1")
        m_view.addSeparator()
        self._act(m_view, "Toggle Rulers", self.toggle_rulers, "Ctrl+R")
        self._act(m_view, "Toggle Grid", self.toggle_grid, "Ctrl+'")
        self._act(m_view, "Snap to Guides", self.toggle_snap_guides)
        self._act(m_view, "Snap to Layers", self.toggle_snap_layers)
        m_view.addSeparator()
        self._act(m_view, "Toggle Mask Painting", self.toggle_mask_paint)

        m_help = mb.addMenu("&Help")
        self._act(m_help, "Keyboard Shortcuts", self.show_shortcuts)
        self._act(m_help, "About", self.about)

    def _build_toolbar(self):
        tb = QToolBar("Tools")
        tb.setMovable(False)
        tb.setStyleSheet("QToolBar { background:#1b1b1f; border:none; }")
        self.addToolBar(tb)
        order = [
            (NavigationTool.MOVE, "Move", "V"),
            (NavigationTool.MARQUEE, "Marquee", "M"),
            (NavigationTool.LASSO, "Lasso", "L"),
            (NavigationTool.WAND, "Wand", "W"),
            (NavigationTool.CROP, "Crop", "C"),
            (NavigationTool.BRUSH, "Brush", "B"),
            (NavigationTool.SPOT_HEALING, "Heal", "J"),
            (NavigationTool.CLONE_STAMP, "Clone", "S"),
            (NavigationTool.BLUR, "Blur", "R"),
            (NavigationTool.GRADIENT, "Gradient", "G"),
            (NavigationTool.SHAPE, "Shape", "U"),
            (NavigationTool.TYPE, "Type", "T"),
            (NavigationTool.EYEDROPPER, "Picker", "I"),
            (NavigationTool.HAND, "Hand", "H"),
            (NavigationTool.ZOOM, "Zoom", "Z"),
        ]
        self._tool_actions = {}
        for tool, label, key in order:
            a = tb.addAction(label)
            a.setToolTip(f"{label} ({key})")
            a.triggered.connect(lambda checked=False, t=tool: self.set_tool(t))
            self._tool_actions[tool] = a
        tb.addSeparator()
        tb.addAction("FG").triggered.connect(lambda: self._pick_color(True))
        tb.addAction("BG").triggered.connect(lambda: self._pick_color(False))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def set_tool(self, t: NavigationTool):
        self.session.tool = t
        self.mode_label.setText(t.value)
        self.options.refresh()
        self.canvas.update()

    def toggle_rulers(self):
        self.session.shows_rulers = not self.session.shows_rulers
        self.canvas.update()

    def toggle_grid(self):
        self.session.shows_grid = not self.session.shows_grid
        self.canvas.update()

    def toggle_snap_guides(self):
        self.session.snap_to_guides = not self.session.snap_to_guides

    def toggle_snap_layers(self):
        self.session.snap_to_layers = not self.session.snap_to_layers

    def toggle_mask_paint(self):
        self.session.painting_on_mask = not self.session.painting_on_mask
        self.mode_label.setText(("Mask" if self.session.painting_on_mask
                                 else "Layer") + " paint")
        self.canvas.update()

    def invert_selection(self):
        s = self.session
        if not s.document or s.document.selection is None:
            return
        s.begin_edit("Invert Selection")
        sel = s.document.selection
        cov = 1.0 - sel.coverage
        from .core.editing import selection_from_coverage
        s.document.selection = selection_from_coverage(cov)
        s.end_edit()

    def crop_to_selection(self):
        s = self.session
        if not s.document or s.document.selection is None:
            return
        bbox = s.document.selection.bounding_box()
        if bbox:
            s.crop(bbox)

    def canvas_size(self):
        from PyQt6.QtWidgets import QInputDialog
        s = self.session
        if not s.document:
            return
        w, ok1 = QInputDialog.getInt(self, "Canvas Width", "px:",
                                     s.document.width, 1, 30000)
        if not ok1:
            return
        h, ok2 = QInputDialog.getInt(self, "Canvas Height", "px:",
                                     s.document.height, 1, 30000)
        if ok2:
            s.set_canvas_size(w, h)

    def image_size(self):
        from PyQt6.QtWidgets import QInputDialog
        s = self.session
        if not s.document:
            return
        w, ok1 = QInputDialog.getInt(self, "Image Width", "px:",
                                     s.document.width, 1, 30000)
        if not ok1:
            return
        h, ok2 = QInputDialog.getInt(self, "Image Height", "px:",
                                     s.document.height, 1, 30000)
        if ok2:
            s.set_image_size(w, h)

    def layer_effects(self):
        lyr = self.session.active_layer
        if lyr is None:
            return
        LayerEffectsDialog(self.session, lyr, self).exec()
        self.refresh_all()

    def add_adjustment(self, kind: AdjustmentKind):
        lyr = self.session.add_adjustment_layer(kind)
        if lyr is not None and kind.is_editable:
            AdjustmentDialog(self.session, lyr, self).exec()
        self.refresh_all()

    def _selection_amount(self, fn, title):
        from PyQt6.QtWidgets import QInputDialog
        v, ok = QInputDialog.getDouble(self, title, "Pixels:", 5.0, 0.1, 500, 1)
        if ok:
            fn(v)
            self.refresh_all()

    def _filter_dialog(self, title, fields, apply_fn):
        from PyQt6.QtWidgets import (QDialog, QDialogButtonBox, QDoubleSpinBox,
                                     QFormLayout)
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        form = QFormLayout(dlg)
        spins = []
        for label, lo, hi, val in fields:
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setValue(val)
            form.addRow(label + ":", sp)
            spins.append(sp)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            apply_fn([sp.value() for sp in spins])
            self.refresh_all()

    def _pick_color(self, foreground=True):
        from PyQt6.QtGui import QColorDialog
        cur = self.session.fg_color if foreground else self.session.bg_color
        c = QColorDialog.getColor(
            QColor(int(cur[0] * 255), int(cur[1] * 255), int(cur[2] * 255)),
            self, "Foreground" if foreground else "Background")
        if c.isValid():
            rgba = (c.redF(), c.greenF(), c.blueF(), c.alphaF())
            if foreground:
                self.session.fg_color = rgba
            else:
                self.session.bg_color = rgba
            self.options.refresh()

    # Clipboard (merged) --------------------------------------------------
    def copy_merged(self):
        if not self.session.document:
            return
        rgba = copy_merged_rgba(self.session.document)
        QApplication.clipboard().setImage(_rgba_to_qimage(rgba))

    def cut_merged(self):
        self.copy_merged()
        self.session.delete_selection_contents()

    def paste_layer(self):
        from PyQt6.QtGui import QImage
        img = QApplication.clipboard().image()
        if img.isNull():
            return
        rgba = _qimage_to_rgba(img.convertToFormat(QImage.Format.Format_RGBA8888))
        self.session.add_image_layer(rgba, "Pasted Layer")
        self.refresh_all()

    # Documents / projects -------------------------------------------------
    def new_document(self):
        dlg = NewDocumentDialog(self)
        if dlg.exec() == NewDocumentDialog.DialogCode.Accepted:
            self.session.new_document(dlg.w.value(), dlg.h.value())
            self.session.document.resolution = dlg.res.value()
            self.canvas.zoom_fit()
            self.refresh_all()

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "",
            "Compositor Projects (*.compositor);;All Files (*)")
        if path:
            self._open_project(path)

    def _open_project(self, path):
        try:
            doc = self.store.load(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Open failed", str(e))
            return
        self.session.set_document(doc, path)
        self.canvas.zoom_fit()
        self.refresh_all()

    def save_project(self):
        if not self.session.document:
            return
        if not self.session.project_url:
            return self.save_project_as()
        self._write_project(self.session.project_url)

    def save_project_as(self):
        if not self.session.document:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "untitled.compositor",
            "Compositor Projects (*.compositor)")
        if path:
            if not path.endswith(".compositor"):
                path += ".compositor"
            self.session.project_url = path
            self._write_project(path)

    def _write_project(self, path):
        active = self.session.active_layer
        try:
            self.store.save(self.session.document, path,
                            active.id if active else None)
            self.session.dirty = False
            self._update_title()
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(e))

    def import_image(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import Images", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.webp *.bmp *.heic *.psd *.psb);;"
            "All Files (*)")
        for p in paths:
            try:
                rgba, name = import_image(p)
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, "Import failed", f"{p}\n{e}")
                continue
            if self.session.document is None:
                self.session.new_document(rgba.shape[1], rgba.shape[0])
            self.session.add_image_layer(rgba, name)
        if paths:
            self.canvas.zoom_fit()
            self.refresh_all()

    def import_psd(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import PSD", "", "Photoshop files (*.psd *.psb)")
        if not path:
            return
        try:
            doc = read_psd(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "PSD import failed", str(e))
            return
        self.session.set_document(doc, None)
        self.canvas.zoom_fit()
        self.refresh_all()

    def export_png(self):
        if not self.session.document:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export PNG", "export.png",
                                              "PNG images (*.png)")
        if path:
            export_png(self.session.document, path)

    def export_jpeg(self):
        if not self.session.document:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export JPEG", "export.jpg",
                                              "JPEG images (*.jpg *.jpeg)")
        if path:
            from PyQt6.QtWidgets import QInputDialog
            q, ok = QInputDialog.getInt(self, "JPEG quality", "Quality:", 90, 1, 100)
            if ok:
                export_jpeg(self.session.document, path, quality=q)

    # Misc -----------------------------------------------------------------
    def show_shortcuts(self):
        QMessageBox.information(self, "Keyboard Shortcuts", SHORTCUTS_TEXT)

    def about(self):
        QMessageBox.about(
            self, "About Compositor",
            "Compositor — a layered raster image editor.\n\n"
            "Python/Qt cross-platform port of the original Swift app.")

    def closeEvent(self, ev):
        if self.session.dirty:
            res = QMessageBox.question(
                self, "Unsaved changes", "Save before closing?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel)
            if res == QMessageBox.StandardButton.Save:
                self.save_project()
                if self.session.dirty:
                    ev.ignore()
                    return
            elif res == QMessageBox.StandardButton.Cancel:
                ev.ignore()
                return
        ev.accept()


SHORTCUTS_TEXT = """\
V Move · M Marquee · L Lasso · W Wand · C Crop · B Brush · J Heal · S Clone \
· R Blur · G Gradient · U Shape · T Type · I Picker · H Hand · Z Zoom
[ / ] brush size · , / . hardness · E eraser
Ctrl+Z undo · Ctrl+Shift+Z redo · Ctrl+A select all · Ctrl+D deselect
Delete delete contents · Arrows nudge (Shift ×10)
Ctrl+0 fit · Ctrl+1 100% · Ctrl+= / - zoom in/out · Ctrl+R rulers"""


def _rgba_to_qimage(rgba):
    from PyQt6.QtGui import QImage
    import numpy as np
    arr = np.ascontiguousarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8))
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, 4 * w,
                  QImage.Format.Format_RGBA8888).copy()


def _qimage_to_rgba(img):
    import numpy as np
    ptr = img.constBits()
    buf = ptr.setsize(img.sizeInBytes()) if hasattr(ptr, "setsize") else ptr
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(
        img.height(), img.bytesPerLine() // 4, 4)[:, :img.width(), :]
    return arr.astype(np.float32) / 255.0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True) \
        if hasattr(Qt, "AA_ShareOpenGLContexts") else None
    app = QApplication(argv)
    app.setApplicationName("Compositor")
    app.setStyle("Fusion")
    project = argv[0] if argv and os.path.isfile(argv[0]) else None
    win = MainWindow(project)
    win.show()
    return app.exec()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        raise
