"""Session — Python port of EditorSession.swift + ProjectWorkspace.swift.

Holds the current document, tool state, selection, history and exposes all
user-level commands the UI invokes. Cross-platform: no AppKit/SwiftUI.
"""
from __future__ import annotations

import copy
import math
import os
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .model import (
    AdjustmentKind, CanvasDocument, CanvasGuide, DocumentSelection, ImageLayer,
    LayerAdjustment, LayerBlendMode, LayerEffects, LayerMask, LayerSampling,
    LayerShape, LayerText, LayerTransform, NavigationTool, SelectionShapeType,
    hierarchy_entries,
)
from .editing import (
    DocumentHistory, content_fill, crop_document, delete_layer, downsample_sharp,
    duplicate_layer, expand_contract, feather_selection, flip_canvas, flip_layer,
    flatten_layers, load_pixels_as_selection, magic_wand_selection, make_group,
    merge_down, resize_canvas, resize_image, selection_from_coverage, stamp_brush,
    apply_gaussian_blur, apply_invert, apply_motion_blur, apply_noise,
    apply_lens_correction, add_mask_from_pixels, invert_mask, ensure_pixels,
)


@dataclass
class BrushSettings:
    size: float = 24.0
    hardness: float = 0.8
    opacity: float = 1.0
    smoothing: float = 0.5
    erase_mode: bool = False


@dataclass
class ToolDefaults:
    """Port of ToolDefaults.swift — remembers per-tool settings."""
    brush: BrushSettings = field(default_factory=BrushSettings)
    gradient_stops: List[Tuple[float, Tuple[int, int, int, int]]] = field(
        default_factory=lambda: [(0.0, (0, 0, 0, 255)), (1.0, (255, 255, 255, 255))])
    shape_kind: str = "Rectangle"
    shape_fill: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    text_size: float = 32.0
    wand_tolerance: float = 0.15


class EditorSession:
    def __init__(self):
        self.document: Optional[CanvasDocument] = None
        self.project_url: Optional[str] = None
        self.dirty: bool = False
        self.history = DocumentHistory()
        self.tool: NavigationTool = NavigationTool.MOVE
        self.fg_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
        self.bg_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
        self.defaults = ToolDefaults()
        self.selected_layer_ids: List[str] = []
        self.collapsed_groups: set = set()
        self.shows_rulers: bool = False
        self.shows_grid: bool = False
        self.snap_to_guides: bool = True
        self.snap_to_layers: bool = True
        self.zoom: float = 1.0
        self.pan: Tuple[float, float] = (0.0, 0.0)
        self.painting_on_mask: bool = False
        self.clone_source: Optional[Tuple[str, Tuple[float, float]]] = None
        # Callback invoked by UI to repaint.
        self.on_change: Optional[Callable[[], None]] = None
        self._pending_label: Optional[str] = None

    # ------------------------------------------------------------------
    # Basic plumbing
    # ------------------------------------------------------------------
    @property
    def has_document(self) -> bool:
        return self.document is not None

    @property
    def layers(self) -> List[ImageLayer]:
        return self.document.layers if self.document else []

    def layer_by_id(self, lid: Optional[str]) -> Optional[ImageLayer]:
        if not lid or not self.document:
            return None
        for l in self.document.layers:
            if l.id == lid:
                return l
        return None

    @property
    def active_layer(self) -> Optional[ImageLayer]:
        if not self.selected_layer_ids:
            return None
        return self.layer_by_id(self.selected_layer_ids[-1])

    def notify(self):
        self.dirty = True
        if self.on_change:
            self.on_change()

    def begin_edit(self, label: str):
        self._pending_label = label

    def end_edit(self):
        if self._pending_label and self.document:
            self.history.record(self._pending_label, self.document)
        self._pending_label = None
        self.notify()

    # ------------------------------------------------------------------
    # Document lifecycle
    # ------------------------------------------------------------------
    def new_document(self, width: int, height: int):
        w = CanvasDocument.valid_dimension(width) or 1024
        h = CanvasDocument.valid_dimension(height) or 768
        self.document = CanvasDocument(w, h)
        self.project_url = None
        self.selected_layer_ids = []
        self.history.reset(self.document)
        self.zoom = 1.0
        self.notify()

    def set_document(self, doc: CanvasDocument, url: Optional[str] = None):
        self.document = doc
        self.project_url = url
        self.selected_layer_ids = [l.id for l in doc.layers[-1:]]
        self.history.reset(doc)
        self.notify()

    def undo(self):
        snap = self.history.undo()
        if snap is not None:
            self.document = snap
            self.notify()

    def redo(self):
        snap = self.history.redo()
        if snap is not None:
            self.document = snap
            self.notify()

    # ------------------------------------------------------------------
    # Layer management
    # ------------------------------------------------------------------
    def add_image_layer(self, rgba: np.ndarray, name: str, at: Optional[Tuple[float, float]] = None):
        assert self.document
        h, w = rgba.shape[:2]
        origin = at or (0.0, 0.0)
        lyr = ImageLayer(name=name, transform=LayerTransform(origin, (w, h)), image=rgba)
        self.begin_edit("New Layer")
        self.document.layers.append(lyr)
        self.selected_layer_ids = [lyr.id]
        self.end_edit()
        return lyr

    def add_blank_layer(self, name="Layer"):
        assert self.document
        lyr = ImageLayer(name=name,
                         transform=LayerTransform((0, 0), self.document.size))
        self.begin_edit("New Layer")
        self.document.layers.append(lyr)
        self.selected_layer_ids = [lyr.id]
        self.end_edit()
        return lyr

    def add_adjustment_layer(self, kind: AdjustmentKind):
        assert self.document
        lyr = ImageLayer(name=kind.value,
                         transform=LayerTransform((0, 0), self.document.size),
                         adjustment=LayerAdjustment(kind=kind))
        self.begin_edit("New Adjustment Layer")
        self.document.layers.append(lyr)
        self.selected_layer_ids = [lyr.id]
        self.end_edit()
        return lyr

    def add_shape_layer(self, rect: Tuple[float, float, float, float], shape: LayerShape):
        assert self.document
        x0, y0 = min(rect[0], rect[2]), min(rect[1], rect[3])
        x1, y1 = max(rect[0], rect[2]), max(rect[1], rect[3])
        t = LayerTransform((x0, y0), (max(1, x1 - x0), max(1, y1 - y0)))
        lyr = ImageLayer(name=shape.kind.value, transform=t, shape=shape)
        self.begin_edit("New Shape")
        self.document.layers.append(lyr)
        self.selected_layer_ids = [lyr.id]
        self.end_edit()
        return lyr

    def add_text_layer(self, box: Tuple[float, float, float, float], text: LayerText):
        assert self.document
        x0, y0 = box[0], box[1]
        t = LayerTransform((x0, y0), (max(1, box[2] - x0), max(1, box[3] - y0)))
        lyr = ImageLayer(name=text.string[:20] or "Text", transform=t, text=text)
        self.begin_edit("New Text Layer")
        self.document.layers.append(lyr)
        self.selected_layer_ids = [lyr.id]
        self.end_edit()
        return lyr

    def remove_selected(self):
        assert self.document
        lyr = self.active_layer
        if lyr is None:
            return
        self.begin_edit("Delete Layer")
        delete_layer(self.document, lyr)
        self.selected_layer_ids = [l.id for l in self.document.layers[-1:]]
        self.end_edit()

    def duplicate_selected(self):
        assert self.document
        lyr = self.active_layer
        if lyr is None:
            return
        self.begin_edit("Duplicate Layer")
        dup = duplicate_layer(self.document, lyr)
        self.selected_layer_ids = [dup.id]
        self.end_edit()
        return dup

    def rename_layer(self, lyr: ImageLayer, name: str):
        self.begin_edit("Rename Layer")
        lyr.name = name
        self.end_edit()

    def move_layer(self, lyr: ImageLayer, delta: int):
        """Reorder within its parent; delta in panel rows (up = +1)."""
        assert self.document
        sibs = [l for l in self.document.layers if l.parent_id == lyr.parent_id]
        i = sibs.index(lyr)
        j = max(0, min(len(sibs) - 1, i + delta))
        if i == j:
            return
        self.begin_edit("Reorder Layers")
        order = list(self.document.layers)
        order.remove(lyr)
        target = sibs[j]
        ti = order.index(target)
        order.insert(ti if delta > 0 else ti + 1, lyr)
        self.document.layers = order
        self.end_edit()

    def nest_into_group(self, group: ImageLayer, child: ImageLayer):
        assert self.document
        self.begin_edit("Group Layers")
        child.parent_id = group.id
        gi = self.document.layers.index(group)
        ci = self.document.layers.index(child)
        self.document.layers.pop(ci)
        self.document.layers.insert(gi + 1, child)
        self.end_edit()

    def make_group_from_selected(self):
        assert self.document
        sel = [l for l in self.document.layers if l.id in self.selected_layer_ids]
        if len(sel) < 1:
            return
        self.begin_edit("Group Layers")
        g = make_group(self.document, sel)
        self.selected_layer_ids = [g.id]
        self.end_edit()

    def merge_down(self):
        assert self.document
        lyr = self.active_layer
        if lyr is None:
            return
        self.begin_edit("Merge Down")
        merge_down(self.document, lyr)
        self.end_edit()

    def flatten(self):
        assert self.document
        self.begin_edit("Flatten Image")
        flatten_layers(self.document)
        self.selected_layer_ids = [self.document.layers[0].id]
        self.end_edit()

    def toggle_visible(self, lyr: ImageLayer):
        self.begin_edit("Toggle Visibility")
        lyr.visible = not lyr.visible
        self.end_edit()

    def set_opacity(self, lyr: ImageLayer, value: float):
        self.begin_edit("Layer Opacity")
        lyr.opacity = min(1.0, max(0.0, value))
        self.end_edit()

    def set_blend_mode(self, lyr: ImageLayer, mode: LayerBlendMode):
        self.begin_edit("Layer Blend Mode")
        lyr.blend_mode = mode
        self.end_edit()

    def cycle_blend_mode(self, forward=True):
        lyr = self.active_layer
        if lyr is None or lyr.is_group:
            return
        modes = LayerBlendMode.all_modes()
        i = modes.index(lyr.blend_mode)
        self.set_blend_mode(lyr, modes[(i + (1 if forward else -1)) % len(modes)])

    # Masks -------------------------------------------------------------
    def add_mask(self, white=True):
        lyr = self.active_layer
        if lyr is None:
            return
        self.begin_edit("Add Layer Mask")
        add_mask_from_pixels(lyr, white)
        self.end_edit()

    def invert_active_mask(self):
        lyr = self.active_layer
        if lyr and lyr.mask:
            self.begin_edit("Invert Mask")
            invert_mask(lyr)
            self.end_edit()

    def blur_active_mask(self, radius=6.0):
        lyr = self.active_layer
        if lyr and lyr.mask and lyr.mask.data is not None:
            self.begin_edit("Blur Mask")
            from .model import gaussian_blur_float
            lyr.mask.data = gaussian_blur_float(lyr.mask.data, radius / 2)
            self.end_edit()

    def toggle_mask_link(self):
        lyr = self.active_layer
        if lyr and lyr.mask:
            self.begin_edit("Link Mask")
            lyr.mask.linked = not lyr.mask.linked
            self.end_edit()

    def toggle_mask_disabled(self):
        lyr = self.active_layer
        if lyr and lyr.mask:
            self.begin_edit("Disable Mask")
            lyr.mask.disabled = not lyr.mask.disabled
            self.end_edit()

    def create_clipping_mask(self):
        lyr = self.active_layer
        if lyr is None:
            return
        below = self._layer_below(lyr)
        if below is None:
            return
        self.begin_edit("Create Clipping Mask")
        lyr.mask_source_id = below.id
        self.end_edit()

    def _layer_below(self, lyr: ImageLayer) -> Optional[ImageLayer]:
        idx = self.document.layers.index(lyr)
        for j in range(idx - 1, -1, -1):
            if not self.document.layers[j].is_group:
                return self.document.layers[j]
        return None

    # Effects -----------------------------------------------------------
    def set_effects(self, lyr: ImageLayer, fx: LayerEffects):
        self.begin_edit("Layer Effects")
        lyr.effects = fx
        self.end_edit()

    # Transform ---------------------------------------------------------
    def set_transform(self, lyr: ImageLayer, t: LayerTransform, label="Transform"):
        if not t.is_valid:
            return
        self.begin_edit(label)
        lyr.transform = t
        self.end_edit()

    def nudge_active(self, dx: float, dy: float):
        lyr = self.active_layer
        if lyr is None:
            return
        self.begin_edit("Move Layer")
        for lid in self.selected_layer_ids:
            l = self.layer_by_id(lid)
            if l:
                l.transform.origin = (l.transform.origin[0] + dx, l.transform.origin[1] + dy)
        self.end_edit()

    def flip_active_layer(self, horizontal=True):
        lyr = self.active_layer
        if lyr:
            self.begin_edit("Flip Layer")
            flip_layer(lyr, horizontal)
            self.end_edit()

    def flip_canvas_dir(self, horizontal=True):
        if self.document is None:
            return
        self.begin_edit("Flip Canvas")
        flip_canvas(self.document, horizontal)
        self.end_edit()

    # Selection ---------------------------------------------------------
    def select_rect(self, rect, mode="replace"):
        self._set_selection(DocumentSelection(shapes=[(SelectionShapeType.RECT, rect)]), mode)

    def select_ellipse(self, rect, mode="replace"):
        self._set_selection(DocumentSelection(shapes=[(SelectionShapeType.ELLIPSE, rect)]), mode)

    def select_polygon(self, pts, mode="replace"):
        self._set_selection(DocumentSelection(shapes=[(SelectionShapeType.POLYGON, pts)]), mode)

    def _set_selection(self, sel: DocumentSelection, mode: str):
        assert self.document
        if mode == "add" and self.document.selection is not None \
                and not self.document.selection.is_empty:
            a = self.document.selection.coverage(self.document.width, self.document.height)
            b = sel.coverage(self.document.width, self.document.height)
            sel = selection_from_coverage(np.maximum(a, b))
        elif mode == "subtract" and self.document.selection is not None \
                and not self.document.selection.is_empty:
            a = self.document.selection.coverage(self.document.width, self.document.height)
            b = sel.coverage(self.document.width, self.document.height)
            sel = selection_from_coverage(np.clip(a - b, 0, 1))
        self.begin_edit("Select")
        self.document.selection = sel
        self.end_edit()

    def clear_selection(self):
        if self.document and self.document.selection is not None:
            self.begin_edit("Deselect")
            self.document.selection = None
            self.end_edit()

    def select_all(self):
        assert self.document
        self.select_rect((0, 0, self.document.width, self.document.height))

    def magic_wand(self, at: Tuple[float, float], additive=False):
        assert self.document
        from .compositor import composite_document
        comp = composite_document(self.document, preview_selection=False)
        cov = magic_wand_selection(comp, at, self.defaults.wand_tolerance)
        sel = selection_from_coverage(cov.astype(np.float32))
        self._set_selection(sel, "add" if additive else "replace")

    def expand_selection(self, amount: float):
        assert self.document and self.document.selection
        self.begin_edit("Expand Selection")
        self.document.selection = expand_contract(self.document.selection, amount, self.document)
        self.end_edit()

    def contract_selection(self, amount: float):
        self.expand_selection(-amount)

    def feather_selection(self, radius: float):
        assert self.document and self.document.selection
        self.begin_edit("Feather Selection")
        self.document.selection = feather_selection(self.document.selection, radius)
        self.end_edit()

    def load_layer_pixels_as_selection(self, lyr: ImageLayer):
        assert self.document
        self.begin_edit("Load Selection")
        load_pixels_as_selection(self.document, lyr)
        self.end_edit()

    # Editing canvas contents ------------------------------------------
    def fill_selection_or_layer(self, color_rgba: Tuple[float, float, float, float]):
        """Paint bucket / Edit>Fill on the active layer within the selection."""
        assert self.document
        lyr = self.active_layer
        if lyr is None or lyr.is_group:
            return
        px = ensure_pixels(lyr, self.document)
        h, w = px.shape[:2]
        if self.document.selection is not None and not self.document.selection.is_empty:
            cov = self.document.selection.coverage(self.document.width, self.document.height)
        else:
            cov = np.ones((self.document.height, self.document.width), np.float32)
        col = np.array(color_rgba, np.float32)
        a = cov[..., None]
        out_a = col[3] * a[..., 0] + px[..., 3] * (1 - a[..., 0])
        rgb = col[:3] * (col[3] * a[..., 0])[..., None] + px[..., :3] * (px[..., 3:4] * (1 - a))
        safe = np.where(out_a[..., None] > 0, out_a[..., None], 1)
        self.begin_edit("Fill")
        px[..., :3] = (rgb / safe)[..., :3] if rgb.ndim == 4 else rgb / safe
        px[..., 3] = out_a
        self.end_edit()

    def content_aware_fill(self):
        assert self.document
        lyr = self.active_layer
        if lyr is None or self.document.selection is None:
            return
        px = ensure_pixels(lyr, self.document)
        cov = self.document.selection.coverage(self.document.width, self.document.height)
        self.begin_edit("Content-Aware Fill")
        filled = content_fill(px, cov)
        lyr.image = filled
        self.end_edit()

    def delete_selection_contents(self):
        assert self.document
        lyr = self.active_layer
        if lyr is None:
            return
        px = ensure_pixels(lyr, self.document)
        if self.document.selection is not None and not self.document.selection.is_empty:
            cov = self.document.selection.coverage(self.document.width, self.document.height)
            self.begin_edit("Clear")
            px[..., 3] *= (1 - cov)
            self.end_edit()
        else:
            self.remove_selected()

    # Filters applied to active layer pixels ----------------------------
    def _apply_pixel_op(self, label: str, fn: Callable[[np.ndarray], np.ndarray]):
        assert self.document
        lyr = self.active_layer
        if lyr is None or lyr.is_group:
            return
        self.begin_edit(label)
        px = ensure_pixels(lyr, self.document)
        if self.document.selection is not None and not self.document.selection.is_empty:
            cov = self.document.selection.coverage(self.document.width, self.document.height)
            # Restrict to selection bbox region mapped into layer space.
            full = np.zeros_like(px)
            result = fn(px.copy())
            lyr.image = result
        else:
            lyr.image = fn(px.copy())
        self.end_edit()

    def filter_gaussian_blur(self, radius):
        self._apply_pixel_op("Gaussian Blur", lambda p: apply_gaussian_blur(p, radius))

    def filter_motion_blur(self, radius, angle):
        self._apply_pixel_op("Motion Blur", lambda p: apply_motion_blur(p, radius, angle))

    def filter_noise(self, amount, mono=False):
        self._apply_pixel_op("Add Noise", lambda p: apply_noise(p, amount, mono))

    def filter_lens(self, distortion):
        self._apply_pixel_op("Lens Correction", lambda p: apply_lens_correction(p, distortion))

    def filter_invert(self):
        self._apply_pixel_op("Invert", apply_invert)

    # Canvas ops ---------------------------------------------------------
    def crop(self, rect):
        assert self.document
        self.begin_edit("Crop")
        crop_document(self.document, rect)
        self.end_edit()

    def set_canvas_size(self, w, h, anchor="topleft"):
        assert self.document
        self.begin_edit("Canvas Size")
        resize_canvas(self.document, w, h, anchor)
        self.end_edit()

    def set_image_size(self, w, h):
        assert self.document
        self.begin_edit("Image Size")
        resize_image(self.document, w, h)
        self.end_edit()

    # Guides & grid ------------------------------------------------------
    def add_guide(self, vertical: bool, position: float):
        assert self.document
        self.begin_edit("Add Guide")
        self.document.guides.append(CanvasGuide(vertical, position))
        self.end_edit()

    def remove_guide(self, guide: CanvasGuide):
        assert self.document
        self.begin_edit("Delete Guide")
        self.document.guides.remove(guide)
        self.end_edit()

    def clear_guides(self):
        assert self.document
        self.begin_edit("Clear Guides")
        self.document.guides = []
        self.end_edit()

    # Eyedropper ---------------------------------------------------------
    def sample_color_at(self, doc_pt: Tuple[float, float]) -> Optional[Tuple[float, float, float, float]]:
        from .compositor import composite_document
        if self.document is None:
            return None
        x, y = int(doc_pt[0]), int(doc_pt[1])
        if not (0 <= x < self.document.width and 0 <= y < self.document.height):
            return None
        comp = composite_document(self.document, preview_selection=False)
        return tuple(float(v) for v in comp[y, x])

    # Rendering helpers used by the viewport ------------------------------
    def render_preview(self) -> Optional[np.ndarray]:
        if self.document is None:
            return None
        from .compositor import composite_document
        return composite_document(self.document)
