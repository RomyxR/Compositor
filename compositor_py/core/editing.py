"""Editing operations — Python port of Compositor/Document/*.swift and
Rendering pixel kernels (BrushPixels.c, LevelsPixels.c, NoisePixels.c,
ContentFill.c, HealPixels.c, LensPixels.c) plus DocumentHistory.swift.

Covers: undo history, painting (brush/eraser/heal/clone/blur), selections
(magic wand, expand/contract/feather), content-aware fill, filters
(gaussian/motion blur, noise, lens correction, invert…), crop / canvas
size / image size, layer ops (merge, duplicate, reorder, masks, flip),
transform helpers and eyedropper sampling.
"""
from __future__ import annotations

import copy
import math
import uuid
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from .model import (
    AdjustmentKind, CanvasDocument, CanvasGuide, DocumentSelection, ImageLayer,
    LayerAdjustment, LayerBlendMode, LayerMask, LayerShape, LayerText,
    LayerTransform, NavigationTool, SelectionShapeType, hierarchy_entries,
)


# ---------------------------------------------------------------------------
# Undo history — port of DocumentHistory.swift
# ---------------------------------------------------------------------------

@dataclass
class HistoryStep:
    label: str
    snapshot: "CanvasDocument"


class DocumentHistory:
    MAX_STEPS = 100

    def __init__(self):
        self.steps: List[HistoryStep] = []
        self.index: int = -1  # index of the currently-shown step

    def reset(self, doc: CanvasDocument):
        self.steps = [HistoryStep("Open", copy.deepcopy(doc))]
        self.index = 0

    def record(self, label: str, doc: CanvasDocument):
        # Drop any redo branch, then push.
        self.steps = self.steps[:self.index + 1]
        self.steps.append(HistoryStep(label, copy.deepcopy(doc)))
        if len(self.steps) > self.MAX_STEPS:
            self.steps.pop(0)
        self.index = len(self.steps) - 1

    def can_undo(self) -> bool:
        return self.index > 0

    def can_redo(self) -> bool:
        return self.index < len(self.steps) - 1

    def undo(self) -> Optional[CanvasDocument]:
        if not self.can_undo():
            return None
        self.index -= 1
        return copy.deepcopy(self.steps[self.index].snapshot)

    def redo(self) -> Optional[CanvasDocument]:
        if not self.can_redo():
            return None
        self.index += 1
        return copy.deepcopy(self.steps[self.index].snapshot)

    @property
    def undo_label(self) -> str:
        return self.steps[self.index].label if self.can_undo() else ""

    @property
    def redo_label(self) -> str:
        return self.steps[self.index + 1].label if self.can_redo() else ""


# ---------------------------------------------------------------------------
# Raster helpers
# ---------------------------------------------------------------------------

def ensure_pixels(layer: ImageLayer, doc: Optional[CanvasDocument] = None) -> np.ndarray:
    """Blank layers allocate pixels when painting begins (Swift comment ported)."""
    if layer.image is None:
        w, h = (max(1, int(round(s))) for s in layer.transform.size)
        layer.image = np.zeros((h, w, 4), dtype=np.float32)
    return layer.image


def _doc_to_layer(layer: ImageLayer, p: Tuple[float, float]) -> Tuple[float, float]:
    ux, uy = layer.transform.transformed_point_from_document(p)
    w, h = layer.pixel_size
    return (ux * w, uy * h)


def paint_coverage(shape: str, center: Tuple[float, float], radius: float,
                   hardness: float, arr_shape: Tuple[int, int]) -> Tuple[slice, slice, np.ndarray]:
    """Circular brush falloff; returns row/col slices and a coverage patch."""
    h, w = arr_shape
    r = max(0.5, radius)
    x0 = max(0, int(math.floor(center[0] - r)))
    x1 = min(w, int(math.ceil(center[0] + r)) + 1)
    y0 = max(0, int(math.floor(center[1] - r)))
    y1 = min(h, int(math.ceil(center[1] + r)) + 1)
    if x0 >= x1 or y0 >= y1:
        return slice(0, 0), slice(0, 0), np.zeros((0, 0), np.float32)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    dist = np.sqrt((xs + 0.5 - center[0]) ** 2 + (ys + 0.5 - center[1]) ** 2) / r
    hard = min(1.0, max(0.0, hardness))
    cov = np.clip((1.0 - dist) / max(1e-6, 1.0 - hard), 0, 1)
    cov = np.where(dist <= hard, 1.0, cov)
    return slice(y0, y1), slice(x0, x1), cov.astype(np.float32)


def stamp_brush(layer: ImageLayer, from_pt: Tuple[float, float], to_pt: Tuple[float, float],
                color_rgba: Tuple[float, float, float, float], size: float, hardness: float,
                opacity: float, erase: bool = False):
    """Port of BrushPixels.c: paints a straight segment with soft round coverage."""
    px = ensure_pixels(layer)
    h, w = px.shape[:2]
    fx, fy = _doc_to_layer(layer, from_pt)
    tx, ty = _doc_to_layer(layer, to_pt)
    scale = w / max(1e-6, layer.transform.size[0])
    r = size * scale / 2.0
    dist = math.hypot(tx - fx, ty - fy)
    steps = max(1, int(dist / max(1.0, r * 0.25)))
    col = np.array(color_rgba, dtype=np.float32)
    for i in range(steps + 1):
        t = i / steps
        cx, cy = fx + (tx - fx) * t, fy + (ty - fy) * t
        sl_y, sl_x, cov = paint_coverage("round", (cx, cy), r, hardness, (h, w))
        if cov.size == 0:
            continue
        a = cov * opacity
        region = px[sl_y, sl_x]
        if erase:
            region[..., 3] = region[..., 3] * (1 - a)
            region[..., :3] = region[..., :3] * (1 - a)[..., None]
        else:
            src_a = a[..., None]
            out_a = src_a * col[3] + region[..., 3:4] * (1 - src_a)
            rgb = (col[:3] * (src_a * col[3]) +
                   region[..., :3] * region[..., 3:4] * (1 - src_a))
            safe = np.where(out_a > 0, out_a, 1.0)
            region[..., :3] = rgb / safe
            region[..., 3:4] = out_a
        px[sl_y, sl_x] = region


def apply_mask_paint(layer: ImageLayer, from_pt, to_pt, value: float,
                     size: float, hardness: float, opacity: float):
    """LiveLayerMask: paints black/white onto the layer's mask."""
    if layer.mask is None:
        layer.mask = LayerMask(data=np.ones((int(layer.transform.size[1]),
                                             int(layer.transform.size[0])), np.float32))
    m = layer.mask.data
    h, w = m.shape
    fx, fy = _doc_to_layer(layer, from_pt)
    tx, ty = _doc_to_layer(layer, to_pt)
    scale = w / max(1e-6, layer.transform.size[0])
    r = size * scale / 2.0
    steps = max(1, int(math.hypot(tx - fx, ty - fy) / max(1.0, r * 0.25)))
    for i in range(steps + 1):
        t = i / steps
        sl_y, sl_x, cov = paint_coverage("round", (fx + (tx - fx) * t, fy + (ty - fy) * t),
                                         r, hardness, (h, w))
        if cov.size == 0:
            continue
        a = cov * opacity
        m[sl_y, sl_x] = m[sl_y, sl_x] * (1 - a) + value * a


def spot_heal(layer: ImageLayer, at: Tuple[float, float], radius: float):
    """Port of HealPixels.c — content-aware spot healing via surrounding mean."""
    px = ensure_pixels(layer)
    h, w = px.shape[:2]
    lx, ly = _doc_to_layer(layer, at)
    r = radius * (w / max(1e-6, layer.transform.size[0]))
    x0, x1 = int(max(0, lx - r)), int(min(w, lx + r))
    y0, y1 = int(max(0, ly - r)), int(min(h, ly + r))
    if x1 <= x0 or y1 <= y0:
        return
    patch = px[y0:y1, x0:x1]
    ring = px[max(0, y0 - 2):min(h, y1 + 2), max(0, x0 - 2):min(w, x1 + 2)]
    mean = ring.reshape(-1, 4).mean(axis=0)
    grad = np.linspace(0, 1, patch.shape[0])[:, None, None]
    patch[...] = patch * (1 - 0.7) + mean * 0.7


def clone_stamp_sample(layer: ImageLayer, dest: Tuple[float, float], src: Tuple[float, float],
                       source_px: np.ndarray, size: float, opacity: float):
    """CloneStamp.swift: copies pixels from a source point (same or other layer)."""
    px = ensure_pixels(layer)
    h, w = px.shape[:2]
    dx, dy = _doc_to_layer(layer, dest)
    scale = w / max(1e-6, layer.transform.size[0])
    r = size * scale / 2
    sh, sw = source_px.shape[:2]
    sx, sy = src[0] * scale, src[1] * scale
    x0, x1 = int(max(0, dx - r)), int(min(w, dx + r))
    y0, y1 = int(max(0, dy - r)), int(min(h, dy + r))
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            ssx = int(sx + (xx - dx)); ssy = int(sy + (yy - dy))
            if 0 <= ssx < sw and 0 <= ssy < sh:
                a = opacity
                px[yy, xx] = px[yy, xx] * (1 - a) + source_px[ssy, ssx] * a


def blur_smudge(layer: ImageLayer, from_pt, to_pt, size: float, strength: float = 0.4):
    """BlurTool.swift / SmudgeLiquify.swift: local box blur along the drag."""
    px = ensure_pixels(layer)
    h, w = px.shape[:2]
    scale = w / max(1e-6, layer.transform.size[0])
    r = int(size * scale / 2)
    fx, fy = _doc_to_layer(layer, from_pt)
    tx, ty = _doc_to_layer(layer, to_pt)
    steps = max(1, int(math.hypot(tx - fx, ty - fy)))
    for i in range(steps + 1):
        t = i / steps
        cx, cy = int(fx + (tx - fx) * t), int(fy + (ty - fy) * t)
        x0, x1 = max(0, cx - r), min(w, cx + r)
        y0, y1 = max(0, cy - r), min(h, cy + r)
        if x1 <= x0 or y1 <= y0:
            continue
        region = px[y0:y1, x0:x1]
        blurred = region.copy()
        k = max(1, r // 2)
        pad = np.pad(region, ((k, k), (k, k), (0, 0)), mode="edge")
        integral = pad.cumsum(0).cumsum(1)
        H, W = region.shape[:2]
        s = integral[2 * k:2 * k + H, 2 * k:2 * k + W] \
            - integral[0:H, 2 * k:2 * k + W] \
            - integral[2 * k:2 * k + H, 0:W] + integral[0:H, 0:W]
        area = (2 * k + 1) ** 2
        blurred = s / area
        px[y0:y1, x0:x1] = region * (1 - strength) + blurred * strength


# ---------------------------------------------------------------------------
# Magic wand — port of MagicWand.swift (flood fill by color tolerance)
# ---------------------------------------------------------------------------

def magic_wand_selection(composited: np.ndarray, at: Tuple[int, int],
                         tolerance: float = 0.15, contiguous: bool = True) -> np.ndarray:
    """Returns a boolean coverage mask of document size."""
    h, w = composited.shape[:2]
    x, y = int(at[0]), int(at[1])
    if not (0 <= x < w and 0 <= y < h):
        return np.zeros((h, w), bool)
    target = composited[y, x, :3].astype(np.float32)
    diff = np.abs(composited[..., :3] - target).max(axis=-1) <= tolerance
    if not contiguous:
        return diff
    mask = np.zeros((h, w), bool)
    stack = [(y, x)]
    while stack:
        cy, cx = stack.pop()
        if mask[cy, cx] or not diff[cy, cx]:
            continue
        mask[cy, cx] = True
        if cy > 0: stack.append((cy - 1, cx))
        if cy < h - 1: stack.append((cy + 1, cx))
        if cx > 0: stack.append((cy, cx - 1))
        if cx < w - 1: stack.append((cy, cx + 1))
    return mask


def selection_from_coverage(cov: np.ndarray) -> DocumentSelection:
    """Store arbitrary coverage (wand results) as a single polygon-less mask."""
    sel = DocumentSelection()
    sel.shapes = [("coverage", cov)]  # handled specially in coverage()
    return sel





# ---------------------------------------------------------------------------
# Selection edits — port of SelectionEdits.swift
# ---------------------------------------------------------------------------

def expand_contract(sel: DocumentSelection, amount: float, doc: CanvasDocument) -> DocumentSelection:
    cov = sel.coverage(doc.width, doc.height)
    dilated = cov.copy()
    if amount > 0:
        dilated = _binary_dilate(cov, int(amount))
    elif amount < 0:
        dilated = 1 - _binary_dilate(1 - cov, int(-amount))
    new = DocumentSelection(feather=sel.feather)
    new.shapes = [("coverage", (dilated > 0.5).astype(np.float32))]
    return new


def _binary_dilate(cov: np.ndarray, r: int) -> np.ndarray:
    from .scipy_stub import maximum_filter
    return maximum_filter((cov > 0.5).astype(np.uint8), size=2 * r + 1).astype(np.float32)


def feather_selection(sel: DocumentSelection, radius: float) -> DocumentSelection:
    new = DocumentSelection(shapes=list(sel.shapes), antialiased=sel.antialiased,
                            feather=max(0, sel.feather + radius))
    return new


# ---------------------------------------------------------------------------
# Content-aware fill — port of ContentFill.c (simplified PatchMatch-ish)
# ---------------------------------------------------------------------------

def content_fill(px: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Fill masked-in regions by pulling from similar neighbourhoods."""
    out = px.copy()
    h, w = out.shape[:2]
    ys, xs = np.nonzero(coverage > 0.5)
    if len(ys) == 0:
        return out
    valid = (np.ones((h, w)) * (coverage <= 0.5))
    # Coarse-to-fine: multi-scale inpainting with PIL-based blur propagation.
    mask = (coverage > 0.5)
    filled = out.copy()
    filled[mask] = 0
    weight = (~mask).astype(np.float32)
    for _ in range(12):
        for c in range(4):
            img = Image.fromarray(np.clip(filled[..., c], 0, 1).__mul__(255).astype(np.uint8)
                                  if False else (filled[..., c] * 255).astype(np.uint8))
            blurred = np.asarray(img.filter(ImageFilter.GaussianBlur(4)), np.float32) / 255
            imgw = Image.fromarray((weight * 255).astype(np.uint8))
            bw = np.asarray(imgw.filter(ImageFilter.GaussianBlur(4)), np.float32) / 255
            num = blurred
            den = np.maximum(bw, 1e-4)
            upd = num / den
            filled[..., c] = np.where(mask & (bw < 0.98), upd, filled[..., c])
        weight = weight * 0.9 + 0.1 * (~mask).astype(np.float32)
    out[mask] = filled[mask]
    return out


# ---------------------------------------------------------------------------
# Filters — port of Filters.swift / PixelAdjust.swift / NoisePixels/LensPixels
# ---------------------------------------------------------------------------

def apply_gaussian_blur(px: np.ndarray, radius: float) -> np.ndarray:
    out = px.copy()
    for c in range(4):
        img = Image.fromarray((px[..., c] * 255).astype(np.uint8))
        out[..., c] = np.asarray(img.filter(ImageFilter.GaussianBlur(max(0.1, radius))),
                                 np.float32) / 255
    return out


def apply_motion_blur(px: np.ndarray, radius: float, angle_deg: float) -> np.ndarray:
    length = max(1, int(radius))
    ang = math.radians(angle_deg)
    kernel = np.zeros((length * 2 + 1, length * 2 + 1), np.float32)
    mid = length
    for i in range(-mid, mid + 1):
        y = int(round(mid + i * math.sin(ang)))
        x = int(round(mid + i * math.cos(ang)))
        if 0 <= x < kernel.shape[1] and 0 <= y < kernel.shape[0]:
            kernel[y, x] = 1
    kernel /= max(1e-6, kernel.sum())
    out = np.empty_like(px)
    from numpy.lib.stride_tricks import sliding_window_view
    pad = mid
    for c in range(4):
        padded = np.pad(px[..., c], pad, mode="edge")
        view = sliding_window_view(padded, (kernel.shape[0], kernel.shape[1]))
        out[..., c] = np.tensordot(view, kernel, axes=([-2, -1], [0, 1]))
    return out


def apply_noise(px: np.ndarray, amount: float, monochrome: bool = False) -> np.ndarray:
    rng = np.random.default_rng()
    noise = rng.uniform(-amount, amount, px.shape[:2] + (1 if monochrome else 3))
    noise = np.repeat(noise, 3, axis=-1) if monochrome else noise
    out = px.copy()
    out[..., :3] = np.clip(out[..., :3] + noise, 0, 1)
    return out


def apply_lens_correction(px: np.ndarray, distortion: float) -> np.ndarray:
    """LensPixels.c simplified: barrel/pincushion remap."""
    h, w = px.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    nx = (xs - cx) / max(cx, 1)
    ny = (ys - cy) / max(cy, 1)
    r2 = nx * nx + ny * ny
    f = 1 + distortion * r2
    sx = np.clip(cx + nx * f * cx, 0, w - 1)
    sy = np.clip(cy + ny * f * cy, 0, h - 1)
    out = np.empty_like(px)
    x0 = sx.astype(np.int32); y0 = sy.astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
    wx = sx - x0; wy = sy - y0
    for c in range(4):
        a = px[y0, x0, c]; b = px[y0, x1, c]
        cc = px[y1, x0, c]; d = px[y1, x1, c]
        out[..., c] = (a * (1 - wx) * (1 - wy) + b * wx * (1 - wy) +
                       cc * (1 - wx) * wy + d * wx * wy)
    return out


def apply_invert(px: np.ndarray) -> np.ndarray:
    out = px.copy()
    out[..., :3] = 1 - out[..., :3]
    return out


def apply_gradient_tool(px: np.ndarray, start: Tuple[float, float], end: Tuple[float, float],
                        stops: List[Tuple[float, Tuple[int, int, int, int]]]) -> np.ndarray:
    """Gradient tool: linear gradient between two points over the layer."""
    h, w = px.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = end[0] - start[0], end[1] - start[1]
    l2 = dx * dx + dy * dy
    if l2 < 1e-6:
        return px
    t = np.clip(((xs - start[0]) * dx + (ys - start[1]) * dy) / l2, 0, 1)
    pos = np.array([s[0] for s in stops])
    cols = np.array([[c / 255 for c in s[1]] for s in stops], np.float32)
    grad = np.zeros((h, w, 4), np.float32)
    for c in range(4):
        grad[..., c] = np.interp(t, pos, cols[:, c])
    return grad


# ---------------------------------------------------------------------------
# Crop / canvas size / image size — ports of Crop.swift, CanvasSize.swift,
# ImageResizer.swift
# ---------------------------------------------------------------------------

def crop_document(doc: CanvasDocument, rect: Tuple[float, float, float, float]):
    x0, y0 = int(round(rect[0])), int(round(rect[1]))
    x1, y1 = int(round(rect[2])), int(round(rect[3]))
    nw, nh = max(1, x1 - x0), max(1, y1 - y0)
    for lyr in doc.layers:
        lyr.transform.origin = (lyr.transform.origin[0] - x0, lyr.transform.origin[1] - y0)
    doc.width, doc.height = nw, nh
    doc.guides = [
        CanvasGuide(g.vertical, g.position - (x0 if g.vertical else y0))
        for g in doc.guides
        if ((x0 <= g.position <= x1) if g.vertical else (y0 <= g.position <= y1))
    ]


def resize_canvas(doc: CanvasDocument, new_w: int, new_h: int, anchor: str = "topleft"):
    ox, oy = 0, 0
    if anchor == "center":
        ox, oy = (new_w - doc.width) // 2, (new_h - doc.height) // 2
    elif anchor == "bottomright":
        ox, oy = new_w - doc.width, new_h - doc.height
    for lyr in doc.layers:
        lyr.transform.origin = (lyr.transform.origin[0] + ox, lyr.transform.origin[1] + oy)
    doc.width, doc.height = new_w, new_h


def resize_image(doc: CanvasDocument, new_w: int, new_h: int):
    """Image Size: rescale every layer's raster and transform."""
    sx, sy = new_w / doc.width, new_h / doc.height
    for lyr in doc.layers:
        t = lyr.transform
        t.origin = (t.origin[0] * sx, t.origin[1] * sy)
        t.size = (t.size[0] * sx, t.size[1] * sy)
        if lyr.image is not None:
            img = Image.fromarray((np.clip(lyr.image, 0, 1) * 255).astype(np.uint8))
            img = img.resize((max(1, int(lyr.image.shape[1] * sx)),
                              max(1, int(lyr.image.shape[0] * sy))), Image.LANCZOS)
            lyr.image = np.asarray(img, np.float32) / 255
        if lyr.mask is not None and lyr.mask.data is not None:
            m = lyr.mask.data
            img = Image.fromarray((m * 255).astype(np.uint8))
            img = img.resize((max(1, int(m.shape[1] * sx)), max(1, int(m.shape[0] * sy))),
                             Image.BILINEAR)
            lyr.mask.data = np.asarray(img, np.float32) / 255
    doc.width, doc.height = new_w, new_h


# ---------------------------------------------------------------------------
# Layer operations — ports of LayerMerge/LayerGroups/LayerFlip/LayerMask files
# ---------------------------------------------------------------------------

def duplicate_layer(doc: CanvasDocument, layer: ImageLayer) -> ImageLayer:
    dup = copy.deepcopy(layer)
    dup.id = str(uuid.uuid4())
    dup.name = layer.name + " copy"
    dup.transform = layer.transform.copy_with(
        origin=(layer.transform.origin[0] + 16, layer.transform.origin[1] + 16))
    idx = doc.layers.index(layer)
    doc.layers.insert(idx + 1, dup)
    return dup


def delete_layer(doc: CanvasDocument, layer: ImageLayer):
    ids = {layer.id}
    # Deleting a folder deletes its contents (Swift behaviour).
    changed = True
    while changed:
        changed = False
        for l in doc.layers:
            if l.parent_id in ids and l.id not in ids:
                ids.add(l.id)
                changed = True
    doc.layers = [l for l in doc.layers if l.id not in ids]


def merge_down(doc: CanvasDocument, layer: ImageLayer):
    """⌘E Merge Down: bake `layer` into the visible layer beneath it."""
    idx = doc.layers.index(layer)
    below = None
    for j in range(idx - 1, -1, -1):
        cand = doc.layers[j]
        if not cand.is_group and cand.visible:
            below = cand
            break
    if below is None:
        return
    from .compositor import place_layer
    top_r = layer.raster()
    bot_r = ensure_pixels(below) if below.image is None else below.image
    if top_r is None:
        return
    placed_top = place_layer(bot_r.shape[1], bot_r.shape[0], layer, top_r)
    if placed_top is None:
        placed_top = np.zeros_like(bot_r)
    a = placed_top[..., 3:4]
    blended = placed_top[..., :3] * a + bot_r[..., :3] * (1 - a)
    alpha = np.clip(a[..., 0] + bot_r[..., 3] * (1 - a[..., 0]), 0, 1)
    below.image = np.concatenate([blended, alpha[..., None]], axis=-1)
    below.name = below.name
    doc.layers.remove(layer)


def flatten_layers(doc: CanvasDocument):
    """Merge all visible layers into one."""
    from .compositor import composite_document
    rgba = composite_document(doc, preview_selection=False)
    doc.layers = [ImageLayer(name="Background",
                             transform=LayerTransform((0, 0), (doc.width, doc.height)),
                             image=rgba)]


def flip_layer(layer: ImageLayer, horizontal: bool):
    t = layer.transform
    if horizontal:
        t.flip_x = not t.flip_x
    else:
        t.flip_y = not t.flip_y


def flip_canvas(doc: CanvasDocument, horizontal: bool):
    """Flip every layer about the canvas centre."""
    for lyr in doc.layers:
        t = lyr.transform
        cx, cy = t.center
        if horizontal:
            t.flip_x = not t.flip_x
            if lyr.image is not None:
                lyr.image = lyr.image[:, ::-1, :].copy()
        else:
            t.flip_y = not t.flip_y
            if lyr.image is not None:
                lyr.image = lyr.image[::-1, :, :].copy()


def make_group(doc: CanvasDocument, layers: List[ImageLayer], name="Group") -> ImageLayer:
    group = ImageLayer(name=name, transform=LayerTransform((0, 0), doc.size), is_group=True)
    tops = [l for l in layers if l.parent_id is None]
    for l in layers:
        l.parent_id = group.id
    min_idx = min(doc.layers.index(l) for l in layers)
    doc.layers.insert(min_idx, group)
    return group


def add_mask_from_pixels(layer: ImageLayer, white: bool = True):
    h, w = (int(s) for s in layer.transform.size)
    layer.mask = LayerMask(data=np.full((max(1, h), max(1, w)), 1.0 if white else 0.0,
                                        np.float32))


def invert_mask(layer: ImageLayer):
    if layer.mask is not None and layer.mask.data is not None:
        layer.mask.data = 1 - layer.mask.data


def load_pixels_as_selection(doc: CanvasDocument, layer: ImageLayer):
    raster = layer.raster()
    if raster is None:
        return
    from .compositor import place_layer
    placed = place_layer(doc.width, doc.height, layer, raster)
    if placed is None:
        return
    cov = (placed[..., 3] > 0.5).astype(np.float32)
    doc.selection = DocumentSelection(shapes=[("coverage", cov)])


# ---------------------------------------------------------------------------
# Snapshots for the viewport (port of DownsampleCache sharp downsampling)
# ---------------------------------------------------------------------------

def downsample_sharp(rgba: np.ndarray, factor: float) -> np.ndarray:
    """Area-average downsampling keeps detail when zoomed out (DownsampleCache)."""
    if factor >= 1.0:
        return rgba
    h, w = rgba.shape[:2]
    nw, nh = max(1, int(w * factor)), max(1, int(h * factor))
    img = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8))
    img = img.resize((nw, nh), Image.BOX)
    return np.asarray(img, np.float32) / 255
