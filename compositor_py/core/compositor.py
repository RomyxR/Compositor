"""Layer compositing — Python port of Rendering/LayerRenderer.swift,
SeparableBlend.swift and RasterSnapshot.swift.

Composites a CanvasDocument's layers bottom-to-top into an RGBA float array,
honouring transforms (position/scale/rotation/flip), per-layer and folder
opacity, blend modes, masks, clipping masks, adjustment layers and layer
effects (stroke / drop shadow).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from .model import (
    CanvasDocument, DocumentSelection, ImageLayer, LayerBlendMode, LayerSampling,
    blend_pixels, effective_opacity, gaussian_blur_float, hierarchy_entries,
)

SAMPLING_FILTER = {
    LayerSampling.NEAREST: Image.NEAREST,
    LayerSampling.SMOOTH: Image.BILINEAR,
    LayerSampling.HIGH: Image.LANCZOS,
}


def warp_layer(layer_raster: np.ndarray, target_w: int, target_h: int,
               sampling: LayerSampling) -> np.ndarray:
    """Resize a layer raster to its placed size in document space."""
    if target_w < 1 or target_h < 1:
        return np.zeros((max(1, target_h), max(1, target_w), 4), dtype=np.float32)
    src_h, src_w = layer_raster.shape[:2]
    if (src_w, src_h) == (target_w, target_h):
        return layer_raster.copy()
    img = Image.fromarray(np.clip(layer_raster * 255, 0, 255).astype(np.uint8))
    img = img.resize((target_w, target_h), SAMPLING_FILTER[sampling])
    return np.asarray(img, dtype=np.float32) / 255.0


def _rotate_rgba(arr: np.ndarray, degrees: float, sampling: LayerSampling) -> np.ndarray:
    """Rotate about center, expanding the canvas so nothing is clipped."""
    resample = SAMPLING_FILTER[sampling]
    if resample == Image.NEAREST:
        resample = Image.NEAREST
    else:
        # Pillow's rotate supports BICUBIC/BILINEAR/NEAREST; map LANCZOS->BICUBIC.
        resample = {Image.LANCZOS: Image.BICUBIC}.get(resample, resample)
    img = Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))
    out = img.rotate(-degrees, expand=True, resample=resample, fillcolor=(0, 0, 0, 0))
    return np.asarray(out, dtype=np.float32) / 255.0


def place_layer(doc_w: int, doc_h: int, layer: ImageLayer,
                raster: np.ndarray) -> Optional[np.ndarray]:
    """Produce a full-document RGBA array with `raster` placed per layer.transform."""
    t = layer.transform
    sx, sy = 1.0, 1.0
    if t.flip_x:
        raster = raster[:, ::-1, :].copy()
    if t.flip_y:
        raster = raster[::-1, :, :].copy()
    # Target unrotated size in whole document pixels.
    tw, th = max(1, int(round(t.size[0]))), max(1, int(round(t.size[1])))
    placed = warp_layer(raster, tw, th, t.sampling)
    if abs(math.remainder(t.rotation, 360.0)) > 1e-6:
        placed = _rotate_rgba(placed, t.rotation, t.sampling)
        rh, rw = placed.shape[:2]
        # Keep the rotated result centered on the transform center.
        cx = t.origin[0] + t.size[0] / 2
        cy = t.origin[1] + t.size[1] / 2
        ox = int(round(cx - rw / 2))
        oy = int(round(cy - rh / 2))
    else:
        ox = int(round(t.origin[0]))
        oy = int(round(t.origin[1]))
    # Clip against the document.
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(doc_w, ox + placed.shape[1]), min(doc_h, oy + placed.shape[0])
    if x0 >= x1 or y0 >= y1:
        return None
    sub = placed[y0 - oy:y1 - oy, x0 - ox:x1 - ox, :]
    out = np.zeros((doc_h, doc_w, 4), dtype=np.float32)
    out[y0:y1, x0:x1, :] = sub
    return out


def apply_effects(doc_w: int, doc_h: int, placed: np.ndarray, layer: ImageLayer) -> np.ndarray:
    """Port of LayerEffectsSurface: stroke + drop shadow around the layer."""
    fx = layer.effects
    if fx is None:
        return placed
    alpha = placed[..., 3]
    result = placed.copy()
    if fx.drop_shadow_enabled:
        silhouette = np.zeros_like(alpha)
        silhouette[...] = np.clip(alpha + (1 - alpha) * 0, 0, 1)
        mask_img = Image.fromarray((np.clip(silhouette, 0, 1) * 255).astype(np.uint8))
        blur_px = max(0.5, fx.drop_shadow_blur)
        mask_img = mask_img.filter(Image.ImageFilter.GaussianBlur(radius=blur_px / 2)
                                   if False else _gauss(blur_px))
        sh_alpha = np.asarray(mask_img, dtype=np.float32) / 255.0
        sh_alpha = np.roll(sh_alpha, int(round(fx.drop_shadow_dy)), axis=0)
        sh_alpha = np.roll(sh_alpha, int(round(fx.drop_shadow_dx)), axis=1)
        sh_alpha *= fx.drop_shadow_color[3]
        col = np.array(fx.drop_shadow_color[:3], dtype=np.float32)
        inv = 1 - sh_alpha[..., None]
        rgb = result[..., :3] * inv + col * sh_alpha[..., None]
        a = result[..., 3] + sh_alpha * (1 - result[..., 3])
        result = np.concatenate([rgb, a[..., None]], axis=-1)
    if fx.stroke_rgba is not None and fx.stroke_width > 0:
        dilated = _dilate(alpha, fx.stroke_width)
        ring = np.clip(dilated - alpha, 0, 1) * fx.stroke_rgba[3]
        col = np.array(fx.stroke_rgba[:3], dtype=np.float32)
        inv = 1 - ring[..., None]
        rgb = result[..., :3] * inv + col * ring[..., None]
        a = result[..., 3] + ring * (1 - result[..., 3])
        result = np.concatenate([rgb, a[..., None]], axis=-1)
    if fx.color_overlay is not None:
        col = np.array(fx.color_overlay[:3], dtype=np.float32)
        wgt = fx.color_overlay[3]
        result[..., :3] = result[..., :3] * (1 - wgt) + col * wgt
    return result


def _gauss(sigma: float):
    from PIL import ImageFilter
    return ImageFilter.GaussianBlur(radius=max(0.1, sigma / 2))


def _dilate(alpha: np.ndarray, radius: float) -> np.ndarray:
    r = max(1, int(round(radius)))
    pad = np.pad(alpha, r, mode="constant")
    out = np.zeros_like(alpha)
    h, w = alpha.shape
    for dy in range(0, 2 * r + 1, max(1, r // 2)):
        for dx in range(0, 2 * r + 1, max(1, r // 2)):
            window = pad[dy:dy + h, dx:dx + w]
            out = np.maximum(out, window)
    return out


def composite_document(doc: CanvasDocument,
                       transparent_base: bool = True,
                       preview_selection: bool = True) -> np.ndarray:
    """Render the document to an RGBA float array (doc.height, doc.width, 4).

    Port of LayerRenderer.render: folders are pass-through; each layer blends
    onto everything below it, multiplied by its effective (folder-aware)
    opacity, masked, optionally clipped to the layer beneath.
    """
    base = np.zeros((doc.height, doc.width, 4), dtype=np.float32)
    if transparent_base:
        # Checkerboard-free: keep alpha; exporters flatten themselves.
        pass
    by_id: Dict[str, ImageLayer] = {l.id: l for l in doc.layers}
    entries = hierarchy_entries(doc.layers, top_first=False)

    clip_stack: List[Optional[ImageLayer]] = []  # last drawn non-group layer chain
    prev_placed: Optional[np.ndarray] = None
    prev_layer: Optional[ImageLayer] = None

    sel_cov: Optional[np.ndarray] = None
    if preview_selection and doc.selection is not None and not doc.selection.is_empty:
        sel_cov = doc.selection.coverage(doc.width, doc.height)

    for layer, depth, visible in entries:
        if not visible:
            continue
        if layer.is_group:
            prev_placed = None
            prev_layer = None
            continue
        if layer.adjustment is not None:
            # Adjustment layers affect everything below (unless clipped).
            if layer.mask_source_id is None and prev_layer is not None and \
                    prev_layer.mask_source_id == layer.id:
                pass  # clipped adjustments handled below via mask_source_id
            target_below = base
            if layer.mask_source_id is not None:
                # Clipped adjustment: affects only the layer directly below.
                continue  # simplified: clipping adjustments applied at merge time
            blended = layer.adjustment.apply(base)
            cov = np.ones((doc.height, doc.width), dtype=np.float32)
            if layer.mask is not None and not layer.mask.disabled:
                cov = layer.mask.coverage_for(doc.height, doc.width)
            cov = cov * effective_opacity(layer, by_id)
            if sel_cov is not None:
                cov = cov * sel_cov
            base = base * (1 - cov[..., None]) + blended * cov[..., None]
            base[..., 3] = np.maximum(base[..., 3], (blended[..., 3] * cov))
            prev_placed = None
            prev_layer = layer
            continue

        raster = layer.raster()
        if raster is None:
            # Blank layer with no pixels yet — nothing to draw.
            prev_placed = None
            prev_layer = layer
            continue
        placed = place_layer(doc.width, doc.height, layer, raster)
        if placed is None:
            prev_placed = None
            prev_layer = layer
            continue
        placed = apply_effects(doc.width, doc.height, placed, layer)

        alpha = placed[..., 3].copy()
        if layer.mask is not None and not layer.mask.disabled:
            alpha = alpha * layer.mask.coverage_for(doc.height, doc.width)
        if layer.mask_source_id is not None and prev_placed is not None:
            # Clipping mask: clip to the transparency of the layer below.
            alpha = alpha * prev_placed[..., 3]
        alpha = alpha * effective_opacity(layer, by_id)
        if sel_cov is not None:
            alpha = alpha * sel_cov

        src_rgb = placed[..., :3]
        base_rgb = base[..., :3]
        blended_rgb = blend_pixels(layer.blend_mode, base_rgb, src_rgb)
        a_src = alpha[..., None]
        a_dst = base[..., 3:4]
        out_a = a_src + a_dst * (1 - a_src)
        # Photoshop-style source-over with blending: where there is backdrop,
        # the blended result contributes; over transparency the raw source shows.
        has_backdrop = a_dst > 0
        contrib = np.where(has_backdrop, blended_rgb, src_rgb)
        out_rgb = contrib * a_src + base_rgb * a_dst * (1 - a_src)
        safe_a = np.where(out_a > 0, out_a, 1.0)
        out_rgb = out_rgb / safe_a
        base = np.concatenate([np.clip(out_rgb, 0, 1), np.clip(out_a, 0, 1)], axis=-1)
        prev_placed = placed
        prev_layer = layer

    return np.clip(base, 0, 1)


def flatten_to_rgb(doc: CanvasDocument, background=(1, 1, 1)) -> np.ndarray:
    """Composite then flatten over a solid background; returns RGB float."""
    rgba = composite_document(doc)
    bg = np.array(background, dtype=np.float32)
    a = rgba[..., 3:4]
    return rgba[..., :3] * a + bg * (1 - a)


def checkerboard_preview(rgba: np.ndarray, cell: int = 8) -> np.ndarray:
    """RGB preview with a checkerboard behind transparency (canvas display)."""
    h, w = rgba.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    light = ((xx // cell + yy // cell) % 2 == 0)
    c1 = np.where(light[..., None], 0.85, 0.70) * np.ones((1, 1, 3), np.float32)
    a = rgba[..., 3:4]
    return rgba[..., :3] * a + c1 * (1 - a)
