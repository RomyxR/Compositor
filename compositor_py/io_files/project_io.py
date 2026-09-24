"""Project & image file I/O — Python port of IO/ProjectStore.swift,
ImageImporter.swift, ImageExporter.swift, RawImporter.swift and the PSD
reader (IO/PSD/*).

Project format: a .compositor zip package containing manifest.json plus one
PNG per layer image/mask — the same layout as the Swift ProjectStore.
"""
from __future__ import annotations

import io
import json
import os
import struct
import uuid
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..core.model import (
    AdjustmentKind, BlackWhiteSettings, CanvasDocument, CanvasGuide,
    ColorBalanceSettings, CurvesSettings, ExposureSettings, GradientMapSettings,
    GrainSettings, HueSaturationSettings, ImageLayer, LayerAdjustment,
    LayerBlendMode, LayerEffects, LayerMask, LayerSampling, LayerShape,
    LayerText, LayerTransform, ShapeKind,
)


def rgba_to_png_bytes(rgba: np.ndarray) -> bytes:
    img = Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def png_bytes_to_rgba(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGBA")
    return np.asarray(img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Manifest serialisation (JSON mirrors ProjectManifest / ProjectLayerRecord)
# ---------------------------------------------------------------------------

def transform_to_dict(t: LayerTransform) -> dict:
    return {"origin": [t.origin[0], t.origin[1]], "size": [t.size[0], t.size[1]],
            "rotation": t.rotation, "flipX": t.flip_x, "flipY": t.flip_y,
            "sampling": t.sampling.value}


def transform_from_dict(d: dict) -> LayerTransform:
    sampling_map = {s.value: s for s in LayerSampling}
    return LayerTransform(tuple(d["origin"]), tuple(d["size"]), d.get("rotation", 0.0),
                          d.get("flipX", False), d.get("flipY", False),
                          sampling_map.get(d.get("sampling", "High quality"),
                                           LayerSampling.HIGH))


def adjustment_to_dict(a: LayerAdjustment) -> dict:
    out = {"kind": a.kind.value}
    if a.kind == AdjustmentKind.HSV:
        out["hsv"] = vars(a.hsv)
    elif a.kind == AdjustmentKind.LEVELS:
        out["levels"] = vars(a.levels)
    elif a.kind == AdjustmentKind.CURVES:
        out["curves"] = {"rgb_points": a.curves.rgb_points, "r_points": a.curves.r_points,
                         "g_points": a.curves.g_points, "b_points": a.curves.b_points}
    elif a.kind == AdjustmentKind.EXPOSURE:
        out["exposure"] = vars(a.exposure)
    elif a.kind == AdjustmentKind.GRADIENT_MAP:
        out["gradient_map"] = {"stops": a.gradient_map.stops}
    elif a.kind == AdjustmentKind.GRAIN:
        out["grain"] = vars(a.grain)
    elif a.kind == AdjustmentKind.BLACK_WHITE:
        out["black_white"] = vars(a.black_white)
    elif a.kind == AdjustmentKind.COLOR_BALANCE:
        out["color_balance"] = vars(a.color_balance)
    return out


def adjustment_from_dict(d: dict) -> LayerAdjustment:
    kind = next(k for k in AdjustmentKind if k.value == d["kind"])
    a = LayerAdjustment(kind=kind)
    if "hsv" in d:
        a.hsv = HueSaturationSettings(**d["hsv"])
    if "levels" in d:
        a.levels = type(a.levels)(**d["levels"])
    if "curves" in d:
        c = d["curves"]
        a.curves = CurvesSettings([tuple(p) for p in c.get("rgb_points", [])],
                                  [tuple(p) for p in c.get("r_points", [])],
                                  [tuple(p) for p in c.get("g_points", [])],
                                  [tuple(p) for p in c.get("b_points", [])])
    if "exposure" in d:
        a.exposure = ExposureSettings(**d["exposure"])
    if "gradient_map" in d:
        a.gradient_map = GradientMapSettings([(s[0], tuple(s[1])) for s in d["gradient_map"]["stops"]])
    if "grain" in d:
        a.grain = GrainSettings(**d["grain"])
    if "black_white" in d:
        a.black_white = BlackWhiteSettings(**d["black_white"])
    if "color_balance" in d:
        cb = d["color_balance"]
        a.color_balance = ColorBalanceSettings(tuple(cb["shadows_cmy"]),
                                               tuple(cb["midtones_cmy"]),
                                               tuple(cb["highlights_cmy"]))
    return a


def shape_to_dict(s: LayerShape) -> dict:
    return {"kind": s.kind.value, "fill": list(s.fill_rgba),
            "stroke": list(s.stroke_rgba) if s.stroke_rgba else None,
            "strokeWidth": s.stroke_width, "cornerRadius": s.corner_radius}


def shape_from_dict(d: dict) -> LayerShape:
    kind = next(k for k in ShapeKind if k.value == d["kind"])
    return LayerShape(kind, tuple(d["fill"]),
                      tuple(d["stroke"]) if d.get("stroke") else None,
                      d.get("strokeWidth", 2.0), d.get("cornerRadius", 0.0))


def text_to_dict(t: LayerText) -> dict:
    return {"string": t.string, "fontFamily": t.font_family, "fontSize": t.font_size,
            "color": list(t.color_rgba), "alignment": t.alignment,
            "lineSpacing": t.line_spacing, "letterSpacing": t.letter_spacing}


def text_from_dict(d: dict) -> LayerText:
    return LayerText(d["string"], d.get("fontFamily", "Sans Serif"),
                     d.get("fontSize", 32.0), tuple(d.get("color", [0, 0, 0, 1])),
                     d.get("alignment", "left"), d.get("lineSpacing", 1.2),
                     d.get("letterSpacing", 0.0))


def effects_to_dict(fx: LayerEffects) -> dict:
    return {"stroke": list(fx.stroke_rgba) if fx.stroke_rgba else None,
            "strokeWidth": fx.stroke_width,
            "dropShadowColor": list(fx.drop_shadow_color),
            "dropShadowDX": fx.drop_shadow_dx, "dropShadowDY": fx.drop_shadow_dy,
            "dropShadowBlur": fx.drop_shadow_blur,
            "dropShadowEnabled": fx.drop_shadow_enabled,
            "colorOverlay": list(fx.color_overlay) if fx.color_overlay else None}


def effects_from_dict(d: dict) -> LayerEffects:
    return LayerEffects(tuple(d["stroke"]) if d.get("stroke") else None,
                        d.get("strokeWidth", 2.0),
                        tuple(d.get("dropShadowColor", [0, 0, 0, 0.5])),
                        d.get("dropShadowDX", 4.0), d.get("dropShadowDY", 4.0),
                        d.get("dropShadowBlur", 6.0), d.get("dropShadowEnabled", False),
                        tuple(d["colorOverlay"]) if d.get("colorOverlay") else None)


# ---------------------------------------------------------------------------
# ProjectStore — save/load .compositor packages
# ---------------------------------------------------------------------------

class ProjectStore:
    FORMAT = "com.compositor.project"
    VERSION = 8

    def save(self, doc: CanvasDocument, path: str, active_layer_id: Optional[str] = None):
        manifest = {
            "format": self.FORMAT, "version": self.VERSION, "colorSpace": "sRGB",
            "resolution": doc.resolution, "documentID": doc.id,
            "width": doc.width, "height": doc.height, "activeLayerID": active_layer_id,
            "layers": [], "guides": [{"vertical": g.vertical, "position": g.position}
                                     for g in doc.guides],
        }
        files: Dict[str, bytes] = {}
        for lyr in doc.layers:
            rec = {
                "id": lyr.id, "name": lyr.name, "isVisible": lyr.visible,
                "transform": transform_to_dict(lyr.transform),
                "imageFile": None, "parentID": lyr.parent_id,
                "isGroup": lyr.is_group or None,
                "opacity": lyr.opacity,
                "blendMode": lyr.blend_mode.value if lyr.blend_mode != LayerBlendMode.NORMAL else None,
                "maskFile": None, "maskSourceID": lyr.mask_source_id,
                "adjustment": adjustment_to_dict(lyr.adjustment) if lyr.adjustment else None,
                "maskLinked": None if (lyr.mask is None or lyr.mask.linked) else False,
                "shape": shape_to_dict(lyr.shape) if lyr.shape else None,
                "effects": effects_to_dict(lyr.effects) if lyr.effects else None,
                "text": text_to_dict(lyr.text) if lyr.text else None,
            }
            raster = lyr.raster()
            if raster is not None and not lyr.is_group:
                fname = f"images/{lyr.id}.png"
                files[fname] = rgba_to_png_bytes(raster)
                rec["imageFile"] = fname
            if lyr.mask is not None and lyr.mask.data is not None:
                mname = f"masks/{lyr.id}.png"
                m = (np.clip(lyr.mask.data, 0, 1) * 255).astype(np.uint8)
                buf = io.BytesIO()
                Image.fromarray(m, "L").save(buf, "PNG")
                files[mname] = buf.getvalue()
                rec["maskFile"] = mname
                rec["maskEnabled"] = not lyr.mask.disabled
            manifest["layers"].append(rec)
        tmp = path + ".tmp.zip"
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("manifest.json", json.dumps(manifest, indent=2))
            for name, data in files.items():
                z.writestr(name, data)
        os.replace(tmp, path)

    def load(self, path: str) -> CanvasDocument:
        with zipfile.ZipFile(path) as z:
            manifest = json.loads(z.read("manifest.json"))
            if manifest.get("format") != self.FORMAT:
                raise ValueError("Not a Compositor project")
            if manifest.get("version", 0) > self.VERSION:
                raise ValueError(f"Unsupported project version {manifest['version']}")
            layers: List[ImageLayer] = []
            for rec in manifest["layers"]:
                img = None
                if rec.get("imageFile"):
                    img = png_bytes_to_rgba(z.read(rec["imageFile"]))
                mask = None
                if rec.get("maskFile"):
                    mimg = Image.open(io.BytesIO(z.read(rec["maskFile"]))).convert("L")
                    mask = LayerMask(data=np.asarray(mimg, np.float32) / 255,
                                     disabled=not rec.get("maskEnabled", True),
                                     linked=rec.get("maskLinked", True))
                blend = LayerBlendMode.NORMAL
                if rec.get("blendMode"):
                    blend = next(b for b in LayerBlendMode if b.value == rec["blendMode"])
                lyr = ImageLayer(
                    id=rec["id"], name=rec["name"],
                    transform=transform_from_dict(rec["transform"]),
                    image=img, visible=rec.get("isVisible", True),
                    parent_id=rec.get("parentID"), is_group=bool(rec.get("isGroup")),
                    opacity=rec.get("opacity", 1.0), blend_mode=blend,
                    mask=mask, mask_source_id=rec.get("maskSourceID"),
                    adjustment=adjustment_from_dict(rec["adjustment"]) if rec.get("adjustment") else None,
                    shape=shape_from_dict(rec["shape"]) if rec.get("shape") else None,
                    effects=effects_from_dict(rec["effects"]) if rec.get("effects") else None,
                    text=text_from_dict(rec["text"]) if rec.get("text") else None,
                )
                layers.append(lyr)
            guides = [CanvasGuide(g["vertical"], g["position"])
                      for g in (manifest.get("guides") or [])]
            doc = CanvasDocument(width=manifest["width"], height=manifest["height"],
                                 resolution=manifest.get("resolution", 72.0),
                                 layers=layers, guides=guides)
            doc.id = manifest.get("documentID", doc.id)
            return doc


# ---------------------------------------------------------------------------
# Image import / export — ports of ImageImporter.swift / ImageExporter.swift
# ---------------------------------------------------------------------------

IMPORT_TYPES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".gif",
                ".heic", ".psd", ".psb"}


def import_image(path: str) -> Tuple[np.ndarray, str]:
    """Returns (RGBA float array, display name). HEIC needs pillow-heif; PSD
    goes through our own reader."""
    ext = os.path.splitext(path)[1].lower()
    name = os.path.splitext(os.path.basename(path))[0]
    if ext in (".psd", ".psb"):
        doc = read_psd(path)
        from ..core.compositor import composite_document
        return composite_document(doc, preview_selection=False), name
    img = Image.open(path)
    if ext == ".heic":
        try:
            import pillow_heif  # optional dependency
            pillow_heif.register_heif_opener()
            img = Image.open(path)
        except ImportError:
            raise RuntimeError("HEIC support requires the 'pillow-heif' package")
    img = img.convert("RGBA")
    return np.asarray(img, dtype=np.float32) / 255.0, name


def export_png(doc, path: str):
    from ..core.compositor import composite_document
    rgba = composite_document(doc, preview_selection=False)
    Image.fromarray((np.clip(rgba, 0, 1) * 255).astype(np.uint8), "RGBA").save(path, "PNG")


def export_jpeg(doc, path: str, quality: int = 90, background=(1.0, 1.0, 1.0)):
    from ..core.compositor import flatten_to_rgb
    rgb = flatten_to_rgb(doc, background)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB").save(
        path, "JPEG", quality=quality)


def copy_merged_rgba(doc) -> np.ndarray:
    from ..core.compositor import composite_document
    return composite_document(doc, preview_selection=False)


# ---------------------------------------------------------------------------
# Minimal PSD reader — port of IO/PSD/PSDReader.swift (8-bit RGB only)
# ---------------------------------------------------------------------------

_PSD_BLEND_MAP = {
    b"pass": LayerBlendMode.NORMAL, b"norm": LayerBlendMode.NORMAL,
    b"dark": LayerBlendMode.DARKEN, b"mul ": LayerBlendMode.MULTIPLY,
    b"idiv": LayerBlendMode.COLOR_BURN, b"lbrn": LayerBlendMode.LINEAR_BURN,
    b"lite": LayerBlendMode.LIGHTEN, b"scrn": LayerBlendMode.SCREEN,
    b"idod": LayerBlendMode.COLOR_DODGE, b"lddg": LayerBlendMode.LINEAR_DODGE,
    b"over": LayerBlendMode.OVERLAY, b"sLit": LayerBlendMode.SOFT_LIGHT,
    b"hLit": LayerBlendMode.HARD_LIGHT, b"diff": LayerBlendMode.DIFFERENCE,
    b"smud": LayerBlendMode.EXCLUSION, b"hdis": LayerBlendMode.HUE,
    b"sat ": LayerBlendMode.SATURATION, b"colr": LayerBlendMode.COLOR,
    b"lum ": LayerBlendMode.LUMINOSITY,
}


def read_psd(path: str) -> CanvasDocument:
    """Read an 8-bit RGB PSD into a layered CanvasDocument. Folders, masks and
    a subset of blend modes stay editable; other vector data becomes pixels
    (mirrors PSDDocumentBuilder's conversion report approach)."""
    data = open(path, "rb").read()
    pos = 0

    def u16():
        nonlocal pos
        v = struct.unpack_from(">H", data, pos)[0]; pos += 2; return v

    def u32():
        nonlocal pos
        v = struct.unpack_from(">I", data, pos)[0]; pos += 4; return v

    def i16():
        nonlocal pos
        v = struct.unpack_from(">h", data, pos)[0]; pos += 2; return v

    def i32():
        nonlocal pos
        v = struct.unpack_from(">i", data, pos)[0]; pos += 4; return v

    def u8():
        nonlocal pos
        v = data[pos]; pos += 1; return v

    def sig(n):
        nonlocal pos
        v = data[pos:pos + n]; pos += n; return v

    if sig(4) != b"8BPS":
        raise ValueError("Not a PSD file")
    version = u16(); u16(); sig(4)
    channels = u16(); height = u32(); width = u32(); depth = u16(); color_mode = u16()
    if version != 1:
        raise ValueError("PSB not supported (8-bit RGB PSD only)")
    if depth != 8 or color_mode not in (3, 1):
        raise ValueError("Only 8-bit RGB PSD files are supported")
    cmd_len = u32(); pos += cmd_len          # colour mode data
    ir_len = u32(); pos += ir_len            # image resources
    lmi_len = u32()
    layers: List[ImageLayer] = []
    if lmi_len >= 4:
        lmi_end = pos + lmi_len
        tag = data[pos:pos + 4]
        if tag in (b"8BIM", b"Lr16") and lmi_len >= 16:
            # Legacy layer info then extra blocks — parse legacy part only.
            pass
        count = i16()
        abs_count = abs(count)
        raw_layers = []
        for _ in range(abs_count):
            top, left, bottom, right = i32(), i32(), i32(), i32()
            ch_count = u16()
            chan_info = []
            for _c in range(ch_count):
                cid = i16(); clen = u32()
                chan_info.append((cid, clen))
            blend = sig(4)
            opacity = u8(); clipping = u8(); flags = u8(); u8(); u8(); u8()
            extras_len = u32()
            extras_start = pos
            name = ""
            nl = u8()
            padded = (nl + 4) & ~3
            name = data[pos:pos + nl].decode("mac-roman", errors="replace")
            pos = extras_start + padded - 1
            channel_data = []
            for cid, clen in chan_info:
                start = pos
                if clen >= 2:
                    comp = struct.unpack_from(">H", data, pos)[0]
                    body = data[start + 2:start + clen]
                    channel_data.append((cid, comp, body))
                pos = start + clen
                if clen % 2:
                    pos += 1
            raw_layers.append(dict(top=top, left=left, bottom=bottom, right=right,
                                   blend=blend, opacity=opacity, clipping=clipping,
                                   flags=flags, name=name, channels=channel_data))
        pos = lmi_end
        for rec in reversed(raw_layers):
            lw = rec["right"] - rec["left"]; lh = rec["bottom"] - rec["top"]
            if lw <= 0 or lh <= 0:
                continue
            chans = {}
            for cid, comp, body in rec["channels"]:
                try:
                    chans[cid] = _decode_channel(body, comp, lh, lw)
                except Exception:
                    continue
            r = chans.get(0); g = chans.get(1); b = chans.get(2)
            a = chans.get(-1)
            if r is None or g is None or b is None:
                continue
            rgba = np.zeros((lh, lw, 4), np.float32)
            rgba[..., 0] = r; rgba[..., 1] = g; rgba[..., 2] = b
            rgba[..., 3] = a if a is not None else 1.0
            hidden = bool(rec["flags"] & 2)
            is_folder = bool(rec["flags"] & 8)
            blend_mode = _PSD_BLEND_MAP.get(rec["blend"], LayerBlendMode.NORMAL)
            lyr = ImageLayer(
                name=rec["name"] or "Layer",
                transform=LayerTransform((rec["left"], rec["top"]), (lw, lh)),
                image=None if is_folder else rgba,
                visible=not hidden, is_group=is_folder,
                opacity=rec["opacity"] / 255.0, blend_mode=blend_mode)
            if rec["clipping"] == 1 and layers:
                lyr.mask_source_id = layers[-1].id
            layers.append(lyr)
    return CanvasDocument(width=width, height=height, layers=list(reversed(layers)))


def _decode_channel(body: bytes, compression: int, lh: int, lw: int) -> np.ndarray:
    if compression == 0:
        arr = np.frombuffer(body[:lh * lw], dtype=np.uint8).reshape(lh, lw)
        return arr.astype(np.float32) / 255.0
    if compression == 1:
        counts = np.frombuffer(body[:lh * 2], dtype=">u2")
        pos = lh * 2
        rows = []
        for n in counts:
            rows.append(_packbits(body[pos:pos + n])); pos += n
        arr = np.array(rows, dtype=np.uint8)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < lw:
            arr = np.pad(arr, ((0, 0), (0, lw - arr.shape[1])))
        return arr[:, :lw].astype(np.float32) / 255.0
    return np.zeros((lh, lw), np.float32)


def _packbits(chunk: bytes) -> np.ndarray:
    out = bytearray()
    i = 0
    while i < len(chunk):
        n = chunk[i]
        if n < 128:
            out += chunk[i + 1:i + 1 + n + 1]; i += n + 2
        elif n > 128:
            out += bytes([chunk[i + 1]]) * (257 - n); i += 2
        else:
            i += 1
    return np.frombuffer(bytes(out), dtype=np.uint8)
