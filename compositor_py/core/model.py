"""Core document model — Python port of Compositor's Swift data types.

Ports: EditorSession.swift (ImageLayer, CanvasDocument, NavigationTool),
LayerAppearance.swift (LayerBlendMode), LayerTransform.swift (LayerTransform,
LayerSampling), LayerGroups.swift (LayerHierarchy, LayerOpacity),
LayerAdjustment.swift (AdjustmentKind, LayerAdjustment), Selection.swift
(DocumentSelection geometry).

Images are float32 RGBA arrays in [0, 1] with shape (height, width, 4),
y-down, matching the Swift rendering pipeline's premultiplied-free model.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Geometry helpers (replaces CoreGraphics CGPoint/CGSize/CGAffineTransform)
# ---------------------------------------------------------------------------

Point = Tuple[float, float]
Size = Tuple[float, float]


def deg2rad(d: float) -> float:
    return d * math.pi / 180.0


# ---------------------------------------------------------------------------
# Blend modes — port of LayerBlendMode
# ---------------------------------------------------------------------------

class LayerBlendMode(Enum):
    NORMAL = "Normal"
    DARKEN = "Darken"
    MULTIPLY = "Multiply"
    COLOR_BURN = "Color Burn"
    LINEAR_BURN = "Linear Burn"
    LIGHTEN = "Lighten"
    SCREEN = "Screen"
    COLOR_DODGE = "Color Dodge"
    LINEAR_DODGE = "Linear Dodge (Add)"
    OVERLAY = "Overlay"
    SOFT_LIGHT = "Soft Light"
    HARD_LIGHT = "Hard Light"
    VIVID_LIGHT = "Vivid Light"
    LINEAR_LIGHT = "Linear Light"
    PIN_LIGHT = "Pin Light"
    HARD_MIX = "Hard Mix"
    DIFFERENCE = "Difference"
    EXCLUSION = "Exclusion"
    SUBTRACT = "Subtract"
    DIVIDE = "Divide"
    HUE = "Hue"
    SATURATION = "Saturation"
    COLOR = "Color"
    LUMINOSITY = "Luminosity"

    @classmethod
    def all_modes(cls) -> List["LayerBlendMode"]:
        return list(cls)

    @property
    def is_separable(self) -> bool:
        """Component (HSL) modes need the base color converted to HSL."""
        return self not in (LayerBlendMode.HUE, LayerBlendMode.SATURATION,
                            LayerBlendMode.COLOR, LayerBlendMode.LUMINOSITY)


BLEND_MODE_GROUPS = [
    [LayerBlendMode.NORMAL],
    [LayerBlendMode.DARKEN, LayerBlendMode.MULTIPLY, LayerBlendMode.COLOR_BURN,
     LayerBlendMode.LINEAR_BURN],
    [LayerBlendMode.LIGHTEN, LayerBlendMode.SCREEN, LayerBlendMode.COLOR_DODGE,
     LayerBlendMode.LINEAR_DODGE],
    [LayerBlendMode.OVERLAY, LayerBlendMode.SOFT_LIGHT, LayerBlendMode.HARD_LIGHT,
     LayerBlendMode.VIVID_LIGHT, LayerBlendMode.LINEAR_LIGHT, LayerBlendMode.PIN_LIGHT,
     LayerBlendMode.HARD_MIX],
    [LayerBlendMode.DIFFERENCE, LayerBlendMode.EXCLUSION, LayerBlendMode.SUBTRACT,
     LayerBlendMode.DIVIDE],
    [LayerBlendMode.HUE, LayerBlendMode.SATURATION, LayerBlendMode.COLOR,
     LayerBlendMode.LUMINOSITY],
]


def _clip01(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 1.0)


def _sep_blend(mode: LayerBlendMode, b: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Separable blend of source s over base b, channel-wise (Photoshop formulas)."""
    if mode == LayerBlendMode.MULTIPLY:
        return b * s
    if mode == LayerBlendMode.SCREEN:
        return 1 - (1 - b) * (1 - s)
    if mode == LayerBlendMode.OVERLAY:
        return np.where(b <= 0.5, 2 * b * s, 1 - 2 * (1 - b) * (1 - s))
    if mode == LayerBlendMode.DARKEN:
        return np.minimum(b, s)
    if mode == LayerBlendMode.LIGHTEN:
        return np.maximum(b, s)
    if mode == LayerBlendMode.COLOR_BURN:
        return np.where(b <= 0, 0.0, _clip01(1 - (1 - s) / np.maximum(b, 1e-9)))
    if mode == LayerBlendMode.COLOR_DODGE:
        return np.where(b >= 1, 1.0, _clip01(s / np.maximum(1 - b, 1e-9)))
    if mode == LayerBlendMode.LINEAR_BURN:
        return _clip01(b + s - 1)
    if mode == LayerBlendMode.LINEAR_DODGE:
        return _clip01(b + s)
    if mode == LayerBlendMode.HARD_LIGHT:
        return np.where(s <= 0.5, 2 * b * s, 1 - 2 * (1 - b) * (1 - s))
    if mode == LayerBlendMode.SOFT_LIGHT:
        # Photoshop's softer formula (w3c soft-light).
        r = np.where(b <= 0.25, ((16 * b - 12) * b + 4) * b, np.sqrt(np.maximum(b, 0)))
        return np.where(s <= 0.5, b - (1 - 2 * s) * b * (1 - b), b + (2 * s - 1) * (r - b))
    if mode == LayerBlendMode.VIVID_LIGHT:
        d = np.where(s <= 0.5, _clip01(b / np.maximum(2 * s, 1e-9)),
                     np.where(s >= 1, 1.0, _clip01(1 - (1 - b) / np.maximum(2 * (1 - s), 1e-9))))
        return d
    if mode == LayerBlendMode.LINEAR_LIGHT:
        return _clip01(b + 2 * s - 1)
    if mode == LayerBlendMode.PIN_LIGHT:
        return np.where(s <= 0.5, np.minimum(b, 2 * s), np.maximum(b, 2 * s - 1))
    if mode == LayerBlendMode.HARD_MIX:
        return np.where(_clip01(b + 2 * s - 1) < 0.5, 0.0, 1.0)
    if mode == LayerBlendMode.DIFFERENCE:
        return np.abs(b - s)
    if mode == LayerBlendMode.EXCLUSION:
        return b + s - 2 * b * s
    if mode == LayerBlendMode.SUBTRACT:
        return _clip01(b - s)
    if mode == LayerBlendMode.DIVIDE:
        return _clip01(b / np.maximum(s, 1e-9))
    raise ValueError(f"not separable: {mode}")


def _rgb_hsl(rgb: np.ndarray) -> np.ndarray:
    """RGB (N.., 3) -> HSL, h in [0,1). Vectorised."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    l = (mx + mn) / 2
    d = mx - mn
    s = np.where(l <= 0.5, np.where(mx > 0, d / np.maximum(mx, 1e-9), 0),
                 np.where(mx < 1, d / np.maximum(1 - mn, 1e-9), 0))
    h = np.zeros_like(mx)
    safe = d > 1e-9
    hr = np.where(safe, (((g - b) / np.where(safe, d, 1)) % 6) / 6, 0)
    hg = np.where(safe, (((b - r) / np.where(safe, d, 1)) + 2) / 6, 0)
    hb = np.where(safe, (((r - g) / np.where(safe, d, 1)) + 4) / 6, 0)
    h = np.where(mx == r, hr, np.where(mx == g, hg, hb))
    return np.stack([h, s, l], axis=-1)


def _hsl_rgb(hsl: np.ndarray) -> np.ndarray:
    h, s, l = hsl[..., 0] % 1.0, hsl[..., 1], hsl[..., 2]

    def f(n):
        k = (n + h * 12) % 12
        return l - s * np.minimum(l, 1 - l) * np.maximum(-1, np.minimum(k - 3, np.minimum(9 - k, 1)))

    return np.stack([f(0), f(8), f(4)], axis=-1)


def _lum(rgb: np.ndarray) -> np.ndarray:
    return 0.3 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]


def _comp_blend(mode: LayerBlendMode, b: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Non-separable (component) blend modes operating in HSL space."""
    bh = _rgb_hsl(b)
    sh = _rgb_hsl(s)
    out = bh.copy()
    if mode == LayerBlendMode.HUE:
        out[..., 0] = sh[..., 0]
    elif mode == LayerBlendMode.SATURATION:
        out[..., 1] = sh[..., 1]
    elif mode == LayerBlendMode.COLOR:
        out[..., 0] = sh[..., 0]
        out[..., 1] = sh[..., 1]
    elif mode == LayerBlendMode.LUMINOSITY:
        # Operate on luminance directly: shift base luminance to source's.
        result = b + (_lum(s) - _lum(b))[..., None]
        return _clip01(result)
    return _clip01(_hsl_rgb(out))


def blend_pixels(mode: LayerBlendMode, base: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Blend src RGB over base RGB (both float (H,W,3)); returns blended RGB."""
    if mode == LayerBlendMode.NORMAL:
        return src
    if mode.is_separable:
        return _sep_blend(mode, base, src)
    return _comp_blend(mode, base, src)


# ---------------------------------------------------------------------------
# Sampling quality — port of LayerSampling
# ---------------------------------------------------------------------------

class LayerSampling(Enum):
    NEAREST = "Nearest"
    SMOOTH = "Smooth"
    HIGH = "High quality"


# ---------------------------------------------------------------------------
# Layer transform — port of LayerTransform
# ---------------------------------------------------------------------------

@dataclass
class LayerTransform:
    """Unrotated bounds in document pixels; rotation is clockwise around center."""
    origin: Point = (0.0, 0.0)
    size: Size = (1.0, 1.0)
    rotation: float = 0.0
    flip_x: bool = False
    flip_y: bool = False
    sampling: LayerSampling = LayerSampling.HIGH

    @property
    def center(self) -> Point:
        return (self.origin[0] + self.size[0] / 2, self.origin[1] + self.size[1] / 2)

    @property
    def radians(self) -> float:
        return deg2rad(math.remainder(self.rotation, 360.0))

    @property
    def is_valid(self) -> bool:
        vals = [self.origin[0], self.origin[1], self.size[0], self.size[1], self.rotation]
        return all(math.isfinite(v) for v in vals) \
            and 1 <= self.size[0] <= 300_000 and 1 <= self.size[1] <= 300_000 \
            and abs(self.origin[0]) <= 1_000_000 and abs(self.origin[1]) <= 1_000_000

    def point(self, unit: Point) -> Point:
        """Map a unit-square point (0..1, y down) into document space."""
        x = (unit[0] - 0.5) * self.size[0]
        y = (unit[1] - 0.5) * self.size[1]
        c, s = math.cos(self.radians), math.sin(self.radians)
        cx, cy = self.center
        return (cx + x * c - y * s, cy + x * s + y * c)

    def corners(self) -> List[Point]:
        return [self.point(u) for u in [(0, 0), (1, 0), (1, 1), (0, 1)]]

    def contains(self, p: Point) -> bool:
        cx, cy = self.center
        x, y = p[0] - cx, p[1] - cy
        c, s = math.cos(self.radians), math.sin(self.radians)
        return abs(x * c + y * s) <= self.size[0] / 2 and abs(-x * s + y * c) <= self.size[1] / 2

    def scale_percent(self, pixel_size: Size) -> float:
        return self.size[0] / max(1.0, pixel_size[0]) * 100.0

    def scaled_to_percent(self, percent: float, pixel_size: Size) -> "LayerTransform":
        size = (pixel_size[0] * percent / 100.0, pixel_size[1] * percent / 100.0)
        cx, cy = self.center
        return LayerTransform((cx - size[0] / 2, cy - size[1] / 2), size,
                              self.rotation, self.flip_x, self.flip_y, self.sampling)

    def rounded(self) -> "LayerTransform":
        return LayerTransform((round(self.origin[0]), round(self.origin[1])),
                              (max(1, round(self.size[0])), max(1, round(self.size[1]))),
                              round(self.rotation), self.flip_x, self.flip_y, self.sampling)

    def transformed_point_from_document(self, p: Point) -> Point:
        """Inverse of point(): document coords -> unit coords of the layer."""
        cx, cy = self.center
        x, y = p[0] - cx, p[1] - cy
        c, s = math.cos(-self.radians), math.sin(-self.radians)
        ux = (x * c - y * s) / self.size[0] + 0.5
        uy = (x * s + y * c) / self.size[1] + 0.5
        return (ux, uy)

    def copy_with(self, **kw) -> "LayerTransform":
        t = LayerTransform(self.origin, self.size, self.rotation,
                           self.flip_x, self.flip_y, self.sampling)
        for k, v in kw.items():
            setattr(t, k, v)
        return t


# ---------------------------------------------------------------------------
# Adjustment layers — port of LayerAdjustment / AdjustmentKind
# ---------------------------------------------------------------------------

class AdjustmentKind(Enum):
    HSV = "Hue/Saturation"
    LEVELS = "Levels"
    CURVES = "Curves"
    EXPOSURE = "Exposure"
    GRADIENT_MAP = "Gradient Map"
    GRAIN = "Grain"
    INVERT = "Invert"
    BLACK_WHITE = "Black & White"
    COLOR_BALANCE = "Color Balance"

    @property
    def is_editable(self) -> bool:
        return self != AdjustmentKind.INVERT


@dataclass
class HueSaturationSettings:
    hue: float = 0.0          # degrees, -180..180
    saturation: float = 0.0   # -100..100
    lightness: float = 0.0    # -100..100
    colorize: bool = False


@dataclass
class LevelsSettings:
    black: float = 0.0     # 0..1 input black
    white: float = 1.0     # 0..1 input white
    gamma: float = 1.0     # 0.01..99
    output_min: float = 0.0
    output_max: float = 1.0

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        a = (rgb - self.black) / max(1e-6, self.white - self.black)
        a = _clip01(a)
        a = np.power(a, 1.0 / max(0.01, self.gamma))
        return _clip01(a * (self.output_max - self.output_min) + self.output_min)


@dataclass
class CurvesSettings:
    """Control points per channel: list of (input 0..1, output 0..1), monotone."""
    rgb_points: List[Tuple[float, float]] = field(default_factory=list)
    r_points: List[Tuple[float, float]] = field(default_factory=list)
    g_points: List[Tuple[float, float]] = field(default_factory=list)
    b_points: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def is_identity(self) -> bool:
        return not (self.rgb_points or self.r_points or self.g_points or self.b_points)

    @staticmethod
    def _lut(points: List[Tuple[float, float]]) -> Optional[np.ndarray]:
        if len(points) < 2:
            return None
        pts = sorted(points)
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        x = np.linspace(0, 1, 256)
        return np.interp(x, xs, ys)

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        out = rgb
        lut = self._lut(self.rgb_points)
        if lut is not None:
            idx = np.clip((out * 255).astype(np.int32), 0, 255)
            out = lut[idx]
        for i, pts in enumerate((self.r_points, self.g_points, self.b_points)):
            ch_lut = self._lut(pts)
            if ch_lut is not None:
                c = out[..., i]
                idx = np.clip((c * 255).astype(np.int32), 0, 255)
                out = out.copy()
                out[..., i] = ch_lut[idx]
        return _clip01(out)


@dataclass
class ExposureSettings:
    exposure: float = 0.0   # stops (-10..10)
    offset: float = 0.0     # -0.5..0.5
    gamma: float = 1.0      # 0.01..9.99

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        a = rgb * (2.0 ** self.exposure) + self.offset
        a = _clip01(a)
        return _clip01(np.power(a, 1.0 / max(0.01, min(9.99, self.gamma))))


@dataclass
class GradientMapSettings:
    stops: List[Tuple[float, Tuple[int, int, int]]] = field(
        default_factory=lambda: [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))])

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        lum = _lum(rgb)
        stops = sorted(self.stops)
        pos = np.array([s[0] for s in stops])
        colors = np.array([[c / 255.0 for c in s[1]] for s in stops])
        out = np.zeros((*lum.shape, 3), dtype=np.float32)
        for i in range(3):
            out[..., i] = np.interp(lum, pos, colors[:, i])
        return out


@dataclass
class GrainSettings:
    intensity: float = 0.5   # 0..1
    contrast: float = 0.5    # 0..1
    grain_type: str = "regular"

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(12345)
        noise = rng.standard_normal(rgb.shape[:2]).astype(np.float32)
        noise = np.sign(noise) * (np.abs(noise) ** (1.0 - self.contrast * 0.9))
        noise *= self.intensity * 0.3
        return _clip01(rgb + noise[..., None])


@dataclass
class BlackWhiteSettings:
    reds: float = 0.40
    yellows: float = 0.60
    greens: float = 0.40
    cyans: float = 0.60
    blues: float = 0.20
    magentas: float = 0.80

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        w = np.array([self.reds, self.yellows, self.greens, self.cyans,
                      self.blues, self.magentas], dtype=np.float32)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        mx = np.max(rgb, axis=-1)
        mn = np.min(rgb, axis=-1)
        d = mx - mn
        gray = (w[0] * r + w[1] * g + w[2] * b)
        norm = np.maximum(w.sum(), 1e-6)
        val = _clip01(gray / norm)
        return np.repeat(val[..., None], 3, axis=-1)


@dataclass
class ColorBalanceSettings:
    shadows_cmy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    midtones_cmy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    highlights_cmy: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        lum = _lum(rgb)
        shadow_w = _clip01(1.0 - lum * 2)
        high_w = _clip01(lum * 2 - 1.0)
        mid_w = 1.0 - shadow_w - high_w
        out = rgb.copy()
        for i in range(3):
            adj = (self.shadows_cmy[i] * shadow_w +
                   self.midtones_cmy[i] * mid_w +
                   self.highlights_cmy[i] * high_w) * 0.5
            out[..., i] = _clip01(out[..., i] + adj)
        return out


@dataclass
class LayerAdjustment:
    kind: AdjustmentKind
    hsv: HueSaturationSettings = field(default_factory=HueSaturationSettings)
    levels: LevelsSettings = field(default_factory=LevelsSettings)
    curves: CurvesSettings = field(default_factory=CurvesSettings)
    exposure: ExposureSettings = field(default_factory=ExposureSettings)
    gradient_map: GradientMapSettings = field(default_factory=GradientMapSettings)
    grain: GrainSettings = field(default_factory=GrainSettings)
    black_white: BlackWhiteSettings = field(default_factory=BlackWhiteSettings)
    color_balance: ColorBalanceSettings = field(default_factory=ColorBalanceSettings)

    def apply(self, rgba: np.ndarray) -> np.ndarray:
        """Apply this adjustment to an RGBA float image (H,W,4)."""
        rgb = rgba[..., :3]
        alpha = rgba[..., 3:4]
        if self.kind == AdjustmentKind.HSV:
            out = _apply_hsv(rgb, self.hsv)
        elif self.kind == AdjustmentKind.LEVELS:
            out = self.levels.apply(rgb)
        elif self.kind == AdjustmentKind.CURVES:
            out = self.curves.apply(rgb)
        elif self.kind == AdjustmentKind.EXPOSURE:
            out = self.exposure.apply(rgb)
        elif self.kind == AdjustmentKind.GRADIENT_MAP:
            out = self.gradient_map.apply(rgb)
        elif self.kind == AdjustmentKind.GRAIN:
            out = self.grain.apply(rgb)
        elif self.kind == AdjustmentKind.BLACK_WHITE:
            out = self.black_white.apply(rgb)
        elif self.kind == AdjustmentKind.COLOR_BALANCE:
            out = self.color_balance.apply(rgb)
        elif self.kind == AdjustmentKind.INVERT:
            out = 1.0 - rgb
        else:
            out = rgb
        return np.concatenate([_clip01(out), alpha], axis=-1)


def _apply_hsv(rgb: np.ndarray, s: HueSaturationSettings) -> np.ndarray:
    hsl = _rgb_hsl(rgb)
    h, sat, l = hsl[..., 0], hsl[..., 1], hsl[..., 2]
    if s.colorize:
        h = (s.hue % 360.0) / 360.0
        sat = _clip01(np.full_like(sat, s.saturation / 100.0))
    else:
        h = (h + s.hue / 360.0) % 1.0
        sat = _clip01(sat * (1.0 + s.saturation / 100.0))
    l = _clip01(l + s.lightness / 200.0)
    return _hsl_rgb(np.stack([h, sat, l], axis=-1))


# ---------------------------------------------------------------------------
# Shapes / text / effects / masks (lightweight ports)
# ---------------------------------------------------------------------------

class ShapeKind(Enum):
    RECTANGLE = "Rectangle"
    ROUNDED_RECTANGLE = "Rounded Rectangle"
    ELLIPSE = "Ellipse"
    LINE = "Line"


@dataclass
class LayerShape:
    kind: ShapeKind = ShapeKind.RECTANGLE
    fill_rgba: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    stroke_rgba: Optional[Tuple[float, float, float, float]] = None
    stroke_width: float = 2.0
    corner_radius: float = 0.0


@dataclass
class LayerText:
    string: str = ""
    font_family: str = "Sans Serif"
    font_size: float = 32.0
    color_rgba: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    alignment: str = "left"       # left | center | right
    line_spacing: float = 1.2
    letter_spacing: float = 0.0


@dataclass
class LayerEffects:
    stroke_rgba: Optional[Tuple[float, float, float, float]] = None
    stroke_width: float = 2.0
    drop_shadow_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.5)
    drop_shadow_dx: float = 4.0
    drop_shadow_dy: float = 4.0
    drop_shadow_blur: float = 6.0
    drop_shadow_enabled: bool = False
    color_overlay: Optional[Tuple[float, float, float, float]] = None
    inner_shadow_enabled: bool = False
    outer_glow_enabled: bool = False


@dataclass
class LayerMask:
    """Grayscale coverage array (float 0..1, H×W at layer-pixel resolution)."""
    data: Optional[np.ndarray] = None
    disabled: bool = False
    linked: bool = True

    def coverage_for(self, h: int, w: int) -> np.ndarray:
        if self.data is None:
            return np.ones((h, w), dtype=np.float32)
        if self.data.shape == (h, w):
            return self.data
        from PIL import Image
        img = Image.fromarray((self.data * 255).astype(np.uint8))
        img = img.resize((w, h), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Layer & document — port of ImageLayer / CanvasDocument
# ---------------------------------------------------------------------------

@dataclass
class ImageLayer:
    name: str
    transform: LayerTransform
    image: Optional[np.ndarray] = None      # RGBA float (h,w,4), native resolution
    visible: bool = True
    parent_id: Optional[str] = None
    is_group: bool = False
    opacity: float = 1.0
    blend_mode: LayerBlendMode = LayerBlendMode.NORMAL
    mask: Optional[LayerMask] = None
    mask_source_id: Optional[str] = None
    adjustment: Optional[LayerAdjustment] = None
    shape: Optional[LayerShape] = None
    effects: Optional[LayerEffects] = None
    text: Optional[LayerText] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def size(self) -> Size:
        return self.transform.size

    @property
    def pixel_size(self) -> Size:
        if self.image is not None:
            return (float(self.image.shape[1]), float(self.image.shape[0]))
        return self.transform.size

    def raster(self) -> Optional[np.ndarray]:
        """Native-resolution RGBA pixels for this layer (shape/text rendered)."""
        if self.image is not None:
            return self.image
        if self.shape is not None:
            return render_shape(self)
        if self.text is not None:
            return render_text(self)
        return None


def render_shape(layer: ImageLayer) -> np.ndarray:
    w, h = (max(1, int(round(s))) for s in layer.transform.size)
    img = np.zeros((h, w, 4), dtype=np.float32)
    from PIL import Image, ImageDraw
    sh = layer.shape
    pil = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(pil)
    fill = tuple(int(round(c * 255)) for c in sh.fill_rgba)
    stroke = tuple(int(round(c * 255)) for c in sh.stroke_rgba) if sh.stroke_rgba else None
    if sh.kind == ShapeKind.RECTANGLE:
        d.rectangle([0, 0, w - 1, h - 1], fill=fill, outline=stroke,
                    width=int(sh.stroke_width) if stroke else 0)
    elif sh.kind == ShapeKind.ROUNDED_RECTANGLE:
        d.rounded_rectangle([0, 0, w - 1, h - 1], radius=sh.corner_radius, fill=fill,
                            outline=stroke, width=int(sh.stroke_width) if stroke else 0)
    elif sh.kind == ShapeKind.ELLIPSE:
        d.ellipse([0, 0, w - 1, h - 1], fill=fill, outline=stroke,
                  width=int(sh.stroke_width) if stroke else 0)
    elif sh.kind == ShapeKind.LINE:
        col = fill if fill[3] > 0 else (0, 0, 0, 255)
        d.line([0, h / 2, w, h / 2], fill=col, width=max(1, int(sh.stroke_width)))
    return np.asarray(pil, dtype=np.float32) / 255.0


def render_text(layer: ImageLayer) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont
    t = layer.text
    w, h = (max(1, int(round(s))) for s in layer.transform.size)
    pil = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(pil)
    try:
        font = ImageFont.load_default(size=int(max(6, t.font_size)))
    except TypeError:  # older Pillow
        font = ImageFont.load_default()
    color = tuple(int(round(c * 255)) for c in t.color_rgba)
    lines = t.string.split("\n")
    lh = t.font_size * t.line_spacing
    y = (h - lh * len(lines)) / 2
    for line in lines:
        bbox = d.textbbox((0, 0), line, font=font)
        lw = bbox[2] - bbox[0]
        if t.alignment == "center":
            x = (w - lw) / 2
        elif t.alignment == "right":
            x = w - lw
        else:
            x = 0
        d.text((x, y), line, font=font, fill=color)
        y += lh
    return np.asarray(pil, dtype=np.float32) / 255.0


@dataclass
class CanvasGuide:
    vertical: bool
    position: float  # document px


# ---------------------------------------------------------------------------
# Selection — port of DocumentSelection (path replaced by polygon/mask model)
# ---------------------------------------------------------------------------

class SelectionShapeType(Enum):
    RECT = "rect"
    ELLIPSE = "ellipse"
    POLYGON = "polygon"


@dataclass
class DocumentSelection:
    """A document-space selection outline clipped to canvas."""
    shapes: List[Tuple[SelectionShapeType, object]] = field(default_factory=list)
    feather: float = 0.0
    antialiased: bool = True

    @property
    def is_empty(self) -> bool:
        return not self.shapes

    def bounding_box(self) -> Optional[Tuple[float, float, float, float]]:
        if self.shapes and all(k == "coverage" for k, _ in self.shapes):
            cov = self.shapes[0][1]
            ys, xs = np.nonzero(cov > 0.5)
            if len(xs) == 0:
                return None
            return (float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1)
        xs, ys = [], []
        for kind, obj in self.shapes:
            if kind == SelectionShapeType.RECT:
                x0, y0, x1, y1 = obj
                xs += [x0, x1]
                ys += [y0, y1]
            elif kind == SelectionShapeType.ELLIPSE:
                x0, y0, x1, y1 = obj
                xs += [x0, x1]
                ys += [y0, y1]
            else:
                for (px, py) in obj:
                    xs.append(px)
                    ys.append(py)
        if not xs:
            return None
        return (min(xs), min(ys), max(xs), max(ys))

    def coverage(self, width: int, height: int) -> np.ndarray:
        """Float coverage mask (0..1) at document resolution."""
        cov = np.zeros((height, width), dtype=np.float32)
        if not self.shapes:
            return cov
        if all(k == "coverage" for k, _ in self.shapes):
            c = self.shapes[0][1]
            if c.shape != (height, width):
                from PIL import Image
                img = Image.fromarray((c * 255).astype(np.uint8)).resize((width, height))
                c = np.asarray(img, np.float32) / 255
            if self.feather > 0:
                c = gaussian_blur_float(c, self.feather / 2)
            return c
        from PIL import Image, ImageDraw
        big = Image.new("L", (width, height), 0)
        d = ImageDraw.Draw(big)
        for kind, obj in self.shapes:
            if kind == SelectionShapeType.RECT:
                x0, y0, x1, y1 = obj
                d.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=255)
            elif kind == SelectionShapeType.ELLIPSE:
                x0, y0, x1, y1 = obj
                d.ellipse([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=255)
            else:
                d.polygon(list(obj), fill=255)
        cov = np.asarray(big, dtype=np.float32) / 255.0
        if self.feather > 0:
            cov = gaussian_blur_float(cov, self.feather / 2.0)
        return cov


def gaussian_blur_float(a: np.ndarray, sigma: float) -> np.ndarray:
    from PIL import Image, ImageFilter
    if sigma <= 0:
        return a
    img = Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.asarray(img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Layer hierarchy — port of LayerHierarchy / LayerOpacity
# ---------------------------------------------------------------------------

def hierarchy_entries(layers: List[ImageLayer], top_first: bool = False,
                      collapsed: Optional[set] = None) -> List[Tuple[ImageLayer, int, bool]]:
    collapsed = collapsed or set()
    children: dict = {}
    for lyr in layers:
        children.setdefault(lyr.parent_id, []).append(lyr)
    result: List[Tuple[ImageLayer, int, bool]] = []

    def visit(parent, depth, visible):
        if depth > 64:
            return
        siblings = children.get(parent, [])
        if top_first:
            siblings = list(reversed(siblings))
        for lyr in siblings:
            eff = visible and lyr.visible
            result.append((lyr, depth, eff))
            if lyr.is_group and lyr.id not in collapsed:
                visit(lyr.id, depth + 1, eff)

    visit(None, 0, True)
    return result


def effective_opacity(layer: ImageLayer, by_id: dict) -> float:
    """A folder's opacity multiplies into everything inside it."""
    opacity = layer.opacity
    node_id = layer.parent_id
    depth = 0
    while node_id is not None and depth < 64:
        node = by_id.get(node_id)
        if node is None:
            break
        opacity *= node.opacity
        node_id = node.parent_id
        depth += 1
    return opacity


# ---------------------------------------------------------------------------
# CanvasDocument — port of CanvasDocument
# ---------------------------------------------------------------------------

@dataclass
class CanvasDocument:
    width: int
    height: int
    resolution: float = 72.0
    layers: List[ImageLayer] = field(default_factory=list)  # bottom to top
    guides: List[CanvasGuide] = field(default_factory=list)
    selection: Optional[DocumentSelection] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def size(self) -> Size:
        return (float(self.width), float(self.height))

    @staticmethod
    def valid_dimension(value) -> Optional[int]:
        try:
            n = int(str(value).strip())
        except (TypeError, ValueError):
            return None
        return n if 1 <= n <= 30_000 else None


# ---------------------------------------------------------------------------
# Navigation tools — port of NavigationTool
# ---------------------------------------------------------------------------

class NavigationTool(Enum):
    MOVE = "move"
    MARQUEE = "marquee"
    LASSO = "lasso"
    WAND = "wand"
    CROP = "crop"
    BRUSH = "brush"
    SPOT_HEALING = "spotHealing"
    CLONE_STAMP = "cloneStamp"
    BLUR = "blur"
    GRADIENT = "gradient"
    SHAPE = "shape"
    TYPE = "type"
    EYEDROPPER = "eyedropper"
    HAND = "hand"
    ZOOM = "zoom"
    IDLE = "idle"

    @property
    def is_brush_tool(self) -> bool:
        return self in (NavigationTool.BRUSH, NavigationTool.SPOT_HEALING,
                        NavigationTool.CLONE_STAMP, NavigationTool.BLUR)

    @property
    def is_selection_tool(self) -> bool:
        return self in (NavigationTool.MARQUEE, NavigationTool.LASSO, NavigationTool.WAND)

    @property
    def label(self) -> str:
        return {
            NavigationTool.MOVE: "Move / Transform (V)",
            NavigationTool.MARQUEE: "Marquee (M)",
            NavigationTool.LASSO: "Lasso (L)",
            NavigationTool.WAND: "Magic (W)",
            NavigationTool.CROP: "Crop (C)",
            NavigationTool.BRUSH: "Brush (B) · Eraser (E)",
            NavigationTool.SPOT_HEALING: "Spot Healing Brush (J)",
            NavigationTool.CLONE_STAMP: "Clone Stamp (S)",
            NavigationTool.BLUR: "Smear (R)",
            NavigationTool.GRADIENT: "Gradient (G)",
            NavigationTool.SHAPE: "Shape (U)",
            NavigationTool.TYPE: "Type (T)",
            NavigationTool.EYEDROPPER: "Eyedropper (I)",
            NavigationTool.HAND: "Hand (H)",
            NavigationTool.ZOOM: "Zoom (Z)",
            NavigationTool.IDLE: "No Tool (A)",
        }[self]

    @property
    def shortcut(self) -> str:
        return {
            NavigationTool.MOVE: "V", NavigationTool.MARQUEE: "M",
            NavigationTool.LASSO: "L", NavigationTool.WAND: "W",
            NavigationTool.CROP: "C", NavigationTool.BRUSH: "B",
            NavigationTool.SPOT_HEALING: "J", NavigationTool.CLONE_STAMP: "S",
            NavigationTool.BLUR: "R", NavigationTool.GRADIENT: "G",
            NavigationTool.SHAPE: "U", NavigationTool.TYPE: "T",
            NavigationTool.EYEDROPPER: "I", NavigationTool.HAND: "H",
            NavigationTool.ZOOM: "Z", NavigationTool.IDLE: "A",
        }[self]
