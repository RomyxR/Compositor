"""Tiny scipy-free image morphology helpers used by editing.py."""
from __future__ import annotations

import numpy as np


def maximum_filter(a: np.ndarray, size: int = 3) -> np.ndarray:
    """Max filter over a square neighbourhood (scipy.ndimage replacement)."""
    r = size // 2
    pad = np.pad(a, r, mode="edge")
    out = a.copy()
    H, W = a.shape
    for dy in range(size):
        for dx in range(size):
            out = np.maximum(out, pad[dy:dy + H, dx:dx + W])
    return out


def minimum_filter(a: np.ndarray, size: int = 3) -> np.ndarray:
    return -maximum_filter(-a, size)
