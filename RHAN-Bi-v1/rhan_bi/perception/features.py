"""Anomaly features from a single frame.

The features are physics-motivated, not learned: oriented dark/bright line responses, a low-pass dark
deviation (shadow), a sharpness ratio, glare, occlusion, contrast and SNR. They are z-scored against
a calibration null before the evidence model sees them (docs/03 §4).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, uniform_filter

from rhan_bi.config import FeatureConfig

FEATURE_NAMES: tuple[str, ...] = (
    "ridge_dark",
    "ridge_bright",
    "soft_dark",
    "sharp_ratio",
    "glare",
    "occlusion",
    "contrast",
    "snr",
)

ANGLES = (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4)


def plate_roi(image: np.ndarray, margin: int = 4) -> tuple[int, int, int, int]:
    """The region that contains the plate in a full frame, or an inset of a zoomed frame."""
    h, w = image.shape[:2]
    inset = max(margin + 16, h // 6)
    return (inset, inset, w - inset, h - inset)


def _line_response(highpass: np.ndarray, cfg: FeatureConfig, sign: float) -> tuple[float, np.ndarray]:
    """Max over 4 orientations of the mean high-pass along a short oriented segment."""
    h, w = highpass.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    half = cfg.line_length // 2
    best = np.zeros((h, w), dtype=np.float32)
    for angle in ANGLES:
        dx, dy = math.cos(angle), math.sin(angle)
        acc = np.zeros((h, w), dtype=np.float32)
        for t in range(-half, half + 1):
            acc += map_coordinates(
                highpass,
                [yy + t * dy, xx + t * dx],
                order=1,
                mode="nearest",
            )
        acc /= float(cfg.line_length)
        best = np.maximum(best, sign * acc)
    return float(best.max()), best


def z_extract(image: np.ndarray, cfg: FeatureConfig) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Raw feature values and the maps used for pair evidence."""
    img = image.astype(np.float32)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = plate_roi(img, cfg.roi_margin)

    highpass = img - gaussian_filter(img, cfg.highpass_sigma)
    mad = float(np.median(np.abs(highpass - np.median(highpass)))) + 1e-6
    noise = 1.4826 * mad

    dark_peak_raw, dark_map = _line_response(highpass, cfg, sign=-1.0)
    bright_peak_raw, bright_map = _line_response(highpass, cfg, sign=+1.0)
    dark_peak = float(dark_map[y0:y1, x0:x1].max())
    bright_peak = float(bright_map[y0:y1, x0:x1].max())

    low = gaussian_filter(img, 4.0)
    plate_median = float(np.median(low[y0:y1, x0:x1]))
    soft = -(low - plate_median)

    # sharpness at the strongest dark response inside the ROI
    roi_dark = dark_map[y0:y1, x0:x1]
    soft_norm = np.clip(soft[y0:y1, x0:x1], 0, None) / 0.30
    combined = roi_dark / (noise * 3.0) + soft_norm
    iy, ix = np.unravel_index(int(np.argmax(combined)), combined.shape)
    cy, cx = y0 + iy, x0 + ix
    r = 4
    sl = (slice(max(0, cy - r), min(h, cy + r + 1)), slice(max(0, cx - r), min(w, cx + r + 1)))
    hp_e = float(np.mean(highpass[sl] ** 2))
    low_detail = gaussian_filter(img, 1.5) - low
    lp_e = float(np.mean(low_detail[sl] ** 2))
    sharp = hp_e / (hp_e + 4.0 * lp_e + 1e-9)

    glare = float(np.mean(img[y0:y1, x0:x1] >= 0.98))
    local_mean = uniform_filter(img, size=5)
    local_var = uniform_filter(img * img, size=5) - local_mean ** 2
    local_std = np.sqrt(np.clip(local_var, 0.0, None))
    flat = (local_std < 0.02) & (img > 0.26) & (img < 0.34)
    occlusion = float(np.mean(flat[y0:y1, x0:x1]))
    p95, p5 = np.percentile(img[y0:y1, x0:x1], [95, 5])
    contrast = float(p95 - p5)
    snr = float(np.max(np.abs(highpass[y0:y1, x0:x1])) / noise)

    features = {
        "ridge_dark": float(dark_peak / (noise * 3.0)),
        "ridge_bright": float(bright_peak / (noise * 3.0)),
        "soft_dark": float(np.clip(soft[y0:y1, x0:x1].max(), 0, None) / 0.30),
        "sharp_ratio": float(np.clip(sharp, 0.0, 1.0)),
        "glare": glare,
        "occlusion": occlusion,
        "contrast": contrast,
        "snr": snr,
    }
    roi_mask = np.zeros_like(img, dtype=bool)
    roi_mask[y0:y1, x0:x1] = True
    maps = {
        "dark": ((highpass < -cfg.ridge_threshold * noise) & roi_mask).astype(np.uint8),
        "bright": ((highpass > cfg.ridge_threshold * noise) & roi_mask).astype(np.uint8),
        "dark_response": (dark_map * roi_mask).astype(np.float32),
        "bright_response": (bright_map * roi_mask).astype(np.float32),
        "highpass": highpass,
    }
    return features, maps
