"""Observation quality (docs/04 §1). Quality is an image-side index, not a probability."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import laplace

from rhan_bi.types import QualityReport


def assess(image: np.ndarray, features: dict[str, float]) -> QualityReport:
    img = image.astype(np.float32)
    # Laplacian variance, normalised by the number of pixels so it is resolution independent.
    lap = laplace(img)
    lapvar = float(np.var(lap))
    blur = float(np.clip(lapvar / 0.002, 0.0, 1.0))

    p95, p5 = np.percentile(img, [95, 5])
    contrast = float(np.clip((p95 - p5) / 0.25, 0.0, 1.0))
    exposure = float(np.clip(1.0 - abs(float(img.mean()) - 0.45) / 0.45, 0.0, 1.0))

    glare = float(np.clip(features.get("glare", 0.0), 0.0, 1.0))
    occlusion = float(np.clip(features.get("occlusion", 0.0), 0.0, 1.0))
    snr = float(np.clip(features.get("snr", 0.0) / 8.0, 0.0, 1.0))

    score = blur * contrast * exposure * (1.0 - glare) * (1.0 - occlusion) * snr

    flags: list[str] = []
    if blur < 0.25:
        flags.append("blurred")
    if glare > 0.02:
        flags.append("glared")
    if occlusion > 0.15:
        flags.append("occluded")
    if contrast < 0.10:
        flags.append("low_contrast")
    if exposure < 0.4:
        flags.append("saturated")

    return QualityReport(
        score=float(np.clip(score, 0.0, 1.0)),
        flags=tuple(flags),
        components={
            "blur": blur,
            "contrast": contrast,
            "exposure": exposure,
            "glare": glare,
            "occlusion": occlusion,
            "snr": snr,
        },
    )
