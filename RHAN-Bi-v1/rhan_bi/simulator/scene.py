"""The inspection scene: parameters, renderer and ground truth.

The renderer is intentionally simple and deterministic. Its job is to make one physical distinction
measurable: an object-fixed anomaly (a defect) versus an illumination-fixed anomaly (a reflection)
versus a soft geometry effect (a shadow). See docs/08.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, zoom as nd_zoom

from rhan_bi.config import SimConfig
from rhan_bi.util import sha256_image

CAUSE_LIST = ("defect", "reflection", "shadow", "none")


@dataclass(frozen=True)
class AnomalySpec:
    cx: float
    cy: float
    angle: float
    length: float
    width: float
    amplitude: float


@dataclass(frozen=True)
class Occluder:
    x0: int
    x1: int
    y0: int
    y1: int


@dataclass(frozen=True)
class Scene:
    scene_id: str
    seed: int
    cause: str
    scenario: str
    gloss: float
    anomaly: AnomalySpec | None
    occluder: Occluder | None
    texture_seed: int

    # ------------------------------------------------------------------ geometry
    def anomaly_centre(self, view: int, light: int, cfg: SimConfig) -> tuple[float, float]:
        """Centre of the *observable* anomaly under a view/light setting.

        A defect is fixed to the object: it only translates with parallax. A reflection is tied to
        the illumination geometry: it translates with the light and, additionally, with the view.
        A shadow is soft and translates with the light. ``none`` has no anomaly.
        """
        if self.anomaly is None:
            return (0.0, 0.0)
        dx = view * cfg.view_parallax
        dy = 0.0
        if self.cause == "reflection":
            dx += (light - 1) * cfg.reflection_light_shift
            dx += view * cfg.reflection_view_shift
        elif self.cause == "shadow":
            dx += (light - 1) * cfg.shadow_light_shift
        return (self.anomaly.cx + dx, self.anomaly.cy + dy)

    # ------------------------------------------------------------------ truth
    def ground_truth(self) -> dict[str, Any]:
        mask = self.defect_mask()
        return {
            "scene_id": self.scene_id,
            "seed": int(self.seed),
            "cause": self.cause,
            "scenario": self.scenario,
            "defect_mask_sha256": sha256_image(mask.astype(np.uint8)),
            "defect_centre": None if self.cause != "defect" or self.anomaly is None
            else [round(self.anomaly.cx, 3), round(self.anomaly.cy, 3)],
            "defect_angle_rad": None if self.cause != "defect" or self.anomaly is None
            else round(self.anomaly.angle, 4),
            "gloss": round(self.gloss, 4),
            "occluder": None if self.occluder is None
            else [self.occluder.x0, self.occluder.x1, self.occluder.y0, self.occluder.y1],
        }

    def defect_mask(self) -> np.ndarray:
        """Rasterised physical defect mask (all zeros unless the cause is a defect)."""
        cfg = SimConfig()
        h = w = cfg.canvas
        out = np.zeros((h, w), dtype=np.uint8)
        if self.cause != "defect" or self.anomaly is None:
            return out
        yy, xx = np.mgrid[0:h, 0:w]
        a = self.anomaly
        u = (xx - a.cx) * math.cos(a.angle) + (yy - a.cy) * math.sin(a.angle)
        v = -(xx - a.cx) * math.sin(a.angle) + (yy - a.cy) * math.cos(a.angle)
        out[(np.abs(u) <= a.length / 2) & (np.abs(v) <= max(1.0, 2.0 * a.width))] = 1
        return out


# ---------------------------------------------------------------------------- construction
def _plate_texture(seed: int, cfg: SimConfig) -> np.ndarray:
    rng = np.random.default_rng([int(seed), 1])
    raw = rng.normal(size=(cfg.canvas, cfg.canvas))
    smooth = gaussian_filter(raw, 2.0)
    smooth = smooth / (smooth.std() + 1e-9)
    return smooth.astype(np.float32)


def make_scene(split: str, index: int, cfg: SimConfig | None = None) -> Scene:
    """Create a scene from a split name and an index.

    Seeds are the split's base plus the index; cause is balanced by construction
    (``index % 4``), and the scenario is conditional on the cause.
    """
    cfg = cfg or SimConfig()
    base = {"train": 0, "cal": 4000, "dev": 5000, "test": 10000}[split]
    seed = base + int(index) + cfg.seed_offset
    rng = np.random.default_rng([seed, 2])
    cause = CAUSE_LIST[int(index) % 4]

    if cause == "defect":
        scenario = "S1" if index % 2 == 0 else "S3"
    elif cause == "reflection":
        scenario = "S1" if index % 2 == 0 else "S2"
    elif cause == "shadow":
        scenario = "S3"
    else:
        scenario = "core" if index % 3 else "S2"

    gloss = 0.85 + float(rng.normal(0, 0.04)) if cause == "reflection" else 0.30 + float(rng.normal(0, 0.04))
    gloss = float(np.clip(gloss, 0.05, 0.98))

    inset = cfg.plate_inset
    lo, hi = inset + 10, cfg.canvas - inset - 10
    anomaly = AnomalySpec(
        cx=float(rng.uniform(lo, hi)),
        cy=float(rng.uniform(lo, hi)),
        angle=float(rng.uniform(0, math.pi)),
        length=float(rng.uniform(22, 38)),
        width=float(rng.uniform(0.8, 1.1)),
        amplitude=float(rng.uniform(0.15, 0.30)) if cause != "shadow" else float(rng.uniform(0.12, 0.22)),
    ) if cause != "none" else None

    occluder = None
    occl_prob = 0.85 if scenario == "S2" else (cfg.occluder_rate if cause == "none" else 0.05)
    if rng.random() < occl_prob:
        width = int(rng.uniform(0.15, 0.25) * (cfg.canvas - 2 * inset))
        x0 = int(rng.integers(inset, cfg.canvas - inset - width))
        y0 = int(rng.integers(inset, inset + 8))
        occluder = Occluder(x0=x0, x1=x0 + width, y0=y0, y1=cfg.canvas - inset)

    scene_id = f"{split}-{index:05d}-{seed}"
    return Scene(
        scene_id=scene_id,
        seed=int(seed),
        cause=cause,
        scenario=scenario,
        gloss=gloss,
        anomaly=anomaly,
        occluder=occluder,
        texture_seed=int(seed) + 17,
    )


# ---------------------------------------------------------------------------- rendering
def _line_field(
    xx: np.ndarray, yy: np.ndarray, cx: float, cy: float, angle: float, length: float, width: float
) -> np.ndarray:
    u = (xx - cx) * math.cos(angle) + (yy - cy) * math.sin(angle)
    v = -(xx - cx) * math.sin(angle) + (yy - cy) * math.cos(angle)
    along = np.clip(1.0 - (np.abs(u) - length / 2.0) / 1.5, 0.0, 1.0)
    perp = np.exp(-(v ** 2) / (2.0 * width ** 2))
    return (along * perp).astype(np.float32)


def render(
    scene: Scene,
    view: int,
    light: int,
    cfg: SimConfig,
    step: int = 0,
    action_index: int = 0,
    zoom_box: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Render one frame. Deterministic given (scene, view, light, zoom_box, step, action_index)."""
    h = w = cfg.canvas
    inset = cfg.plate_inset
    tex = _plate_texture(scene.texture_seed, cfg)

    img = np.full((h, w), 0.45, dtype=np.float32) + 0.02 * tex
    img[inset:h - inset, inset:w - inset] = 0.55 + 0.03 * tex[inset:h - inset, inset:w - inset]

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    if scene.anomaly is not None:
        cx, cy = scene.anomaly_centre(view, light, cfg)
        a = scene.anomaly
        if scene.cause == "defect":
            field = _line_field(xx, yy, cx, cy, a.angle, a.length, a.width)
            img -= a.amplitude * field
        elif scene.cause == "reflection":
            angle = a.angle + 0.10 * (light - 1)
            field = _line_field(xx, yy, cx, cy, angle, a.length, a.width)
            img -= a.amplitude * field
            halo = _line_field(xx, yy, cx, cy + 1.6, angle, a.length, a.width * 1.8)
            img += 0.02 * scene.gloss * halo
            glare = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * (0.35 * cfg.canvas) ** 2)))
            img += 0.0 * scene.gloss * glare
        elif scene.cause == "shadow":
            shade = np.exp(-((xx - cx) ** 2) / (2.0 * 18.0 ** 2))
            shade *= np.clip((yy - inset) / 4.0, 0, 1) * np.clip((h - inset - yy) / 4.0, 0, 1)
            img -= a.amplitude * shade

    if scene.occluder is not None:
        o = scene.occluder
        patch = 0.30 + 0.005 * tex[o.y0:o.y1, o.x0:o.x1]
        img[o.y0:o.y1, o.x0:o.x1] = patch

    img = np.clip(img, 0.0, 1.0)

    if zoom_box is not None:
        x0, y0, x1, y1 = zoom_box
        px0, px1 = int(round(x0 * w)), int(round(x1 * w))
        py0, py1 = int(round(y0 * h)), int(round(y1 * h))
        crop = img[py0:py1, px0:px1]
        if crop.size:
            img = nd_zoom(crop, (h / max(1, crop.shape[0]), w / max(1, crop.shape[1])), order=1)

    rng = np.random.default_rng([int(scene.seed), int(step), int(action_index), 99])
    img = img + rng.normal(0.0, cfg.noise, size=img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def mutate(scene: Scene, mutation: str, cfg: SimConfig | None = None) -> Scene:
    """Apply the S4 world change: introduce a defect into a previously clean scene."""
    if mutation != "introduce_defect":
        raise ValueError(f"unknown mutation {mutation!r}")
    if scene.cause != "none":
        raise ValueError("introduce_defect requires a scene with cause 'none'")
    cfg = cfg or SimConfig()
    rng = np.random.default_rng([scene.seed, 77])
    inset = cfg.plate_inset
    lo, hi = inset + 10, cfg.canvas - inset - 10
    anomaly = AnomalySpec(
        cx=float(rng.uniform(lo, hi)),
        cy=float(rng.uniform(lo, hi)),
        angle=float(rng.uniform(0, math.pi)),
        length=float(rng.uniform(22, 38)),
        width=float(rng.uniform(0.8, 1.1)),
        amplitude=float(rng.uniform(0.20, 0.30)),
    )
    return replace(scene, cause="defect", anomaly=anomaly, scenario="S4")
