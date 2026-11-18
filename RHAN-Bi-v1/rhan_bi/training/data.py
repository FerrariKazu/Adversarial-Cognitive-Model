"""Data generation for the stages. Everything is a pure function of split + index + config."""
from __future__ import annotations

from typing import Any

import numpy as np

from rhan_bi.config import Config
from rhan_bi.evidence import EvidenceBuilder
from rhan_bi.ledger import EvidenceLedger
from rhan_bi.perception.backbone import Backbone
from rhan_bi.simulator.lab import InspectionLab
from rhan_bi.simulator.scene import Scene, make_scene
from rhan_bi.types import CAUSES, Action


def scene_list(split: str, n: int, cfg: Config) -> list[Scene]:
    return [make_scene(split, i, cfg.sim) for i in range(int(n))]


def collect_base_rows(split: str, n: int, cfg: Config, backbone: Backbone) -> dict[str, Any]:
    """One base observation per scene: raw features, labels, quality and localization maps."""
    rows_by_cause: dict[str, list[dict[str, float]]] = {c: [] for c in CAUSES}
    labels: list[str] = []
    quality: list[dict[str, float]] = []
    quality_by_cause: dict[str, list[dict[str, float]]] = {c: [] for c in CAUSES}
    maps: list[np.ndarray] = []
    scenes: list[Scene] = []
    for scene in scene_list(split, n, cfg):
        lab = InspectionLab(scene, cfg)
        obs = lab.reset()
        enc = backbone.encode(obs.image)
        rows_by_cause[scene.cause].append(enc.features)
        labels.append(scene.cause)
        q = np.array([enc.features.get(k, 0.0) for k in
                      ("glare", "occlusion", "contrast", "snr")], dtype=float)
        qrow = {"glare": float(q[0]), "occlusion": float(q[1]),
                "contrast": float(q[2]), "snr": float(q[3])}
        quality.append(qrow)
        quality_by_cause[scene.cause].append(qrow)
        maps.append(enc.maps["dark_response"])
        scenes.append(scene)
    return {"rows": rows_by_cause, "labels": labels, "quality": quality,
            "quality_by_cause": quality_by_cause, "maps": maps, "scenes": scenes}


def collect_calibration(n: int, cfg: Config, backbone: Backbone) -> dict[str, Any]:
    """Calibration episodes with a fixed, policy-independent schedule base -> light -> view -> zoom.

    Returns the raw feature rows needed to fit the evidence model, the pair features (same view,
    different light), and the per-(cause, action kind) rows for the forward model.
    """
    base = collect_base_rows("cal", n, cfg, backbone)
    pair_rows: dict[str, list[tuple[float, float]]] = {c: [] for c in CAUSES}
    rows_all: dict[str, list[dict[str, float]]] = {c: [] for c in CAUSES}
    forward_raw: list[tuple[str, str, dict[str, float]]] = []
    quality_rows: list[dict[str, Any]] = []

    for scene in scene_list("cal", n, cfg):
        lab = InspectionLab(scene, cfg, budget=3)
        builder = EvidenceBuilder(cfg, backbone, model=None)
        ledger = EvidenceLedger()
        obs0 = lab.reset()
        rec0 = builder.build(obs0, ledger, lab.episode_id)
        ledger.add(rec0, image=obs0.image)
        base_feats = backbone.encode(obs0.image).features
        forward_raw.append((scene.cause, "base", base_feats))
        rows_all[scene.cause].append(base_feats)

        try:
            obs1 = lab.observe(Action(kind="light", light=2, cost=cfg.planner.cost_light))
        except ValueError:
            continue
        rec1 = builder.build(obs1, ledger, lab.episode_id)
        ledger.add(rec1, image=obs1.image)
        light_feats = backbone.encode(obs1.image).features
        forward_raw.append((scene.cause, "light", light_feats))
        rows_all[scene.cause].append(light_feats)
        if rec1.pair:
            pair_rows[scene.cause].append((float(rec1.pair["persistence"]),
                                           float(rec1.pair["bright_shift"])))
        quality_rows.append({
            "components": rec1.quality["components"],
            "score": rec1.quality["score"],
        })

        # the remaining scheduled observations are only for the forward model
        for action in (Action(kind="view", view=1, cost=cfg.planner.cost_view),
                       Action(kind="zoom", roi=(0.25, 0.25, 0.75, 0.75), cost=cfg.planner.cost_zoom)):
            try:
                obsx = lab.observe(action)
            except ValueError:
                continue
            feats_x = backbone.encode(obsx.image).features
            forward_raw.append((scene.cause, action.kind, feats_x))
            rows_all[scene.cause].append(feats_x)

    return {"base": base, "pair_rows": pair_rows, "rows_all": rows_all,
            "forward_raw": forward_raw, "quality_rows": quality_rows}
