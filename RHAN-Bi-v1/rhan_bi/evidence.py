"""Turn an observation into an immutable evidence record (docs/02 [2], docs/03)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from rhan_bi import __version__
from rhan_bi.config import Config
from rhan_bi.dependence import weight_for
from rhan_bi.ledger import EvidenceLedger
from rhan_bi.likelihood import EvidenceModel
from rhan_bi.perception.backbone import Backbone
from rhan_bi.quality import assess
from rhan_bi.types import CAUSES, FEATURE_VERSION, LEDGER_VERSION, EvidenceRecord, Observation
from rhan_bi.util import canon


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = a.astype(float), b.astype(float)
    denom = math.sqrt(float(aa.sum()) * float(bb.sum()))
    if denom <= 0:
        return 0.0
    return float((aa * bb).sum() / denom)


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


class EvidenceBuilder:
    def __init__(self, config: Config, backbone: Backbone, model: EvidenceModel | None = None):
        self.cfg = config
        self.backbone = backbone
        self.model = model

    def build(self, obs: Observation, ledger: EvidenceLedger, episode_id: str) -> EvidenceRecord:
        enc = self.backbone.encode(obs.image)
        quality = assess(obs.image, enc.features)

        if self.model is not None and self.model.fitted:
            z = self.model.z_row(enc.features)
            features = {k: float(v) for k, v in zip(self.model.feature_names, z)}
            llr = self.model.llr_design(z)
        else:
            features = {k: float(v) for k, v in enc.features.items()}
            llr = {c: 0.0 for c in CAUSES}

        pair = self._pair(obs, ledger, enc.maps, features)
        if pair is not None and self.model is not None and self.model.fitted:
            pair["llr"] = self.model.pair_llr(pair["persistence"], pair["bright_shift"])
        elif pair is not None:
            pair["llr"] = {c: 0.0 for c in CAUSES}

        weight, rels = weight_for(obs.action.to_dict(), ledger.records, self.cfg.evidence, quality.score)

        provenance = {
            "code_version": __version__,
            "feature_version": FEATURE_VERSION,
            "model_version": f"evidence-model-{self.model.hash()}" if self.model is not None else "unfitted",
            "backbone": getattr(self.backbone, "name", "unknown"),
            "camera": {"view": obs.view, "light": obs.light,
                       "roi": list(obs.roi) if obs.roi is not None else None},
            "parent_ids": [pair["with_record_id"]] if pair else [],
        }

        return EvidenceRecord(
            ledger_version=LEDGER_VERSION,
            episode_id=episode_id,
            scene_id=obs.scene_id,
            step=int(obs.step),
            action=obs.action.to_dict(),
            image_sha256=obs.image_sha256,
            features=canon(features),
            quality={
                "score": round(quality.score, 6),
                "flags": list(quality.flags),
                "components": {k: round(v, 6) for k, v in quality.components.items()},
            },
            llr={c: round(float(llr[c]), 6) for c in CAUSES},
            pair=pair,
            dependence={
                "rho_max": max([r["rho"] for r in rels], default=0.0),
                "relations": rels,
                "dependency_weight": round(weight, 6),
                "weight": round(weight, 6),
            },
            provenance=provenance,
        )

    # ------------------------------------------------------------------ pair evidence
    def _pair(self, obs: Observation, ledger: EvidenceLedger, maps: dict[str, np.ndarray],
              features: dict[str, float]) -> dict[str, Any] | None:
        if obs.action.kind != "light":
            return None
        candidates = [
            r for r in ledger.records
            if r.action.get("view") == obs.view
            and r.action.get("light") != obs.light
            and (r.action.get("roi") == (list(obs.roi) if obs.roi is not None else None))
        ]
        if not candidates:
            return None
        prev = candidates[-1]
        prev_img = ledger.image(prev.image_sha256)
        if prev_img is None:
            return None
        prev_enc = self.backbone.encode(prev_img)
        # A persistence comparison is only meaningful when both observations actually contain a
        # dark anomaly. On clean scenes the thresholded maps are texture, and correlating them
        # produces a large, meaningless pair LLR that can flip a correct NONE decision.
        if float(maps["dark"].sum()) < 25 or float(prev_enc.maps["dark"].sum()) < 25:
            return None
        persistence = _overlap(maps["dark"], prev_enc.maps["dark"])
        c_new, c_old = _centroid(maps["bright"]), _centroid(prev_enc.maps["bright"])
        shift = 0.0
        if c_new is not None and c_old is not None:
            shift = float(math.hypot(c_new[0] - c_old[0], c_new[1] - c_old[1]))
        return {
            "with_record_id": prev.record_id,
            "persistence": round(float(persistence), 6),
            "bright_shift": round(float(shift), 6),
            "llr": {c: 0.0 for c in CAUSES},
        }
