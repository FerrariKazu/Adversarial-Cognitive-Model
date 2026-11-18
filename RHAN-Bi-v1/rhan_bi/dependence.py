"""Evidence dependence and effective sample size (docs/03 §5, docs/04 §4)."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from rhan_bi.config import EvidenceConfig


def _roi_key(roi: Any) -> tuple | None:
    if roi is None:
        return None
    return tuple(round(float(x), 4) for x in roi)


def rho(action_a: dict[str, Any], action_b: dict[str, Any], cfg: EvidenceConfig) -> tuple[float, str]:
    """Structural dependence between two observations, with the reason."""
    ka, kb = action_a.get("kind"), action_b.get("kind")
    va, vb = action_a.get("view"), action_b.get("view")
    la, lb = action_a.get("light"), action_b.get("light")
    ra, rb = _roi_key(action_a.get("roi")), _roi_key(action_b.get("roi"))

    same_roi = ra == rb
    if ka == kb and va == vb and la == lb and same_roi:
        return cfg.dependence_same_action, "same action"
    if va == vb and la != lb and same_roi:
        return cfg.dependence_same_view, "same view, different light"
    if la == lb and va != vb:
        return cfg.dependence_same_light, "same light, different view"
    if va == vb and la == lb and not same_roi:
        return cfg.dependence_overlap_roi, "same view and light, different ROI"
    return cfg.dependence_scene, "common scene only"


def relations(action: dict[str, Any], records: Iterable[Any], cfg: EvidenceConfig) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in records:
        r, reason = rho(action, rec.action, cfg)
        out.append({"record_id": rec.record_id, "rho": float(r), "reason": reason})
    return out


def weight_for(action: dict[str, Any], records: Iterable[Any], cfg: EvidenceConfig,
               quality_score: float = 1.0) -> tuple[float, list[dict[str, Any]]]:
    """``1 / max(1, 1 + Σρ)`` times the quality factor (0.5 + 0.5 q). Never above 1."""
    rels = relations(action, records, cfg)
    total = float(sum(r["rho"] for r in rels))
    dep = 1.0 / max(1.0, 1.0 + total)
    qf = 0.5 + 0.5 * float(np.clip(quality_score, 0.0, 1.0))
    return float(np.clip(dep * qf, 0.0, 1.0)), rels


def effective_n(rhos: list[float]) -> float:
    n = len(rhos)
    if n <= 1:
        return float(n)
    avg = float(np.mean(rhos))
    return float(n / (1.0 + (n - 1) * max(0.0, avg)))


def mean_pairwise_rho(rels: list[dict[str, Any]]) -> float:
    vals = [float(r["rho"]) for r in rels]
    return float(np.mean(vals)) if vals else 0.0


def vif(feature_matrix: np.ndarray) -> list[float]:
    """Variance inflation factors from regressing each column on the others."""
    x = np.asarray(feature_matrix, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2 or x.shape[0] < x.shape[1] + 2:
        return [float("nan")] * (x.shape[1] if x.ndim == 2 else 0)
    out: list[float] = []
    for j in range(x.shape[1]):
        y = x[:, j]
        z = np.delete(x, j, axis=1)
        z = np.column_stack([np.ones(len(z)), z])
        beta, *_ = np.linalg.lstsq(z, y, rcond=None)
        resid = y - z @ beta
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out.append(float(1.0 / max(1e-6, 1.0 - r2)))
    return out
