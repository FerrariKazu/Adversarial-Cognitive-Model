"""Belief state: posterior over causes, contradictions, evidence debt, revisions (docs/05)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rhan_bi.config import BeliefConfig
from rhan_bi.ledger import EvidenceLedger
from rhan_bi.types import CAUSES, EvidenceRecord
from rhan_bi.util import canon


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / e.sum()


@dataclass
class BeliefState:
    posterior: dict[str, float]
    log_weights: dict[str, float]
    debt: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    revisions: list[dict[str, Any]] = field(default_factory=list)
    n_records: int = 0
    n_eff: float = 0.0
    prior: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)

    # ------------------------------------------------------------------ read-outs
    def map_cause(self) -> str:
        return max(CAUSES, key=lambda c: self.posterior[c])

    def map_confidence(self) -> float:
        return float(max(self.posterior.values()))

    def margin(self) -> float:
        vals = sorted(self.posterior.values(), reverse=True)
        return float(vals[0] - vals[1]) if len(vals) > 1 else 1.0

    def entropy(self) -> float:
        return float(-sum(p * math.log(max(p, 1e-12)) for p in self.posterior.values()))

    def risk(self) -> float:
        return float(1.0 - self.map_confidence())

    def debt_total(self, cause: str | None = None) -> float:
        cause = cause or self.map_cause()
        return float(min(1.0, sum(item["severity"] for item in self.debt.get(cause, []))))

    def to_dict(self) -> dict[str, Any]:
        return canon({
            "posterior": self.posterior,
            "log_weights": self.log_weights,
            "map": self.map_cause(),
            "confidence": self.map_confidence(),
            "margin": self.margin(),
            "entropy": self.entropy(),
            "risk": self.risk(),
            "debt": self.debt,
            "debt_total": {c: self.debt_total(c) for c in CAUSES},
            "conflicts": self.conflicts,
            "revisions": self.revisions,
            "n_records": self.n_records,
            "n_eff": self.n_eff,
        })

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_ledger(
        cls,
        ledger: EvidenceLedger,
        cfg: BeliefConfig | None = None,
        prev: "BeliefState | None" = None,
        weight_mode: str = "ledger",
        include_pair: bool = True,
    ) -> "BeliefState":
        """Recompute the posterior from the full ledger.

        ``weight_mode`` and ``include_pair`` exist only for the Gate D ablations: ``"ledger"``
        uses the dependence weights, ``"normal"`` uses weight 1 for every record; ``include_pair``
        False drops the light-change pair evidence. They are not tuning knobs.
        """
        cfg = cfg or BeliefConfig()
        prior = np.array(cfg.prior, dtype=float)
        if prior.shape != (len(CAUSES),) or np.any(prior <= 0):
            raise ValueError("prior must be a positive vector over the four causes")
        prior = prior / prior.sum()

        logits = np.log(prior)
        for rec in ledger.records:
            w = rec.weight() if weight_mode == "ledger" else 1.0
            total = rec.total_llr() if include_pair else {c: float(rec.llr.get(c, 0.0)) for c in CAUSES}
            for i, cause in enumerate(CAUSES):
                logits[i] += w * float(total[cause])
        if not np.all(np.isfinite(logits)):
            raise FloatingPointError("non-finite belief logits")
        post = _softmax(logits)
        posterior = {c: float(post[i]) for i, c in enumerate(CAUSES)}
        log_weights = {c: float(logits[i]) for i, c in enumerate(CAUSES)}

        state = cls(
            posterior=posterior,
            log_weights=log_weights,
            n_records=len(ledger),
            n_eff=ledger.n_eff(),
            prior=cfg.prior,
        )
        state.conflicts = _conflicts(ledger, posterior, cfg)
        state.debt = _debt(ledger, posterior, state.conflicts, cfg)
        if prev is not None:
            state.revisions = list(prev.revisions)
            if prev.map_cause() != state.map_cause() and ledger.records:
                last = ledger.records[-1]
                state.revisions.append({
                    "step": int(last.step),
                    "from": prev.map_cause(),
                    "to": state.map_cause(),
                    "caused_by": last.record_id,
                    "delta": round(log_weights[state.map_cause()] - prev.log_weights[prev.map_cause()], 6),
                    "reason": _revision_reason(last, prev.map_cause(), state.map_cause()),
                })
        return state


def _conflicts(ledger: EvidenceLedger, posterior: dict[str, float], cfg: BeliefConfig) -> list[dict[str, Any]]:
    leader = max(CAUSES, key=lambda c: posterior[c])
    out: list[dict[str, Any]] = []
    for rec in ledger.records:
        total = rec.total_llr()
        if total[leader] <= -cfg.conflict_log:
            challenger = max(CAUSES, key=lambda c: total[c])
            out.append({
                "record_id": rec.record_id,
                "supports": challenger,
                "against": leader,
                "magnitude": round(float(-total[leader]), 6),
                "kind": "llr",
                "weak_source": rec.quality.get("score", 1.0) < 0.35,
            })
        elif rec.pair and rec.pair.get("llr", {}).get(leader, 0.0) <= -cfg.conflict_log:
            out.append({
                "record_id": rec.record_id,
                "supports": max(CAUSES, key=lambda c: rec.pair["llr"].get(c, 0.0)),
                "against": leader,
                "magnitude": round(float(-rec.pair["llr"][leader]), 6),
                "kind": "pair",
                "weak_source": rec.quality.get("score", 1.0) < 0.35,
            })
    return out


def _debt(
    ledger: EvidenceLedger,
    posterior: dict[str, float],
    conflicts: list[dict[str, Any]],
    cfg: BeliefConfig,
) -> dict[str, list[dict[str, Any]]]:
    records = ledger.records
    leader = max(CAUSES, key=lambda c: posterior[c])
    margin = sorted(posterior.values(), reverse=True)
    margin_val = margin[0] - margin[1]
    has_light_change = any(r.action.get("kind") == "light" for r in records)
    supports = {c: [r for r in records if r.total_llr()[c] > 0.3] for c in CAUSES}
    out: dict[str, list[dict[str, Any]]] = {}

    for cause in CAUSES:
        items: list[dict[str, Any]] = []

        occ = max((float(r.quality.get("components", {}).get("occlusion", 0.0)) for r in records), default=0.0)
        if occ > 0.15:
            items.append({"code": "OCCLUDED", "severity": round(0.6 * occ, 6),
                          "reason": f"{occ:.2f} of the region is occluded"})

        glare = max((float(r.quality.get("components", {}).get("glare", 0.0)) for r in records), default=0.0)
        if glare > 0.02:
            items.append({"code": "GLARE", "severity": round(0.5 * min(1.0, glare / 0.1), 6),
                          "reason": f"{glare:.3f} of the region is saturated"})

        if any(float(r.quality.get("score", 1.0)) < 0.35 for r in records):
            items.append({"code": "LOW_QUALITY", "severity": 0.5,
                          "reason": "at least one observation is below the quality floor"})

        sup = supports[cause]
        if len(sup) >= 2:
            ids = {r.record_id for r in sup}
            rhos = [rel["rho"] for r in sup for rel in r.dependence.get("relations", [])
                    if rel["record_id"] in ids]
            if rhos and float(np.mean(rhos)) >= 0.8:
                items.append({"code": "CORRELATED_ONLY", "severity": 0.4,
                              "reason": "supporting observations are near-duplicates"})

        if any(cf["against"] == cause and not cf["weak_source"] for cf in conflicts):
            items.append({"code": "CONFLICT", "severity": 0.5,
                          "reason": "an observation contradicts this hypothesis"})

        if cause == leader and margin_val < 0.15 and not has_light_change:
            items.append({"code": "NO_DISAMBIGUATION", "severity": 0.5,
                          "reason": "no illumination change has been observed"})

        if any(max(abs(v) for v in r.features.values()) > 4.0 for r in records):
            items.append({"code": "OUT_OF_RANGE", "severity": 0.3,
                          "reason": "a feature is outside the calibrated range"})

        if len(sup) == 1:
            items.append({"code": "SINGLE_SOURCE", "severity": 0.3,
                          "reason": "the hypothesis rests on one observation"})

        out[cause] = items
    return out


def _revision_reason(rec: EvidenceRecord, old: str, new: str) -> str:
    total = rec.total_llr()
    if rec.pair:
        return (f"pair persistence {rec.pair.get('persistence', 0):.2f}, "
                f"LLR {total[new]:+.2f} for {new} vs {total[old]:+.2f} for {old}")
    return f"single-record LLR {total[new]:+.2f} for {new} vs {total[old]:+.2f} for {old}"
