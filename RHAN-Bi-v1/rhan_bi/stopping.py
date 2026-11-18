"""Stopping rules: commit, abstain, or keep looking (docs/06 §7)."""
from __future__ import annotations

from dataclasses import dataclass

from rhan_bi.belief import BeliefState
from rhan_bi.config import BeliefConfig


@dataclass(frozen=True)
class StopDecision:
    action: str            # "commit" | "abstain" | "continue"
    cause: str | None
    reason: str
    confidence: float
    debt: float
    margin: float


def decide(
    belief: BeliefState,
    budget_left: int,
    cfg: BeliefConfig | None = None,
    best_value: float | None = None,
    next_cost: float = 1.0,
) -> StopDecision:
    cfg = cfg or BeliefConfig()
    cause = belief.map_cause()
    conf = belief.map_confidence()
    debt = belief.debt_total(cause)
    margin = belief.margin()

    if conf >= cfg.tau_commit and debt <= cfg.tau_debt:
        return StopDecision("commit", cause, "confidence and debt thresholds met", conf, debt, margin)
    if budget_left <= 0:
        if margin < cfg.tau_abstain_margin:
            return StopDecision("abstain", None, "budget exhausted with no margin", conf, debt, margin)
        return StopDecision("commit", cause, "budget exhausted", conf, debt, margin)
    if best_value is not None and best_value < next_cost:
        if margin < cfg.tau_abstain_margin:
            return StopDecision("abstain", None, "no observation worth its cost", conf, debt, margin)
        return StopDecision("commit", cause, "no observation worth its cost", conf, debt, margin)
    return StopDecision("continue", None, "evidence insufficient", conf, debt, margin)
