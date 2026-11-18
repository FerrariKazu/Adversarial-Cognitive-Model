"""Observation selection (docs/06).

Every policy scores the same predicted outcomes; only the score differs. The forward model is
fitted on calibration episodes and predicts features per (cause, action kind); it is the agent's
model of its own sensor, not access to ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rhan_bi.belief import BeliefState
from rhan_bi.config import Config
from rhan_bi.dependence import rho
from rhan_bi.ledger import EvidenceLedger
from rhan_bi.likelihood import PAIR_CAUSES, EvidenceModel
from rhan_bi.perception.features import FEATURE_NAMES
from rhan_bi.types import CAUSES, Action
from rhan_bi.util import canonical_json, sha256_text

POLICIES = ("falsification", "eig", "confidence", "random", "fixed_light_first", "oracle")

KIND_QUALITY = {"base": 1.0, "light": 0.95, "zoom": 0.90, "view": 0.90}


@dataclass
class ForwardModel:
    mu: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    sd: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    pair_mu: dict[str, np.ndarray] = field(default_factory=dict)
    pair_sd: dict[str, np.ndarray] = field(default_factory=dict)
    fitted: bool = False

    @classmethod
    def fit(cls, rows: list[tuple[str, str, np.ndarray]],
            pair_rows: dict[str, list[tuple[float, float]]] | None = None) -> "ForwardModel":
        out = cls()
        groups: dict[tuple[str, str], list[np.ndarray]] = {}
        for cause, kind, z in rows:
            groups.setdefault((cause, kind), []).append(np.asarray(z, dtype=float))
        for key, vals in groups.items():
            arr = np.stack(vals)
            out.mu[key] = arr.mean(0)
            out.sd[key] = np.maximum(arr.std(0), 0.20)
        if pair_rows:
            for cause in PAIR_CAUSES:
                vals = pair_rows.get(cause, [])
                if vals:
                    arr = np.array(vals, dtype=float)
                    out.pair_mu[cause] = arr.mean(0)
                    out.pair_sd[cause] = np.maximum(arr.std(0), 0.05)
        out.fitted = True
        return out

    def sample(self, rng: np.random.Generator, cause: str, kind: str) -> tuple[np.ndarray, np.ndarray | None]:
        key = (cause, kind)
        if key not in self.mu:
            key = (cause, "base")
        if key not in self.mu:
            mu, sd = np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES))
        else:
            mu, sd = self.mu[key], self.sd[key]
        z = rng.normal(mu, sd)
        pair = None
        if kind == "light" and cause in self.pair_mu:
            pair = rng.normal(self.pair_mu[cause], self.pair_sd[cause])
        return z, pair

    def hash(self) -> str:
        payload = {
            "mu": {f"{c}|{k}": v.tolist() for (c, k), v in self.mu.items()},
            "sd": {f"{c}|{k}": v.tolist() for (c, k), v in self.sd.items()},
            "pair_mu": {c: v.tolist() for c, v in self.pair_mu.items()},
        }
        return sha256_text(canonical_json(payload))[:16]


class Planner:
    def __init__(self, config: Config, model: EvidenceModel, forward: ForwardModel):
        self.cfg = config
        self.model = model
        self.forward = forward

    # ------------------------------------------------------------------ public
    def choose(self, belief: BeliefState, ledger: EvidenceLedger, actions: list[Action],
               policy: str = "falsification", rng: np.random.Generator | None = None,
               ground_truth: str | None = None) -> tuple[Action | None, dict[str, Any]]:
        if not actions:
            return None, {"policy": policy, "scores": {}}
        rng = rng or np.random.default_rng(0)
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}")
        if policy == "random":
            idx = int(rng.integers(0, len(actions)))
            return actions[idx], {"policy": policy, "scores": {a.label(): None for a in actions}}
        if policy == "fixed_light_first":
            return self._fixed(actions, ledger)

        scored: dict[str, dict[str, float]] = {}
        for action in actions:
            scored[action.label()] = self._score_action(belief, ledger, action, rng, policy, ground_truth)

        def key(a: Action):
            s = scored[a.label()].get("score", float("-inf"))
            order = {"light": 0, "zoom": 1, "view": 2}.get(a.kind, 3)
            return (-s, order, a.label())

        best = sorted(actions, key=key)[0]
        return best, {"policy": policy, "scores": scored}

    # ------------------------------------------------------------------ scoring
    def _score_action(self, belief: BeliefState, ledger: EvidenceLedger, action: Action,
                      rng: np.random.Generator, policy: str,
                      ground_truth: str | None) -> dict[str, float]:
        pc = self.cfg
        w = self._hypothetical_weight(action, ledger)
        n = max(4, pc.planner.n_samples)
        p_prior = np.array([belief.posterior[c] for c in CAUSES])
        leader = belief.map_cause()

        if policy == "oracle":
            # Reported-only upper bound: sample outcomes from the *true* cause and pick the action
            # that maximizes the probability of ending with the correct MAP. Not deployable.
            if ground_truth is None:
                return {"score": 0.0, "oracle": 0.0}
            n_or = max(8, pc.planner.n_samples // 2)
            causes = np.full(n_or, CAUSES.index(ground_truth), dtype=int)
            post = self._simulate(p_prior, w, action, causes, rng)
            score = float(np.mean([CAUSES[int(np.argmax(row))] == ground_truth for row in post]))
            return {"score": score, "oracle": 1.0}

        risk_before = belief.risk()
        h_before = belief.entropy()
        causes = rng.choice(len(CAUSES), size=n, p=p_prior)
        post = self._simulate(p_prior, w, action, causes, rng)      # (n, 4)
        leader_idx = CAUSES.index(leader)
        disp = 1.0 - post[:, leader_idx]
        risk_after = 1.0 - post.max(axis=1)
        risk_gain = risk_before - risk_after
        ent = -(post * np.log(np.clip(post, 1e-12, 1.0))).sum(axis=1)
        ent_gain = h_before - ent
        max_post = post.max(axis=1)

        mean = lambda xs: float(np.mean(xs))
        redundancy = self._redundancy(action, ledger)
        quality = KIND_QUALITY.get(action.kind, 0.9)
        cost = float(action.cost or 0.0) / max(1e-9, pc.planner.cost_view)
        penalty = pc.planner.lam_cost * cost + pc.planner.lam_redundancy * redundancy + \
            pc.planner.lam_quality * (1.0 - quality)

        base = {
            "displacement": mean(disp),
            "risk_gain": mean(risk_gain),
            "entropy_gain": mean(ent_gain),
            "expected_max_posterior": mean(max_post),
            "redundancy": redundancy,
            "weight": w,
            "penalty": penalty,
        }
        if policy == "falsification":
            score = pc.planner.w_falsify * base["displacement"] + pc.planner.w_risk * base["risk_gain"] - penalty
        elif policy == "eig":
            score = base["entropy_gain"] - penalty
        elif policy == "confidence":
            score = base["expected_max_posterior"] - penalty
        else:  # pragma: no cover
            score = -penalty
        base["score"] = float(score)
        return base

    def _simulate(self, p_prior: np.ndarray, weight: float, action: Action,
                  causes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Vectorised posterior after one hypothetical observation, for each sampled cause."""
        n = len(causes)
        llr = np.zeros((n, len(CAUSES)))
        log_prior = np.log(np.clip(p_prior, 1e-12, 1.0))
        for ci, cause in enumerate(CAUSES):
            mask = causes == ci
            k = int(mask.sum())
            if k == 0:
                continue
            key = (cause, action.kind)
            if key not in self.forward.mu:
                key = (cause, "base")
            mu = self.forward.mu.get(key, np.zeros(len(FEATURE_NAMES)))
            sd = self.forward.sd.get(key, np.ones(len(FEATURE_NAMES)))
            z = rng.normal(mu, sd, size=(k, len(FEATURE_NAMES)))
            logits = z @ self.model.coef.T + self.model.intercept
            logits = logits - logits.max(axis=1, keepdims=True)
            logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
            row = logp - logp[:, [self.model.classes.index("none")]]
            if action.kind == "light" and cause in self.forward.pair_mu:
                pair = rng.normal(self.forward.pair_mu[cause], self.forward.pair_sd[cause], size=(k, 2))
                row = row + self.model.pair_llr_matrix(pair)
            llr[mask] = row
        logits = log_prior[None, :] + weight * llr
        logits = logits - logits.max(axis=1, keepdims=True)
        post = np.exp(logits)
        return post / post.sum(axis=1, keepdims=True)

    def _hypothetical_weight(self, action: Action, ledger: EvidenceLedger) -> float:
        rels = [rho(action.to_dict(), rec.action, self.cfg.evidence)[0] for rec in ledger.records]
        return 1.0 / max(1.0, 1.0 + sum(rels))

    def _redundancy(self, action: Action, ledger: EvidenceLedger) -> float:
        if not ledger.records:
            return 0.0
        return max(rho(action.to_dict(), rec.action, self.cfg.evidence)[0] for rec in ledger.records)

    def _fixed(self, actions: list[Action], ledger: EvidenceLedger) -> tuple[Action, dict[str, Any]]:
        """A strong hand-designed protocol: change the light first, then move, then change again."""
        step = len(ledger.records)
        preferred = {
            1: ("light", 2),
            2: ("view", -1),
            3: ("light", 0),
        }.get(step, ("zoom", None))
        for action in actions:
            if action.kind == preferred[0] and (preferred[1] is None or
                                                (action.kind == "light" and action.light == preferred[1]) or
                                                (action.kind == "view" and action.view == preferred[1])):
                return action, {"policy": "fixed_light_first", "scores": {action.label(): 1.0}}
        return actions[0], {"policy": "fixed_light_first", "scores": {actions[0].label(): 1.0}}
