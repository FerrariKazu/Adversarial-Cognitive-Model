"""The evidence model.

Two fitted pieces:

1. **Single-observation evidence** — a multinomial logistic model over the eight anomaly features,
   fitted on calibration scenes, converted to per-cause log-likelihood ratios against ``none``.
   A logistic model is used rather than a naive-Bayes Gaussian because the features are correlated
   and the Gaussian version saturated: with eight dimensions, small mean differences divided by
   small per-class standard deviations produced LLRs of ±10 from a single look, and the posterior
   collapsed to 1.0 on scenes where the evidence was weak. The logistic posterior is calibrated on
   its own fitting split by construction, and Stage C measures any residual miscalibration.

2. **Pair evidence** — per-cause Gaussians over ``(persistence, bright_shift)`` fitted on
   calibration pairs from the fixed base→light schedule. Centred (each cause minus the mean over
   causes) and scaled down, because the pair measurement is the strongest single cue in the task and
   an un-shrunk likelihood would let it dominate the posterior on its own.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from rhan_bi.perception.features import FEATURE_NAMES
from rhan_bi.types import CAUSES, FEATURE_VERSION
from rhan_bi.util import canonical_json

PAIR_CAUSES = ("defect", "reflection", "shadow")
PAIR_SCALE = 0.5
PAIR_CLIP = 4.0


@dataclass
class EvidenceModel:
    feature_names: tuple[str, ...] = FEATURE_NAMES
    scaler_mean: np.ndarray = field(default_factory=lambda: np.zeros(len(FEATURE_NAMES)))
    scaler_scale: np.ndarray = field(default_factory=lambda: np.ones(len(FEATURE_NAMES)))
    coef: np.ndarray = field(default_factory=lambda: np.zeros((len(CAUSES), len(FEATURE_NAMES))))
    intercept: np.ndarray = field(default_factory=lambda: np.zeros(len(CAUSES)))
    classes: tuple[str, ...] = CAUSES
    pair_means: dict[str, np.ndarray] = field(default_factory=dict)
    pair_sds: dict[str, np.ndarray] = field(default_factory=dict)
    version: str = FEATURE_VERSION
    fitted: bool = False

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(
        cls,
        rows_by_cause: dict[str, list[dict[str, float]]],
        pair_rows_by_cause: dict[str, list[tuple[float, float]]] | None = None,
    ) -> "EvidenceModel":
        x, y = [], []
        for cause in CAUSES:
            for row in rows_by_cause.get(cause, []):
                x.append([float(row[k]) for k in FEATURE_NAMES])
                y.append(cause)
        if len(set(y)) < 2:
            raise ValueError("the evidence model needs at least two causes in the calibration rows")
        X = np.asarray(x, dtype=float)
        scaler = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=3000, C=1.0, multi_class="multinomial")
        clf.fit(scaler.transform(X), y)
        model = cls(
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            coef=clf.coef_,
            intercept=clf.intercept_,
            classes=tuple(clf.classes_),
        )
        if pair_rows_by_cause:
            for cause in PAIR_CAUSES:
                rows = pair_rows_by_cause.get(cause, [])
                if not rows:
                    continue
                arr = np.array(rows, dtype=float)
                model.pair_means[cause] = arr.mean(0)
                model.pair_sds[cause] = np.maximum(arr.std(0), 0.15)
        model.fitted = True
        return model

    # ------------------------------------------------------------------ scoring
    def design(self, features: dict[str, float] | np.ndarray) -> np.ndarray:
        raw = (np.array([float(features[k]) for k in self.feature_names], dtype=float)
               if isinstance(features, dict) else np.asarray(features, dtype=float))
        return (raw - self.scaler_mean) / self.scaler_scale

    # the planner's forward model works in this standardised space
    z_row = design

    def z_rows(self, rows: Iterable[dict[str, float]]) -> np.ndarray:
        return np.array([self.design(r) for r in rows], dtype=float)

    def log_proba_design(self, z: np.ndarray) -> dict[str, float]:
        """Log-probabilities from an already standardised design vector."""
        z = np.asarray(z, dtype=float)
        logits = self.coef @ z + self.intercept
        logits = logits - logits.max()
        logp = logits - np.log(np.exp(logits).sum())
        return {c: float(logp[self.classes.index(c)]) for c in CAUSES}

    def log_proba(self, features: dict[str, float]) -> dict[str, float]:
        return self.log_proba_design(self.design(features))

    def proba(self, features: dict[str, float]) -> dict[str, float]:
        lp = self.log_proba(features)
        m = max(lp.values())
        e = {c: float(np.exp(v - m)) for c, v in lp.items()}
        s = sum(e.values())
        return {c: v / s for c, v in e.items()}

    def llr_design(self, z: np.ndarray) -> dict[str, float]:
        lp = self.log_proba_design(z)
        base = lp["none"]
        return {c: float(lp[c] - base) for c in CAUSES}

    def llr(self, features: dict[str, float]) -> dict[str, float]:
        return self.llr_design(self.design(features))

    def pair_llr_matrix(self, points: np.ndarray) -> np.ndarray:
        """Vectorised centred pair LLR, shape (n, 4), ordered as CAUSES."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        out = np.zeros((points.shape[0], len(CAUSES)))
        if not self.pair_means:
            return out
        logs = np.zeros((points.shape[0], len(self.pair_means)))
        for j, (cause, mu) in enumerate(self.pair_means.items()):
            sd = self.pair_sds[cause]
            logs[:, j] = -0.5 * np.sum(((points - mu) / sd) ** 2, axis=1) - np.sum(np.log(sd))
        logs = logs - logs.mean(axis=1, keepdims=True)
        for j, cause in enumerate(self.pair_means):
            out[:, CAUSES.index(cause)] = np.clip(PAIR_SCALE * logs[:, j], -PAIR_CLIP, PAIR_CLIP)
        return out

    def pair_llr(self, persistence: float, bright_shift: float) -> dict[str, float]:
        if not self.pair_means:
            return {c: 0.0 for c in CAUSES}
        vec = self.pair_llr_matrix(np.array([[persistence, bright_shift]]))[0]
        return {c: float(vec[i]) for i, c in enumerate(CAUSES)}

    # ------------------------------------------------------------------ io
    def to_json(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "scaler_mean": self.scaler_mean.tolist(),
            "scaler_scale": self.scaler_scale.tolist(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept.tolist(),
            "classes": list(self.classes),
            "pair_means": {c: v.tolist() for c, v in self.pair_means.items()},
            "pair_sds": {c: v.tolist() for c, v in self.pair_sds.items()},
            "version": self.version,
            "fitted": self.fitted,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "EvidenceModel":
        return cls(
            feature_names=tuple(data["feature_names"]),
            scaler_mean=np.array(data["scaler_mean"], dtype=float),
            scaler_scale=np.array(data["scaler_scale"], dtype=float),
            coef=np.array(data["coef"], dtype=float),
            intercept=np.array(data["intercept"], dtype=float),
            classes=tuple(data["classes"]),
            pair_means={c: np.array(v, dtype=float) for c, v in data.get("pair_means", {}).items()},
            pair_sds={c: np.array(v, dtype=float) for c, v in data.get("pair_sds", {}).items()},
            version=data.get("version", FEATURE_VERSION),
            fitted=bool(data.get("fitted", True)),
        )

    def hash(self) -> str:
        return hashlib.sha256(canonical_json(self.to_json()).encode()).hexdigest()[:16]

    def require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("evidence model has not been fitted")
