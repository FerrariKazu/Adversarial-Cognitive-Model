"""Metric definitions used by every gate (docs/10)."""
from __future__ import annotations

import math

import numpy as np


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    return float(np.mean([a == b for a, b in zip(y_true, y_pred)])) if y_true else float("nan")


def balanced_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    labels = sorted(set(y_true))
    recalls = []
    for lab in labels:
        idx = [i for i, y in enumerate(y_true) if y == lab]
        if idx:
            recalls.append(np.mean([y_pred[i] == lab for i in idx]))
    return float(np.mean(recalls)) if recalls else float("nan")


def confusion(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[list[int]]:
    idx = {lab: i for i, lab in enumerate(labels)}
    out = [[0] * len(labels) for _ in labels]
    for a, b in zip(y_true, y_pred):
        if a in idx and b in idx:
            out[idx[a]][idx[b]] += 1
    return out


def _as_prob(prob: dict[str, float], labels: list[str]) -> np.ndarray:
    return np.array([float(prob.get(lab, 0.0)) for lab in labels], dtype=float)


def nll(probs: list[dict[str, float]], y: list[str], labels: list[str]) -> float:
    vals = []
    for p, label in zip(probs, y):
        vals.append(-math.log(max(1e-12, float(p.get(label, 0.0)))))
    return float(np.mean(vals)) if vals else float("nan")


def brier(probs: list[dict[str, float]], y: list[str], labels: list[str]) -> float:
    vals = []
    for p, label in zip(probs, y):
        vec = _as_prob(p, labels)
        target = np.array([1.0 if lab == label else 0.0 for lab in labels])
        vals.append(float(np.sum((vec - target) ** 2)))
    return float(np.mean(vals)) if vals else float("nan")


def ece(probs: list[dict[str, float]], y: list[str], bins: int = 15) -> float:
    conf = np.array([max(p.values()) for p in probs], dtype=float)
    correct = np.array([max(p, key=p.get) == label for p, label in zip(probs, y)], dtype=float)
    if len(conf) == 0:
        return float("nan")
    order = np.argsort(conf)
    chunks = np.array_split(order, bins)
    total = 0.0
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        total += len(chunk) * abs(float(conf[chunk].mean()) - float(correct[chunk].mean()))
    return float(total / len(conf))


def risk_coverage(probs: list[dict[str, float]], y: list[str], steps: int = 50) -> list[dict[str, float]]:
    conf = np.array([max(p.values()) for p in probs], dtype=float)
    correct = np.array([max(p, key=p.get) == label for p, label in zip(probs, y)], dtype=float)
    out = []
    if len(conf) == 0:
        return out
    for q in np.linspace(0.05, 1.0, steps):
        n = max(1, int(round(q * len(conf))))
        idx = np.argsort(-conf)[:n]
        out.append({"coverage": float(n / len(conf)),
                    "risk": float(1.0 - correct[idx].mean()),
                    "threshold": float(conf[idx][-1])})
    return out


def selective_at_coverage(probs: list[dict[str, float]], y: list[str], coverage: float) -> dict[str, float]:
    conf = np.array([max(p.values()) for p in probs], dtype=float)
    correct = np.array([max(p, key=p.get) == label for p, label in zip(probs, y)], dtype=float)
    n = max(1, int(round(coverage * len(conf))))
    idx = np.argsort(-conf)[:n]
    return {"coverage": float(n / len(conf)), "risk": float(1.0 - correct[idx].mean()),
            "threshold": float(conf[idx][-1]) if n else 1.0}


def temperature_scale(logits: list[dict[str, float]], y: list[str], labels: list[str]) -> float:
    """Fit one temperature by minimizing NLL (grid + golden section, deterministic)."""
    arr = np.array([[logit[c] for c in labels] for logit in logits], dtype=float)
    yi = np.array([labels.index(label) for label in y])

    def loss(t: float) -> float:
        z = arr / max(t, 1e-3)
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-logp[np.arange(len(yi)), yi].mean())

    grid = np.exp(np.linspace(math.log(0.25), math.log(4.0), 40))
    best = min(grid, key=loss)
    lo, hi = best / 1.15, best * 1.15
    for _ in range(60):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if loss(m1) < loss(m2):
            hi = m2
        else:
            lo = m1
    return float((lo + hi) / 2)


def apply_temperature(prob: dict[str, float], temperature: float) -> dict[str, float]:
    logs = {k: math.log(max(1e-12, v)) / max(temperature, 1e-3) for k, v in prob.items()}
    m = max(logs.values())
    e = {k: math.exp(v - m) for k, v in logs.items()}
    z = sum(e.values())
    return {k: v / z for k, v in e.items()}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
