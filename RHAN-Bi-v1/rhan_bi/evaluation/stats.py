"""Paired statistics (docs/10)."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import binomtest


def bootstrap_ci(a: np.ndarray, b: np.ndarray | None = None, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> dict[str, float]:
    """Percentile bootstrap of mean(a) (or mean(a - b) for paired vectors)."""
    a = np.asarray(a, dtype=float)
    diff = a if b is None else a - np.asarray(b, dtype=float)
    if diff.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    return {
        "mean": float(diff.mean()),
        "lo": float(np.quantile(means, alpha / 2)),
        "hi": float(np.quantile(means, 1 - alpha / 2)),
        "n": int(diff.size),
    }


def mcnemar(correct_a: list[bool], correct_b: list[bool]) -> dict[str, float]:
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    b01 = int(np.sum(~a & b))
    b10 = int(np.sum(a & ~b))
    n = b01 + b10
    p = 1.0 if n == 0 else float(binomtest(b10, n, 0.5).pvalue)
    return {"b01": b01, "b10": b10, "n_discordant": n, "p": p}


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, float]]:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict[str, float]] = {}
    prev = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        prev = adj
        out[name] = {"p": float(p), "p_holm": float(adj), "reject": bool(adj < alpha)}
    return out


def paired_table(errors_a: list[int], errors_b: list[int]) -> dict[str, float]:
    a = np.asarray(errors_a, dtype=float)
    b = np.asarray(errors_b, dtype=float)
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "diff": float((a - b).mean())}


def normal_power(delta: float, discordance: float, n: int, alpha: float = 0.05) -> float:
    """Approximate McNemar power (docs/10 §5), reported as an approximation only."""
    var = max(1e-9, (discordance - delta ** 2) / n)
    z = delta / math.sqrt(var)
    from scipy.stats import norm
    crit = norm.ppf(1 - alpha / 2)
    return float(norm.cdf(z - crit) + norm.cdf(-z - crit))
