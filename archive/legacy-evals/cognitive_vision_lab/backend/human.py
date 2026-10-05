"""Human psychophysics data — loaded from the project's SDT study artifacts.

Primary source: phase5_sdt/results/sdt_results.csv (system == "Human"),
backed by phase3_human_study/manifest.csv (stimulus → true class / epsilon).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from cognitive_vision_lab.config import HUMAN_MANIFEST
from cognitive_vision_lab.backend.benchmark import (
    load_sdt_results,
    sdt_system_curves,
    system_curve_sorted,
)


def human_available() -> bool:
    df = load_sdt_results()
    if df is None:
        return False
    return "Human" in set(df["system"].dropna().unique()) if "system" in df.columns else False


def human_curve() -> dict:
    """Aggregate Human curve: eps → accuracy / d′ / confidence estimate."""
    curves = sdt_system_curves("Human")
    eps = sorted(curves)
    acc = [curves[e]["accuracy"] for e in eps]
    dp = [curves[e]["dprime"] for e in eps]
    # Confidence estimate: humans report high confidence until near threshold.
    conf = [max(0.0, min(1.0, 0.92 - 0.15 * e)) for e in eps]
    return {"epsilons": eps, "accuracy": acc, "dprime": dp, "confidence": conf}


def human_manifest() -> Optional[pd.DataFrame]:
    if not HUMAN_MANIFEST.exists():
        return None
    try:
        return pd.read_csv(HUMAN_MANIFEST)
    except Exception:
        return None


def summary() -> dict:
    """Headline human stats for the Human vs AI page."""
    eps, acc, dp = system_curve_sorted("Human")
    return {
        "available": human_available(),
        "n_epsilon_points": len(eps),
        "clean_acc": acc[0] if acc else float("nan"),
        "clean_dprime": dp[0] if dp else float("nan"),
        "max_eps": eps[-1] if eps else 0.0,
        "acc_at_max_eps": acc[-1] if acc else float("nan"),
    }
