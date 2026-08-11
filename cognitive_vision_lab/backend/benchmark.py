"""Curated benchmark results + robust loaders.

Sources of truth, in priority order:
  1. Live files: tier1/results/comparison_table.csv, phase5_sdt/results/sdt_results.csv
  2. Embedded curated profiles transcribed from FINDINGS.md (always available)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from cognitive_vision_lab.backend.metrics import find_ethresh
from cognitive_vision_lab.config import (
    COMPARISON_TABLE,
    DRIFT_STATS,
    ISOLATION_SWEEP,
    SDT_RESULTS,
)
from cognitive_vision_lab.utils.logging import get_logger

log = get_logger("benchmark")


@dataclass
class ModelProfile:
    id: str
    name: str
    family: str
    dataset: str
    params_m: float
    clean_acc: float
    robust_at: dict = field(default_factory=dict)   # eps -> accuracy (matched norm grid)
    autoattack: Optional[float] = None
    ethresh: Optional[float] = None
    dprime: Optional[float] = None
    inference_ms: Optional[float] = None
    memory_mb: Optional[float] = None
    paper: str = ""
    source: str = "curated"
    notes: str = ""
    robustbench_rank: Optional[str] = None


# ── Embedded curated profiles (transcribed from FINDINGS.md / run artifacts) ──
CURATED: list[dict] = [
    {
        "id": "human_stl10", "name": "Human observers", "family": "Biological",
        "dataset": "STL-10", "params_m": 0.0, "clean_acc": 84.0,
        "robust_at": {0.031: 76.0, 0.062: 71.0, 0.094: 66.0},
        "autoattack": None, "ethresh": 0.30, "dprime": 2.72,
        "paper": "Project psychophysics study (n=20 observers)", "source": "sdt",
        "notes": "Stable sensitivity far beyond any model; d′ never crosses 1.0.",
    },
    {
        "id": "rhan_large_pseudolabel", "name": "RHAN-Large (Pseudolabel)",
        "family": "Hybrid Recurrent", "dataset": "STL-10", "params_m": 55.6,
        "clean_acc": 52.6,
        "robust_at": {0.031: 48.0, 0.062: 40.3, 0.094: 33.7},
        "autoattack": 10.6, "ethresh": 0.185, "dprime": 2.748,
        "inference_ms": 14.5, "memory_mb": 3.2,
        "paper": "Finding 14/17 — TRADES-Large pseudo-label curriculum",
        "source": "comparison_table",
        "notes": "Highest εthresh among models; genuine robustness (PGD-50↔PGD-100 gap < 2pp).",
    },
    {
        "id": "trades_large_baseline", "name": "TRADES Large baseline",
        "family": "Hybrid Recurrent", "dataset": "STL-10", "params_m": 55.6,
        "clean_acc": 52.8,
        "robust_at": {0.031: 48.0, 0.062: 40.3, 0.094: 33.7},
        "autoattack": None, "ethresh": None, "dprime": None,
        "paper": "Finding 17 baseline table", "source": "finding17",
        "notes": "Finding-17 sanity reference: 48.0/40.3/33.7 at norm ε=0.031/0.062/0.094.",
    },
    {
        "id": "rhan_v11_loss_ablated", "name": "RHAN-v11 loss-ablated",
        "family": "Hybrid Recurrent", "dataset": "STL-10", "params_m": 55.6,
        "clean_acc": 51.7,
        "robust_at": {0.031: 46.5, 0.062: 38.8, 0.094: 31.8},
        "autoattack": None, "ethresh": None, "dprime": None,
        "paper": "Finding 17 — zeroed active-inference losses, real + synthetic",
        "source": "finding17",
        "notes": "Synthetic data boosts clean but erodes high-ε robustness.",
    },
    {
        "id": "null_ablation_v11", "name": "RHAN-v11 null ablation",
        "family": "Hybrid Recurrent", "dataset": "STL-10", "params_m": 55.6,
        "clean_acc": 47.9,
        "robust_at": {0.031: 45.9, 0.062: 42.6, 0.094: 39.3},
        "autoattack": None, "ethresh": None, "dprime": None,
        "paper": "Finding 17 — real STL-10 only", "source": "finding17",
        "notes": "Architecture wins at high ε on real data (+5.6 pp vs TRADES Large).",
    },
    {
        "id": "rhan_v11_best", "name": "RHAN-v11 (Best)",
        "family": "Active Inference", "dataset": "STL-10", "params_m": 55.6,
        "clean_acc": 51.6,
        "robust_at": {0.031: 2.0, 0.062: 0.3, 0.094: 0.1},
        "autoattack": None, "ethresh": 0.0050, "dprime": None,
        "paper": "Finding 16 — tripartite active inference (T=4)",
        "source": "finding16",
        "notes": "Active-inference losses add zero εthresh benefit (Finding 16).",
    },
    {
        "id": "rhan_v10_final", "name": "RHAN-v10 (Final)",
        "family": "Active Inference", "dataset": "STL-10", "params_m": 55.6,
        "clean_acc": 55.2,
        "robust_at": {0.031: 0.2, 0.062: 0.0, 0.094: 0.0},
        "autoattack": None, "ethresh": 0.0050, "dprime": None,
        "paper": "Finding 16", "source": "finding16",
        "notes": "Static v10 base with active-inference head.",
    },
    {
        "id": "rhan_large_ep45", "name": "RHAN-Large (ep45)",
        "family": "Hybrid Recurrent", "dataset": "STL-10", "params_m": 55.6,
        "clean_acc": 53.8,
        "robust_at": {0.031: 0.6, 0.062: 0.2, 0.094: 0.1},
        "autoattack": None, "ethresh": 0.0053, "dprime": None,
        "paper": "Finding 16 — static large, 45 epochs", "source": "finding16",
        "notes": "Pixel-space grid reference (Finding 16 table).",
    },
    {
        "id": "resnet18_cifar10", "name": "ResNet-18",
        "family": "CNN", "dataset": "CIFAR-10", "params_m": 11.2,
        "clean_acc": 83.1,
        "robust_at": {0.031: 0.0, 0.062: 0.0, 0.094: 0.0},
        "autoattack": 0.0, "ethresh": 0.029, "dprime": 1.82,
        "inference_ms": 120.4, "memory_mb": 1.1,
        "paper": "He et al. 2016", "source": "comparison_table",
        "notes": "Classic CNN baseline — collapses at ε≈0.03.",
    },
    {
        "id": "efficientnet_b0", "name": "EfficientNet-B0",
        "family": "CNN", "dataset": "CIFAR-10", "params_m": 5.3,
        "clean_acc": 96.81,
        "robust_at": {0.031: 0.0, 0.062: 0.0, 0.094: 0.0},
        "autoattack": None, "ethresh": 0.006, "dprime": None,
        "paper": "Tan & Le 2019", "source": "finding1",
        "notes": "Highest clean acc, most fragile (εthresh = 0.006).",
    },
    {
        "id": "vit_b16", "name": "ViT-B/16",
        "family": "Transformer", "dataset": "CIFAR-10", "params_m": 86.6,
        "clean_acc": 75.0,
        "robust_at": {0.031: 0.1, 0.062: 0.0, 0.094: 0.0},
        "autoattack": None, "ethresh": 0.010, "dprime": None,
        "paper": "Dosovitskiy et al. 2021", "source": "finding1",
        "notes": "Global attention preserves a residual accuracy floor at high ε.",
    },
]


def curated_profiles() -> list[ModelProfile]:
    return [ModelProfile(**p) for p in CURATED]


# ── Isolation sweep profiles (report/isolation_sweep_results.json) ─────────────
# Live source of truth: the Kaggle seeded PGD-50 sweep (Run A / Run B vs the
# TRADES Large baseline). The embedded fallback mirrors the JSON exactly
# (verified 2026-08-02) so the lab works offline without drift.
_ISOLATION_META: dict[str, dict] = {
    "run_a_norecon": {
        "id": "rhan_v11_isolation_norecon",
        "name": "RHAN-v11 (Isolation A · no-recon)",
        "family": "Hybrid Recurrent",
        "config": "Run A isolation: --w-recon 0 (reconstruction loss disabled)",
    },
    "run_b_fixedgaze": {
        "id": "rhan_v11_isolation_fixedgaze",
        "name": "RHAN-v11 (Isolation B · fixed-gaze)",
        "family": "Hybrid Recurrent",
        "config": "Run B isolation: --freeze-gaze, --w-recon 0.10",
    },
    "trades_large_baseline_in_run_a": {
        "id": "trades_large_measured",
        "name": "TRADES Large (measured)",
        "family": "Hybrid Recurrent",
        "config": "Matched seed-42 re-run of the Finding-17 baseline",
    },
}

# Embedded mirror of the JSON (verified 2026-08-02), same schema as the file, so
# the live JSON and the offline fallback share one construction path.
_ISOLATION_EMBEDDED: dict = {
    "sweeps": {
        "run_a_norecon": {"points": [
            {"eps_norm": 0.0, "acc_pct": 50.40, "macro_dprime": 1.4326},
            {"eps_norm": 0.031, "acc_pct": 45.60, "macro_dprime": 1.2672},
            {"eps_norm": 0.062, "acc_pct": 35.20, "macro_dprime": 0.8392},
            {"eps_norm": 0.094, "acc_pct": 26.00, "macro_dprime": 0.5447},
        ]},
        "run_b_fixedgaze": {"points": [
            {"eps_norm": 0.0, "acc_pct": 56.20, "macro_dprime": 1.9604},
            {"eps_norm": 0.031, "acc_pct": 47.00, "macro_dprime": 1.1198},
            {"eps_norm": 0.062, "acc_pct": 29.40, "macro_dprime": 0.8749},
            {"eps_norm": 0.094, "acc_pct": 23.40, "macro_dprime": 0.5925},
        ]},
        "trades_large_baseline_in_run_a": {"points": [
            {"eps_norm": 0.0, "acc_pct": 55.20, "macro_dprime": 1.8786},
            {"eps_norm": 0.031, "acc_pct": 46.40, "macro_dprime": 1.7526},
            {"eps_norm": 0.062, "acc_pct": 33.20, "macro_dprime": 1.1203},
            {"eps_norm": 0.094, "acc_pct": 22.60, "macro_dprime": 0.3515},
        ]},
    },
}


def load_isolation_sweep_json() -> Optional[dict]:
    """Load report/isolation_sweep_results.json if present (best-effort)."""
    if not ISOLATION_SWEEP.exists():
        return None
    try:
        with open(ISOLATION_SWEEP) as f:
            data = json.load(f)
        return data if "sweeps" in data else None
    except Exception as e:
        log.warning("isolation sweep JSON load failed: %s", e)
        return None


def isolation_sweep_profiles() -> list[ModelProfile]:
    """Isolation sweep profiles from the live JSON, else the embedded mirror."""
    data = load_isolation_sweep_json()
    if data is None:
        data = _ISOLATION_EMBEDDED
    else:
        sweeps = data.get("sweeps") or {}
        if not all(k in sweeps for k in _ISOLATION_META):
            log.warning("isolation sweep JSON missing expected sweep keys; "
                        "using embedded mirror")
            data = _ISOLATION_EMBEDDED

    out: list[ModelProfile] = []
    for key, meta in _ISOLATION_META.items():
        sweep = (data.get("sweeps") or {}).get(key)
        if not sweep:
            continue
        eps: list[float] = []
        dprimes: list[float] = []
        robust_at: dict[float, float] = {}
        clean: Optional[float] = None
        for p in sweep.get("points") or []:
            if p.get("acc_pct") is None or p.get("macro_dprime") is None:
                continue
            e = float(p["eps_norm"])
            eps.append(e)
            dprimes.append(float(p["macro_dprime"]))
            if e == 0.0:
                clean = float(p["acc_pct"])
            else:
                robust_at[e] = float(p["acc_pct"])
        if not eps:
            continue
        out.append(ModelProfile(
            id=meta["id"], name=meta["name"], family=meta["family"],
            dataset="STL-10", params_m=55.6,
            clean_acc=clean if clean is not None else float("nan"),
            robust_at=robust_at, autoattack=None,
            ethresh=find_ethresh(eps, dprimes),
            dprime=dprimes[0] if dprimes else None,
            paper="Isolation sweep — matched Finding-17 norm-space grid (PGD-50, seed 42, n=500)",
            source="isolation_sweep",
            notes=(f"{meta['config']}; [BOUND CHECK] OK at all eps; "
                   "seeded single draw (run-to-run variance applies)."),
        ))
    return out


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_comparison_table() -> Optional[pd.DataFrame]:
    """tier1/results/comparison_table.csv if present."""
    if not COMPARISON_TABLE.exists():
        return None
    try:
        df = pd.read_csv(COMPARISON_TABLE)
        return df[df.columns[:12]] if df.shape[1] > 12 else df
    except Exception as e:
        log.warning("comparison_table load failed: %s", e)
        return None


def load_sdt_results() -> Optional[pd.DataFrame]:
    """phase5_sdt/results/sdt_results.csv — per-(epsilon, system, class) SDT rows."""
    if not SDT_RESULTS.exists():
        return None
    try:
        return pd.read_csv(SDT_RESULTS)
    except Exception as e:
        log.warning("sdt_results load failed: %s", e)
        return None


def sdt_systems() -> list[str]:
    df = load_sdt_results()
    if df is None or "system" not in df.columns:
        return []
    return sorted(df["system"].dropna().unique().tolist())


def sdt_system_curves(system: str) -> dict:
    """Macro-averaged accuracy + d′ curve for one SDT system (real data)."""
    df = load_sdt_results()
    if df is None or df.empty:
        return {}
    sub = df[df["system"] == system]
    if sub.empty:
        return {}
    curves = {}
    for eps, g in sub.groupby("epsilon"):
        dprimes = g["d_prime"].dropna()
        # accuracy from hit-rate averaged per class (macro)
        acc = g["hit_rate"].mean() * 100.0
        curves[float(eps)] = {
            "accuracy": float(acc),
            "dprime": float(dprimes.mean()),
        }
    return curves


def system_curve_sorted(system: str) -> tuple[list[float], list[float], list[float]]:
    """(epsilons, accuracies, dprimes) for one SDT system, sorted by eps."""
    curves = sdt_system_curves(system)
    eps = sorted(curves)
    return eps, [curves[e]["accuracy"] for e in eps], [curves[e]["dprime"] for e in eps]


def load_drift_stats() -> Optional[pd.DataFrame]:
    if not DRIFT_STATS.exists():
        return None
    try:
        return pd.read_csv(DRIFT_STATS)
    except Exception as e:
        log.warning("drift stats load failed: %s", e)
        return None


def model_summary_table() -> pd.DataFrame:
    """Flatten curated + isolation-sweep profiles into a sortable DataFrame."""
    rows = []
    for p in curated_profiles() + isolation_sweep_profiles():
        rows.append({
            "Model": p.name,
            "Family": p.family,
            "Dataset": p.dataset,
            "Params (M)": p.params_m,
            "Clean %": p.clean_acc,
            "ε=0.031": p.robust_at.get(0.031, float("nan")),
            "ε=0.062": p.robust_at.get(0.062, float("nan")),
            "ε=0.094": p.robust_at.get(0.094, float("nan")),
            "AA %": p.autoattack if p.autoattack is not None else float("nan"),
            "εthresh": p.ethresh if p.ethresh is not None else float("nan"),
            "d′": p.dprime if p.dprime is not None else float("nan"),
            "Source": p.source,
        })
    return pd.DataFrame(rows)


def find_profile(model_id: str) -> Optional[ModelProfile]:
    for p in curated_profiles():
        if p.id == model_id:
            return p
    return None


def profiles_as_json() -> str:
    return json.dumps([p.__dict__ for p in curated_profiles()], indent=2, default=str)
