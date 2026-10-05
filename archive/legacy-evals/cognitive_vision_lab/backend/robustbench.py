"""RobustBench leaderboard integration.

Embedded curated subset (CIFAR-10 Linf) so the page always renders; a live
fetch from the RobustBench GitHub CSV can be enabled at runtime.

The project's own RHAN entry is a deliberate cross-dataset reference (STL-10)
and is therefore always surfaced on the page — it must survive both the
CIFAR-10 dataset filter and a live leaderboard fetch.
"""
from __future__ import annotations

import pandas as pd

# The project's own model, kept visible as a cross-dataset reference point.
RHAN_OURS: dict = {
    "model": "RHAN-Large (Ours)",
    "arch": "Hybrid Recurrent",
    "method": "TRADES + pseudo-label curriculum",
    "dataset": "STL-10",
    "clean": 52.6,
    "robust": 10.6,   # RHAN's own AutoAttack number on STL-10
    "params_m": 55.6,
    "source": "This project",
}

# Curated RobustBench entries (CIFAR-10, Linf threat model).
# accuracy, robust accuracy, params — transcribed from robustbench.github.io.
ROBUSTBENCH_CIFAR10_LINF: list[dict] = [
    RHAN_OURS,
    {"model": "Wang2023Better_WRN-70-16", "arch": "WideResNet-70-16", "method": "Better diffusion",
     "dataset": "CIFAR-10", "clean": 92.23, "robust": 66.95, "params_m": 266.8, "source": "RobustBench 2023"},
    {"model": "Peng2023Robust_WRN28-10", "arch": "WideResNet-28-10", "method": "WAT + reconstruction",
     "dataset": "CIFAR-10", "clean": 88.5, "robust": 61.07, "params_m": 36.5, "source": "RobustBench 2023"},
    {"model": "Cui2023Improved_WRN28-10", "arch": "WideResNet-28-10", "method": "DA + WAT",
     "dataset": "CIFAR-10", "clean": 87.65, "robust": 59.78, "params_m": 36.5, "source": "RobustBench 2023"},
    {"model": "Gowal2021Improving_70-16-ddt-extra", "arch": "WideResNet-70-16", "method": "Extra data + DA",
     "dataset": "CIFAR-10", "clean": 91.79, "robust": 66.1, "params_m": 266.8, "source": "RobustBench 2021"},
    {"model": "Rebuffi2021Fixing_70-16_cutout_extra", "arch": "WideResNet-70-16", "method": "Extra data + cutout",
     "dataset": "CIFAR-10", "clean": 92.41, "robust": 66.58, "params_m": 266.8, "source": "RobustBench 2021"},
    {"model": "Rebuffi2021Fixing_R18_cutout_extra", "arch": "ResNet-18", "method": "Extra data + cutout",
     "dataset": "CIFAR-10", "clean": 87.33, "robust": 56.87, "params_m": 11.2, "source": "RobustBench 2021"},
    {"model": "Sehwag2021Proxy_R18", "arch": "ResNet-18", "method": "Proxy robustness",
     "dataset": "CIFAR-10", "clean": 88.8, "robust": 56.07, "params_m": 11.2, "source": "RobustBench 2021"},
    {"model": "Wu2020Adversarial_wide", "arch": "WideResNet-34-10", "method": "Adversarial training",
     "dataset": "CIFAR-10", "clean": 85.57, "robust": 46.22, "params_m": 46.2, "source": "RobustBench 2020"},
    {"model": "Madry2018_8_8", "arch": "ResNet-18", "method": "PGD adversarial training",
     "dataset": "CIFAR-10", "clean": 87.14, "robust": 44.04, "params_m": 11.2, "source": "RobustBench 2018"},
]

_THREATS = {
    "Linf": ROBUSTBENCH_CIFAR10_LINF,
}


def leaderboard(dataset: str = "CIFAR-10", threat: str = "Linf",
                fetch_live: bool = False) -> pd.DataFrame:
    """Return a leaderboard DataFrame, optionally refreshing from GitHub.

    The curated RHAN entry is always retained: it is a cross-dataset reference
    (STL-10) and is intentionally shown alongside the CIFAR-10 entries.
    """
    rows = list(_THREATS.get(threat, _THREATS["Linf"]))
    if fetch_live:
        live = _fetch_live(threat)
        if live is not None:
            rows = live
    df = pd.DataFrame(rows)
    if dataset != "All":
        df = df[df["dataset"].isin(["All", dataset])
               | (df["model"] == RHAN_OURS["model"])]
    return df.sort_values("robust", ascending=False).reset_index(drop=True)


def leaderboard_with_ours(dataset: str = "CIFAR-10", threat: str = "Linf",
                          fetch_live: bool = False) -> pd.DataFrame:
    """Like :func:`leaderboard` but guarantees the RHAN entry is present.

    A successful live fetch replaces the embedded list entirely, so the
    curated reference point is re-appended afterwards.
    """
    df = leaderboard(dataset=dataset, threat=threat, fetch_live=fetch_live)
    if RHAN_OURS["model"] in set(df["model"]):
        return df
    combined = pd.concat([pd.DataFrame([RHAN_OURS]), df], ignore_index=True)
    return combined.sort_values("robust", ascending=False).reset_index(drop=True)

def _fetch_live(threat: str) -> list | None:
    """Fetch the official RobustBench leaderboard CSV (best-effort)."""
    url_map = {
        "Linf": "https://raw.githubusercontent.com/RobustBench/robustbench/master/robustbench/model_info/cifar10_linf.csv",
    }
    url = url_map.get(threat)
    if not url:
        return None
    try:
        import io as _io
        import urllib.request

        with urllib.request.urlopen(url, timeout=10) as r:
            raw = r.read().decode()
        df = pd.read_csv(_io.StringIO(raw))
        out = []
        for _, row in df.iterrows():
            out.append({
                "model": row.get("model_name", str(_)),
                "arch": row.get("model_arch", ""),
                "method": row.get("model_type", ""),
                "dataset": "CIFAR-10",
                "clean": float(row.get("clean_acc", 0.0)),
                "robust": float(row.get("robust_acc", 0.0)),
                "params_m": float(row.get("model_params", 0.0)),
                "source": "RobustBench (live)",
            })
        return out
    except Exception:
        return None


def available_threats() -> list[str]:
    return list(_THREATS.keys())
