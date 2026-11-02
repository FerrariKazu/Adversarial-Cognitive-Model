#!/usr/bin/env python3
"""
Stage 0 Closeout Part 2:
- T4c disaggregated per-seed slopes (epochs 10-15)
- Documented McNemar methodology & pairwise 6-cell matrix
- Seed 17 forensics report comparing seeds 17 vs 42
"""

import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

OUT_DIR = Path("evaluation/stage0_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. T4c: Per-Seed Slopes
# ---------------------------------------------------------------------------

def calc_slope(epochs, vals):
    xs = epochs
    ys = vals
    n = len(xs)
    xm, ym = sum(xs) / n, sum(ys) / n
    num = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    den = sum((x - xm) ** 2 for x in xs)
    return num / den if den > 0 else 0.0

eps = [10, 11, 12, 13, 14, 15]

# Extracted from gen2_epoch_log commit history
data_series = {
    "recurrence_112_seed17": {
        "seed": 17,
        "phase": "recurrence_112",
        "clean_ep10_15": [0.6630, 0.6648, 0.6752, 0.6732, 0.6768, 0.6704],
        "robust_ep10_15": [0.3242, 0.3320, 0.3262, 0.3223, 0.3418, 0.3340],
        "loss_ep10_15": [0.9652, 0.8540, 0.8317, 0.7417, 0.7396, 0.6780],
    },
    "recurrence_112_seed42": {
        "seed": 42,
        "phase": "recurrence_112",
        "clean_ep10_15": [0.7260, 0.7386, 0.7514, 0.7564, 0.7586, 0.7610],
        "robust_ep10_15": [0.3652, 0.4043, 0.3965, 0.4062, 0.4102, 0.4004],
        "loss_ep10_15": [0.9425, 0.8914, 0.8011, 0.7544, 0.7167, 0.6991],
    },
    "ff_112_seed42": {
        "seed": 42,
        "phase": "ff_112",
        "clean_ep10_15": [0.7098, 0.7260, 0.7292, 0.7358, 0.7438, 0.7460],
        "robust_ep10_15": [0.3496, 0.3770, 0.3730, 0.3887, 0.3809, 0.3828],
        "loss_ep10_15": [0.9412, 0.8943, 0.8442, 0.8045, 0.7649, 0.7255],
    },
    "ff_112_seed99": {
        "seed": 99,
        "phase": "ff_112",
        "clean_ep10_15": [0.7080, 0.7186, 0.7298, 0.7362, 0.7402, 0.7458],
        "robust_ep10_15": [0.3633, 0.3574, 0.3691, 0.3789, 0.3789, 0.3809],
        "loss_ep10_15": [0.9368, 0.8972, 0.8411, 0.7977, 0.7319, 0.7083],
    },
}

t4c_results = {
    "protocol_id": "consolidated-flagship-112/1",
    "epochs": eps,
    "arms": {},
}

for arm_id, d in data_series.items():
    s_clean = calc_slope(eps, d["clean_ep10_15"])
    s_rob = calc_slope(eps, d["robust_ep10_15"])
    s_loss = calc_slope(eps, d["loss_ep10_15"])
    t4c_results["arms"][arm_id] = {
        "seed": d["seed"],
        "phase": d["phase"],
        "slope_clean_pp_per_epoch": float(s_clean * 100),
        "slope_robust_pp_per_epoch": float(s_rob * 100),
        "slope_loss_per_epoch": float(s_loss),
        "clean_trajectory": d["clean_ep10_15"],
        "robust_trajectory": d["robust_ep10_15"],
        "loss_trajectory": d["loss_ep10_15"],
    }

with open(OUT_DIR / "t4c_per_seed_slopes.json", "w") as f:
    json.dump(t4c_results, f, indent=2)

print("Saved t4c_per_seed_slopes.json")

# ---------------------------------------------------------------------------
# 2. McNemar: Documented Pooling & Pairwise Matrix
# ---------------------------------------------------------------------------

rec17 = np.load(OUT_DIR / "recurrence_112_seed17_per_sample.npz")["corr_worst"]
rec42 = np.load(OUT_DIR / "recurrence_112_seed42_per_sample.npz")["corr_worst"]
ff17  = np.load(OUT_DIR / "ff_112_seed17_per_sample.npz")["corr_worst"]
ff42  = np.load(OUT_DIR / "ff_112_seed42_per_sample.npz")["corr_worst"]
ff99  = np.load(OUT_DIR / "ff_112_seed99_per_sample.npz")["corr_worst"]

def mcnemar_paired(a, b):
    disc1 = np.sum((a == 1) & (b == 0))  # a win
    disc2 = np.sum((a == 0) & (b == 1))  # b win
    n_disc = disc1 + disc2
    delta = (a.mean() - b.mean()) * 100
    p = binomtest(min(disc1, disc2), n_disc, 0.5).pvalue if n_disc > 0 else 1.0
    se = np.sqrt(n_disc) / len(a) * 100
    ci_low = delta - 1.96 * se
    ci_high = delta + 1.96 * se
    return {
        "b_flagship_win": int(disc1),
        "c_baseline_win": int(disc2),
        "n_disc": int(n_disc),
        "delta_pp": float(delta),
        "p_exact": float(p),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
    }

pairwise_matrix = {
    "rec17_vs_ff17": mcnemar_paired(rec17, ff17),
    "rec17_vs_ff42": mcnemar_paired(rec17, ff42),
    "rec17_vs_ff99": mcnemar_paired(rec17, ff99),
    "rec42_vs_ff17": mcnemar_paired(rec42, ff17),
    "rec42_vs_ff42": mcnemar_paired(rec42, ff42),
    "rec42_vs_ff99": mcnemar_paired(rec42, ff99),
}

# The pooled method concatenates across all (flagship_seed, ff_seed) evaluation pairs
# flagship_pooled: [rec17, rec17, rec17, rec42, rec42, rec42] (30,000 observations)
# ff_pooled:       [ff17,  ff42,  ff99,  ff17,  ff42,  ff99]  (30,000 observations)
flag_pooled = np.concatenate([rec17, rec17, rec17, rec42, rec42, rec42])
ff_pooled   = np.concatenate([ff17,  ff42,  ff99,  ff17,  ff42,  ff99])
pooled_res = mcnemar_paired(flag_pooled, ff_pooled)

mcnemar_doc = {
    "protocol_id": "consolidated-flagship-112/1",
    "methodology": {
        "sample_level": "Per-image boolean indicator: correct under clean AND all robust attacks",
        "pooling_structure": "Full Cartesian pairing: 2 flagship seeds (17, 42) x 3 baseline seeds (17, 42, 99) x 5,000 validation samples = 30,000 paired sample evaluations (mean = 15,000 per flagship seed)",
        "exact_test": "Exact two-sided binomial test on discordant pairs: min(b, c) ~ Binomial(n_disc, 0.5)",
        "confidence_interval": "Wald 95% CI on difference in proportions: delta +/- 1.96 * (sqrt(n_disc) / N)",
    },
    "pooled_gating_test": pooled_res,
    "descriptive_pairwise_matrix": pairwise_matrix,
}

with open(OUT_DIR / "mcnemar_pairwise_and_doc.json", "w") as f:
    json.dump(mcnemar_doc, f, indent=2)

print("Saved mcnemar_pairwise_and_doc.json")
