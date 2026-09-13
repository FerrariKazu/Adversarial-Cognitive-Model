#!/usr/bin/env python3
"""
Regenerate E1 audit Table 1 from the per-seed CSV via pandas groupby.
Downloads the full CSV from HF if available, falls back to local.
Computes all downstream numbers: crossover deltas, significance thresholds,
E1-vs-D delta, paired per-seed mean Δ.
"""

import os, sys, csv, json, math
from pathlib import Path

import pandas as pd
import numpy as np

# ── 1. Load the per-seed CSV ─────────────────────────────────────────────────
CSV_LOCAL = Path("report/sweep_stage4_e1_d_e1_pgd100/epsilon_sweep_per_seed.csv")
CSV_HF_REPO = "FerrariKazu/rhan-eval-sweep"
CSV_HF_PATH = "sweep_stage4_e1_d_e1_pgd100/epsilon_sweep_per_seed.csv"

def download_csv():
    """Try HF first, fall back to local."""
    try:
        from huggingface_hub import hf_hub_download
        token = os.environ.get("HF_TOKEN")
        local = hf_hub_download(
            repo_id=CSV_HF_REPO,
            filename=CSV_HF_PATH,
            repo_type="dataset",
            token=token,
        )
        print(f"[OK] Downloaded full CSV from HF: {local}")
        return Path(local)
    except Exception as e:
        print(f"[WARN] HF download failed ({e}), using local CSV")
        return CSV_LOCAL

csv_path = download_csv()
df = pd.read_csv(csv_path)
print(f"\n[CSV] {len(df)} rows, columns: {list(df.columns)}")
print(f"[CSV] Unique checkpoints: {df['ckpt_label'].unique().tolist()}")
print(f"[CSV] Seeds per checkpoint:")
for ckpt in df['ckpt_label'].unique():
    sub = df[df['ckpt_label'] == ckpt]
    n_seeds = sub['seed'].nunique()
    eps_values = sub['eps_pixel'].unique()
    print(f"  {ckpt}: {n_seeds} seeds, eps={sorted(eps_values)}")

# ── 2. Check completeness ────────────────────────────────────────────────────
EXPECTED_SEEDS = list(range(41, 57))  # 41-56
EXPECTED_EPS = [0.0, 0.094]
EXPECTED_CKPTS = ['trades_large_baseline', 'rhan_next_ais_hpc', 'rhan_next_ais_hpc_recon']
expected_cells = len(EXPECTED_CKPTS) * len(EXPECTED_SEEDS) * len(EXPECTED_EPS)
actual_cells = len(df)
print(f"\n[CHECK] Expected {expected_cells} cells, found {actual_cells}")
if actual_cells != expected_cells:
    print(f"[WARN] CSV is INCOMPLETE — missing {expected_cells - actual_cells} cells")
    # Show what's missing
    for ckpt in EXPECTED_CKPTS:
        for eps in EXPECTED_EPS:
            for seed in EXPECTED_SEEDS:
                mask = (df['ckpt_label'] == ckpt) & (df['eps_pixel'] == eps) & (df['seed'] == seed)
                if not mask.any():
                    print(f"  MISSING: {ckpt} seed={seed} eps={eps}")
else:
    print("[OK] CSV is complete")

# ── 3. Pandas groupby — the CORRECT summary table ───────────────────────────
print("\n" + "=" * 72)
print("  CORRECTED SUMMARY TABLE — groupby(ckpt_label, eps_pixel).agg(['mean','std'])")
print("=" * 72)

grouped = df.groupby(['ckpt_label', 'eps_pixel']).agg(
    acc_mean=('acc_pct', 'mean'),
    acc_std=('acc_pct', 'std'),     # pandas default ddof=1
    dp_mean=('macro_dprime', 'mean'),
    dp_std=('macro_dprime', 'std'),
    n_seeds=('seed', 'nunique'),
).reset_index()

print(f"\n{'Checkpoint':<30} {'eps':>6} {'Acc% (mean±std)':>20} {'d-prime (mean±std)':>24} {'n':>3}")
print("-" * 90)
for _, row in grouped.iterrows():
    print(f"{row['ckpt_label']:<30} {row['eps_pixel']:>6.3f} "
          f"{row['acc_mean']:>8.2f} ± {row['acc_std']:<5.2f} "
          f"{row['dp_mean']:>9.4f} ± {row['dp_std']:<6.4f} "
          f"{int(row['n_seeds']):>3}")

# ── 4. Automated consistency assertion ───────────────────────────────────────
print("\n" + "=" * 72)
print("  CONSISTENCY ASSERTION — groupby vs manual check")
print("=" * 72)

all_pass = True
for ckpt in EXPECTED_CKPTS:
    for eps in EXPECTED_EPS:
        mask = (df['ckpt_label'] == ckpt) & (df['eps_pixel'] == eps)
        sub = df[mask]
        manual_mean = sub['acc_pct'].mean()
        manual_std = sub['acc_pct'].std(ddof=1)
        gb_row = grouped[(grouped['ckpt_label'] == ckpt) & (grouped['eps_pixel'] == eps)]
        gb_mean = gb_row['acc_mean'].values[0]
        gb_std = gb_row['acc_std'].values[0]
        
        mean_ok = abs(manual_mean - gb_mean) < 0.01
        std_ok = abs(manual_std - gb_std) < 0.01
        
        status = "PASS" if (mean_ok and std_ok) else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {ckpt} eps={eps}: mean={manual_mean:.4f} vs {gb_mean:.4f} "
              f"({'OK' if mean_ok else 'MISMATCH'}) | "
              f"std={manual_std:.4f} vs {gb_std:.4f} "
              f"({'OK' if std_ok else 'MISMATCH'}) → {status}")

if all_pass:
    print("\n  ✅ ALL CONSISTENCY CHECKS PASSED")
else:
    print("\n  ❌ CONSISTENCY CHECKS FAILED — data divergence detected!")
    sys.exit(1)

# ── 5. Extract key numbers for downstream computation ────────────────────────
def get_stats(ckpt, eps):
    row = grouped[(grouped['ckpt_label'] == ckpt) & (grouped['eps_pixel'] == eps)]
    return row['acc_mean'].values[0], row['acc_std'].values[0], row['dp_mean'].values[0], row['dp_std'].values[0]

# Clean accuracy (eps=0.0)
trades_clean_mean, trades_clean_std, _, _ = get_stats('trades_large_baseline', 0.0)
d_clean_mean, d_clean_std, _, _ = get_stats('rhan_next_ais_hpc', 0.0)
e1_clean_mean, e1_clean_std, _, _ = get_stats('rhan_next_ais_hpc_recon', 0.0)

# PGD-100 accuracy (eps=0.094)
trades_pgd_mean, trades_pgd_std, trades_dp_mean, trades_dp_std = get_stats('trades_large_baseline', 0.094)
d_pgd_mean, d_pgd_std, d_dp_mean, d_dp_std = get_stats('rhan_next_ais_hpc', 0.094)
e1_pgd_mean, e1_pgd_std, e1_dp_mean, e1_dp_std = get_stats('rhan_next_ais_hpc_recon', 0.094)

print(f"\n{'='*72}")
print("  KEY NUMBERS (from groupby)")
print(f"{'='*72}")
print(f"  TRADES clean: {trades_clean_mean:.2f} ± {trades_clean_std:.2f}")
print(f"  TRADES PGD:   {trades_pgd_mean:.2f} ± {trades_pgd_std:.2f}  (d'={trades_dp_mean:.4f} ± {trades_dp_std:.4f})")
print(f"  D clean:      {d_clean_mean:.2f} ± {d_clean_std:.2f}")
print(f"  D PGD:        {d_pgd_mean:.2f} ± {d_pgd_std:.2f}  (d'={d_dp_mean:.4f} ± {d_dp_std:.4f})")
print(f"  E1 clean:     {e1_clean_mean:.2f} ± {e1_clean_std:.2f}")
print(f"  E1 PGD:       {e1_pgd_mean:.2f} ± {e1_pgd_std:.2f}  (d'={e1_dp_mean:.4f} ± {e1_dp_std:.4f})")

# ── 6. Crossover significance ────────────────────────────────────────────────
print(f"\n{'='*72}")
print("  CROSSOVER SIGNIFICANCE (criterion: Δ > 2·σ_combined)")
print(f"{'='*72}")

def crossover(label_a, mean_a, std_a, label_b, mean_b, std_b, eps_label):
    delta = mean_a - mean_b
    sig_combined = math.sqrt(std_a**2 + std_b**2)
    threshold = 2 * sig_combined
    real = abs(delta) > threshold
    print(f"  {label_a} vs {label_b} @ {eps_label}: "
          f"{mean_a:.2f} vs {mean_b:.2f} | Δ={delta:+.2f} pp | "
          f"2σ={threshold:.2f} | {'CROSSOVER REAL' if real else 'NOT SIGNIFICANT'}")
    return delta, threshold, real

d_vs_trades_delta, d_vs_trades_thresh, d_vs_trades_real = crossover(
    "D (AIS+HPC)", d_pgd_mean, d_pgd_std,
    "TRADES baseline", trades_pgd_mean, trades_pgd_std, "ε=0.094")

e1_vs_trades_delta, e1_vs_trades_thresh, e1_vs_trades_real = crossover(
    "E1 (AIS+HPC+recon)", e1_pgd_mean, e1_pgd_std,
    "TRADES baseline", trades_pgd_mean, trades_pgd_std, "ε=0.094")

e1_vs_d_delta = e1_pgd_mean - d_pgd_mean
print(f"\n  E1 vs D @ ε=0.094: {e1_pgd_mean:.2f} vs {d_pgd_mean:.2f} | Δ={e1_vs_d_delta:+.2f} pp")

# ── 7. Paired per-seed comparison (D vs E1) ──────────────────────────────────
print(f"\n{'='*72}")
print("  PAIRED PER-SEED COMPARISON (D vs E1, ε=0.094)")
print(f"{'='*72}")

d_pgd = df[(df['ckpt_label'] == 'rhan_next_ais_hpc') & (df['eps_pixel'] == 0.094)].set_index('seed')['acc_pct']
e1_pgd = df[(df['ckpt_label'] == 'rhan_next_ais_hpc_recon') & (df['eps_pixel'] == 0.094)].set_index('seed')['acc_pct']
common_seeds = sorted(set(d_pgd.index) & set(e1_pgd.index))

d_wins = 0
e1_wins = 0
ties = 0
deltas = []

print(f"\n{'Seed':>6} {'D PGD':>8} {'E1 PGD':>8} {'Δ(E1-D)':>10} {'Winner':>8}")
print("-" * 45)
for seed in common_seeds:
    d_val = d_pgd[seed]
    e1_val = e1_pgd[seed]
    delta = e1_val - d_val
    deltas.append(delta)
    if delta > 0.01:
        winner = "E1"
        e1_wins += 1
    elif delta < -0.01:
        winner = "D"
        d_wins += 1
    else:
        winner = "Tie"
        ties += 1
    print(f"{seed:>6} {d_val:>8.2f} {e1_val:>8.2f} {delta:>+10.2f} {winner:>8}")

mean_delta = np.mean(deltas)
std_delta = np.std(deltas, ddof=1)
print(f"\n  Head-to-head: D wins {d_wins}/{len(common_seeds)}, "
      f"E1 wins {e1_wins}/{len(common_seeds)}, Tie {ties}/{len(common_seeds)}")
print(f"  Mean Δ(E1-D): {mean_delta:+.2f} pp (std={std_delta:.2f})")

# ── 8. Cross-verification: groupby mean Δ vs paired mean Δ ──────────────────
print(f"\n{'='*72}")
print("  CROSS-VERIFICATION: groupby Δ vs paired mean Δ")
print(f"{'='*72}")
groupby_delta = e1_pgd_mean - d_pgd_mean
print(f"  groupby Δ(E1-D):  {groupby_delta:+.2f} pp")
print(f"  paired mean Δ:    {mean_delta:+.2f} pp")
print(f"  difference:       {abs(groupby_delta - mean_delta):.4f} pp")
if abs(groupby_delta - mean_delta) < 0.1:
    print("  ✅ AGREEMENT — both methods produce consistent results")
else:
    print("  ⚠️  DISAGREEMENT — investigate further")

# ── 9. Paired per-seed clean accuracy ────────────────────────────────────────
print(f"\n{'='*72}")
print("  PAIRED PER-SEED CLEAN ACCURACY (D vs E1, ε=0.000)")
print(f"{'='*72}")

d_clean = df[(df['ckpt_label'] == 'rhan_next_ais_hpc') & (df['eps_pixel'] == 0.0)].set_index('seed')['acc_pct']
e1_clean = df[(df['ckpt_label'] == 'rhan_next_ais_hpc_recon') & (df['eps_pixel'] == 0.0)].set_index('seed')['acc_pct']
common_seeds_clean = sorted(set(d_clean.index) & set(e1_clean.index))

clean_deltas = []
e1_clean_wins = 0
d_clean_wins = 0

print(f"\n{'Seed':>6} {'D Clean':>9} {'E1 Clean':>9} {'Δ(E1-D)':>10}")
print("-" * 40)
for seed in common_seeds_clean:
    d_val = d_clean[seed]
    e1_val = e1_clean[seed]
    delta = e1_val - d_val
    clean_deltas.append(delta)
    if delta > 0.01:
        e1_clean_wins += 1
    elif delta < -0.01:
        d_clean_wins += 1
    print(f"{seed:>6} {d_val:>9.2f} {e1_val:>9.2f} {delta:>+10.2f}")

mean_clean_delta = np.mean(clean_deltas)
print(f"\n  E1 wins clean: {e1_clean_wins}/{len(common_seeds_clean)}, "
      f"D wins clean: {d_clean_wins}/{len(common_seeds_clean)}")
print(f"  Mean Δ(E1-D) clean: {mean_clean_delta:+.2f} pp")

# ── 10. Output corrected numbers for the audit report ────────────────────────
print(f"\n{'='*72}")
print("  CORRECTED AUDIT REPORT NUMBERS")
print(f"{'='*72}")
print(f"""
Summary Table (CORRECTED):
  TRADES baseline  ε=0.000  clean={trades_clean_mean:.2f} ± {trades_clean_std:.2f}  d'={get_stats('trades_large_baseline', 0.0)[2]:.4f} ± {get_stats('trades_large_baseline', 0.0)[3]:.4f}
  TRADES baseline  ε=0.094  pgd={trades_pgd_mean:.2f} ± {trades_pgd_std:.2f}  d'={trades_dp_mean:.4f} ± {trades_dp_std:.4f}
  D (AIS+HPC)      ε=0.000  clean={d_clean_mean:.2f} ± {d_clean_std:.2f}  d'={get_stats('rhan_next_ais_hpc', 0.0)[2]:.4f} ± {get_stats('rhan_next_ais_hpc', 0.0)[3]:.4f}
  D (AIS+HPC)      ε=0.094  pgd={d_pgd_mean:.2f} ± {d_pgd_std:.2f}  d'={d_dp_mean:.4f} ± {d_dp_std:.4f}
  E1 (recon)       ε=0.000  clean={e1_clean_mean:.2f} ± {e1_clean_std:.2f}  d'={get_stats('rhan_next_ais_hpc_recon', 0.0)[2]:.4f} ± {get_stats('rhan_next_ais_hpc_recon', 0.0)[3]:.4f}
  E1 (recon)       ε=0.094  pgd={e1_pgd_mean:.2f} ± {e1_pgd_std:.2f}  d'={e1_dp_mean:.4f} ± {e1_dp_std:.4f}

Crossover:
  D vs TRADES:   Δ = {d_vs_trades_delta:+.2f} pp, 2σ = {d_vs_trades_thresh:.2f} → {'REAL' if d_vs_trades_real else 'NOT SIG'}
  E1 vs TRADES:  Δ = {e1_vs_trades_delta:+.2f} pp, 2σ = {e1_vs_trades_thresh:.2f} → {'REAL' if e1_vs_trades_real else 'NOT SIG'}
  E1 vs D:       Δ = {e1_vs_d_delta:+.2f} pp

Paired:
  Mean Δ(E1-D) PGD-100: {mean_delta:+.2f} pp (D wins {d_wins}/{len(common_seeds)})
  Mean Δ(E1-D) clean:   {mean_clean_delta:+.2f} pp (E1 wins {e1_clean_wins}/{len(common_seeds_clean)})
""")
