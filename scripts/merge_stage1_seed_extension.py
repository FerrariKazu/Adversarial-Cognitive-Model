#!/usr/bin/env python3
"""Merge a completed seed-averaged sweep with a seed-extension leg → 8-seed verdict.

The Stage 1 5-seed crossover (2026-08-09) was real but razor-thin (PGD-50
+8.13 vs the 8.02 threshold; the PGD-100 leg did NOT reach significance), so
the protocol extends the seed set to 8 by running seeds 46-48 at eps=0.094
ONLY (both the PGD-50 and PGD-100 legs) and MERGING the new per-seed rows
with the existing 5-seed rows — nothing already completed is re-run.

This script is the pure-math half:
  1. reads the main (5-seed) and extension (3-seed) per-seed CSVs;
  2. verifies the expected seed coverage on both sides;
  3. recomputes the aggregated mean / unbiased sample std (ddof=1) exactly
     like eval_full_epsilon_sweep.py (same rounding);
  4. re-runs the conservative 2-sigma crossover verdicts (d > 2·√(σ_r²+σ_b²))
     exactly like eval_rhan.py's _crossover_verdicts;
  5. writes an 8-seed eval_provenance.json in the EXACT eval_rhan schema so
     the notebook's record_verdict() / masking_verdict() read it unchanged.

Usage (per leg):
    python3 scripts/merge_stage1_seed_extension.py \\
        --main-dir report/sweep_stage1_ais_v1_halting_only \\
        --ext-dir  report/sweep_stage1_ais_v1_halting_only_c2_seeds46_48 \\
        --out-dir  report/sweep_stage1_ais_v1_halting_only_merged \\
        --baseline-label trades_large_baseline \\
        --pgd-steps 50 --n-samples 300 --batch-size 64 \\
        --ckpt-specs rhan_next_ais_v1_halting_only:checkpoints/x_rolling.pth:next \\
                     trades_large_baseline:checkpoints/y.pth:large \\
        --main-seeds 41 42 43 44 45 --ext-seeds 46 47 48

Pure stdlib + numpy — no torch, no GPU; runnable anywhere the CSVs exist.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

PER_SEED_FIELDS = ['ckpt_label', 'seed', 'eps_pixel',
                   'eps_norm_R', 'eps_norm_G', 'eps_norm_B',
                   'acc_pct', 'macro_dprime']
AGG_FIELDS = ['ckpt_label', 'eps_pixel',
              'eps_norm_R', 'eps_norm_G', 'eps_norm_B',
              'acc_mean', 'acc_std', 'macro_dprime_mean', 'macro_dprime_std',
              'n_seeds']


def _mean_std(vals):
    """(mean, unbiased sample std ddof=1) — identical to the frozen sweep."""
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return float('nan'), float('nan')
    return float(arr.mean()), (float(arr.std(ddof=1)) if arr.size > 1 else 0.0)


def load_per_seed(path):
    """{(label, eps4): {seed: row}} from an epsilon_sweep_per_seed.csv."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            label = row['ckpt_label']
            eps4 = round(float(row['eps_pixel']), 4)
            out.setdefault((label, eps4), {})[int(row['seed'])] = row
    return out


def _git_sha():
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _git_branch():
    try:
        out = subprocess.run(['git', 'branch', '--show-current'], cwd=REPO,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "detached"
    except Exception:
        return "unknown"


def _sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            while True:
                block = f.read(chunk)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def crossover_verdicts(agg_rows, baseline_label):
    """Same conservative criterion as eval_rhan._crossover_verdicts.

    agg_rows: list of dicts with acc_mean/acc_std/n_seeds (floats OK).
    """
    verdicts = []
    eps_list = sorted({round(float(r['eps_pixel']), 4) for r in agg_rows})
    for eps in eps_list:
        if eps == 0.0:
            continue
        b = [r for r in agg_rows
             if r['ckpt_label'] == baseline_label
             and round(float(r['eps_pixel']), 4) == eps]
        if not b or int(b[0].get('n_seeds', 0) or 0) < 2:
            continue
        bm, bs = float(b[0]['acc_mean']), float(b[0]['acc_std'])
        for r in agg_rows:
            if r['ckpt_label'] == baseline_label or \
                    round(float(r['eps_pixel']), 4) != eps:
                continue
            rm, rs = float(r['acc_mean']), float(r['acc_std'])
            diff = rm - bm
            combined = math.sqrt(rs ** 2 + bs ** 2)
            verdict = ("CROSSOVER REAL" if diff > 2.0 * combined
                       else ("positive but NOT significant" if diff > 0
                             else "at or below baseline"))
            verdicts.append({
                "eps": eps, "checkpoint": r['ckpt_label'],
                "acc_mean": rm, "acc_std": rs,
                "baseline": baseline_label, "baseline_acc_mean": bm,
                "diff_pp": round(diff, 2),
                "threshold_2sig": round(2.0 * combined, 2),
                "verdict": verdict,
            })
    return verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--main-dir', required=True)
    ap.add_argument('--ext-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--baseline-label', default='trades_large_baseline')
    ap.add_argument('--pgd-steps', type=int, required=True)
    ap.add_argument('--n-samples', type=int, required=True)
    ap.add_argument('--batch-size', type=int, required=True)
    ap.add_argument('--ckpt-specs', nargs='*', default=None,
                    help='label:path:arch[:freeze] — recorded in provenance')
    ap.add_argument('--main-seeds', type=int, nargs='+', required=True)
    ap.add_argument('--ext-seeds', type=int, nargs='+', required=True)
    args = ap.parse_args()

    main_csv = os.path.join(args.main_dir, 'epsilon_sweep_per_seed.csv')
    ext_csv = os.path.join(args.ext_dir, 'epsilon_sweep_per_seed.csv')
    os.makedirs(args.out_dir, exist_ok=True)

    main_rows = load_per_seed(main_csv)
    ext_rows = load_per_seed(ext_csv)

    if not main_rows:
        sys.exit(f"[merge] FATAL: no per-seed rows in {main_csv}. Re-run the "
                 f"5-seed Step C legs first (or restore the CSVs from HF).")
    if not ext_rows:
        sys.exit(f"[merge] FATAL: no per-seed rows in {ext_csv}. The seed "
                 f"extension eval did not produce results.")

    # ── Verify seed coverage ────────────────────────────────────────────────
    # The extension runs eps=0.094 ONLY; every (label, eps) it contains must
    # have ALL extension seeds, and the same (label, eps) must also exist in
    # the main run with ALL main seeds (otherwise merging would silently
    # produce a partial-seed row).
    ext_eps = sorted({e for _, e in ext_rows})
    for (label, eps) in ext_rows:
        have = set(ext_rows[(label, eps)])
        missing = set(args.ext_seeds) - have
        if missing:
            sys.exit(f"[merge] FATAL: {label} eps={eps} extension rows are "
                     f"missing seeds {sorted(missing)} — re-run the extension "
                     f"leg before merging.")
    for (label, eps) in ext_rows:
        m = main_rows.get((label, eps))
        if m is None:
            sys.exit(f"[merge] FATAL: {label} eps={eps} exists in the "
                     f"extension but NOT in the main 5-seed run — cannot "
                     f"merge a seed set onto nothing.")
        missing = set(args.main_seeds) - set(m)
        if missing:
            sys.exit(f"[merge] FATAL: main run {label} eps={eps} is missing "
                     f"seeds {sorted(missing)} — re-run Step C before merging.")

    # ── Merge per-seed rows (main first, then extension) ───────────────────
    merged_seed_rows = []
    merged = {}   # (label, eps4) -> {'accs': [], 'dps': [], 'eps_norm': (r,g,b)}
    for source in (main_rows, ext_rows):
        for (label, eps), seed_map in sorted(source.items()):
            rec = merged.setdefault((label, eps), {'accs': [], 'dps': []})
            for seed in sorted(seed_map):
                row = seed_map[seed]
                merged_seed_rows.append(row)
                rec['accs'].append(float(row['acc_pct']))
                rec['dps'].append(float(row['macro_dprime']))
                rec['eps_norm'] = (float(row['eps_norm_R']),
                                   float(row['eps_norm_G']),
                                   float(row['eps_norm_B']))

    # ── Aggregated CSV (identical schema/rounding to the frozen sweep) ─────
    agg_rows = []
    for (label, eps), rec in sorted(merged.items()):
        am, astd = _mean_std(rec['accs'])
        dm, dstd = _mean_std(rec['dps'])
        r, g, b = rec['eps_norm']
        agg_rows.append({
            'ckpt_label': label, 'eps_pixel': round(eps, 4),
            'eps_norm_R': round(r, 4), 'eps_norm_G': round(g, 4),
            'eps_norm_B': round(b, 4),
            'acc_mean': round(am, 2), 'acc_std': round(astd, 2),
            'macro_dprime_mean': round(dm, 4), 'macro_dprime_std': round(dstd, 4),
            'n_seeds': len(rec['accs']),
        })

    out_per_seed = os.path.join(args.out_dir, 'epsilon_sweep_per_seed.csv')
    with open(out_per_seed, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=PER_SEED_FIELDS)
        w.writeheader()
        # de-dup by (label, seed, eps) keeping the FIRST occurrence (main wins)
        seen = set()
        for row in sorted(merged_seed_rows,
                          key=lambda r: (r['ckpt_label'], r['eps_pixel'],
                                         int(r['seed']))):
            key = (row['ckpt_label'], int(row['seed']),
                   round(float(row['eps_pixel']), 4))
            if key in seen:
                continue
            seen.add(key)
            w.writerow(row)

    out_agg = os.path.join(args.out_dir, 'epsilon_sweep_results.csv')
    with open(out_agg, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=AGG_FIELDS)
        w.writeheader()
        w.writerows(agg_rows)

    # ── Provenance (eval_rhan schema; results rows read back as strings,
    #    exactly like eval_rhan.py reads its own aggregated CSV) ────────────
    with open(out_agg, newline='') as f:
        results_rows = list(csv.DictReader(f))

    specs = []
    for s in args.ckpt_specs or []:
        parts = s.split(':')
        if len(parts) < 3:
            continue
        freeze = len(parts) > 3 and parts[3].strip().lower() in (
            'freeze', 'freeze-gaze', '1', 'true')
        specs.append({'label': parts[0], 'path': parts[1], 'arch': parts[2],
                      'freeze': freeze,
                      'sha256': _sha256_file(parts[1])})

    seeds = sorted(set(args.main_seeds) | set(args.ext_seeds))
    prov = {
        "schema": "eval_rhan_provenance_v1",
        "tool": "scripts/merge_stage1_seed_extension.py",
        "git_sha": _git_sha(),
        "git_branch": _git_branch(),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec='seconds'),
        "eps_mode": "norm_space (forced by eval_rhan.py; Finding-17 "
                    "matched convention)",
        "seeds": seeds,
        "n_samples": args.n_samples,
        "pgd_steps": args.pgd_steps,
        "batch_size": args.batch_size,
        "baseline_label": args.baseline_label,
        "eps_list": sorted({round(float(r['eps_pixel']), 4)
                            for r in results_rows}),
        "checkpoints": specs,
        "results_csv": out_agg,
        "results": results_rows,
        "crossover_verdicts": crossover_verdicts(results_rows,
                                                 args.baseline_label),
        "seed_extension": {
            "applied": True,
            "main_seeds": sorted(args.main_seeds),
            "extended_seeds": sorted(args.ext_seeds),
            "eps": 0.094,
            "note": ("8-seed merged verdict: seeds %s (main) + %s (extension) "
                     "at eps=0.094, both PGD-50 and PGD-100 legs, same "                             "checkpoints/protocol. Extension rows were generated by "
                             "eval_sweep_next.py (the frozen sweep + eval_rhan's "
                             "arch registry) because eval_rhan.py enforces a "
                             ">=5-seed floor by design."
                     % (sorted(args.main_seeds), sorted(args.ext_seeds))),
        },
    }

    out_prov = os.path.join(args.out_dir, 'eval_provenance.json')
    with open(out_prov, 'w') as f:
        json.dump(prov, f, indent=2, sort_keys=True)

    # ── Console summary ─────────────────────────────────────────────────────
    print("=" * 72, flush=True)
    print("  MERGED RESULTS TABLE - mean +- std over seeds (ddof=1)")
    print("=" * 72, flush=True)
    print(f"{'Checkpoint':<30} {'eps':>6} {'Acc% (mean+-std)':>18} "
          f"{'d-prime (mean+-std)':>22} {'n':>3}", flush=True)
    print("-" * 72, flush=True)
    for row in agg_rows:
        print(f"{row['ckpt_label']:<30} {float(row['eps_pixel']):>6.3f} "
              f"{row['acc_mean']:>7.2f}+-{row['acc_std']:<5.2f} "
              f"{row['macro_dprime_mean']:>8.4f}+-{row['macro_dprime_std']:<6.4f} "
              f"{row['n_seeds']:>3}", flush=True)
    print("=" * 72, flush=True)
    print("  CROSSOVER SIGNIFICANCE (criterion: d > 2*sig_combined)", flush=True)
    for cv in prov["crossover_verdicts"]:
        print(f"    eps={cv['eps']}: {cv['checkpoint']:<28} "
              f"{cv['acc_mean']:6.2f}+-{cv['acc_std']:4.2f} vs "
              f"{cv['baseline']} {cv['baseline_acc_mean']:6.2f} | "
              f"d={cv['diff_pp']:+5.2f} pp | 2*sig={cv['threshold_2sig']:5.2f} "
              f"| {cv['verdict']}", flush=True)
    print(f"  Wrote: {out_per_seed}", flush=True)
    print(f"  Wrote: {out_agg}", flush=True)
    print(f"  Wrote: {out_prov}", flush=True)


if __name__ == '__main__':
    main()
