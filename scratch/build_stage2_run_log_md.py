#!/usr/bin/env python3
"""
build_stage2_run_log_md.py — assemble report/stage2_hpc_run_log.md.

Source of truth for the per-epoch numbers: the HF-synced trainer diag
(report/rhan_next_hpc_only_diag.jsonl, 27 rows: the prepended epoch-1
resume baseline + epochs 35-60 of the Step B run that finished 2026-08-16
under commit b606c55). The printed per-epoch diagnostic blocks in the Kaggle
log are exactly these rows, so the tables here match the log 1:1.

Step C eval numbers come from the HF-synced sweep CSVs
(report/sweep_stage2_hpc_only/epsilon_sweep_{per_seed,results}.csv —
the re-run under commit 3c50d51 completed the full THREE-WAY PGD-50 sweep
(A baseline / B AIS-v1 / C HPC-only) on 2026-08-18 and syncs each leg to HF
immediately, so these are the exact final numbers). The PGD-100 leg was still
in progress when the log was captured, so its partial rows are transcribed
from the pasted log below.

Metadata that is NOT in the diag (pseudo-label distribution, per-epoch
loss/throughput/epoch-duration from the printed epoch lines) is transcribed
from the pasted Kaggle log and sanity-checked against the diag where they
overlap (tr_acc / te_acc / beta / recon / gate / hpc_err / pi_d /
hpc_err_per_class).
"""
import csv
import json
import os

DIAG = "report/rhan_next_hpc_only_diag.jsonl"
OUT = "report/stage2_hpc_run_log.md"
SWEEP_DIR = "report/sweep_stage2_hpc_only"
SWEEP_RESULTS = os.path.join(SWEEP_DIR, "epsilon_sweep_results.csv")
SWEEP_PER_SEED = os.path.join(SWEEP_DIR, "epsilon_sweep_per_seed.csv")

CLASSES = ["airplane", "bird", "car", "cat", "deer",
           "dog", "horse", "monkey", "ship", "truck"]

# ── Transcribed from the printed epoch lines (not present in the diag) ──────
# epoch -> (loss, throughput img/s, epochs/hr, epoch seconds)
EPOCH_LINE = {
    35: (0.820, 7.64, 2.86, 1257), 36: (0.819, 7.68, 2.88, 1250),
    37: (0.810, 7.65, 2.87, 1254), 38: (0.815, 7.65, 2.87, 1255),
    39: (0.814, 7.66, 2.87, 1253), 40: (0.815, 7.67, 2.87, 1252),
    41: (1.011, 7.67, 2.88, 1251), 42: (0.970, 7.66, 2.87, 1253),
    43: (0.945, 7.67, 2.87, 1252), 44: (0.940, 7.66, 2.87, 1253),
    45: (0.932, 7.65, 2.87, 1254), 46: (0.934, 7.65, 2.87, 1254),
    47: (0.932, 7.66, 2.87, 1253), 48: (0.936, 7.66, 2.87, 1254),
    49: (0.933, 7.66, 2.87, 1254), 50: (0.919, 7.66, 2.87, 1253),
    51: (0.912, 7.65, 2.87, 1255), 52: (0.923, 7.64, 2.87, 1256),
    53: (0.920, 7.64, 2.87, 1256), 54: (0.922, 7.63, 2.86, 1257),
    55: (0.911, 7.61, 2.85, 1261), 56: (0.918, 7.63, 2.86, 1258),
    57: (0.905, 7.62, 2.86, 1260), 58: (0.909, 7.61, 2.86, 1261),
    59: (0.905, 7.61, 2.85, 1261), 60: (0.916, 7.64, 2.86, 1257),
}

# ── Pseudo-label distribution (printed by the train split step) ─────────────
PSEUDO = [  # (class, images, mean confidence)
    ("airplane", 6190, 0.8283), ("bird", 4545, 0.7657), ("car", 5235, 0.8344),
    ("cat", 674, 0.6975), ("deer", 3005, 0.7455), ("dog", 3520, 0.7314),
    ("horse", 4336, 0.7647), ("monkey", 2444, 0.7056), ("ship", 5065, 0.8168),
    ("truck", 6642, 0.8171),
]

# ── Step C PGD-100 partial rows (transcribed from the log; leg still running) ─
# The PGD-100 leg (eps=0.094, 5 seeds × 300 samples) was captured mid-run at
# A (trades_large_baseline) seed 44. (label, seed, acc, dprime)
PGD100_PARTIAL = [
    ("trades_large_baseline", 41, 19.00, 0.2352),
    ("trades_large_baseline", 42, 21.33, 0.1366),
    ("trades_large_baseline", 43, 20.67, 0.6272),
]

# Crossover verdicts from eval_provenance.json (report/sweep_stage2_hpc_only/)
# (checkpoint, diff_pp, threshold_2sig)
CROSSOVER = [
    ("rhan_next_ais_v1_halting_only", 12.13, 4.57),
    ("rhan_next_hpc_only", 7.33, 5.16),
]


def load_rows():
    rows = [json.loads(l) for l in open(DIAG) if l.strip()]
    by_epoch = {r["epoch"]: r for r in rows}
    assert set(by_epoch) == set([1] + list(EPOCH_LINE)), \
        f"diag epochs {sorted(by_epoch)} != expected"
    return rows, by_epoch


def load_sweep():
    """Read the three-way PGD-50 CSVs (synced to HF by the re-run's per-leg sync)."""
    assert os.path.exists(SWEEP_RESULTS), \
        f"missing {SWEEP_RESULTS} — download sweep_stage2_hpc_only_* from " \
        "FerrariKazu/rhan-checkpoints-rolling (dataset) first"
    agg = []   # (label, eps, acc_mean, acc_std, dprime_mean, dprime_std)
    with open(SWEEP_RESULTS, newline="") as f:
        for r in csv.DictReader(f):
            agg.append((r["ckpt_label"], float(r["eps_pixel"]),
                        r["acc_mean"], r["acc_std"],
                        r["macro_dprime_mean"], r["macro_dprime_std"]))
    per_seed = []  # (label, seed, eps, acc, dprime)
    with open(SWEEP_PER_SEED, newline="") as f:
        for r in csv.DictReader(f):
            per_seed.append((r["ckpt_label"], int(r["seed"]),
                             float(r["eps_pixel"]),
                             float(r["acc_pct"]), float(r["macro_dprime"])))
    return agg, per_seed


def fmt(v, nd=4):
    if v is None:
        return "—"
    return f"{v:.{nd}f}"


def main():
    rows, by_epoch = load_rows()
    agg, per_seed = load_sweep()

    lines = []
    A = lines.append

    A("# Stage 2 Run Log — HPC-only (matrix C): 60-epoch training + three-way Step C eval")
    A("")
    A("> Recorded 2026-08-16 from the Kaggle run that completed Step B "
      "(60/60 epochs) and hit the 12 h timeout mid-Step-C PGD-100. Updated "
      "2026-08-18 with the **completed three-way PGD-50 eval** (A baseline / "
      "B AIS-v1 / C HPC-only) from the re-run under commit `3c50d51` (per-cell "
      "`--resume` + per-leg HF sync). Per-epoch numbers below are the exact "
      "rows of the trainer diag `report/rhan_next_hpc_only_diag.jsonl` "
      "(synced to HF); the printed diagnostic blocks in the run log are those "
      "same rows. Step C numbers are the exact HF-synced sweep CSVs "
      "(`report/sweep_stage2_hpc_only/epsilon_sweep_*.csv`).")
    A("")
    A("## 0. Run metadata")
    A("")
    A("| field | value |")
    A("|---|---|")
    A("| config | `RHANNextConfig([HPC(L=1)])` — enable_ais=False, enable_hpc=True, hpc_num_levels=1, w_hpc=0.10 |")
    A("| params | 76,663,734 |")
    A("| base checkpoint | `checkpoints/rhan_next_ais_v1_halting_only_best.pth` (validated AIS-v1) |")
    A("| resume | HF rolling `rhan_next_hpc_only_rolling.pth` @ epoch 34 → resumed from epoch 35 (best val 56.81% at resume) |")
    A("| curriculum | 1–20 @ ε=0.031 (prior session) → 21–40 @ ε=0.062 (lr 0.002) → 41–60 @ ε=0.094 (lr 0.001) |")
    A("| dataset | 5000 real + 41656 pseudo + 0 synthetic = 46656 (41.7% pseudo-kept) |")
    A("| dataloader | num_workers=4, persistent_workers=True, prefetch_factor=4 |")
    A("| diag rows | 27 (prepended epoch-1 resume baseline + epochs 35–60) |")
    A("| final | Training complete, Best **56.81%**; rolling epoch 60; truck-rank WATCH series logged (27 epochs, final rank=3, margin −0.0169) |")
    A("| Step C PGD-50 | **COMPLETE** (three-way A/B/C, 5 seeds × 300 samples, eps ∈ {0.0, 0.094}, norm-space) — 2026-08-18, commit `3c50d51`, provenance `report/sweep_stage2_hpc_only/eval_provenance.json` |")
    A("| Step C PGD-100 | **IN PROGRESS** — A seeds 41–43 done at log cut (A seed 44 running, B/C not started) |")
    A("")
    A("## 1. Pseudo-label distribution (train-split step, this session)")
    A("")
    A("Total pseudo-labeled: **41656 / 100000 (41.7%)**; combined 5000 real + 41656 pseudo = 46656.")
    A("")
    A("| class | images | mean confidence |")
    A("|---|---|---|")
    for c, n, conf in PSEUDO:
        A(f"| {c} | {n} | {conf:.4f} |")
    A("")
    A("## 2. Per-epoch diagnostics (epochs 35–60; epoch 1 = prepended resume baseline)")
    A("")
    A("`eps` = adversarial-noise curriculum phase; β_dyn = dynamic β (mean/std); "
      "`gate α` = foveal gate; `HPC err` = mean HPC prediction error; "
      "`map max/std` = HPC error-map max/std; `truck` = truck Π_D; "
      "`r/m` = truck Π_D rank among top-3 / truck−#2 margin. "
      "Loss/throughput from the printed epoch lines (not in the diag).")
    A("")
    A("| ep | eps | loss | TrAcc% | TeAcc% | img/s | β_dyn | recon | gate α | HPC err | map max | map std | truck | r | m |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for e in [1] + list(EPOCH_LINE):
        r = by_epoch[e]
        loss, thr, eph, secs = EPOCH_LINE.get(e, (None, None, None, None))
        loss_s = f"{loss:.3f}" if loss is not None else "—"
        thr_s = f"{thr:.2f}" if thr is not None else "—"
        beta = f"{r['beta_dyn_mean']:.4f}/{r['beta_dyn_std']:.4f}"
        pd = r["pi_d_per_class"]
        truck = pd.get("truck")
        margin = r.get("truck_pi_d_vs_2_margin")
        margin_s = f"{margin:+.4f}" if margin is not None else "—"
        A(f"| {e} | {r['eps']:.3f} | {loss_s} | {r['tr_acc']:.1f} | "
          f"{r['te_acc']:.1f} | {thr_s} | {beta} | {r['recon_mse']:.4f} | "
          f"{r['gate_alpha']:.4f} | {r['hpc_error_mean']:.4f} | "
          f"{r['hpc_error_map_max']:.4f} | {r['hpc_error_map_std']:.4f} | "
          f"{fmt(truck)} | {r.get('truck_pi_d_rank','—')} | {margin_s} |")
    A("")
    A("## 3. Π_D per class (mean over batch, per epoch)")
    A("")
    hdr = "| ep | " + " | ".join(CLASSES) + " |"
    A(hdr)
    A("|" + "---|" * (len(CLASSES) + 1))
    for e in [1] + list(EPOCH_LINE):
        pd = by_epoch[e]["pi_d_per_class"]
        A("| " + str(e) + " | " + " | ".join(fmt(pd[c]) for c in CLASSES) + " |")
    A("")
    A("## 4. HPC prediction error per class (mean, per epoch)")
    A("")
    A(hdr)
    A("|" + "---|" * (len(CLASSES) + 1))
    for e in [1] + list(EPOCH_LINE):
        he = by_epoch[e]["hpc_error_per_class"]
        A("| " + str(e) + " | " + " | ".join(fmt(he[c]) for c in CLASSES) + " |")
    A("")
    A("## 5. Truck-rank WATCH series (gate amendment 2026-08-16, non-blocking)")
    A("")
    A("| ep | truck Π_D | rank (top-3) | in top-3 | truck−#2 margin |")
    A("|---|---|---|---|---|")
    for e in [1] + list(EPOCH_LINE):
        r = by_epoch[e]
        margin = r.get("truck_pi_d_vs_2_margin")
        margin_s = f"{margin:+.4f}" if margin is not None else "—"
        A(f"| {e} | {fmt(r['pi_d_per_class']['truck'])} | "
          f"{r.get('truck_pi_d_rank','—')} | "
          f"{r.get('truck_pi_d_in_top3','—')} | {margin_s} |")
    A("")
    A("The margin **narrowed** through the 0.062/0.094 phases (−0.0285 @ e10 "
      "in the earlier session → −0.0169 final): truck converged toward #2 "
      "instead of diverging — the WATCH's flag threshold (< −0.05) never "
      "tripped. Truck's per-class HPC error was the **lowest** of the "
      "car/airplane/truck contenders in every logged epoch.")
    A("")
    A("## 6. Step C — PGD-50 matched eval (THREE-WAY: A baseline / B AIS-v1 / C HPC-only)")
    A("")
    A("5 seeds × 300 samples, eps ∈ {0.0, 0.094}, PGD-50, norm-space "
      "(Finding-17 matched convention), baseline `trades_large_baseline`. "
      "Run 2026-08-18 under commit `3c50d51` — the registry now declares C's "
      "trained checkpoint (previously `None`, which silently dropped C from "
      "the first sweep), and both legs use per-cell `--resume` + per-leg HF "
      "sync. Numbers are the exact HF-synced CSVs.")
    A("")
    A("### 6.1 Aggregated (mean ± std over seeds)")
    A("")
    A("| checkpoint | eps | Acc% | d′ |")
    A("|---|---|---|---|")
    for lab, eps, acc_m, acc_s, dp_m, dp_s in agg:
        A(f"| {lab} | {eps:.3f} | {float(acc_m):.2f}±{float(acc_s):.2f} | "
          f"{float(dp_m):.4f}±{float(dp_s):.4f} |")
    A("")
    A("Crossover @ eps=0.094 (criterion: diff > 2·σ_comb):")
    A("")
    A("| checkpoint | diff (pp) | 2·σ_comb | verdict |")
    A("|---|---|---|---|")
    for lab, diff, thresh in CROSSOVER:
        A(f"| {lab} | **+{diff:.2f}** | {thresh:.2f} | **CROSSOVER REAL** |")
    A("")
    A("### 6.2 Per-seed")
    A("")
    A("| checkpoint | seed | eps | Acc% | d′ |")
    A("|---|---|---|---|---|")
    for lab, seed, eps, acc, dp in per_seed:
        A(f"| {lab} | {seed} | {eps:.3f} | {acc:.2f} | {dp:.4f} |")
    A("")
    A("## 7. Step C — PGD-100 leg (eps=0.094, masking re-confirmation)")
    A("")
    A("In progress at the time the log was captured (12 h Kaggle budget, "
      "session still running). Completed cells at the cut:")
    A("")
    A("| checkpoint | seed | Acc% | d′ |")
    A("|---|---|---|---|")
    for lab, seed, acc, dp in PGD100_PARTIAL:
        A(f"| {lab} | {seed} | {acc:.2f} | {dp:.4f} |")
    A("")
    A("Remaining at the cut: A seed 44 (running) + seed 45, then all of "
      "B (AIS-v1) and C (HPC-only). Because the leg syncs to HF only after "
      "completing, and `--resume` skips already-evaluated `(ckpt, seed, eps)` "
      "cells, any re-run continues from exactly the completed cells — "
      "nothing already computed is recomputed.")
    A("")
    A("## 8. Key observations")
    A("")
    A("- Best test acc **56.81%** (set in the epoch-1–34 segment; epochs 35–60 "
      "traded 52.4–56.5% around it — best synced to HF).")
    A("- HPC prediction error converged 0.4432 (e10, prior session) → 0.1506 "
      "(e35) → **0.1492 (e60)**; error-map std pinned ≈0.266, max slowly "
      "crept 1.48 → 1.87 — no collapse/explosion (trend check PASS).")
    A("- Truck Π_D stayed rank 3 the whole segment (car/airplane contested #1: "
      "car held #1 at e35–e40 boundary, airplane at several 0.094-phase "
      "epochs); truck−#2 margin closed from −0.0239 (e11, prior) to −0.0169.")
    A("- β_dyn stepped up with the curriculum: 1.57 (ε=0.062) → 1.94–1.96 "
      "(ε=0.094, min 1.75/max 3.25) — the precision controller responded to "
      "the heavier perturbation.")
    A("- `frac_halted_any: 0.000`, effective steps pinned at the hard cap 4 — "
      "expected for the HPC-only variant (entropy-gated halting is AIS-only).")
    A("- Recon MSE kept falling through the 0.094 phase (0.901 → 0.858) — the "
      "generative prior keeps improving even under the strongest perturbation.")
    A("- **Three-way eval (PGD-50):** HPC-only (C) has the **best clean "
      "accuracy** of the three (55.20±3.67 vs baseline 53.47±2.87, AIS-v1 "
      "49.40±3.48) but the **weakest robustness** of the two RHANNext "
      "variants at ε=0.094 (27.73±2.28 vs AIS-v1 32.53±1.94). Both variants "
      "cross over the TRADES baseline: AIS-v1 **+12.13 pp** (2·σ = 4.57), "
      "HPC-only **+7.33 pp** (2·σ = 5.16) — both CROSSOVER REAL.")
    A("- HPC-only's clean-acc edge over AIS-v1 (+5.8 pp) flips to a **−4.8 pp "
      "robustness deficit** under attack — the auxiliary HPC signal helps "
      "clean generalization but does not add adversarial margin the way the "
      "AIS halting/precision machinery does.")
    A("")

    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
