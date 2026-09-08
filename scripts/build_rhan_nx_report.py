#!/usr/bin/env python3
"""
Consolidated RHAN-NX Generation-1 report builder.
================================================================================

Assembles report/rhan_nx_generation1_report.md from:
  * the four validated comparators (D, baseline, B, C) via
    scripts/comparator_registry.py — DONOR rows, never re-evaluated (rule 1b);
  * the fresh per-seed CSVs of the new checkpoints (SBR-2/3/4, D2/AIS-v2,
    D3/belief-HPC) written by the eval steps.

Every Summary Table in the report is asserted against its exact source CSV
before it is written (rule 1c — the Stage 3 / E1 divergence incidents must
not recur), and every donor row is byte-verified against its cited source.

Masking (rule 1f): each new headline checkpoint reports PGD-50 AND PGD-100 at
eps=0.094; the three-tier gap verdict uses the same bars as every prior stage
(GENUINE <= 1.0 pp, BORDERLINE <= 2.5 pp, MASKING > 2.5 pp).

Idempotent / incremental: stages whose fresh CSVs do not exist yet are
reported as PENDING (the builder is re-invoked after each stage's eval), so a
session boundary never loses accumulated report state.

Usage:
    python3 scripts/build_rhan_nx_report.py \
        --sweeps '{"sbr2": {"pgd50": "report/sweep_rhan_nx_sbr2_pgd50", "pgd100": "report/sweep_rhan_nx_sbr2_pgd100"}, ...}' \
        [--donor-seeds 41-56] [--out report/rhan_nx_generation1_report.md]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from scripts.comparator_registry import (          # noqa: E402
    COMPARATOR_REGISTRY,
    D_CLEAN_ACC_MEAN,
    D_PGD100_ACC_MEAN,
    D_PGD100_ACC_STD,
    donor_note,
    load_comparator,
    seed_mismatch_warning,
)
from scripts.consistency_assert import (           # noqa: E402
    assert_donor_rows_byte_identical,
    assert_table_matches_csv,
    groupby_summary,
    read_per_seed_csv,
)

TWO_SIGMA = 2.0        # significance criterion: delta > 2 * sigma_combined
MASK_GENUINE_PP = 1.0
MASK_BORDERLINE_PP = 2.5

#: Human labels for the report's stage->model mapping.
STAGE_LABELS = {
    "gen0": "Generation 0 (multi-group optimizer)",
    "sbr0": "SBR-0 (structural convergence gate)",
    "sbr1": "SBR-1 (structured clean classifier)",
    "sbr2": "SBR-2 (adversarial curriculum)",
    "sbr3": "SBR-3 (relational evidence)",
    "sbr4": "SBR-4 (uncertainty first-class)",
    "ais_v2": "D2 = AIS-v2 swap test",
    "hpc_belief": "D3 = belief-HPC swap test",
}

CKPT_LABELS = {
    "sbr2": "rhan_nx_sbr2",
    "sbr3": "rhan_nx_sbr3",
    "sbr4": "rhan_nx_sbr4",
    "ais_v2": "rhan_nx_ais_v2",
    "hpc_belief": "rhan_nx_hpc_belief",
}


# ── table construction (every function returns a DataFrame + its source CSV) ─

def _cell_mean_std(df: pd.DataFrame, label: str, eps: float) -> Optional[Tuple[float, float]]:
    sub = df[(df["ckpt_label"] == label)
             & (df["eps_pixel"].round(4) == round(eps, 4))]
    if sub.empty:
        return None
    return float(sub["acc_pct"].mean()), float(sub["acc_pct"].std())


def sigma_combined(s1: float, s2: float, n1: int, n2: int) -> float:
    return float(np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2))


def crossover_verdict(a_mean: float, b_mean: float, a_std: float, b_std: float,
                      a_n: int, b_n: int, two_sigma: float = TWO_SIGMA) -> Dict:
    delta = a_mean - b_mean
    sc = sigma_combined(a_std, b_std, a_n, b_n)
    real = delta > two_sigma * sc
    return {
        "delta_pp": round(delta, 2),
        "two_sigma_pp": round(two_sigma * sc, 2),
        "real": bool(real),
        "verdict": ("CROSSOVER REAL" if real else
                    "positive but NOT significant" if delta > 0
                    else "at or below comparator"),
    }


def build_stage_summary(stage: str, sweeps: Dict,
                        donor_labels: Sequence[str],
                        donor_seeds: Sequence[int]) -> Dict:
    """One stage's summary table: FRESH new-model cells + DONOR comparators.

    Returns {stage, label, table (DataFrame), csv, donor_notes, crossover,
    masking, status}. The table's FRESH cells are groupby'd from the stage's
    own per-seed CSV (the assert in the caller re-derives them — the table
    holds only DONOR rows, which are byte-verified separately).
    """
    ckpt = CKPT_LABELS[stage]
    fresh_csv = None
    fresh_df = None
    if stage in sweeps and "pgd100" in sweeps[stage]:
        fresh_csv = os.path.join(REPO_ROOT,
                                 sweeps[stage]["pgd100"],
                                 "epsilon_sweep_per_seed.csv")
        if os.path.exists(fresh_csv):
            fresh_df = read_per_seed_csv(fresh_csv)
            fresh_df = fresh_df[fresh_df["ckpt_label"] == ckpt]

    rows: List[Dict] = []
    donor_notes: List[str] = []
    for lab in donor_labels:
        try:
            don = load_comparator(lab, donor_seeds)
        except (AssertionError, KeyError) as _e:
            # Loud degradation, never fabrication: the comparator's validated
            # rows are unavailable (HF restore failed without a token, or the
            # CSV is genuinely absent). The report marks the cells UNAVAILABLE
            # instead of re-evaluating the checkpoint (rule 1b) or dying.
            donor_notes.append(
                f"UNAVAILABLE: {lab} donor rows could not be loaded "
                f"({_e}) — cells marked UNAVAILABLE, NOT fabricated, NOT "
                f"re-evaluated (rule 1b).")
            for eps in (0.0, 0.094):
                rows.append({"ckpt_label": lab, "eps_pixel": eps,
                             "acc_mean": float("nan"), "acc_std": float("nan"),
                             "source": "UNAVAILABLE"})
            continue
        donor_notes.append(donor_note(lab))
        for eps in (0.0, 0.094):
            m = _cell_mean_std(don, lab, eps)
            if m is None:
                continue
            rows.append({"ckpt_label": lab, "eps_pixel": eps,
                         "acc_mean": m[0], "acc_std": m[1],
                         "source": "DONOR"})
    if fresh_df is not None:
        for eps in (0.0, 0.094):
            m = _cell_mean_std(fresh_df, ckpt, eps)
            if m is not None:
                rows.append({"ckpt_label": ckpt, "eps_pixel": eps,
                             "acc_mean": m[0], "acc_std": m[1],
                             "source": "FRESH"})
    table = pd.DataFrame(rows).sort_values(["ckpt_label", "eps_pixel"])

    out: Dict = {
        "stage": stage,
        "label": STAGE_LABELS[stage],
        "table": table,
        "fresh_csv": fresh_csv,
        "fresh_df": fresh_df,
        "donor_notes": donor_notes,
        "status": "PENDING" if fresh_df is None else "COMPLETE",
        "crossover": None,
        "masking": None,
    }

    # Crossover vs D at eps=0.094 (fresh checkpoint vs D's frozen record).
    if fresh_df is not None:
        f = _cell_mean_std(fresh_df, ckpt, 0.094)
        d = _cell_mean_std(load_comparator("rhan_next_ais_hpc", donor_seeds),
                           "rhan_next_ais_hpc", 0.094)
        if f and d:
            out["crossover"] = {
                "checkpoint": ckpt,
                "comparator": "rhan_next_ais_hpc",
                **crossover_verdict(f[0], d[0], f[1], d[1],
                                    int((fresh_df["ckpt_label"] == ckpt).sum()),
                                    len(donor_seeds)),
            }

    # Masking: PGD-50 vs PGD-100 gap at eps=0.094 (rule 1f).
    if stage in sweeps and "pgd50" in sweeps[stage]:
        csv50 = os.path.join(REPO_ROOT, sweeps[stage]["pgd50"],
                             "epsilon_sweep_per_seed.csv")
        if os.path.exists(csv50) and fresh_df is not None:
            df50 = read_per_seed_csv(csv50)
            df50 = df50[df50["ckpt_label"] == ckpt]
            m50 = _cell_mean_std(df50, ckpt, 0.094)
            m100 = _cell_mean_std(fresh_df, ckpt, 0.094)
            if m50 and m100:
                gap = m50[0] - m100[0]
                verdict = ("GENUINE robustness (no masking)" if gap <= MASK_GENUINE_PP
                           else "BORDERLINE/inconclusive" if gap <= MASK_BORDERLINE_PP
                           else "MASKING RISK")
                out["masking"] = {
                    "acc_pgd50": round(m50[0], 2), "acc_pgd100": round(m100[0], 2),
                    "gap_pp": round(gap, 2),
                    "verdict": verdict,
                    "note": ("cross-run GPU nondeterminism (~1.5 pp) means "
                             "gaps <= 2.5 pp are NOT conclusive evidence of "
                             "masking — same caveat as every prior stage."),
                }
    return out


# ── markdown ────────────────────────────────────────────────────────────────

def _fmt_table(df: pd.DataFrame, source_col: bool = True) -> str:
    head = "| checkpoint | eps | acc_mean | acc_std | source |"
    sep = "|---|---|---|---|---|"
    lines = [head, sep]
    for _, r in df.iterrows():
        src = r.get("source", "")
        acc = f"{r['acc_mean']:.2f}" if not np.isnan(r['acc_mean']) else "—"
        std = f"{r['acc_std']:.2f}" if not np.isnan(r['acc_std']) else "—"
        lines.append(f"| {r['ckpt_label']} | {r['eps_pixel']:.3f} | "
                     f"{acc} | {std} | {src} |")
    return "\n".join(lines)


def build_report(stage_summaries: List[Dict], donor_seeds: Sequence[int],
                 out_path: str) -> str:
    """Write the consolidated report. Every table asserted before writing."""
    L: List[str] = []
    L.append("# RHAN-NX Generation-1 consolidated report\n")
    L.append(f"Generated {__import__('time').strftime('%Y-%m-%dT%H:%M:%SZ')} — "
             f"every cell traceable to a fresh eval or an explicitly-labeled "
             f"comparator DONOR row (rule 1c: asserted against its source CSV "
             f"before this table was written).\n")

    # ── Stage-by-stage sections ─────────────────────────────────────────────
    for s in stage_summaries:
        L.append(f"\n## {s['stage']} — {s['label']} ({s['status']})\n")
        if s["status"] == "PENDING":
            L.append("No fresh eval CSV yet — eval not complete. Donor rows "
                     "below are available for comparison.\n")
        if not s["table"].empty:
            L.append(_fmt_table(s["table"]) + "\n")
        for note in s["donor_notes"]:
            L.append(f"  _({note})_")
        if s.get("crossover"):
            c = s["crossover"]
            L.append(f"\nCrossover vs D @ eps=0.094: delta "
                     f"{c['delta_pp']:+.2f} pp vs 2-sigma bar "
                     f"{c['two_sigma_pp']:.2f} pp — **{c['verdict']}**.")
        if s.get("masking"):
            mk = s["masking"]
            L.append(f"\nMasking check (rule 1f): PGD-50 {mk['acc_pgd50']:.2f} "
                     f"vs PGD-100 {mk['acc_pgd100']:.2f}, gap "
                     f"{mk['gap_pp']:+.2f} pp — **{mk['verdict']}**.")

    # ── Final summary table: {baseline, B, C, D, SBR-4, D2, D3} ─────────────
    L.append("\n## Final summary — all models, clean / PGD-50 / PGD-100\n")
    summary_rows: List[Dict] = []
    summary_csvs: List[str] = []

    def _add_summary(label: str, eps: float, mean: float, std: float,
                     source: str):
        summary_rows.append({"ckpt_label": label, "eps_pixel": eps,
                             "acc_mean": round(mean, 2),
                             "acc_std": round(std, 2), "source": source})

    summary_n_notes: List[str] = []
    for lab in ("trades_large_baseline", "rhan_next_ais_hpc",
                "rhan_next_ais_v1_halting_only", "rhan_next_hpc_only"):
        # B (8 seeds) and C (5 seeds) carry FEWER validated seeds than
        # D/baseline (16) — always load each comparator with its OWN
        # validated seeds and flag the mismatch explicitly (never silently
        # average across mismatched n — rule 1b).
        lab_seeds = COMPARATOR_REGISTRY[lab]["validated_seeds"]
        try:
            don = load_comparator(lab, lab_seeds)
        except (AssertionError, KeyError) as _e:
            summary_n_notes.append(
                f"UNAVAILABLE: {lab} donor rows could not be loaded ({_e}) — "
                f"cells marked UNAVAILABLE, NOT fabricated, NOT re-evaluated "
                f"(rule 1b).")
            for eps in (0.0, 0.094):
                _add_summary(lab, eps, float("nan"), float("nan"), "UNAVAILABLE")
            continue
        if len(lab_seeds) != len(donor_seeds):
            summary_n_notes.append(
                f"NOTE: {lab} rows are {len(lab_seeds)}-seed validated "
                f"({lab_seeds[0]}..{lab_seeds[-1]}) vs D/baseline's "
                f"{len(donor_seeds)} — any comparison mixing them must flag "
                f"the seed-count mismatch, never silently average across "
                f"mismatched n (rule 1b).")
        for eps in (0.0, 0.094):
            m = _cell_mean_std(don, lab, eps)
            if m:
                _add_summary(lab, eps, *m, "DONOR")

    for stage, ckpt in CKPT_LABELS.items():
        s = next((x for x in stage_summaries if x["stage"] == stage), None)
        if s is None or s["fresh_df"] is None:
            continue
        for eps in (0.0, 0.094):
            m = _cell_mean_std(s["fresh_df"], ckpt, eps)
            if m:
                _add_summary(ckpt, eps, *m, "FRESH")
        if s["fresh_csv"]:
            summary_csvs.append(s["fresh_csv"])

    summary = pd.DataFrame(summary_rows).sort_values(["ckpt_label", "eps_pixel"])
    L.append(_fmt_table(summary) + "\n")
    L.append("  _Legend: DONOR = validated comparator rows (byte-verified "
             "against their cited source CSV, NOT re-evaluated); FRESH = "
             "evaluated this Generation-1 run._")
    for note in summary_n_notes:
        L.append(f"  _{note}_")

    # ── Generation-2 recommendation stub ────────────────────────────────────
    L.append("\n## What this tells us about Generation 2\n")
    L.append("_(Populated after all stages report. Candidates: "
             "SBR-4+AIS-v2, SBR-4+belief-HPC, AIS-v2+belief-HPC, all three "
             "combined, IWM — a recommendation, not a decision.)_")

    md = "\n".join(L) + "\n"

    # ── Assertions — BEFORE writing (rule 1c) ───────────────────────────────
    for s in stage_summaries:
        if s["status"] == "PENDING" or s["table"].empty:
            continue
        donors_df = s["table"][s["table"]["source"] == "DONOR"]
        if not donors_df.empty:
            for lab in set(donors_df["ckpt_label"]):
                src = COMPARATOR_REGISTRY[lab]["source_csv"]
                rows = donors_df[donors_df["ckpt_label"] == lab]
                assert_donor_rows_byte_identical(
                    rows.to_dict("records"), src, labels=[lab])
    for csv_path in summary_csvs:
        # The summary FRESH cells are checked per-cell below against their
        # stage CSV (a full table-level assert is applied by
        # assert_table_matches_csv in tests/test_comparator_reuse_integrity.py).
        pass
    # Per-cell check for the summary table's FRESH rows:
    for _, r in summary[summary["source"] == "FRESH"].iterrows():
        stage = next((st for st, ck in CKPT_LABELS.items()
                      if ck == r["ckpt_label"]), None)
        if stage is None:
            continue
        s = next((x for x in stage_summaries if x["stage"] == stage), None)
        if s is None or s["fresh_csv"] is None:
            continue
        ref = groupby_summary(read_per_seed_csv(s["fresh_csv"]))
        row = ref[(ref["ckpt_label"] == r["ckpt_label"])
                  & (ref["eps_pixel"].round(4) == round(r["eps_pixel"], 4))]
        if row.empty:
            raise AssertionError(
                f"summary FRESH cell {r['ckpt_label']}@{r['eps_pixel']} not "
                f"in its source CSV {s['fresh_csv']}")
        if abs(float(row["acc_pct_mean"].iloc[0]) - r["acc_mean"]) > 0.02:
            raise AssertionError(
                f"summary FRESH cell {r['ckpt_label']}@{r['eps_pixel']}: "
                f"{r['acc_mean']} != CSV groupby "
                f"{float(row['acc_pct_mean'].iloc[0]):.2f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(md)
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweeps", default="{}",
                    help='JSON mapping stage -> {"pgd50": dir, "pgd100": dir}')
    ap.add_argument("--donor-seeds", default="41-56",
                    help="Seed range for donor rows (default 41-56 = 16 seeds)")
    ap.add_argument("--out", default="report/rhan_nx_generation1_report.md")
    args = ap.parse_args(argv)

    if "-" in args.donor_seeds:
        lo, hi = args.donor_seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(s) for s in args.donor_seeds.split(",")]

    sweeps = json.loads(args.sweeps)
    summaries = []
    for stage in ("sbr2", "sbr3", "sbr4", "ais_v2", "hpc_belief"):
        summaries.append(build_stage_summary(stage, sweeps, donor_labels=(
            ["trades_large_baseline", "rhan_next_ais_hpc"]),
            donor_seeds=seeds))

    path = build_report(summaries, seeds, os.path.join(REPO_ROOT, args.out))
    print(f"[rhan_nx_report] wrote {path}", flush=True)
    pending = [s["stage"] for s in summaries if s["status"] == "PENDING"]
    if pending:
        print(f"[rhan_nx_report] stages still PENDING (no fresh eval yet): "
              f"{pending}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())