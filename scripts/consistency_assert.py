#!/usr/bin/env python3
"""
Structural consistency assertion — RHAN-NX rule 1c.
================================================================================

The Stage 3 and Stage 4-E1 incidents (a headline Summary Table silently
diverging from its own per-seed CSV) must not recur. Every report-generation
script in the RHAN-NX task MUST, before writing any Summary Table, assert that
every (checkpoint, eps) cell's mean/std equals a fresh pandas groupby of the
exact CSV being cited as that table's source. Fail loudly, do not write the
table, if the assertion fails.

Two functions:

  * assert_table_matches_csv(table, csv_path, ...) — the headline rule above:
    every (ckpt_label, eps) cell in `table` must reproduce a fresh groupby of
    `csv_path` within tolerance. Raises AssertionError with a cell-by-cell
    diff when it does not.

  * assert_donor_rows_byte_identical(table_rows, source_csv, labels, seeds) —
    comparator-reuse integrity: rows cited as "DONOR row from <source_csv>"
    must match their cited source BYTE-FOR-BYTE (every column), so a reused
    comparator cell can never be silently edited between the source sweep and
    the new report.

Usage (every report generator):

    from scripts.consistency_assert import assert_table_matches_csv, \
        assert_donor_rows_byte_identical

    # BEFORE writing the summary table:
    assert_table_matches_csv(summary_table, "report/sweep_xxx/epsilon_sweep_per_seed.csv")
    assert_donor_rows_byte_identical(donor_rows, "report/.../epsilon_sweep_per_seed.csv",
                                     labels={"rhan_next_ais_hpc"},
                                     seeds=list(range(41, 57)))

Both raise (never write) on mismatch — the caller is expected to let the
exception abort the report write.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

#: Column names of the canonical per-seed eval CSV (matches
#: phase2_attacks/seed_sweep_comparators.py FIELDNAMES and eval_rhan.py output).
CSV_FIELDNAMES = ["ckpt_label", "seed", "eps_pixel",
                  "eps_norm_R", "eps_norm_G", "eps_norm_B",
                  "acc_pct", "macro_dprime"]

#: Default numeric tolerance for the groupby-vs-table comparison (percent
#: points for acc_pct, absolute for macro_dprime). The CSV stores acc_pct
#: rounded to 2 decimals, so a strict 1e-6 tolerance would false-fail on the
#: CSV's own rounding — the mean of rounded values equals the table value only
#: within the CSV's rounding granularity.
ACC_ATOL = 0.02          # pp (CSV stores 2 decimals)
DPRIME_ATOL = 0.001


def read_per_seed_csv(csv_path: str) -> pd.DataFrame:
    """Read a per-seed eval CSV into a DataFrame with typed columns.

    Fails loudly when the file is missing or malformed — a report must never
    silently proceed with a wrong/unreadable source (the E1 incident class).
    """
    if not os.path.exists(csv_path):
        raise AssertionError(
            f"consistency_assert: source CSV missing: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:  # pandas raises a family of errors
        raise AssertionError(
            f"consistency_assert: cannot read {csv_path}: {e}")
    for col in ("ckpt_label", "seed", "eps_pixel", "acc_pct"):
        if col not in df.columns:
            raise AssertionError(
                f"consistency_assert: {csv_path} lacks required column "
                f"{col!r} (columns: {list(df.columns)})")
    df["seed"] = df["seed"].astype(int)
    df["eps_pixel"] = df["eps_pixel"].astype(float).round(4)
    df["acc_pct"] = df["acc_pct"].astype(float)
    if "macro_dprime" in df.columns:
        df["macro_dprime"] = df["macro_dprime"].astype(float)
    return df


def groupby_summary(df: pd.DataFrame,
                    group_cols: Sequence[str] = ("ckpt_label", "eps_pixel"),
                    value_cols: Sequence[str] = ("acc_pct", "macro_dprime"),
                    ) -> pd.DataFrame:
    """Fresh pandas groupby of a per-seed CSV -> one row per (ckpt, eps) cell.

    Columns: ckpt_label, eps_pixel, <value>_mean, <value>_std per value col.
    This is THE reference the summary tables must reproduce.
    """
    value_cols = [c for c in value_cols if c in df.columns]
    g = df.groupby(list(group_cols), sort=False)[value_cols]
    agg = g.agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    out = agg.reset_index()
    return out


def _cell_key(row: Mapping) -> Tuple[str, float]:
    return (str(row["ckpt_label"]), round(float(row["eps_pixel"]), 4))


def assert_table_matches_csv(
        table: pd.DataFrame,
        csv_path: str,
        label_col: str = "ckpt_label",
        eps_col: str = "eps_pixel",
        acc_col: str = "acc_mean",
        acc_std_col: str = "acc_std",
        dprime_col: str = "macro_dprime_mean",
        dprime_std_col: str = "macro_dprime_std",
        acc_atol: float = ACC_ATOL,
        dprime_atol: float = DPRIME_ATOL) -> pd.DataFrame:
    """Assert every (ckpt, eps) cell in `table` matches a fresh groupby of the
    cited CSV. Raises AssertionError (cell-by-cell diff) on any mismatch.

    `table` is the Summary Table DataFrame BEFORE it is written (one row per
    (ckpt_label, eps) cell, with acc_mean / acc_std / macro_dprime_mean /
    macro_dprime_std columns — the naming used by eval_rhan.py provenance).
    Returns the reference groupby (for the caller to embed in provenance).
    """
    df = read_per_seed_csv(csv_path)
    ref = groupby_summary(df)

    # Match on (ckpt_label, eps_pixel) — round both sides identically.
    t = table.copy()
    t["_k"] = t.apply(lambda r: (str(r[label_col]),
                                 round(float(r[eps_col]), 4)), axis=1)
    ref["_k"] = ref.apply(lambda r: (str(r["ckpt_label"]),
                                     round(float(r["eps_pixel"]), 4)), axis=1)
    ref_idx = {k: i for i, k in enumerate(ref["_k"])}

    problems: List[str] = []
    for _, row in t.iterrows():
        k = row["_k"]
        if k not in ref_idx:
            problems.append(f"cell {k} is NOT in {csv_path} (extra/missing row)")
            continue
        r = ref.iloc[ref_idx[k]]
        acc_mean = float(row[acc_col])
        acc_std = float(row[acc_std_col])
        if abs(acc_mean - float(r["acc_pct_mean"])) > acc_atol:
            problems.append(
                f"cell {k}: acc_mean {acc_mean:.4f} != CSV groupby "
                f"{float(r['acc_pct_mean']):.4f} (delta "
                f"{acc_mean - float(r['acc_pct_mean']):+.4f} pp)")
        if abs(acc_std - float(r["acc_pct_std"])) > acc_atol:
            problems.append(
                f"cell {k}: acc_std {acc_std:.4f} != CSV groupby "
                f"{float(r['acc_pct_std']):.4f}")
        if dprime_col in row and "macro_dprime_mean" in ref.columns:
            d_mean = float(row[dprime_col])
            if abs(d_mean - float(r["macro_dprime_mean"])) > dprime_atol:
                problems.append(
                    f"cell {k}: {dprime_col} {d_mean:.4f} != CSV groupby "
                    f"{float(r['macro_dprime_mean']):.4f}")
            if dprime_std_col in row:
                d_std = float(row[dprime_std_col])
                if abs(d_std - float(r["macro_dprime_std"])) > dprime_atol:
                    problems.append(
                        f"cell {k}: {dprime_std_col} {d_std:.4f} != CSV "
                        f"groupby {float(r['macro_dprime_std']):.4f}")

    if problems:
        head = "\n".join(problems[:40])
        more = "" if len(problems) <= 40 else f"\n  ... and {len(problems)-40} more"
        raise AssertionError(
            f"STRUCTURAL CONSISTENCY FAILURE: Summary Table diverges from its "
            f"own source CSV {csv_path}:\n  {head}{more}\n"
            f"Refusing to write the table (RHAN-NX rule 1c).")
    return ref


def assert_donor_rows_byte_identical(
        donor_rows: Iterable[Mapping],
        source_csv: str,
        labels: Optional[Sequence[str]] = None,
        seeds: Optional[Sequence[int]] = None,
        key_cols: Sequence[str] = ("ckpt_label", "seed", "eps_pixel"),
        all_columns: bool = True) -> None:
    """Assert comparator-reuse rows are byte-identical to their cited source.

    Every row a report labels "DONOR row from <source_csv>" must equal the
    source CSV's row for the same (label, seed, eps) key on EVERY column
    (when all_columns=True — the default and the honest check) so a reused
    comparator cell can never be silently edited between sweeps.

    Args:
        donor_rows: the rows the report intends to reuse (as dicts / Series).
        source_csv: the cited donor CSV path.
        labels: restrict the check to these ckpt_labels (None = all).
        seeds: restrict to these seeds (None = all).
        key_cols: columns forming the row identity.
        all_columns: when True (default), compare ALL shared columns
            byte-for-byte; when False, compare only the key + acc_pct +
            macro_dprime (the values a report would cite).

    Raises AssertionError listing every mismatched / absent donor row.
    """
    df = read_per_seed_csv(source_csv)
    src = {(str(r["ckpt_label"]), int(r["seed"]),
            round(float(r["eps_pixel"]), 4)): r.to_dict()
           for _, r in df.iterrows()}

    problems: List[str] = []
    n_checked = 0
    for row in donor_rows:
        key = tuple(row.get(c) for c in key_cols)
        if len(key) == 3:
            key = (str(key[0]), int(key[1]), round(float(key[2]), 4))
        if labels is not None and str(key[0]) not in set(labels):
            continue
        if seeds is not None and int(key[1]) not in set(seeds):
            continue
        n_checked += 1
        if key not in src:
            problems.append(f"donor row {key} not present in {source_csv}")
            continue
        s = src[key]
        if all_columns:
            compare_cols = [c for c in set(row.keys()) & set(s.keys())
                            if c not in key_cols]
        else:
            compare_cols = [c for c in ("acc_pct", "macro_dprime")
                            if c in row and c in s]
        for c in compare_cols:
            a, b = str(row[c]), str(s[c])
            if a != b:
                problems.append(
                    f"donor row {key}: column {c!r} '{a}' != source '{b}' "
                    f"({source_csv}) — reused rows must be byte-identical")

    if n_checked == 0:
        raise AssertionError(
            f"donor-row check: no donor rows matched labels={labels} "
            f"seeds={seeds} in the provided rows — nothing verified")
    if problems:
        head = "\n".join(problems[:40])
        more = "" if len(problems) <= 40 else f"\n  ... and {len(problems)-40} more"
        raise AssertionError(
            f"DONOR-ROW INTEGRITY FAILURE vs {source_csv}:\n  {head}{more}\n"
            f"Refusing to cite these rows as comparator donors (rule 1b/1c).")