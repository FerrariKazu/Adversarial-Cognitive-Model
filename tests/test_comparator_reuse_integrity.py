"""
Comparator-reuse integrity tests (rules 1b/1c).
================================================================================

Pins the two integrity guarantees of the RHAN-NX comparison protocol:

  1. load_comparator() returns the EXACT validated rows from the cited source
     CSV — never re-runs PGD, never silently returns fewer rows:
       * unknown label -> KeyError;
       * requested seeds NOT a subset of the comparator's validated seeds ->
         loud AssertionError (never a silent partial return);
       * source CSV missing locally AND on HF -> loud AssertionError
         (the rows a report would cite do not exist);
       * local checkpoint present with a mismatched sha256 -> loud
         AssertionError (swapped/corrupted checkpoint must never be cited
         with its old numbers).
  2. DONOR rows are byte-identical to their source CSV: a report may only
     label a row "DONOR row from <csv>" if every column matches the source
     byte-for-byte — assert_donor_rows_byte_identical enforces it, and
     assert_table_matches_csv enforces the headline rule (every Summary
     Table cell reproduces a fresh groupby of its cited CSV).

The D / baseline entries point at the completed E1 sweep CSV, which exists
in the repo (report/sweep_stage4_e1_d_e1_pgd100/epsilon_sweep_per_seed.csv)
— those tests run against real validated rows. The B / C entries point at
CSVs absent locally (HF-only), so their missing-file path is exercised with
no HF token: the module must raise loudly, never fabricate.
"""
import csv
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.comparator_registry as cr
from scripts.consistency_assert import (
    assert_donor_rows_byte_identical,
    assert_table_matches_csv,
    groupby_summary,
    read_per_seed_csv,
)

E1_CSV = ROOT / "report" / "sweep_stage4_e1_d_e1_pgd100" / "epsilon_sweep_per_seed.csv"
E2_CSV = ROOT / "report" / "sweep_stage2_hpc_only" / "epsilon_sweep_per_seed.csv"


# ── load_comparator: exact rows, verified ───────────────────────────────────

def test_load_comparator_returns_exact_rows():
    rows = cr.load_comparator("rhan_next_ais_hpc", range(41, 57))
    src = read_per_seed_csv(str(E1_CSV))
    expected = src[src["ckpt_label"] == "rhan_next_ais_hpc"]
    expected = expected[expected["seed"].isin(range(41, 57))]
    assert len(rows) == len(expected) == 32        # 16 seeds x 2 eps
    # Exact values, untouched.
    pd.testing.assert_frame_equal(
        rows.reset_index(drop=True)[["ckpt_label", "seed", "eps_pixel",
                                     "acc_pct", "macro_dprime"]],
        expected.reset_index(drop=True)[["ckpt_label", "seed", "eps_pixel",
                                         "acc_pct", "macro_dprime"]])


def test_load_comparator_subset_ok():
    rows = cr.load_comparator("trades_large_baseline", [41, 42, 43])
    assert sorted(rows["seed"].astype(int).unique()) == [41, 42, 43]
    assert len(rows) == 6                          # 3 seeds x 2 eps


def test_load_comparator_unknown_label_raises():
    with pytest.raises(KeyError):
        cr.load_comparator("not_a_model", [41])


def test_load_comparator_unvalidated_seed_raises():
    """Asking B (validated 41-48) for seed 56 must RAISE, never silently
    return fewer rows than expected."""
    with pytest.raises(AssertionError, match="NEVER evaluated"):
        cr.load_comparator("rhan_next_ais_v1_halting_only", [41, 56])


def test_load_comparator_b_missing_csv_raises():
    """B's source CSV is HF-only; with no HF token the module must raise
    loudly rather than fabricate rows."""
    os.environ.pop("HF_TOKEN", None)
    with pytest.raises(AssertionError, match="not found locally"):
        cr.load_comparator("rhan_next_ais_v1_halting_only", [41, 42])


def test_load_comparator_swapped_checkpoint_detected(monkeypatch):
    """A local checkpoint whose sha256 differs from the registry is caught."""
    # Corrupt the registry's recorded hash for D -> the local D checkpoint
    # (present in the repo) will mismatch -> loud failure.
    entry = dict(cr.COMPARATOR_REGISTRY["rhan_next_ais_hpc"])
    entry["sha256_checkpoint"] = "0" * 64
    monkeypatch.setitem(cr.COMPARATOR_REGISTRY, "rhan_next_ais_hpc", entry)
    if not (ROOT / "checkpoints" / "rhan_next_ais_hpc_best.pth").exists():
        pytest.skip("D checkpoint not present locally")
    with pytest.raises(AssertionError, match="swapped/corrupted"):
        cr.load_comparator("rhan_next_ais_hpc", [41])


def test_seed_mismatch_warning_only_for_b_c():
    assert cr.seed_mismatch_warning("rhan_next_ais_v1_halting_only")
    assert cr.seed_mismatch_warning("rhan_next_hpc_only")
    assert cr.seed_mismatch_warning("rhan_next_ais_hpc") is None
    assert cr.seed_mismatch_warning("trades_large_baseline") is None


# ── DONOR byte-identity + headline consistency ──────────────────────────────

def test_donor_rows_byte_identical_pass():
    rows = cr.load_comparator("rhan_next_ais_hpc", range(41, 57))
    assert_donor_rows_byte_identical(
        rows.to_dict("records"),
        str(E1_CSV),
        labels=["rhan_next_ais_hpc"],
        seeds=range(41, 57))


def test_donor_rows_byte_identical_detects_edit():
    rows = cr.load_comparator("rhan_next_ais_hpc", [41, 42]).to_dict("records")
    tampered = dict(rows[0])
    tampered["acc_pct"] = str(float(tampered["acc_pct"]) + 5.0)
    with pytest.raises(AssertionError, match="byte-identical"):
        assert_donor_rows_byte_identical([tampered], str(E1_CSV),
                                         labels=["rhan_next_ais_hpc"])


def test_donor_rows_absent_key_detected():
    """A donor row key that does not exist in the source CSV is caught."""
    bogus = {"ckpt_label": "rhan_next_ais_hpc", "seed": 99,
             "eps_pixel": 0.0, "acc_pct": "56.0"}
    with pytest.raises(AssertionError, match="not present"):
        assert_donor_rows_byte_identical([bogus], str(E1_CSV),
                                         labels=["rhan_next_ais_hpc"])


def test_donor_check_nothing_matched_raises():
    rows = cr.load_comparator("rhan_next_ais_hpc", [41]).to_dict("records")
    with pytest.raises(AssertionError, match="nothing verified"):
        assert_donor_rows_byte_identical(rows, str(E1_CSV),
                                         labels=["nonexistent_label"])


def test_assert_table_matches_csv_pass_and_fail():
    """A summary table built from the CSV's own groupby passes; a hand-edited
    cell fails loudly (the Stage 3 / E1 divergence incident, prevented)."""
    df = read_per_seed_csv(str(E1_CSV))
    ref = groupby_summary(df)
    table = pd.DataFrame({
        "ckpt_label": ref["ckpt_label"],
        "eps_pixel": ref["eps_pixel"],
        "acc_mean": ref["acc_pct_mean"],
        "acc_std": ref["acc_pct_std"],
        "macro_dprime_mean": ref["macro_dprime_mean"],
        "macro_dprime_std": ref["macro_dprime_std"],
    })
    assert_table_matches_csv(table, str(E1_CSV))     # must pass

    bad = table.copy()
    bad.loc[0, "acc_mean"] = bad.loc[0, "acc_mean"] + 3.0
    with pytest.raises(AssertionError, match="STRUCTURAL CONSISTENCY"):
        assert_table_matches_csv(bad, str(E1_CSV))


def test_assert_table_matches_csv_missing_source():
    table = pd.DataFrame({"ckpt_label": ["x"], "eps_pixel": [0.0],
                          "acc_mean": [1.0], "acc_std": [0.1]})
    with pytest.raises(AssertionError, match="missing"):
        assert_table_matches_csv(table, "report/does_not_exist/x.csv")


# ── registry metadata sanity ────────────────────────────────────────────────

def test_registry_seed_counts_recorded():
    assert cr.COMPARATOR_REGISTRY["rhan_next_ais_hpc"]["validated_seeds"] == \
        list(range(41, 57))
    assert cr.COMPARATOR_REGISTRY["rhan_next_ais_v1_halting_only"]["validated_seeds"] == \
        list(range(41, 49))       # 8 seeds
    assert cr.COMPARATOR_REGISTRY["rhan_next_hpc_only"]["validated_seeds"] == \
        list(range(41, 46))       # 5 seeds


def test_registry_checkpoint_hashes_match_roadmap():
    """The registry's recorded checkpoint hashes must match the validated
    roadmap records (B and C are pinned in docs/rhan_next_roadmap.json)."""
    assert cr.COMPARATOR_REGISTRY["rhan_next_ais_v1_halting_only"]["sha256_checkpoint"] \
        == "19582ff4b32afdb2a46e88a7089167844a2ac3b24d01d6c34cac79ebbf1118e3"
    assert cr.COMPARATOR_REGISTRY["rhan_next_hpc_only"]["sha256_checkpoint"] \
        == "5b7dce1e37e9e26d95a742f8a6dc0ec4e1c895766ca42e0316ec78a90f6f55a0"
    assert cr.COMPARATOR_REGISTRY["trades_large_baseline"]["sha256_checkpoint"] \
        == "37a4eee0c37b77adc291cb45cc19310de1249c7cd81b0e779b2ebaea6d466afc"