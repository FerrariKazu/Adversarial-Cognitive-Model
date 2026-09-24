"""
Agent A contract test — the structural-consistency assertion catches
corruption. Deliberately mismatch a summary table against its own cited
CSV and assert the failure is LOUD (AssertionError naming the cell), not
a silent pass.
"""
import pytest
import pandas as pd

from noesis_vision.core.consistency_assert import (
    assert_donor_rows_byte_identical,
    assert_table_matches_csv,
    groupby_summary,
    read_per_seed_csv,
)

LABELS = ["full_step6", "backbone_only"]


def _write_csv(path, label="full_step6", acc=50.0, n_seeds=3):
    rows = []
    for seed in range(41, 41 + n_seeds):
        rows.append({"ckpt_label": label, "seed": seed,
                     "eps_pixel": 0.094, "eps_norm_R": 0.05,
                     "eps_norm_G": 0.05, "eps_norm_B": 0.05,
                     "acc_pct": acc, "macro_dprime": 1.0})
    pd.DataFrame(rows).to_csv(path, index=False)


def _correct_table(csv_paths):
    """Build the table the honest way: from a fresh groupby of the CSVs."""
    dfs = [read_per_seed_csv(p) for p in csv_paths]
    ref = groupby_summary(pd.concat(dfs, ignore_index=True))
    table = ref.rename(columns={
        "acc_pct_mean": "acc_mean", "acc_pct_std": "acc_std",
        "macro_dprime_mean": "macro_dprime_mean",
        "macro_dprime_std": "macro_dprime_std"})
    return table.drop(columns=[c for c in ref.columns
                               if c not in ("ckpt_label", "eps_pixel")
                               and not c.endswith(("_mean", "_std"))])


def test_honest_table_passes(tmp_path):
    # One CSV containing both arms; the honest table is its fresh groupby.
    csv1 = str(tmp_path / "a.csv")
    dfs = []
    for label, acc in (("full_step6", 50.0), ("backbone_only", 45.0)):
        _write_csv(str(tmp_path / f"_{label}.csv"), label, acc=acc)
        dfs.append(read_per_seed_csv(str(tmp_path / f"_{label}.csv")))
    pd.concat(dfs, ignore_index=True).to_csv(csv1, index=False)
    table = _correct_table([csv1])
    ref = assert_table_matches_csv(table, csv1)  # no exception
    assert len(ref) == 2


def test_corrupted_mean_fails_loudly(tmp_path):
    csv1 = str(tmp_path / "a.csv")
    _write_csv(csv1, "full_step6", acc=50.0)
    table = _correct_table([csv1])
    # Corrupt ONE cell's mean — the Stage-3/4-E1 incident shape.
    table.loc[0, "acc_mean"] = table.loc[0, "acc_mean"] + 5.0
    with pytest.raises(AssertionError) as ei:
        assert_table_matches_csv(table, csv1)
    msg = str(ei.value)
    assert "STRUCTURAL CONSISTENCY FAILURE" in msg
    assert "full_step6" in msg and "0.094" in msg  # the (label, eps) cell


def test_corrupted_std_fails_loudly(tmp_path):
    csv1 = str(tmp_path / "a.csv")
    _write_csv(csv1, "full_step6", acc=50.0)
    table = _correct_table([csv1])
    table.loc[0, "acc_std"] = table.loc[0, "acc_std"] + 1.0
    with pytest.raises(AssertionError, match="acc_std"):
        assert_table_matches_csv(table, csv1)


def test_extra_or_missing_cell_fails(tmp_path):
    csv1 = str(tmp_path / "a.csv")
    _write_csv(csv1, "full_step6", acc=50.0)
    table = _correct_table([csv1])
    table.loc[len(table)] = {**table.iloc[0].to_dict(),
                             "ckpt_label": "phantom_arm"}  # never measured
    with pytest.raises(AssertionError, match="phantom_arm"):
        assert_table_matches_csv(table, csv1)


def test_missing_csv_fails(tmp_path):
    table = pd.DataFrame({"ckpt_label": ["x"], "eps_pixel": [0.094],
                          "acc_mean": [50.0], "acc_std": [1.0]})
    with pytest.raises(AssertionError, match="missing"):
        assert_table_matches_csv(table, str(tmp_path / "nope.csv"))


def test_donor_rows_byte_identical(tmp_path):
    csv1 = str(tmp_path / "a.csv")
    _write_csv(csv1, "full_step6", acc=50.0)
    df = read_per_seed_csv(csv1)
    # Honest donor rows (byte-equal to source) pass on every column.
    assert_donor_rows_byte_identical(df.to_dict("records"), csv1)
    # ONE edited value -> loud refusal.
    edited = df.to_dict("records")
    edited[0]["acc_pct"] = 99.0
    with pytest.raises(AssertionError, match="byte-identical"):
        assert_donor_rows_byte_identical(edited, csv1)
    # A summary row without per-seed identity is rejected explicitly —
    # never silently "verified".
    with pytest.raises(AssertionError, match="key columns"):
        assert_donor_rows_byte_identical([{"ckpt_label": "full_step6",
                                           "acc_mean": 50.0}], csv1)
    # Zero matching rows -> nothing verified -> failure.
    with pytest.raises(AssertionError, match="nothing verified"):
        assert_donor_rows_byte_identical(df.to_dict("records"), csv1,
                                         labels=["not_a_label"])
