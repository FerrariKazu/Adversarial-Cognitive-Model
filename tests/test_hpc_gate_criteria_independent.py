"""Stage 2 — gate criteria are scored and reported INDEPENDENTLY (REQUIRED gate test).

2026-08-15: checks 2 (HPC error trend) and 4 (Π_D car/truck ordering) were
both labeled from the SAME combined boolean `_v2["healthy"]`. A Π_D failure
therefore flipped the trend check to FAIL even when the trend had genuinely
passed — the smoke-v4 run printed "HPC prediction error trend OK: ratio
0.23" under a `[2_hpc_error_trend] FAIL` banner, and the real failure was
only `[4_pi_d_car_truck]`.

2026-08-16 (gate AMENDMENT): check 4 was replaced by a two-tier criterion —
BLOCKING only on genuine collapse/explosion (any class outside the
Stage-1-validated reference envelope [PI_D_REF_BAND_LO, PI_D_REF_BAND_HI])
or car losing #1; truck's rank / truck-vs-#2 margin became a NON-BLOCKING
WATCH metric (roadmap stages['2'].watch_metrics). This pins the amended
behavior: the real v4 smoke — whose truck/airplane rank-flicker previously
FAILED the gate — must now re-score HEALTHY without retraining, while a
class leaving the reference envelope or car losing #1 must still block.

The REAL gate is exercised: the notebooks are executable scripts (they train
on import), so the functions are extracted from their source via AST and run
in a minimal namespace — the same pattern as test_hpc_gate_trend_across_resume.py.
Both notebooks are tested so a future edit to one gate can never silently
drift from the other.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = [
    ROOT / "cloud_setup" / "colab_notebook_noesis.py",
    ROOT / "cloud_setup" / "Kaggle_NOESIS.py",
]

# Real smoke-v4 telemetry (HF report/rhan_next_hpc_only_smoke_v4_diag.jsonl).
# Epoch 1 -> epoch 15 trend passes (ratio 0.23); the final row's Π_D top-2 is
# car/airplane with truck THIRD — the rank-flicker the amendment stops gating
# on. All classes sit inside the reference envelope [0.1722, 0.4616] and car
# holds #1, so the amended gate reads HEALTHY with a truck WATCH.
ROW1 = {"epoch": 1, "eps": 0.031, "hpc_error_mean": 0.686788,
        "hpc_error_map_min": 9.1e-05, "hpc_error_map_max": 1.152582,
        "hpc_error_map_std": 0.277792,
        "pi_d_per_class": {"car": 0.42, "truck": 0.38, "airplane": 0.36}}
ROW15_V4 = {"epoch": 15, "eps": 0.031, "hpc_error_mean": 0.160782,
            "hpc_error_map_min": 2.1e-05, "hpc_error_map_max": 1.871881,
            "hpc_error_map_std": 0.268991,
            "pi_d_per_class": {"airplane": 0.3360, "car": 0.3425,
                               "truck": 0.3237}}
# Flat trend (epoch-1 0.6868 -> 0.6906, ratio ~1.01 -> no >=10% decrease) but
# Π_D within the envelope: the mirror-direction regression.
ROW15_FLAT_HEALTHY_PI_D = {"epoch": 15, "eps": 0.031,
                           "hpc_error_mean": 0.690600,
                           "hpc_error_map_min": 1e-4, "hpc_error_map_max": 1.9986,
                           "hpc_error_map_std": 0.5072,
                           "pi_d_per_class": {"car": 0.40, "truck": 0.39,
                                              "airplane": 0.37}}
# Genuine collapse: airplane's Π_D collapses to 0.05 (way below the 0.1722
# band floor) while everything else stays put — must BLOCK.
ROW15_COLLAPSE = {"epoch": 15, "eps": 0.031, "hpc_error_mean": 0.160782,
                  "hpc_error_map_min": 2.1e-05, "hpc_error_map_max": 1.871881,
                  "hpc_error_map_std": 0.268991,
                  "pi_d_per_class": {"airplane": 0.0500, "car": 0.3425,
                                     "truck": 0.3237}}
# Car loses #1 (airplane overtakes) while everything stays in-envelope — the
# second BLOCKING branch — must FAIL.
ROW15_CAR_NOT_1 = {"epoch": 15, "eps": 0.031, "hpc_error_mean": 0.160782,
                   "hpc_error_map_min": 2.1e-05, "hpc_error_map_max": 1.871881,
                   "hpc_error_map_std": 0.268991,
                   "pi_d_per_class": {"airplane": 0.3600, "car": 0.3425,
                                      "truck": 0.3237}}


def _extract(nb_path, fn_name):
    """Pull a REAL function out of a notebook's source via AST.

    The notebooks are executable scripts (they train on import), so they
    cannot be imported; the function is extracted by AST and run in a minimal
    namespace holding the constants and helpers it references.
    """
    tree = ast.parse(nb_path.read_text())
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == fn_name), None)
    assert fn is not None, f"{fn_name} not found in {nb_path}"
    mod = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {"HPC_TREND_MIN_DECREASE": 0.10, "HPC_EXPLOSION_RATIO": 10.0,
          "PI_D_REF_MIN": 0.2478, "PI_D_REF_MAX": 0.3860, "PI_D_REF_STD": 0.0378,
          "PI_D_REF_BAND_LO": 0.1722, "PI_D_REF_BAND_HI": 0.4616}
    exec(compile(mod, str(nb_path), "exec"), ns)
    return ns[fn_name]


def _gate_for(nb_path):
    """Assemble health_verdict_stage2 with its real truck_watch_from_pi_d
    helper in the same namespace (the gate calls it)."""
    tree = ast.parse(nb_path.read_text())
    names = ["truck_watch_from_pi_d", "health_verdict_stage2"]
    ns = {"HPC_TREND_MIN_DECREASE": 0.10, "HPC_EXPLOSION_RATIO": 10.0,
          "PI_D_REF_MIN": 0.2478, "PI_D_REF_MAX": 0.3860, "PI_D_REF_STD": 0.0378,
          "PI_D_REF_BAND_LO": 0.1722, "PI_D_REF_BAND_HI": 0.4616}
    for name in names:
        fn = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == name), None)
        assert fn is not None, f"{name} not found in {nb_path}"
        mod = ast.Module(body=[fn], type_ignores=[])
        ast.fix_missing_locations(mod)
        exec(compile(mod, str(nb_path), "exec"), ns)
    return ns["health_verdict_stage2"]


GATES = {p.name: _gate_for(p) for p in NOTEBOOKS}
LABELS = {p.name: _extract(p, "gate_check_label") for p in NOTEBOOKS}


def _printed_checks(gate, label, rows):
    """Replicate the notebooks' check assembly + print format exactly."""
    v = gate(rows)
    checks = {
        "2_hpc_error_trend": label(v["hpc_error_trend_pass"]),
        "4_pi_d_reference_envelope": label(v["pi_d_reference_envelope_pass"]),
    }
    return v, checks, "\n".join(f"    [{k}] {v_}" for k, v_ in checks.items())


@pytest.mark.parametrize("name", list(GATES), ids=list(GATES))
def test_v4_smoke_re_scores_healthy_under_amended_gate(name):
    """THE regression (gate AMENDMENT 2026-08-16): the real v4 smoke — the
    run that FAILED the old car/truck-top-2 gate — must now read HEALTHY,
    with truck's rank-3 recorded as a NON-BLOCKING WATCH, not a failure."""
    v, checks, printed = _printed_checks(GATES[name], LABELS[name],
                                         [ROW1, ROW15_V4])
    joined = " ".join(v["reasons"])
    assert "trend OK" in joined, joined
    assert "within reference envelope" in joined, joined
    assert v["hpc_error_trend_pass"] is True, joined
    assert v["pi_d_reference_envelope_pass"] is True, joined
    assert v["healthy"] is True, joined
    assert checks["2_hpc_error_trend"] == "PASS", printed
    assert checks["4_pi_d_reference_envelope"] == "PASS", printed
    assert "[4_pi_d_reference_envelope] PASS" in printed, printed
    # Truck WATCH is emitted, non-blocking, with the actual v4 numbers.
    tw = v.get("truck_watch")
    assert tw is not None, joined
    assert tw["truck_rank"] == 3, tw
    assert tw["truck_in_top3"] is True, tw
    assert tw["truck_vs_2_margin"] == pytest.approx(0.3237 - 0.3360, abs=1e-4), tw
    assert "truck WATCH (non-blocking)" in joined, joined


@pytest.mark.parametrize("name", list(GATES), ids=list(GATES))
def test_envelope_collapse_still_blocks(name):
    """A class leaving the Stage-1-validated reference envelope (genuine
    collapse) must still FAIL check 4 and block Step B."""
    v, checks, printed = _printed_checks(GATES[name], LABELS[name],
                                         [ROW1, ROW15_COLLAPSE])
    joined = " ".join(v["reasons"])
    assert "reference-envelope VIOLATED" in joined, joined
    assert v["pi_d_reference_envelope_pass"] is False, joined
    assert v["healthy"] is False, joined
    assert checks["4_pi_d_reference_envelope"] == "FAIL", printed
    assert "[4_pi_d_reference_envelope] FAIL" in printed, printed


@pytest.mark.parametrize("name", list(GATES), ids=list(GATES))
def test_car_losing_1_blocks(name):
    """Car losing the #1 Π_D slot (second BLOCKING branch) fails check 4."""
    v, checks, printed = _printed_checks(GATES[name], LABELS[name],
                                         [ROW1, ROW15_CAR_NOT_1])
    joined = " ".join(v["reasons"])
    assert "#1 class changed" in joined, joined
    assert v["pi_d_reference_envelope_pass"] is False, joined
    assert v["healthy"] is False, joined
    assert checks["4_pi_d_reference_envelope"] == "FAIL", printed


@pytest.mark.parametrize("name", list(GATES), ids=list(GATES))
def test_trend_fail_pi_d_pass_reported_independently(name):
    """Mirror direction: flat trend + in-envelope Π_D prints [2] FAIL,
    [4] PASS — one criterion's failure never flips the other's label."""
    v, checks, printed = _printed_checks(GATES[name], LABELS[name],
                                         [ROW1, ROW15_FLAT_HEALTHY_PI_D])
    joined = " ".join(v["reasons"])
    assert "did NOT decrease" in joined, joined
    assert "within reference envelope" in joined, joined
    assert v["hpc_error_trend_pass"] is False, joined
    assert v["pi_d_reference_envelope_pass"] is True, joined
    assert v["healthy"] is False, joined
    assert checks["2_hpc_error_trend"] == "FAIL", printed
    assert checks["4_pi_d_reference_envelope"] == "PASS", printed


@pytest.mark.parametrize("name", list(GATES), ids=list(GATES))
def test_both_pass_reports_both_pass(name):
    """Both criteria healthy: both flags True, both labels PASS, gate open."""
    v, checks, printed = _printed_checks(GATES[name], LABELS[name],
                                         [ROW1, ROW15_FLAT_HEALTHY_PI_D]
                                         if False else [ROW1, ROW15_V4])
    joined = " ".join(v["reasons"])
    assert v["hpc_error_trend_pass"] is True, joined
    assert v["pi_d_reference_envelope_pass"] is True, joined
    assert v["healthy"] is True, joined
    assert checks["2_hpc_error_trend"] == "PASS", printed
    assert checks["4_pi_d_reference_envelope"] == "PASS", printed
