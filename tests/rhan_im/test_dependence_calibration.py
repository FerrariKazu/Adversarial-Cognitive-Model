"""The simulated claims behind the within-look dependence handling: Lemma 1b (equicorrelated noise) and Lemma 1c (shared difficulty)."""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("dep_check", ROOT / "scripts" / "rhan_im" / "dependence_calibration_check.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)

RHO = {"independent": 0.0, "equicorrelated": 0.35, "shared_difficulty": 0.0, "combined_mild": 0.10, "combined_strong": 0.35}


@pytest.fixture(scope="module")
def worlds():
    return {w: D.simulate(w, 60000, 16, 0.6, RHO[w], 0.4, seed=3) for w in D.KINDS}


def test_in_the_independent_world_nothing_is_corrected_and_all_posteriors_agree(worlds):
    w = worlds["independent"]
    assert w["vif_used"] == 1.0 and w["cv_estimated"] == 0.0
    assert w["naive"]["ece"] < 0.01 and w["exact"]["ece"] < 0.01
    assert w["naive"]["nll"] == pytest.approx(w["exact"]["nll"], abs=1e-9)
    assert w["full_rule"]["nll"] == pytest.approx(w["naive"]["nll"], abs=1e-9)


def test_ignoring_correlated_noise_is_badly_overconfident_and_the_vif_correction_fixes_it(worlds):
    w = worlds["equicorrelated"]
    assert w["vif_null"] == pytest.approx(1 + 15 * 0.35, rel=0.06) and w["vif_used"] == pytest.approx(w["vif_null"])
    assert w["naive"]["ece"] > 0.15 and w["naive"]["mean_confidence"] - w["naive"]["accuracy"] > 0.15
    assert w["vif_corrected"]["ece"] < 0.015 and w["exact"]["ece"] < 0.015
    assert w["vif_corrected"]["nll"] < w["naive"]["nll"]
    assert w["cv_estimated"] == 0.0                                                    # no difficulty is invented where there is none


def test_shared_difficulty_is_flagged_not_shrunk_and_the_fitted_mixture_calibrates(worlds):
    w = worlds["shared_difficulty"]
    assert w["vif_null"] == pytest.approx(1.0, abs=0.05) and w["vif_pos"] > 1.5 and w["vif_used"] == 1.0   # shrinkage is not applied
    assert w["cv_estimated"] == pytest.approx(0.4, abs=0.06)                                                # the true spread is recovered from the VIFs alone
    assert w["vif_corrected"]["ece"] == pytest.approx(w["naive"]["ece"], abs=1e-9)
    assert w["full_rule"]["ece"] < 0.015 < w["naive"]["ece"]
    assert w["full_rule"]["nll"] < w["naive"]["nll"]
    assert w["retired_max_rule"]["ece"] > w["naive"]["ece"]                                            # shrinking by VIF_pos made it worse: why the first draft rule was replaced


def test_with_both_effects_the_null_vif_is_subtracted_before_the_spread_is_read_off(worlds):
    w = worlds["combined_mild"]
    assert w["vif_used"] == pytest.approx(1 + 15 * 0.10, rel=0.08) and w["vif_pos"] / w["vif_null"] > 1.25
    assert w["cv_estimated"] == pytest.approx(0.4, abs=0.08)
    assert w["full_rule"]["ece"] < w["vif_corrected"]["ece"] < w["naive"]["ece"]
    assert w["full_rule"]["ece"] < 0.012 and w["full_rule"]["nll"] <= w["vif_corrected"]["nll"] + 1e-6


def test_the_blind_spot_is_strong_noise_dependence_and_the_vif_correction_alone_still_removes_most_of_the_error(worlds):
    w = worlds["combined_strong"]
    assert w["vif_pos"] / w["vif_null"] < 1.25 and w["cv_estimated"] == 0.0           # the difficulty flag cannot fire here ...
    assert w["vif_corrected"]["ece"] < 0.03 and w["naive"]["ece"] > 0.15              # ... and what it misses is small compared with what the VIF removes
    assert w["exact"]["ece"] < 0.015


def test_lemma_1b_is_exact_for_an_isolated_block_and_only_conservative_when_the_noise_extends_beyond_the_footprint():
    """A reviewer's finding. Dividing z and d by sqrt(VIF) reproduces the exact Gaussian log-likelihood ratio when the n footprint tokens are
    the whole observation and have equal d. If the same equicorrelated noise also sits on tokens outside the footprint, those tokens carry
    information about the shared noise (a noise reference: negative weights in Sigma^-1 d) that the block-only form discards, so it
    under-states the information; the pipeline stays calibrated for the block's own statistic, but a gate that reads d_block from it can
    fail a trunk that is informative."""
    import torch
    n_win, m, d = 64, 16, 0.6
    for rho in (0.1, 0.35):
        vif = 1 + (m - 1) * rho
        closed = m * d * d / vif
        sub = (1 - rho) * torch.eye(m, dtype=torch.float64) + rho * torch.ones(m, m, dtype=torch.float64)
        dv_b = torch.full((m,), d, dtype=torch.float64)
        assert float(dv_b @ torch.linalg.solve(sub, dv_b)) == pytest.approx(closed, rel=1e-10)               # isolated block: exact
        full = (1 - rho) * torch.eye(n_win, dtype=torch.float64) + rho * torch.ones(n_win, n_win, dtype=torch.float64)
        dv = torch.zeros(n_win, dtype=torch.float64)
        dv[:m] = d
        w = torch.linalg.solve(full, dv)
        assert float(dv @ w) > closed + 0.5                                                                    # block inside a correlated window: the closed form is conservative
        assert float(w[m:].min()) < 0                                                                          # the other tokens enter as a noise reference
    # unequal d inside an equicorrelated block: sum/VIF is no longer the exact weighting (Sigma^-1 d is not proportional to d)
    rho = 0.35
    sub = (1 - rho) * torch.eye(m, dtype=torch.float64) + rho * torch.ones(m, m, dtype=torch.float64)
    dprof = torch.linspace(0.3, 0.9, m, dtype=torch.float64)
    exact_info = float(dprof @ torch.linalg.solve(sub, dprof))
    assert exact_info > float((dprof ** 2).sum()) / (1 + (m - 1) * rho) + 0.1
