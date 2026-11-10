import math

import pytest
import torch

from rhan_im import gate1 as g1
from rhan_im.layout import UnitLayout
from rhan_im.trunk_adapter import WindowSensor
from rhan_im.window_head import EvidenceCalibration, EvidenceHead

from test_layout_adapter import PatchMeanTrunk, make_sensor


def test_paste_and_overlap_geometry_agree_with_the_sensor():
    # The compositing geometry (where the target is, which tokens overlap it) must be the SAME geometry the
    # sensor scatters scores with. Plant a 2x2-unit square, look through the mean-pixel stand-in trunk and
    # require: score is non-zero exactly where the overlap says the token overlaps the target.
    lay = UnitLayout(canvas=112, patch=14, window=56, target_units=2)
    sensor, _ = make_sensor(lay)
    bg = torch.zeros(1, 3, 112, 112)
    patch = torch.ones(1, 3, 28, 28)
    top, left = torch.tensor([4 * 14]), torch.tensor([5 * 14])
    canvas = g1.paste_squares(bg, patch, top, left, torch.tensor([1.0]))
    ov = g1.unit_overlap(top, left, 28, lay.g, lay.patch)
    assert ov.shape == (1, 8, 8) and ov.sum().item() == pytest.approx(4.0) and ov[0, 4, 5] == 1 and ov[0, 3, 5] == 0
    for k in range(sensor.K):
        tl = sensor.topleft[k:k + 1]
        cov = g1.window_token_overlap(ov, tl, lay)[0] > 0
        z, _ = sensor.look(canvas, torch.tensor([k]))
        idx, valid = lay.window_units(tl)
        z_tok = z[0, :lay.n_fine, 0][idx[0]] * valid[0]
        assert torch.equal(z_tok > 0, cov)


def test_partial_overlap_and_gist_overlap():
    lay = UnitLayout(canvas=112, patch=14, window=56, target_units=2)
    ov = g1.unit_overlap(torch.tensor([7]), torch.tensor([21]), 28, lay.g, lay.patch)     # half a unit off in both axes
    assert ov.max().item() == pytest.approx(1.0) and ov.sum().item() == pytest.approx(4.0)
    assert ov[0, 0, 1].item() == pytest.approx(0.5 * 1.0 * 1.0 / 1.0 * 0.5 + 0.0, abs=1.0)   # sanity: a fraction
    full = g1.unit_overlap(torch.tensor([0]), torch.tensor([0]), 28, lay.g, lay.patch)
    gist = g1.gist_token_overlap(full, lay)
    assert gist.shape == (1, 16) and gist[0, 0].item() == pytest.approx(1.0) and gist[0, 1:].sum().item() == 0.0


def synth(n, T, C, d_map, leak, mu_c, sd_c, seed, p_cov=0.35, p_has=0.7):
    g = torch.Generator().manual_seed(seed)
    label = torch.where(torch.rand(n, generator=g) < p_has, torch.randint(0, C, (n,), generator=g), torch.full((n,), -1))
    cover = (torch.rand(n, T, generator=g) < p_cov) & (label >= 0)[:, None]
    z = torch.randn(n, T, C, generator=g)
    own = torch.nn.functional.one_hot(label.clamp_min(0), C).bool()[:, None, :] & cover[:, :, None]
    wrong = (~torch.nn.functional.one_hot(label.clamp_min(0), C).bool())[:, None, :] & cover[:, :, None]
    z = z + own * d_map[None, :, None] + wrong * leak
    return z * sd_c + mu_c, cover, label


def test_measure_tokens_recovers_known_values():
    T, C, tok = 16, 5, 4
    d_map = torch.linspace(0.8, 3.2, T)
    mu_c, sd_c = torch.arange(C).float() * 0.5, 1.0 + 0.2 * torch.arange(C).float()
    cal = synth(8000, T, C, d_map, 0.3, mu_c, sd_c, seed=1)
    tst = synth(8000, T, C, d_map, 0.3, mu_c, sd_c, seed=2)
    mu0, sd0 = g1.fit_standardizer(*cal)
    assert (mu0 - mu_c).abs().max() < 0.06 and (sd0 - sd_c).abs().max() < 0.06
    m = g1.measure_tokens(*tst, mu0, sd0, tok)
    assert (m["d_shift"].flatten() - d_map).abs().max() < 0.2
    assert (m["d_prime"].flatten() - d_map).abs().max() < 0.2
    assert (m["null_sd"] - 1).abs().max() < 0.05 and m["null_mean"].abs().max() < 0.05
    assert (m["leakage"] - 0.3).abs().max() < 0.12
    assert (m["zroc_slope"][torch.isfinite(m["zroc_slope"])] - 1).abs().max() < 0.2


def test_measure_tokens_flags_unequal_variance():
    T, C, tok = 4, 3, 2
    g = torch.Generator().manual_seed(0)
    n = 20000
    label = torch.randint(0, C, (n,), generator=g)
    cover = torch.rand(n, T, generator=g) < 0.4
    z = torch.randn(n, T, C, generator=g)
    own = torch.nn.functional.one_hot(label, C).bool()[:, None, :] & cover[:, :, None]
    z = torch.where(own, 2.0 + 2.0 * torch.randn(n, T, C, generator=g), z)        # positives have sd 2
    mu0, sd0 = g1.fit_standardizer(z, cover, label)
    m = g1.measure_tokens(z, cover, label, mu0, sd0, tok)
    assert (m["pos_sd"] - 2).abs().max() < 0.15 and (m["zroc_slope"] - 0.5).abs().max() < 0.1      # null_sd / pos_sd


def test_paired_noise_correlation_and_effective_looks():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(50000, generator=g)
    for rho in (0.0, 0.6, 0.95):
        b = rho * a + math.sqrt(1 - rho ** 2) * torch.randn(50000, generator=g)
        assert g1.paired_noise_correlation(a, b) == pytest.approx(rho, abs=0.02)
    assert g1.effective_looks(8, 0.0) == pytest.approx(8.0) and g1.effective_looks(8, 1.0) == pytest.approx(1.0)
    assert g1.effective_looks(4, 0.5) == pytest.approx(4 / 2.5)


def test_train_evidence_head_learns_planted_class_features_and_exports_log_odds():
    D, C, T, b, tok = 24, 4, 4, 96, 2

    def batches(step):
        gg = torch.Generator().manual_seed(100 + step)
        label = torch.where(torch.rand(b, generator=gg) < 0.7, torch.randint(0, C, (b,), generator=gg), torch.full((b,), -1))
        cover = (torch.rand(b, T, generator=gg) < 0.4) & (label >= 0)[:, None]
        x = torch.randn(b, T, D, generator=gg)
        oh = torch.nn.functional.one_hot(label.clamp_min(0), C).float()
        x[..., :C] += 2.5 * oh[:, None, :] * cover[:, :, None].float()
        return x, cover, label, torch.ones(b, T)

    head, log = g1.train_evidence_head(batches, D, C, steps=250, lr=5e-3)
    assert log[-1][1] < log[0][1] and head.proj.out_features == C
    cal = [batches(10_000 + i) for i in range(20)]
    tst = [batches(20_000 + i) for i in range(20)]

    def score(chunks):
        with torch.no_grad():
            return (torch.cat([head(c[0]) for c in chunks]), torch.cat([c[1] for c in chunks]), torch.cat([c[2] for c in chunks]))

    mu0, sd0 = g1.fit_standardizer(*score(cal))
    m = g1.measure_tokens(*score(tst), mu0, sd0, tok, min_pos=20)
    assert m["d_shift"].min() > 2.0 and (m["null_sd"] - 1).abs().max() < 0.15


def test_verdict_draft_rule():
    ok = dict(zroc_slope_median=1.0, null_sd_median=1.0, pos_sd_median=1.0, d_native_centre=3.0, d_gist=1.0, rho_null=0.05,
              vif_null_op=1.0, vif_pos_op=1.0, op_block_tokens=16.0)
    v = g1.verdict(ok)
    assert v["pass_all"] and v["combination_rule"] == "sum" and v["native_over_gist"] == pytest.approx(3.0)
    assert not g1.verdict({**ok, "zroc_slope_median": 0.5})["sdt_adequate"]
    nan = float("nan")
    assert g1.verdict({**ok, "zroc_slope_median": nan})["sdt_adequate"]                       # falls back to sd_null / sd_pos = 1
    assert not g1.verdict({**ok, "zroc_slope_median": nan, "pos_sd_median": 2.5})["sdt_adequate"]
    assert not g1.verdict({**ok, "d_native_centre": 1.0})["informative"]
    assert not g1.verdict({**ok, "d_gist": 2.9})["looking_closer_pays"]
    assert g1.verdict({**ok, "rho_null": 0.9})["combination_rule"] == "best"
    assert g1.verdict({**ok, "rho_null": 0.4})["combination_rule"].startswith("best (conservative")


# ------------------------------------------------------------------ within-look dependence (block_vif)
def equicorrelated(n, T, C, rho, seed):
    """Unit-variance scores whose tokens (same scene, same class) have correlation rho."""
    g = torch.Generator().manual_seed(seed)
    shared = torch.randn(n, 1, C, generator=g)
    return math.sqrt(1 - rho) * torch.randn(n, T, C, generator=g) + math.sqrt(rho) * shared


def aligned_blocks(n, tok, block, seed):
    """(n, tok*tok) bool cover: one block x block square at a random aligned position per scene."""
    g = torch.Generator().manual_seed(seed)
    r = torch.randint(0, tok - block + 1, (n,), generator=g)
    c = torch.randint(0, tok - block + 1, (n,), generator=g)
    cover = torch.zeros(n, tok, tok, dtype=torch.bool)
    for i in range(n):
        cover[i, r[i]:r[i] + block, c[i]:c[i] + block] = True
    return cover.flatten(1)


@pytest.mark.parametrize("rho", [0.0, 0.4, 0.8])
def test_block_vif_null_matches_one_plus_n_minus_one_rho(rho):
    tok, block, C, n = 4, 2, 3, 3000
    z = equicorrelated(n, tok * tok, C, rho, seed=1)
    out = g1.block_vif(z, torch.zeros(n, tok * tok, dtype=torch.bool), torch.full((n,), -1), tok, block)
    assert out["vif_null"] == pytest.approx(1 + (block * block - 1) * rho, rel=0.06, abs=0.06)
    assert math.isnan(out["vif_pos"]) and out["n_pos_scenes"] == 0


@pytest.mark.parametrize("rho", [0.0, 0.5])
def test_block_vif_positive_scenes_match_the_same_formula(rho):
    tok, block, C, n = 4, 2, 3, 4000
    g = torch.Generator().manual_seed(2)
    label = torch.randint(0, C, (n,), generator=g)
    cover = aligned_blocks(n, tok, block, seed=3)
    z = equicorrelated(n, tok * tok, C, rho, seed=4)
    d_map = torch.linspace(1.0, 3.0, tok * tok)
    own = torch.nn.functional.one_hot(label, C).bool()[:, None, :] & cover[:, :, None]
    z = z + own * d_map[None, :, None]                                       # per-position shift: removed by the per-position mean
    out = g1.block_vif(z, cover, label, tok, block)                          # equal weights
    assert out["vif_pos"] == pytest.approx(1 + (block * block - 1) * rho, rel=0.06, abs=0.06)
    assert out["n_pos_scenes"] == n and math.isnan(out["vif_null"])         # no scene without a target


def test_block_vif_weighted_null_matches_the_analytic_value():
    tok, block, C, n, rho = 4, 2, 4, 6000, 0.5
    z = equicorrelated(n, tok * tok, C, rho, seed=5)
    w = torch.linspace(0.5, 3.0, tok * tok)
    out = g1.block_vif(z, torch.zeros(n, tok * tok, dtype=torch.bool), torch.full((n,), -1), tok, block, weights=w)
    num = den = 0.0
    for r in range(tok - block + 1):
        for c in range(tok - block + 1):
            wb = torch.tensor([float(w[(r + i) * tok + (c + j)]) for i in range(block) for j in range(block)])
            num += float((wb ** 2).sum() + rho * (wb.sum() ** 2 - (wb ** 2).sum()))
            den += float((wb ** 2).sum())
    assert out["vif_null"] == pytest.approx(num / den, rel=0.06)


def test_block_vif_sees_shared_image_difficulty_only_in_positives():
    # A target that is easy at one token is easy at its neighbours (kappa_i ~ N(1, 0.5^2) scales d for the whole
    # scene). Noise is independent, so the null VIF stays 1 while the positive VIF is
    # (n + (sum d)^2 s2) / (n + sum d^2 s2) = (4 + 64 * 0.25) / (4 + 16 * 0.25) = 2.5 for d = 2 on a 2 x 2 block.
    tok, block, C, n, s2 = 4, 2, 3, 6000, 0.25
    g = torch.Generator().manual_seed(6)
    label = torch.where(torch.rand(n, generator=g) < 0.5, torch.randint(0, C, (n,), generator=g), torch.full((n,), -1))
    cover = aligned_blocks(n, tok, block, seed=7) & (label >= 0)[:, None]
    kappa = 1.0 + math.sqrt(s2) * torch.randn(n, generator=g)
    z = torch.randn(n, tok * tok, C, generator=g)
    own = torch.nn.functional.one_hot(label.clamp_min(0), C).bool()[:, None, :] & cover[:, :, None]
    z = z + own * (2.0 * kappa)[:, None, None]
    out = g1.block_vif(z, cover, label, tok, block)
    assert out["vif_null"] == pytest.approx(1.0, abs=0.08)
    assert out["vif_pos"] == pytest.approx(2.5, rel=0.1)


def test_block_vif_pos_counts_only_looks_that_cover_the_whole_target_and_pooling_partial_coverage_biases_it_low():
    # A VIF grows with the number of tokens summed. If half of the windows cut the target (cover 2 of its 4 tokens) and the
    # rest contain all of it, the pooled estimate sits between VIF(2) and VIF(4) and is NOT comparable with the 4-token null VIF.
    tok, block, C, n, rho = 4, 2, 3, 6000, 0.5
    g = torch.Generator().manual_seed(11)
    label = torch.randint(0, C, (n,), generator=g)
    cover = aligned_blocks(n, tok, block, seed=12)
    cut = torch.rand(n, generator=g) < 0.5
    cover2 = cover.clone()
    for i in torch.nonzero(cut).squeeze(1).tolist():                         # drop the last two covered tokens of a "cut" target
        idx = torch.nonzero(cover[i]).squeeze(1)
        cover2[i, idx[2:]] = False
    z = equicorrelated(n, tok * tok, C, rho, seed=13)
    full4 = 1 + (block * block - 1) * rho
    only_full = g1.block_vif(z, cover2, label, tok, block)                    # default: scenes covering all 4 tokens
    pooled = g1.block_vif(z, cover2, label, tok, block, full_only=False)
    assert only_full["n_pos_scenes"] == int((~cut).sum()) and only_full["vif_pos"] == pytest.approx(full4, rel=0.08)
    assert pooled["n_pos_scenes"] == n and pooled["vif_pos"] < 0.93 * full4                  # the pooled version is biased low
    assert pooled["vif_pos"] > 1 + rho * 0.7                                                  # and sits between VIF(2) and VIF(4)


def test_block_vif_pos_is_nan_below_the_minimum_number_of_full_scenes_and_for_single_token_blocks():
    tok, C, n = 4, 3, 2000
    g = torch.Generator().manual_seed(14)
    label = torch.randint(0, C, (n,), generator=g)
    z = torch.randn(n, tok * tok, C, generator=g)
    cover = aligned_blocks(n, tok, 2, seed=15)
    assert math.isfinite(g1.block_vif(z, cover, label, tok, 2)["vif_pos"])
    assert math.isnan(g1.block_vif(z[:40], cover[:40], label[:40], tok, 2, min_pos=2)["vif_pos"])      # 40 scenes < min_pos_scenes = 50
    assert math.isnan(g1.block_vif(z, cover, label, tok, 1)["vif_pos"])                                # one token cannot be dependent


def test_block_vif_too_few_samples_is_nan_not_a_number():
    z = torch.randn(20, 16, 2)
    out = g1.block_vif(z, torch.zeros(20, 16, dtype=torch.bool), torch.full((20,), -1), 4, 2)
    assert math.isnan(out["vif_null"]) and math.isnan(out["vif_pos"])
    assert math.isnan(g1.block_vif(torch.randn(500, 16, 2), torch.zeros(500, 16, dtype=torch.bool), torch.full((500,), -1), 4, 1)["vif_null"])


def test_vif_correction_reproduces_the_exact_equicorrelated_likelihood_ratio():
    # For z ~ N(d 1, Sigma), Sigma = (1 - rho) I + rho 1 1^T and equal d, the exact log-likelihood ratio against
    # N(0, Sigma) is  d 1^T Sigma^-1 z - d^2 1^T Sigma^-1 1 / 2  =  (1 / VIF) * sum_u (d z_u - d^2 / 2),  VIF = 1 + (n - 1) rho.
    # So scaling z and d by 1 / sqrt(VIF) inside the independent-token update is exact in that case.
    n, rho, d = 6, 0.35, 1.7
    sigma = (1 - rho) * torch.eye(n, dtype=torch.float64) + rho * torch.ones(n, n, dtype=torch.float64)
    z = torch.randn(n, dtype=torch.float64, generator=torch.Generator().manual_seed(0)) + 0.3
    dv = torch.full((n,), d, dtype=torch.float64)
    exact = dv @ torch.linalg.solve(sigma, z) - 0.5 * dv @ torch.linalg.solve(sigma, dv)
    vif = 1 + (n - 1) * rho
    zc, dc = z / math.sqrt(vif), dv / math.sqrt(vif)
    corrected = (dc * zc - dc ** 2 / 2).sum()
    naive = (dv * z - dv ** 2 / 2).sum()
    assert float(corrected) == pytest.approx(float(exact), abs=1e-10)
    assert abs(float(naive) - float(exact)) > 1.0                           # the independent-token sum is badly off at this rho


def test_verdict_separates_noise_dependence_from_shared_difficulty_and_uses_the_block_level_detectability():
    ok = dict(zroc_slope_median=1.0, null_sd_median=1.0, pos_sd_median=1.0, d_native_centre=3.0, d_gist=1.0, rho_null=0.05)
    v = g1.verdict({**ok, "vif_null_op": 1.1, "vif_pos_op": 1.2, "op_block_tokens": 16.0})
    assert v["token_evidence"] == "independent" and v["vif_used"] == 1.0 and not v["shared_difficulty"]
    assert v["d_block_effective"] == pytest.approx(12.0) and v["pass_all"]
    v = g1.verdict({**ok, "vif_null_op": 4.0, "vif_pos_op": 4.2, "op_block_tokens": 16.0})                   # correlated noise: corrected by VIF_null
    assert v["token_evidence"].startswith("correct") and v["vif_used"] == 4.0 and not v["shared_difficulty"]
    assert v["d_block_effective"] == pytest.approx(6.0) and v["pass_all"]
    v = g1.verdict({**ok, "vif_null_op": 1.0, "vif_pos_op": 2.5, "op_block_tokens": 16.0})                   # dependence only on target scenes: flagged, not shrunk
    assert v["token_evidence"] == "independent" and v["vif_used"] == 1.0 and v["shared_difficulty"] and v["d_block_effective"] == pytest.approx(12.0)
    v = g1.verdict({**ok, "d_native_centre": 1.6, "vif_null_op": 16.0, "vif_pos_op": 16.0, "op_block_tokens": 16.0})   # fully dependent tokens
    assert v["d_block_effective"] == pytest.approx(1.6) and not v["informative"] and not v["pass_all"]
    v = g1.verdict({**ok, "vif_null_op": float("nan"), "vif_pos_op": 3.0, "op_block_tokens": 16.0})           # null VIF unmeasurable: cannot separate the causes
    assert not v["vif_measured"] and v["vif_used"] == 1.0 and not v["shared_difficulty"] and not v["shared_difficulty_assessed"]


def test_verdict_fails_closed_when_the_vif_or_the_block_size_is_missing():
    """A reviewer's finding: an unmeasured VIF_null used to read as 'independent tokens' (d_block 7.2 instead of 2.08 for
    d_centre 1.8 and VIF 12) and a missing block size skipped the block test, so a missing number could only help a pass."""
    nan = float("nan")
    base = dict(zroc_slope_median=1.0, null_sd_median=1.0, pos_sd_median=1.0, d_native_centre=1.8, d_gist=1.0, rho_null=0.05,
                vif_null_op=12.0, vif_pos_op=12.0, op_block_tokens=16.0)
    v = g1.verdict(base)                                                                   # measured: corrected to 1.8 * sqrt(16 / 12) = 2.08
    assert v["d_block_effective"] == pytest.approx(2.0785, abs=1e-3) and v["informative"] and v["pass_all"] and v["unmeasured"] == []
    for bad in (nan, None):
        v = g1.verdict({**base, "vif_null_op": bad})
        assert not v["informative"] and not v["pass_all"] and v["unmeasured"] == ["vif_null_op"] and math.isnan(v["d_block_effective"])
        assert v["token_evidence"] == "unmeasured" and not v["vif_measured"]
    for bad in (nan, 0.0, -1.0):
        v = g1.verdict({**base, "op_block_tokens": bad})
        assert not v["informative"] and not v["pass_all"] and v["unmeasured"] == ["op_block_tokens"]
    no_key = {k: x for k, x in base.items() if k != "op_block_tokens"}
    assert not g1.verdict(no_key)["pass_all"] and g1.verdict(no_key)["unmeasured"] == ["op_block_tokens"]
    both = {k: x for k, x in base.items() if k not in ("op_block_tokens", "vif_null_op")}
    assert g1.verdict(both)["unmeasured"] == ["vif_null_op", "op_block_tokens"]
    v = g1.verdict({**base, "op_block_tokens": 25.0})                                      # the block size is read, not assumed to be 16
    assert v["d_block_effective"] == pytest.approx(1.8 * math.sqrt(25 / 12.0), abs=1e-9)
    assert not g1.verdict({**base, "vif_pos_op": nan})["shared_difficulty_assessed"] and g1.verdict(base)["shared_difficulty_assessed"]


def test_difficulty_cv_from_vif_inverts_the_analytic_vif_and_handles_the_edges():
    n, d = 16, 0.6
    for cv in (0.1, 0.4, 0.8):
        c = d * d * cv * cv
        vif = (1 + n * c) / (1 + c)
        assert g1.difficulty_cv_from_vif(vif, n, d) == pytest.approx(cv, rel=1e-9)
    assert g1.difficulty_cv_from_vif(0.97, n, d) == 0.0                         # VIF <= 1: no heterogeneity evidence
    for rho in (0.1, 0.35):                                                    # equicorrelated noise AND shared difficulty: the null VIF is subtracted
        vn = 1 + (n - 1) * rho
        c = d * d * 0.4 ** 2
        vp = (vn + n * c) / (1 + c)
        assert g1.difficulty_cv_from_vif(vp, n, d, vif_null=vn) == pytest.approx(0.4, rel=1e-9)
        assert g1.difficulty_cv_from_vif(vn, n, d, vif_null=vn) == 0.0
    assert g1.difficulty_cv_from_vif(2.0, n, d, vif_null=0.8) == g1.difficulty_cv_from_vif(2.0, n, d)   # null VIF below 1 is clamped
    assert math.isnan(g1.difficulty_cv_from_vif(float(n), n, d)) and math.isnan(g1.difficulty_cv_from_vif(float("nan"), n, d))


# ------------------------------------------------------------------ spillover (footprint check) and the best-case gist comparator
def _ring(cover, tok):
    """Reference (loop) implementation of 'uncovered but within one token of a covered token'."""
    n = cover.shape[0]
    out = torch.zeros_like(cover)
    for i in range(n):
        g = cover[i].reshape(tok, tok)
        for r in range(tok):
            for c in range(tok):
                if g[r, c]:
                    continue
                if g[max(0, r - 1):r + 2, max(0, c - 1):c + 2].any():
                    out[i, r * tok + c] = True
    return out


def test_spillover_recovers_planted_leakage_onto_adjacent_tokens_only():
    tok, C, n = 5, 3, 3000
    T = tok * tok
    g = torch.Generator().manual_seed(8)
    label = torch.where(torch.rand(n, generator=g) < 0.7, torch.randint(0, C, (n,), generator=g), torch.full((n,), -1))
    cover = aligned_blocks(n, tok, 2, seed=9) & (label >= 0)[:, None]
    cover[torch.rand(n, generator=g) < 0.2] = False                                   # target outside the window: must not dilute
    ring = _ring(cover, tok)
    own = torch.nn.functional.one_hot(label.clamp_min(0), C).bool()[:, None, :]
    z = torch.randn(n, T, C, generator=g)
    z = z + (own & cover[:, :, None]) * 2.0 + (own & ring[:, :, None]) * 0.8
    out = g1.spillover(z, cover, label, tok)
    assert out["adjacent"] == pytest.approx(0.8, abs=0.1) and abs(out["far"]) < 0.1
    assert out["n_adjacent"] > 1000 and out["n_far"] > 1000


def test_spillover_is_zero_for_a_clean_sensor_and_nan_without_enough_data():
    tok, C, n = 4, 3, 3000
    g = torch.Generator().manual_seed(10)
    label = torch.where(torch.rand(n, generator=g) < 0.7, torch.randint(0, C, (n,), generator=g), torch.full((n,), -1))
    cover = aligned_blocks(n, tok, 2, seed=11) & (label >= 0)[:, None]
    own = torch.nn.functional.one_hot(label.clamp_min(0), C).bool()[:, None, :]
    z = torch.randn(n, tok * tok, C, generator=g) + (own & cover[:, :, None]) * 2.5
    out = g1.spillover(z, cover, label, tok)
    assert abs(out["adjacent"]) < 0.1 and abs(out["far"]) < 0.1
    few = g1.spillover(z[:50], cover[:50], label[:50], tok)
    assert math.isnan(few["adjacent"]) and math.isnan(few["far"])


def test_verdict_compares_against_the_best_case_gist():
    # the pooled gist d includes partially covered tokens (weaker); the comparator for "looking closer pays" is the best case
    ok = dict(zroc_slope_median=1.0, null_sd_median=1.0, pos_sd_median=1.0, d_native_centre=3.0, d_gist=1.0, rho_null=0.05)
    assert g1.verdict(ok)["looking_closer_pays"]
    assert not g1.verdict({**ok, "d_gist_full": 2.5})["looking_closer_pays"]          # 3.0 / 2.5 = 1.2 < 1.5
    assert g1.verdict({**ok, "d_gist_full": float("nan")})["looking_closer_pays"]     # unmeasurable: fall back to the pooled value
    assert g1.verdict({**ok, "d_gist_full": 0.4})["native_over_gist"] == pytest.approx(3.0)   # the larger of the two is used


# ------------------------------------------------------------------------------------------------ fail closed for every quantity a criterion needs
def _base():
    return dict(zroc_slope_median=1.0, null_sd_median=1.0, pos_sd_median=1.0, d_native_centre=3.0, d_gist=1.0, rho_null=0.05,
                vif_null_op=1.0, vif_pos_op=1.0, op_block_tokens=16.0)


def test_every_quantity_a_criterion_needs_is_listed_as_unmeasured_and_never_passes():
    base = _base()
    assert g1.verdict(base)["unmeasured"] == [] and g1.verdict(base)["pass_all"]
    for key in ("d_native_centre", "null_sd_median", "vif_null_op", "op_block_tokens", "d_gist"):
        for bad in (None, float("nan"), float("inf"), float("-inf")):
            v = g1.verdict({**base, key: bad})
            assert key in v["unmeasured"] and not v["pass_all"], (key, bad)
    for key in ("vif_null_op", "op_block_tokens"):                                      # a variance-inflation factor or a token count is positive or it was not measured
        for bad in (0.0, -1.0):
            v = g1.verdict({**base, key: bad})
            assert key in v["unmeasured"] and not v["pass_all"] and not v["informative"], (key, bad)
    v = g1.verdict({**base, "d_gist": None, "d_gist_full": 1.0})                         # either gist measure is enough
    assert v["unmeasured"] == [] and v["pass_all"]
    v = g1.verdict({**base, "zroc_slope_median": float("nan")})                          # the sd-ratio fallback measures the slope
    assert v["unmeasured"] == [] and v["pass_all"]
    for psd in (float("nan"), float("inf"), float("-inf")):
        v = g1.verdict({**base, "zroc_slope_median": float("nan"), "pos_sd_median": psd})
        assert "zroc_slope_median" in v["unmeasured"] and not v["sdt_adequate"], psd


def _oracle(s, thr):
    """An independent statement of the documented rule (07 section 6), written from the text, not from verdict()."""
    nan = float("nan")
    fin = lambda x: x is not None and math.isfinite(x)                                    # noqa: E731
    s = {k: (nan if v is None else float(v)) for k, v in s.items()}
    slope, nsd, psd = s["zroc_slope_median"], s["null_sd_median"], s["pos_sd_median"]
    if not fin(slope):
        slope = nsd / max(psd, 1e-6) if (fin(nsd) and fin(psd)) else nan
    sdt = fin(slope) and fin(nsd) and thr["zroc_slope"][0] <= slope <= thr["zroc_slope"][1] and thr["null_sd"][0] <= nsd <= thr["null_sd"][1]
    dc, vn, n = s["d_native_centre"], s["vif_null_op"], s["op_block_tokens"]
    block_known = fin(vn) and vn > 0 and fin(n) and n > 0
    used = vn if (block_known and vn > thr["vif_independent_max"]) else 1.0
    informative = bool(fin(dc) and block_known and dc >= thr["d_centre_min"] and dc * math.sqrt(n / used) >= thr["d_block_min"])
    gists = [g for g in (s["d_gist"], s.get("d_gist_full", nan)) if fin(g)]
    pays = bool(gists) and fin(dc) and dc / max(max(gists), 1e-6) >= thr["native_over_gist_min"]
    missing = set()
    if not (fin(vn) and vn > 0):
        missing.add("vif_null_op")
    if not (fin(n) and n > 0):
        missing.add("op_block_tokens")
    if not fin(dc):
        missing.add("d_native_centre")
    if not gists:
        missing.add("d_gist")
    if not fin(nsd):
        missing.add("null_sd_median")
    if not fin(slope):
        missing.add("zroc_slope_median")
    return bool(sdt), informative, bool(pays), missing


def test_verdict_agrees_with_an_independent_statement_of_the_rule_on_random_and_degenerate_summaries():
    import random
    rnd = random.Random(7)
    pool = {"zroc_slope_median": [None, float("nan"), 0.79, 0.8, 1.0, 1.25, 1.26, 2.0, -1.0, float("inf")],
            "null_sd_median": [None, float("nan"), 0.84, 0.85, 1.0, 1.2, 1.21, 0.0, float("inf")],
            "pos_sd_median": [None, float("nan"), 0.0, 1e-9, 0.5, 1.0, 3.0, float("inf"), float("-inf")],
            "d_native_centre": [None, float("nan"), 0.0, 1.49, 1.5, 3.0, 7.0, -2.0, float("inf")],
            "d_gist": [None, float("nan"), -0.5, 0.0, 1.0, 2.0, 3.0, 100.0, float("inf")],
            "d_gist_full": [None, float("nan"), 0.5, 1.0, 2.0, float("-inf")],
            "rho_null": [None, float("nan"), 0.0, 0.2, 0.4, 0.6, 0.9],
            "vif_null_op": [None, float("nan"), -1.0, 0.0, 0.9, 1.0, 1.25, 1.3, 4.0, 40.0, float("inf")],
            "vif_pos_op": [None, float("nan"), 1.0, 5.0],
            "op_block_tokens": [None, float("nan"), -16.0, 0.0, 4.0, 16.0, 36.0]}
    good = {"zroc_slope_median": [1.0], "null_sd_median": [1.0], "pos_sd_median": [1.0], "d_native_centre": [3.0, 7.0], "d_gist": [1.0, 2.0], "rho_null": [0.05],
            "vif_null_op": [1.0, 4.0], "vif_pos_op": [1.0], "op_block_tokens": [16.0, 36.0]}          # typical values, so that the pass side of the rule is exercised too
    seen_pass = seen_unmeasured = 0
    for _ in range(6000):
        s = {k: rnd.choice(good[k] if (k in good and rnd.random() < 0.8) else v) for k, v in pool.items()}
        if rnd.random() < 0.5:
            del s["d_gist_full"]
        sdt, inf, pays, missing = _oracle(s, g1.DEFAULT_THRESHOLDS)
        v = g1.verdict(s)
        assert (v["sdt_adequate"], v["informative"], v["looking_closer_pays"]) == (sdt, inf, pays), s
        assert v["pass_all"] == (sdt and inf and pays) and set(v["unmeasured"]) == missing, s
        if v["unmeasured"]:
            assert not v["pass_all"], s
        seen_pass += v["pass_all"]
        seen_unmeasured += bool(v["unmeasured"])
    assert seen_pass > 100 and seen_unmeasured > 1000                                      # the sample exercises both sides of the rule
