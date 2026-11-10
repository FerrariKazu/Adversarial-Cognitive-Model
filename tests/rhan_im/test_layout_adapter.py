import torch

from rhan_im.layout import UnitLayout
from rhan_im.trunk_adapter import WindowSensor, native_crop
from rhan_im.window_head import EvidenceCalibration, EvidenceHead, fit_calibration


def test_layout_sizes_and_footprint():
    lay = UnitLayout(canvas=112, patch=14, window=56, target_units=2)          # g = 8, tok = 4, coarse_block = 2
    assert (lay.g, lay.tok, lay.coarse_block, lay.N, lay.V) == (8, 4, 2, 64, 64 + 16)
    F = lay.footprint()
    assert F.shape == (64, 80) and set(F.unique().tolist()) <= {0.0, 1.0}
    inside = 2 * 8 + 2                                                         # anchor (2, 2): fine units (2..3, 2..3), all in coarse block (1, 1)
    assert F[inside, :64].sum() == 4 and F[inside, 64:].sum() == 1
    straddle = 3 * 8 + 3                                                       # anchor (3, 3): fine units (3..4, 3..4) touch four coarse blocks
    assert F[straddle, :64].sum() == 4 and F[straddle, 64:].sum() == 4
    corner = 7 * 8 + 7
    assert F[corner, :64].sum() == 1 and F[corner, 64:].sum() == 1             # clipped at the border


def test_native_crop_is_a_pure_slice_with_zero_padding():
    x = torch.randn(2, 3, 56, 56)
    tl = torch.tensor([[14, 28], [0, 0]])
    out = native_crop(x, tl, 28)
    assert torch.equal(out[0], x[0, :, 14:42, 28:56]) and torch.equal(out[1], x[1, :, 0:28, 0:28])
    far = native_crop(x, torch.tensor([[-14, 42], [0, 0]]), 28)               # partly outside
    assert far[0, :, :14].abs().sum() == 0 and torch.equal(far[0, :, 14:, :14], x[0, :, 0:14, 42:56])


class PatchMeanTrunk:
    """Transparent stand-in for a ViT: token (i, j) = mean pixel of patch (i, j)."""

    def __init__(self, patch):
        self.patch = patch

    def __call__(self, x):
        t = F_avg(x.mean(1, keepdim=True), self.patch)                         # (B, 1, tok, tok)
        tokens = t.flatten(2).transpose(1, 2)                                  # (B, tok*tok, 1)
        return tokens.mean(1), tokens


def F_avg(x, p):
    return torch.nn.functional.avg_pool2d(x, p)


def make_sensor(lay, C=3):
    head = EvidenceHead(1, C)
    with torch.no_grad():
        head.proj.weight.zero_()
        head.proj.bias.zero_()
        head.proj.weight[0, 0] = 1.0                                           # class 0 score = mean pixel
    calib = EvidenceCalibration(C, lay.tok)
    calib.d_tok.fill_(2.0)
    calib.d_gist.fill_(0.5)
    return WindowSensor(PatchMeanTrunk(lay.patch), head, lay, calib, lay.candidates(2)), head


def test_planted_marker_lands_on_the_right_unit_and_vis_matches():
    lay = UnitLayout(canvas=112, patch=14, window=56, target_units=2)
    sensor, _ = make_sensor(lay)
    canvas = torch.zeros(1, 3, 112, 112)
    r, c = 4, 5                                                                # fine cell of the marker
    canvas[:, :, r * 14:(r + 1) * 14, c * 14:(c + 1) * 14] = 1.0
    ks = [k for k in range(sensor.K) if (lambda tl: tl[0] <= r < tl[0] + 4 and tl[1] <= c < tl[1] + 4)(sensor.topleft[k].tolist())]
    assert ks
    for k in ks:
        z, d = sensor.look(canvas, torch.tensor([k]))
        assert z.shape == (1, lay.V, 3) and d.shape == (1, lay.V)
        assert z[0, r * 8 + c, 0] == 1.0 and z[0, :, 0].sum() == 1.0 and z[..., 1:].abs().sum() == 0
        tl = sensor.topleft[k]
        in_win = torch.zeros(lay.V, dtype=torch.bool)
        for i in range(4):
            for j in range(4):
                in_win[(tl[0] + i) * 8 + tl[1] + j] = True
        assert torch.equal(d[0] > 0, in_win) and torch.all(d[0][in_win] == 2.0)


def test_gist_look_fills_only_coarse_units():
    lay = UnitLayout(canvas=112, patch=14, window=56, target_units=2)
    sensor, _ = make_sensor(lay)
    canvas = torch.zeros(1, 3, 112, 112)
    canvas[:, :, :28, :28] = 1.0                                               # top-left coarse token (covers 2x2 fine units)
    z, d = sensor.look(canvas, torch.tensor([sensor.K]))
    assert z[0, :lay.n_fine].abs().sum() == 0 and z[0, lay.n_fine, 0] == 1.0
    assert torch.all(d[0, :lay.n_fine] == 0) and torch.all(d[0, lay.n_fine:] == 0.5)


def test_fit_calibration_recovers_known_values():
    g = torch.Generator().manual_seed(0)
    null = 3.0 + 2.0 * torch.randn(100_000, generator=g)
    pos = 3.0 + 2.0 * torch.randn(100_000, generator=g) + 2.0 * 1.8
    r = fit_calibration(pos, null)
    assert abs(r["mu0"] - 3.0) < 0.05 and abs(r["sd0"] - 2.0) < 0.05 and abs(r["d_prime"] - 1.8) < 0.05 and abs(r["zroc_slope"] - 1.0) < 0.08


def _posterior_after_one_look(marker_cells, amplitude=2.0):
    from rhan_im.belief import SearchPosterior
    lay = UnitLayout(canvas=112, patch=14, window=56, target_units=2)
    sensor, _ = make_sensor(lay)
    canvas = torch.zeros(1, 3, 112, 112)
    for r, c in marker_cells:
        canvas[:, :, r * 14:(r + 1) * 14, c * 14:(c + 1) * 14] = amplitude
    post = SearchPosterior(lay.N, 3, 1, footprint=lay.footprint(), n_units=lay.V, mode="best", p_null=0.4)
    k = torch.tensor([[j for j in range(sensor.K) if sensor.topleft[j].tolist() == [2, 4]][0]])   # window rows 2..5, cols 4..7
    z, d = sensor.look(canvas, k.reshape(1))
    post.update(z, d)
    return lay, post


def test_sensor_feeds_the_posterior_end_to_end():
    lay, post = _posterior_after_one_look([(4, 5), (4, 6), (5, 5), (5, 6)])    # a marker exactly as large as a target
    assert post.p_cells().sum().item() + post.p_null().item() == __import__("pytest").approx(1.0, abs=1e-5)
    assert post.p_loc()[0].argmax().item() == 4 * 8 + 5                        # the anchor whose 2x2 footprint is the marker
    assert post.decision()[0].item() == 1                                      # class 0 (index 0 is "absent")


def test_evidence_of_the_wrong_size_is_evidence_against():
    # A one-unit blip inside the window is NOT what a 2x2 target looks like: every observed anchor that
    # needs the other three units to light up loses mass, so unobserved anchors end up more probable.
    lay, post = _posterior_after_one_look([(4, 5)])
    p = post.p_loc()[0]
    observed = torch.zeros(lay.N, dtype=torch.bool)
    for r in range(2, 6):
        for c in range(4, 8):
            observed[r * 8 + c] = True
    unobserved_prior = p[~observed].max().item()
    assert p[observed].max().item() < unobserved_prior


def test_vif_calibration_scales_scores_and_detectabilities_and_defaults_to_identity():
    lay = UnitLayout(canvas=112, patch=14, window=56, target_units=2)
    sensor, _ = make_sensor(lay)
    canvas = torch.zeros(1, 3, 112, 112)
    canvas[:, :, 4 * 14:5 * 14, 5 * 14:6 * 14] = 1.0
    k = torch.tensor([sensor.K - 1])
    z1, d1 = sensor.look(canvas, k)
    g = torch.tensor([sensor.K])
    zg1, dg1 = sensor.look(canvas, g)
    sensor.calib.vif_tok.fill_(4.0)
    sensor.calib.vif_gist.fill_(9.0)
    z4, d4 = sensor.look(canvas, k)
    zg9, dg9 = sensor.look(canvas, g)
    assert torch.allclose(z4, z1 / 2.0) and torch.allclose(d4, d1 / 2.0)                 # native look: 1 / sqrt(4)
    assert torch.allclose(zg9, zg1 / 3.0) and torch.allclose(dg9, dg1 / 3.0)             # gist look:  1 / sqrt(9)
    # per-unit evidence d z - d^2 / 2 is divided by the factor itself (the exact equicorrelated correction)
    llr1 = (d1 * z1[..., 0] - d1 ** 2 / 2).sum().detach()
    llr4 = (d4 * z4[..., 0] - d4 ** 2 / 2).sum().detach()
    assert float(llr4) == __import__("pytest").approx(float(llr1) / 4.0, abs=1e-5)
