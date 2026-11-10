"""End-to-end run of the whole chain at the REAL geometry, with random weights: the real CompactViTGen2 architecture on the
ff path, the window sensor, the Bayes state, the Koopman policy and the stopping rule. Random weights mean the numbers say
nothing about perception; what is checked is that the pieces fit together, that the tensor shapes quoted in the dossier
(04 section 9) are the shapes that actually occur, and that a scene's result does not depend on its batch neighbours."""
import pytest
import torch

from gen2_foundation.backbone import CompactViTGen2
from rhan_im.agent import run_search
from rhan_im.layout import UnitLayout
from rhan_im.trunk_adapter import WindowSensor, ff_token_fn
from rhan_im.window_head import EvidenceCalibration, EvidenceHead

C = 100


class CanvasScene:
    """The interface run_search expects (B, C, label, measure), backed by the real trunk through the sensor."""

    def __init__(self, sensor, canvas, label):
        self.sensor, self.canvas, self.label = sensor, canvas, label
        self.B, self.C = canvas.shape[0], sensor.head.proj.out_features
        self.looks = []

    def measure(self, k):
        z, d = self.sensor.look(self.canvas, k)
        self.looks.append((k.clone(), z, d))
        return z


@pytest.fixture(scope="module")
def chain():
    torch.manual_seed(0)
    trunk = CompactViTGen2(img_size=112, num_refine_iters=2).eval()
    lay = UnitLayout(canvas=224, patch=14, window=112, target_units=4)
    cand = lay.candidates(2)
    head = EvidenceHead(384, C).eval()
    cal = EvidenceCalibration(C, lay.tok)
    cal.d_tok.fill_(2.0)                                                      # stand-ins for the measured values (gate 1)
    cal.d_gist.fill_(1.2)
    return trunk, lay, WindowSensor(ff_token_fn(trunk), head, lay, cal, cand)


def test_the_shapes_in_the_dossier_are_the_shapes_that_occur(chain):
    trunk, lay, sensor = chain
    B = 2
    canvas = torch.randn(B, 3, 224, 224)
    k = torch.tensor([0, sensor.K])                                           # one native window, one gist look
    with torch.no_grad():
        crop_in = torch.randn(B, 3, 112, 112)
        assert trunk._prep(crop_in).shape == (B, 66, 384)                     # cls + register + 64 patch tokens
        cls, tokens = ff_token_fn(trunk)(crop_in)
        assert cls.shape == (B, 384) and tokens.shape == (B, 64, 384)
        z, d = sensor.look(canvas, k)
    assert (lay.tok, lay.N, lay.V, sensor.n_candidates) == (8, 256, 320, 26)
    assert z.shape == (B, 320, C) and d.shape == (B, 320) and sensor.vis().shape == (26, 320)
    assert lay.footprint().shape == (256, 320)
    assert torch.isfinite(z).all() and torch.isfinite(d).all()


def test_a_look_only_writes_to_the_units_it_exposes(chain):
    _, lay, sensor = chain
    canvas = torch.randn(2, 3, 224, 224)
    k = torch.tensor([7, sensor.K])
    with torch.no_grad():
        z, d = sensor.look(canvas, k)
    tl = sensor.topleft[7]
    window = torch.zeros(lay.g, lay.g, dtype=torch.bool)
    window[tl[0]: tl[0] + lay.tok, tl[1]: tl[1] + lay.tok] = True
    fine = z[0, : lay.n_fine].abs().sum(1) > 0
    assert torch.equal(fine.reshape(lay.g, lay.g), window)                    # native look: exactly its 8 x 8 fine units
    assert z[0, lay.n_fine:].abs().sum() == 0                                 # and no coarse unit
    assert z[1, : lay.n_fine].abs().sum() == 0 and (z[1, lay.n_fine:].abs().sum(1) > 0).all()   # gist look: every coarse unit, no fine unit
    assert (d[0, : lay.n_fine] > 0).sum() == lay.tok ** 2 and (d[1, lay.n_fine:] > 0).all() and d[1, : lay.n_fine].abs().sum() == 0


def test_a_scene_does_not_depend_on_its_batch_neighbours(chain):
    _, _, sensor = chain
    torch.manual_seed(1)
    canvas = torch.randn(3, 3, 224, 224)
    k = torch.tensor([3, sensor.K, 12])
    with torch.no_grad():
        z, d = sensor.look(canvas, k)
        for i in range(3):
            zi, di = sensor.look(canvas[i: i + 1], k[i: i + 1])
            assert torch.allclose(z[i], zi[0], atol=1e-4, rtol=1e-4) and torch.equal(d[i], di[0])


def test_the_full_loop_runs_keeps_valid_probabilities_and_is_deterministic(chain):
    _, lay, sensor = chain
    torch.manual_seed(2)
    canvas = torch.randn(2, 3, 224, 224)
    label = torch.tensor([5, 9])
    seen = []

    def hook(t, post, k, z):
        p = post.p_cells().flatten(1).sum(1) + post.p_null()
        seen.append((t, post.log_cells.shape, post.u_contrib.shape, float((p - 1).abs().max())))

    kw = dict(vis=sensor.vis(), policy="koopman", K_max=3, p_null=0.3, footprint=lay.footprint(), unit_cap=8.0, kappa=4.0)
    a = run_search(CanvasScene(sensor, canvas, label), hook=hook, **kw)
    b = run_search(CanvasScene(sensor, canvas, label), **kw)
    assert a["dec"].shape == (2, 4) and a["risk"].shape == (2, 4) and a["fix"].shape == (2, 3)
    assert ((a["risk"] >= 0) & (a["risk"] <= 1)).all() and torch.isfinite(a["ent"]).all()
    assert (a["n_fix"] == 3).all()
    assert all(s[1] == (2, 256, C) and s[2] == (2, 320, C) and s[3] < 1e-4 for s in seen) and [s[0] for s in seen] == [0, 1, 2, 3]
    assert torch.equal(a["fix"], b["fix"]) and torch.equal(a["dec"], b["dec"]) and torch.allclose(a["risk"], b["risk"])


def test_a_stopping_threshold_that_is_always_met_stops_after_the_first_look(chain):
    _, lay, sensor = chain
    torch.manual_seed(3)
    canvas = torch.randn(2, 3, 224, 224)
    scene = CanvasScene(sensor, canvas, torch.tensor([1, 2]))
    out = run_search(scene, sensor.vis(), "koopman", 4, 0.3, footprint=lay.footprint(), unit_cap=8.0, kappa=4.0, stop_lambda=1.0)
    assert (out["n_fix"] == 1).all()
    assert torch.equal(out["dec"][:, 1], out["dec"][:, -1])                   # the posterior is frozen after the stop
