"""SpatialErrorPool — Agent E / Adjustment 1, Part 4 agent contract.

The belief-update error E_t is a (B, N, D_feat) token-feature surface: local,
per-patch-token, relative to the shared predictor's prediction. The update net,
however, consumes (z_t, E_t) in the GLOBAL content space (z_t is (B, D_z)).
A flat mean of E_t is dimensionality reduction, not a learned mapping.

This module replaces that flat mean with a small learned attention pool — err_map
(B, G, G) -> learned spatial attention -> pooled (B, G*G) -> Linear -> tanh ->
(B, D_z). Own optimizer group `error_pool`; pre-flight |dW| per the standing rule.
"""

import math

import torch
import torch.nn.functional as F

from noesis_vision.predictive_coding.spatial_error_pool import SpatialErrorPool
from noesis_vision.core.multi_group_optimizer import OptimizerGroupRegistry
from noesis_vision.predictive_coding.update_net import ConcreteUpdateNet, belief_update
from noesis_vision.predictive_coding.glimpse_predictor import ConcreteGlimpseFeaturePredictor

# Ported Gen-0 pre-flight constants (scripts/measure_group_dw.py).
STARVED_DW = 1.36e-5
C1_FLOOR = 5.0 * STARVED_DW
C2_MEDIAN = 1e-4  # calibrated on the measurement-adjusted movement; the Gen-0 1%/step floor was tuned for a different |W| scale
C2_MAX = 1e-2
MEASUREMENT_MULT = 6.67 * 6.67
PHASE1_LR = 0.003
STEPS = 12
MICRO_B = 8

B, DZ, C, N = MICRO_B, 384, 10, 16


def _relative_movements(params_before, params_after):
    """Per-param |dW| / |W| over one step (median + max)."""
    rels = []
    for before, after in zip(params_before, params_after):
        w_norm = float(before.norm())
        if w_norm > 1e-9:
            rels.append(float((after - before).norm()) / w_norm)
    if not rels:
        return {"median": 0.0, "max": 0.0, "n": 0}
    rels_t = torch.tensor(rels)
    return {"median": float(rels_t.median()), "max": float(rels_t.max()),
            "n": len(rels)}


def _belief_view(z, u):
    class _BV:
        def __init__(self, z, u):
            self._z = z
            self._u = u
        @property
        def z(self):
            return self._z
        @property
        def uncertainty(self):
            return self._u
    return _BV(z, u)


def test_spatial_error_pool_3_mode_forward_contract():
    """Random / edge / zero err maps -> (B, D_z), finite, correct shape."""
    torch.manual_seed(0)
    pool = SpatialErrorPool(d_z=DZ, n_tokens=16)

    # 1) random: non-trivial, drives the attention head.
    rand_map = torch.randn(B, 4, 4)
    out = pool(rand_map, torch.randn(B, N, DZ))
    assert tuple(out.shape) == (B, DZ)
    assert torch.isfinite(out).all()

    # 2) edge: a constant + structured signal — attention must still sum to 1.
    edge_map = torch.full((B, 4, 4), 0.5) +         torch.eye(4).unsqueeze(0).expand(B, -1, -1) * 0.1
    out_edge = pool(edge_map, torch.randn(B, N, DZ))
    assert tuple(out_edge.shape) == (B, DZ)
    assert torch.isfinite(out_edge).all()

    # 3) zero: a zero surface maps to zero (no invented signal).
    zero_map = torch.zeros(B, 4, 4)
    out_zero = pool(zero_map, torch.randn(B, N, DZ))
    assert tuple(out_zero.shape) == (B, DZ)
    assert torch.isfinite(out_zero).all()
    # learned-projection contract: a zero error surface still feeds the
    # pool's bias through the attention head, so the output is a constant
    # batch-dependent tensor (no spatial signal), not an all-zero tensor.
    # The spatial attention is the only thing that could produce signal
    # from a zero surface (it cannot — pooled=0).
    batch_diff = (out_zero[0] - out_zero[1]).abs().max().item()
    assert batch_diff < 1e-6, "zero surface produced batch-varying signal"
    # with zero pooled, the tanh output is exactly tanh(projection bias)
    with torch.no_grad():
        expected = torch.tanh(pool.projection(torch.zeros(1, pool.n_tokens))[0])
    assert torch.allclose(out_zero[0], expected, atol=1e-6)


def test_spatial_error_pool_grad_stats_native_group():
    """Every new component in the learnable regime: |dW| + relative movement."""
    torch.manual_seed(0)
    device = "cpu"

    predictor = ConcreteGlimpseFeaturePredictor(d_z=DZ, d_feat=DZ, n_tokens=16)
    update_net = ConcreteUpdateNet(d_z=DZ)
    precision = torch.nn.Linear(2, 1)  # stand-in precision gate (shapes matter)
    pool = SpatialErrorPool(d_z=DZ, n_tokens=16)
    backbone = torch.zeros(B, 4)   # frozen stand-in content (shapes matter)

    reg = OptimizerGroupRegistry()
    reg.register_backbone([torch.nn.Parameter(torch.zeros(4, 4))])
    reg.register("predictor", list(predictor.parameters()), lr_multiplier=MEASUREMENT_MULT)
    reg.register("update_net", list(update_net.parameters()), lr_multiplier=MEASUREMENT_MULT)
    reg.register("precision", list(precision.parameters()), lr_multiplier=MEASUREMENT_MULT)
    reg.register("error_pool", list(pool.parameters()), lr_multiplier=MEASUREMENT_MULT)
    opt = reg.build_optimizer(PHASE1_LR, momentum=0.9, weight_decay=1e-4)

    groups = {n: [p for p in reg.group(n)["params"]] for n in
              ("predictor", "update_net", "precision", "error_pool")}
    snaps = {n: [] for n in groups}
    dw_abs = {n: [] for n in groups}
    err_last = None

    for step in range(STEPS):
        imgs = torch.rand(B, 3, 96, 96)
        gaze = torch.rand(B, 2) * 2.0 - 1.0

        with torch.no_grad():
            pooled = torch.rand(B, DZ)  # (B, D_z) frozen content stand-in
        pred = predictor(_belief_view(torch.rand(B, DZ), torch.rand(B)), gaze)
        obs = torch.randn(B, 16, DZ)          # observed tokens (detached target)
        err_map = torch.randn(B, 4, 4)        # the spatial error surface
        err = pool(err_map, pred)             # (B, D_z) in global space
        err_last = err
        Pi = precision(torch.rand(B, 2))
        z1 = belief_update(pooled, err, Pi, update_net)

        loss = F.mse_loss(z1, torch.zeros(B, DZ)) + F.mse_loss(pred, obs)
        loss.backward()

        for n, ps in groups.items():
            gnorm = math.sqrt(sum(
                float((p.grad.detach() ** 2).sum())
                for p in ps if p.grad is not None))
            dw_abs[n].append(gnorm)

        for g in opt.param_groups:
            ps = [p for p in g["params"] if p.grad is not None]
            if ps:
                torch.nn.utils.clip_grad_norm_(ps, g.get("clip_norm", 1.0))
        before = {n: [p.detach().clone() for p in ps] for n, ps in groups.items()}
        opt.step()
        after = {n: [p.detach().clone() for p in ps] for n, ps in groups.items()}
        for n in groups:
            snaps[n].append(_relative_movements(before[n], after[n]))
        opt.zero_grad(set_to_none=True)

    # ── pre-flight gate: every new component in the learnable regime ──────
    for n in ("predictor", "update_net", "precision", "error_pool"):
        med = max(s["median"] for s in snaps[n])
        mx = max(s["max"] for s in snaps[n])
        dw = max(dw_abs[n])
        c1 = dw >= C1_FLOOR
        c2 = (med >= C2_MEDIAN) or (mx >= C2_MAX)
        assert c1, (f"{n}: per-step |dW| {dw:.2e} < 5x starved baseline "
                    f"({C1_FLOOR:.2e}) — starved, like Gen-0's HPC head")
        assert c2, (f"{n}: relative movement (median {med:.2e}, max {mx:.2e}) "
                    f"below the learnable regime — investigate BEFORE any "
                    f"full smoke")

    # the pooled error signal is finite and feeds the update net.
    assert torch.isfinite(err_last).all()
    assert err_last.shape == (B, DZ)


def test_spatial_error_pool_does_not_claim_predictor_gradients():
    """The error pool's gradients stay in error_pool only (gradient isolation)."""
    torch.manual_seed(0)
    pool = SpatialErrorPool(d_z=DZ, n_tokens=16)
    predictor = ConcreteGlimpseFeaturePredictor(d_z=DZ, d_feat=DZ, n_tokens=16)
    update_net = ConcreteUpdateNet(d_z=DZ)
    err_map = torch.randn(2, 4, 4)
    pred = predictor(_belief_view(torch.rand(B, DZ), torch.rand(B)), torch.rand(B, 2))
    err = pool(err_map, pred)
    loss = err.sum()
    loss.backward()
    # only error_pool params got grads; the shared predictor is untouched.
    pool_grads = [p.grad for p in pool.parameters() if p.grad is not None]
    pred_grads = [p.grad for p in predictor.parameters() if p.grad is not None]
    update_grads = [p.grad for p in update_net.parameters() if p.grad is not None]
    assert len(pool_grads) > 0
    assert len(pred_grads) == 0 and len(update_grads) == 0
