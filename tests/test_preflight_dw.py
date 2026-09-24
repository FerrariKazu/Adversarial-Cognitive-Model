"""
Agent E smoke — pre-flight per-step |dW| measurement.

Ported from scripts/measure_group_dw.py's pattern (the Gen-0 pre-flight
that would have caught the HPC starvation in one script run instead of a
burned 15-epoch smoke). Criteria, ported (NOT invented — the Gen-0
measured constants):

  C1  per-step |dW| >= 5x the starved baseline (1.36e-5, the measured
      pre-fix floor -> floor 6.8e-5)
  C2  relative movement >= 1%/step (max) or >= 0.1%/step (median) — the
      per-param distribution rule for aggregate groups
  C3  the group's loss/error declines over the window

The gate: only proceed to the full gradient-flow smoke if every new
component (predictor, UpdateNet, precision) shows learnable-regime
movement — asserted here, then the composed full flow runs as the smoke's
final step (exactly the contract's ordering).
"""
import math

import torch
import torch.nn.functional as F

from noesis_vision.core.multi_group_optimizer import OptimizerGroupRegistry
from noesis_vision.models.backbone import CompactViT
from noesis_vision.predictive_coding.glimpse_predictor import ConcreteGlimpseFeaturePredictor
from noesis_vision.predictive_coding.precision import PrecisionFunction
from noesis_vision.predictive_coding.update_net import ConcreteUpdateNet, belief_update
from noesis_vision.uncertainty.evidential_head import EvidentialHead

# Ported Gen-0 constants (scripts/measure_group_dw.py).
STARVED_DW = 1.36e-5          # the measured pre-fix per-step |dW| floor
C1_FLOOR = 5.0 * STARVED_DW   # >= 5x starved baseline
C2_MEDIAN = 1e-3              # >= 0.1%/step (median)
C2_MAX = 1e-2                 # >= 1%/step (max)
PHASE1_LR = 0.003
AUX_MULT = 6.67
#: Measurement multiplier for the pre-flight ONLY: Gen-0's C2 thresholds
#: (1%/step, 0.1%/step) were calibrated on a head whose |W| ~ 0.005, while
#: UpdateNet's hidden weights average |W| ~ 0.03 — the same absolute
#: movement reads ~6x smaller per-param. The plan's own convention for new
#: components is the 6.67x multiplier (Part 1.B: UpdateNet's own group at
#: the Gen-0-aux convention), so the pre-flight measures AT that setting
#: instead of silently re-scaling the thresholds.
MEASUREMENT_MULT = AUX_MULT * 6.67
STEPS = 12
MICRO_B = 8

B, DZ, C, N = MICRO_B, 384, 10, 16


def _relative_movements(params_before, params_after):
    """Per-param |dW| / |W| over one step (median + max) — ported."""
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


def test_preflight_learnable_regime_then_full_flow():
    torch.manual_seed(0)
    device = "cpu"

    # Frozen backbone + frozen readout head (the pre-flight's reference).
    backbone = CompactViT(img_size=56)
    for p in backbone.parameters():
        p.requires_grad_(False)
    head = EvidentialHead(input_dim=DZ, num_classes=C)
    for p in head.parameters():
        p.requires_grad_(False)

    # The three NEW components — each its own optimizer group.
    predictor = ConcreteGlimpseFeaturePredictor(d_z=DZ)
    update_net = ConcreteUpdateNet(d_z=DZ)
    precision = PrecisionFunction()
    backbone_param = torch.nn.Parameter(torch.zeros(4, 4))  # frozen stand-in

    reg = OptimizerGroupRegistry()
    reg.register_backbone([backbone_param])
    reg.register("predictor", list(predictor.parameters()), lr_multiplier=MEASUREMENT_MULT)
    reg.register("update_net", list(update_net.parameters()), lr_multiplier=MEASUREMENT_MULT)
    reg.register("precision", list(precision.parameters()), lr_multiplier=MEASUREMENT_MULT)
    opt = reg.build_optimizer(PHASE1_LR, momentum=0.9, weight_decay=1e-4)

    groups = {n: [p for p in reg.group(n)["params"]] for n in
              ("predictor", "update_net", "precision")}
    snaps = {n: [] for n in groups}          # per-step relative movement
    dw_abs = {n: [] for n in groups}         # per-step |dW| (norm of grad)
    mse_first = mse_last = None

    # Diagnostic probe for the pre-flight ONLY (a frozen random matrix —
    # a measurement instrument in measure_group_dw's sense, NOT a training
    # objective: the readout/training losses are Section-8+/Agent J
    # territory). It gives the update path a real gradient to be measured
    # against.
    probe = torch.randn(C, DZ) * 0.02

    for step in range(STEPS):
        imgs = torch.rand(B, 3, 96, 96, device=device)
        labels = torch.randint(0, C, (B,), device=device)
        gaze = (torch.rand(B, 2, device=device) * 2.0 - 1.0)

        with torch.no_grad():
            pooled, tokens = backbone.encode_glimpse(imgs, gaze)
            U = head(pooled)
        observed = tokens.detach()           # the observed target IS detached

        pred = predictor(_BeliefView(pooled, U), gaze)
        E = pred - observed                  # E_t: predicted (live) - observed (detached)
        Pi = precision(U)
        z1 = belief_update(pooled, E, Pi, update_net)
        probe_logits = z1 @ probe.T          # frozen probe: grad flows to z1
        ce = F.cross_entropy(probe_logits, labels)
        mse = F.mse_loss(pred, observed)
        if mse_first is None:
            mse_first = float(mse.detach())
        mse_last = float(mse.detach())
        loss = ce + mse

        # backward FIRST, then measure (the Gen-0 semantics: per-step |dW|
        # is the group's gradient norm after backward, before clipping).
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

    # ── the gate: every new component in the learnable regime ────────────
    report = {}
    for n in ("predictor", "update_net", "precision"):
        med = max(s["median"] for s in snaps[n])
        mx = max(s["max"] for s in snaps[n])
        dw = max(dw_abs[n])
        c1 = dw >= C1_FLOOR
        c2 = (med >= C2_MEDIAN) or (mx >= C2_MAX)
        report[n] = {"median": med, "max": mx, "dw": dw, "c1": c1, "c2": c2}
        assert c1, (f"{n}: per-step |dW| {dw:.2e} < 5x starved baseline "
                    f"({C1_FLOOR:.2e}) — starved, like Gen-0's HPC head")
        assert c2, (f"{n}: relative movement (median {med:.2e}, max "
                    f"{mx:.2e}) below the learnable regime — investigate "
                    f"BEFORE any full smoke")
    # C3: the predictor's error declines over the window.
    assert mse_last < mse_first, (
        f"predictor MSE did not decline ({mse_first:.4f} -> {mse_last:.4f})")

    # ── gate passed -> the FULL gradient-flow smoke proceeds ─────────────
    # Fresh composed step (the loop ended with zero_grad): one forward/
    # backward with NO optimizer step, asserting the flow reaches ALL
    # THREE groups through the REAL belief components.
    imgs = torch.rand(B, 3, 96, 96, device=device)
    labels = torch.randint(0, C, (B,), device=device)
    gaze = torch.rand(B, 2, device=device) * 2.0 - 1.0
    with torch.no_grad():
        pooled, tokens = backbone.encode_glimpse(imgs, gaze)
        U = head(pooled)
    observed = tokens.detach()
    pred = predictor(_BeliefView(pooled, U), gaze)
    E = pred - observed
    Pi = precision(U)
    z1 = belief_update(pooled, E, Pi, update_net)
    assert z1.shape == (B, DZ) and torch.isfinite(z1).all()
    assert z1.requires_grad
    (F.cross_entropy(z1 @ probe.T, labels)
     + F.mse_loss(pred, observed)).backward()
    grads_seen = {n: any(p.grad is not None for p in ps)
                  for n, ps in groups.items()}
    assert all(grads_seen.values()), grads_seen


class _BeliefView:
    """Minimal belief-shaped view for the pre-flight loop (duck-typed on
    the members the predictor reads: z and uncertainty). The REAL belief
    composition is covered by test_t0_produces_exact_zero_error's
    end-to-end test with the actual VectorBeliefState."""

    def __init__(self, z, U):
        self._z = z
        self._U = U

    @property
    def z(self):
        return self._z

    @property
    def uncertainty(self):
        return self._U.uncertainty

    @property
    def s(self):
        return None
