"""
Agent E contract test — the predictor is CHEAP (numeric threshold, auditable).

FLOPs here = multiply-accumulate (MAC) counts; bias terms are negligible
and excluded (documented). The backbone reference is the Agent C substrate
at the DEFAULT operating point: image 96x96, fovea 56 (4x14 patch grid):

  patch embed conv: 3*384*14*14 per image            ~= 0.226M
  per trunk block:  N_tot * 384 * (qkv 1152 + proj 384
                                   + fc1 1536 + fc2 1536)
                    = N_tot * 384 * 4608,  N_tot = 2 + (56/14)^2 = 18
    x12 blocks                                        ~= 382.21M
  ------------------------------------------------------------
  backbone ~= 382.43M MACs

The predictor's MACs are counted from its ACTUAL nn.Linear modules
(each fires once per candidate pass — no per-token Linears), so the test
survives architecture tweaks and the threshold stays meaningful.
"""
import torch

from noesis_vision.models.backbone import PATCH_SIZE, TRUNK_DEPTH, TRUNK_WIDTH, CompactViT
from noesis_vision.predictive_coding.glimpse_predictor import ConcreteGlimpseFeaturePredictor
from noesis_vision.predictive_coding.interfaces import GlimpseFeaturePredictor

THRESHOLD = 0.10  # predictor MACs / one backbone forward MACs < 10%


def _backbone_macs(img=96, fovea=56):
    n_tot = 2 + (fovea // PATCH_SIZE) ** 2
    patch = 3 * TRUNK_WIDTH * PATCH_SIZE * PATCH_SIZE
    per_block = n_tot * TRUNK_WIDTH * (3 * TRUNK_WIDTH + TRUNK_WIDTH
                                       + 4 * TRUNK_WIDTH + 4 * TRUNK_WIDTH)
    return patch + TRUNK_DEPTH * per_block


def _module_linear_macs(module):
    total = 0
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            total += m.in_features * m.out_features
    return total


def test_predictor_under_10_percent_of_backbone():
    bb = _backbone_macs()
    p = ConcreteGlimpseFeaturePredictor()  # defaults match the substrate
    pred_macs = _module_linear_macs(p)
    ratio = pred_macs / bb
    assert ratio < THRESHOLD, (
        f"predictor MACs {pred_macs/1e6:.2f}M = {ratio:.2%} of one backbone "
        f"pass ({bb/1e6:.2f}M) — the 10% cheapness contract is violated; "
        f"the predictor must stay a narrow head, NOT a second backbone")
    assert isinstance(p, GlimpseFeaturePredictor)  # Adjustment 1 conformance


def test_even_k8_scoring_stays_under_10_percent():
    """The Part 1.E compute constraint, demonstrated: K=8 candidate passes
    of the CHEAP predictor still cost less than 10% of one backbone pass —
    candidate scoring must never approach K backbone passes."""
    bb = _backbone_macs()
    p = ConcreteGlimpseFeaturePredictor()
    ratio_k8 = 8 * _module_linear_macs(p) / bb
    assert ratio_k8 < THRESHOLD, (
        f"even K=8 scoring costs {ratio_k8:.2%} of one backbone pass — "
        f"must stay under {THRESHOLD:.0%}")


def test_predictor_not_a_second_backbone_structurally():
    p = ConcreteGlimpseFeaturePredictor()
    kinds = {type(m).__name__ for m in p.modules()}
    assert "Conv2d" not in kinds, (
        "no patch embed — a conv stem would make this a second backbone")
    # Depth: 6 Linears (cond 2, res 2, token head 1) — not a 12-block trunk.
    n_linear = sum(1 for m in p.modules() if isinstance(m, torch.nn.Linear))
    assert n_linear <= 8


def test_predictions_not_detached():
    """Hard non-detached assertion at the substrate's real width (d_z=384):
    the predicted path must carry autograd (Part 1.A gradient rules — the
    Gen-0 failure class)."""
    from noesis_vision.beliefs.factory import populate_belief
    from noesis_vision.beliefs.vector_belief import DirichletParams, GazeState
    import torch.nn.functional as F
    B, DZ, C, N = 2, 384, 10, 16
    p = ConcreteGlimpseFeaturePredictor()  # defaults match the substrate
    z = torch.randn(B, DZ, requires_grad=True)
    U = DirichletParams(evidence=F.softplus(torch.randn(B, C)))
    E = torch.randn(B, N, DZ)
    b = populate_belief(z=z, U=U, E=E,
                        A=GazeState(gaze_history=[torch.zeros(B, 2)],
                                    current_glimpse_idx=1))
    pred = p.predict_features(b, torch.zeros(B, 2))
    assert pred.grad_fn is not None, (
        "the predicted path must NEVER be detached (Part 1.A gradient "
        "rules — the Gen-0 failure class)")
