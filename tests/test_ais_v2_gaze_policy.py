"""
Agent F contract tests — AIS-v2 policy, candidate sampler, GazeState.
================================================================================

Covers the contract's named tests:
  test_shares_agent_e_predictor          — identity on the injected class
  test_candidate_scoring_shapes          — (B,K,2)/(B,K)/(B,2), K range
  test_soft_selection_gradient_flow      — policy AND Agent E's predictor
                                           AND Agent D's head receive grad
  test_hard_selection_no_gradient_at_inference
  test_t0_uses_saliency_only             — the uncertainty-reduction term
                                           is STRUCTURALLY ABSENT at t=0

plus the smoke experiment (K=6, in-bounds coords, cross-boundary grad).
"""
from __future__ import annotations

import pytest
import torch

from noesis_vision.beliefs.factory import populate_belief
from noesis_vision.beliefs.vector_belief import GazeState
from noesis_vision.core.schema import AIS_T0_SCORING
from noesis_vision.gaze.ais_v2_policy import AISv2GazePolicy
from noesis_vision.gaze.candidate_sampler import (
    HeuristicCandidateSampler,
    token_error_map,
    token_grid_centers,
)
from noesis_vision.models.backbone import CompactViT
from noesis_vision.predictive_coding.glimpse_predictor import (
    ConcreteGlimpseFeaturePredictor,
)
from noesis_vision.uncertainty.evidential_head import (
    DirichletParams,
    EvidentialHead,
)

B, C, DZ, N, DF = 3, 8, 384, 16, 384
IMG = 56


def build_substrate(seed: int = 0):
    torch.manual_seed(seed)
    model = CompactViT(img_size=IMG)
    predictor = ConcreteGlimpseFeaturePredictor(d_z=DZ, d_feat=DF, n_tokens=N)
    head = EvidentialHead(input_dim=DF, num_classes=C)
    policy = AISv2GazePolicy(predictor=predictor, evidential_head=head,
                             num_candidates=6,
                             selection_mode="straight_through")
    return model, predictor, head, policy


def make_belief(seed: int = 1, t: int = 1, B_: int = B):
    torch.manual_seed(seed)
    ev = torch.rand(B_, C) + 0.5
    E = torch.zeros(B_, N, DF) if t == 0 else torch.randn(B_, N, DF)
    A = GazeState(gaze_history=[torch.zeros(B_, 2)], current_glimpse_idx=t)
    return populate_belief(z=torch.randn(B_, DZ),
                           U=DirichletParams(evidence=ev),
                           E=E, A=A)


def step_once(policy, model, belief, t: int, training: bool = True,
              seed: int = 0):
    """One cycle: encode glimpse 0, (t>=1) predict, generate/score/select.

    t = 0 calls the policy with predicted_tokens=None — the policy itself
    must not need a prediction (LOCKED fallback); the TEST is what must
    not fabricate a U_0-conditioned prediction.
    """
    torch.manual_seed(seed)
    x = torch.rand(B, 3, IMG, IMG)
    a0 = torch.zeros(B, 2)
    _z0, obs0 = model.encode_glimpse(x, a0)
    pred0 = None if t == 0 else policy.predictor.predict_features(belief, a0)
    sel = policy(belief, obs0.detach(), a0, predicted_tokens=pred0,
                 training=training)
    return sel, pred0, obs0


# ── test_shares_agent_e_predictor ────────────────────────────────────────────
def test_shares_agent_e_predictor():
    """Identity check: the policy's predictor IS Agent E's object — not a
    copy, not a subclass instance, not a reimplementation."""
    _model, predictor, head, policy = build_substrate()
    assert policy.predictor is predictor
    assert type(policy.predictor) is ConcreteGlimpseFeaturePredictor
    assert policy.predictor.__class__ is predictor.__class__
    # The policy's DIRECT parameters are exactly logit_scale (submodule
    # params belong to the injected shared modules, not to this policy):
    # a second predictor/scoring head would show up as extra direct params.
    own = set(policy._parameters.keys())
    assert own == {"logit_scale"}, (
        f"the policy grew its own learnable scoring machinery: {own} — "
        f"a second scoring head is LOCKED OUT (Part 1.E)")
    # ...and the same object identity holds on the uncertainty side.
    assert policy.evidential_head is head


# ── test_candidate_scoring_shapes ────────────────────────────────────────────
def test_candidate_scoring_shapes():
    _model, predictor, head, policy = build_substrate()
    belief = make_belief(t=1)
    sel, _pred0, _obs0 = step_once(policy, _model, belief, t=1)
    K = policy.num_candidates
    assert sel.candidates.shape == (B, K, 2)             # candidates
    assert sel.scores.scores.shape == (B, K)             # scores
    assert sel.selected.shape == (B, 2)                  # selected location
    assert sel.scores.reduction.shape == (B, K)
    assert sel.scores.kind == "uncertainty_reduction"
    # Coordinates stay in bounds (clamp at +/-0.9, the Gen-0 convention).
    assert sel.candidates.abs().max().item() <= 0.9 + 1e-6
    # K range enforcement (LOCKED 4-8) on the policy and the sampler.
    with pytest.raises(ValueError):
        AISv2GazePolicy(predictor=predictor, evidential_head=head,
                        num_candidates=3)
    with pytest.raises(ValueError):
        HeuristicCandidateSampler(num_candidates=9)
    # Token-grid centers land on patch centers: with gaze (0,0) and a
    # full-frame fovea, patch centers sit at 1/8, 3/8, 5/8, 7/8 of the
    # crop -> normalized offsets {-0.75, -0.25, +0.25, +0.75}.
    centers = token_grid_centers(torch.zeros(1, 2), fovea_size=IMG,
                                 img_size=IMG)
    expected = torch.tensor([-0.75, -0.25, 0.25, 0.75])
    assert torch.allclose(centers[0, :, :, 0].unique(), expected, atol=1e-6)


# ── test_soft_selection_gradient_flow ────────────────────────────────────────
def test_soft_selection_gradient_flow():
    """Backprop from a dummy loss through the soft-selected coordinates
    reaches the policy AND Agent E's predictor AND Agent D's head —
    the shared-predictor design proven end to end (the smoke's
    cross-boundary gradient check)."""
    _model, predictor, head, policy = build_substrate()
    belief = make_belief(t=1)
    sel, _pred0, _obs0 = step_once(policy, _model, belief, t=1)
    assert sel.weights is not None and sel.weights.requires_grad
    ref = torch.randn(B, 2)
    loss = (sel.selected * ref).sum()
    loss.backward()

    assert policy.logit_scale.grad is not None, \
        "the policy's own parameters received no gradient"
    pred_grads = [p.grad for p in predictor.parameters() if p.grad is not None]
    assert pred_grads, \
        "gradient did NOT reach Agent E's predictor across the agent boundary"
    head_grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert head_grads, "gradient did not reach Agent D's evidential head"
    # Selection weights sum to one (Gumbel-softmax property).
    assert torch.allclose(sel.weights.sum(dim=1), torch.ones(B), atol=1e-5)


# ── smoke experiment (contract) ──────────────────────────────────────────────
def test_smoke_k6_in_bounds_and_cross_boundary_gradient():
    """K=6 candidates on a dummy batch; coordinates in bounds; soft-path
    gradient reaches Agent E's predictor parameters (not just F's own)."""
    _model, predictor, _head, policy = build_substrate(seed=7)
    belief = make_belief(t=1, seed=8)
    sel, _pred0, _obs0 = step_once(policy, _model, belief, t=1, seed=9)
    assert policy.num_candidates == 6
    assert sel.candidates.shape == (B, 6, 2)
    assert sel.candidates.abs().max().item() <= 0.9
    (sel.selected.sum()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in predictor.parameters())


# ── test_hard_selection_no_gradient_at_inference ─────────────────────────────
def test_hard_selection_no_gradient_at_inference():
    """Hard argmax at inference: chosen coordinates carry NO gradient, and
    no Gumbel noise (deterministic across repeated calls)."""
    _model, _predictor, _head, policy = build_substrate()
    belief = make_belief(t=1)
    sel1, _, _ = step_once(policy, _model, belief, t=1, training=False,
                           seed=3)
    assert sel1.weights is None and sel1.chosen_idx is not None
    assert sel1.selected.requires_grad is False
    assert sel1.selected.grad_fn is None
    # Same candidates AND same scores -> identical choice: the hard path
    # adds NO sampling noise of its own (generation noise is the
    # sampler's, seeded via its generator — a different thing entirely).
    torch.manual_seed(3)
    x = torch.rand(B, 3, IMG, IMG)
    _z0, obs0 = _model.encode_glimpse(x, torch.zeros(B, 2))
    pred0 = policy.predictor.predict_features(belief, torch.zeros(B, 2))
    gen_a = torch.Generator().manual_seed(77)
    gen_b = torch.Generator().manual_seed(77)
    sel_a = policy(belief, obs0.detach(), torch.zeros(B, 2),
                   predicted_tokens=pred0, training=False, generator=gen_a)
    sel_b = policy(belief, obs0.detach(), torch.zeros(B, 2),
                   predicted_tokens=pred0, training=False, generator=gen_b)
    assert torch.equal(sel_a.candidates, sel_b.candidates)
    assert torch.equal(sel_a.selected, sel_b.selected), \
        "inference selection must be deterministic (no sampling noise)"
    # And the scored candidate that won is the one that was chosen.
    idx = sel1.scores.scores.argmax(dim=1)
    assert torch.equal(sel1.chosen_idx, idx)


# ── test_t0_uses_saliency_only ───────────────────────────────────────────────
def test_t0_uses_saliency_only():
    """t = 0 (LOCKED): candidates are scored by heuristic saliency ONLY —
    the uncertainty-reduction term is STRUCTURALLY ABSENT (None in the
    score record, no predictor call, no Dirichlet, no entropy), not
    present-and-zero. No gradient reaches the predictor from a t=0 step."""
    _model, predictor, head, policy = build_substrate()
    for p in list(predictor.parameters()) + list(head.parameters()):
        p.grad = None                                    # hermetic: fresh slate
    belief = make_belief(t=0)
    sel, pred0, obs0 = step_once(policy, _model, belief, t=0,
                                 training=True, seed=5)
    assert pred0 is None, \
        "a t=0 step must not fabricate a U_0-conditioned prediction"
    cs = sel.scores
    K = policy.num_candidates
    assert cs.kind == "heuristic_saliency" and cs.t == 0
    assert cs.reduction is None, (
        "the uncertainty-reduction term must be structurally absent at "
        "t=0 (None), not a zero tensor")
    assert cs.saliency is not None and cs.saliency.shape == (B, K)
    assert "heuristic-saliency" in repr(policy) or AIS_T0_SCORING in repr(policy)
    # Deterministic surface: candidate 0 IS the anchor (the argmax token),
    # so its saliency equals the map's max.
    err_map = token_error_map(obs0.detach(), None, 4)
    assert torch.allclose(
        cs.saliency[:, 0], err_map.reshape(B, -1).max(dim=1).values)
    # Soft selection still runs at t=0 (weights over the detached surface)…
    assert sel.weights is not None
    (sel.selected.sum()).backward()
    # …but NO gradient may reach the shared stack from the t=0 path.
    assert all(p.grad is None for p in predictor.parameters()), (
        "a t=0 step must not train the predictor — the U_0-conditioned "
        "term does not exist (LOCKED)")
    assert all(p.grad is None for p in head.parameters())


# ── sampler determinism + error-map semantics ────────────────────────────────
def test_sampler_deterministic_given_generator_and_map_semantics():
    torch.manual_seed(0)
    sampler = HeuristicCandidateSampler(num_candidates=4)
    g1 = torch.Generator().manual_seed(11)
    g2 = torch.Generator().manual_seed(11)
    obs = torch.randn(2, N, DF)
    pred = torch.randn(2, N, DF)
    m1 = token_error_map(obs, pred, 4)
    m2 = token_error_map(obs, pred, 4)
    assert torch.equal(m1, m2)
    c1, _ = sampler.forward(m1, torch.zeros(2, 2), generator=g1)
    c2, _ = sampler.forward(m1, torch.zeros(2, 2), generator=g2)
    assert torch.equal(c1, c2)                     # seeded generation
    assert torch.equal(c1[:, 0], c2[:, 0])         # candidate 0 = anchor
    # The anchor is the argmax token center of the given map.
    b = 1
    am = m1[b].argmax()
    r, c = am // 4, am % 4
    expected = torch.tensor([(c.item() / 2.0 - 0.75),
                             (r.item() / 2.0 - 0.75)])   # gaze (0,0), full frame
    assert torch.allclose(c1[b, 0], expected, atol=1e-6)
    # Error map: prediction-error branch differs from the t=0 magnitude
    # branch, and the t=0 branch has NO predicted term in its formula.
    m_t0 = token_error_map(obs, None, 4)
    assert not torch.allclose(m1, m_t0)
    assert torch.allclose(m_t0, obs.pow(2).mean(dim=-1).reshape(2, 4, 4))
