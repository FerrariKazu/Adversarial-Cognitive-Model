"""
AIS-v2 — genuine one-step-lookahead EIG gaze: gradient-flow + mechanism tests.
================================================================================

The AIS-v2 swap test (D2) hinges on three things this file pins:

  1. GRADIENT FLOW (project lesson #1, hard assertion): backward through the
     full model must reach the candidate-evaluation head
     (gaze_policy_v2.candidate_head.*). The head's input features are
     detached by design (documented approximation 3), so its ONLY gradient
     path is the one-step TD pair (predicted vs observed surprise) collected
     in the trajectory — the trainer's L_eig. A regression that detaches or
     drops that pair fails LOUD here.
  2. MECHANISM: with return_trajectory, eig_pairs is populated across
     foraging steps; the policy's chosen candidate is the argmax of the
     predicted-surprise set (max-EIG selection); halted samples keep their
     previous gaze.
  3. BACKWARD COMPAT: ais_variant="info_gain_v2" ONLY changes the gaze
     mechanism — the module is named gaze_policy_v2, AIS-v1's step_net is
     absent, and the default (halting_only) model has no gaze_policy_v2 at
     all (state-dict compat with every pre-RHAN-NX checkpoint).

The candidate-preference gate (predicted-vs-observed surprise correlation
above ~0.0) is a TRAINING-time gate (smoke) — this file checks the head is
trainable through its real gradient path, which is the prerequisite.
"""
import gc
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "phase1_training"))

from rhan_core.config.pillar_config import RHANNextConfig
from rhan_core.model import RHANNext
from train_rhan_next import eig_loss_from_trajectory

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_B = 2


def _cfg_v2():
    return RHANNextConfig(enable_ais=True, ais_variant="info_gain_v2",
                          ais_halt_enabled=True,
                          ais_precision_recon_enabled=False,
                          enable_hpc=True, hpc_num_levels=1,
                          hpc_error_weight=0.10)


def _cfg_v1():
    return RHANNextConfig(enable_ais=True, ais_variant="halting_only",
                          ais_halt_enabled=True,
                          ais_precision_recon_enabled=False,
                          enable_hpc=True, hpc_num_levels=1,
                          hpc_error_weight=0.10)


# ── 1. Gradient flow ────────────────────────────────────────────────────────

def test_ais_v2_gradient_reaches_candidate_head():
    """The FULL trainer loss (CE + w_eig * L_eig) reaches the candidate head.

    CE alone CANNOT reach it by design (documented approximation 3: candidate
    features are detached; the head's ONLY gradient path is the one-step TD
    pair collected in the trajectory). The trainer always adds L_eig, so the
    honest full-loss assertion is CE + w_eig * L_eig.
    """
    model = RHANNext(config=_cfg_v2()).to(_DEVICE)
    x = torch.randn(_B, 3, 96, 96, device=_DEVICE)
    y = torch.randint(0, 10, (_B,), device=_DEVICE)
    model.train()
    with torch.enable_grad():
        logits, traj = model(x, return_trajectory=True)
        loss = (torch.nn.functional.cross_entropy(logits, y)
                + 0.05 * eig_loss_from_trajectory(traj, x.device))
        loss.backward()
    head_grads = {}
    for name, p in model.named_parameters():
        if "gaze_policy_v2" in name and p.grad is not None:
            head_grads[name] = float(p.grad.norm())
    assert head_grads, "no gaze_policy_v2 params received gradients " \
        "(CE alone cannot reach the head by design — L_eig must be in the loss)"
    for name, g in head_grads.items():
        assert g > 0, f"candidate_head param {name} has ZERO gradient"
    del model
    torch.cuda.empty_cache()
    gc.collect()


def test_eig_loss_is_the_head_gradient_path():
    """L_eig (predicted vs observed surprise MSE) alone must move the head."""
    model = RHANNext(config=_cfg_v2()).to(_DEVICE)
    # Zero out CE gradients so ONLY the EIG path can move the head.
    model.eval()
    x = torch.randn(_B, 3, 96, 96, device=_DEVICE)
    with torch.enable_grad():
        _, traj = model(x, return_trajectory=True)
        l_eig = eig_loss_from_trajectory(traj, x.device)
        assert l_eig.requires_grad, "L_eig is detached — head untrainable"
        l_eig.backward()
    head_grads = [float(p.grad.norm()) for name, p in
                  model.named_parameters()
                  if "gaze_policy_v2" in name
                  and p.grad is not None and p.grad.norm() > 0]
    assert head_grads, "L_eig backward did not reach candidate_head params"
    del model
    torch.cuda.empty_cache()
    gc.collect()


# ── 2. Mechanism ────────────────────────────────────────────────────────────

def test_trajectory_has_eig_pairs():
    model = RHANNext(config=_cfg_v2()).to(_DEVICE).eval()
    x = torch.randn(_B, 3, 96, 96, device=_DEVICE)
    with torch.no_grad():
        _, traj = model(x, return_trajectory=True)
    pairs = traj.get("eig_pairs") or []
    # max_foraging_steps=4 -> up to 3 select_action calls -> 3 pairs max.
    assert len(pairs) >= 1, "expected >=1 eig_pairs in trajectory"
    for pred, obs in pairs:
        assert pred.shape == (_B,), pred.shape
        assert obs.shape == (_B,), obs.shape
        assert not obs.requires_grad, "observed surprise must be detached"
    del model
    torch.cuda.empty_cache()
    gc.collect()


def test_candidate_selection_is_argmax():
    """The policy moves toward the max-predicted-surprise candidate (max-EIG
    selection), and a halted sample keeps its previous gaze."""
    model = RHANNext(config=_cfg_v2()).to(_DEVICE).eval()
    policy = model.gaze_policy_v2
    x = torch.randn(_B, 3, 96, 96, device=_DEVICE)
    with torch.no_grad():
        model(x, return_trajectory=True)
    assert policy.last_candidates is not None
    assert policy.last_candidates.shape == (_B, policy.num_candidates, 2)
    surprises = policy.last_candidate_surprises          # (B, K)
    chosen = policy.last_chosen
    # Chosen == candidates at the argmax index.
    idx = surprises.argmax(dim=1)
    for b in range(_B):
        assert torch.allclose(
            chosen[b], policy.last_candidates[b, idx[b]],
            atol=1e-5), f"chosen != argmax candidate for sample {b}"
    # Halted samples keep the previous gaze: the history halt flag is False
    # here (fresh run, entropy gate likely open), so at minimum verify the
    # mechanism: force halt via a direct call with ctx['halt'] = True.
    from rhan_core.beliefs.vector_belief import VectorBeliefState
    prev_action = torch.tensor([[0.3, -0.2], [-0.1, 0.4]], device=_DEVICE)
    ctx = {"action": prev_action, "image": x,
           "belief_tensor": torch.randn(_B, 512, device=_DEVICE),
           "precision": torch.rand(_B, device=_DEVICE),
           "halt": torch.tensor([True, False], device=_DEVICE)}
    a_new = policy.select_action(None, [ctx])
    assert torch.allclose(a_new[0], prev_action[0], atol=1e-6), \
        "halted sample must keep its previous gaze"
    del model
    torch.cuda.empty_cache()
    gc.collect()


def test_eig_loss_zero_when_no_pairs():
    from train_rhan_next import eig_loss_from_trajectory
    l = eig_loss_from_trajectory({}, _DEVICE)
    assert l.item() == 0.0 and not l.requires_grad


# ── 3. Backward compat / isolation ──────────────────────────────────────────

def test_default_config_has_no_gaze_policy_v2():
    """AIS-v1 default (halting_only) must not build the v2 head — state dict
    compatibility with every pre-RHAN-NX checkpoint."""
    model = RHANNext(config=_cfg_v1())
    assert not hasattr(model, "gaze_policy_v2")
    assert "gaze_policy_v2" not in dict(model.state_dict())
    # And AIS-v1's step_net IS present (the swapped-out mechanism).
    assert hasattr(model.gaze_policy, "step_net")


def test_v2_has_no_ais_v1_step_net():
    model = RHANNext(config=_cfg_v2())
    assert hasattr(model, "gaze_policy_v2")
    assert not hasattr(model.gaze_policy_v2, "step_net")