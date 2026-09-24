"""
test_j2_ais_v2_integration — Agent J2 (Part 2 steps 5-6).
================================================================================
Integration tests for the J2 wiring of Agent F's AISv2GazePolicy into the
foundation trainer:

  * the machine walks SIX phases, J2's at the end, in ladder order;
  * F's reuse boundary holds in the trainer composition (the policy uses
    THE SAME predictor and evidential head — no second copies) and the
    policy's OWN optimizer group contains ONLY logit_scale;
  * gradient reach: the AIS-v2 selection path trains the shared stack AND
    the policy (the standing rule), while hard (eval) selection trains
    nothing through logit_scale (F's phase rule);
  * the canonical GazeState records EXACTLY T fixations (its cap raises
    loudly — the loop must respect the LOCKED T=4);
  * resume-guard behavior for the J2 phases (per-phase layouts);
  * the placeholder supersession is COMPLETE (Agent B's placeholders are
    GONE; the canonical identities are re-exported).

Synthetic tensors only — no dataset, no HF, no CUDA requirement.
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from noesis_vision.gaze.gaze_state import GazeState as _CanonicalGazeState
from noesis_vision.uncertainty.evidential_head import (
    DirichletParams as _CanonicalDirichletParams)
from training.stage_state_machine import (  # noqa: E402
    FOUNDATION_PHASES,
    advance,
    ensure_foundation_state,
    get_next_action,
)
from training.train_generation1_foundation import (  # noqa: E402
    AIS_V2_GAZE_LABEL,
    FoundationConfig,
    FoundationModel,
    gaze_scheme_for_phase,
)

B, DZ = 4, 384


def _cfg(tmp_path, **kw) -> FoundationConfig:
    cfg = FoundationConfig()
    cfg.use_hf = False            # NEVER network in tests
    cfg.amp = False
    cfg.force_fresh = False
    cfg.ckpt_dir = str(tmp_path / "ckpts")
    cfg.report_dir = str(tmp_path / "report")
    cfg.runs_dir = str(tmp_path / "runs")
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _fresh_model(cfg, phase):
    torch.manual_seed(0)
    return FoundationModel(cfg, phase)


def _build_registry(model):
    from noesis_vision.core.multi_group_optimizer import (
        OptimizerGroupRegistry)
    registry = OptimizerGroupRegistry()
    groups = model.group_params()
    registry.register_backbone(groups["backbone"])
    for name in ("classifier", "evidential_head", "predictor", "update_net",
                 "precision", "gaze_policy"):
        if name in groups:
            registry.register(name, groups[name])
    return registry


# ── machine ─────────────────────────────────────────────────────────────────
def test_machine_walks_six_phases_in_order(tmp_path):
    """Steps 1-6, in ladder order, J2's phases at the end; 'done' after."""
    roadmap_path = str(tmp_path / "rm.json")
    roadmap: dict = {}
    ensure_foundation_state(roadmap)
    os.makedirs(os.path.dirname(roadmap_path), exist_ok=True)
    with open(roadmap_path, "w") as f:
        json.dump(roadmap, f)
    seen = []
    for _ in range(len(FOUNDATION_PHASES) + 1):
        action = get_next_action(json.loads(json.dumps(roadmap)))
        if action.phase is None:
            break
        assert action.phase == FOUNDATION_PHASES[len(seen)]
        seen.append(action.phase)
        roadmap = advance(action.phase, "done", roadmap_path=roadmap_path)
    assert seen == list(FOUNDATION_PHASES), \
        "the machine must walk EXACTLY the six Part 2 phases in order"
    assert seen[-2:] == ["ais_v2_swap", "gen1_core"], \
        "J2's steps 5-6 must close the foundation sequence"
    assert get_next_action(
        json.load(open(roadmap_path))).substep == "done"


def test_machine_rejects_seventh_phase():
    """Steps 7+ are NOT this machine: advancing an unknown phase raises."""
    with pytest.raises(ValueError, match="unknown phase"):
        advance("s_t_arm", "running", roadmap_path="/tmp/nonexistent_rm.json")


# ── F's reuse boundary in the trainer composition ───────────────────────────
def test_trainer_policy_shares_predictor_and_head(tmp_path):
    """The policy's predictor/head are THE SAME modules the trainer owns
    (F's contract), and the gaze_policy group holds ONLY logit_scale."""
    cfg = _cfg(tmp_path)
    m = _fresh_model(cfg, "ais_v2_swap")
    assert m.gaze_policy.predictor is m.predictor, \
        "the policy must use the trainer's shared predictor, not a copy"
    assert m.gaze_policy.evidential_head is m.evidential_head, \
        "the policy must use the trainer's shared evidential head"
    # F's ONLY learned state is logit_scale (recursion into the injected
    # shared modules would silently duplicate their optimizer membership).
    assert set(m.gaze_policy._parameters) == {"logit_scale"}
    groups = m.group_params()
    assert set(groups) == {"backbone", "classifier", "evidential_head",
                           "predictor", "update_net", "precision",
                           "gaze_policy"}
    assert groups["gaze_policy"] == [m.gaze_policy.logit_scale]
    # The composition must be REGISTRABLE without the registry's
    # double-claim guard firing (the whole point of registering only
    # logit_scale under gaze_policy).
    _build_registry(m)   # raises if any parameter is claimed twice


def test_step6_identical_mechanism_set(tmp_path):
    """Step 6 is the integrated system: same mechanisms as step 5 (plus
    nothing new) — S_t stays None, L_stab is not a training objective."""
    cfg = _cfg(tmp_path)
    m5 = _fresh_model(cfg, "ais_v2_swap")
    m6 = _fresh_model(cfg, "gen1_core")
    assert sorted(m5.group_params()) == sorted(m6.group_params()), \
        "step 6 adds NO mechanism beyond step 5's (the frozen reference)"
    assert not hasattr(m6, "s_slots") and not hasattr(m6, "l_stab"), \
        "no S_t slots and no L_stab objective may exist in the core build"


# ── gradient reach (the standing rule, checked explicitly) ──────────────────
def test_ais_v2_phase_gradients_reach_gaze_stack(tmp_path):
    """Soft selection must train logit_scale AND the shared predictor/head;
    the dynamics path must still reach update_net."""
    cfg = _cfg(tmp_path)
    m = _fresh_model(cfg, "ais_v2_swap").to("cpu")
    m.train()
    x = torch.rand(B, 3, 96, 96)
    y = torch.randint(0, cfg.num_classes, (B,))
    loss = torch.nn.functional.cross_entropy(m(x), y)
    loss.backward()
    g = m.gaze_policy.logit_scale.grad
    assert g is not None and torch.isfinite(g).all() and g.abs() > 0, \
        "the gaze selection's gradient never reached logit_scale"
    upd = next(m.update_net.parameters()).grad
    assert upd is not None and upd.abs().sum() > 0, \
        "the belief-dynamics gradient never reached update_net"
    pred = next(m.predictor.parameters()).grad
    assert pred is not None and pred.abs().sum() > 0, \
        "the prediction path's gradient never reached the shared predictor"


def test_ais_v2_eval_uses_hard_selection(tmp_path):
    """Inference: hard argmax, no Gumbel noise, nothing trains through the
    policy — and determinism: two eval passes give identical logits."""
    cfg = _cfg(tmp_path)
    m = _fresh_model(cfg, "gen1_core").to("cpu")
    m.eval()
    x = torch.rand(B, 3, 96, 96)
    y = torch.randint(0, cfg.num_classes, (B,))
    m.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(m(x), y).backward()
    assert m.gaze_policy.logit_scale.grad is None, \
        "hard (eval) selection must not train the policy"
    with torch.no_grad():
        l1, l2 = m(x), m(x)
    assert torch.allclose(l1, l2), \
        "eval-mode AIS-v2 selection must be deterministic"


# ── canonical GazeState accounting in the live loop ─────────────────────────
def test_gaze_state_records_exactly_T_fixations(tmp_path):
    """The live loop's GazeState ends with EXACTLY T records (its cap
    raises loudly at T+1 — the loop must respect the LOCKED T=4)."""
    cfg = _cfg(tmp_path)
    m = _fresh_model(cfg, "ais_v2_swap")
    T = cfg.num_glimpses
    # Re-run the loop's bookkeeping via the policy's record(): start from
    # the fixed first fixation and record T-1 selections.
    class _Sel:  # minimal stand-in: record() only consumes .selected
        selected = torch.full((B, 2), 0.1)

    a0 = torch.zeros(B, 2)
    gs = _CanonicalGazeState(gaze_history=[a0], current_glimpse_idx=0)
    for _ in range(T - 1):
        gs = m.gaze_policy.record(gs, _Sel())
    assert len(gs.gaze_history) == T, \
        f"the loop must record EXACTLY T={T} fixations"
    with pytest.raises(ValueError, match="capacity"):
        m.gaze_policy.record(gs, _Sel())   # T+1 must raise LOUDLY


# ── resume discipline for the J2 phases ─────────────────────────────────────
def test_j2_phase_resume_guard(tmp_path):
    """A J2-phase rolling checkpoint resumes under the SAME phase's group
    layout and is REFUSED under an earlier phase's (fewer groups)."""
    from noesis_vision.core.checkpoint import save_rolling
    cfg = _cfg(tmp_path)
    m5 = _fresh_model(cfg, "ais_v2_swap")
    rolling = os.path.join(cfg.ckpt_dir, "foundation_ais_v2_swap_rolling.pth")
    optimizer = _build_registry(m5).build_optimizer(cfg.lr)
    save_rolling(rolling, epoch=1, model=m5, optimizer=optimizer)

    state = torch.load(rolling, map_location="cpu", weights_only=False)
    same = _build_registry(_fresh_model(cfg, "ais_v2_swap"))
    assert same.resume_guard(state["optimizer"]) is True
    earlier = _build_registry(_fresh_model(cfg, "belief_with_f"))
    assert earlier.resume_guard(state["optimizer"]) is False, \
        "a step-5 checkpoint must never silently restore into a step-4 layout"


# ── supersession COMPLETE (the flagged Agent-J integration task) ────────────
def test_placeholder_gaze_state_is_gone():
    """Agent B's placeholders must NOT coexist with the canonical classes
    (non-improvisation rule): the beliefs module now re-exports the
    canonical identities."""
    from noesis_vision.beliefs import vector_belief
    assert vector_belief.GazeState is _CanonicalGazeState, \
        "beliefs.vector_belief.GazeState must BE the canonical class"
    assert vector_belief.DirichletParams is _CanonicalDirichletParams, \
        "beliefs.vector_belief.DirichletParams must BE the canonical class"
    src = open(os.path.join(REPO_ROOT, "noesis_vision", "beliefs",
                            "vector_belief.py")).read()
    for marker in ("class DirichletParams:", "class GazeState:"):
        assert marker not in src, \
            f"a placeholder definition still coexists: {marker!r}"


# ── gaze-scheme honesty (never mislabel) ────────────────────────────────────
def test_gaze_scheme_labels_are_honest(tmp_path):
    cfg = _cfg(tmp_path)
    assert "AIS" in gaze_scheme_for_phase("ais_v2_swap").upper()
    assert "AIS" in gaze_scheme_for_phase("gen1_core").upper()
    for p in ("recurrence_only", "belief_no_f", "belief_with_f"):
        assert "not AIS-v2" in gaze_scheme_for_phase(p), \
            "steps 1-4 must keep the PLACEHOLDER label (never AIS-v2)"
    assert gaze_scheme_for_phase("backbone_only") == "single_center_fixation"
    m = _fresh_model(cfg, "ais_v2_swap")
    assert m.gaze_scheme == AIS_V2_GAZE_LABEL
