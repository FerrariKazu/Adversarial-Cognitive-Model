"""
lens — read-only introspection layer for RHAN / RHAN-Next checkpoints.
=====================================================================

A diagnostic/demo tool that loads any NOESIS / RHAN-Next checkpoint and
visualises its internal state on a single input, live, as it processes it.

This is NOT a fifth pillar:

  * it does not modify RHANNextConfig,
  * it does not touch train_rhan_next.py or any training/evaluation path,
  * it sits entirely outside the gated pillar system (no resume-gate, no
    --force-restart, no checkpoint writes).

The only contact point with a live model is `hooks.HookRegistry`, which uses
plain torch forward hooks (nn.Module.register_forward_hook) on modules the
model ALREADY executes inside its foraging loop — model code is never edited.

Package layout:

    hooks.py    forward-hook registry (non-invasive observation of the
                per-step modules: foveal_stream, generative_prior, hpc_level1)
    capture.py  per-forward-pass snapshot: StepCapture (one per recurrent
                step) + ForwardResult (classification summary)
    session.py  LensSession — loads a checkpoint (config auto-detected via
                the same logic eval_rhan.py uses), wraps it with hooks, and
                exposes .run(image) as a step-by-step generator

Reuse contract (no duplicated logic): checkpoint config auto-detection and
model loading are delegated to phase2_attacks/eval_rhan.py; HuggingFace
download/cache is delegated to phase1_training/eval_rhan_v11.py; the PGD
attack and STL-10 normalization constants are delegated to
phase2_attacks/eval_full_epsilon_sweep.py. lens/ implements only the
observation + snapshot assembly.
"""
from __future__ import annotations

from rhan_core.lens.hooks import HookRegistry, STEP_MODULES
from rhan_core.lens.capture import (
    StepCapture, ForwardResult, build_forward_result,
    compute_belief_drift, belief_drift_summary,
    gaze_perturbation_correlation, aggregate_gaze_correlation,
)
from rhan_core.lens.session import LensSession, run_captures, batch_belief_drift

__all__ = [
    "HookRegistry",
    "STEP_MODULES",
    "StepCapture",
    "ForwardResult",
    "build_forward_result",
    "compute_belief_drift",
    "belief_drift_summary",
    "gaze_perturbation_correlation",
    "aggregate_gaze_correlation",
    "LensSession",
    "run_captures",
    "batch_belief_drift",
]
