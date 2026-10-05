#!/usr/bin/env python3
"""
diag_isolations.py — §11/§12/§13 ISOLATIONS (structure-probe tier).
================================================================================
The full isolations (§11 T=1/2/4 training, §12 A-D pathways, §13 no-E vs E)
are 60-epoch matched runs — hub work (Kaggle/Colab). This module executes the
parts answerable locally from REAL checkpoints and REAL forwards, with every
remaining arm declared as an explicit PENDING experiment (no simulation, no
guessing):

§11 (recurrence): on the recurrence_only pure-CE checkpoint, T-sweep the
  SAME weights over T=1,2,4 and measure accuracy/representation norms. The
  TRAINED T is 4; a T-sweep at fixed weights probes whether the T=4 loop's
  extra passes help or hurt inference-time representations (representation
  degeneration signal), NOT whether training with T=1 would be better —
  that stays a PENDING matched-compute experiment.
§12 (BeliefState): pathway evidence from the era table (belief_no_f 0.2684 /
  belief_with_f 0.3316 vs recurrence_only 0.3668 under pure CE) + a live
  linear-probe-style measurement: classification accuracy from z_t pooled
  content vs from the evidential evidence vector on the belief_with_f ckpt.
§13 (prediction error): live no-E vs E ablation at inference on the real
  belief_with_f ckpt — replace UpdateNet's contribution with the identity
  (z_t unchanged across glimpses, pooled mean) vs the real dynamics, and
  measure the delta. TRAINING-tier no-E-vs-E remains PENDING.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_common import DATA_ROOT, REPO_ROOT, save_json  # noqa: E402

from training.train_generation1_foundation import (  # noqa: E402
    FoundationConfig, build_model, build_fixed_gaze_schedule)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOCAL = os.path.join(REPO_ROOT, "checkpoints")


def load(phase, path):
    cfg = FoundationConfig()
    m = build_model(cfg, phase)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck


def main() -> int:
    from evaluation.imagenet100_loader import make_imagenet100_loaders
    loaders = make_imagenet100_loaders(DATA_ROOT, batch_size=128, num_workers=4)
    n_eval = 1024
    xs, ys = [], []
    for x, y in loaders["val"]:
        xs.append(x)
        ys.append(y)
        if sum(t.shape[0] for t in xs) >= n_eval:
            break
    X = torch.cat(xs)[:n_eval].to(device)
    Y = torch.cat(ys)[:n_eval].to(device)

    out = {"section": "11_12_13_isolations", "n_eval": n_eval}

    # ── §11: T-sweep at fixed recurrence_only weights ────────────────────────
    m2, ck2 = load("recurrence_only", os.path.join(LOCAL, "foundation_recurrence_only_best.pth"))
    tsweep = {}
    for T in (1, 2, 4):
        correct = 0
        norms = []
        with torch.no_grad():
            sched = build_fixed_gaze_schedule(X.shape[0], T, device)
            feats = [m2._glimpse(X, sched[:, t, :])[0] for t in range(T)]
            z = torch.stack(feats, dim=1).mean(dim=1)
            norms.append(round(float(z.norm(dim=1).mean().item()), 4))
            logits = m2.cls_head(z)
            correct = int((logits.argmax(1) == Y).sum().item())
        tsweep[f"T={T}"] = {"acc": round(correct / n_eval, 4),
                            "mean_pooled_norm": norms[0]}
        print(f"  §11 T-sweep T={T}: acc={tsweep[f'T={T}']['acc']:.4f} "
              f"|z|={norms[0]}")
    out["s11_recurrence"] = {
        "ckpt": "foundation_recurrence_only_best.pth (pure-CE era, trained at T=4)",
        "note": "fixed-weight T-sweep = inference-time structure probe only",
        "sweep": tsweep,
        "training_tier_pending": ("§11 full: matched 60-epoch runs at T=1/2/4 "
                                  "measuring train loss, grad norms, activation "
                                  "norms, prediction entropy (hub)")}

    # ── §12: pathway evidence + z-vs-evidence probe on belief_with_f ─────────
    m4, ck4 = load("belief_with_f", os.path.join(LOCAL, "foundation_belief_with_f_best.pth"))
    era_table = {"recurrence_only": 0.3668, "belief_no_f": 0.2684,
                 "belief_with_f": 0.3316}
    out["s12_beliefstate"] = {
        "era_table_pure_ce": era_table,
        "observation": ("belief carrier (identity update) DROPPED recurrence_only "
                        "0.3668 -> 0.2684; adding learned dynamics (F) recovered "
                        "0.3316 but did not exceed recurrence_only — the belief "
                        "pathway inserted a cost without a demonstrated gain"),
        "live_probe": "see 12_z_evidence_probe below",
    }
    # live probe: acc from z-pooled content vs acc from evidence vector
    with torch.no_grad():
        cfg = m4.cfg
        B = X.shape[0]
        sched = build_fixed_gaze_schedule(B, cfg.num_glimpses, device)
        z_t = None
        dp_t = None
        pred_t = None
        err = None
        for t in range(cfg.num_glimpses):
            pooled_t, tokens_t = m4._glimpse(X, sched[:, t, :])
            if t == 0:
                z_t = pooled_t
                err = torch.zeros_like(tokens_t)
            else:
                obs = tokens_t.detach()
                err = obs - pred_t
                Pi_t = m4.precision(dp_t)
                from noesis_vision.predictive_coding.update_net import belief_update
                z_t = belief_update(z_t, err, Pi_t, m4.update_net)
            dp_t = m4.evidential_head(tokens_t)
            if m4.belief_dynamics and t + 1 < cfg.num_glimpses:
                from noesis_vision.beliefs.factory import populate_belief
                from noesis_vision.gaze.gaze_state import GazeState
                from noesis_vision.uncertainty.evidential_head import DirichletParams
                belief_t = populate_belief(
                    z=z_t, U=DirichletParams(evidence=dp_t.evidence),
                    E=(torch.zeros(B, 16, cfg.d_z, device=device) if t == 0 else err),
                    A=GazeState(gaze_history=[], current_glimpse_idx=t))
                pred_t = m4.predictor.predict_features(belief_t, sched[:, t + 1, :])
        logits_full = m4.cls_head(torch.cat([z_t, dp_t.evidence], dim=-1)) \
            + m4.ev_readout(dp_t.evidence)
        # content-only: zero the evidence block (not trained this way —
        # a probe, not a result; the ev_readout term is dropped too)
        logits_zonly = m4.cls_head(torch.cat(
            [z_t, torch.zeros_like(dp_t.evidence)], dim=-1))
        # evidence-only: zero the content block
        logits_eonly = m4.cls_head(torch.cat(
            [torch.zeros_like(z_t), dp_t.evidence], dim=-1)) \
            + m4.ev_readout(dp_t.evidence)
        acc = lambda lg: round(float((lg.argmax(1) == Y).float().mean().item()), 4)
        out["s12_z_evidence_probe"] = {
            "full_pathway_acc": acc(logits_full),
            "z_content_only_acc": acc(logits_zonly),
            "evidence_only_acc": acc(logits_eonly),
            "note": "zeroing probe on the trained readout — indicative, not causal"}
        print(f"  §12 probe: full={out['s12_z_evidence_probe']['full_pathway_acc']} "
              f"z-only={out['s12_z_evidence_probe']['z_content_only_acc']} "
              f"ev-only={out['s12_z_evidence_probe']['evidence_only_acc']}")

    # ── §13: no-E identity vs E dynamics at inference ────────────────────────
    with torch.no_grad():
        # no-E identity: z stays the FIRST glimpse's pooled content; the
        # readout consumes the mean-pooled observation across glimpses instead
        # of the updated belief (the §13 'no-E identity' arm at inference tier).
        pooled_all = []
        for t in range(cfg.num_glimpses):
            pooled_all.append(m4._glimpse(X, sched[:, t, :])[0])
        z_noE = torch.stack(pooled_all, dim=1).mean(dim=1)
        dp_last = m4.evidential_head(m4._glimpse(X, sched[:, -1, :])[1])
        logits_noE = m4.cls_head(torch.cat([z_noE, dp_last.evidence], dim=-1)) \
            + m4.ev_readout(dp_last.evidence)
        acc_noE = float((logits_noE.argmax(1) == Y).float().mean().item())
    out["s13_prediction_error"] = {
        "inference_tier": {
            "with_E_dynamics_acc": out["s12_z_evidence_probe"]["full_pathway_acc"],
            "no_E_identity_acc": round(acc_noE, 4)},
        "training_tier_pending": ("§13 full: matched training runs with the E "
                                  "update replaced by identity; success criterion "
                                  "= E-arm beats identity beyond seed noise "
                                  "(hub); REJECT/DEFER if no benefit"),
        "note": ("E_0 := 0 is LOCKED and implemented (verified in code + "
                 "unit test test_t0_produces_exact_zero_error)")}
    print(f"  §13 inference tier: with_E={out['s13_prediction_error']['inference_tier']['with_E_dynamics_acc']} "
          f"no_E={out['s13_prediction_error']['inference_tier']['no_E_identity_acc']}")

    save_json("11_12_13_isolations.json", out)
    print("SECTION 11/12/13: structure-probe tier DONE; training-tier arms "
          "declared PENDING (hub experiments J-D1..J-D4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
