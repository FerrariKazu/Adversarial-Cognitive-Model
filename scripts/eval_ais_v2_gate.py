#!/usr/bin/env python3
"""
AIS-v2 smoke gate — the two mechanism-specific checks beyond gradient flow.
================================================================================

The D2 smoke gate requires, beyond the gradient-flow test
(tests/test_ais_v2_gradient_flow.py):

  G1. GAZE-SHIFT DISTANCE: the policy must actually move the gaze (mean
      per-step |Δa| over the foraging loop above a pre-registered floor).
  G2. CANDIDATE PREFERENCE (the genuinely new check): candidate selection
      must show MEASURABLE preference for higher-expected-gain candidates
      over random selection. The policy's own one-step TD signal makes this
      directly testable: over the probe set, the surprise the head PREDICTED
      at the chosen candidate must correlate with the surprise ACTUALLY
      observed at that fixation next step (the trajectory's eig_pairs). A
      head that learned "where surprise lands" has corr > 0; a random
      selector sits at ~0. This is tested directly — not just "gaze moves".

Uses the smoke checkpoint's own eig_pairs (predicted vs observed surprise
per step/sample) from forward passes over a held-out probe subset. Verdict
written to report/rhan_nx_ais_v2_smoke_gate.json.

Criteria (pre-registered 2026-09-08):
  G1: mean |Δa|/step >= 0.02 (v12's base gaze step is 0.20 — an order of
      magnitude below that means the policy is stuck at the anchor).
  G2: Pearson corr(predicted, observed) > 0.05 with n_samples >= 64 —
      measurably above the ~0.0 a random-selection policy would show.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

GAZE_SHIFT_FLOOR = 0.02
CANDIDATE_CORR_FLOOR = 0.05
MIN_PAIRS = 64

VERDICT_PATH = os.path.join(REPO_ROOT, "report",
                            "rhan_nx_ais_v2_smoke_gate.json")


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.size < 2:
        return 0.0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def evaluate(model, loader, device, max_samples: int = 512) -> Dict:
    """Run the two gate checks over a probe subset of the test set."""
    preds: List[float] = []
    obs: List[float] = []
    gaze_shifts: List[float] = []
    model.eval()
    n = 0
    with torch.no_grad():
        for x, _y in loader:
            x = x.to(device)
            logits, traj = model(x, return_trajectory=True)
            for p, o in traj.get("eig_pairs") or []:
                preds.extend(p.cpu().tolist())
                obs.extend(o.cpu().tolist())
            # Per-step gaze displacement: trajectory 'actions' are not
            # collected by default, so measure via the policy's last_chosen
            # against the anchor (first candidate = stay put). Mean over the
            # probe of ||chosen - anchor||.
            pol = getattr(model, "gaze_policy_v2", None)
            if pol is not None and pol.last_chosen is not None \
                    and pol.last_candidates is not None:
                anchor = pol.last_candidates[:, 0]
                shift = (pol.last_chosen - anchor).norm(dim=-1)
                gaze_shifts.extend(shift.cpu().tolist())
            n += x.shape[0]
            if n >= max_samples:
                break
    preds = np.array(preds)
    obs = np.array(obs)
    gaze_shifts = np.array(gaze_shifts)

    corr = pearson(preds, obs) if preds.size >= 2 else 0.0
    n_pairs = int(preds.size)
    mean_shift = float(gaze_shifts.mean()) if gaze_shifts.size else 0.0
    n_shift = int(gaze_shifts.size)

    g1 = mean_shift >= GAZE_SHIFT_FLOOR
    g2 = (n_pairs >= MIN_PAIRS) and (corr > CANDIDATE_CORR_FLOOR)
    return {
        "g1_gaze_shift": {
            "passed": bool(g1),
            "mean_shift_per_step": round(mean_shift, 6),
            "floor": GAZE_SHIFT_FLOOR,
            "n_samples": n_shift,
        },
        "g2_candidate_preference": {
            "passed": bool(g2),
            "pearson_corr_predicted_vs_observed": round(corr, 6),
            "floor": CANDIDATE_CORR_FLOOR,
            "n_pairs": n_pairs,
            "min_pairs": MIN_PAIRS,
            "note": "corr > 0.05 with >= 64 pairs = measurably above the "
                    "~0.0 a random-selection policy shows (the thing AIS-v1's "
                    "name claimed but never demonstrated).",
        },
        "passed": bool(g1 and g2),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True,
                    help="AIS-v2 smoke checkpoint (best or rolling).")
    ap.add_argument("--out", default=VERDICT_PATH)
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)

    sys.path.insert(0, REPO_ROOT)
    sys.path.insert(0, os.path.join(REPO_ROOT, "phase1_training"))
    from torch.utils.data import DataLoader

    from checkpoint_utils import compat_load
    from rhan_core.config.pillar_config import RHANNextConfig
    from rhan_core.model import RHANNext
    from dataset_stl10 import get_stl10_test

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt
    if not os.path.exists(ckpt):
        cand = os.path.join(REPO_ROOT, "checkpoints", ckpt)
        if os.path.exists(cand):
            ckpt = cand
    print(f"[ais_v2_gate] loading {ckpt}", flush=True)
    state = compat_load(ckpt, map_location="cpu")
    cfg = RHANNextConfig.from_dict(state["config"])
    assert cfg.ais_variant == "info_gain_v2", \
        f"expected ais_variant=info_gain_v2, got {cfg.ais_variant!r}"
    model = RHANNext(config=cfg).to(device)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    print(f"  loaded {len(state['model'])-len(missing)}/{len(state['model'])} "
          f"keys (missing={len(missing)} unexpected={len(unexpected)})",
          flush=True)
    model.eval()

    loader = DataLoader(get_stl10_test(root=os.path.join(REPO_ROOT, "data")),
                        batch_size=args.batch_size, shuffle=False)
    verdict = evaluate(model, loader, device, max_samples=args.samples)
    verdict.update({
        "schema": "ais_v2_smoke_gate_v1",
        "ckpt": ckpt,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(verdict, f, indent=2)
        f.write("\n")
    print(json.dumps(verdict, indent=2), flush=True)
    print("  =>", "PASS — proceed to the 60-epoch AIS-v2 run" if verdict["passed"]
          else "FAIL — STOP and diagnose the candidate-evaluation head",
          flush=True)
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())