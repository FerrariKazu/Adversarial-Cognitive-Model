#!/usr/bin/env python3
"""RHAN-NXA Revision-1 audit diagnostics (READ-ONLY, exploratory).

Nothing here writes to report/, checkpoints/, or any registered artifact.
Each check is a small deterministic synthetic test.

Run:  .venv/bin/python scratch/audit_rev1/diag_metrics.py
"""
from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OUT = {}


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 1. Reproduce the CAL2 "NLL" formula and compare with true classification NLL
# ---------------------------------------------------------------------------
hr("CHECK 1 — CAL2 'NLL' vs true classification NLL")

# 5 samples, 4 classes, known probabilities. Two samples are confidently WRONG.
probs = torch.tensor([
    [0.70, 0.10, 0.10, 0.10],   # correct
    [0.05, 0.90, 0.03, 0.02],   # correct
    [0.60, 0.20, 0.15, 0.05],   # correct
    [0.80, 0.10, 0.05, 0.05],   # WRONG (label 1)
    [0.55, 0.15, 0.25, 0.05],   # WRONG (label 2)
], dtype=torch.float64)
labels = torch.tensor([0, 1, 2, 1, 2])

logits = torch.log(probs)
true_nll = F.cross_entropy(logits, labels).item()          # -log p(y)
cal2_nll = -torch.log(probs.max(dim=-1).values.clamp(min=1e-8)).mean().item()

print(f"  true classification NLL (-log p(y)) : {true_nll:.6f}")
print(f"  CAL2 implementation  (-log max p)   : {cal2_nll:.6f}")
print(f"  identical?                          : {abs(true_nll - cal2_nll) < 1e-9}")
OUT["cal2_nll"] = {
    "true_nll": true_nll,
    "cal2_nll": cal2_nll,
    "identical": abs(true_nll - cal2_nll) < 1e-9,
}

# Direction test: make the model confidently wrong. True NLL must RISE,
# -log(max p) must FALL. This is the reported "NLL decrease under attack".
conf_wrong = torch.tensor([
    [0.02, 0.96, 0.01, 0.01],
    [0.02, 0.96, 0.01, 0.01],
    [0.02, 0.96, 0.01, 0.01],
    [0.96, 0.02, 0.01, 0.01],   # confident WRONG on label 1
    [0.96, 0.02, 0.01, 0.01],   # confident WRONG on label 2
], dtype=torch.float64)
lg2 = torch.log(conf_wrong)
t2 = F.cross_entropy(lg2, labels).item()
c2 = -torch.log(conf_wrong.max(dim=-1).values.clamp(min=1e-8)).mean().item()
print(f"  after 'confidently wrong' shift: true NLL={t2:.4f} (rises), "
      f"CAL2 NLL={c2:.4f} (falls)")
OUT["cal2_nll_direction"] = {"true_nll_conf_wrong": t2, "cal2_nll_conf_wrong": c2}

# ---------------------------------------------------------------------------
# 2. Reproduce the fp16 entropy NaN in g2_cal2_uncertainty.py
# ---------------------------------------------------------------------------
hr("CHECK 2 — fp16 entropy NaN reproduction")

# g2_cal2_uncertainty.py computes:
#   probs = F.softmax(logits, -1)            # OUTSIDE autocast -> inherits logits dtype
#   entropy = -(probs * (probs + 1e-8).log()).sum(-1)
# Under autocast the model returns fp16 logits. 1e-8 is not representable in
# fp16 (min subnormal 5.96e-8), so the epsilon is flushed to 0 and any class
# whose fp16 probability underflows to exactly 0 yields log(0) = -inf and
# 0 * -inf = NaN.
torch.manual_seed(0)
logits32 = torch.randn(8, 100) * 12.0   # sharp logits, as a trained 100-class head
logits16 = logits32.half()

for name, lg in (("fp32", logits32), ("fp16", logits16)):
    p = F.softmax(lg, dim=-1)
    ent = -(p * (p + 1e-8).log()).sum(dim=-1)
    n_zero = int((p == 0).sum())
    print(f"  {name}: probs==0 count={n_zero:5d}  "
          f"entropy finite={bool(torch.isfinite(ent).all())}  "
          f"mean={ent.mean().item()}")
    OUT[f"entropy_{name}"] = {
        "n_zero_probs": n_zero,
        "all_finite": bool(torch.isfinite(ent).all()),
        "mean": (None if not torch.isfinite(ent).all() else float(ent.mean())),
    }

# Confirm what torch.argsort does with NaN (used by the AUROC rank sum).
s = torch.tensor([0.5, float("nan"), 0.2, 0.9])
order = torch.argsort(s).tolist()
print(f"  argsort([0.5, nan, 0.2, 0.9]) -> {order} (NaN sorted last)")
OUT["argsort_nan_last"] = order

# ---------------------------------------------------------------------------
# 3. epsilon-constraint verification on the ACTUAL eval PGD implementation
# ---------------------------------------------------------------------------
hr("CHECK 3 — run_pgd_ce_eval epsilon constraint (L-inf, normalized space)")

try:
    from training.train_gen2_foundation import run_pgd_ce_eval

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.f = torch.nn.Linear(3 * 8 * 8, 5)

        def forward(self, x):
            return self.f(x.flatten(1))

    torch.manual_seed(0)
    m = Toy().eval()
    x = torch.randn(4, 3, 8, 8)
    y = torch.randint(0, 5, (4,))
    for eps in (0.031, 0.094):
        adv = run_pgd_ce_eval(m, x, y, eps=eps, steps=10)
        linf = (adv - x).abs().max().item()
        print(f"  eps={eps:.3f}  measured ||adv-x||_inf={linf:.8f}  "
              f"violated={linf > eps + 1e-6}")
        OUT[f"eps_{eps}"] = {"measured_linf": linf, "violated": linf > eps + 1e-6}

    # alpha = eps/steps (NOT 2*eps/steps): first step moves at most eps/50.
    eps = 0.094
    adv1 = run_pgd_ce_eval(m, x, y, eps=eps, steps=1)
    print(f"  steps=1: ||adv-x||_inf={float((adv1 - x).abs().max()):.6f} "
          f"(=-alpha={eps:.4f}); PGD-50 alpha={eps/50:.6f}")
    OUT["pgd_alpha"] = {"steps1_linf": float((adv1 - x).abs().max()),
                        "alpha_steps50": eps / 50}
except Exception as exc:  # noqa: BLE001
    print(f"  could not import run_pgd_ce_eval: {exc}")
    OUT["pgd_eps"] = {"error": str(exc)}

# ---------------------------------------------------------------------------
# 4. EOT helper: does it actually average over stochastic forward passes?
# ---------------------------------------------------------------------------
hr("CHECK 4 — gen2_foundation.eot semantics")

try:
    from gen2_foundation.eot import eot_pgd_attack

    calls = {"n": 0}

    class Counter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Linear(3 * 4 * 4, 3)

        def forward(self, x):
            calls["n"] += 1
            return self.w(x.flatten(1))

    cm = Counter().eval()
    x = torch.randn(2, 3, 4, 4)
    y = torch.tensor([0, 1])
    calls["n"] = 0
    _ = eot_pgd_attack(cm, x, y, eps=0.1, alpha=0.01, steps=5, n_eot=4,
                       clamp_min=-4.0, clamp_max=4.0)
    print(f"  forward calls for steps=5, n_eot=4: {calls['n']} "
          f"(expected 20 = steps*n_eot)")
    OUT["eot_forward_calls"] = calls["n"]
except Exception as exc:  # noqa: BLE001
    print(f"  eot import failed: {exc}")
    OUT["eot"] = {"error": str(exc)}

# ---------------------------------------------------------------------------
# 5. Checkpoint key compatibility with strict=False loading
# ---------------------------------------------------------------------------
hr("CHECK 5 — checkpoint <-> model key match (eval scripts use strict=False)")

try:
    from gen2_foundation.model import Gen2FoundationModel

    for path, phase in [
        ("checkpoints/foundation_recurrence_only_final.pth", "recurrence_only"),
        ("checkpoints/foundation_gen1_core_best_clean.pth", "gen1_core"),
    ]:
        saved = torch.load(os.path.join(REPO_ROOT, path), map_location="cpu",
                           weights_only=False)
        sd = saved["model"] if "model" in saved else saved
        m = Gen2FoundationModel(phase=phase, arm="v1")
        own = set(m.state_dict().keys())
        ck = set(sd.keys())
        missing = sorted(own - ck)
        unexpected = sorted(ck - own)
        meta = {k: saved[k] for k in ("epoch", "phase", "best_clean_acc",
                                      "best_robust_acc", "step") if k in saved}
        print(f"  {path}")
        print(f"    phase_arg={phase} keys_in_ckpt={len(ck)} missing={len(missing)} "
              f"unexpected={len(unexpected)}")
        print(f"    metadata={meta}")
        if missing:
            print(f"    MISSING (not loaded): {missing[:8]}")
        if unexpected:
            print(f"    UNEXPECTED (ignored) : {unexpected[:8]}")
        OUT[path] = {"n_missing": len(missing), "n_unexpected": len(unexpected),
                     "missing": missing[:20], "unexpected": unexpected[:20],
                     "metadata": {k: str(v) for k, v in meta.items()}}
except Exception as exc:  # noqa: BLE001
    print(f"  checkpoint check failed: {exc}")
    OUT["ckpt"] = {"error": str(exc)}

# ---------------------------------------------------------------------------
# 6. Does the observation pathway ever see more than 56x56?
# ---------------------------------------------------------------------------
hr("CHECK 6 — foveal_sample geometry at the evaluated 56x56 resolution")

from noesis_vision.models.foveation import foveal_sample

x = torch.arange(3 * 56 * 56, dtype=torch.float32).reshape(1, 3, 56, 56)
g0 = torch.zeros(1, 2)
g1 = torch.tensor([[0.3, 0.3]])
c0 = foveal_sample(x, g0, fovea_size=56)
c1 = foveal_sample(x, g1, fovea_size=56)
# identical? (pure translation with border padding)
print(f"  crop(gaze=0).shape={tuple(c0.shape)}  "
      f"equal_to_input={torch.allclose(c0, x)}")
print(f"  crop(gaze=(0.3,0.3)) mean abs diff vs crop(gaze=0): "
      f"{(c1 - c0).abs().mean().item():.2f}")
print(f"  overlap fraction (non-border pixels): "
      f"{float(((c1 != c1[0,0,0,0]).sum()) / c1.numel()):.3f} (non-constant proxy)")
OUT["foveal"] = {
    "crop_equal_input_at_zero_gaze": bool(torch.allclose(c0, x)),
    "mean_abs_diff_shifted": float((c1 - c0).abs().mean()),
}

print("\n" + "=" * 78)
print(json.dumps(OUT, indent=2, default=str))
