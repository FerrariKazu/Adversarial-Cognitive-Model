#!/usr/bin/env python3
"""
diag_output_sanity.py — §7 MODEL OUTPUT SANITY / COLLAPSE CHECK (real ckpts).
================================================================================
For EVERY preserved best checkpoint (pure-CE era, code cf6ce8a) and the
adversarial-era rolling checkpoints (code fef50f3, downloaded to
diagnosis_artifacts/hf_rolling_adv/), on a fixed real validation batch:
  logits mean/std/min/max, softmax entropy, max-prob, unique predicted
  classes, prediction histogram concentration, top-1 accuracy.
Answers §7 directly: has the network collapsed toward a few classes?
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_common import DATA_ROOT, REPO_ROOT, save_json  # noqa: E402

from training.train_generation1_foundation import (  # noqa: E402
    FoundationConfig, build_model)

PHASES = ["backbone_only", "recurrence_only", "belief_no_f", "belief_with_f"]
LOCAL_DIR = os.path.join(REPO_ROOT, "checkpoints")
ADV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "hf_rolling_adv")


def load_for(phase: str, path: str):
    cfg = FoundationConfig()
    model = build_model(cfg, phase)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    model.load_state_dict(sd)
    return model.to(device).eval(), ck


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> int:
    from evaluation.imagenet100_loader import make_imagenet100_loaders
    loaders = make_imagenet100_loaders(DATA_ROOT, batch_size=256, num_workers=4)
    xb, yb = next(iter(loaders["val"]))
    x, y = xb.to(device), yb.to(device)

    results = {}
    for era, base in (("pure_ce_cf6ce8a", LOCAL_DIR),
                      ("adversarial_fef50f3", ADV_DIR)):
        for ph in PHASES:
            p = os.path.join(base, f"foundation_{ph}_best.pth" if era.startswith("pure")
                             else f"foundation_{ph}_rolling.pth")
            if not os.path.exists(p):
                results[f"{era}/{ph}"] = {"missing": True}
                continue
            try:
                model, ck = load_for(ph, p)
                with torch.no_grad():
                    logits = model(x)
                    probs = F.softmax(logits.float(), dim=1)
                    ent = -(probs * probs.clamp_min(1e-12).log()).sum(1)
                    pred = logits.argmax(1)
                hist = torch.bincount(pred, minlength=100).float()
                results[f"{era}/{ph}"] = {
                    "ckpt_epoch": ck.get("epoch"), "code_commit": ck.get("code_commit"),
                    "logit_mean": round(logits.mean().item(), 4),
                    "logit_std": round(logits.std().item(), 4),
                    "logit_min": round(logits.min().item(), 3),
                    "logit_max": round(logits.max().item(), 3),
                    "softmax_entropy_mean": round(ent.mean().item(), 4),
                    "max_softmax_prob_mean": round(probs.max(1).values.mean().item(), 4),
                    "unique_predicted_classes": int(hist.gt(0).sum().item()),
                    "top1_acc_on_batch": round((pred.cpu() == yb).float().mean().item(), 4),
                    "pred_hist_top5_share": round(
                        float(hist.topk(5).values.sum().item()) / x.shape[0], 4),
                }
                r = results[f"{era}/{ph}"]
                print(f"  [{era}] {ph:15s} ep={r['ckpt_epoch']} "
                      f"acc={r['top1_acc_on_batch']:.4f} "
                      f"entropy={r['softmax_entropy_mean']:.3f} "
                      f"uniq={r['unique_predicted_classes']} "
                      f"top5share={r['pred_hist_top5_share']:.3f}")
            except Exception as e:  # noqa: BLE001 — report, never hide
                results[f"{era}/{ph}"] = {"error": str(e)[:200]}

    collapsed = [k for k, v in results.items()
                 if isinstance(v, dict) and v.get("unique_predicted_classes", 100) <= 5]
    out = {"section": "7_output_sanity", "batch_size": x.shape[0],
           "results": results,
           "collapsed_models": collapsed,
           "verdict": ("COLLAPSE DETECTED in: " + ", ".join(collapsed)
                       if collapsed else
                       "No hard collapse in any preserved checkpoint "
                       "(predictions spread over >5 classes)")}
    save_json("07_output_sanity.json", out)
    print(f"SECTION 7 VERDICT: {out['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
