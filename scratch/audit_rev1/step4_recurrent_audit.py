#!/usr/bin/env python3
"""Step 4: Recurrent Perceptual Dynamics & Belief Revision Audit
================================================================
Implements:
  1. G2-REV1 (Contradiction Injection):
     - Presents conflicting evidence sequences from two different classes (A and B):
         * [A, A, A, A] (Control A)
         * [B, B, B, B] (Control B)
         * [A, A, B, B] (Late Shift: 2 steps A, 2 steps B)
         * [B, B, A, A] (Early Shift: 2 steps B, 2 steps A)
         * [A, A, A, B] (Single Late Contradiction)
         * [B, A, A, A] (Single Early Contradiction)
     - Tests: Observation averaging vs Recency preference vs Order sensitivity.
  2. G2-PRB1 (Surprise & Selective Revision Probe):
     - Monitors step-by-step:
         * Latent state norm ||z_t||
         * Prediction error norm ||err_t|| = ||pred_t - tok_t||
         * Precision scalar prec_t
         * Dirichlet uncertainty U_t
         * Logit margin (Logit_A - Logit_B)
     - Evaluates whether revision tracks evidence quality (clean vs noisy injection).

Evaluated on:
  - recurrence_only_final (Phase 2 Control)
  - gen1_core_best_clean (Phase 6 Full Gen-2)

Output: report/audit_rev1/step4_recurrent_audit.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gen2_foundation.model import Gen2FoundationModel
from noesis_vision.models.foveation import DEFAULT_FOVEA_SIZE, foveal_sample
from training.curriculum_gen2 import IMAGENET_MEAN, IMAGENET_STD

GAZE_SCHEDULE = [
    (0.0, 0.0),
    (0.3, 0.3),
    (-0.3, 0.3),
    (0.0, -0.3),
]


def get_sample_pairs(data_root: str, n_pairs: int = 50, seed: int = 42):
    """Selects n_pairs of test samples from completely distinct classes."""
    tf = transforms.Compose([
        transforms.Resize((DEFAULT_FOVEA_SIZE, DEFAULT_FOVEA_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    ds = datasets.ImageFolder(os.path.join(data_root, "val"), transform=tf)
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(ds), generator=g).tolist()

    pairs = []
    seen_classes = set()
    idx_i = 0
    while len(pairs) < n_pairs and idx_i < len(indices) - 1:
        i1 = indices[idx_i]
        i2 = indices[idx_i + 1]
        x1, y1 = ds[i1]
        x2, y2 = ds[i2]
        idx_i += 2
        if y1 != y2:
            pairs.append(((x1, y1), (x2, y2)))

    return pairs


def run_sequence_recurrence_only(model: Gen2FoundationModel, image_seq: List[torch.Tensor]) -> Dict[str, Any]:
    """Runs recurrence_only model with a custom sequence of 4 glimpse images."""
    device = image_seq[0].device
    B = image_seq[0].shape[0]

    # Gist from first image in sequence
    gist_tokens = model.gist_encoder(image_seq[0])
    z_t = gist_tokens.mean(dim=1)

    step_z_norms = [float(z_t.norm(dim=-1).mean().item())]

    for t in range(4):
        x_t = image_seq[t]
        gx, gy = GAZE_SCHEDULE[t]
        gaze = torch.tensor([[gx, gy]], device=device, dtype=torch.float32).expand(B, -1)
        crop = foveal_sample(x_t, gaze, fovea_size=model.fovea_size)
        pooled_g, _ = model.backbone(crop)
        z_t = 0.5 * z_t + 0.5 * pooled_g
        step_z_norms.append(float(z_t.norm(dim=-1).mean().item()))

    logits = model.cls_head(z_t)
    return {
        "logits": logits,
        "step_z_norms": step_z_norms,
    }


def run_sequence_gen1_core(
    model: Gen2FoundationModel,
    image_seq: List[torch.Tensor],
    fixed_gaze: bool = True,
) -> Dict[str, Any]:
    """Runs gen1_core model with step-by-step diagnostic monitoring."""
    device = image_seq[0].device
    B = image_seq[0].shape[0]

    gist_tokens = model.gist_encoder(image_seq[0])
    z_t = gist_tokens.mean(dim=1)
    dirichlet = model.evidential_head(z_t)

    current_gaze = torch.zeros(B, 2, device=device)
    prev_tokens = None
    pred_feat = None

    history = {
        "step_z_norms": [float(z_t.norm(dim=-1).mean().item())],
        "step_err_norms": [],
        "step_precisions": [],
        "step_uncertainties": [float(dirichlet.uncertainty.mean().item())],
        "step_delta_z_norms": [],
    }

    for t in range(4):
        x_t = image_seq[t]
        if fixed_gaze or t == 0:
            gx, gy = GAZE_SCHEDULE[t]
            current_gaze = torch.tensor([[gx, gy]], device=device, dtype=torch.float32).expand(B, -1)
        else:
            belief_obj = SimpleNamespace(
                z=z_t, uncertainty=dirichlet.uncertainty, evidence=dirichlet.evidence
            )
            sel = model.gaze_policy.select_next_location(
                belief_obj,
                observed_tokens=prev_tokens.detach() if prev_tokens is not None else torch.zeros(B, 16, model.d_z, device=device),
                current_gaze=current_gaze,
                predicted_tokens=pred_feat,
                training=False,
            )
            current_gaze = sel.selected.clamp(-1.0, 1.0)

        crop = foveal_sample(x_t, current_gaze, fovea_size=model.fovea_size)
        pooled_g, tokens_g = model.backbone(crop)
        prev_tokens = tokens_g

        if model.belief_dynamics:
            prec = model.precision(dirichlet).unsqueeze(-1)
            belief_obj = SimpleNamespace(
                z=z_t, uncertainty=dirichlet.uncertainty, evidence=dirichlet.evidence
            )
            pred_feat = model.predictor.predict_features(belief_obj, current_gaze)
            err = pred_feat - tokens_g
            delta_z = model.update_net(z_t, err)

            history["step_err_norms"].append(float(err.norm(dim=-1).mean().item()))
            history["step_precisions"].append(float(prec.mean().item()))
            history["step_delta_z_norms"].append(float(delta_z.norm(dim=-1).mean().item()))

            z_t = z_t + prec * delta_z
        else:
            z_t = 0.5 * z_t + 0.5 * pooled_g

        dirichlet = model.evidential_head(z_t)
        history["step_z_norms"].append(float(z_t.norm(dim=-1).mean().item()))
        history["step_uncertainties"].append(float(dirichlet.uncertainty.mean().item()))

    feats = torch.cat([z_t, dirichlet.evidence], dim=-1)
    logits = model.cls_head(feats) + model.ev_readout(dirichlet.evidence)
    history["logits"] = logits
    return history


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Step 4 Recurrent Audit] Device: {device}")

    data_root = os.path.join(REPO_ROOT, "data", "imagenet100")
    pairs = get_sample_pairs(data_root, n_pairs=50, seed=42)
    print(f"[Step 4] Generated {len(pairs)} distinct (Class A, Class B) sample pairs.")

    models = {
        "recurrence_only_final": {
            "path": "checkpoints/foundation_recurrence_only_final.pth",
            "phase": "recurrence_only",
            "arm": "v1",
        },
        "gen1_core_best_clean": {
            "path": "checkpoints/foundation_gen1_core_best_clean.pth",
            "phase": "gen1_core",
            "arm": "v1",
        },
    }

    audit_results: Dict[str, Any] = {
        "n_pairs": len(pairs),
        "models": {},
    }

    # Sequences to evaluate per pair:
    # A = img_A, B = img_B, B_noisy = img_B + noise
    seq_patterns = [
        "AAAA",   # Homogeneous A
        "BBBB",   # Homogeneous B
        "AABB",   # Late shift: 2 steps A, 2 steps B
        "BBAA",   # Early shift: 2 steps B, 2 steps A
        "AAAB",   # Single late B
        "BAAA",   # Single early B
        "AAB_noisyB_noisy", # Late shift with low-quality/noisy evidence
    ]

    for model_name, info in models.items():
        print(f"\n{'='*75}\n[Step 4] Testing Model: {model_name}\n{'='*75}")
        model = Gen2FoundationModel(phase=info["phase"], arm=info["arm"]).to(device)
        saved = torch.load(info["path"], map_location=device, weights_only=False)
        sd = saved["model"] if "model" in saved else saved
        model.load_state_dict(sd, strict=False)
        model.eval()

        pattern_stats = {pat: {"margin_A_minus_B": [], "p_pred_A": [], "p_pred_B": [], "flips_to_B": []} for pat in seq_patterns}
        surprise_stats = {"transition_err": [], "non_transition_err": [], "transition_prec": [], "transition_U": []}

        for (x_a, y_a), (x_b, y_b) in pairs:
            xa = x_a.unsqueeze(0).to(device)
            xb = x_b.unsqueeze(0).to(device)
            # Noisy B: add gaussian noise to simulate degraded evidence quality
            xb_noisy = torch.clamp(xb + 0.5 * torch.randn_like(xb), -4.0, 4.0)

            seq_map = {
                "AAAA": [xa, xa, xa, xa],
                "BBBB": [xb, xb, xb, xb],
                "AABB": [xa, xa, xb, xb],
                "BBAA": [xb, xb, xa, xa],
                "AAAB": [xa, xa, xa, xb],
                "BAAA": [xb, xa, xa, xa],
                "AAB_noisyB_noisy": [xa, xa, xb_noisy, xb_noisy],
            }

            for pat, seq in seq_map.items():
                with torch.no_grad():
                    if info["phase"] == "recurrence_only":
                        res = run_sequence_recurrence_only(model, seq)
                    else:
                        res = run_sequence_gen1_core(model, seq, fixed_gaze=True)

                logits = res["logits"][0]
                p = F.softmax(logits, dim=-1)
                margin = float((logits[y_a] - logits[y_b]).item())
                prob_a = float(p[y_a].item())
                prob_b = float(p[y_b].item())
                pred_cls = int(logits.argmax().item())

                pattern_stats[pat]["margin_A_minus_B"].append(margin)
                pattern_stats[pat]["p_pred_A"].append(prob_a)
                pattern_stats[pat]["p_pred_B"].append(prob_b)
                pattern_stats[pat]["flips_to_B"].append(1 if pred_cls == y_b else 0)

                # Collect surprise stats on AABB transition at step t=2 for gen1_core
                if pat == "AABB" and "step_err_norms" in res and len(res["step_err_norms"]) >= 3:
                    # t=0, t=1 are A; t=2 is first observation of B -> transition!
                    err_t2 = res["step_err_norms"][2]
                    err_t1 = res["step_err_norms"][1]
                    surprise_stats["transition_err"].append(err_t2)
                    surprise_stats["non_transition_err"].append(err_t1)
                    surprise_stats["transition_prec"].append(res["step_precisions"][2])
                    surprise_stats["transition_U"].append(res["step_uncertainties"][2])

        model_summary = {}
        for pat in seq_patterns:
            model_summary[pat] = {
                "mean_margin_A_minus_B": float(np.mean(pattern_stats[pat]["margin_A_minus_B"])),
                "std_margin_A_minus_B": float(np.std(pattern_stats[pat]["margin_A_minus_B"])),
                "mean_prob_A": float(np.mean(pattern_stats[pat]["p_pred_A"])),
                "mean_prob_B": float(np.mean(pattern_stats[pat]["p_pred_B"])),
                "flip_to_B_rate": float(np.mean(pattern_stats[pat]["flips_to_B"])),
            }
            print(f"  Pattern [{pat:<16}]: Margin(A-B)={model_summary[pat]['mean_margin_A_minus_B']:+6.2f} "
                  f"| P(A)={model_summary[pat]['mean_prob_A']*100:5.1f}% | P(B)={model_summary[pat]['mean_prob_B']*100:5.1f}% "
                  f"| Flip to B={model_summary[pat]['flip_to_B_rate']*100:5.1f}%")

        if surprise_stats["transition_err"]:
            trans_mean = float(np.mean(surprise_stats["transition_err"]))
            non_trans_mean = float(np.mean(surprise_stats["non_transition_err"]))
            prec_mean = float(np.mean(surprise_stats["transition_prec"]))
            u_mean = float(np.mean(surprise_stats["transition_U"]))
            model_summary["surprise_probe"] = {
                "transition_error_norm": trans_mean,
                "homogeneous_error_norm": non_trans_mean,
                "error_ratio_transition_vs_homo": trans_mean / max(non_trans_mean, 1e-6),
                "precision_at_transition": prec_mean,
                "uncertainty_at_transition": u_mean,
            }
            print(f"\n  Surprise Probe at Transition (t=2, A -> B):")
            print(f"    Prediction Error Norm: Transition={trans_mean:.4f} vs Homogeneous={non_trans_mean:.4f} "
                  f"(Ratio = {trans_mean/non_trans_mean:.2f}x)")
            print(f"    Precision at Transition: {prec_mean:.4f} | Uncertainty U: {u_mean:.4f}")

        audit_results["models"][model_name] = model_summary
        del model
        torch.cuda.empty_cache()

    out_file = os.path.join(REPO_ROOT, "report", "audit_rev1", "step4_recurrent_audit.json")
    with open(out_file, "w") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\n[Step 4] Recurrent dynamics audit complete. Saved to: {out_file}")


if __name__ == "__main__":
    main()
