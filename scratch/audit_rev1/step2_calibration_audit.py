#!/usr/bin/env python3
"""Step 2: Calibration, Uncertainty & Temperature Scaling Audit
=============================================================
Audit of probabilistic calibration and uncertainty metrics:
  - Recomputes true classification NLL: -(1/N) * sum log p(y_i | x_i) in fp32.
  - Documents contrast with the flawed metric: -log(max p).
  - Fits Temperature Scaling (T*) on designated Calibration Split (n=500, seed 101).
  - Evaluates on Held-Out Split (n=500, seed 42) under Clean and PGD-10 (eps=0.094).
  - Metrics: True NLL, Flawed NLL, ECE (10 bins), Brier score, AUROC error detection.
  - Compares Softmax Entropy vs Evidential Dirichlet Uncertainty U_T for gen1_core.

Output: report/audit_rev1/step2_calibration_audit.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Dict, List, Tuple, Any

import numpy as np
import scipy.stats as stats
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
from training.train_gen2_foundation import run_pgd_ce_eval

CALIB_SEED = 101
EVAL_SEED = 42
SAMPLE_SIZE = 500
BATCH_SIZE = 32
EPS_NORM = 0.094
N_BINS = 10


def get_split_loader(data_root: str, seed: int, sample_size: int, batch_size: int) -> DataLoader:
    tf = transforms.Compose([
        transforms.Resize((DEFAULT_FOVEA_SIZE, DEFAULT_FOVEA_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    ds = datasets.ImageFolder(os.path.join(data_root, "val"), transform=tf)
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(ds), generator=g)[:sample_size].tolist()
    return DataLoader(Subset(ds, indices), batch_size=batch_size, shuffle=False, num_workers=2)


def compute_true_nll(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return F.cross_entropy(logits.float(), targets).item()


def compute_flawed_cal2_nll(probs: torch.Tensor) -> float:
    # -log(max p) without ground-truth label
    return -torch.log(probs.max(dim=-1).values.clamp(min=1e-8)).mean().item()


def compute_brier_score(probs: torch.Tensor, targets: torch.Tensor) -> float:
    # Brier = (1/N) * sum_i sum_k (p_ik - y_ik)^2
    n_classes = probs.size(-1)
    one_hot = F.one_hot(targets, num_classes=n_classes).float()
    return torch.mean(torch.sum((probs - one_hot) ** 2, dim=-1)).item()


def compute_ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = N_BINS) -> float:
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(targets)
    ece = 0.0
    n = len(confidences)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        prop_in_bin = in_bin.float().mean().item()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_auroc(scores: np.ndarray, error_labels: np.ndarray) -> float:
    # error_labels: 1 if prediction is WRONG, 0 if CORRECT
    # scores: higher score indicates higher uncertainty / higher risk of error
    n_pos = np.sum(error_labels == 1)
    n_neg = np.sum(error_labels == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Mann-Whitney U
    ranks = stats.rankdata(scores)
    pos_ranks = np.sum(ranks[error_labels == 1])
    u_stat = pos_ranks - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))


class TemperatureScaler(nn.Module):
    """Fits a single positive scalar temperature T using L-BFGS to minimize NLL."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, targets: torch.Tensor, max_iter: int = 50) -> float:
        self.to(logits.device)
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), targets)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return float(self.temperature.item())


def forward_with_uncertainty(model: Gen2FoundationModel, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Runs model and captures final Dirichlet uncertainty if present."""
    if not model.carry_belief:
        return model(x), None

    base = model
    B = x.shape[0]
    device = x.device

    gist_tokens = base.gist_encoder(x)
    z_t = gist_tokens.mean(dim=1)
    dirichlet = base.evidential_head(z_t)
    current_gaze = torch.zeros(B, 2, device=device)
    prev_tokens = None
    pred_feat = None

    GAZE_SCHEDULE = [(0.0, 0.0), (0.3, 0.3), (-0.3, 0.3), (0.0, -0.3)]

    for t in range(base.num_glimpses):
        if base.use_ais_v2 and t > 0:
            belief_obj = SimpleNamespace(
                z=z_t, uncertainty=dirichlet.uncertainty, evidence=dirichlet.evidence
            )
            sel = base.gaze_policy.select_next_location(
                belief_obj,
                observed_tokens=prev_tokens.detach() if prev_tokens is not None else torch.zeros(B, 16, base.d_z, device=device),
                current_gaze=current_gaze,
                predicted_tokens=pred_feat,
                training=base.training,
            )
            current_gaze = sel.selected.clamp(-1.0, 1.0)
        elif t < len(GAZE_SCHEDULE):
            gx, gy = GAZE_SCHEDULE[t]
            current_gaze = torch.tensor([[gx, gy]], device=device, dtype=torch.float32).expand(B, -1)
        else:
            current_gaze = torch.zeros(B, 2, device=device)

        crop = foveal_sample(x, current_gaze, fovea_size=base.fovea_size)
        pooled_g, tokens_g = base.backbone(crop)
        prev_tokens = tokens_g

        if base.belief_dynamics:
            prec = base.precision(dirichlet).unsqueeze(-1)
            belief_obj = SimpleNamespace(
                z=z_t, uncertainty=dirichlet.uncertainty, evidence=dirichlet.evidence
            )
            pred_feat = base.predictor.predict_features(belief_obj, current_gaze)
            if base.arm == "v2" and base.update_net_v2 is not None:
                err_map = (pred_feat - tokens_g).norm(dim=-1).view(B, 4, 4)
                delta_z = base.update_net_v2(z_t, err_map, observed=pooled_g, U=dirichlet.uncertainty)
            else:
                err = pred_feat - tokens_g
                delta_z = base.update_net(z_t, err)
            z_t = z_t + prec * delta_z
        else:
            z_t = 0.5 * z_t + 0.5 * pooled_g

        dirichlet = base.evidential_head(z_t)

    feats = torch.cat([z_t, dirichlet.evidence], dim=-1)
    logits = base.cls_head(feats) + base.ev_readout(dirichlet.evidence)
    return logits, dirichlet.uncertainty


def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device, eps: float = 0.0, steps: int = 0
) -> Dict[str, torch.Tensor]:
    model.eval()
    all_logits = []
    all_targets = []
    all_u = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if eps > 0.0 and steps > 0:
            x = run_pgd_ce_eval(model, x, y, eps=eps, steps=steps)

        with torch.no_grad():
            if isinstance(model, Gen2FoundationModel) and model.carry_belief:
                logits, u = forward_with_uncertainty(model, x)
                all_u.append(u.cpu())
            else:
                logits = model(x)
        all_logits.append(logits.float().cpu())
        all_targets.append(y.cpu())

    res = {
        "logits": torch.cat(all_logits, dim=0),
        "targets": torch.cat(all_targets, dim=0),
    }
    if all_u:
        res["uncertainty"] = torch.cat(all_u, dim=0)
    return res


def evaluate_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 1.0,
    uncertainties: torch.Tensor | None = None,
) -> Dict[str, Any]:
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    preds = probs.argmax(dim=-1)
    correct = preds.eq(targets)
    error_labels = (~correct).numpy().astype(int)

    # fp32 entropy: compute with high precision, guard with clamp
    probs_clamped = probs.clamp(min=1e-12)
    entropy = -torch.sum(probs * torch.log(probs_clamped), dim=-1).numpy()

    acc = correct.float().mean().item()
    true_nll = compute_true_nll(scaled_logits, targets)
    flawed_nll = compute_flawed_cal2_nll(probs)
    brier = compute_brier_score(probs, targets)
    ece = compute_ece(probs, targets, N_BINS)
    auroc_entropy = compute_auroc(entropy, error_labels)

    metrics = {
        "accuracy": acc,
        "true_nll": true_nll,
        "flawed_cal2_nll": flawed_nll,
        "brier_score": brier,
        "ece": ece,
        "auroc_entropy": auroc_entropy,
        "mean_entropy": float(np.mean(entropy)),
        "std_entropy": float(np.std(entropy)),
    }

    if uncertainties is not None:
        u_np = uncertainties.numpy()
        metrics["mean_dirichlet_u"] = float(np.mean(u_np))
        metrics["std_dirichlet_u"] = float(np.std(u_np))
        metrics["auroc_dirichlet_u"] = compute_auroc(u_np, error_labels)
        # Correlation between entropy and Dirichlet uncertainty
        corr = float(np.corrcoef(entropy, u_np)[0, 1])
        metrics["entropy_u_correlation"] = corr

    return metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Step 2 Calibration Audit] Device: {device}")

    data_root = os.path.join(REPO_ROOT, "data", "imagenet100")
    calib_loader = get_split_loader(data_root, CALIB_SEED, SAMPLE_SIZE, BATCH_SIZE)
    eval_loader = get_split_loader(data_root, EVAL_SEED, SAMPLE_SIZE, BATCH_SIZE)
    print(f"[Step 2] Calibration split: n={SAMPLE_SIZE} (seed {CALIB_SEED})")
    print(f"[Step 2] Held-out eval split: n={SAMPLE_SIZE} (seed {EVAL_SEED})")

    checkpoints = [
        {
            "name": "recurrence_only_final",
            "path": "checkpoints/foundation_recurrence_only_final.pth",
            "phase": "recurrence_only",
            "arm": "v1",
        },
        {
            "name": "gen1_core_best_clean",
            "path": "checkpoints/foundation_gen1_core_best_clean.pth",
            "phase": "gen1_core",
            "arm": "v1",
        },
    ]

    all_results = {
        "calibration_seed": CALIB_SEED,
        "evaluation_seed": EVAL_SEED,
        "sample_size": SAMPLE_SIZE,
        "models": {},
    }

    for ckpt in checkpoints:
        name = ckpt["name"]
        path = os.path.join(REPO_ROOT, ckpt["path"])
        phase = ckpt["phase"]

        print(f"\n{'='*75}\n[Step 2] Auditing Checkpoint: {name}\n{'='*75}")
        model = Gen2FoundationModel(phase=phase, arm=ckpt["arm"]).to(device)
        saved = torch.load(path, map_location=device, weights_only=False)
        sd = saved["model"] if "model" in saved else saved
        model.load_state_dict(sd, strict=False)
        model.eval()

        # 1. Collect clean predictions on Calibration Split to fit Temperature
        print("  Collecting calibration split predictions (clean)...")
        calib_preds = collect_predictions(model, calib_loader, device, eps=0.0, steps=0)
        scaler = TemperatureScaler()
        t_star = scaler.fit(calib_preds["logits"].to(device), calib_preds["targets"].to(device))
        print(f"  Fitted Temperature T* = {t_star:.4f}")

        # 2. Collect held-out predictions for Clean and PGD-10
        print("  Collecting held-out split predictions (clean)...")
        eval_clean = collect_predictions(model, eval_loader, device, eps=0.0, steps=0)

        print("  Collecting held-out split predictions (PGD-10 @ 0.094)...")
        eval_pgd10 = collect_predictions(model, eval_loader, device, eps=EPS_NORM, steps=10)

        ckpt_results = {
            "temperature_t_star": t_star,
            "clean": {
                "uncalibrated": evaluate_metrics(
                    eval_clean["logits"], eval_clean["targets"],
                    temperature=1.0, uncertainties=eval_clean.get("uncertainty")
                ),
                "temperature_scaled": evaluate_metrics(
                    eval_clean["logits"], eval_clean["targets"],
                    temperature=t_star, uncertainties=eval_clean.get("uncertainty")
                ),
            },
            "pgd10": {
                "uncalibrated": evaluate_metrics(
                    eval_pgd10["logits"], eval_pgd10["targets"],
                    temperature=1.0, uncertainties=eval_pgd10.get("uncertainty")
                ),
                "temperature_scaled": evaluate_metrics(
                    eval_pgd10["logits"], eval_pgd10["targets"],
                    temperature=t_star, uncertainties=eval_pgd10.get("uncertainty")
                ),
            },
        }

        print("\n  Summary of Held-Out Metrics:")
        for cond in ["clean", "pgd10"]:
            uc = ckpt_results[cond]["uncalibrated"]
            ts = ckpt_results[cond]["temperature_scaled"]
            print(f"    [{cond.upper()}] Uncalibrated: Acc={uc['accuracy']*100:.2f}%, "
                  f"True NLL={uc['true_nll']:.4f}, Flawed NLL={uc['flawed_cal2_nll']:.4f}, "
                  f"ECE={uc['ece']:.4f}, AUROC(H)={uc['auroc_entropy']:.4f}")
            print(f"    [{cond.upper()}] Temp-Scaled : Acc={ts['accuracy']*100:.2f}%, "
                  f"True NLL={ts['true_nll']:.4f}, ECE={ts['ece']:.4f}, AUROC(H)={ts['auroc_entropy']:.4f}")
            if "auroc_dirichlet_u" in uc:
                print(f"    [{cond.upper()}] Dirichlet U: Mean={uc['mean_dirichlet_u']:.4f}, "
                      f"AUROC(U)={uc['auroc_dirichlet_u']:.4f}, Corr(H, U)={uc['entropy_u_correlation']:.4f}")

        all_results["models"][name] = ckpt_results
        del model
        torch.cuda.empty_cache()

    out_file = os.path.join(REPO_ROOT, "report", "audit_rev1", "step2_calibration_audit.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Step 2] Calibration audit complete. Saved to: {out_file}")


if __name__ == "__main__":
    main()
