#!/usr/bin/env python3
"""Step 1: G2-D9 Adversarial Attack Audit & Robustness Verification
===================================================================
Executes exhaustive attack audit on available Gen-2 final checkpoints:
  1. foundation_g2_gist_only_final.zip (Phase 1 Final, Ep 30)
  2. foundation_recurrence_only_final.pth (Phase 2 Final, Ep 15)
  3. foundation_gen1_core_best_clean.pth (Phase 6 Final, Ep 14)

Evaluates:
  - In-house PGD reproduction (PGD-10, PGD-50 at alpha = eps/steps)
  - Factor ablation:
      * FP32 vs Autocast during attack
      * Cross-Entropy vs TRADES KL objective
      * Step sizes: alpha in {eps/50, 2*eps/50, 2.5*eps/50}
      * Random start enabled vs disabled
      * Zero / non-finite gradient checks on initial attack step
  - Maintained library attacks (torchattacks):
      * APGD-CE-100
      * APGD-DLR-100 (targeted / margin-based)
      * MI-FGSM-100 (decay=0.9)
  - Large-epsilon sanity checks at eps_pixel in {16/255, 32/255}
  - Per-sample binary outcomes for paired McNemar and paired difference statistics.

Output: report/audit_rev1/step1_attack_audit.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Tuple, Any

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchattacks

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from gen2_foundation.model import Gen2FoundationModel
from noesis_vision.models.foveation import DEFAULT_FOVEA_SIZE
from training.curriculum_gen2 import IMAGENET_MEAN, IMAGENET_STD, MEAN_STD

SEED = 42
SAMPLE_SIZE = 500
BATCH_SIZE = 32
EPS_NORM = 0.094
EPS_PIXEL = EPS_NORM * MEAN_STD  # ~0.0212 (~5.4/255)
CLAMP_MIN = -4.0
CLAMP_MAX = 4.0

CHECKPOINTS = [
    {
        "name": "g2_gist_only_final",
        "path": "checkpoints/foundation_g2_gist_only_final.zip",
        "phase": "g2_gist_only",
        "arm": "v1",
        "role": "phase1_gist_only",
    },
    {
        "name": "recurrence_only_final",
        "path": "checkpoints/foundation_recurrence_only_final.pth",
        "phase": "recurrence_only",
        "arm": "v1",
        "role": "phase2_control",
    },
    {
        "name": "gen1_core_best_clean",
        "path": "checkpoints/foundation_gen1_core_best_clean.pth",
        "phase": "gen1_core",
        "arm": "v1",
        "role": "phase6_full_gen2",
    },
]


def get_eval_loaders(data_root: str, seed: int, sample_size: int, batch_size: int):
    # Normalized loader (for in-house attacks)
    tf_norm = transforms.Compose([
        transforms.Resize((DEFAULT_FOVEA_SIZE, DEFAULT_FOVEA_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    # Unnormalized loader in [0, 1] (for torchattacks library)
    tf_unnorm = transforms.Compose([
        transforms.Resize((DEFAULT_FOVEA_SIZE, DEFAULT_FOVEA_SIZE)),
        transforms.ToTensor(),
    ])

    val_dir = os.path.join(data_root, "val")
    ds_norm = datasets.ImageFolder(val_dir, transform=tf_norm)
    ds_unnorm = datasets.ImageFolder(val_dir, transform=tf_unnorm)

    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(ds_norm), generator=g)[:sample_size].tolist()

    loader_norm = DataLoader(Subset(ds_norm, indices), batch_size=batch_size, shuffle=False, num_workers=2)
    loader_unnorm = DataLoader(Subset(ds_unnorm, indices), batch_size=batch_size, shuffle=False, num_workers=2)
    return loader_norm, loader_unnorm, indices


# --- In-House Parametric PGD Attack Implementation ---
def run_inhouse_pgd(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = EPS_NORM,
    steps: int = 50,
    alpha: float = None,
    random_start: bool = True,
    loss_fn: str = "ce",  # "ce" | "trades_kl"
    use_autocast: bool = False,
    device: torch.device = torch.device("cuda"),
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if alpha is None:
        alpha = eps / max(steps, 1)

    diag = {"initial_grad_zeros": 0, "initial_grad_nan_inf": 0}

    if random_start:
        x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, CLAMP_MIN, CLAMP_MAX)
    else:
        x_adv = x.clone().detach()

    # Pre-compute clean probabilities for TRADES KL if needed
    probs_clean = None
    if loss_fn == "trades_kl":
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_autocast):
                probs_clean = F.softmax(model(x).float(), dim=-1)

    for step_i in range(steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            with torch.amp.autocast("cuda", enabled=use_autocast):
                logits = model(x_adv)
                if loss_fn == "ce":
                    loss = F.cross_entropy(logits.float(), y)
                elif loss_fn == "trades_kl":
                    loss = F.kl_div(
                        F.log_softmax(logits.float(), dim=-1),
                        probs_clean,
                        reduction="batchmean",
                    )
                else:
                    raise ValueError(f"Unknown loss_fn: {loss_fn}")

        grad = torch.autograd.grad(loss, x_adv, create_graph=False)[0]

        if step_i == 0:
            diag["initial_grad_zeros"] = int((grad == 0).sum().item())
            diag["initial_grad_nan_inf"] = int((~torch.isfinite(grad)).sum().item())

        with torch.no_grad():
            x_adv = x_adv.detach() + alpha * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
            x_adv = x_adv.clamp(CLAMP_MIN, CLAMP_MAX)

    return x_adv.detach(), diag


class NormalizedModelWrapper(nn.Module):
    """Wraps model taking [0, 1] inputs and normalizes with ImageNet mean/std."""
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.mean) / self.std
        return self.base_model(x_norm)


def evaluate_batch_predictions(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Tuple[int, List[int]]:
    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(-1)
        correct_mask = (preds == y).cpu().numpy().astype(int).tolist()
        correct_count = sum(correct_mask)
    return correct_count, correct_mask


def paired_mcnemar_test(outcomes_a: List[int], outcomes_b: List[int]) -> Dict[str, float]:
    """Computes McNemar test between two binary outcome vectors on identical samples."""
    a = np.array(outcomes_a)
    b = np.array(outcomes_b)
    # b00: both wrong, b01: a wrong, b right, b10: a right, b wrong, b11: both right
    b01 = np.sum((a == 0) & (b == 1))
    b10 = np.sum((a == 1) & (b == 0))
    diff = float(np.mean(a) - np.mean(b))
    n = len(a)
    # Standard error of difference for paired proportions
    se_diff = float(np.sqrt((b01 + b10) - (b10 - b01)**2 / n) / n) if n > 0 else 0.0

    # McNemar with continuity correction
    if (b01 + b10) > 0:
        stat = float((abs(b10 - b01) - 1.0)**2 / (b10 + b01))
        p_val = float(stats.chi2.sf(stat, df=1))
    else:
        stat = 0.0
        p_val = 1.0

    return {
        "diff": diff,
        "se_diff": se_diff,
        "mcnemar_stat": stat,
        "p_value": p_val,
        "n_a_only": int(b10),
        "n_b_only": int(b01),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[G2-D9 Audit] Running on device: {device}")

    out_dir = os.path.join(REPO_ROOT, "report", "audit_rev1")
    os.makedirs(out_dir, exist_ok=True)

    data_root = os.path.join(REPO_ROOT, "data", "imagenet100")
    loader_norm, loader_unnorm, sample_indices = get_eval_loaders(
        data_root, SEED, SAMPLE_SIZE, BATCH_SIZE
    )
    print(f"[G2-D9 Audit] Fixed sample size: {SAMPLE_SIZE} images (seed {SEED})")

    audit_results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "torch_version": torch.__version__,
        "torchattacks_version": getattr(torchattacks, "__version__", "unknown"),
        "seed": SEED,
        "sample_size": SAMPLE_SIZE,
        "sample_indices": sample_indices[:20],  # first 20 for provenance
        "eps_norm": EPS_NORM,
        "eps_pixel": EPS_PIXEL,
        "checkpoints": {},
    }

    models_dict = {}

    for ckpt in CHECKPOINTS:
        name = ckpt["name"]
        path = os.path.join(REPO_ROOT, ckpt["path"])
        phase = ckpt["phase"]
        arm = ckpt["arm"]

        if not os.path.exists(path):
            print(f"[G2-D9 Audit] Checkpoint {path} not found. Skipping.")
            audit_results["checkpoints"][name] = {"error": "file_not_found"}
            continue

        print(f"\n{'='*75}\n[G2-D9 Audit] Evaluating: {name} (phase={phase})\n{'='*75}")
        model = Gen2FoundationModel(phase=phase, arm=arm).to(device)
        saved = torch.load(path, map_location=device, weights_only=False)
        sd = saved["model"] if "model" in saved else saved
        missing, unexpected = model.load_state_dict(sd, strict=False)
        model.eval()

        models_dict[name] = model

        ckpt_res: Dict[str, Any] = {
            "path": ckpt["path"],
            "phase": phase,
            "epoch": saved.get("epoch") if isinstance(saved, dict) else None,
            "best_clean_acc": saved.get("best_clean_acc") if isinstance(saved, dict) else None,
            "best_rob_acc": saved.get("best_rob_acc") if isinstance(saved, dict) else None,
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "outcomes": {},
            "accuracies": {},
            "runtimes": {},
            "diagnostics": {},
        }

        # -------------------------------------------------------------
        # 1. Clean Baseline
        # -------------------------------------------------------------
        t0 = time.time()
        clean_correct = 0
        clean_mask = []
        for x, y in loader_norm:
            x, y = x.to(device), y.to(device)
            c, m = evaluate_batch_predictions(model, x, y)
            clean_correct += c
            clean_mask.extend(m)
        clean_acc = clean_correct / SAMPLE_SIZE
        ckpt_res["accuracies"]["clean"] = clean_acc
        ckpt_res["outcomes"]["clean"] = clean_mask
        ckpt_res["runtimes"]["clean"] = time.time() - t0
        print(f"  Clean Accuracy: {clean_acc*100:.2f}% ({ckpt_res['runtimes']['clean']:.1f}s)")

        # -------------------------------------------------------------
        # 2. Reproduction of In-House Attacks (PGD-10, PGD-50)
        # -------------------------------------------------------------
        for steps in [10, 50]:
            t0 = time.time()
            correct = 0
            mask = []
            for x, y in loader_norm:
                x, y = x.to(device), y.to(device)
                x_adv, diag = run_inhouse_pgd(
                    model, x, y, eps=EPS_NORM, steps=steps, alpha=EPS_NORM/steps,
                    random_start=True, loss_fn="ce", use_autocast=False, device=device
                )
                c, m = evaluate_batch_predictions(model, x_adv, y)
                correct += c
                mask.extend(m)
            acc = correct / SAMPLE_SIZE
            key = f"inhouse_pgd_{steps}"
            ckpt_res["accuracies"][key] = acc
            ckpt_res["outcomes"][key] = mask
            ckpt_res["runtimes"][key] = time.time() - t0
            print(f"  In-House PGD-{steps} (alpha=eps/{steps}): {acc*100:.2f}% ({time.time()-t0:.1f}s)")

        # -------------------------------------------------------------
        # 3. Factor Ablation on PGD-50
        # -------------------------------------------------------------
        ablation_configs = [
            ("pgd50_alpha_2x", {"steps": 50, "alpha": (2.0 * EPS_NORM) / 50.0, "random_start": True, "loss_fn": "ce", "use_autocast": False}),
            ("pgd50_alpha_2_5x", {"steps": 50, "alpha": (2.5 * EPS_NORM) / 50.0, "random_start": True, "loss_fn": "ce", "use_autocast": False}),
            ("pgd50_no_random_start", {"steps": 50, "alpha": EPS_NORM / 50.0, "random_start": False, "loss_fn": "ce", "use_autocast": False}),
            ("pgd50_trades_kl", {"steps": 50, "alpha": EPS_NORM / 50.0, "random_start": True, "loss_fn": "trades_kl", "use_autocast": False}),
            ("pgd50_autocast", {"steps": 50, "alpha": EPS_NORM / 50.0, "random_start": True, "loss_fn": "ce", "use_autocast": True}),
        ]

        total_zeros = 0
        total_nan_infs = 0

        for cond_name, cfg in ablation_configs:
            t0 = time.time()
            correct = 0
            mask = []
            for x, y in loader_norm:
                x, y = x.to(device), y.to(device)
                x_adv, diag = run_inhouse_pgd(
                    model, x, y, eps=EPS_NORM, steps=cfg["steps"], alpha=cfg["alpha"],
                    random_start=cfg["random_start"], loss_fn=cfg["loss_fn"],
                    use_autocast=cfg["use_autocast"], device=device
                )
                total_zeros += diag["initial_grad_zeros"]
                total_nan_infs += diag["initial_grad_nan_inf"]
                c, m = evaluate_batch_predictions(model, x_adv, y)
                correct += c
                mask.extend(m)
            acc = correct / SAMPLE_SIZE
            ckpt_res["accuracies"][cond_name] = acc
            ckpt_res["outcomes"][cond_name] = mask
            ckpt_res["runtimes"][cond_name] = time.time() - t0
            print(f"  Ablation {cond_name}: {acc*100:.2f}% ({time.time()-t0:.1f}s)")

        ckpt_res["diagnostics"]["initial_grad_zeros"] = total_zeros
        ckpt_res["diagnostics"]["initial_grad_nan_infs"] = total_nan_infs
        print(f"  Gradient diagnostics: zeros={total_zeros}, NaN/inf={total_nan_infs}")

        # -------------------------------------------------------------
        # 4. Maintained Library Attacks (torchattacks on wrapped model)
        # -------------------------------------------------------------
        wrapped_model = NormalizedModelWrapper(model).to(device)
        wrapped_model.eval()

        lib_attacks = [
            ("lib_apgd_ce_100", torchattacks.APGD(wrapped_model, eps=EPS_PIXEL, steps=100, loss="ce")),
            ("lib_apgd_dlr_100", torchattacks.APGD(wrapped_model, eps=EPS_PIXEL, steps=100, loss="dlr")),
            ("lib_mifgsm_100", torchattacks.MIFGSM(wrapped_model, eps=EPS_PIXEL, steps=100, decay=0.9)),
        ]

        for atk_name, atk_obj in lib_attacks:
            atk_obj.set_device(device)
            t0 = time.time()
            correct = 0
            mask = []
            for x_unnorm, y in loader_unnorm:
                x_unnorm, y = x_unnorm.to(device), y.to(device)
                x_adv_unnorm = atk_obj(x_unnorm, y)
                with torch.no_grad():
                    logits = wrapped_model(x_adv_unnorm)
                    preds = logits.argmax(-1)
                    c_mask = (preds == y).cpu().numpy().astype(int).tolist()
                    correct += sum(c_mask)
                    mask.extend(c_mask)
            acc = correct / SAMPLE_SIZE
            ckpt_res["accuracies"][atk_name] = acc
            ckpt_res["outcomes"][atk_name] = mask
            ckpt_res["runtimes"][atk_name] = time.time() - t0
            print(f"  Library {atk_name}: {acc*100:.2f}% ({time.time()-t0:.1f}s)")

        # -------------------------------------------------------------
        # 5. Large-Epsilon Sanity Checks (16/255 and 32/255)
        # -------------------------------------------------------------
        for large_eps_num in [16, 32]:
            large_eps = large_eps_num / 255.0
            atk_large = torchattacks.APGD(wrapped_model, eps=large_eps, steps=50, loss="ce")
            atk_large.set_device(device)
            t0 = time.time()
            correct = 0
            mask = []
            for x_unnorm, y in loader_unnorm:
                x_unnorm, y = x_unnorm.to(device), y.to(device)
                x_adv_unnorm = atk_large(x_unnorm, y)
                with torch.no_grad():
                    logits = wrapped_model(x_adv_unnorm)
                    preds = logits.argmax(-1)
                    c_mask = (preds == y).cpu().numpy().astype(int).tolist()
                    correct += sum(c_mask)
                    mask.extend(c_mask)
            acc = correct / SAMPLE_SIZE
            key = f"sanity_eps_{large_eps_num}_255"
            ckpt_res["accuracies"][key] = acc
            ckpt_res["outcomes"][key] = mask
            ckpt_res["runtimes"][key] = time.time() - t0
            print(f"  Sanity APGD @ eps={large_eps_num}/255: {acc*100:.2f}% ({time.time()-t0:.1f}s)")

        audit_results["checkpoints"][name] = ckpt_res

    # -----------------------------------------------------------------
    # 6. Paired Statistical Comparisons between Checkpoints
    # -----------------------------------------------------------------
    print("\n" + "=" * 75 + "\nPAIRED STATISTICAL COMPARISONS (McNemar Test)\n" + "=" * 75)
    paired_comp = {}
    ref_name = "recurrence_only_final"
    comp_name = "gen1_core_best_clean"

    if ref_name in audit_results["checkpoints"] and comp_name in audit_results["checkpoints"]:
        ref_out = audit_results["checkpoints"][ref_name]["outcomes"]
        cmp_out = audit_results["checkpoints"][comp_name]["outcomes"]
        for test_key in ref_out:
            if test_key in cmp_out:
                p_res = paired_mcnemar_test(ref_out[test_key], cmp_out[test_key])
                paired_comp[test_key] = p_res
                print(f"  {test_key:<25} | diff (rec - gen1): {p_res['diff']*100:+5.2f}% "
                      f"(SE={p_res['se_diff']*100:4.2f}%) | p={p_res['p_value']:.4e} "
                      f"(rec_only={p_res['n_a_only']}, gen1_only={p_res['n_b_only']})")

    audit_results["paired_comparisons"] = paired_comp

    out_file = os.path.join(out_dir, "step1_attack_audit.json")
    with open(out_file, "w") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\n[G2-D9 Audit] Results saved successfully to {out_file}")


if __name__ == "__main__":
    main()
