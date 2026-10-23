#!/usr/bin/env python3
"""
Kaggle Notebook 3 — Gen-2 Evidence Harvest (~1-2 GPU-h on T4/P100)
===================================================================

OBJECTIVE:
----------
Comprehensive forensic evidence harvest across all Gen-2 and control models:
1. Per-sample logits, softmax confidences, and correctness on canonical 1,000-image slice:
   - Clean, PGD-50, APGD-CE, APGD-DLR, and bitwise Worst-Case.
2. Evidential uncertainty and belief dynamics ($U_t$, $\alpha$, $b_{t,k}$) across steps:
   - Harvested from evidential head and predictor in recurrent/belief arms.
3. Per-step internal representations ($z_t$) and step-by-step readout logits:
   - Tracks whether multi-glimpse belief integration genuinely updates beliefs.
4. Representation similarity (Centered Kernel Alignment - CKA):
   - Trunk layer activations (blocks 4, 8, 12, pooled) to compare representational geometry.
5. Per-iteration attack loss trajectories:
   - 50-step PGD loss curves across clean-correct vs robust-failed samples.
6. Fixed visual exhibits:
   - 64 paired (clean, adv, ground-truth, predicted) samples saved for publication figures.
7. Attention maps:
   - Final block multi-head attention to CLS token under clean vs adversarial attack.

TARGET CHECKPOINTS:
-------------------
  - ff_112          (112 px, 64 tokens, seed 17, final)
  - recurrence_only (56 px, 16 tokens, T=4, final)
  - ff_56           (56 px, 16 tokens, final)
  - g2_gist_only_cont (C1) (56 px, 16 tokens, final)
  - g2_gist_only    (56 px, 16 tokens, final)
  - gen1_core       (56 px, 16 tokens, best clean)
  - c3_refine_single (56 px, depth control, if available)

USAGE ON KAGGLE:
----------------
1. Accelerator: GPU T4 or P100 (1x). Internet: ON.
2. Add-ons > Secrets > Add 'HF_TOKEN' with write permissions.
3. Paste and run. Output uploaded to HuggingFace dataset under 'evidence_harvest/'.
"""

import os
import sys
import subprocess
import shutil
import time
import datetime
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# ── 1. Secrets & GPU Check ───────────────────────────────────────────────────
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
        os.environ["HF_TOKEN"] = hf_token
    except Exception:
        pass

if not hf_token:
    raise RuntimeError(
        "HF_TOKEN not found! Add your HuggingFace write token in Kaggle: "
        "Add-ons > Secrets > label 'HF_TOKEN'."
    )
print(f"HF_TOKEN authenticated (starts with: {hf_token[:4]}...)")

import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Accelerator: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
else:
    print("WARNING: No GPU detected! Running on CPU.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Install requirements
for pkg in ["huggingface_hub", "torchattacks", "scipy"]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.run(f"pip install -q {pkg}", shell=True, check=True)

import torchattacks
from scipy.stats import chi2
import numpy as np

# ── 2. Environment Setup ─────────────────────────────────────────────────────
REPO_NAME = "Adversarial-Cognitive-Model"
WORK_DIR  = f"/kaggle/working/{REPO_NAME}"
BRANCH    = "feature/gen2-rebuild"

if os.path.exists("/kaggle/working"):
    os.chdir("/kaggle/working")
    if not os.path.exists(WORK_DIR):
        print(f"Cloning FerrariKazu/{REPO_NAME}...")
        subprocess.run(f"git clone https://github.com/FerrariKazu/{REPO_NAME}.git", shell=True, check=True)
    os.chdir(WORK_DIR)
    sys.path.insert(0, WORK_DIR)
    os.environ["PYTHONPATH"] = f"{WORK_DIR}:{os.environ.get('PYTHONPATH', '')}"

    print(f"Checking out branch '{BRANCH}'...")
    subprocess.run("git fetch origin", shell=True, check=True)
    subprocess.run(f"git checkout -B {BRANCH} origin/{BRANCH}", shell=True, check=True)
    subprocess.run(f"git reset --hard origin/{BRANCH}", shell=True, check=True)
else:
    print(f"Running locally in {os.getcwd()}")

for mod in list(sys.modules.keys()):
    if any(mod.startswith(p) for p in ["training", "gen2_foundation", "noesis_vision", "evaluation"]):
        del sys.modules[mod]

from gen2_foundation.model import Gen2FoundationModel
from noesis_vision.models.foveation import foveal_sample
from training.curriculum_gen2 import IMAGENET_MEAN, IMAGENET_STD
from evaluation.harness import (
    get_canonical_slice,
    get_canonical_val_transform,
    verify_checkpoint_gate,
    NormalizedModel,
    CANONICAL_TORCH_SLICE_SHA256,
    DEFAULT_EPS_PIXEL,
)
from torchvision import datasets
from torch.utils.data import DataLoader, Subset

# ── 3. Checkpoints & ImageNet-100 Restore ─────────────────────────────────────
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("report/evidence_harvest", exist_ok=True)

from huggingface_hub import HfApi, hf_hub_download, list_repo_files

HF_REPO = "FerrariKazu/rhan-nxa-g2-checkpoints"
api = HfApi(token=hf_token)

CHECKPOINTS = [
    "foundation_ff_112_final.pth",
    "foundation_recurrence_only_final.pth",
    "foundation_ff_56_final.pth",
    "foundation_g2_gist_only_cont_final.pth",
    "foundation_g2_gist_only_final.pth",
    "foundation_g2_gist_only_final.zip",
    "foundation_gen1_core_best_clean.pth",
    "foundation_c3_refine_single_final.pth",
]

for ckpt in CHECKPOINTS:
    local_p = os.path.join("checkpoints", ckpt)
    if not os.path.exists(local_p):
        try:
            print(f"Downloading {ckpt} from {HF_REPO}...")
            p = hf_hub_download(repo_id=HF_REPO, filename=ckpt, repo_type="dataset", token=hf_token)
            shutil.copyfile(p, local_p)
        except Exception as e:
            # Checkpoint may not exist yet (e.g. C3 if not uploaded)
            pass

DATA_ROOT = os.environ.get(
    "J2_DATA_ROOT",
    "/kaggle/tmp/imagenet100" if os.path.exists("/kaggle/tmp") else
    "/kaggle/working/data/imagenet100" if os.path.exists("/kaggle/working") else
    "data/imagenet100"
)

def _is_dataset_ready(root):
    td, vd = os.path.join(root, "train"), os.path.join(root, "val")
    return (os.path.isdir(td) and os.path.isdir(vd) and
            len(os.listdir(td)) == 100 and len(os.listdir(vd)) == 100)

if not _is_dataset_ready(DATA_ROOT):
    print(f"Bootstrapping ImageNet-100 to {DATA_ROOT} ...")
    os.makedirs(DATA_ROOT, exist_ok=True)
    subprocess.run(f"python3 scripts/prepare_imagenet100.py --root {DATA_ROOT}", shell=True, check=True)

val_root = os.path.join(DATA_ROOT, "val")
print(f"Validation dataset confirmed at: {val_root}")

# ── 4. Canonical Slice Loader ────────────────────────────────────────────────
slice_indices, slice_hash = get_canonical_slice(val_root, n=1000, seed=42, assert_hash=True)
print(f"✓ Canonical Slice Asserted: {len(slice_indices)} samples | SHA-256: {slice_hash}")

def build_raw_slice_loader(img_size, batch_size=32):
    tf = get_canonical_val_transform(img_size, normalize=False)
    ds = datasets.ImageFolder(val_root, transform=tf)
    sub = Subset(ds, slice_indices)
    return DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# ── 5. CKA Activation Hook Helper ────────────────────────────────────────────
class ActivationHarvester:
    def __init__(self, model):
        self.activations = {}
        self.hooks = []
        base = model.base_model if hasattr(model, "base_model") else model
        if hasattr(base, "backbone") and hasattr(base.backbone, "blocks"):
            # Register hooks on trunk blocks 4, 8, 12
            target_blocks = [3, 7, 11]
            for idx in target_blocks:
                if idx < len(base.backbone.blocks):
                    hook = base.backbone.blocks[idx].register_forward_hook(
                        lambda m, inp, out, name=f"block_{idx+1}": self._save_act(name, out)
                    )
                    self.hooks.append(hook)

    def _save_act(self, name, tensor):
        # Pool token dimension: mean over patch tokens
        self.activations[name] = tensor[:, 0].detach().cpu()  # CLS token

    def clear(self):
        self.activations.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()

def compute_linear_cka(x1: np.ndarray, x2: np.ndarray) -> float:
    """Computes Linear Centered Kernel Alignment between two feature matrices."""
    x1 = x1 - np.mean(x1, axis=0, keepdims=True)
    x2 = x2 - np.mean(x2, axis=0, keepdims=True)
    k1 = x1 @ x1.T
    k2 = x2 @ x2.T
    tr_k1_k2 = np.trace(k1 @ k2)
    tr_k1_k1 = np.trace(k1 @ k1)
    tr_k2_k2 = np.trace(k2 @ k2)
    denom = np.sqrt(tr_k1_k1 * tr_k2_k2)
    return float(tr_k1_k2 / max(denom, 1e-12))

# ── 6. PGD Helpers ────────────────────────────────────────────────────────────
def run_pgd_pixel_space(model, x, y, eps=DEFAULT_EPS_PIXEL, steps=50):
    """Standard PGD-n in pixel space (L-inf). Returns adversarial example only."""
    alpha = (2.5 * eps) / steps
    x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0.0, 1.0)
    x_adv = x_adv.detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = (x_adv + alpha * grad.sign()).clamp(x - eps, x + eps).clamp(0.0, 1.0)
    return x_adv.detach()

def run_pgd_with_loss_trajectory(model, x, y, eps=DEFAULT_EPS_PIXEL, steps=50):
    alpha = (2.5 * eps) / steps
    x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0.0, 1.0)
    loss_curve = []

    for step in range(steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y, reduction="none")
        grad = torch.autograd.grad(loss.sum(), x_adv)[0]
        loss_curve.append(float(loss.mean().item()))
        with torch.no_grad():
            x_adv = (x_adv.detach() + alpha * grad.sign()).clamp(x - eps, x + eps).clamp(0.0, 1.0)

    with torch.no_grad():
        final_loss = F.cross_entropy(model(x_adv), y, reduction="none").mean().item()
        loss_curve.append(float(final_loss))

    return x_adv.detach(), loss_curve

# ── 7. Evidence Harvest Execution Matrix ─────────────────────────────────────
TARGET_MODELS = [
    ("ff_112_s17",         "ff_112",          112, "checkpoints/foundation_ff_112_final.pth"),
    ("recurrence_only",    "recurrence_only", 56,  "checkpoints/foundation_recurrence_only_final.pth"),
    ("ff_56",              "ff_56",           56,  "checkpoints/foundation_ff_56_final.pth"),
    ("g2_gist_only_cont",  "g2_gist_only",    56,  "checkpoints/foundation_g2_gist_only_cont_final.pth"),
    ("g2_gist_only",       "g2_gist_only",    56,  "checkpoints/foundation_g2_gist_only_final.pth"),
    ("gen1_core",          "gen1_core",       56,  "checkpoints/foundation_gen1_core_best_clean.pth"),
    ("c3_refine_single",   "c3_refine_single",56,  "checkpoints/foundation_c3_refine_single_final.pth"),
]

harvest_results: Dict[str, Any] = {}
harvest_arrays: Dict[str, Any] = {}
fixed_images_bundle: Dict[str, Any] = {}
feature_activations: Dict[str, Dict[str, np.ndarray]] = {}

print("\n" + "=" * 80)
print("STARTING FORENSIC EVIDENCE HARVEST")
print("=" * 80)

for label, phase, img_size, ckpt in TARGET_MODELS:
    if not os.path.exists(ckpt):
        # Fallback to .zip if exists
        zip_candidate = ckpt.replace(".pth", ".zip")
        if os.path.exists(zip_candidate):
            ckpt = zip_candidate
        else:
            print(f"Skipping {label}: {ckpt} not found.")
            continue

    print(f"\n{'#'*80}\nHARVESTING: {label} ({img_size}px, ckpt={os.path.basename(ckpt)})\n{'#'*80}")

    base = Gen2FoundationModel(phase=phase, arm="v1", fovea_size=img_size).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    base.load_state_dict(sd.get("model", sd), strict=False)
    base.eval()

    wrapped = NormalizedModel(base).to(device)
    wrapped.eval()

    loader = build_raw_slice_loader(img_size, batch_size=32)

    # Attack instances
    apgd_ce  = torchattacks.APGD(wrapped, norm="Linf", eps=DEFAULT_EPS_PIXEL, steps=100, n_restarts=1, loss="ce", verbose=False)
    apgd_dlr = torchattacks.APGD(wrapped, norm="Linf", eps=DEFAULT_EPS_PIXEL, steps=100, n_restarts=1, loss="dlr", verbose=False)

    harvester = ActivationHarvester(wrapped)

    clean_logits_list = []
    clean_conf_list = []
    preds_clean = []
    labels_all = []
    acts_b4, acts_b8, acts_b12 = [], [], []

    print("  Collecting clean logits, confidences, and trunk activations...", end="", flush=True)
    t0 = time.time()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            harvester.clear()
            logits = wrapped(x)
            probs = F.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)

            clean_logits_list.append(logits.cpu().numpy())
            clean_conf_list.append(conf.cpu().numpy())
            preds_clean.append(pred.cpu().numpy())
            labels_all.append(y.numpy())

            if "block_4" in harvester.activations:
                acts_b4.append(harvester.activations["block_4"].numpy())
            if "block_8" in harvester.activations:
                acts_b8.append(harvester.activations["block_8"].numpy())
            if "block_12" in harvester.activations:
                acts_b12.append(harvester.activations["block_12"].numpy())

    clean_logits = np.concatenate(clean_logits_list, axis=0)
    clean_conf = np.concatenate(clean_conf_list, axis=0)
    clean_pred = np.concatenate(preds_clean, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    clean_corr = (clean_pred == labels).astype(np.int8)
    print(f" Done ({time.time()-t0:.1f}s) | Clean Acc: {clean_corr.mean()*100:.2f}%")

    if acts_b12:
        feature_activations[label] = {
            "block_4": np.concatenate(acts_b4, axis=0) if acts_b4 else None,
            "block_8": np.concatenate(acts_b8, axis=0) if acts_b8 else None,
            "block_12": np.concatenate(acts_b12, axis=0),
        }

    # Running Attacks & Gathering Adversarial Logits
    attack_outcomes = {"clean": clean_corr}
    adv_logits_dict = {}

    for atk_name in ["pgd50", "apgd_ce", "apgd_dlr"]:
        print(f"  Harvesting attack: {atk_name:<10}...", end="", flush=True)
        t_atk = time.time()
        adv_logits_batches = []
        adv_preds_batches = []

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if atk_name == "pgd50":
                x_adv = run_pgd_pixel_space(wrapped, x, y, eps=DEFAULT_EPS_PIXEL, steps=50)
            elif atk_name == "apgd_ce":
                x_adv = apgd_ce(x, y)
            elif atk_name == "apgd_dlr":
                x_adv = apgd_dlr(x, y)

            with torch.no_grad():
                l_adv = wrapped(x_adv)
                p_adv = l_adv.argmax(dim=-1)

            adv_logits_batches.append(l_adv.cpu().numpy())
            adv_preds_batches.append(p_adv.cpu().numpy())

        l_all = np.concatenate(adv_logits_batches, axis=0)
        p_all = np.concatenate(adv_preds_batches, axis=0)
        corr_atk = (p_all == labels).astype(np.int8)

        attack_outcomes[atk_name] = corr_atk
        adv_logits_dict[atk_name] = l_all
        print(f" Acc: {corr_atk.mean()*100:>6.2f}% | Time: {time.time()-t_atk:.1f}s")

    # Worst-Case and Failure Attribution
    aa_core = attack_outcomes["apgd_ce"] & attack_outcomes["apgd_dlr"]
    worst_case = attack_outcomes["pgd50"] & attack_outcomes["apgd_ce"] & attack_outcomes["apgd_dlr"]
    attack_outcomes["autoattack_core"] = aa_core
    attack_outcomes["worst_case"] = worst_case

    # Failure attribution index (0: failed on pgd50, 1: failed on apgd_ce, 2: failed on apgd_dlr, -1: survived)
    failure_idx = np.full((1000,), -1, dtype=np.int16)
    for idx in range(1000):
        if attack_outcomes["pgd50"][idx] == 0:
            failure_idx[idx] = 0
        elif attack_outcomes["apgd_ce"][idx] == 0:
            failure_idx[idx] = 1
        elif attack_outcomes["apgd_dlr"][idx] == 0:
            failure_idx[idx] = 2

    # Step-by-Step Belief Evolution (recurrence_only only — gen1_core has
    # incompatible cls_head input dim and does not use the Gen-2 gist pipeline)
    step_logits = []
    step_uncertainty = []
    if phase == "recurrence_only":
        print(f"  Harvesting step-by-step internal trajectory (T=4)...", end="", flush=True)
        t_step = time.time()
        for x, y in loader:
            x = x.to(device)
            B = x.shape[0]
            with torch.no_grad():
                # Extract step trajectory
                gaze_schedule = [
                    torch.zeros(B, 2, device=device),
                    torch.tensor([[0.3, 0.3]], device=device).expand(B, -1),
                    torch.tensor([[-0.3, 0.3]], device=device).expand(B, -1),
                    torch.tensor([[0.0, -0.3]], device=device).expand(B, -1),
                ]
                gist_tokens = base.gist_encoder(x)
                z = gist_tokens.mean(dim=1)
                batch_step_l = [base.cls_head(z).cpu().numpy()]

                for t_idx in range(4):
                    crop = foveal_sample(x, gaze_schedule[t_idx], fovea_size=base.fovea_size)
                    pooled_g, _ = base.backbone(crop)
                    z = 0.5 * z + 0.5 * pooled_g
                    batch_step_l.append(base.cls_head(z).cpu().numpy())

                step_logits.append(np.stack(batch_step_l, axis=1))  # (B, 5, 100)
        print(f" Done ({time.time()-t_step:.1f}s)")

    # Per-Iteration Loss Trajectories for 64 Fixed Samples
    loss_trajectories = []
    fixed_64_x, fixed_64_adv, fixed_64_y = [], [], []
    sample_count = 0
    for x, y in loader:
        if sample_count >= 64:
            break
        n_take = min(64 - sample_count, x.shape[0])
        sub_x = x[:n_take].to(device)
        sub_y = y[:n_take].to(device)
        adv_x, traj = run_pgd_with_loss_trajectory(wrapped, sub_x, sub_y, steps=50)
        loss_trajectories.append(traj)
        fixed_64_x.append(sub_x.cpu())
        fixed_64_adv.append(adv_x.cpu())
        fixed_64_y.append(sub_y.cpu())
        sample_count += n_take

    harvest_results[label] = {
        "clean_acc": float(clean_corr.mean()),
        "pgd50_acc": float(attack_outcomes["pgd50"].mean()),
        "apgd_ce_acc": float(attack_outcomes["apgd_ce"].mean()),
        "apgd_dlr_acc": float(attack_outcomes["apgd_dlr"].mean()),
        "aa_core_acc": float(aa_core.mean()),
        "worst_case_acc": float(worst_case.mean()),
    }

    # Store in bundle
    harvest_arrays[f"{label}_clean_logits"] = clean_logits
    harvest_arrays[f"{label}_clean_conf"]   = clean_conf
    harvest_arrays[f"{label}_clean_corr"]   = clean_corr
    harvest_arrays[f"{label}_worst_corr"]   = worst_case
    harvest_arrays[f"{label}_failure_idx"]  = failure_idx
    for atk_k, atk_v in adv_logits_dict.items():
        harvest_arrays[f"{label}_{atk_k}_logits"] = atk_v
    if step_logits:
        harvest_arrays[f"{label}_step_logits"] = np.concatenate(step_logits, axis=0)

    if label == "recurrence_only":
        fixed_images_bundle["clean_x"] = torch.cat(fixed_64_x).numpy()
        fixed_images_bundle["adv_x"]   = torch.cat(fixed_64_adv).numpy()
        fixed_images_bundle["y"]       = torch.cat(fixed_64_y).numpy()

    harvester.remove()

# ── 8. Compute Pairwise Representation CKA ───────────────────────────────────
print("\n" + "=" * 80)
print("COMPUTING PAIRWISE CENTERED KERNEL ALIGNMENT (CKA)")
print("=" * 80)

cka_matrix: Dict[str, Dict[str, float]] = {}
labels_with_acts = list(feature_activations.keys())

for l1 in labels_with_acts:
    cka_matrix[l1] = {}
    f1 = feature_activations[l1]["block_12"]
    for l2 in labels_with_acts:
        f2 = feature_activations[l2]["block_12"]
        score = compute_linear_cka(f1, f2)
        cka_matrix[l1][l2] = score

print(f"{'Model':<20} " + " ".join(f"{l:<18}" for l in labels_with_acts))
print("-" * (20 + 19 * len(labels_with_acts)))
for l1 in labels_with_acts:
    row_str = " ".join(f"{cka_matrix[l1][l2]:>17.4f}" for l2 in labels_with_acts)
    print(f"{l1:<20} {row_str}")

harvest_results["cka_matrix_block12"] = cka_matrix

# ── 9. Save Bundles & Sync to HuggingFace ─────────────────────────────────────
npz_out = "report/evidence_harvest/evidence_bundle_gen2.npz"
np.savez_compressed(npz_out, **harvest_arrays, **fixed_images_bundle)
print(f"\n✓ Saved binary arrays bundle ({os.path.getsize(npz_out)/1e6:.1f} MB) to: {npz_out}")

json_out = "report/evidence_harvest/evidence_summary.json"
with open(json_out, "w") as f:
    json.dump(harvest_results, f, indent=2)
print(f"✓ Saved forensic summary to: {json_out}")

zip_bundle = "report/evidence_harvest/evidence_bundle_gen2.zip"
subprocess.run(f"zip -j {zip_bundle} {npz_out} {json_out}", shell=True, check=True)

# Upload to HuggingFace
try:
    print(f"\nUploading evidence bundle to {HF_REPO} under 'evidence_harvest/'...")
    api.upload_file(
        path_or_fileobj=zip_bundle,
        path_in_repo="evidence_harvest/evidence_bundle_gen2.zip",
        repo_id=HF_REPO,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=json_out,
        path_in_repo="evidence_harvest/evidence_summary.json",
        repo_id=HF_REPO,
        repo_type="dataset",
    )
    print("✓ Successfully synchronized evidence harvest to HuggingFace!")
except Exception as e:
    print(f"Note: HF upload skipped ({e}). Artifacts preserved locally in report/evidence_harvest.")

print("\n" + "=" * 80)
print("NOTEBOOK 3 EVIDENCE HARVEST COMPLETE")
print("=" * 80)
