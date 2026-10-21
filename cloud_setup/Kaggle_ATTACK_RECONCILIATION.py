#!/usr/bin/env python3
"""
Kaggle Notebook -- J2 Attack Reconciliation Diagnostic (<1 GPU-h)
==================================================================

OBJECTIVE:
----------
Reconcile the 16-27 pp gap between PGD-50 and APGD-CE observed in J2 by:
1. Wrapping models in a NormalizedModel layer:
     DataLoader yields raw images in [0, 1].
     Attacks perturb directly in [0, 1] pixel space at eps = 5.417/255 (0.02124).
     NormalizedModel handles ImageNet mean/std inside forward().
2. Asserting and logging realized Linf bound in pixel units (<= 5.417/255) for EVERY attack.
3. Running APGD-CE in fp32 vs under torch.cuda.amp.autocast() on identical 1,000 images
   to verify whether fp16 underflow explains the discrepancy.
4. Running PGD-200 with 3 restarts and alpha = eps/10 as an upper-strength PGD baseline.
5. Running standard AutoAttack core (APGD-CE + APGD-DLR).
6. Running a transfer attack from feedforward ff_112 to recurrence_only / g2_gist_only.

Target Checkpoints (Evaluated on shared 1,000-image slice, seed=42):
  - ff_112          (112 px, 64 tokens, seed 17, final)
  - recurrence_only (56 px, 16 tokens, T=4, final)
  - g2_gist_only    (56 px, 16 tokens, final)
  - gen1_core       (56 px, 16 tokens, best clean)

USAGE ON KAGGLE:
----------------
1. Attach Accelerator: GPU T4 or P100 (1 GPU is sufficient, runtime < 45 min).
2. Internet: ON.
3. Add-ons > Secrets > Add 'HF_TOKEN' with write permissions.
4. Paste and execute this script. Results are uploaded to HuggingFace dataset
   FerrariKazu/rhan-nxa-g2-checkpoints under reconciliation/.
"""

import os
import sys
import subprocess
import shutil
import time
import datetime
import json
from pathlib import Path

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
    print("WARNING: No GPU detected! Attach a GPU accelerator (T4 or P100) in Notebook Settings.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Install missing libraries
for pkg in ["huggingface_hub", "torchattacks", "scipy"]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.run(f"pip install -q {pkg}", shell=True, check=True)

import torchattacks
from scipy.stats import chi2
import numpy as np

# ── 2. Clone & Checkout Repository ───────────────────────────────────────────
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
    sha = subprocess.check_output("git rev-parse --short HEAD", shell=True, text=True).strip()
    print(f"Checked out {BRANCH} @ commit {sha}")
else:
    print(f"Running locally in {os.getcwd()}")
    sha = subprocess.check_output("git rev-parse --short HEAD", shell=True, text=True).strip()

# Evict cached module references
for mod in list(sys.modules.keys()):
    if any(mod.startswith(p) for p in ["training", "gen2_foundation", "noesis_vision"]):
        del sys.modules[mod]

from gen2_foundation.model import Gen2FoundationModel
from training.curriculum_gen2 import IMAGENET_MEAN, IMAGENET_STD

# ── 3. Checkpoints & ImageNet-100 Dataset Restore ─────────────────────────────
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("report/audit_rev1", exist_ok=True)

from huggingface_hub import HfApi, hf_hub_download, list_repo_files

HF_REPO = "FerrariKazu/rhan-nxa-g2-checkpoints"
api = HfApi(token=hf_token)

REQUIRED_CHECKPOINTS = [
    "foundation_ff_112_final.pth",
    "foundation_recurrence_only_final.pth",
    "foundation_g2_gist_only_final.pth",
    "foundation_gen1_core_best_clean.pth",
]

for ckpt in REQUIRED_CHECKPOINTS:
    local_path = os.path.join("checkpoints", ckpt)
    if not os.path.exists(local_path) or os.path.getsize(local_path) < 1024 * 1024:
        print(f"Downloading {ckpt} from HuggingFace {HF_REPO}...")
        hf_hub_download(repo_id=HF_REPO, filename=ckpt, repo_type="dataset",
                        local_dir="checkpoints", token=hf_token)

# DINOv2-small backbone
DINO_PATH = "checkpoints/dinov2_vits14_pretrain.pth"
DINO_URL  = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
if not os.path.exists(DINO_PATH) or os.path.getsize(DINO_PATH) < 80 * 1024 * 1024:
    print("Downloading DINOv2-small weights...")
    subprocess.run(f"wget -q -c {DINO_URL} -O {DINO_PATH}", shell=True, check=True)

# ImageNet-100 dataset bootstrap
# Use /kaggle/tmp (larger quota, survives inter-cell) rather than /working
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
    print(f"Bootstrapping ImageNet-100 dataset to {DATA_ROOT} ...")
    os.makedirs(DATA_ROOT, exist_ok=True)
    subprocess.run(f"python3 scripts/prepare_imagenet100.py --root {DATA_ROOT}", shell=True, check=True)
    assert _is_dataset_ready(DATA_ROOT), f"Dataset bootstrap failed at {DATA_ROOT}!"
print(f"ImageNet-100 validated at {DATA_ROOT}")

# ── 4. Normalization Wrappers (Pixel Space Execution) ─────────────────────────
class NormalizedModel(nn.Module):
    """Wraps base model with ImageNet normalization.

    Attacks see and perturb raw images in [0, 1] pixel space.
    Forward pass normalizes x with ImageNet mean/std inside the module.
    Gradients backpropagate through normalization directly into pixel space.
    """
    def __init__(self, base_model, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(std,  dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x):
        # x is in [0, 1]
        x_norm = (x - self.mean) / self.std
        return self.base_model(x_norm)


class AutocastNormalizedModel(nn.Module):
    """Executes base model under fp16 autocast() to test fp16 gradient effects."""
    def __init__(self, base_model, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(std,  dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            x_norm = (x - self.mean) / self.std
            logits = self.base_model(x_norm)
        return logits.float()


# ── 5. Upper-Strength PGD Implementation (Pixel Space) ────────────────────────
def run_pgd_pixel_space(model, x, y, eps, steps=50, alpha=None, restarts=1):
    """Run PGD in [0, 1] pixel space with projection to [x-eps, x+eps] cap [0, 1]."""
    if alpha is None:
        alpha = (2.5 * eps) / max(steps, 1)

    best_loss = torch.full((x.shape[0],), -1e9, device=x.device)
    best_x_adv = x.clone()

    for _ in range(restarts):
        x_adv = x.clone() + torch.empty_like(x).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for _ in range(steps):
            x_adv.requires_grad_(True)
            with torch.enable_grad():
                logits = model(x_adv)
                loss = F.cross_entropy(logits, y, reduction="none")
            grad = torch.autograd.grad(loss.sum(), x_adv)[0]
            with torch.no_grad():
                x_adv = x_adv.detach() + alpha * grad.sign()
                x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)

        with torch.no_grad():
            final_loss = F.cross_entropy(model(x_adv), y, reduction="none")
            better = final_loss > best_loss
            best_loss[better] = final_loss[better]
            best_x_adv[better] = x_adv[better]

    return best_x_adv


# ── 6. Evaluation Loader & Infrastructure ─────────────────────────────────────
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

EVAL_SEED   = 42
EVAL_N      = 1000
BATCH_SIZE  = 32
EPS_PIXEL   = 5.417 / 255.0  # Exact target budget in [0, 1] pixel space: ~0.021243

print("\n" + "=" * 70)
print(f"J2 ATTACK RECONCILIATION DIAGNOSTIC  (n={EVAL_N}, seed={EVAL_SEED})")
print(f"  Target Budget: eps = {EPS_PIXEL:.5f} ({EPS_PIXEL*255:.3f}/255 pixel units)")
print("=" * 70)

def build_raw_loader(img_size):
    """Loads raw images in [0, 1] pixel space (no Normalize in loader!)."""
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    val_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"), transform=tf)
    g = torch.Generator().manual_seed(EVAL_SEED)
    idx = torch.randperm(len(val_ds), generator=g)[:EVAL_N].tolist()
    return DataLoader(Subset(val_ds, idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

def load_wrapped_models(phase, img_size, ckpt_path):
    base_m = Gen2FoundationModel(phase=phase, arm="v1", fovea_size=img_size).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    sd = sd.get("model", sd)
    base_m.load_state_dict(sd, strict=False)
    base_m.eval()

    fp32_m = NormalizedModel(base_m).to(device)
    fp32_m.eval()

    amp_m = AutocastNormalizedModel(base_m).to(device)
    amp_m.eval()
    return fp32_m, amp_m

# ── 7. Target Models Specification ───────────────────────────────────────────
TARGET_MODELS = [
    ("ff_112",          "ff_112",           112, "checkpoints/foundation_ff_112_final.pth"),
    ("recurrence_only", "recurrence_only",  56,  "checkpoints/foundation_recurrence_only_final.pth"),
    ("g2_gist_only",    "g2_gist_only",    56,  "checkpoints/foundation_g2_gist_only_final.pth"),
    ("gen1_core",       "gen1_core",        56,  "checkpoints/foundation_gen1_core_best_clean.pth"),
]

diagnostic_results = {}
surrogate_adv_examples = {}  # for transfer attack

# ── 8. Execute Diagnostic Battery ─────────────────────────────────────────────
for label, phase, img_size, ckpt_path in TARGET_MODELS:
    if not os.path.exists(ckpt_path):
        print(f"Skipping {label}: {ckpt_path} not found.")
        continue

    print(f"\n{'#'*70}")
    print(f"  DIAGNOSING: {label} (resolution={img_size}px, ckpt={os.path.basename(ckpt_path)})")
    print(f"{'#'*70}")

    loader = build_raw_loader(img_size)
    m_fp32, m_amp = load_wrapped_models(phase, img_size, ckpt_path)

    # Attack instances operating on [0, 1] raw pixel space
    apgd_ce_fp32 = torchattacks.APGD(m_fp32, norm="Linf", eps=EPS_PIXEL, steps=100, n_restarts=1, loss="ce", verbose=False)
    apgd_ce_amp  = torchattacks.APGD(m_amp,  norm="Linf", eps=EPS_PIXEL, steps=100, n_restarts=1, loss="ce", verbose=False)
    apgd_dlr     = torchattacks.APGD(m_fp32, norm="Linf", eps=EPS_PIXEL, steps=100, n_restarts=1, loss="dlr", verbose=False)

    outcomes = {}
    realized_linfs = {}

    for attack_name in [
        "clean",
        "pgd50",
        "apgd_ce_fp32",
        "apgd_ce_autocast",
        "pgd200_3restarts",
        "apgd_dlr",
    ]:
        print(f"  Running attack: {attack_name:<20}...", end="", flush=True)
        t0 = time.time()
        preds_list = []
        labels_list = []
        max_linf_seen = 0.0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            if attack_name == "clean":
                x_adv = x
            elif attack_name == "pgd50":
                x_adv = run_pgd_pixel_space(m_fp32, x, y, eps=EPS_PIXEL, steps=50, restarts=1)
            elif attack_name == "apgd_ce_fp32":
                x_adv = apgd_ce_fp32(x, y)
            elif attack_name == "apgd_ce_autocast":
                x_adv = apgd_ce_amp(x, y)
            elif attack_name == "pgd200_3restarts":
                # PGD-200 with 3 restarts and alpha = eps / 10
                alpha_200 = EPS_PIXEL / 10.0
                x_adv = run_pgd_pixel_space(m_fp32, x, y, eps=EPS_PIXEL, steps=200, alpha=alpha_200, restarts=3)
            elif attack_name == "apgd_dlr":
                x_adv = apgd_dlr(x, y)

            # Assert and track realized Linf bound in pixel space
            if attack_name != "clean":
                batch_linf = (x_adv - x).abs().max().item()
                max_linf_seen = max(max_linf_seen, batch_linf)
                assert batch_linf <= EPS_PIXEL + 1e-4, (
                    f"VIOLATION: Attack {attack_name} exceeded budget! "
                    f"Realized: {batch_linf*255:.3f}/255 > {EPS_PIXEL*255:.3f}/255"
                )

            with torch.no_grad():
                preds = m_fp32(x_adv).argmax(-1)
            preds_list.extend(preds.cpu().numpy().tolist())
            labels_list.extend(y.cpu().numpy().tolist())

            # Save surrogate adversarial examples from ff_112 for transfer test
            if label == "ff_112" and attack_name == "pgd50":
                surrogate_adv_examples.setdefault("x_adv_ff112", []).append(x_adv.cpu())
                surrogate_adv_examples.setdefault("y", []).append(y.cpu())

        dt = time.time() - t0
        corr = (np.array(preds_list) == np.array(labels_list)).astype(np.int8)
        acc  = float(corr.mean()) * 100.0
        outcomes[attack_name] = corr
        realized_linfs[attack_name] = float(max_linf_seen * 255.0)
        print(f" Acc: {acc:>6.2f}% | Max L_inf: {realized_linfs[attack_name]:.3f}/255 | Time: {dt:.1f}s")

    # Standard AutoAttack Linf core (intersection of APGD-CE and APGD-DLR)
    aa_core_corr = outcomes["apgd_ce_fp32"] & outcomes["apgd_dlr"]
    outcomes["autoattack_core"] = aa_core_corr
    print(f"  AutoAttack Core (CE & DLR)    ... Acc: {aa_core_corr.mean()*100:>6.2f}%")

    # Per-sample worst-case robust accuracy across all attacks
    all_adv_corr = [v for k, v in outcomes.items() if k != "clean"]
    worst_case_corr = np.bitwise_and.reduce(all_adv_corr)
    outcomes["worst_case_all"] = worst_case_corr
    print(f"  Worst-Case Robust (All Attacks)... Acc: {worst_case_corr.mean()*100:>6.2f}%")

    diagnostic_results[label] = {
        "clean_acc": float(outcomes["clean"].mean()),
        "pgd50_acc": float(outcomes["pgd50"].mean()),
        "apgd_ce_fp32_acc": float(outcomes["apgd_ce_fp32"].mean()),
        "apgd_ce_amp_acc": float(outcomes["apgd_ce_autocast"].mean()),
        "pgd200_3restarts_acc": float(outcomes["pgd200_3restarts"].mean()),
        "apgd_dlr_acc": float(outcomes["apgd_dlr"].mean()),
        "autoattack_core_acc": float(outcomes["autoattack_core"].mean()),
        "worst_case_all_acc": float(outcomes["worst_case_all"].mean()),
        "realized_linfs_pix255": realized_linfs,
    }

# ── 9. Transfer Attack Evaluation ─────────────────────────────────────────────
if "x_adv_ff112" in surrogate_adv_examples:
    print(f"\n{'#'*70}")
    print("  RUNNING TRANSFER ATTACK: Surrogate = ff_112 (PGD-50, eps=5.42/255)")
    print(f"{'#'*70}")
    x_adv_all = torch.cat(surrogate_adv_examples["x_adv_ff112"])
    y_all     = torch.cat(surrogate_adv_examples["y"])

    for label, phase, img_size, ckpt_path in TARGET_MODELS:
        if label == "ff_112":
            continue
        m_fp32, _ = load_wrapped_models(phase, img_size, ckpt_path)
        # Interpolate spatial resolution if surrogate resolution (112px) != target resolution (56px)
        if x_adv_all.shape[-1] != img_size:
            x_transfer = F.interpolate(x_adv_all, size=(img_size, img_size), mode="bilinear", align_corners=False)
        else:
            x_transfer = x_adv_all

        transfer_preds = []
        for i in range(0, len(x_transfer), BATCH_SIZE):
            bx = x_transfer[i:i+BATCH_SIZE].to(device)
            with torch.no_grad():
                preds = m_fp32(bx).argmax(-1)
            transfer_preds.extend(preds.cpu().numpy().tolist())

        t_corr = (np.array(transfer_preds) == y_all.numpy()).astype(np.int8)
        t_acc  = float(t_corr.mean()) * 100.0
        print(f"  Transfer ff_112 -> {label:<16} Robust Acc: {t_acc:.2f}% (Transfer Fooling Rate: {100-t_acc:.2f}%)")
        diagnostic_results[label]["transfer_from_ff112_acc"] = float(t_corr.mean())

# ── 10. Summary Reconciliation Matrix ─────────────────────────────────────────
print(f"\n{'='*95}")
print(f"  ATTACK RECONCILIATION SUMMARY MATRIX  (All attacks at eps = {EPS_PIXEL*255:.2f}/255, n={EVAL_N})")
print(f"{'='*95}")
print(f"  {'Model':<16} {'Clean':>7} {'PGD-50':>8} {'APGD(fp32)':>12} {'APGD(amp)':>11} {'PGD-200':>9} {'AA-Core':>9} {'Worst':>7}")
print(f"  {'-'*16} {'-'*7} {'-'*8} {'-'*12} {'-'*11} {'-'*9} {'-'*9} {'-'*7}")

for lbl, r in diagnostic_results.items():
    print(f"  {lbl:<16} "
          f"{r['clean_acc']*100:>6.2f}% "
          f"{r['pgd50_acc']*100:>7.2f}% "
          f"{r['apgd_ce_fp32_acc']*100:>11.2f}% "
          f"{r['apgd_ce_amp_acc']*100:>10.2f}% "
          f"{r['pgd200_3restarts_acc']*100:>8.2f}% "
          f"{r['autoattack_core_acc']*100:>8.2f}% "
          f"{r['worst_case_all_acc']*100:>6.2f}%")

print(f"\nRealized L_inf bounds (must all be <= {EPS_PIXEL*255:.3f}/255):")
for lbl, r in diagnostic_results.items():
    linf_str = ", ".join(f"{k}: {v:.2f}/255" for k, v in r["realized_linfs_pix255"].items() if k != "clean")
    print(f"  {lbl:<16}: {linf_str}")

# ── 11. Save & Sync to HuggingFace ────────────────────────────────────────────
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_json = f"report/audit_rev1/attack_reconciliation_{ts}.json"
with open(out_json, "w") as f:
    json.dump({
        "timestamp": ts,
        "eval_seed": EVAL_SEED,
        "eval_n": EVAL_N,
        "target_eps_pixel_255": round(EPS_PIXEL * 255.0, 4),
        "target_eps_pixel_float": float(EPS_PIXEL),
        "results": diagnostic_results,
    }, f, indent=2)

print(f"\nSaved local artifact to {out_json}")

try:
    print(f"Uploading diagnostic report to HuggingFace {HF_REPO}...")
    api.upload_file(
        path_or_fileobj=out_json,
        path_in_repo=f"reconciliation/attack_reconciliation_{ts}.json",
        repo_id=HF_REPO,
        repo_type="dataset",
    )
    print("Uploaded successfully to HuggingFace!")
except Exception as e:
    print(f"Note: HF sync skipped ({e})")
