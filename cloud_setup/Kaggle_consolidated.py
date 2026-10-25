#!/usr/bin/env python3
"""
Kaggle Notebook -- Consolidated 112px Recurrence Flagship Training & Rigorous Evaluation
========================================================================================
Protocol ID: consolidated-flagship-112/1
Target Model: recurrence_only at 112px (4 shifted views, fovea = full frame)
Baselines:    ff_112 (seeds 17, 42, 99)

THE CONSOLIDATED FLAGSHIP SPECIFICATION
---------------------------------------
Part         Choice                                                 Evidence
Backbone     DINOv2-S/14 warm start, matched fraction >=95% logged  proven
Input        112 px, 4 shifted views at existing offsets, fovea=112  resolution proven
Fusion       gist prior z_0, EMA 0.5/0.5, tied refinement x2        proven pipeline
Recipe       TRADES beta=2.5, eps=5.417/255 (0.0940 norm), PGD-4,   proven recipe
             Rung 3 from epoch 1, 15 epochs, same data as ff_112
Evaluation T always 4, never different                              T-sweep
Excluded     belief, precision, gaze policy, uncertainty, halting   inert / unwired

TRAINING RULES
--------------
1. 3 seeds: 17, 42, 99.
2. Final-epoch checkpoint only (epoch 15) with NO selection on eval images.
3. Fingerprint every checkpoint (phase, eps, beta, epochs, seed, data hash, git commit).
4. Abort guardrails:
   - Abort if eps != 0.0940 +/- 0.001 or KL <= 0.0 on any adversarial batch.
   - Abort if PGD-10 training monitor robust accuracy < 15.0% by epoch 3.
5. Gate 5k clean accuracy against the epoch-15 monitor (within 1.0 pp).
6. Effective batch size = 64 (micro-batch size 16, grad_accum_steps 4).

PRE-REGISTERED DECISION RULE
----------------------------
Flagship is declared BETTER only if:
  1. min(recurrence_112) > max(ff_112) on worst-case robust accuracy across all 5,000 images.
  2. Pooled McNemar 95% confidence interval on discordant pairs excludes zero (p < 0.01).
Otherwise:
  Conclude that at 4x token compute, recurrent fusion does not earn its cost at 112px.
"""

import os
import sys
import json
import time
import math
import shutil
import tempfile
import hashlib
import datetime
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

SESSION_T0 = time.time()

def elapsed_h() -> float:
    return (time.time() - SESSION_T0) / 3600.0

def banner(title: str) -> None:
    print("\n" + "=" * 88)
    print(f"  {title}")
    print("=" * 88, flush=True)


# ============================================================================
# A. PRE-REGISTRATION -- LOCKED BEFORE ANY MODEL LOADS OR TRAINS
# ============================================================================

PROTOCOL = {
    "protocol_id": "consolidated-flagship-112/1",
    "model_phase": "recurrence_only",
    "fovea_size": 112,
    "num_glimpses": 4,
    "seeds": [17, 42, 99],
    "effective_batch_size": 64,
    "micro_batch_size": 16,
    "grad_accum_steps": 4,
    "n_epochs": 15,
    "trades_beta": 2.5,
    "eps_pixel_255": 5.417,
    "eps_norm": 0.0940,  # 5.417 / 255 / 0.226
    "trades_steps": 4,
    "alpha": 0.05875,    # 2.5 * 0.0940 / 4
    "eval_t": 4,
    "clean_set": {"n": 5000, "note": "full val split"},
    "robust_set": {"n": 5000, "note": "full val split, worst-of-3 attacks"},
    "attacks": {
        "pgd50": {"steps": 50, "restarts": 1, "loss": "ce"},
        "apgd_ce": {"steps": 100, "restarts": 1, "loss": "ce"},
        "apgd_dlr": {"steps": 100, "restarts": 1, "loss": "dlr"},
    },
    "worst_case": "bitwise AND of pgd50, apgd_ce, apgd_dlr",
    "checkpoint_gate_pp": 1.0,
    "early_health_epoch": 3,
    "early_health_floor_rob": 0.15,
    "full_val_path_sha256": "6a78a2fde02bfc1ff93b063eeddb32fb32759d9da4994421adda7353636a3be7",
}

RULES = {
    "decision_rule_superiority": (
        "Flagship is declared superior iff min_seed(rec_112) > max_seed(ff_112) on 5k worst-case "
        "robust accuracy AND pooled McNemar exact two-sided p < 0.01 with net delta > 0."
    ),
    "decision_rule_null": (
        "If min_seed(rec_112) <= max_seed(ff_112) or McNemar p >= 0.05, conclusion is that "
        "recurrent fusion does not earn its 4x compute cost at 112px."
    ),
    "abort_epsilon": (
        "Training immediately aborts if eps_norm != 0.0940 +/- 0.001 or if loss_kl <= 0.0."
    ),
    "abort_epoch3_floor": (
        "Training immediately aborts if PGD-10 monitor robust accuracy < 15.0% by epoch 3."
    ),
    "selection_ban": (
        "Strictly evaluate epoch 15 final checkpoint. Best-validation checkpoint selection is forbidden."
    ),
}

def prereg_hash() -> str:
    payload = json.dumps({"protocol": PROTOCOL, "rules": RULES}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()

PROTOCOL_HASH = prereg_hash()

def print_preregistration() -> None:
    banner("A. PRE-REGISTRATION (LOCKED: consolidated-flagship-112/1)")
    print(f"  protocol id   : {PROTOCOL['protocol_id']}")
    print(f"  rules hash    : {PROTOCOL_HASH}")
    print(f"  resolution    : {PROTOCOL['fovea_size']} px (4 shifted views, fovea=full frame)")
    print(f"  recipe        : TRADES beta={PROTOCOL['trades_beta']}, eps={PROTOCOL['eps_pixel_255']}/255 ({PROTOCOL['eps_norm']:.4f} norm)")
    print(f"  inner PGD     : {PROTOCOL['trades_steps']} steps, alpha={PROTOCOL['alpha']:.5f}")
    print(f"  epochs        : {PROTOCOL['n_epochs']} (Rung 3 from epoch 1, no warmup)")
    print(f"  seeds         : {PROTOCOL['seeds']}")
    print(f"  effective bs  : {PROTOCOL['effective_batch_size']} (micro {PROTOCOL['micro_batch_size']} x {PROTOCOL['grad_accum_steps']} accum)")
    print(f"  clean set     : {PROTOCOL['clean_set']['n']} images (full val)")
    print(f"  robust set    : {PROTOCOL['robust_set']['n']} images (full val, worst-of-3 attacks)")
    print(f"  decision rule : min(rec_112) > max(ff_112) AND pooled McNemar p < 0.01")
    print(f"  abort triggers: eps != 0.094, KL <= 0, or epoch 3 PGD-10 < 15.0%")
    print("=" * 88, flush=True)


# ============================================================================
# B. HARDWARE & ENVIRONMENT
# ============================================================================

# Auto-install dependencies on Kaggle
for pkg, mod in [
    ("huggingface_hub", "huggingface_hub"),
    ("torchattacks", "torchattacks"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("pyarrow", "pyarrow"),
]:
    try:
        __import__(mod)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.run(f"pip install -q {pkg}", shell=True, check=True)

# HuggingFace Auth
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
        os.environ["HF_TOKEN"] = hf_token
    except Exception:
        pass

# Kaggle repo bootstrap
if os.path.exists("/kaggle/working"):
    work_dir = Path("/kaggle/working/Adversarial-Cognitive-Model")
    if not work_dir.exists():
        print("Cloning repository on Kaggle...")
        subprocess.run("git clone https://github.com/FerrariKazu/Adversarial-Cognitive-Model.git /kaggle/working/Adversarial-Cognitive-Model", shell=True, check=True)
    os.chdir(str(work_dir))
    subprocess.run("git fetch origin && git checkout feature/gen2-rebuild", shell=True, check=False)
    REPO_DIR = work_dir
else:
    REPO_DIR = Path(".").resolve()

sys.path.insert(0, str(REPO_DIR))
os.environ["PYTHONPATH"] = f"{REPO_DIR}:{os.environ.get('PYTHONPATH', '')}"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import scipy.stats as stats
import torchattacks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM)")
    torch.backends.cudnn.benchmark = True


# Import substrate architecture
from gen2_foundation.model import Gen2FoundationModel
from gen2_foundation.backbone import CompactViTGen2
from noesis_vision.models.foveation import foveal_sample

DINO_PATH = str(REPO_DIR / "checkpoints" / "dinov2_vits14_pretrain.pth")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EPS_PIXEL = PROTOCOL["eps_pixel_255"] / 255.0
EPS_NORM = PROTOCOL["eps_norm"]

OUT_DIR = Path("/kaggle/working/output") if Path("/kaggle/working").exists() else Path("output").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = Path("/kaggle/working/checkpoints") if Path("/kaggle/working").exists() else Path("checkpoints").resolve()
CKPT_DIR.mkdir(parents=True, exist_ok=True)
DATA_ROOT = Path("/kaggle/working/data/imagenet100") if Path("/kaggle/working").exists() else Path("data/imagenet100").resolve()
VAL_ROOT = str(DATA_ROOT / "val")
TRAIN_ROOT = str(DATA_ROOT / "train")


# ============================================================================
# C. DATA BOOTSTRAP & DINO WEIGHTS
# ============================================================================

def ensure_dino_weights(path: str = DINO_PATH) -> str:
    p = Path(path)
    if p.exists() and p.stat().st_size > 80 * 1024 * 1024:
        return str(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official DINOv2-small weights to {p} ...")
    url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
    subprocess.run(["curl", "-sSL", "-o", str(p), url], check=True)
    assert p.exists() and p.stat().st_size > 80 * 1024 * 1024, "DINO download failed"
    return str(p)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def bootstrap_imagenet100(data_dir: Path) -> None:
    if (data_dir / "val").exists() and (data_dir / "train").exists():
        n_val = sum(1 for _ in (data_dir / "val").glob("*/*.JPEG")) + sum(1 for _ in (data_dir / "val").glob("*/*.jpg"))
        if n_val == 5000:
            print(f"ImageNet-100 already bootstrapped at {data_dir} (val=5000).")
            return

    data_dir.mkdir(parents=True, exist_ok=True)
    banner("BOOTSTRAPPING IMAGENET-100 FROM HUGGINGFACE")
    import pyarrow.parquet as pq
    import io
    from PIL import Image
    from huggingface_hub import HfApi, hf_hub_download

    repo_id = "clane9/imagenet-100"
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    train_files = sorted([f for f in files if f.startswith("data/train-") and f.endswith(".parquet")])
    val_files = sorted([f for f in files if f.startswith("data/val-") and f.endswith(".parquet")])

    for split_name, shard_list in [("val", val_files), ("train", train_files)]:
        split_dir = data_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        print(f"Unpacking {split_name} ({len(shard_list)} shards) ...")
        for shard in shard_list:
            local_parquet = hf_hub_download(repo_id=repo_id, filename=shard, repo_type="dataset")
            table = pq.read_table(local_parquet)
            cols = table.to_dict()
            imgs = cols["image"]
            lbls = cols["label"]
            for img_dict, lbl in zip(imgs, lbls):
                class_name = str(lbl)
                target_folder = split_dir / class_name
                target_folder.mkdir(parents=True, exist_ok=True)
                img_bytes = img_dict["bytes"]
                img_path = target_folder / f"{hashlib.md5(img_bytes).hexdigest()[:12]}.JPEG"
                if not img_path.exists():
                    with open(img_path, "wb") as f_out:
                        f_out.write(img_bytes)

    print("ImageNet-100 bootstrap complete.")


# ============================================================================
# D. ATTACKS & LOSSES (STANDALONE HARNESS & TRADES)
# ============================================================================

def run_pgd_kl_attack(
    model: nn.Module,
    x: torch.Tensor,
    eps: float,
    steps: int = 4,
    alpha: Optional[float] = None,
) -> torch.Tensor:
    """Inner PGD loop for TRADES objective (runs in fp32 outside autocast)."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits_clean = model(x).float()
            probs_clean = F.softmax(logits_clean, dim=1)

        x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, -4.0, 4.0)
        if alpha is None:
            alpha = (2.5 * eps) / max(steps, 1)

        for _ in range(steps):
            x_adv.requires_grad_(True)
            with torch.enable_grad():
                logits_adv = model(x_adv).float()
                loss_kl = F.kl_div(
                    F.log_softmax(logits_adv, dim=1),
                    probs_clean,
                    reduction="batchmean",
                )
            grad = torch.autograd.grad(loss_kl, x_adv)[0]
            with torch.no_grad():
                x_adv = x_adv.detach() + alpha * grad.sign()
                x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
                x_adv = x_adv.clamp(-4.0, 4.0)
        return x_adv.detach()
    finally:
        if was_training:
            model.train()


def run_pgd_ce_monitor(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    steps: int = 10,
    alpha: Optional[float] = None,
) -> torch.Tensor:
    """PGD-10 CE attack used strictly for the training monitor."""
    was_training = model.training
    model.eval()
    try:
        if alpha is None:
            alpha = (2.5 * eps) / max(steps, 1)
        x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, -4.0, 4.0)
        for _ in range(steps):
            x_adv.requires_grad_(True)
            with torch.enable_grad():
                logits = model(x_adv).float()
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            with torch.no_grad():
                x_adv = x_adv.detach() + alpha * grad.sign()
                x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
                x_adv = x_adv.clamp(-4.0, 4.0)
        return x_adv.detach()
    finally:
        if was_training:
            model.train()


# ============================================================================
# E. DATA LOADERS & DATASET HASH
# ============================================================================

def get_loaders(
    data_root: Path,
    img_size: int = 112,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.35, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(str(data_root / "train"), transform=tf_train)
    val_ds = datasets.ImageFolder(str(data_root / "val"), transform=tf_val)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # Fixed 512-image subset for monitor
    g_mon = torch.Generator().manual_seed(42)
    mon_idx = torch.randperm(len(val_ds), generator=g_mon)[:512].tolist()
    monitor_loader = DataLoader(
        Subset(val_ds, mon_idx), batch_size=64, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, monitor_loader


# ============================================================================
# F. TRAINING FUNCTION FOR THE CONSOLIDATED FLAGSHIP
# ============================================================================

def train_consolidated_seed(seed: int) -> Path:
    banner(f"TRAINING CONSOLIDATED FLAGSHIP -- SEED {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    final_ckpt = CKPT_DIR / f"foundation_recurrence_112_seed{seed}_final.pth"
    rolling_ckpt = CKPT_DIR / f"foundation_recurrence_112_seed{seed}_rolling.pth"

    if final_ckpt.exists():
        print(f"Final checkpoint already exists at {final_ckpt}. Skipping training.")
        return final_ckpt

    ensure_dino_weights(DINO_PATH)
    train_loader, val_loader, monitor_loader = get_loaders(
        DATA_ROOT, img_size=112, batch_size=PROTOCOL["micro_batch_size"]
    )

    model = Gen2FoundationModel(
        phase="recurrence_only",
        arm="v1",
        fovea_size=112,
        num_glimpses=4,
    ).to(DEVICE)

    # Load DINOv2-S/14 warm start and log matched fraction
    rep = model.backbone.load_dino_warm_start(DINO_PATH)
    print(f"DINOv2 Warm-start Load: accepted={rep['accepted']}, "
          f"matched={rep['matched_trunk_keys']}/{rep['trunk_keys']} "
          f"({rep['matched_fraction']*100:.1f}%)")
    assert rep["accepted"], f"DINO warm-start failed to meet strict threshold: {rep}"

    # Optimizer & LR groups
    trunk_params = list(model.backbone.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW([
        {"params": trunk_params, "lr": 5e-5, "weight_decay": 0.05},
        {"params": other_params, "lr": 5e-4, "weight_decay": 0.05},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PROTOCOL["n_epochs"], eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    start_epoch = 1
    if rolling_ckpt.exists():
        print(f"Resuming from rolling checkpoint: {rolling_ckpt}")
        sd = torch.load(rolling_ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(sd["model"])
        optimizer.load_state_dict(sd["optimizer"])
        scheduler.load_state_dict(sd["scheduler"])
        start_epoch = sd["epoch"] + 1
        print(f"Resumed at epoch {start_epoch}")

    # Hash of training manifest
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_hash = "unknown"

    epoch_logs = []
    grad_accum_steps = PROTOCOL["grad_accum_steps"]

    for epoch in range(start_epoch, PROTOCOL["n_epochs"] + 1):
        t0 = time.time()
        model.train()
        tot_loss, tot_ce, tot_kl = 0.0, 0.0, 0.0
        n_batches = len(train_loader)

        optimizer.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

            # Assert perturbation budget
            assert abs(EPS_NORM - 0.0940) < 0.001, f"GUARDRAIL FAIL: eps_norm is {EPS_NORM}, expected 0.0940"

            # Inner PGD-4 attack
            x_adv = run_pgd_kl_attack(
                model, x, eps=EPS_NORM, steps=PROTOCOL["trades_steps"], alpha=PROTOCOL["alpha"]
            )

            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                logits_clean = model(x)
                logits_adv = model(x_adv)
                loss_ce = F.cross_entropy(logits_clean, y)
                loss_kl = F.kl_div(
                    F.log_softmax(logits_adv.float(), dim=1),
                    F.softmax(logits_clean.float(), dim=1),
                    reduction="batchmean",
                )

                # Guardrail: KL divergence must strictly be positive
                if loss_kl.item() <= 0.0:
                    raise RuntimeError(f"GUARDRAIL FAIL: KL divergence <= 0 ({loss_kl.item():.6f}) at step {step}")

                loss = loss_ce + PROTOCOL["trades_beta"] * loss_kl
                loss_scaled = loss / grad_accum_steps

            scaler.scale(loss_scaled).backward()

            if step % grad_accum_steps == 0 or step == n_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            tot_loss += loss.item()
            tot_ce += loss_ce.item()
            tot_kl += loss_kl.item()

            if step % 50 == 0 or step == n_batches:
                print(f"  [Ep {epoch:02d}/{PROTOCOL['n_epochs']:02d} | Step {step:04d}/{n_batches:04d}] "
                      f"Loss: {loss.item():.4f} (CE: {loss_ce.item():.4f}, KL: {loss_kl.item():.4f})", flush=True)

        scheduler.step()

        # Epoch Validation Monitor
        model.eval()
        n_corr_clean, n_corr_rob, n_tot = 0, 0, 0
        with torch.no_grad():
            for xm, ym in monitor_loader:
                xm, ym = xm.to(DEVICE), ym.to(DEVICE)
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    pred_clean = model(xm).argmax(dim=-1)
                n_corr_clean += (pred_clean == ym).sum().item()

                # Robust PGD-10 monitor
                xm_adv = run_pgd_ce_monitor(model, xm, ym, eps=EPS_NORM, steps=10)
                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    pred_rob = model(xm_adv).argmax(dim=-1)
                n_corr_rob += (pred_rob == ym).sum().item()
                n_tot += xm.size(0)

        mon_clean_acc = n_corr_clean / n_tot
        mon_rob_acc = n_corr_rob / n_tot
        ep_duration = time.time() - t0

        print(f"==> Epoch {epoch:02d} Complete ({ep_duration:.1f}s) | "
              f"Monitor Clean: {mon_clean_acc*100:.2f}% | Monitor PGD-10 Rob: {mon_rob_acc*100:.2f}%")

        # Guardrail: Epoch 3 Robust Accuracy Floor Check
        if epoch == PROTOCOL["early_health_epoch"]:
            if mon_rob_acc < PROTOCOL["early_health_floor_rob"]:
                raise RuntimeError(
                    f"GUARDRAIL ABORT: PGD-10 robust accuracy at epoch 3 is {mon_rob_acc*100:.2f}%, "
                    f"which is below the required {PROTOCOL['early_health_floor_rob']*100:.1f}% floor!"
                )

        # Save rolling checkpoint
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "monitor_clean": mon_clean_acc,
            "monitor_robust": mon_rob_acc,
        }, rolling_ckpt)

    # Final epoch completed -- save cryptographically fingerprinted checkpoint
    fingerprint = {
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_hash": PROTOCOL_HASH,
        "phase": "recurrence_112",
        "seed": seed,
        "fovea_size": 112,
        "epoch": 15,
        "total_epochs": 15,
        "eps_pixel_255": PROTOCOL["eps_pixel_255"],
        "eps_norm": PROTOCOL["eps_norm"],
        "trades_beta": PROTOCOL["trades_beta"],
        "trades_steps": PROTOCOL["trades_steps"],
        "git_commit": git_hash,
        "final_monitor_clean": mon_clean_acc,
        "final_monitor_robust": mon_rob_acc,
    }

    torch.save({
        "model": model.state_dict(),
        "meta": fingerprint,
    }, final_ckpt)
    print(f"Saved fingerprinted final checkpoint to {final_ckpt}")
    return final_ckpt


# ============================================================================
# G. FULL 5,000-IMAGE EVALUATION BATTERY
# ============================================================================

def full_5k_battery(model: nn.Module, label: str) -> Dict[str, Any]:
    """Clean + PGD-50 + APGD-CE + APGD-DLR on all 5,000 validation images."""
    banner(f"FULL 5k EVALUATION -- {label}")
    tf_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_ds = datasets.ImageFolder(VAL_ROOT, transform=tf_val)
    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    apgd_ce = torchattacks.APGD(model, norm="Linf", eps=EPS_NORM, steps=100, n_restarts=1, loss="ce", verbose=False)
    apgd_dlr = torchattacks.APGD(model, norm="Linf", eps=EPS_NORM, steps=100, n_restarts=1, loss="dlr", verbose=False)

    preds_dict: Dict[str, List[np.ndarray]] = {"clean": [], "pgd50": [], "apgd_ce": [], "apgd_dlr": []}
    corr_dict: Dict[str, List[np.ndarray]] = {"clean": [], "pgd50": [], "apgd_ce": [], "apgd_dlr": []}

    model.eval()
    for b_idx, (x, y) in enumerate(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_np = y.cpu().numpy()

        # 1. Clean
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                p_clean = model(x).argmax(dim=-1).cpu().numpy()
        preds_dict["clean"].append(p_clean)
        corr_dict["clean"].append((p_clean == y_np).astype(np.int8))

        # 2. PGD-50
        x_pgd50 = run_pgd_ce_monitor(model, x, y, eps=EPS_NORM, steps=50)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                p_pgd50 = model(x_pgd50).argmax(dim=-1).cpu().numpy()
        preds_dict["pgd50"].append(p_pgd50)
        corr_dict["pgd50"].append((p_pgd50 == y_np).astype(np.int8))

        # 3. APGD-CE
        x_apgd_ce = apgd_ce(x, y)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                p_apgd_ce = model(x_apgd_ce).argmax(dim=-1).cpu().numpy()
        preds_dict["apgd_ce"].append(p_apgd_ce)
        corr_dict["apgd_ce"].append((p_apgd_ce == y_np).astype(np.int8))

        # 4. APGD-DLR
        x_apgd_dlr = apgd_dlr(x, y)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                p_apgd_dlr = model(x_apgd_dlr).argmax(dim=-1).cpu().numpy()
        preds_dict["apgd_dlr"].append(p_apgd_dlr)
        corr_dict["apgd_dlr"].append((p_apgd_dlr == y_np).astype(np.int8))

        if (b_idx + 1) % 25 == 0 or (b_idx + 1) == len(loader):
            print(f"  [5k Eval {b_idx+1}/{len(loader)}] Batch processed.", flush=True)

    res = {}
    for k in ("clean", "pgd50", "apgd_ce", "apgd_dlr"):
        res[f"corr_{k}"] = np.concatenate(corr_dict[k])
        res[f"pred_{k}"] = np.concatenate(preds_dict[k])
        res[f"acc_{k}"] = float(res[f"corr_{k}"].mean())

    res["corr_worst_case"] = (res["corr_pgd50"] & res["corr_apgd_ce"] & res["corr_apgd_dlr"]).astype(np.int8)
    res["acc_worst_case"] = float(res["corr_worst_case"].mean())

    print(f"\n{label} Results (N=5,000):")
    print(f"  Clean:      {res['acc_clean']*100:.2f}%")
    print(f"  PGD-50:     {res['acc_pgd50']*100:.2f}%")
    print(f"  APGD-CE:    {res['acc_apgd_ce']*100:.2f}%")
    print(f"  APGD-DLR:   {res['acc_apgd_dlr']*100:.2f}%")
    print(f"  Worst-Case: {res['acc_worst_case']*100:.2f}%\n")
    return res


# ============================================================================
# H. STATISTICAL SYNTHESIS & DECISION RULE
# ============================================================================

def mcnemar_test(corr_a: np.ndarray, corr_b: np.ndarray) -> Dict[str, Any]:
    """Exact two-sided McNemar binomial test with 95% confidence interval."""
    assert len(corr_a) == len(corr_b)
    b = int(((corr_a == 1) & (corr_b == 0)).sum())
    c = int(((corr_a == 0) & (corr_b == 1)).sum())
    n_disc = b + c
    delta = (corr_a.mean() - corr_b.mean()) * 100.0

    if n_disc == 0:
        return {"b": b, "c": c, "p_exact": 1.0, "delta_pp": delta, "ci_95": (delta, delta)}

    p_exact = float(stats.binomtest(b, n_disc, 0.5, alternative="two-sided").pvalue)
    chi2 = float((abs(b - c) - 1.0) ** 2 / n_disc) if n_disc > 0 else 0.0

    # Agresti-Min 95% CI on discordant proportion
    se = math.sqrt((b + c) - ((b - c) ** 2 / len(corr_a))) / len(corr_a) * 100.0
    ci_low = delta - 1.96 * se
    ci_high = delta + 1.96 * se

    return {
        "b": b,
        "c": c,
        "n_discordant": n_disc,
        "delta_pp": delta,
        "p_exact": p_exact,
        "chi2": chi2,
        "ci_95": (ci_low, ci_high),
    }


# ============================================================================
# I. MAIN ORCHESTRATION
# ============================================================================

def main() -> None:
    print_preregistration()

    # 1. Ensure Data & Weights
    ensure_dino_weights(DINO_PATH)
    bootstrap_imagenet100(DATA_ROOT)

    # 2. Train Consolidated Flagship (Seeds 17, 42, 99)
    flagship_ckpts = {}
    for s in PROTOCOL["seeds"]:
        flagship_ckpts[s] = train_consolidated_seed(s)

    # 3. Locate / Download the 3 ff_112 Baselines
    ff112_ckpts = {
        17: CKPT_DIR / "foundation_ff_112_final.pth",
        42: CKPT_DIR / "foundation_ff_112_seed42_final.pth",
        99: CKPT_DIR / "foundation_ff_112_seed99_final.pth",
    }

    # If any ff_112 ckpts missing locally, try fetching from HF
    for s, p in ff112_ckpts.items():
        if not p.exists():
            from huggingface_hub import hf_hub_download
            try:
                print(f"Downloading {p.name} from HuggingFace...")
                dl = hf_hub_download(
                    repo_id="FerrariKazu/rhan-nxa-g2-checkpoints",
                    filename=p.name,
                    local_dir=str(CKPT_DIR),
                )
            except Exception as e:
                print(f"Could not download {p.name}: {e}")

    # 4. Evaluate all 6 checkpoints on full 5,000 images
    eval_results = {}
    per_sample_bundle = {}

    # Evaluate Flagship seeds
    for s in PROTOCOL["seeds"]:
        lbl = f"recurrence_112_seed{s}"
        m = Gen2FoundationModel(phase="recurrence_only", arm="v1", fovea_size=112, num_glimpses=4).to(DEVICE)
        st = torch.load(flagship_ckpts[s], map_location=DEVICE, weights_only=False)
        m.load_state_dict(st.get("model", st))
        res = full_5k_battery(m, lbl)
        eval_results[lbl] = {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
        per_sample_bundle[f"{lbl}_clean"] = res["corr_clean"]
        per_sample_bundle[f"{lbl}_worst"] = res["corr_worst_case"]

    # Evaluate ff_112 Baseline seeds
    for s in PROTOCOL["seeds"]:
        lbl = f"ff_112_seed{s}"
        if ff112_ckpts[s].exists():
            m_ff = Gen2FoundationModel(phase="ff_112", arm="v1", fovea_size=112, num_glimpses=4).to(DEVICE)
            st_ff = torch.load(ff112_ckpts[s], map_location=DEVICE, weights_only=False)
            m_ff.load_state_dict(st_ff.get("model", st_ff))
            res_ff = full_5k_battery(m_ff, lbl)
            eval_results[lbl] = {k: v for k, v in res_ff.items() if not isinstance(v, np.ndarray)}
            per_sample_bundle[f"{lbl}_clean"] = res_ff["corr_clean"]
            per_sample_bundle[f"{lbl}_worst"] = res_ff["corr_worst_case"]
        else:
            print(f"WARNING: Baseline {lbl} not found at {ff112_ckpts[s]}! Skipping.")

    # 5. Pre-registered Hypothesis Testing
    banner("PRE-REGISTERED HYPOTHESIS TESTING & DECISION RULE")
    rec_worst_accs = [eval_results[f"recurrence_112_seed{s}"]["acc_worst_case"] for s in PROTOCOL["seeds"]]
    min_rec = min(rec_worst_accs)

    ff_present = [s for s in PROTOCOL["seeds"] if f"ff_112_seed{s}" in eval_results]
    if len(ff_present) == len(PROTOCOL["seeds"]):
        ff_worst_accs = [eval_results[f"ff_112_seed{s}"]["acc_worst_case"] for s in PROTOCOL["seeds"]]
        max_ff = max(ff_worst_accs)

        # Pooled McNemar test across all 3 seed pairs
        pooled_rec = np.concatenate([per_sample_bundle[f"recurrence_112_seed{s}_worst"] for s in PROTOCOL["seeds"]])
        pooled_ff = np.concatenate([per_sample_bundle[f"ff_112_seed{s}_worst"] for s in PROTOCOL["seeds"]])
        pooled_test = mcnemar_test(pooled_rec, pooled_ff)

        cond1_pass = min_rec > max_ff
        cond2_pass = (pooled_test["p_exact"] < 0.01) and (pooled_test["ci_95"][0] > 0.0)

        print(f"Condition 1: min(rec_112) > max(ff_112)")
        print(f"  min(rec_112) = {min_rec*100:.2f}% | max(ff_112) = {max_ff*100:.2f}% -> {'PASS' if cond1_pass else 'FAIL'}")
        print(f"Condition 2: Pooled McNemar CI excludes 0 (p < 0.01)")
        print(f"  Delta: +{pooled_test['delta_pp']:.2f} pp | 95% CI: [{pooled_test['ci_95'][0]:.2f}, {pooled_test['ci_95'][1]:.2f}] | p = {pooled_test['p_exact']:.3e} -> {'PASS' if cond2_pass else 'FAIL'}")

        if cond1_pass and cond2_pass:
            verdict = "FLAGSHIP CONFIRMED: Recurrent loop achieves superior robustness at 112px."
        else:
            verdict = "NULL / COST INEFFECTIVE: Recurrent loop does not earn its 4x compute cost at 112px."

        print(f"\nFINAL VERDICT:\n  {verdict}\n")
    else:
        verdict = "INCOMPLETE: One or more ff_112 baselines were missing from evaluation."
        pooled_test = {}

    # 6. Save Bundle and Report
    report = {
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_hash": PROTOCOL_HASH,
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "eval_results": eval_results,
        "pooled_test": pooled_test,
        "verdict": verdict,
    }

    report_path = OUT_DIR / "consolidated_flagship_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    bundle_path = OUT_DIR / "per_sample_bundle_112.npz"
    np.savez_compressed(bundle_path, **per_sample_bundle)
    print(f"Saved report to {report_path}")
    print(f"Saved per-sample bundle to {bundle_path}")

    # 7. Push to HuggingFace
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        banner("UPLOADING TO HUGGINGFACE")
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        repo_id = "FerrariKazu/rhan-nxa-g2-checkpoints"
        prefix = f"closure/consolidated_112/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        api.upload_file(path_or_fileobj=str(report_path), path_in_repo=f"{prefix}/report.json", repo_id=repo_id)
        api.upload_file(path_or_fileobj=str(bundle_path), path_in_repo=f"{prefix}/per_sample_bundle.npz", repo_id=repo_id)
        for s in PROTOCOL["seeds"]:
            api.upload_file(path_or_fileobj=str(flagship_ckpts[s]), path_in_repo=f"{prefix}/foundation_recurrence_112_seed{s}_final.pth", repo_id=repo_id)
        print("Upload complete!")

    banner("RUN COMPLETE")


if __name__ == "__main__":
    main()
