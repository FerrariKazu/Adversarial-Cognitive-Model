#!/usr/bin/env python3
"""
Colab Notebook -- Consolidated 112px Recurrence Flagship Training & Rigorous Evaluation
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
Evaluation   T always 4, never different                            T-sweep
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

USAGE ON GOOGLE COLAB
---------------------
  1. New Colab notebook. Runtime > Change runtime type > T4 GPU (or A100/V100).
  2. Secrets (key icon in left sidebar) > Add secret "HF_TOKEN" > Enable notebook access.
  3. Run all cells in sequence (or run via terminal: `python3 cloud_setup/colab_consolidated.py`).
  4. Disconnect resilience:
     - Live sync uploads rolling checkpoints & logs to HuggingFace after every epoch.
     - On restart, Step 4 pulls the latest rolling checkpoint and training automatically resumes.
     - Completed final checkpoints (epoch 15) are cached and never re-trained.
"""

# %% [markdown]
# ## Step 1: Environment, Dependencies & HuggingFace Secrets

# %%
import os
import sys
import json
import time
import math
import shutil
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

# Auto-install dependencies on Colab
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
        subprocess.run(["pip", "install", "--quiet", pkg], check=True)

# HuggingFace Auth (Colab userdata with environment fallback)
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from google.colab import userdata
        hf_token = userdata.get("HF_TOKEN")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
    except Exception:
        pass

if not hf_token:
    print("WARNING: HF_TOKEN not found in environment or Colab secrets.")
    print("Set secret 'HF_TOKEN' with notebook access enabled to enable checkpoint backup and restore.")
else:
    print(f"HF_TOKEN authenticated (starts with: {hf_token[:4]}...)")

# %% [markdown]
# ## Step 2: Repository Clone & Workspace Paths (Colab)

# %%
REPO_NAME = "Adversarial-Cognitive-Model"
BRANCH    = "feature/gen2-rebuild"

if os.path.exists("/content"):
    WORK_DIR = Path(f"/content/{REPO_NAME}")
    if not WORK_DIR.exists():
        print(f"Cloning FerrariKazu/{REPO_NAME}...")
        subprocess.run(
            f"git clone https://github.com/FerrariKazu/{REPO_NAME}.git {WORK_DIR}",
            shell=True,
            check=True,
        )
    os.chdir(str(WORK_DIR))
    subprocess.run("git fetch origin", shell=True, check=True)
    subprocess.run(f"git checkout -B {BRANCH} origin/{BRANCH}", shell=True, check=True)
    subprocess.run(f"git reset --hard origin/{BRANCH}", shell=True, check=True)
    REPO_DIR = WORK_DIR.resolve()
else:
    REPO_DIR = Path(".").resolve()

sys.path.insert(0, str(REPO_DIR))
os.environ["PYTHONPATH"] = f"{REPO_DIR}:{os.environ.get('PYTHONPATH', '')}"

# Evict stale cached modules
for mod in list(sys.modules.keys()):
    if any(mod.startswith(p) for p in ["training", "gen2_foundation", "noesis_vision"]):
        del sys.modules[mod]

try:
    sha = subprocess.check_output("git rev-parse --short HEAD", shell=True, text=True, cwd=str(REPO_DIR)).strip()
    print(f"Checked out {BRANCH} @ commit {sha}")
except Exception:
    sha = "unknown"

# Define canonical directory layout
OUT_DIR    = REPO_DIR / "output"
CKPT_DIR   = REPO_DIR / "checkpoints"
DATA_ROOT  = Path(os.environ.get("J2_DATA_ROOT", "/content/imagenet100" if os.path.exists("/content") else str(REPO_DIR / "data" / "imagenet100"))).resolve()
VAL_ROOT   = str(DATA_ROOT / "val")
TRAIN_ROOT = str(DATA_ROOT / "train")
DINO_PATH  = str(CKPT_DIR / "dinov2_vits14_pretrain.pth")

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "eval").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "audit_rev1").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "report").mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Step 3: Hardware Verification & Locked Pre-Registration

# %%
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
else:
    print("WARNING: No GPU detected! Go to Runtime > Change runtime type > T4 GPU.")

# Import substrate architecture
from gen2_foundation.model import Gen2FoundationModel
from gen2_foundation.backbone import CompactViTGen2
from noesis_vision.models.foveation import foveal_sample

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Locked protocol specification -- identical to Kaggle_consolidated.py
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

EPS_PIXEL = PROTOCOL["eps_pixel_255"] / 255.0
EPS_NORM  = PROTOCOL["eps_norm"]

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

# %% [markdown]
# ## Step 4: HuggingFace Repos, Live Sync & Checkpoint Restore

# %%
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

HF_BEST    = "FerrariKazu/rhan-nxa-g2-checkpoints"
HF_ROLLING = "FerrariKazu/rhan-nxa-g2-checkpoints-rolling"

if hf_token:
    api = HfApi(token=hf_token)
    for r in [HF_BEST, HF_ROLLING]:
        try:
            api.create_repo(repo_id=r, repo_type="dataset", private=True, exist_ok=True)
            print(f"HF repo ready: {r}")
        except Exception as e:
            print(f"  HF check {r}: {e}")

def restore_checkpoints_from_hf() -> None:
    """Pull any checkpoints/logs/reports from HF that are missing locally.
    Enables Colab to resume training seamlessly from a prior session's rolling checkpoint."""
    if not hf_token:
        return
    restored = 0
    api = HfApi(token=hf_token)
    for repo_id in [HF_BEST, HF_ROLLING]:
        try:
            files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=hf_token)
            for f in files:
                bn = os.path.basename(f)
                is_rec_rolling = bn.startswith("foundation_recurrence_112_") and bn.endswith("_rolling.pth")
                is_final_needed = bn.startswith("foundation_") and bn.endswith("_final.pth") and (
                    "recurrence_112" in bn or "ff_112" in bn
                )
                if is_rec_rolling or is_final_needed:
                    local_f = CKPT_DIR / bn
                elif bn.startswith("eval_") and "112" in bn:
                    local_f = OUT_DIR / "eval" / bn
                elif bn == "gen2_epoch_log.jsonl" and not f.startswith("closure/"):
                    local_f = OUT_DIR / bn
                else:
                    continue
                if local_f is not None:
                    # For rolling checkpoints, always refresh from HF to get latest epoch state
                    needs_download = not local_f.exists() or bn.endswith("_rolling.pth")
                    if needs_download:
                        print(f"  Restoring/updating {f} from {repo_id}...")
                        p = hf_hub_download(repo_id=repo_id, filename=f,
                                            repo_type="dataset", token=hf_token)
                        local_f.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(p, str(local_f))
                        restored += 1
        except Exception as e:
            print(f"  Note: {repo_id}: {e}")
    print(f"Restored {restored} file(s) from HF.")

def sync_to_hf() -> None:
    """Upload new checkpoints, eval JSONs, and JSONL logs to HF.
    Called on each completed epoch via epoch_callback to prevent loss of training progress."""
    if not hf_token:
        return
    api = HfApi(token=hf_token)
    # 1. Rolling checkpoints
    for cp in CKPT_DIR.glob("*_rolling.pth"):
        try:
            api.upload_file(path_or_fileobj=str(cp), path_in_repo=cp.name,
                            repo_id=HF_ROLLING, repo_type="dataset", token=hf_token)
        except Exception as e:
            print(f"  Warning: rolling sync {cp.name}: {e}")
    # 2. Final and best checkpoints
    for cp in list(CKPT_DIR.glob("*_final.pth")) + list(CKPT_DIR.glob("*_best_*.pth")):
        try:
            api.upload_file(path_or_fileobj=str(cp), path_in_repo=cp.name,
                            repo_id=HF_BEST, repo_type="dataset", token=hf_token)
        except Exception as e:
            print(f"  Warning: best sync {cp.name}: {e}")
    # 3. JSONL epoch logs
    for log_file in list(OUT_DIR.glob("*.jsonl")) + list((REPO_DIR / "report").glob("*.jsonl")):
        try:
            api.upload_file(path_or_fileobj=str(log_file), path_in_repo=log_file.name,
                            repo_id=HF_ROLLING, repo_type="dataset", token=hf_token)
        except Exception as e:
            print(f"  Warning: log sync {log_file.name}: {e}")

restore_checkpoints_from_hf()

# %% [markdown]
# ## Step 5: DINOv2 Weights & ImageNet-100 Data Bootstrap

# %%
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

def bootstrap_imagenet100(data_dir: Path) -> None:
    data_dir = Path(data_dir).resolve()
    td, vd = data_dir / "train", data_dir / "val"
    if td.is_dir() and vd.is_dir():
        n_val = sum(1 for _ in vd.glob("*/*.JPEG")) + sum(1 for _ in vd.glob("*/*.jpg"))
        if n_val == 5000:
            print(f"ImageNet-100 already bootstrapped at {data_dir} (val=5000).")
            return

    data_dir.mkdir(parents=True, exist_ok=True)
    prep_script = REPO_DIR / "scripts" / "prepare_imagenet100.py"
    if prep_script.exists():
        print(f"Running official {prep_script} --root {data_dir} ...")
        res = subprocess.run([sys.executable, str(prep_script), "--root", str(data_dir)])
        if res.returncode == 0:
            print("ImageNet-100 bootstrap via prepare_imagenet100.py complete.")
            return
        print(f"prepare_imagenet100.py exited with {res.returncode}; falling back to direct parquet extraction...")

    banner("BOOTSTRAPPING IMAGENET-100 FROM HUGGINGFACE")
    import pyarrow.parquet as pq
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
            cols = table.to_pydict()
            imgs = cols["image"]
            lbls = cols["label"]
            for img_data, lbl in zip(imgs, lbls):
                class_name = str(lbl)
                target_folder = split_dir / class_name
                target_folder.mkdir(parents=True, exist_ok=True)
                img_bytes = img_data["bytes"] if isinstance(img_data, dict) else img_data
                img_path = target_folder / f"{hashlib.md5(img_bytes).hexdigest()[:12]}.JPEG"
                if not img_path.exists():
                    with open(img_path, "wb") as f_out:
                        f_out.write(img_bytes)

    print("ImageNet-100 bootstrap complete.")

# %% [markdown]
# ## Step 6: Attacks & Losses (Standalone Harness & TRADES)

# %%
def run_pgd_kl_attack(
    model: nn.Module,
    x: torch.Tensor,
    eps: float = EPS_NORM,
    steps: int = 4,
    alpha: float = 0.05875,
) -> torch.Tensor:
    """Canonical 4-step TRADES inner PGD attack."""
    was_training = model.training
    model.eval()
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, x - eps, x + eps).detach()

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(x.device.type == "cuda")):
            p_clean = F.softmax(model(x), dim=-1)

    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.amp.autocast("cuda", enabled=(x.device.type == "cuda")):
            out_adv = model(x_adv)
            loss_kl = F.kl_div(F.log_softmax(out_adv, dim=-1), p_clean, reduction="batchmean")
        grad = torch.autograd.grad(loss_kl, x_adv)[0]
        x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
        x_adv = torch.clamp(x_adv, x - eps, x + eps).detach()

    if was_training:
        model.train()
    return x_adv.detach()

def run_pgd_ce_monitor(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = EPS_NORM,
    steps: int = 50,
    alpha: Optional[float] = None,
) -> torch.Tensor:
    """Canonical PGD-CE attack for evaluation."""
    if alpha is None:
        alpha = 2.5 * eps / steps
    was_training = model.training
    model.eval()
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, x - eps, x + eps).detach()

    for _ in range(steps):
        x_adv.requires_grad_()
        with torch.amp.autocast("cuda", enabled=(x.device.type == "cuda")):
            out = model(x_adv)
            loss = F.cross_entropy(out, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
        x_adv = torch.clamp(x_adv, x - eps, x + eps).detach()

    if was_training:
        model.train()
    return x_adv.detach()

def trades_loss(
    model: nn.Module,
    x_clean: torch.Tensor,
    y: torch.Tensor,
    beta: float = 2.5,
    eps: float = EPS_NORM,
    steps: int = 4,
    alpha: float = 0.05875,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TRADES loss: L_natural + beta * L_robust_kl."""
    out_clean = model(x_clean)
    loss_nat = F.cross_entropy(out_clean, y)

    x_adv = run_pgd_kl_attack(model, x_clean, eps=eps, steps=steps, alpha=alpha)
    model.train()
    out_adv = model(x_adv)
    loss_kl = F.kl_div(
        F.log_softmax(out_adv, dim=-1),
        F.softmax(out_clean.detach(), dim=-1),
        reduction="batchmean",
    )
    loss_total = loss_nat + beta * loss_kl
    return loss_total, loss_nat.detach(), loss_kl.detach()

# %% [markdown]
# ## Step 7: Data Loaders & Dataset Hash

# %%
def get_loaders(
    data_root: Path,
    img_size: int = 112,
    batch_size: int = 48,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    tf_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_ds = datasets.ImageFolder(str(data_root / "val"), transform=tf_val)
    train_ds = datasets.ImageFolder(str(data_root / "train"), transform=tf_train)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    g_mon = torch.Generator().manual_seed(12345)
    mon_indices = torch.randperm(len(val_ds), generator=g_mon)[:500].tolist()
    monitor_loader = DataLoader(
        Subset(val_ds, mon_indices), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, monitor_loader

# %% [markdown]
# ## Step 8: Training Function for the Consolidated Flagship

# %%
from training.train_gen2_foundation import run_gen2_phase

def train_consolidated_seed(seed: int) -> Path:
    banner(f"TRAINING CONSOLIDATED FLAGSHIP -- SEED {seed}")
    final_ckpt = CKPT_DIR / f"foundation_recurrence_112_seed{seed}_final.pth"
    rolling_ckpt = CKPT_DIR / f"foundation_recurrence_112_seed{seed}_rolling.pth"

    if final_ckpt.exists():
        print(f"Final checkpoint already exists at {final_ckpt}. Skipping training.")
        return final_ckpt

    if rolling_ckpt.exists():
        try:
            st = torch.load(rolling_ckpt, map_location="cpu", weights_only=False)
            ep = st.get("epoch", 0) + 1
            print(f"\n>>> RESUME: rolling checkpoint found at {rolling_ckpt}")
            print(f"    run_gen2_phase will resume from epoch {ep} onwards.")
        except Exception as e:
            print(f"\n>>> RESUME: rolling checkpoint found at {rolling_ckpt} (inspect notice: {e})")
    else:
        print(f"\n>>> FRESH START: no rolling checkpoint for seed {seed}. Training from epoch 1.")

    ensure_dino_weights(DINO_PATH)

    # Invokes the exact canonical ff_112 training engine unchanged,
    # swapping the model class to Gen2FoundationModel(phase="recurrence_only", fovea_size=112)
    # with DINOv2 warm-start and the active abort assertions.
    eff_batch = int(os.environ.get("CONSOLIDATED_BATCH", PROTOCOL.get("effective_batch_size", 48)))
    run_gen2_phase(
        phase="consolidated_112",
        arm="v1",
        data_root=str(DATA_ROOT),
        ckpt_dir=str(CKPT_DIR),
        report_dir=str(OUT_DIR),
        dino_path=DINO_PATH,
        batch_size=eff_batch,
        num_workers=2,
        smoke=False,
        n_epochs=PROTOCOL["n_epochs"],
        training_seed=seed,
        phase_override_name=f"recurrence_112_seed{seed}",
        epoch_callback=lambda p, ep: sync_to_hf(),
    )

    assert final_ckpt.exists(), f"Expected final checkpoint at {final_ckpt}"
    print(f"Verified final checkpoint for seed {seed} at {final_ckpt}")
    sync_to_hf()
    return final_ckpt

# %% [markdown]
# ## Step 9: Full 5,000-Image Evaluation Battery

# %%
class _NormWrap(nn.Module):
    """Re-normalise inside forward so torchattacks can work in [0,1] pixel space."""
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        m = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        s = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("_m", m)
        self.register_buffer("_s", s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x in [0,1]
        return self.base((x - self._m.to(x)) / self._s.to(x))


def full_5k_battery(model: nn.Module, label: str) -> Dict[str, Any]:
    """Clean + PGD-50 + APGD-CE + APGD-DLR on all 5,000 validation images."""
    banner(f"FULL 5k EVALUATION -- {label}")
    # Two transforms: normalised (for PGD/clean) and raw [0,1] (for torchattacks)
    tf_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tf_val_01 = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),  # [0,1] -- no normalise; NormWrap handles it
    ])
    val_ds = datasets.ImageFolder(VAL_ROOT, transform=tf_val)
    val_ds_01 = datasets.ImageFolder(VAL_ROOT, transform=tf_val_01)
    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    loader_01 = DataLoader(val_ds_01, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # NormWrap so torchattacks applies eps in [0,1] pixel domain
    model_01 = _NormWrap(model).to(DEVICE)
    model_01.eval()
    apgd_ce = torchattacks.APGD(model_01, norm="Linf", eps=EPS_PIXEL, steps=100, n_restarts=1, loss="ce", verbose=False)
    apgd_dlr = torchattacks.APGD(model_01, norm="Linf", eps=EPS_PIXEL, steps=100, n_restarts=1, loss="dlr", verbose=False)

    preds_dict: Dict[str, List[np.ndarray]] = {"clean": [], "pgd50": [], "apgd_ce": [], "apgd_dlr": []}
    corr_dict: Dict[str, List[np.ndarray]] = {"clean": [], "pgd50": [], "apgd_ce": [], "apgd_dlr": []}

    model.eval()
    loader_01_iter = iter(loader_01)
    for b_idx, (x, y) in enumerate(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        x_01, _ = next(loader_01_iter)  # unnormalised [0,1] for torchattacks
        x_01 = x_01.to(DEVICE)
        y_np = y.cpu().numpy()

        # 1. Clean (use normalised x)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                p_clean = model(x).argmax(dim=-1).cpu().numpy()
        preds_dict["clean"].append(p_clean)
        corr_dict["clean"].append((p_clean == y_np).astype(np.int8))

        # 2. PGD-50 (normalised domain, custom loop handles clamp correctly)
        x_pgd50 = run_pgd_ce_monitor(model, x, y, eps=EPS_NORM, steps=50)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                p_pgd50 = model(x_pgd50).argmax(dim=-1).cpu().numpy()
        preds_dict["pgd50"].append(p_pgd50)
        corr_dict["pgd50"].append((p_pgd50 == y_np).astype(np.int8))

        # 3. APGD-CE (torchattacks, [0,1] domain via NormWrap, eps=EPS_PIXEL)
        x_apgd_ce = apgd_ce(x_01, y)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                p_apgd_ce = model_01(x_apgd_ce).argmax(dim=-1).cpu().numpy()
        preds_dict["apgd_ce"].append(p_apgd_ce)
        corr_dict["apgd_ce"].append((p_apgd_ce == y_np).astype(np.int8))

        # 4. APGD-DLR (torchattacks, [0,1] domain via NormWrap, eps=EPS_PIXEL)
        x_apgd_dlr = apgd_dlr(x_01, y)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                p_apgd_dlr = model_01(x_apgd_dlr).argmax(dim=-1).cpu().numpy()
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

# %% [markdown]
# ## Step 10: Statistical Synthesis & Decision Rule

# %%
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

# %% [markdown]
# ## Step 11: Main Orchestration & HuggingFace Dual-Upload

# %%
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

    # 7. Push to HuggingFace (dual upload: archive and canonical top-level)
    hf_token_val = os.environ.get("HF_TOKEN")
    if hf_token_val:
        banner("UPLOADING TO HUGGINGFACE")
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token_val)
        repo_id = "FerrariKazu/rhan-nxa-g2-checkpoints"
        _ts_up = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"closure/consolidated_112/{_ts_up}"

        api.upload_file(path_or_fileobj=str(report_path), path_in_repo=f"{prefix}/report.json", repo_id=repo_id)
        api.upload_file(path_or_fileobj=str(bundle_path), path_in_repo=f"{prefix}/per_sample_bundle.npz", repo_id=repo_id)
        for s in PROTOCOL["seeds"]:
            ckpt_path = str(flagship_ckpts[s])
            # Archive copy (timestamped, for provenance)
            api.upload_file(
                path_or_fileobj=ckpt_path,
                path_in_repo=f"{prefix}/foundation_recurrence_112_seed{s}_final.pth",
                repo_id=repo_id,
            )
            # Canonical top-level copy (required for OPAL gate in colab_j2_eval_train.py)
            api.upload_file(
                path_or_fileobj=ckpt_path,
                path_in_repo=f"foundation_recurrence_112_seed{s}_final.pth",
                repo_id=repo_id,
            )
            print(f"  Uploaded flagship seed {s} (archive + top-level canonical)")
        print("Upload complete!")

    banner("RUN COMPLETE")


if __name__ == "__main__":
    main()
