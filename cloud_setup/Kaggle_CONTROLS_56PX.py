#!/usr/bin/env python3
"""
Kaggle Notebook -- Recipe-Matched 56-px Controls  (~10-12 GPU-h total)
=======================================================================

OBJECTIVE
---------
Before the recurrence claim can stand, two epoch-matched feedforward
controls at the *same* resolution (56 px) and *same* threat radius
(Rung 3, eps=0.094) must be evaluated.

  CONTROL 1  (g2_gist_only_cont)  ~4-5 GPU-h on T4
  ---------------------------------------------------
  Continue g2_gist_only training for 15 more epochs at Rung 3
  (eps=0.094, beta=2.5, PGD-4), inheriting the existing
  foundation_g2_gist_only_final.pth checkpoint.
  This is the EPOCH-MATCHED feedforward baseline for recurrence_only.

  CONTROL 2  (ff_56)  ~5-6 GPU-h on T4
  ---------------------------------------
  Train a fresh 56-px feedforward model from DINOv2 warm-start,
  directly at Rung 3 (eps=0.094) for 15 epochs — no clean pre-training.
  This isolates whether the gap is "curriculum" (g2_gist_only_cont) or
  "direct Rung-3 start" (ff_56).

  EVALUATION  (reconciled harness)
  ---------------------------------
  After training, both controls are evaluated with the corrected attack
  harness (NormalizedModel wrapper, pixel-space attacks, eps = 5.417/255)
  on the shared 1,000-image slice (seed=42) used in the reconciliation.

DECISION RULE (preregistered)
------------------------------
  Recurrence earns a robustness claim IFF:
    recurrence_only worst-case > BEST(g2_gist_only_cont, ff_56) worst-case + 2σ
  where σ is the ff_112 inter-seed standard deviation.

USAGE ON KAGGLE
---------------
  1. Create a new Kaggle Notebook.
  2. Accelerator: GPU T4 (1x), Internet: ON, runtime: ~12 h.
  3. Add-ons > Secrets > Add 'HF_TOKEN' (HuggingFace write token).
  4. To run only one control:
       import os; os.environ["CONTROLS_JOBS"] = "C1"     # or "C2" or "C1,C2"
  5. Paste and execute. Rolling checkpoints sync to HF after every epoch.
"""

import os
import sys
import subprocess
import shutil
import time
import json
import datetime
import numpy as np
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
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
else:
    print("WARNING: No GPU detected. Attach T4 in Notebook Settings.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Install missing libraries
for pkg in ["huggingface_hub", "torchattacks", "scipy"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.run(f"pip install -q {pkg}", shell=True, check=True)

import torchattacks
from scipy.stats import chi2

# Jobs to run (default both controls)
CONTROLS_JOBS = set(j.strip().upper() for j in
                    os.environ.get("CONTROLS_JOBS", "C1,C2").split(","))
print(f"Controls scheduled: {sorted(CONTROLS_JOBS)}")

# ── 2. Clone & Checkout feature/gen2-rebuild ──────────────────────────────────
REPO_NAME = "Adversarial-Cognitive-Model"
WORK_DIR  = f"/kaggle/working/{REPO_NAME}"
BRANCH    = "feature/gen2-rebuild"

if os.path.exists("/kaggle/working"):
    os.chdir("/kaggle/working")
    if not os.path.exists(WORK_DIR):
        print(f"Cloning FerrariKazu/{REPO_NAME}...")
        subprocess.run(f"git clone https://github.com/FerrariKazu/{REPO_NAME}.git",
                       shell=True, check=True)
    os.chdir(WORK_DIR)
    sys.path.insert(0, WORK_DIR)
    os.environ["PYTHONPATH"] = f"{WORK_DIR}:{os.environ.get('PYTHONPATH', '')}"
    subprocess.run("git fetch origin", shell=True, check=True)
    subprocess.run(f"git checkout -B {BRANCH} origin/{BRANCH}", shell=True, check=True)
    subprocess.run(f"git reset --hard origin/{BRANCH}", shell=True, check=True)
    sha = subprocess.check_output("git rev-parse --short HEAD", shell=True, text=True).strip()
    print(f"Checked out {BRANCH} @ {sha}")
else:
    print(f"Running locally in {os.getcwd()}")
    sha = subprocess.check_output("git rev-parse --short HEAD", shell=True, text=True).strip()

for mod in list(sys.modules.keys()):
    if any(mod.startswith(p) for p in ["training", "gen2_foundation", "noesis_vision"]):
        del sys.modules[mod]

# ── 3. HuggingFace Repos ──────────────────────────────────────────────────────
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

HF_BEST    = "FerrariKazu/rhan-nxa-g2-checkpoints"
HF_ROLLING = "FerrariKazu/rhan-nxa-g2-checkpoints-rolling"
api = HfApi(token=hf_token)

for r in [HF_BEST, HF_ROLLING]:
    try:
        api.create_repo(repo_id=r, repo_type="dataset", private=True, exist_ok=True)
        print(f"HF repo ready: {r}")
    except Exception as e:
        print(f"  Note: {r}: {e}")

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("report/controls", exist_ok=True)


def restore_checkpoints_from_hf():
    restored = 0
    for repo_id in [HF_BEST, HF_ROLLING]:
        try:
            files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=hf_token)
            for f in files:
                if f.endswith(".pth") or f.endswith(".zip"):
                    local_f = os.path.join("checkpoints", os.path.basename(f))
                    if not os.path.exists(local_f):
                        print(f"  Restoring {os.path.basename(f)} from {repo_id}...")
                        p = hf_hub_download(repo_id=repo_id, filename=f,
                                            repo_type="dataset", token=hf_token)
                        shutil.copyfile(p, local_f)
                        restored += 1
        except Exception as e:
            print(f"  Note: {repo_id}: {e}")
    print(f"Restored {restored} checkpoint file(s) from HF.")


def sync_to_hf():
    for cp in list(Path("checkpoints").glob("*_rolling.pth")):
        try:
            api.upload_file(path_or_fileobj=str(cp), path_in_repo=cp.name,
                            repo_id=HF_ROLLING, repo_type="dataset")
        except Exception as e:
            print(f"  Warning: rolling sync {cp.name}: {e}")
    for cp in (list(Path("checkpoints").glob("*_final.pth")) +
               list(Path("checkpoints").glob("*_best_*.pth"))):
        try:
            api.upload_file(path_or_fileobj=str(cp), path_in_repo=cp.name,
                            repo_id=HF_BEST, repo_type="dataset")
        except Exception as e:
            print(f"  Warning: best sync {cp.name}: {e}")
    for jf in (list(Path("report/controls").glob("*.json")) +
               list(Path("report/controls").glob("*.jsonl"))):
        try:
            api.upload_file(path_or_fileobj=str(jf),
                            path_in_repo=f"controls/{jf.name}",
                            repo_id=HF_BEST, repo_type="dataset")
        except Exception as e:
            print(f"  Warning: report sync {jf.name}: {e}")


restore_checkpoints_from_hf()

# ── 4. DINOv2 & ImageNet-100 ─────────────────────────────────────────────────
DINO_PATH = "checkpoints/dinov2_vits14_pretrain.pth"
DINO_URL  = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
if not os.path.exists(DINO_PATH) or os.path.getsize(DINO_PATH) < 80 * 1024 * 1024:
    print("Downloading DINOv2-small weights (~88 MB)...")
    subprocess.run(f"wget -q -c {DINO_URL} -O {DINO_PATH}", shell=True, check=True)
print(f"DINOv2 checkpoint ready ({os.path.getsize(DINO_PATH)/1e6:.1f} MB)")

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
    subprocess.run(f"python3 scripts/prepare_imagenet100.py --root {DATA_ROOT}",
                   shell=True, check=True)
    assert _is_dataset_ready(DATA_ROOT), f"Bootstrap failed at {DATA_ROOT}!"
print(f"ImageNet-100 ready at {DATA_ROOT}")

BATCH_SIZE = int(os.environ.get("CONTROLS_BATCH", "48"))

# ── 5. Import training infrastructure ────────────────────────────────────────
from training.train_gen2_foundation import run_gen2_phase
from gen2_foundation.model import Gen2FoundationModel
from training.curriculum_gen2 import IMAGENET_MEAN, IMAGENET_STD

# ── 6. CONTROL 1 — g2_gist_only_cont ─────────────────────────────────────────
# Epoch-matched continuation of g2_gist_only at Rung 3 (eps=0.094)
# Inherits foundation_g2_gist_only_final.pth and trains 15 more epochs.

if "C1" not in CONTROLS_JOBS:
    print("\nCONTROL 1 (g2_gist_only_cont) skipped (not in CONTROLS_JOBS).")
else:
    C1_FINAL = "checkpoints/foundation_g2_gist_only_cont_final.pth"
    if os.path.exists(C1_FINAL):
        print(f"\nCONTROL 1 already done ({C1_FINAL}). Skipping to eval.")
    else:
        parent = "checkpoints/foundation_g2_gist_only_final.pth"
        if not os.path.exists(parent):
            raise RuntimeError(
                f"CONTROL 1 requires {parent} to exist.\n"
                f"Run Kaggle_J2_EVAL_TRAIN.py Job A first (or restore from HF)."
            )
        print(f"\n{'='*70}")
        print(f"CONTROL 1: g2_gist_only_cont  (56px, Rung 3, eps=0.094, 15 epochs)")
        print(f"  Inheriting: {parent}")
        print(f"{'='*70}")
        try:
            run_gen2_phase(
                phase="g2_gist_only_cont",
                arm="v1",
                data_root=DATA_ROOT,
                ckpt_dir="checkpoints",
                report_dir="report/controls",
                dino_path=DINO_PATH,
                batch_size=BATCH_SIZE,
                num_workers=4,
                smoke=False,
                img_size=56,
                n_epochs=15,
                epoch_callback=lambda p, ep: sync_to_hf(),
            )
            print("CONTROL 1 training complete.")
        except TypeError as e:
            print(f"  run_gen2_phase signature mismatch ({e}); CLI fallback.")
            subprocess.run(
                f"python3 training/train_gen2_foundation.py "
                f"--phase g2_gist_only_cont "
                f"--arm v1 "
                f"--data-root {DATA_ROOT} "
                f"--ckpt-dir checkpoints "
                f"--report-dir report/controls "
                f"--dino-path {DINO_PATH} "
                f"--batch-size {BATCH_SIZE} "
                f"--img-size 56 "
                f"--n-epochs 15",
                shell=True, check=True,
            )
        sync_to_hf()

# ── 7. CONTROL 2 — ff_56 (fresh DINOv2 start, Rung 3 directly) ───────────────
# Trains a 56-px feedforward from DINOv2 warm-start directly at Rung 3.
# No clean pre-training — tests whether the curriculum matters.

if "C2" not in CONTROLS_JOBS:
    print("\nCONTROL 2 (ff_56) skipped (not in CONTROLS_JOBS).")
else:
    C2_FINAL = "checkpoints/foundation_ff_56_final.pth"
    if os.path.exists(C2_FINAL):
        print(f"\nCONTROL 2 already done ({C2_FINAL}). Skipping to eval.")
    else:
        print(f"\n{'='*70}")
        print(f"CONTROL 2: ff_56  (56px, DINOv2 warm-start, Rung 3, eps=0.094, 15 epochs)")
        print(f"{'='*70}")
        try:
            run_gen2_phase(
                phase="ff_56",
                arm="v1",
                data_root=DATA_ROOT,
                ckpt_dir="checkpoints",
                report_dir="report/controls",
                dino_path=DINO_PATH,
                batch_size=BATCH_SIZE,
                num_workers=4,
                smoke=False,
                img_size=56,
                n_epochs=15,
                epoch_callback=lambda p, ep: sync_to_hf(),
            )
            print("CONTROL 2 training complete.")
        except TypeError as e:
            print(f"  run_gen2_phase signature mismatch ({e}); CLI fallback.")
            subprocess.run(
                f"python3 training/train_gen2_foundation.py "
                f"--phase ff_56 "
                f"--arm v1 "
                f"--data-root {DATA_ROOT} "
                f"--ckpt-dir checkpoints "
                f"--report-dir report/controls "
                f"--dino-path {DINO_PATH} "
                f"--batch-size {BATCH_SIZE} "
                f"--img-size 56 "
                f"--n-epochs 15",
                shell=True, check=True,
            )
        sync_to_hf()

# ── 8. Corrected Evaluation Harness ──────────────────────────────────────────
# Evaluate trained controls with the same NormalizedModel pixel-space harness
# as Kaggle_ATTACK_RECONCILIATION.py, on the identical 1,000-image slice.

print(f"\n{'='*70}")
print("EVALUATING CONTROLS WITH CORRECTED ATTACK HARNESS")
print(f"{'='*70}")

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

EVAL_SEED  = 42
EVAL_N     = 1000
EVAL_BATCH = 32
EPS_PIXEL  = 5.417 / 255.0   # target budget in [0, 1]


class NormalizedModel(nn.Module):
    """Wrap model: attacks stay in [0,1] pixel space, norm happens inside."""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return self.base_model((x - self.mean) / self.std)


def build_raw_loader(img_size):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # [0, 1], NO Normalize
    ])
    val_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"), transform=tf)
    g = torch.Generator().manual_seed(EVAL_SEED)
    idx = torch.randperm(len(val_ds), generator=g)[:EVAL_N].tolist()
    return DataLoader(Subset(val_ds, idx), batch_size=EVAL_BATCH, shuffle=False, num_workers=2)


def run_pgd_pixel(model, x, y, eps, steps=50, restarts=1, alpha=None):
    alpha = alpha or (2.5 * eps / max(steps, 1))
    best_loss = torch.full((x.size(0),), -1e9, device=x.device)
    best_adv  = x.clone()
    for _ in range(restarts):
        x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0.0, 1.0)
        for _ in range(steps):
            x_adv.requires_grad_(True)
            with torch.enable_grad():
                loss = F.cross_entropy(model(x_adv), y, reduction="none")
            grad = torch.autograd.grad(loss.sum(), x_adv)[0]
            with torch.no_grad():
                x_adv = (x_adv.detach() + alpha * grad.sign())
                x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0.0, 1.0)
        with torch.no_grad():
            fl = F.cross_entropy(model(x_adv), y, reduction="none")
        mask = fl > best_loss
        best_loss[mask] = fl[mask]
        best_adv[mask]  = x_adv[mask]
    return best_adv


# Pairs: (label, phase, img_size, ckpt_path)
EVAL_TARGETS = []
for label, phase, img_size, ckpt in [
    ("g2_gist_only_cont", "g2_gist_only", 56, "checkpoints/foundation_g2_gist_only_cont_final.pth"),
    ("ff_56",             "ff_56",         56, "checkpoints/foundation_ff_56_final.pth"),
    # Include recurrence_only for direct comparison
    ("recurrence_only",   "recurrence_only", 56, "checkpoints/foundation_recurrence_only_final.pth"),
]:
    if os.path.exists(ckpt):
        EVAL_TARGETS.append((label, phase, img_size, ckpt))
    else:
        print(f"  SKIP eval {label}: {ckpt} not found.")

eval_results = {}

for label, phase, img_size, ckpt in EVAL_TARGETS:
    print(f"\n--- Evaluating: {label} ({img_size}px) ---")
    base = Gen2FoundationModel(phase=phase, arm="v1", fovea_size=img_size).to(device)
    sd = torch.load(ckpt, map_location=device)
    base.load_state_dict(sd.get("model", sd), strict=False)
    base.eval()

    wrapped = NormalizedModel(base).to(device)
    wrapped.eval()

    loader = build_raw_loader(img_size)
    apgd_ce  = torchattacks.APGD(wrapped, norm="Linf", eps=EPS_PIXEL, steps=100,
                                  n_restarts=1, loss="ce", verbose=False)
    apgd_dlr = torchattacks.APGD(wrapped, norm="Linf", eps=EPS_PIXEL, steps=100,
                                  n_restarts=1, loss="dlr", verbose=False)

    outcomes = {}
    for atk_name in ["clean", "pgd50", "apgd_ce", "apgd_dlr"]:
        preds_list, labels_list = [], []
        max_linf = 0.0
        print(f"  {atk_name:<16}", end="", flush=True)
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if atk_name == "clean":
                x_adv = x
            elif atk_name == "pgd50":
                x_adv = run_pgd_pixel(wrapped, x, y, eps=EPS_PIXEL, steps=50)
            elif atk_name == "apgd_ce":
                x_adv = apgd_ce(x, y)
            elif atk_name == "apgd_dlr":
                x_adv = apgd_dlr(x, y)

            if atk_name != "clean":
                bl = (x_adv - x).abs().max().item()
                assert bl <= EPS_PIXEL + 1e-4, f"L_inf violation: {bl*255:.3f}/255"
                max_linf = max(max_linf, bl)

            with torch.no_grad():
                preds = wrapped(x_adv).argmax(-1)
            preds_list.extend(preds.cpu().numpy().tolist())
            labels_list.extend(y.cpu().numpy().tolist())

        corr = np.array(preds_list) == np.array(labels_list)
        acc  = corr.mean() * 100.0
        outcomes[atk_name] = corr.astype(np.int8)
        linf_str = f"  L_inf={max_linf*255:.3f}/255" if atk_name != "clean" else ""
        print(f"  Acc={acc:.2f}%{linf_str}  ({time.time()-t0:.1f}s)")

    # Worst-case across all attacks
    adv_corr = [v for k, v in outcomes.items() if k != "clean"]
    worst = np.bitwise_and.reduce(adv_corr)
    print(f"  {'worst_case':<16}  Acc={worst.mean()*100:.2f}%")

    eval_results[label] = {
        "clean_acc":   float(outcomes["clean"].mean()),
        "pgd50_acc":   float(outcomes["pgd50"].mean()),
        "apgd_ce_acc": float(outcomes["apgd_ce"].mean()),
        "apgd_dlr_acc": float(outcomes["apgd_dlr"].mean()),
        "worst_case_acc": float(worst.mean()),
    }

# ── 9. Summary Table ─────────────────────────────────────────────────────────
print(f"\n{'='*75}")
print(f"  56-PX CONTROLS EVALUATION  (eps={EPS_PIXEL*255:.2f}/255, n={EVAL_N}, seed={EVAL_SEED})")
print(f"{'='*75}")
print(f"  {'Model':<22} {'Clean':>7} {'PGD-50':>8} {'APGD-CE':>9} {'APGD-DLR':>10} {'Worst':>7}")
print(f"  {'-'*22} {'-'*7} {'-'*8} {'-'*9} {'-'*10} {'-'*7}")
for lbl, r in eval_results.items():
    print(f"  {lbl:<22} "
          f"{r['clean_acc']*100:>6.2f}% "
          f"{r['pgd50_acc']*100:>7.2f}% "
          f"{r['apgd_ce_acc']*100:>8.2f}% "
          f"{r['apgd_dlr_acc']*100:>9.2f}% "
          f"{r['worst_case_acc']*100:>6.2f}%")

# ── 10. Decision Rule Evaluation (requires 2σ bar from ff_112 seeds) ──────────
# 2σ bar must be loaded from the reconciliation results (ff_112 seed variance).
# For now, print a reminder and record the raw numbers.

print("\n[Decision Rule Check]")
if "recurrence_only" in eval_results and eval_results:
    rec_worst = eval_results.get("recurrence_only", {}).get("worst_case_acc", None)
    best_ctrl = max(
        (v["worst_case_acc"] for k, v in eval_results.items() if k != "recurrence_only"),
        default=None
    )
    if rec_worst is not None and best_ctrl is not None:
        gap_pp = (rec_worst - best_ctrl) * 100.0
        print(f"  recurrence_only worst-case: {rec_worst*100:.2f}%")
        print(f"  best 56px control worst-case: {best_ctrl*100:.2f}%")
        print(f"  Gap: {gap_pp:+.2f} pp")
        print(f"  NOTE: 2σ bar requires ff_112 inter-seed σ from reconciliation results.")
        print(f"        Load reconciliation JSON and compare gap vs 2σ before drawing conclusion.")
    else:
        print("  Insufficient data to apply decision rule.")

# ── 11. Save & Upload ─────────────────────────────────────────────────────────
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_json = f"report/controls/controls_eval_{ts}.json"
with open(out_json, "w") as f:
    json.dump({
        "timestamp": ts,
        "eval_seed": EVAL_SEED,
        "eval_n": EVAL_N,
        "eps_pixel_255": round(EPS_PIXEL * 255.0, 4),
        "controls_jobs": sorted(CONTROLS_JOBS),
        "results": eval_results,
    }, f, indent=2)
print(f"\nSaved eval results to {out_json}")
sync_to_hf()
print("Controls run complete.")
