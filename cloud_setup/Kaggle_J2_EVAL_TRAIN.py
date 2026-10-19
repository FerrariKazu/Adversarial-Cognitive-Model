#!/usr/bin/env python3
"""
Kaggle Notebook -- RHAN-NXA J2: Matched-eps Eval + 56px FF Control + ff_112 Seed Replication
==============================================================================================

CONTEXT & OBJECTIVES
--------------------
This notebook is the Kaggle dispatch script for the Gen-2 evidence-gathering sequence.
It provides the exact same functionality as cloud_setup/colab_j2_eval_train.py adapted
for the Kaggle environment (/kaggle/working, Kaggle Secrets, /kaggle/tmp).

Three independent jobs are controlled via the GEN2_JOBS environment variable:

  JOB A -- Matched-epsilon Cross-Checkpoint Evaluation (Eval-only, ~1.5-2 GPU-h on T4/P100):
      Re-evaluates all four checkpoints on an identical 1,000-image slice with all headline
      attacks at eps_norm=0.094 (training threat radius, approx 5.4/255).
      Checkpoints:
        - g2_gist_only    (56px, TRADES-PGD4)
        - recurrence_only (56px, T=4 recurrence)
        - gen1_core       (56px, full Gen-2 core)
        - ff_112          (112px, 64 tokens, feedforward baseline)
      Attacks (fp32):
        - PGD-50          (alpha = 2.5*eps/50, random start)
        - APGD-CE-100     (torchattacks, 1 restart)
        - APGD-T-100      (torchattacks, 1 restart, 10 target classes)
      Headline: Per-sample worst-case robust accuracy (union of fooled sets).
      Sanity checks for ff_112: APGD-CE at 8/255, 16/255, and 32/255.
      Statistical significance: Paired two-sided McNemar tests for every pair.

  JOB B -- 56-px Feedforward Control Training (ff_56, ~1-1.5 GPU-h):
      Trains single-pass trunk at 56 px under corrected TRADES-PGD4 recipe for 15 epochs.
      Isolates resolution gain (56px -> 112px) from the attack recipe correction.

  JOB C -- Two Additional ff_112 Training Seeds (Seeds 42 & 99, ~9.4 GPU-h total):
      Trains seeds 42 and 99 (seed 17 was the original ff_112) for 15 epochs each.
      Sets the 2-sigma variance bar required by Gate B for all future Gen-3 arms.

USAGE ON KAGGLE:
----------------
  1. Create a new Kaggle Notebook.
  2. Accelerator: GPU T4 or P100 (or T4 x2), Internet: ON.
  3. Add-ons > Secrets > Add secret 'HF_TOKEN' (HuggingFace write token).
  4. In the first code cell, optionally configure GEN2_JOBS:
       import os; os.environ["GEN2_JOBS"] = "A"      # Run eval only (~2h)
       # or os.environ["GEN2_JOBS"] = "A,B,C"        # Run all jobs
  5. Paste and run this script. Rolling checkpoints sync to HF after every epoch.
     If the session expires or disconnects, re-running automatically resumes.
"""

# %% [markdown]
# ## Step 1: Environment, Dependencies & HuggingFace Secrets

# %%
import os
import sys
import subprocess
import shutil
import time
import json
from pathlib import Path

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# HuggingFace token from Kaggle Secrets (or environment)
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

# GPU Accelerator Verification
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Accelerator Active: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
else:
    print("WARNING: No GPU detected! Attach a GPU accelerator (T4 or P100) in Notebook Settings.")

# Install dependencies if missing
for mod, pkg in [
    ("huggingface_hub", "huggingface_hub"),
    ("pyarrow", "pyarrow"),
    ("torchattacks", "torchattacks"),
    ("scipy", "scipy"),
]:
    try:
        __import__(mod)
    except ImportError:
        subprocess.run(["pip", "install", "--quiet", pkg], check=True)

# Scheduled jobs (default: A, B, C)
GEN2_JOBS = set(j.strip().upper() for j in os.environ.get("GEN2_JOBS", "A,B,C").split(","))
print(f"Jobs scheduled: {sorted(GEN2_JOBS)}")

# %% [markdown]
# ## Step 2: Clone & Checkout branch feature/gen2-rebuild

# %%
REPO_NAME = "Adversarial-Cognitive-Model"
WORK_DIR  = f"/kaggle/working/{REPO_NAME}"
BRANCH    = "feature/gen2-rebuild"

os.chdir("/kaggle/working")
if not os.path.exists(WORK_DIR):
    print(f"Cloning FerrariKazu/{REPO_NAME}...")
    subprocess.run(f"git clone https://github.com/FerrariKazu/{REPO_NAME}.git",
                   shell=True, check=True)

os.chdir(WORK_DIR)
sys.path.insert(0, WORK_DIR)
os.environ["PYTHONPATH"] = f"{WORK_DIR}:{os.environ.get('PYTHONPATH', '')}"

print(f"Fetching and checking out branch '{BRANCH}'...")
subprocess.run("git fetch origin", shell=True, check=True)
subprocess.run(f"git checkout -B {BRANCH} origin/{BRANCH}", shell=True, check=True)
subprocess.run(f"git reset --hard origin/{BRANCH}", shell=True, check=True)

sha = subprocess.check_output("git rev-parse --short HEAD", shell=True, text=True).strip()
print(f"Checked out {BRANCH} @ commit {sha}")

# Evict any cached module references
for mod in list(sys.modules.keys()):
    if any(mod.startswith(p) for p in ["training", "gen2_foundation", "noesis_vision"]):
        del sys.modules[mod]

# %% [markdown]
# ## Step 3: DINOv2-small Checkpoint

# %%
DINO_PATH = "checkpoints/dinov2_vits14_pretrain.pth"
DINO_URL  = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("report/eval", exist_ok=True)
os.makedirs("report/audit_rev1", exist_ok=True)

if not os.path.exists(DINO_PATH) or os.path.getsize(DINO_PATH) < 80 * 1024 * 1024:
    print(f"Downloading DINOv2-small weights (~88 MB)...")
    subprocess.run(f"wget -q -c {DINO_URL} -O {DINO_PATH}", shell=True, check=True)
print(f"DINOv2 checkpoint ready ({os.path.getsize(DINO_PATH)/1e6:.1f} MB)")

# %% [markdown]
# ## Step 4: HuggingFace Checkpoint Sync & Restore

# %%
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


def restore_checkpoints_from_hf():
    """Download checkpoints, audit reports, and logs from HuggingFace."""
    restored = 0
    for repo_id in [HF_BEST, HF_ROLLING]:
        try:
            files = list_repo_files(repo_id=repo_id, repo_type="dataset", token=hf_token)
            for f in files:
                local_f = None
                if f.endswith(".pth") or f.endswith(".zip"):
                    local_f = os.path.join("checkpoints", os.path.basename(f))
                elif f.endswith(".jsonl"):
                    local_f = os.path.join("report", os.path.basename(f))
                elif f.endswith(".json"):
                    bn = os.path.basename(f)
                    if bn.startswith("eval_"):
                        local_f = os.path.join("report/eval", bn)
                    elif bn.startswith("joba_") or bn.startswith("step"):
                        local_f = os.path.join("report/audit_rev1", bn)
                if local_f and not os.path.exists(local_f):
                    print(f"  Restoring {f} from {repo_id}...")
                    p = hf_hub_download(repo_id=repo_id, filename=f,
                                        repo_type="dataset", token=hf_token)
                    os.makedirs(os.path.dirname(local_f), exist_ok=True)
                    shutil.copyfile(p, local_f)
                    restored += 1
        except Exception as e:
            print(f"  Note: {repo_id}: {e}")
    print(f"Restored {restored} file(s) from HF.")


def sync_to_hf():
    """Upload checkpoints, eval JSONs, and JSONL logs to HuggingFace."""
    for cp in Path("checkpoints").glob("*_rolling.pth"):
        try:
            api.upload_file(path_or_fileobj=str(cp), path_in_repo=cp.name,
                            repo_id=HF_ROLLING, repo_type="dataset", token=hf_token)
        except Exception as e:
            print(f"  Warning: rolling sync {cp.name}: {e}")
    for cp in list(Path("checkpoints").glob("*_final.pth")) + \
               list(Path("checkpoints").glob("*_best_*.pth")):
        try:
            api.upload_file(path_or_fileobj=str(cp), path_in_repo=cp.name,
                            repo_id=HF_BEST, repo_type="dataset", token=hf_token)
        except Exception as e:
            print(f"  Warning: best sync {cp.name}: {e}")
    for jf in list(Path("report/eval").glob("*.json")) + \
               list(Path("report/audit_rev1").glob("*.json")):
        try:
            api.upload_file(path_or_fileobj=str(jf), path_in_repo=jf.name,
                            repo_id=HF_BEST, repo_type="dataset", token=hf_token)
        except Exception as e:
            print(f"  Warning: json sync {jf.name}: {e}")
    for log in Path("report").glob("*.jsonl"):
        try:
            api.upload_file(path_or_fileobj=str(log), path_in_repo=log.name,
                            repo_id=HF_ROLLING, repo_type="dataset", token=hf_token)
        except Exception as e:
            print(f"  Warning: log sync {log.name}: {e}")


restore_checkpoints_from_hf()

# %% [markdown]
# ## Step 5: ImageNet-100 Data Bootstrap (Kaggle /kaggle/tmp)

# %%
DATA_ROOT = os.environ.get("J2_DATA_ROOT", "/kaggle/tmp/imagenet100")
CONVERTER = "scripts/prepare_imagenet100.py"

def is_dataset_ready(root):
    td, vd = os.path.join(root, "train"), os.path.join(root, "val")
    return (os.path.isdir(td) and os.path.isdir(vd) and
            len(os.listdir(td)) == 100 and len(os.listdir(vd)) == 100)

if is_dataset_ready(DATA_ROOT):
    print(f"ImageNet-100 ready at {DATA_ROOT}.")
else:
    print(f"== Bootstrapping ImageNet-100 -> {DATA_ROOT} ==")
    os.makedirs(DATA_ROOT, exist_ok=True)
    subprocess.run(f"python3 {CONVERTER} --root {DATA_ROOT}", shell=True, check=True)
    assert is_dataset_ready(DATA_ROOT), "Dataset bootstrap failed!"
    print("ImageNet-100 bootstrap complete.")

# %% [markdown]
# ## JOB A: Matched-epsilon Cross-Checkpoint Evaluation
#
# Evaluates all four checkpoints on a shared 1,000-image slice.
# Headline attacks evaluated at eps_norm=0.094 (approx 5.4/255).
# ff_112 also evaluated at 8/255, 16/255 sanity, 32/255 sanity.
# Per-sample worst-case robust accuracy and paired McNemar tests.

# %%
if "A" not in GEN2_JOBS:
    print("JOB A skipped (not in GEN2_JOBS).")
else:
    import datetime
    import numpy as np
    from scipy.stats import chi2
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    for mod in list(sys.modules.keys()):
        if any(mod.startswith(p) for p in ["training", "gen2_foundation", "noesis_vision"]):
            del sys.modules[mod]

    from gen2_foundation.model import Gen2FoundationModel
    from training.curriculum_gen2 import IMAGENET_MEAN, IMAGENET_STD, MEAN_STD
    from training.train_gen2_foundation import run_pgd_ce_eval

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Constants ────────────────────────────────────────────────────────────
    EVAL_SEED   = 42
    EVAL_N      = 1000
    BATCH_SIZE  = 32
    EPS_NORM    = 0.094
    EPS_PIX     = EPS_NORM * MEAN_STD       # approx 0.0212, approx 5.4/255
    EPS_N_8     = (8.0  / 255.0) / MEAN_STD
    EPS_N_16    = (16.0 / 255.0) / MEAN_STD
    EPS_N_32    = (32.0 / 255.0) / MEAN_STD

    print("\n" + "=" * 70)
    print(f"JOB A  --  Matched-epsilon Evaluation  (n={EVAL_N}, seed={EVAL_SEED})")
    print(f"  train eps: norm={EPS_NORM}  pixel={EPS_PIX*255:.2f}/255")
    print("=" * 70)

    # ── Shared evaluation loader ─────────────────────────────────────────────
    def build_loader(img_size):
        tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        val_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"), transform=tf)
        g = torch.Generator().manual_seed(EVAL_SEED)
        idx = torch.randperm(len(val_ds), generator=g)[:EVAL_N].tolist()
        return DataLoader(Subset(val_ds, idx), batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2)

    def load_model(phase, img_size, ckpt_path):
        m = Gen2FoundationModel(phase=phase, arm="v1", fovea_size=img_size).to(device)
        sd = torch.load(ckpt_path, map_location=device)
        sd = sd.get("model", sd)
        m.load_state_dict(sd, strict=False)
        m.eval()
        return m

    def collect_outcomes(model, loader, eps_val, attack="pgd", pgd_steps=50) -> np.ndarray:
        """Return int8 array of shape (N,): 1=correct under attack, 0=fooled."""
        outcomes = []
        model.eval()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if attack == "clean":
                x_atk = x
            elif attack == "pgd":
                x_atk = run_pgd_ce_eval(model, x, y, eps=eps_val, steps=pgd_steps)
            elif attack in ("apgd_ce", "apgd_t"):
                import torchattacks
                if attack == "apgd_ce":
                    atk = torchattacks.APGD(model, norm="Linf", eps=eps_val,
                                            steps=100, n_restarts=1, loss="ce", verbose=False)
                else:
                    atk = torchattacks.APGDT(model, norm="Linf", eps=eps_val,
                                             steps=100, n_restarts=1, n_classes=10, verbose=False)
                x_atk = atk(x, y)
            else:
                raise ValueError(f"Unknown attack: {attack}")
            with torch.no_grad():
                preds = model(x_atk).argmax(-1)
            outcomes.extend((preds == y).cpu().numpy().tolist())
        return np.array(outcomes, dtype=np.int8)

    def mcnemar_p(a: np.ndarray, b: np.ndarray) -> float:
        """Two-sided McNemar p-value, continuity-corrected."""
        b01 = int(((a == 1) & (b == 0)).sum())
        b10 = int(((a == 0) & (b == 1)).sum())
        n = b01 + b10
        if n == 0:
            return 1.0
        stat = (abs(b01 - b10) - 1.0) ** 2 / n
        return float(1.0 - chi2.cdf(stat, df=1))

    # ── Checkpoint registry ──────────────────────────────────────────────────
    CHECKPOINTS = [
        ("g2_gist_only",    "g2_gist_only",    56,  "checkpoints/foundation_g2_gist_only_final.pth"),
        ("recurrence_only", "recurrence_only",  56,  "checkpoints/foundation_recurrence_only_final.pth"),
        ("gen1_core",       "gen1_core",        56,  "checkpoints/foundation_gen1_core_best_clean.pth"),
        ("ff_112",          "ff_112",           112, "checkpoints/foundation_ff_112_final.pth"),
    ]

    # Extract any zip-wrapped checkpoints if corresponding .pth is missing
    import zipfile
    for zpath in Path("checkpoints").glob("*.zip"):
        pname = zpath.stem + ".pth"
        ppath = zpath.parent / pname
        if not ppath.exists():
            print(f"Extracting {zpath.name} -> {pname}...")
            with zipfile.ZipFile(zpath) as zf:
                for m in zf.namelist():
                    if m.endswith(".pth"):
                        zf.extract(m, "checkpoints")
                        extracted = os.path.join("checkpoints", m)
                        if extracted != str(ppath):
                            shutil.move(extracted, str(ppath))
            print(f"Extracted {pname}")

    # ── Main evaluation loop ─────────────────────────────────────────────────
    all_results  = {}
    all_outcomes = {}

    for label, phase, img_size, ckpt_path in CHECKPOINTS:
        if not os.path.exists(ckpt_path):
            print(f"\n  SKIP {label}: checkpoint not found ({ckpt_path})")
            continue

        loader = build_loader(img_size)
        model  = load_model(phase, img_size, ckpt_path)
        print(f"\n{'─'*60}")
        print(f"  {label}  (img={img_size}px  ckpt={os.path.basename(ckpt_path)})")
        print(f"{'─'*60}")

        res      = {"phase": phase, "img_size": img_size, "checkpoint": ckpt_path}
        outcomes = {}

        # Clean
        c = collect_outcomes(model, loader, 0.0, attack="clean")
        res["clean"] = {"acc": float(c.mean()), "eps_norm": 0.0, "eps_pixel_255": 0.0}
        outcomes["clean"] = c
        print(f"    clean                          : {c.mean()*100:.2f}%  (n={len(c)})")

        # PGD-50 @ training eps (fp32, corrected alpha)
        p50 = collect_outcomes(model, loader, EPS_NORM, attack="pgd", pgd_steps=50)
        res["pgd50_train"] = {"acc": float(p50.mean()), "eps_norm": EPS_NORM,
                               "eps_pixel_255": round(EPS_PIX * 255, 3)}
        outcomes["pgd50"] = p50
        print(f"    PGD-50   @ {EPS_PIX*255:.2f}/255 (train eps): {p50.mean()*100:.2f}%")

        # APGD-CE-100 @ training eps
        ace = collect_outcomes(model, loader, EPS_NORM, attack="apgd_ce")
        if len(ace):
            res["apgd_ce_train"] = {"acc": float(ace.mean()), "eps_norm": EPS_NORM,
                                     "eps_pixel_255": round(EPS_PIX * 255, 3)}
            outcomes["apgd_ce"] = ace
            print(f"    APGD-CE  @ {EPS_PIX*255:.2f}/255 (train eps): {ace.mean()*100:.2f}%")

        # APGD-T-100 @ training eps
        at_ = collect_outcomes(model, loader, EPS_NORM, attack="apgd_t")
        if len(at_):
            res["apgd_t_train"] = {"acc": float(at_.mean()), "eps_norm": EPS_NORM,
                                    "eps_pixel_255": round(EPS_PIX * 255, 3)}
            outcomes["apgd_t"] = at_
            print(f"    APGD-T   @ {EPS_PIX*255:.2f}/255 (train eps): {at_.mean()*100:.2f}%")

        # Per-sample worst-case robust accuracy (union of fooled sets)
        headline = [v for k, v in outcomes.items() if k != "clean" and len(v) == EVAL_N]
        if headline:
            worst = np.stack(headline, axis=0).min(axis=0)  # 1 iff correct on ALL attacks
            res["worst_case_headline"] = {
                "acc": float(worst.mean()),
                "eps_norm": EPS_NORM,
                "eps_pixel_255": round(EPS_PIX * 255, 3),
                "note": "per-sample min across PGD-50, APGD-CE, APGD-T at training eps",
            }
            outcomes["worst"] = worst
            print(f"    Worst-case (union of fooled): {worst.mean()*100:.2f}%")

        # ff_112-specific extra rows
        if label == "ff_112":
            print(f"\n  [ff_112 extra epsilon rows]")
            for eps_n, eps_pix_val, atk_lbl, key in [
                (EPS_N_8,   8.0 / 255.0,  "APGD-CE @ 8/255",          "apgd_ce_8_255"),
                (EPS_N_16,  16.0 / 255.0, "APGD-CE @ 16/255 (sanity)", "apgd_ce_16_255"),
                (EPS_N_32,  32.0 / 255.0, "APGD-CE @ 32/255 (sanity)", "apgd_ce_32_255"),
            ]:
                xa = collect_outcomes(model, loader, eps_n, attack="apgd_ce")
                if len(xa):
                    res[key] = {"acc": float(xa.mean()),
                                "eps_norm": round(eps_n, 4),
                                "eps_pixel_255": round(eps_pix_val * 255, 1)}
                    print(f"    {atk_lbl}: {xa.mean()*100:.2f}%")

        all_results[label]  = res
        all_outcomes[label] = outcomes

    # ── Paired McNemar tests (PGD-50 @ train eps) ────────────────────────────
    print(f"\n{'='*70}")
    print("Paired McNemar tests  --  PGD-50 @ training eps, n=1000")
    print(f"{'='*70}")
    paired_labels = [l for l in all_outcomes
                     if "pgd50" in all_outcomes[l] and len(all_outcomes[l]["pgd50"]) == EVAL_N]
    mcnemar_results = {}
    for i, la in enumerate(paired_labels):
        for lb in paired_labels[i+1:]:
            a_arr = all_outcomes[la]["pgd50"]
            b_arr = all_outcomes[lb]["pgd50"]
            p_val = mcnemar_p(a_arr, b_arr)
            delta = float(a_arr.mean() - b_arr.mean()) * 100
            key   = f"{la}_vs_{lb}"
            mcnemar_results[key] = {"delta_pp": round(delta, 2), "mcnemar_p": round(p_val, 6)}
            sig = ("***" if p_val < 0.001 else
                   ("**"  if p_val < 0.01  else
                    ("*"   if p_val < 0.05  else "ns")))
            print(f"  {la:>20} vs {lb:<20}  delta={delta:+.2f}pp  p={p_val:.4f}  {sig}")

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  {'Model':<20} {'Clean':>8} {'PGD-50':>8} {'APGD-CE':>9} {'APGD-T':>8} {'Worst':>8}")
    print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*9} {'─'*8} {'─'*8}")
    for lbl, r in all_results.items():
        cl  = r.get("clean",              {}).get("acc", float("nan"))
        p5  = r.get("pgd50_train",        {}).get("acc", float("nan"))
        ace = r.get("apgd_ce_train",      {}).get("acc", float("nan"))
        at_ = r.get("apgd_t_train",       {}).get("acc", float("nan"))
        wc  = r.get("worst_case_headline", {}).get("acc", float("nan"))
        print(f"  {lbl:<20} {cl*100:>7.2f}% {p5*100:>7.2f}% "
              f"{ace*100:>8.2f}% {at_*100:>7.2f}% {wc*100:>7.2f}%")
    print(f"\n  All rows at eps_norm={EPS_NORM} (approx {EPS_PIX*255:.2f}/255).")
    print(f"  Worst-case = per-sample min across PGD-50, APGD-CE, APGD-T.")

    # ── Save artifact and sync to HF ─────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"report/audit_rev1/joba_matched_eps_{ts}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    artifact = {
        "timestamp": ts,
        "git_sha": sha,
        "eval_seed": EVAL_SEED,
        "eval_n": EVAL_N,
        "eps_norm_train": EPS_NORM,
        "eps_pixel_train_255": round(EPS_PIX * 255, 3),
        "results": all_results,
        "mcnemar_pgd50": mcnemar_results,
    }
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nJOB A artifact: {out_path}")
    sync_to_hf()

# %% [markdown]
# ## JOB B: 56-px Feedforward Control Training (ff_56)
#
# Same corrected TRADES-PGD4 recipe as ff_112 but at 56-px / 16 patch tokens.
# Separates resolution gain from the attack correction.
# Expected: approx 1-1.5 GPU-h on T4/P100 for 15 epochs.

# %%
if "B" not in GEN2_JOBS:
    print("JOB B skipped (not in GEN2_JOBS).")
else:
    for mod in list(sys.modules.keys()):
        if any(mod.startswith(p) for p in ["training", "gen2_foundation", "noesis_vision"]):
            del sys.modules[mod]

    import time as _t_mod
    from training.train_gen2_foundation import (
        run_gen2_phase, run_pgd_kl_attack, compute_trades_loss, get_gen2_loaders
    )
    from gen2_foundation.model import Gen2FoundationModel

    BATCH_B    = int(os.environ.get("GEN2_BATCH", "48"))
    FF56_FINAL = "checkpoints/foundation_ff_56_final.pth"

    if os.path.exists(FF56_FINAL):
        print(f"\nff_56 already trained ({FF56_FINAL}). Running S7 eval only.")
    else:
        print(f"\n{'='*70}")
        print(f"JOB B: Training ff_56  (56-px feedforward, corrected TRADES-PGD4, 15 epochs)")
        print(f"{'='*70}")

        # Pre-flight latency benchmark (5 batches)
        _, _tl, _, _ = get_gen2_loaders(DATA_ROOT, batch_size=BATCH_B, smoke=True, img_size=56)
        _bm = Gen2FoundationModel(phase="g2_gist_only", fovea_size=56).cuda()
        _bo = torch.optim.AdamW(_bm.parameters(), lr=1e-4)
        _bm.train()
        torch.cuda.synchronize(); _t0 = _t_mod.time()
        for _i, (_bx, _by) in enumerate(_tl):
            if _i >= 5: break
            _bx, _by = _bx.cuda(), _by.cuda()
            _bo.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                _adv = run_pgd_kl_attack(_bm, _bx, eps=0.094, steps=4)
                _lss, _, _ = compute_trades_loss(_bm, _bx, _by, _adv, beta=2.5, w_trades=0.55)
            _lss.backward(); _bo.step()
        torch.cuda.synchronize()
        _spb = (_t_mod.time() - _t0) / 5
        _ep_min = (126689 // BATCH_B * _spb) / 60
        print(f"  ff_56 latency: {_spb:.3f} s/batch | ~{_ep_min:.1f} min/epoch "
              f"| ~{_ep_min*15/60:.1f} GPU-h for 15 epochs")
        del _bm, _bo, _tl

        # Training invocation with CLI fallback
        try:
            res_b = run_gen2_phase(
                phase="g2_gist_only",
                arm="v1",
                data_root=DATA_ROOT,
                ckpt_dir="checkpoints",
                report_dir="report",
                dino_path=DINO_PATH,
                batch_size=BATCH_B,
                num_workers=4,
                smoke=False,
                phase_override_name="ff_56",
                img_size=56,
                n_epochs=15,
                epoch_callback=lambda p, ep: sync_to_hf(),
            )
            print(f"ff_56 training complete.")
        except TypeError as e:
            print(f"  run_gen2_phase signature mismatch ({e}); using CLI fallback.")
            cmd = (
                f"python3 training/train_gen2_foundation.py "
                f"--phase g2_gist_only "
                f"--arm v1 "
                f"--data-root {DATA_ROOT} "
                f"--ckpt-dir checkpoints "
                f"--report-dir report "
                f"--dino-path {DINO_PATH} "
                f"--batch-size {BATCH_B} "
                f"--img-size 56 "
                f"--n-epochs 15 "
                f"--phase-name ff_56"
            )
            subprocess.run(cmd, shell=True, check=True)
        sync_to_hf()

    # S7 Evaluation for ff_56
    ff56_eval_ckpt = (FF56_FINAL if os.path.exists(FF56_FINAL) else
                      "checkpoints/foundation_ff_56_best_clean.pth")
    if os.path.exists(ff56_eval_ckpt):
        print(f"\n--- ff_56 S7 Evaluation ({ff56_eval_ckpt}) ---")
        subprocess.run(
            f"python3 training/eval_gen2_foundation.py "
            f"--ckpt-path {ff56_eval_ckpt} "
            f"--phase g2_gist_only "
            f"--data-root {DATA_ROOT} "
            f"--final-verify",
            shell=True, check=True,
        )
        sync_to_hf()
    else:
        print(f"  WARN: ff_56 checkpoint not found; skipping S7 eval.")

# %% [markdown]
# ## JOB C: Two Additional ff_112 Training Seeds
#
# Seeds 42 and 99 (original was seed 17).
# Sets the 2-sigma variance bar for Gate B.
# Each seed runs independently for 15 epochs, followed by S7 evaluation.
# Expected: approx 4.7 GPU-h per seed on T4/P100.

# %%
if "C" not in GEN2_JOBS:
    print("JOB C skipped (not in GEN2_JOBS).")
else:
    for mod in list(sys.modules.keys()):
        if any(mod.startswith(p) for p in ["training", "gen2_foundation", "noesis_vision"]):
            del sys.modules[mod]

    from training.train_gen2_foundation import run_gen2_phase

    BATCH_C = int(os.environ.get("GEN2_BATCH", "48"))

    FF112_EXTRA_SEEDS = [
        {"seed": 42, "suffix": "seed42"},
        {"seed": 99, "suffix": "seed99"},
    ]

    for cfg in FF112_EXTRA_SEEDS:
        s   = cfg["seed"]
        sfx = cfg["suffix"]
        final_pt = f"checkpoints/foundation_ff_112_{sfx}_final.pth"

        print(f"\n{'='*70}")
        print(f"JOB C: ff_112 seed={s} (suffix={sfx})")
        print(f"{'='*70}")

        if os.path.exists(final_pt):
            print(f"  Already trained ({final_pt}). Skipping to S7 eval.")
        else:
            torch.manual_seed(s)
            try:
                run_gen2_phase(
                    phase="ff_112",
                    arm="v1",
                    data_root=DATA_ROOT,
                    ckpt_dir="checkpoints",
                    report_dir="report",
                    dino_path=DINO_PATH,
                    batch_size=BATCH_C,
                    num_workers=4,
                    smoke=False,
                    phase_override_name=f"ff_112_{sfx}",
                    img_size=112,
                    n_epochs=15,
                    training_seed=s,
                    epoch_callback=lambda p, ep: sync_to_hf(),
                )
                print(f"ff_112 seed={s} training complete.")
            except TypeError as e:
                print(f"  run_gen2_phase signature mismatch ({e}); using CLI fallback.")
                subprocess.run(
                    f"python3 training/train_gen2_foundation.py "
                    f"--phase ff_112 "
                    f"--arm v1 "
                    f"--data-root {DATA_ROOT} "
                    f"--ckpt-dir checkpoints "
                    f"--report-dir report "
                    f"--dino-path {DINO_PATH} "
                    f"--batch-size {BATCH_C} "
                    f"--img-size 112 "
                    f"--n-epochs 15 "
                    f"--seed {s} "
                    f"--phase-name ff_112_{sfx}",
                    shell=True, check=True,
                )
            sync_to_hf()

        # S7 eval for this seed
        eval_ckpt = (final_pt if os.path.exists(final_pt) else
                     f"checkpoints/foundation_ff_112_{sfx}_best_clean.pth")
        if os.path.exists(eval_ckpt):
            print(f"\n--- S7 Eval: ff_112 {sfx} ({eval_ckpt}) ---")
            subprocess.run(
                f"python3 training/eval_gen2_foundation.py "
                f"--ckpt-path {eval_ckpt} "
                f"--phase ff_112 "
                f"--data-root {DATA_ROOT} "
                f"--final-verify",
                shell=True, check=True,
            )
            sync_to_hf()
        else:
            print(f"  WARN: {eval_ckpt} not found; skipping eval.")

print("\n" + "=" * 70)
print(f"J2 DISPATCH COMPLETE. Jobs run: {sorted(GEN2_JOBS)}")
print("=" * 70)
