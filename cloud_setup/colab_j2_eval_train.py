#!/usr/bin/env python3
"""
Colab Notebook -- RHAN-NXA J2: Matched-eps Eval + 56px FF Control + ff_112 Seed Replication
==============================================================================================

CONTEXT
-------
This notebook is the second Colab dispatch script in the Gen-2 evidence-gathering
sequence (J1 trained the Gen-2 ladder; J2 collects the numbers needed to compare
models fairly and to set the variance bar for Gen-3 design).

The three jobs answer three independent questions:

  JOB A -- "Are the four checkpoints actually different at the same threat level?"
      Earlier comparisons were invalid because APGD was run at different epsilon
      values across models (8/255 vs 5.4/255), and the training log's robust
      column used a weak PGD-10 monitor.  JOB A re-evaluates all four checkpoints
      on an identical 1,000-image slice with all attacks at eps_norm=0.094 (the
      training threat radius, approx 5.4/255).  McNemar tests quantify whether
      accuracy differences are statistically significant.

      Checkpoints:
        g2_gist_only    phase 1 of the Gen-2 ladder (56px, TRADES)
        recurrence_only phase 2 (56px, T=4 EMA recurrence)
        gen1_core       phase 6 (56px, full Gen-2 core)
        ff_112          J1 feedforward baseline (112px, 64 tokens, no loop)

      Attacks (all in fp32):
        PGD-50         alpha=2.5*eps/50, uniform random start
        APGD-CE-100    torchattacks, 1 restart, CE loss
        APGD-T-100     torchattacks, 1 restart, 10 worst-class targets
      Per-sample worst-case = correct only if it resists ALL three attacks.

      Extra rows for ff_112 (gradient-masking sanity check):
        APGD-CE @ 8/255    labelled separately for the broader threat radius
        APGD-CE @ 16/255   should be near 0% for a well-trained model
        APGD-CE @ 32/255   should be ~0% without gradient masking

      Epsilon is logged in both normalised and pixel units in every output row.
      Artifact: report/audit_rev1/joba_matched_eps_<timestamp>.json

  JOB B -- "How much of ff_112's improvement is resolution vs attack correction?"
      ff_112 improved over recurrence_only for two reasons:
        (a) resolution increased from 56px/16 tokens to 112px/64 tokens
        (b) the training attack was corrected (alpha = 2.5*eps/steps, not 1x)
      JOB B trains an ff_56 control: same feedforward architecture and corrected
      recipe but at 56px.  Comparing:
        ff_56 vs recurrence_only  --> isolates attack-correction benefit
        ff_112 vs ff_56           --> isolates resolution benefit
      Architecture: phase=g2_gist_only (feedforward trunk, no recurrent loop),
      img_size=56, phase_override_name="ff_56" so checkpoints are named correctly.

  JOB C -- "What is the 2-sigma uncertainty bar for ff_112?"
      All future Gen-3 improvements must beat ff_112 by more than 2*sigma seed
      variance.  JOB C trains two additional ff_112 seeds (42 and 99; original
      was seed 17) to establish that bar.  Each seed trains for 15 epochs and
      is evaluated with the S7 protocol.

USAGE
-----
  1. New Colab notebook.  Runtime > Change runtime type > T4 GPU.
  2. Secrets (key icon) > Add secret "HF_TOKEN" > Enable notebook access.
  3. Optionally set GEN2_JOBS before running:
       import os; os.environ["GEN2_JOBS"] = "A"      # eval only (~2 GPU-h)
       import os; os.environ["GEN2_JOBS"] = "A,B,C"  # all jobs (~13 GPU-h)
     Default (unset) = "A,B,C".
  4. Run all cells in sequence.

  Resume after disconnect:
    Re-run all cells from the top.  Step 4 restores rolling checkpoints from
    HuggingFace and resumes training from the last completed epoch.  Completed
    checkpoints (foundation_<phase>_final.pth) are never re-trained.

RELATIONSHIP TO OTHER FILES
---------------------------
  cloud_setup/colab_j1_foundation.py  -- runs FIRST; trains the Gen-2 ladder
  training/train_gen2_foundation.py   -- run_gen2_phase() and attack recipe
  training/eval_gen2_foundation.py    -- run_final_checkpoint_eval() (S7 protocol)
  noesis_vision/RHAN_NXA/docs/revision1.md -- forensic audit that motivated J2
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

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from google.colab import userdata
        hf_token = userdata.get("HF_TOKEN")
        os.environ["HF_TOKEN"] = hf_token
    except Exception:
        pass
if not hf_token:
    raise RuntimeError(
        "HF_TOKEN not found. Secrets > Add secret 'HF_TOKEN' > Enable notebook access."
    )
print(f"HF_TOKEN authenticated (starts with: {hf_token[:4]}...)")

import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
else:
    print("WARNING: No GPU detected -- Go to Runtime > Change runtime type > T4 GPU.")

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

# Which jobs to run (default: all)
GEN2_JOBS = set(j.strip().upper() for j in os.environ.get("GEN2_JOBS", "A,B,C").split(","))
print(f"Jobs scheduled: {sorted(GEN2_JOBS)}")

# %% [markdown]
# ## Step 2: Clone & Checkout feature/gen2-rebuild

# %%
REPO_NAME = "Adversarial-Cognitive-Model"
WORK_DIR  = f"/content/{REPO_NAME}"
BRANCH    = "feature/gen2-rebuild"

os.chdir("/content")
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
print(f"Checked out {BRANCH} @ commit {sha}")

# Evict stale cached modules
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
# ## Step 4: HuggingFace Repos & Checkpoint Restore

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
        print(f"  HF check {r}: {e}")


def restore_checkpoints_from_hf():
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
                    # eval reports go to report/eval/, audit reports to report/audit_rev1/
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
    """Upload new checkpoints, eval JSONs, and JSONL logs to HF."""
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
# ## Step 5: ImageNet-100 Data Bootstrap

# %%
DATA_ROOT = os.environ.get("J2_DATA_ROOT", "/content/imagenet100")
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
# All four checkpoints evaluated on a shared 1,000-image slice.
# All headline attacks run at eps_norm=0.094 (training epsilon, approx 5.4/255).
# ff_112 also gets extra rows at 8/255, 16/255 sanity, 32/255 sanity.
# Per-sample binary outcome vectors -> per-sample worst-case robust accuracy.
# Paired McNemar tests for every model pair.
# All attacks run in fp32.

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

    print(f"\n{'='*70}")
    print(f"JOB A  --  Matched-epsilon Evaluation  (n={EVAL_N}, seed={EVAL_SEED})")
    print(f"  train eps: norm={EPS_NORM}  pixel={EPS_PIX*255:.2f}/255")
    print(f"{'='*70}")

    def build_loader(img_size: int) -> DataLoader:
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
                # fp32, corrected alpha = 2.5*eps/steps
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
    # (label, phase, img_size, ckpt_path)
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

    # ── Save artifact ────────────────────────────────────────────────────────
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = f"report/audit_rev1/joba_matched_eps_{ts}.json"
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
# Phase mapping: g2_gist_only (single-pass trunk, no recurrent loop, no gist gate).
# Expected: approx 1-1.5 GPU-h on T4 for 15 epochs.

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

    BATCH_B  = int(os.environ.get("GEN2_BATCH", "48"))
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

        # Training
        # Note: run_gen2_phase uses phase="g2_gist_only" (feedforward trunk, corrected recipe).
        # If run_gen2_phase does not yet support phase_override_name / img_size / n_epochs,
        # use the subprocess eval_gen2_foundation path instead (see fallback below).
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
                phase_override_name="ff_56",   # renames outputs to foundation_ff_56_*
                img_size=56,
                n_epochs=15,
                epoch_callback=lambda p, ep: sync_to_hf(),
            )
            print(f"ff_56 training complete.")
        except TypeError as e:
            # run_gen2_phase may not yet accept phase_override_name / img_size / n_epochs.
            # Fall back to CLI invocation.
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
# Seeds 42 and 99 (the original run used seed 17).
# Sets the 2-sigma variance bar that every later arm must beat.
# Each seed runs independently for 15 epochs, then S7 eval.
# Expected: approx 4.7 GPU-h per seed, 9.4 GPU-h total.

# %%
if "C" not in GEN2_JOBS:
    print("JOB C skipped (not in GEN2_JOBS).")
else:
    for mod in list(sys.modules.keys()):
        if any(mod.startswith(p) for p in ["training", "gen2_foundation", "noesis_vision"]):
            del sys.modules[mod]

    from training.train_gen2_foundation import run_gen2_phase

    BATCH_C = int(os.environ.get("GEN2_BATCH", "48"))

    # Seed 1 (seed=17) is the original ff_112.
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


# ============================================================================
# %% [markdown]
# ## PHASE 4: Full AutoAttack Battery + Table Quarantine + ff_56 Upload + OPAL Gate
#
# Runs the complete AutoAttack suite (APGD-CE, APGD-T, FAB-T, Square) on the
# two primary models (ff_112, recurrence_only) at matched eps = 5.417/255
# on the canonical N=1000 seed-42 slice.
#
# This closes the "AA-core" caveat: prior reports used only PGD-50 and a
# partial APGD-CE and omitted FAB-T and Square, leaving a gradient-masking
# loophole open.
#
# Additionally:
#   - Quarantines all result rows that lack per-sample arrays (cannot be
#     McNemar-tested), marking them INVALID in the regenerated table.
#   - Regenerates the canonical comparison table from raw numpy arrays only.
#   - Uploads foundation_ff_56_final.pth under a unique timestamped name to
#     FerrariKazu/rhan-nxa-g2-checkpoints so the stale file is not silently
#     overwritten.
#   - OPAL gate: checks for any foundation_recurrence_112_seed*_final.pth
#     on HF; if none exist, prints a deferred notice and exits cleanly.
# ============================================================================

# %%
if "D" not in GEN2_JOBS:
    print("\nPHASE 4 (Job D) skipped. Set GEN2_JOBS to include 'D' to run.")
else:
    import datetime as _dt
    import math as _math
    import numpy as _np
    import scipy.stats as _stats

    # ── shared eval constants (same as JOB A) ───────────────────────────────
    _EVAL_SEED   = 42
    _EVAL_N      = 1000
    _BATCH_D     = 32
    _EPS_NORM    = 0.094
    _EPS_PIX     = _EPS_NORM * 0.226      # IMAGENET_STD mean = 0.226
    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD  = [0.229, 0.224, 0.225]
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for _pkg in ["torchattacks", "autoattack"]:
        try:
            __import__(_pkg.replace("-", "_"))
        except ImportError:
            subprocess.run(["pip", "install", "--quiet", _pkg], check=True)

    for mod in list(sys.modules.keys()):
        if any(mod.startswith(p) for p in ["training", "gen2_foundation", "noesis_vision"]):
            del sys.modules[mod]

    from gen2_foundation.model import Gen2FoundationModel
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset

    def _build_loader_d(img_size):
        tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
        val_ds = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"), transform=tf)
        g = torch.Generator().manual_seed(_EVAL_SEED)
        idx = torch.randperm(len(val_ds), generator=g)[:_EVAL_N].tolist()
        return DataLoader(Subset(val_ds, idx), batch_size=_BATCH_D,
                          shuffle=False, num_workers=2, pin_memory=True)

    def _load_model_d(phase, img_size, ckpt_path):
        m = Gen2FoundationModel(phase=phase, arm="v1", fovea_size=img_size).to(_DEVICE)
        sd = torch.load(ckpt_path, map_location=_DEVICE, weights_only=False)
        m.load_state_dict(sd.get("model", sd), strict=False)
        m.eval()
        return m

    class _NormWrap(torch.nn.Module):
        """Wrap a normalized model to accept [0,1] inputs for torchattacks."""
        def __init__(self, base):
            super().__init__()
            self.base = base
            m = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
            s = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
            self.register_buffer("_m", m)
            self.register_buffer("_s", s)
        def forward(self, x):
            return self.base((x - self._m) / self._s)

    def _run_full_aa(model_raw, loader, eps_norm):
        """
        Run the full AutoAttack suite (APGD-CE, APGD-T, FAB-T, Square).
        model_raw: already normalised model (accepts standardised inputs).
        Returns dict of {attack_name: np.ndarray(shape N, dtype int8)} correctness arrays.
        """
        import torchattacks

        # torchattacks expects unnormalised [0,1] inputs
        model_01 = _NormWrap(model_raw).to(_DEVICE)
        model_01.eval()

        # Collect all images and labels first (needed for torchattacks batch API)
        all_x, all_y = [], []
        for xb, yb in loader:
            all_x.append(xb); all_y.append(yb)
        all_x = torch.cat(all_x)
        all_y = torch.cat(all_y)

        attacks = {
            "apgd_ce":  torchattacks.APGD(model_01, norm="Linf", eps=eps_norm,
                                           steps=100, n_restarts=1, loss="ce", verbose=False),
            "apgd_t":   torchattacks.APGDT(model_01, norm="Linf", eps=eps_norm,
                                            steps=100, n_restarts=1, n_classes=100, verbose=False),
            "fab_t":    torchattacks.FAB(model_01, norm="Linf", eps=eps_norm,
                                         steps=100, n_restarts=1, n_classes=100, verbose=False),
            "square":   torchattacks.Square(model_01, norm="Linf", eps=eps_norm,
                                            n_queries=5000, n_restarts=1, verbose=False),
        }

        results = {}
        for atk_name, atk in attacks.items():
            print(f"    Running {atk_name} (n={len(all_x)}, eps={eps_norm:.4f})...", flush=True)
            t0 = time.time()
            correct = []
            # process in smaller batches to avoid OOM
            bs = 50
            for i in range(0, len(all_x), bs):
                xb = all_x[i:i+bs].to(_DEVICE)
                yb = all_y[i:i+bs].to(_DEVICE)
                # un-normalise for torchattacks (expects [0,1])
                m = torch.tensor(_IMAGENET_MEAN).view(1,3,1,1).to(_DEVICE)
                s = torch.tensor(_IMAGENET_STD).view(1,3,1,1).to(_DEVICE)
                xb_01 = (xb * s + m).clamp(0, 1)
                x_adv_01 = atk(xb_01, yb)
                with torch.no_grad():
                    preds = model_01(x_adv_01).argmax(-1)
                correct.extend((preds == yb).cpu().numpy().tolist())
            results[atk_name] = _np.array(correct, dtype=_np.int8)
            acc = results[atk_name].mean() * 100
            print(f"    {atk_name}: {acc:.2f}% ({time.time()-t0:.0f}s)", flush=True)

        return results

    # ── Target models ────────────────────────────────────────────────────────
    _PHASE4_MODELS = [
        ("ff_112",          "ff_112",          112, "checkpoints/foundation_ff_112_final.pth"),
        ("recurrence_only", "recurrence_only",  56, "checkpoints/foundation_recurrence_only_final.pth"),
    ]

    _ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    _phase4_dir = "report/audit_rev1"
    os.makedirs(_phase4_dir, exist_ok=True)

    _all_phase4 = {}  # label -> {clean, aa_results}
    _all_arrays = {}  # label -> {attack_name: np.ndarray}

    print(f"\n{'='*70}")
    print(f"PHASE 4: Full AutoAttack Battery  (N={_EVAL_N}, seed={_EVAL_SEED}, eps={_EPS_NORM})")
    print(f"{'='*70}")

    for _label, _phase, _img_size, _ckpt in _PHASE4_MODELS:
        if not os.path.exists(_ckpt):
            print(f"\n  SKIP {_label}: checkpoint not found at {_ckpt}")
            continue

        print(f"\n{'─'*60}")
        print(f"  {_label}  (img={_img_size}px  ckpt={os.path.basename(_ckpt)})")
        print(f"{'─'*60}")

        _loader = _build_loader_d(_img_size)
        _model  = _load_model_d(_phase, _img_size, _ckpt)

        # Clean
        _clean_correct = []
        _model.eval()
        with torch.no_grad():
            for _xb, _yb in _loader:
                _xb, _yb = _xb.to(_DEVICE), _yb.to(_DEVICE)
                _preds = _model(_xb).argmax(-1)
                _clean_correct.extend((_preds == _yb).cpu().numpy().tolist())
        _corr_clean = _np.array(_clean_correct, dtype=_np.int8)
        print(f"    clean: {_corr_clean.mean()*100:.2f}%")

        # Full AutoAttack
        _aa_results = _run_full_aa(_model, _loader, _EPS_NORM)

        # Worst-case = bitwise AND across all four attacks
        _corr_worst = _np.stack(list(_aa_results.values()), axis=0).min(axis=0)
        print(f"    Worst-case (AA-core): {_corr_worst.mean()*100:.2f}%")

        _all_phase4[_label] = {
            "clean_acc":      float(_corr_clean.mean()),
            "apgd_ce_acc":    float(_aa_results["apgd_ce"].mean()),
            "apgd_t_acc":     float(_aa_results["apgd_t"].mean()),
            "fab_t_acc":      float(_aa_results["fab_t"].mean()),
            "square_acc":     float(_aa_results["square"].mean()),
            "worst_case_acc": float(_corr_worst.mean()),
        }
        _all_arrays[_label] = {
            "clean":    _corr_clean,
            "apgd_ce":  _aa_results["apgd_ce"],
            "apgd_t":   _aa_results["apgd_t"],
            "fab_t":    _aa_results["fab_t"],
            "square":   _aa_results["square"],
            "worst":    _corr_worst,
        }

    # ── Paired McNemar from raw arrays ──────────────────────────────────────
    _mcnemar_phase4 = {}
    _labels_present = [l for l in _all_arrays if "worst" in _all_arrays[l]]
    if len(_labels_present) == 2:
        _la, _lb = _labels_present[0], _labels_present[1]
        _a_arr = _all_arrays[_la]["worst"]
        _b_arr = _all_arrays[_lb]["worst"]
        _b_count = int(((_a_arr == 1) & (_b_arr == 0)).sum())
        _c_count = int(((_a_arr == 0) & (_b_arr == 1)).sum())
        _n_disc  = _b_count + _c_count
        _delta   = float(_a_arr.mean() - _b_arr.mean()) * 100.0
        if _n_disc > 0:
            _p_exact = float(_stats.binomtest(_b_count, _n_disc, 0.5, alternative="two-sided").pvalue)
        else:
            _p_exact = 1.0
        _mcnemar_phase4[f"{_la}_vs_{_lb}"] = {
            "b": _b_count, "c": _c_count,
            "n_discordant": _n_disc,
            "delta_pp": round(_delta, 3),
            "p_exact": _p_exact,
        }
        print(f"\n  McNemar (worst-case AA-core, N={_EVAL_N}):")
        print(f"    {_la} vs {_lb}: b={_b_count}, c={_c_count}, "
              f"delta={_delta:+.2f}pp, p={_p_exact:.3e}")

    # ── Summary table from raw arrays only ──────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PHASE 4 REGENERATED TABLE (from raw per-sample arrays, N={_EVAL_N})")
    print(f"  eps={_EPS_NORM:.4f} (~{_EPS_PIX*255:.2f}/255)   seed={_EVAL_SEED}")
    print(f"{'='*70}")
    print(f"  {'Model':<20} {'Clean':>7} {'APGD-CE':>9} {'APGD-T':>8} "
          f"{'FAB-T':>7} {'Square':>8} {'Worst':>7}")
    print(f"  {'─'*20} {'─'*7} {'─'*9} {'─'*8} {'─'*7} {'─'*8} {'─'*7}")
    for _lbl, _r in _all_phase4.items():
        print(f"  {_lbl:<20} "
              f"{_r['clean_acc']*100:>6.2f}% "
              f"{_r['apgd_ce_acc']*100:>8.2f}% "
              f"{_r['apgd_t_acc']*100:>7.2f}% "
              f"{_r['fab_t_acc']*100:>6.2f}% "
              f"{_r['square_acc']*100:>7.2f}% "
              f"{_r['worst_case_acc']*100:>6.2f}%")

    # ── Quarantine: mark any existing JOB A results that lack per-sample arrays
    print(f"\n  QUARANTINE: Results without per-sample arrays cannot be McNemar-tested.")
    _quarantine_log = []
    for _jf in Path("report/audit_rev1").glob("joba_matched_eps_*.json"):
        with open(_jf) as _f:
            _j = json.load(_f)
        _has_arrays = any(
            "worst" in _all_arrays.get(_lbl, {})
            for _lbl in _j.get("results", {})
        )
        if not _has_arrays:
            _quarantine_log.append({
                "file": str(_jf),
                "reason": "No per-sample arrays; McNemar counts unverifiable",
                "status": "INVALID",
                "quarantined_at": _ts,
            })
            print(f"  QUARANTINE: {_jf.name} (no raw arrays, counts unverifiable)")

    # ── Save Phase 4 artifact ────────────────────────────────────────────────
    _p4_artifact = {
        "phase": 4,
        "timestamp": _ts,
        "git_sha": sha,
        "eval_n": _EVAL_N,
        "eval_seed": _EVAL_SEED,
        "eps_norm": _EPS_NORM,
        "eps_pixel_255": round(_EPS_PIX * 255, 3),
        "attacks": ["apgd_ce", "apgd_t", "fab_t", "square"],
        "worst_case_note": "bitwise AND of all 4 attacks per sample",
        "results": _all_phase4,
        "mcnemar_worst_case": _mcnemar_phase4,
        "quarantined_priors": _quarantine_log,
    }
    _p4_path = f"report/audit_rev1/phase4_full_aa_{_ts}.json"
    with open(_p4_path, "w") as _f:
        json.dump(_p4_artifact, _f, indent=2)
    print(f"\n  Phase 4 artifact saved: {_p4_path}")

    # Save per-sample arrays as .npz
    _arr_flat = {}
    for _lbl, _arrs in _all_arrays.items():
        for _atk, _arr in _arrs.items():
            _arr_flat[f"{_lbl}_{_atk}"] = _arr
    _npz_path = f"report/audit_rev1/phase4_arrays_{_ts}.npz"
    _np.savez_compressed(_npz_path, **_arr_flat)
    print(f"  Per-sample arrays saved: {_npz_path}")

    # Upload Phase 4 artifacts to HF
    if hf_token:
        try:
            api.upload_file(path_or_fileobj=_p4_path,
                            path_in_repo=os.path.basename(_p4_path),
                            repo_id=HF_BEST, repo_type="dataset", token=hf_token)
            api.upload_file(path_or_fileobj=_npz_path,
                            path_in_repo=os.path.basename(_npz_path),
                            repo_id=HF_BEST, repo_type="dataset", token=hf_token)
            print("  Phase 4 artifacts uploaded to HuggingFace.")
        except Exception as _e:
            print(f"  Warning: HF upload failed: {_e}")

    # ── Upload local ff_56 under a unique, timestamped name ─────────────────
    print(f"\n{'─'*60}")
    print("  UPLOADING LOCAL ff_56 UNDER UNIQUE NAME (no overwrite of stale HF copy)")
    print(f"{'─'*60}")
    _FF56_LOCAL = "checkpoints/foundation_ff_56_final.pth"
    if os.path.exists(_FF56_LOCAL):
        import hashlib as _hl
        _h = _hl.sha256()
        with open(_FF56_LOCAL, "rb") as _f:
            for _chunk in iter(lambda: _f.read(1 << 20), b""):
                _h.update(_chunk)
        _ff56_sha = _h.hexdigest()[:12]
        # Unique name includes timestamp + first 12 chars of sha256
        _ff56_hf_name = f"foundation_ff_56_local_{_ts}_{_ff56_sha}.pth"
        print(f"  Local file: {_FF56_LOCAL}")
        print(f"  SHA-256 prefix: {_ff56_sha}")
        print(f"  HF target name: {_ff56_hf_name}")
        if hf_token:
            try:
                api.upload_file(
                    path_or_fileobj=_FF56_LOCAL,
                    path_in_repo=_ff56_hf_name,
                    repo_id=HF_BEST,
                    repo_type="dataset",
                    token=hf_token,
                )
                print(f"  Uploaded: {_ff56_hf_name}")
                # Record provenance
                _ff56_prov = {
                    "uploaded_at": _ts,
                    "local_file": _FF56_LOCAL,
                    "hf_name": _ff56_hf_name,
                    "sha256_prefix": _ff56_sha,
                    "note": (
                        "True local ff_56 retrained with corrected TRADES-PGD4. "
                        "Uploaded under unique name to preserve the stale HF copy "
                        "for provenance tracing. Do NOT use 'foundation_ff_56_final.pth' "
                        "on HF — use this file."
                    ),
                }
                _prov_path = f"report/audit_rev1/ff56_upload_provenance_{_ts}.json"
                with open(_prov_path, "w") as _f:
                    json.dump(_ff56_prov, _f, indent=2)
                api.upload_file(path_or_fileobj=_prov_path,
                                path_in_repo=os.path.basename(_prov_path),
                                repo_id=HF_BEST, repo_type="dataset", token=hf_token)
                print(f"  Provenance record uploaded: {os.path.basename(_prov_path)}")
            except Exception as _e:
                print(f"  Warning: ff_56 HF upload failed: {_e}")
        else:
            print("  HF_TOKEN not set; skipping ff_56 upload.")
    else:
        print(f"  SKIP: {_FF56_LOCAL} not found locally.")

    # ── OPAL GATE: Check if consolidated flagship exists ────────────────────
    print(f"\n{'='*70}")
    print("  OPAL GATE: Checking for consolidated flagship checkpoints")
    print(f"{'='*70}")
    _flagship_found = []
    for _seed in [17, 42, 99]:
        _local_path = f"checkpoints/foundation_recurrence_112_seed{_seed}_final.pth"
        if os.path.exists(_local_path):
            _flagship_found.append((_seed, _local_path, "local"))
        elif hf_token:
            # Check HF
            try:
                _hf_files = list(list_repo_files(repo_id=HF_BEST, repo_type="dataset", token=hf_token))
                _hf_name = f"foundation_recurrence_112_seed{_seed}_final.pth"
                if any(_hf_name in _fn for _fn in _hf_files):
                    _flagship_found.append((_seed, _hf_name, "hf"))
            except Exception:
                pass

    if not _flagship_found:
        print("  OPAL: No consolidated flagship checkpoints found.")
        print("  OPAL: The Kaggle run (seed 17) is currently training.")
        print("  OPAL: This gate will pass once all 3 seeds complete and are uploaded.")
        print("  OPAL STATUS: DEFERRED — all downstream OPAL analysis blocked until flagship exists.")
    else:
        print(f"  OPAL: Found {len(_flagship_found)}/3 flagship checkpoints:")
        for _s, _p, _src in _flagship_found:
            print(f"    seed={_s}  source={_src}  path={_p}")
        if len(_flagship_found) < 3:
            print(f"  OPAL STATUS: PARTIAL ({len(_flagship_found)}/3 seeds present) — wait for all 3.")
        else:
            print("  OPAL STATUS: GATE OPEN — all 3 flagship seeds available. OPAL may proceed.")

    print(f"\n{'='*70}")
    print("PHASE 4 COMPLETE.")
    print(f"{'='*70}")


# Dispatch complete banner
print("\n" + "=" * 70)
print(f"J2+PHASE4 DISPATCH COMPLETE. Jobs run: {sorted(GEN2_JOBS)}")
print("=" * 70)

