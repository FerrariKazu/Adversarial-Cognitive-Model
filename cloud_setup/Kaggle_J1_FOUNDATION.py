#!/usr/bin/env python3
"""
Kaggle Notebook — RHAN-NXA Generation 1 FOUNDATION (Agent J1) Execution
=======================================================================

Runs Part 2 steps 1-4 (backbone_only -> recurrence_only -> belief_no_f ->
belief_with_f) via training/train_generation1_foundation.py, on the
PLACEHOLDER fixed gaze schedule. Steps 5-6 (the AIS-v2 swap) are J2's file
and are deliberately NOT runnable from here (Agent F's modules exist; the
ladder gate is J2's).

DURABILITY MODEL (mirrors Kaggle_NOESIS.py):
  - /kaggle/working is wiped between sessions, so ALL durable state lives
    on HF. The trainer itself owns the two dedicated repos (Gen-0
    contamination is structurally impossible — separate repos):
        FerrariKazu/rhan-nxa-checkpoints          (best ckpts + eval CSVs)
        FerrariKazu/rhan-nxa-checkpoints-rolling  (rolling ckpts + roadmap)
  - Resume: the trainer calls Agent A's resume_or_abort (local first, then
    the rolling HF repo) + resume_guard; a session dying at ANY point
    resumes by RE-RUNNING THIS CELL. Never pass --force-restart here.
  - One run per cell execution: the dispatch executes the machine's next
    action and exits. Re-run the cell (or the notebook) to continue.
    Phases already marked done in the HF-synced roadmap are skipped.

USAGE:
  1. Kaggle Notebook, accelerator = GPU (T4 x2 or P100); Internet ON.
  2. Add-ons > Secrets > add 'HF_TOKEN' (must match exactly).
  3. Run all cells. First run: SMOKE PROOF cell (fast, synthetic) then the
     dispatch cell. The dispatch STOPs loudly if ImageNet-100 is absent —
     attach the dataset first (see DATA_ROOT below).
  4. Pre-flight without spending compute: set environment variable
     NOESIS_DRY_RUN=1 (Kaggle: Add-ons > Environment variables) — prints
     the exact launch commands, exercises skip/gate logic against LIVE HF
     state, and never launches training, touches git state, or writes to
     HF (roadmap writes are shielded to a scratch copy).

DATA (the one thing this notebook does NOT do for you):
  Attach an ImageNet-100 Kaggle dataset and point DATA_ROOT at it (default
  /kaggle/input/imagenet100 — adjust to your dataset's mount path). The
  layout must be <root>/{train,val}/<wnid>/... (exactly 100 class dirs per
  split). The trainer runs Agent I's structural validation BEFORE any
  launch; a wrong layout is a loud STOP, never a silently-wrong result.
"""
# %% [markdown]
# ## Step 1: Environment — fail fast on HF stalls, then deps

# %%
import os, sys, subprocess, json, shutil

# huggingface_hub freezes HF_HUB_DOWNLOAD_TIMEOUT at import time — set it
# BEFORE any hub import (the dep-check below imports it), and trainer
# subprocesses inherit it. A stalled download raises in ~30s instead of
# hanging the session silently (the 2026-08-09 Step A incident).
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

# ── PRE-FLIGHT (dry-run) MODE ─────────────────────────────────────────────
DRY_RUN = os.environ.get("NOESIS_DRY_RUN", "0") == "1"

# J1's roadmap lives under report/ in-repo; in dry-run, writes are shielded
# to a scratch copy so pre-flight can never mutate protocol state.
ROADMAP_LOCAL = "report/generation1_foundation_roadmap.json"
if DRY_RUN:
    import tempfile, shutil
    _shadow = os.path.join(tempfile.mkdtemp(prefix="j1_dryrun_"),
                           "generation1_foundation_roadmap.json")
    if os.path.exists(ROADMAP_LOCAL):
        shutil.copy(ROADMAP_LOCAL, _shadow)
    ROADMAP_LOCAL = _shadow


def run(cmd, check=True):
    print(f"\n[RUN]: {cmd}", flush=True)
    if DRY_RUN:
        print("  [DRY-RUN] command NOT executed — pre-flight mode.", flush=True)
        return 0
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True, bufsize=1)
    for line in process.stdout:
        print(line, end='', flush=True)
    rc = process.wait()
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return rc


if not DRY_RUN:
    # Kaggle GPU runtimes PREINSTALL torch+cu121 — never upgrade/reinstall
    # torch over the platform build. Only install what is demonstrably
    # missing, one package at a time.
    try:
        import torch  # noqa: F401
        if not torch.cuda.is_available():
            raise ImportError("torch present but CUDA unavailable")
    except Exception:
        run("pip install --quiet torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu121")
    for _mod, _pkg in (("huggingface_hub", "huggingface_hub"),
                       ("pandas", "pandas"),
                       ("PIL", "Pillow")):
        try:
            __import__(_mod)
        except Exception:
            run(f"pip install --quiet {_pkg}")
    # NOTE: `datasets` is deliberately NOT installed — the J1 trainer +
    # Agent I eval chain import torch/torchvision/pandas/huggingface_hub
    # only (verified); installing it would pull pyarrow for nothing.

# %% [markdown]
# ## Step 2: Clone and checkout feature/rhan-next (NOT main!)

# %%
REPO_NAME = 'Adversarial-Cognitive-Model'
WORK_DIR = f'/kaggle/working/{REPO_NAME}'

if not DRY_RUN:
    # /kaggle/working is the writable scratch dir (wiped between sessions —
    # hence the HF durability model). Never put the repo on /kaggle/input.
    os.chdir('/kaggle/working')
    if not os.path.exists(WORK_DIR):
        run(f'git clone https://github.com/FerrariKazu/{REPO_NAME}.git')
    os.chdir(WORK_DIR)
    sys.path.insert(0, WORK_DIR)
    os.environ["PYTHONPATH"] = f"{WORK_DIR}:{os.environ.get('PYTHONPATH', '')}"

    # RHAN-NXA lives on feature/rhan-next. Never reset to origin/main here.
    run('git fetch origin')
    _branch_ok = subprocess.run(
        'git ls-remote --heads origin feature/rhan-next',
        shell=True, capture_output=True, text=True).stdout.strip()
    if not _branch_ok:
        raise RuntimeError(
            "feature/rhan-next is NOT on origin. Push it first:\n"
            "  git push origin feature/rhan-next\n"
            "(RHAN-NXA must not be merged to main until J1/J2 validate.)")
    run('git checkout -B feature/rhan-next origin/feature/rhan-next')
    run('git reset --hard origin/feature/rhan-next')
    _sha = subprocess.run('git rev-parse --short HEAD', shell=True,
                          capture_output=True, text=True).stdout.strip()
    print(f"✓ checked out feature/rhan-next @ {_sha}", flush=True)

    if not os.path.exists("training/train_generation1_foundation.py"):
        raise RuntimeError(
            "training/train_generation1_foundation.py not found — the "
            "checked-out commit predates Agent J1. Push/verify the J1 "
            "commit on feature/rhan-next first.")

# %% [markdown]
# ## Step 3: HF_TOKEN (Kaggle Secrets) + environment

# %%
import torch
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        # Kaggle Secrets are injected as env vars; this fallback reads them
        # via the official client if env injection didn't happen.
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret('HF_TOKEN')
        os.environ["HF_TOKEN"] = hf_token
    except Exception:
        pass
if not hf_token:
    raise RuntimeError("HF_TOKEN not found. Add it to Kaggle Secrets: "
                       "Add-ons > Secrets > 'HF_TOKEN' (key must match "
                       "exactly).")

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
print(f"✓ HF_TOKEN set for user: {hf_token[:4]}...{hf_token[-4:]}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ VRAM: "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠ CPU-only runtime — training will be impossibly slow. "
          "Attach a GPU accelerator.", flush=True)

# %% [markdown]
# ## Step 4: SMOKE PROOF — orchestration chain, synthetic, NOT results

# %%
# One fast end-to-end proof that the 4-phase chain orchestrates on THIS
# runtime (machine walk, gradient-reach checks, parity, eval CSVs). Uses
# synthetic loaders, 1 epoch, NO HF writes, NO dataset. Numbers are NOT
# results. Skip on re-runs: delete the marker to force it again.
_SMOKE_MARK = "/kaggle/working/.j1_smoke_ok"
if DRY_RUN:
    print("[DRY-RUN] would run: python3 training/train_generation1_foundation.py --smoke")
elif not os.path.exists(_SMOKE_MARK):
    run("python3 training/train_generation1_foundation.py --smoke")
    open(_SMOKE_MARK, "w").write("ok\n")
    print("✓ smoke chain verified on this runtime (marker written).",
          flush=True)
else:
    print("✓ smoke already verified on this runtime (marker exists).",
          flush=True)

# %% [markdown]
# ## Step 5: Launch helpers — durable state on HF, one run per cell

# %%
DATA_ROOT = os.environ.get("J1_DATA_ROOT", "/kaggle/input/imagenet100")
TRAINER = "training/train_generation1_foundation.py"
HF_ROLLING = "FerrariKazu/rhan-nxa-checkpoints-rolling"
HF_BEST = "FerrariKazu/rhan-nxa-checkpoints"
ROADMAP_ON_HF = "generation1_foundation_roadmap.json"


def hf_file_exists(repo_id, filename):
    try:
        from huggingface_hub import HfApi
        return filename in HfApi(token=hf_token).list_repo_files(
            repo_id=repo_id, repo_type="dataset")
    except Exception:
        return False


def _roadmap_status():
    """Phase statuses from local roadmap, restoring the HF copy first
    (runtime state beats the fresh-clone baseline; rev-guarded)."""
    local_rev = 0
    if os.path.exists(ROADMAP_LOCAL):
        try:
            local_rev = json.load(open(ROADMAP_LOCAL)).get(
                "generation1_foundation", {}).get("schema_version", 1)
        except Exception:
            pass
    if not DRY_RUN:
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(repo_id=HF_ROLLING, filename=ROADMAP_ON_HF,
                                repo_type="dataset", token=hf_token)
            shutil.copyfile(p, ROADMAP_LOCAL)
            print("  ✓ roadmap restored from HF (runtime state preserved)",
                  flush=True)
        except Exception:
            pass
    try:
        fnd = json.load(open(ROADMAP_LOCAL)).get("generation1_foundation", {})
        return {ph: st.get("status", "not_started")
                for ph, st in fnd.get("phases", {}).items()}
    except Exception:
        return {}


# %% [markdown]
# ## Step 6: THE DISPATCH — one run, resume-safe, data-gated

# %%
# Data pre-flight FIRST (the contract's mandatory check) — before spending
# a single training step. Missing data is a loud STOP with instructions,
# never a silently-wrong run.
data_ready = os.path.isdir(DATA_ROOT)
if DRY_RUN:
    print(f"[DRY-RUN] DATA_ROOT={DATA_ROOT} exists={data_ready}")
    print(f"[DRY-RUN] roadmap statuses: {_roadmap_status()}")
    print(f"[DRY-RUN] would launch: python3 {TRAINER} "
          f"--data-root {DATA_ROOT} --batch-size 96")
    print("[DRY-RUN] no training launched, no state mutated.")
elif not data_ready:
    raise SystemExit(
        f"STOP — ImageNet-100 not found at {DATA_ROOT}.\n"
        "  1) Attach an ImageNet-100 dataset to this notebook (Add Input).\n"
        "  2) Point DATA_ROOT at its mount path (env var J1_DATA_ROOT or "
        "edit Step 5).\n"
        "     Layout required: <root>/{train,val}/<wnid>/... (100 class "
        "dirs per split).\n"
        "The trainer would also refuse (Agent I structural validation), "
        "but refusing BEFORE the GPU bill is the point.")
else:
    statuses = _roadmap_status()
    done = sorted(p for p, s in statuses.items() if s == "done")
    if done:
        print(f"  roadmap: already done = {done} — the trainer skips them.",
              flush=True)
        if len(done) == 4:
            print("ALL FOUR FOUNDATION PHASES COMPLETE — nothing to run. "
                  "Next: Agent J2 (steps 5-6, AIS-v2).", flush=True)
    # ONE run per cell execution. The trainer owns: resume gate (local ->
    # HF rolling), provenance manifest, gradient-reach checks, per-epoch
    # rolling + eval artifacts + roadmap sync (per-file HF sync, rev-guarded).
    # NEVER pass --force-restart here: resume is the protocol, restart is a
    # manually-audible local action.
    rc = run(f"python3 {TRAINER} --data-root {DATA_ROOT} --batch-size 96")
    print(f"\n{'='*70}\nrun finished (rc={rc}). "
          f"Re-run this cell to continue the foundation sequence — "
          f"phases already done are skipped.\n{'='*70}", flush=True)

# %% [markdown]
# ## Re-run loop (no code — read me)
#
# The dispatch cell executes the machine's next action and exits. To run
# the full foundation sequence: re-run Step 6 until it prints
# "ALL FOUR FOUNDATION PHASES COMPLETE". Every re-run is resume-safe:
#
#   - killed mid-training  -> resumes from the HF rolling checkpoint at the
#     last completed epoch (optimizer state + scheduler included);
#   - killed after training, before eval -> the roadmap says trained /
#     eval_pending; the trainer's single-run entry re-enters that phase and
#     completes the remaining substeps;
#   - killed mid-eval -> eval artifacts are rewritten atomically; already-
#     done phases are never re-trained.
#
# Monitor: HF repo FerrariKazu/rhan-nxa-checkpoints-rolling carries the
# roadmap (generation1_foundation_roadmap.json) and per-phase rolling
# checkpoints; FerrariKazu/rhan-nxa-checkpoints carries best checkpoints
# and the Agent I eval CSVs/summary as each phase completes.
