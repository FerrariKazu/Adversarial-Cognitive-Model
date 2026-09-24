#!/usr/bin/env python3
"""
Colab Notebook — RHAN-NXA Generation 1 FOUNDATION (Agent J1) Execution
======================================================================

Colab twin of Kaggle_J1_FOUNDATION.py (identical protocol, platform-native
secrets/paths). Runs Part 2 steps 1-4 (backbone_only -> recurrence_only ->
belief_no_f -> belief_with_f) via training/train_generation1_foundation.py,
on the PLACEHOLDER fixed gaze schedule. Steps 5-6 (the AIS-v2 swap) are
J2's file and are deliberately NOT runnable from here.

DURABILITY MODEL (identical to the Kaggle twin):
  - /content is wiped when the runtime dies, so ALL durable state lives on
    HF. The trainer owns the two dedicated repos (Gen-0 contamination is
    structurally impossible — separate repos):
        FerrariKazu/rhan-nxa-checkpoints          (best ckpts + eval CSVs)
        FerrariKazu/rhan-nxa-checkpoints-rolling  (rolling ckpts + roadmap)
  - Resume: Agent A's resume_or_abort (local first, then the rolling HF
    repo) + resume_guard; a session dying at ANY point resumes by
    RE-RUNNING THE DISPATCH CELL. Never pass --force-restart here.
  - One run per cell execution: the dispatch executes the machine's next
    action and exits. Re-run it until it prints
    "ALL FOUR FOUNDATION PHASES COMPLETE".

USAGE:
  1. Colab notebook, runtime = GPU (T4 is fine); Internet ON.
  2. Secrets (key icon in the sidebar) > add 'HF_TOKEN' (name must match
     exactly; enable notebook access when prompted).
  3. Run all cells. First run: SMOKE PROOF cell (fast, synthetic) then the
     dispatch cell. The dispatch STOPs loudly if ImageNet-100 is absent.

DATA (the one thing this notebook does NOT do for you):
  Put ImageNet-100 on Google Drive (folder layout <root>/{train,val}/
  <wnid>/..., exactly 100 class dirs per split), mount Drive in the DATA
  cell, and point DATA_ROOT at it — or upload a zip to /content and extract
  it. The trainer runs Agent I's structural validation BEFORE any launch; a
  wrong layout is a loud STOP, never a silently-wrong result.
"""
# %% [markdown]
# ## Step 1: Environment — fail fast on HF stalls, then deps

# %%
import os, sys, subprocess, json, shutil

# huggingface_hub freezes HF_HUB_DOWNLOAD_TIMEOUT at import time — set it
# BEFORE any hub import, and trainer subprocesses inherit it. A stalled
# download raises in ~30s instead of hanging the session silently.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

# ── PRE-FLIGHT (dry-run) MODE ─────────────────────────────────────────────
# In Colab there is no per-notebook env-var UI before the first cell; set
# DRY_RUN = True in this cell for a pre-flight pass (prints the exact
# launch command, exercises skip logic against LIVE HF state, and never
# launches training, touches git state, or writes to HF).
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
    # Colab PREINSTALLS torch (GPU build) — never upgrade pip/setuptools or
    # reinstall torch over the platform build (the setuptools-83-vs-torch
    # conflict). Only install what is demonstrably missing, one at a time.
    try:
        import torch  # noqa: F401
        if not torch.cuda.is_available():
            raise ImportError("torch present but CUDA unavailable — "
                              "select a GPU runtime and rerun")
    except Exception as e:
        raise RuntimeError(
            f"CUDA-capable torch required before proceeding ({e}). "
            "Runtime > Change runtime type > GPU, then rerun.") from e
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
WORK_DIR = '/content/' + REPO_NAME

if not DRY_RUN:
    # /content is writable scratch, wiped between sessions — hence the HF
    # durability model. Fresh clone each session (simplest correct form).
    os.chdir('/content')
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
# ## Step 3: HF_TOKEN (Colab Secrets) + environment

# %%
import torch
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from google.colab import userdata
        hf_token = userdata.get('HF_TOKEN')
        os.environ["HF_TOKEN"] = hf_token
    except Exception:
        pass
if not hf_token:
    raise RuntimeError("HF_TOKEN not found. Set it in Colab Secrets "
                       "(key icon in the sidebar) as 'HF_TOKEN' — the key "
                       "must match exactly — and enable notebook access.")

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
print(f"✓ HF_TOKEN set for user: {hf_token[:4]}...{hf_token[-4:]}")
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ VRAM: "
      f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## Step 4: DATA — point DATA_ROOT at ImageNet-100 (Drive or local)

# %%
# OPTION A (recommended): Drive. Mount, then set DATA_ROOT to the folder
# that CONTAINS train/ and val/ (100 class dirs each).
USE_DRIVE = True
DATA_ROOT = "/content/drive/MyDrive/imagenet100"

# OPTION B: a zip uploaded to /content (set USE_DRIVE = False, then):
#   DATA_ROOT = "/content/imagenet100"   # after extracting imagenet100.zip
#   !unzip -q /content/imagenet100.zip -d /content/

if DRY_RUN:
    print(f"[DRY-RUN] would mount Drive and use DATA_ROOT={DATA_ROOT}")
elif USE_DRIVE:
    from google.colab import drive
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")

data_ready = os.path.isdir(DATA_ROOT)
print(f"DATA_ROOT={DATA_ROOT}  exists={data_ready}")
if not DRY_RUN and not data_ready:
    print("⚠ ImageNet-100 not found yet — set DATA_ROOT correctly before "
          "running Step 6 (the dispatch will also STOP loudly).",
          flush=True)

# %% [markdown]
# ## Step 5: SMOKE PROOF + dispatch helpers

# %%
# One fast end-to-end proof that the 4-phase chain orchestrates on THIS
# runtime (machine walk, gradient-reach checks, parity, eval CSVs).
# Synthetic loaders, 1 epoch, NO HF writes, NO dataset. Numbers are NOT
# results. Skips itself on re-runs within the same runtime.
_SMOKE_MARK = "/content/.j1_smoke_ok"
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

TRAINER = "training/train_generation1_foundation.py"
HF_ROLLING = "FerrariKazu/rhan-nxa-checkpoints-rolling"
ROADMAP_ON_HF = "generation1_foundation_roadmap.json"


def _roadmap_status():
    """Phase statuses from local roadmap, restoring the HF copy first
    (runtime state beats the fresh-clone baseline)."""
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
if DRY_RUN:
    print(f"[DRY-RUN] roadmap statuses: {_roadmap_status()}")
    print(f"[DRY-RUN] would launch: python3 {TRAINER} "
          f"--data-root {DATA_ROOT} --batch-size 96")
    print("[DRY-RUN] no training launched, no state mutated.")
elif not data_ready:
    raise SystemExit(
        f"STOP — ImageNet-100 not found at {DATA_ROOT}.\n"
        "  Fix DATA_ROOT in Step 4 (Drive folder containing train/ and "
        "val/ with 100 class dirs each), rerun Step 4, then this cell.")
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
    # rolling + eval artifacts + roadmap sync (per-file HF sync).
    # NEVER pass --force-restart here: resume is the protocol, restart is a
    # manually-audible local action.
    rc = run(f"python3 {TRAINER} --data-root {DATA_ROOT} --batch-size 96")
    print(f"\n{'='*70}\nrun finished (rc={rc}). "
          f"Re-run this cell to continue the foundation sequence — "
          f"phases already done are skipped.\n{'='*70}", flush=True)

# %% [markdown]
# ## Re-run loop (no code — read me)
#
# Colab runtimes die (~12 h on free T4); the protocol is built for it:
# re-run Step 6 in a fresh runtime and it resumes from the HF rolling
# checkpoint at the last completed epoch (optimizer state + scheduler
# included). Killed mid-eval: eval artifacts rewrite atomically. Already-
# done phases are never re-trained. Monitor on HF:
# FerrariKazu/rhan-nxa-checkpoints-rolling (roadmap + rolling ckpts) and
# FerrariKazu/rhan-nxa-checkpoints (best ckpts + Agent I eval CSVs).
#
# TIP: keep this notebook's browser tab open; Colab may still disconnect.
# Everything durable is on HF — a disconnect costs you nothing but the
# current epoch's progress.
