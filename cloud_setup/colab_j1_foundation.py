#!/usr/bin/env python3
"""
Colab Notebook — RHAN-NXA GENERATION 1 FOUNDATION (six phases, J2-era)
======================================================================

Colab twin of Kaggle_J1_FOUNDATION.py (identical protocol, platform-native
secrets/paths). Runs the FULL six-phase foundation ladder
(backbone_only -> recurrence_only -> belief_no_f -> belief_with_f ->
ais_v2_swap -> gen1_core) via training/train_generation1_foundation.py.
Steps 1-4 use the PLACEHOLDER fixed gaze schedule; steps 5-6 run the
AIS-v2 swap (Agent F AISv2GazePolicy) — live in the trainer since the J2
commits, so this notebook needs NO companion file anymore.

DURABILITY MODEL (identical to the Kaggle twin):
  - /content is wiped when the runtime dies, so ALL durable state lives on
    HF. The trainer owns the two dedicated repos (Gen-0 contamination is
    structurally impossible — separate repos):
        FerrariKazu/rhan-nxa-checkpoints          (best ckpts)
        FerrariKazu/rhan-nxa-checkpoints-rolling  (rolling ckpts + roadmap)
    THIS notebook additionally syncs the artifacts the trainer leaves
    host-local (per-phase provenance manifests, eval CSVs/result/verdict
    jsons, compactness reports) to the best repo after each run, so a wiped
    session can still be verified end-to-end.
  - Resume: Agent A's resume_or_abort (local first, then the rolling HF
    repo) + resume_guard; a session dying at ANY point resumes by
    RE-RUNNING THE DISPATCH CELL. Never pass --force-restart here.
  - One run per cell execution: the dispatch runs the trainer once (the
    trainer itself executes the machine's next action and exits). Re-run
    it until the Phase-11 verifier prints RUN COMPLETE & VERIFIED.

DATA — fully automated, NO manual upload required:
  scripts/prepare_imagenet100.py downloads the PINNED source
  (clane9/imagenet-100 @ 0519dc2f…), converts to
  <root>/{train,val}/<wnid>/*.jpg, and structurally verifies + fingerprints
  the tree. Resumable per file; ~1-2 h once per runtime. Set
  USE_DRIVE_DATA = True to convert straight into Google Drive so the tree
  survives runtime deaths (recommended for the multi-day ladder).

USAGE:
  1. Colab notebook, runtime = GPU (T4 is fine); Internet ON.
  2. Secrets (key icon in the sidebar) > add 'HF_TOKEN' (name must match
     exactly; enable notebook access when prompted).
  3. Run all cells: deps -> clone -> token -> SMOKE PROOF (fast, synthetic)
     -> DATA bootstrap (first run only) -> DISPATCH. When the roadmap
     reaches 6/6, the dispatch runs the Phase-11 completion verifier.

BATCH/WORKERS: the dispatch mirrors the FROZEN production defaults
(--batch-size 48 --num-workers 4). Override via J1_BATCH / J1_WORKERS env
vars if the runtime struggles — a different batch is a DIFFERENT config
(it enters the provenance hash); never change it mid-ladder.
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
# launch command, exercises skip/verify logic against LIVE HF state, and
# never launches training, touches git state, or writes to HF).
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


def run(cmd, check=True, noisy=True):
    if noisy:
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
                       ("PIL", "Pillow"),
                       ("pyarrow", "pyarrow"),
                       ("dotenv", "python-dotenv")):
        try:
            __import__(_mod)
        except Exception:
            run(f"pip install --quiet {_pkg}")
    # NOTE: the `datasets` library is deliberately NOT installed — the J1/J2
    # trainer, the Agent I eval chain, AND scripts/prepare_imagenet100.py
    # import torch/torchvision/pandas/huggingface_hub/pyarrow/PIL only
    # (verified); installing it would pull a heavy dependency chain for
    # nothing.

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
# ## Step 4: SMOKE PROOF — orchestration chain, synthetic, NOT results

# %%
# One fast end-to-end proof that the SIX-phase chain orchestrates on THIS
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

# %% [markdown]
# ## Step 5: DATA — automated pinned-source bootstrap (or Drive cache)

# %%
# Primary path: scripts/prepare_imagenet100.py pulls the PINNED source
# (clane9/imagenet-100 @ 0519dc2f… — the byte-identity contract of the
# frozen launch manifest), converts to <root>/{train,val}/<wnid>/*.jpg,
# structurally validates (exactly 100 class dirs per split) and writes
# fingerprint.json. Resumable per file: a killed bootstrap resumes by
# re-running this cell. ~1-2 h once per runtime.
#
# USE_DRIVE_DATA = True converts straight into Google Drive, so the tree
# survives runtime deaths (recommended for the multi-day ladder; Drive I/O
# is slower during training, so /content + re-bootstrap is acceptable for
# short sessions). J1_DATA_ROOT env var overrides both.
TRAINER = "training/train_generation1_foundation.py"
CONVERTER = "scripts/prepare_imagenet100.py"
HF_ROLLING = "FerrariKazu/rhan-nxa-checkpoints-rolling"
HF_BEST = "FerrariKazu/rhan-nxa-checkpoints"
ROADMAP_ON_HF = "generation1_foundation_roadmap.json"

USE_DRIVE_DATA = False
DRIVE_DATA_ROOT = "/content/drive/MyDrive/imagenet100"
if DRY_RUN:
    DATA_ROOT = os.environ.get("J1_DATA_ROOT") or "/content/imagenet100"
else:
    if USE_DRIVE_DATA:
        from google.colab import drive
        if not os.path.ismount("/content/drive"):
            drive.mount("/content/drive")
    DATA_ROOT = (os.environ.get("J1_DATA_ROOT")
                 or (DRIVE_DATA_ROOT if USE_DRIVE_DATA
                     else "/content/imagenet100"))


def _data_ok(root):
    if not os.path.isdir(root):
        return False
    try:
        import subprocess as _sp
        _r = _sp.run(f'python3 -c "import sys; sys.path.insert(0, \'.\'); '
                     f'from evaluation.imagenet100_loader import '
                     f'validate_imagenet100_root; '
                     f'validate_imagenet100_root({root!r}, split=\'train\'); '
                     f'validate_imagenet100_root({root!r}, split=\'val\')"',
                     shell=True, capture_output=True, text=True)
        return _r.returncode == 0
    except Exception:
        return False


if DRY_RUN:
    print(f"[DRY-RUN] DATA_ROOT={DATA_ROOT} exists={os.path.isdir(DATA_ROOT)}")
    if not os.path.isdir(DATA_ROOT):
        print(f"[DRY-RUN] would bootstrap: python3 {CONVERTER} --root {DATA_ROOT}")
elif _data_ok(DATA_ROOT):
    print(f"✓ DATA pre-flight OK: {DATA_ROOT} (100/100 class dirs both splits)")
else:
    print(f"== DATA bootstrap (pinned source, resumable) -> {DATA_ROOT} ==")
    run(f"python3 {CONVERTER} --root {DATA_ROOT}")
    if not _data_ok(DATA_ROOT):
        raise SystemExit(
            f"STOP — bootstrap finished but {DATA_ROOT} failed structural "
            "validation. Read the converter output above; do NOT proceed.")

# %% [markdown]
# ## Step 6: Dispatch helpers — durable state on HF, artifact durability

# %%


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


def _upload(local_path, repo_path, repo_id):
    """Per-file durability sync (Kaggle_NOESIS upload_hf_file pattern)."""
    if DRY_RUN:
        print(f"  [DRY-RUN] would sync {local_path} -> {repo_id}:{repo_path}",
              flush=True)
        return False
    if not os.path.exists(local_path):
        return False
    try:
        from huggingface_hub import HfApi
        HfApi(token=hf_token).upload_file(
            path_or_fileobj=local_path, path_in_repo=repo_path,
            repo_id=repo_id, repo_type="dataset", token=hf_token)
        return True
    except Exception as e:
        print(f"  WARNING: could not sync {local_path} to HF: {e}", flush=True)
        return False


def sync_run_artifacts(statuses):
    """Upload the artifacts the trainer leaves host-local so a wiped
    session (and scripts/verify_run_complete.py on ANY host) can still see
    them: per-phase provenance manifests, eval summary CSV, result json,
    compactness report. Rolling ckpts + roadmap are already synced by the
    trainer itself; best ckpts by the trainer too (all six phases)."""
    done = [p for p, s in statuses.items() if s == "done"]
    for ph in done:
        _upload(f"runs/foundation_{ph}/manifest.json",
                f"runs/foundation_{ph}/manifest.json", HF_BEST)
        _upload(f"report/foundation_{ph}_eval/summary_table.csv",
                f"report/foundation_{ph}_eval/summary_table.csv", HF_BEST)
        _upload(f"report/foundation_{ph}_result.json",
                f"report/foundation_{ph}_result.json", HF_BEST)
        _upload(f"report/foundation_{ph}_compactness.json",
                f"report/foundation_{ph}_compactness.json", HF_BEST)
    if done:
        print(f"  ✓ artifact durability sync done for {len(done)} phase(s)",
              flush=True)

# %% [markdown]
# ## Step 7: THE DISPATCH — one run, resume-safe, then verify at 6/6

# %%
# Batch 48 / workers 4 mirror the FROZEN production defaults (the launcher's
# DEFAULTS recorded in the frozen manifest's overrides_applied_to_hash).
BATCH = os.environ.get("J1_BATCH", "48")
WORKERS = os.environ.get("J1_WORKERS", "4")

statuses = _roadmap_status()
done = sorted(p for p, s in statuses.items() if s == "done")
print(f"  roadmap: {len(done)}/6 done {done}", flush=True)

if DRY_RUN:
    print(f"[DRY-RUN] would launch: python3 {TRAINER} --data-root {DATA_ROOT} "
          f"--batch-size {BATCH} --num-workers {WORKERS}")
    print("[DRY-RUN] no training launched, no state mutated.",
          flush=True)
elif not _data_ok(DATA_ROOT):
    raise SystemExit(
        f"STOP — ImageNet-100 not validated at {DATA_ROOT}.\n"
        "  Run Step 5 (DATA bootstrap) first, or point DATA_ROOT at a "
        "converted Drive folder.\n"
        "  Layout required: <root>/{train,val}/<wnid>/... (100 class dirs "
        "per split). The trainer would also refuse (Agent I structural "
        "validation), but refusing BEFORE the GPU bill is the point.")
else:
    # ONE run per cell execution. The trainer owns: resume gate (local ->
    # HF rolling), provenance manifest, gradient-reach checks, per-epoch
    # rolling + eval artifacts + roadmap sync (per-file HF sync).
    # NEVER pass --force-restart here: resume is the protocol, restart is a
    # manually-audible local action.
    rc = run(f"python3 {TRAINER} --data-root {DATA_ROOT} "
             f"--batch-size {BATCH} --num-workers {WORKERS}")
    statuses = _roadmap_status()
    sync_run_artifacts(statuses)
    n_done = sum(1 for s in statuses.values() if s == "done")
    if n_done == 6:
        print("\nALL SIX FOUNDATION PHASES COMPLETE — running Phase-11 "
              "completion verification.", flush=True)
        # On a fresh session the verifier self-heals missing local artifacts
        # from HF (and pulls the frozen manifest from the rolling repo); if
        # this runtime's dataset root differs from the frozen one, the
        # config-hash compare is recorded as SKIPPED in the report notes
        # (per-phase provenance manifests stay authoritative).
        run("python3 scripts/verify_run_complete.py", check=False)
    else:
        print(f"\n{'='*70}\nrun finished (rc={rc}); {n_done}/6 phases done. "
              f"Re-run this cell to continue — phases already done are "
              f"skipped.\n{'='*70}", flush=True)

# %% [markdown]
# ## Re-run loop (no code — read me)
#
# Colab runtimes die (~12 h on free T4); the protocol is built for it:
# re-run Step 7 in a fresh runtime and it resumes from the HF rolling
# checkpoint at the last completed epoch (optimizer state + scheduler
# included). Killed mid-eval: eval artifacts rewrite atomically. Already-
# done phases are never re-trained. The DATA bootstrap re-runs only if no
# valid tree is present (use USE_DRIVE_DATA = True to avoid re-downloading
# after every runtime death). Monitor on HF:
# FerrariKazu/rhan-nxa-checkpoints-rolling (roadmap + rolling ckpts) and
# FerrariKazu/rhan-nxa-checkpoints (best ckpts, per-phase provenance
# manifests, Agent I eval CSVs/summaries).
#
# TIP: keep this notebook's browser tab open; Colab may still disconnect.
# Everything durable is on HF — a disconnect costs you nothing but the
# current epoch's progress.
