#!/usr/bin/env python3
"""
Colab Notebook — RHAN-NXA GENERATION 1 FOUNDATION (six phases, J2-era)
======================================================================

Colab twin of Kaggle_J1_FOUNDATION.py (identical protocol, platform-native
secrets/paths; guards kept in parity 2026-10-02: curriculum assert,
quota pre-flight + history squash, reset gate, stale-state STOP,
adversarial-recipe post-run guard — adapted to /content being WIPED
every session, unlike /kaggle/working). Runs the FULL six-phase foundation ladder
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
        t0 = time.perf_counter()
        run(f'git clone https://github.com/FerrariKazu/{REPO_NAME}.git', noisy=False)
        print(f"HF sync ok (clone {REPO_NAME}, {time.perf_counter()-t0:.1f}s)", flush=True)
    os.chdir(WORK_DIR)
    sys.path.insert(0, WORK_DIR)
    os.environ["PYTHONPATH"] = f"{WORK_DIR}:{os.environ.get('PYTHONPATH', '')}"

    # RHAN-NXA lives on feature/rhan-next. Never reset to origin/main here.
    t0 = time.perf_counter()
    run('git fetch origin', noisy=False)
    _branch_ok = subprocess.run(
        'git ls-remote --heads origin feature/rhan-next',
        shell=True, capture_output=True, text=True).stdout.strip()
    if not _branch_ok:
        raise RuntimeError(
            "feature/rhan-next is NOT on origin. Push it first:\n"
            "  git push origin feature/rhan-next\n"
            "(RHAN-NXA must not be merged to main until J1/J2 validate.)")
    run('git checkout -B feature/rhan-next origin/feature/rhan-next', noisy=False)
    run('git reset --hard origin/feature/rhan-next', noisy=False)
    _sha = subprocess.run('git rev-parse --short HEAD', shell=True,
                          capture_output=True, text=True).stdout.strip()
    print(f"HF sync ok (checkout feature/rhan-next @ {_sha}, "
          f"{time.perf_counter()-t0:.1f}s)", flush=True)
    log_sync_event(f"checkout feature/rhan-next @ {_sha}", logger)

    if not os.path.exists("training/train_generation1_foundation.py"):
        raise RuntimeError(
            "training/train_generation1_foundation.py not found — the "
            "checked-out commit predates Agent J1. Push/verify the J1 "
            "commit on feature/rhan-next first.")
    # The 2026-09-29 recipe correction (commit 33a180b): the trainer must
    # carry the ported TRADES/PGD curriculum. A stale checkout would launch
    # another pure-CE run — the exact bug the current re-run exists to fix.
    # (Guard ported from the Kaggle twin 2026-10-02.)
    _trainer_src = open("training/train_generation1_foundation.py").read()
    if "adv_curriculum" not in _trainer_src or "--clean-only" not in _trainer_src:
        raise RuntimeError(
            "checked-out trainer LACKS the 2026-09-29 curriculum port "
            "(commit 33a180b). git fetch/pull on feature/rhan-next first — "
            "never launch a ladder from pre-correction code.")
    print("✓ trainer carries the 2026-09-29 TRADES/PGD curriculum port",
          flush=True)

# %% [markdown]
# ## Step 3: HF_TOKEN (Colab Secrets) + environment

# %%
import torch
from training.sync_log import configure_sync_logger, log_sync_event
def run(msg: str, *, noisy: bool = True) -> None:
    """run(cmd) helper wrapper already in scope for this notebook; we patch
    the noisy git/chirp behavior here only. Noisy operations log their full
    stdout/stderr to report/sync.log and print one line to stdout."""
    logger = configure_sync_logger()
    if noisy:
        logger.info(f"RUN START: {msg}")
    try:
        import subprocess
        proc = subprocess.run(msg, shell=True, capture_output=True, text=True)
        if noisy:
            if proc.stdout:
                logger.info(f"RUN STDOUT:\n{proc.stdout}")
            if proc.stderr:
                logger.info(f"RUN STDERR:\n{proc.stderr}")
        if proc.returncode != 0:
            raise RuntimeError(f"command failed (rc={proc.returncode}): {msg}\n{proc.stderr}")
    finally:
        if noisy:
            logger.info(f"RUN END: {msg}")

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
# ## Step 4.5: HF GUARDS — quota pre-flight + reset gate (ported from Kaggle)
#
# The Kaggle twin grew three guards after 2026-09-30; this cell ports the
# two that matter on Colab (where /content is WIPED every session):
#   1. STORAGE-QUOTA pre-flight (2026-10-01 outage): a full private-LFS
#      quota makes every checkpoint upload 403 while training marches on
#      undurably. Probe both repos, then squash history unconditionally
#      (tip preserved: archive/, roadmap, manifests, results). Disable
#      with J1_SKIP_QUOTA_CHECK=1 (set it in Step 1's cell before running).
#   2. RERUN-RESET gate: AUTO-FIRES exactly once if the HF roadmap still
#      records the stale pure-CE ladder and no reset archive exists
#      (marker file + archive dir on HF make it strictly one-time).
#      The Kaggle twin's stranded-artifact rescue cell is NOT ported:
#      /content does not survive a session, so stranded files cannot
#      exist here (the Kaggle rescue proved those artifacts were already
#      durable on HF anyway).

# %%
import glob as _glob
import json as _json
import time as _time
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from training.stage_state_machine import (DEPENDENCIES as _DEPS,
                                          FOUNDATION_PHASES as _PHASES)

# Repo ids + roadmap filename are defined HERE, at FIRST USE — this cell
# runs BEFORE Step 5, which previously owned these constants (a fresh
# top-to-bottom session hit NameError: 'HF_ROLLING' is not defined in the
# gate, which then correctly failed closed). Same fix as the Kaggle twin.
HF_ROLLING = "FerrariKazu/rhan-nxa-checkpoints-rolling"
HF_BEST = "FerrariKazu/rhan-nxa-checkpoints"
ROADMAP_ON_HF = "generation1_foundation_roadmap.json"

_api = HfApi(token=hf_token)
STAMP = _time.strftime("%Y%m%d_%H%M%S")
ARCH = f"archive/gen1_pure_ce_{STAMP}"
_RESET_MARKER = ".j1_pure_ce_reset_done"

def _hf_probe(fn, *a, **k):
    """Retry an HF read probe: transient 429/5xx at cell-run time must not
    silently flip the reset gate. Re-raises after the final attempt."""
    last = None
    for _i in range(3):
        try:
            return fn(*a, **k)
        except Exception as _e:  # noqa: BLE001 — retried below
            last = _e
            _time.sleep(20 * (_i + 1))
    raise last

def _reset_already_done() -> bool:
    """One-time guard: a local marker OR the archive dir on HF (from any
    prior session) means the reset must never fire again. Raises if HF is
    unreachable after retries (the caller fails closed)."""
    if os.path.exists(_RESET_MARKER):
        return True
    names = _hf_probe(list_repo_files, HF_ROLLING, repo_type="dataset",
                      token=hf_token)
    return any(n.startswith("archive/gen1_pure_ce_") for n in names)

def _roadmap_has_pure_ce() -> bool:
    """True while the HF roadmap still records any foundation phase as
    done — the stale pure-CE state this reset exists to clear. Raises if
    HF is unreachable after retries (the caller fails closed)."""
    _p = _hf_probe(hf_hub_download, repo_id=HF_ROLLING,
                   filename=ROADMAP_ON_HF, repo_type="dataset",
                   token=hf_token)
    _d = _json.load(open(_p)).get("generation1_foundation", {})
    return any((v or {}).get("status") == "done"
               for v in _d.get("phases", {}).values())

# ── storage-quota pre-flight (2026-10-01 incident) ────────────────────────
if not DRY_RUN and not os.environ.get("J1_SKIP_QUOTA_CHECK"):
    def _quota_used_gb(repo_id: str) -> float:
        info = _hf_probe(_api.repo_info, repo_id, repo_type="dataset",
                         files_metadata=False)
        return float(getattr(info, "used_storage", 0) or 0) / 1e9

    try:
        _used = max(_quota_used_gb(HF_ROLLING), _quota_used_gb(HF_BEST))
        print(f"✓ HF private storage reachable (largest repo "
              f"~{_used:.1f} GB of LFS revisions)", flush=True)
    except Exception as _e:
        raise RuntimeError(
            f"HF storage probe FAILED before training: {_e}\n"
            "A full private-storage quota makes every checkpoint upload "
            "403 while training continues undurably (2026-10-01: 5h of "
            "rolling+best uploads lost this way). Fix quota first — "
            "squash repo history in the HF UI (Settings), delete stale "
            "repos, or set J1_SKIP_QUOTA_CHECK=1 to override.") from _e
    try:
        # Unconditional: once per session start is cheap, and it does not
        # depend on used_storage being populated by the API response.
        _api.super_squash_history(
            repo_id=HF_ROLLING, repo_type="dataset",
            commit_message="pre-flight: squash history to keep LFS quota under the cap")
        _api.super_squash_history(
            repo_id=HF_BEST, repo_type="dataset",
            commit_message="pre-flight: squash history to keep LFS quota under the cap")
        print("✓ history squashed on both checkpoint repos (tip "
              "preserved: archive/, roadmap, manifests, results)",
              flush=True)
    except Exception as _e:
        print(f"⚠ history squash skipped: {_e}\n"
              "  If uploads later fail with 'storage limit reached', "
              "squash manually in each repo's HF Settings.", flush=True)

FORCED_RESET = os.environ.get("J1_RERUN_RESET", "0") == "1"
_probe_ok, _probe_err, _stale, _already = True, "", False, False
if not DRY_RUN and not FORCED_RESET:
    try:
        _stale = _roadmap_has_pure_ce()
        _already = _reset_already_done()
    except Exception as _e:  # noqa: BLE001 — fail CLOSED below
        _probe_ok, _probe_err = False, str(_e)[:160]
AUTO_RESET = (not DRY_RUN and not FORCED_RESET and _probe_ok
              and _stale and not _already)
RERUN_RESET = (FORCED_RESET or AUTO_RESET) and not DRY_RUN

if DRY_RUN:
    print("[DRY-RUN] RERUN RESET exists (auto-fires once on the stale "
          "pure-CE state; J1_RERUN_RESET=1 force-fires). OFF now.",
          flush=True)
elif not RERUN_RESET and not _probe_ok and not FORCED_RESET:
    print(f"RERUN RESET: SKIPPED — HF probes failed after retries "
          f"({_probe_err}). Failing closed: NOTHING was archived or "
          "deleted. If the dispatch below stops with the stale-ladder "
          "message, simply re-run the notebook — this gate retries and "
          "will fire once HF is reachable.", flush=True)
elif not RERUN_RESET and _stale and _already:
    print("RERUN RESET: reset already done (archive/marker present) — "
          "no action needed.", flush=True)
elif not RERUN_RESET:
    print("RERUN RESET: not needed — HF roadmap has no stale pure-CE "
          "state (already reset or in progress).", flush=True)
elif AUTO_RESET:
    print("RERUN RESET: AUTO-FIRED — the HF roadmap still shows the stale "
          "pure-CE ladder and no archive exists yet. Archiving + resetting "
          "exactly once now.", flush=True)
else:
    print("RERUN RESET: forced via J1_RERUN_RESET=1.", flush=True)

if RERUN_RESET:
    _roles = ((HF_BEST, "_best.pth"), (HF_ROLLING, "_rolling.pth"))
    print(f"== RERUN RESET: pure-CE working set -> {HF_ROLLING}:{ARCH}/ ==",
          flush=True)

    # -- 1. archive every foundation checkpoint from both repos -----------
    _archived = 0
    for repo_id, suffix in _roles:
        try:
            names = [f for f in list_repo_files(repo_id, repo_type="dataset",
                                                token=hf_token)
                     if f.startswith("foundation_") and f.endswith(suffix)]
        except Exception as e:
            print(f"  WARNING: cannot list {repo_id}: {e}", flush=True)
            continue
        for name in names:
            try:
                p = hf_hub_download(repo_id=repo_id, filename=name,
                                    repo_type="dataset", token=hf_token)
                dst = os.path.join(ARCH, "checkpoints", name)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(p, dst)
                _api.upload_file(path_or_fileobj=dst, path_in_repo=dst,
                                 repo_id=HF_ROLLING, repo_type="dataset",
                                 token=hf_token)
                _archived += 1
            except Exception as e:
                print(f"  WARNING: archive failed {repo_id}:{name}: {e}",
                      flush=True)
    print(f"  archived {_archived} checkpoint(s)", flush=True)

    # -- 2. archive README --------------------------------------------------
    _readme = os.path.join(ARCH, "README.txt")
    os.makedirs(os.path.dirname(_readme), exist_ok=True)
    with open(_readme, "w") as f:
        f.write(
            "ARCHIVE: Gen-1 foundation pure-CE run (2026-09-25 -> 2026-09-26)\n"
            f"Archived by colab_j1_foundation.py reset at {STAMP}.\n\n"
            "This run trained with PURE cross-entropy: the frozen trainer had\n"
            "NO adversarial term (confirmed 2026-09-29 by code read). The\n"
            "corrected trainer adds the ported Gen-0 TRADES/PGD curriculum\n"
            "(commit 33a180b) — a different recipe, therefore a NEW\n"
            "experiment; the checkpoints could not be resumed.\n\n"
            "Full extracted results of THIS archived run:\n"
            "  report/GEN1_RESULTS_MASTER.md on the rolling repo\n"
            "  (FerrariKazu/rhan-nxa-checkpoints-rolling). The pure-CE arm\n"
            "  remains citable as the clean-CE control for the re-run.\n")
    _api.upload_file(path_or_fileobj=_readme, path_in_repo=_readme,
                     repo_id=HF_ROLLING, repo_type="dataset", token=hf_token)
    print("  archive README written", flush=True)

    # -- 3. delete the pure-CE working set from HF --------------------------
    _deleted = 0
    for repo_id, suffix in _roles:
        try:
            names = [f for f in list_repo_files(repo_id, repo_type="dataset",
                                                token=hf_token)
                     if f.startswith("foundation_") and f.endswith(suffix)]
        except Exception:
            continue
        for name in names:
            try:
                _api.delete_file(name, repo_id, repo_type="dataset",
                                 token=hf_token)
                _deleted += 1
            except Exception as e:
                print(f"  WARNING: delete failed {repo_id}:{name}: {e}",
                      flush=True)
    # per-phase eval/manifest artifacts live on the BEST repo
    for ph in _PHASES:
        for rel in (f"runs/foundation_{ph}/manifest.json",
                    f"report/foundation_{ph}_result.json",
                    f"report/foundation_{ph}_compactness.json",
                    f"report/foundation_{ph}_eval/summary_table.csv",
                    f"report/foundation_{ph}_eval/epsilon_sweep_per_seed.csv",
                    f"report/foundation_{ph}_eval/eval_provenance.json"):
            try:
                _api.delete_file(rel, HF_BEST, repo_type="dataset",
                                 token=hf_token)
                _deleted += 1
            except Exception:
                pass
    # stale run-wide records on the rolling repo (the old frozen manifest
    # would make the verifier 'verify' the OLD config against the NEW run)
    for name in ("run_completion_report.json", "supervisor.log",
                 "production_launch_manifest.json"):
        try:
            _api.delete_file(name, HF_ROLLING, repo_type="dataset",
                             token=hf_token)
            _deleted += 1
        except Exception:
            pass
    print(f"  deleted {_deleted} stale artifact(s) from the working set",
          flush=True)

    # local tree: a reused runtime must not carry stale artifacts either
    for pat in ("checkpoints/foundation_*_best.pth",
                "checkpoints/foundation_*_rolling.pth",
                "report/foundation_*_eval", "report/foundation_*_result.json",
                "report/foundation_*_compactness.json", "runs/foundation_*",
                "report/run_completion_report.json", "report/supervisor.log",
                "runs/production_launch_manifest.json"):
        for m in _glob.glob(pat):
            shutil.rmtree(m, ignore_errors=True) if os.path.isdir(m) \
                else os.remove(m)

    # -- 4. reset the roadmap ON HF (rev bumped = newer everywhere) --------
    _old_rev = 0
    try:
        p = hf_hub_download(repo_id=HF_ROLLING, filename=ROADMAP_ON_HF,
                            repo_type="dataset", token=hf_token)
        _old_rev = int(json.load(open(p)).get("roadmap_rev", 0) or 0)
    except Exception:
        pass
    _fresh = {
        "schema_version": 1,
        "phases_order": list(_PHASES),
        "current_phase": _PHASES[0],
        "current_substep": "not_started",
        "roadmap_rev": _old_rev + 1,
        "reset": {
            "from": "gen1-pure-ce-2026-09-25",
            "at_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
            "reason": ("pure-CE run had no adversarial term; corrected "
                       "recipe (33a180b) = new experiment. Archive: "
                       f"{ARCH}/"),
        },
        "generation1_foundation": {
            "schema_version": 1,
            "phases_order": list(_PHASES),
            "current_phase": _PHASES[0],
            "current_substep": "not_started",
            "phases": {ph: {"status": "not_started",
                            "depends_on": _DEPS[ph]} for ph in _PHASES},
        },
    }
    os.makedirs(os.path.dirname(ROADMAP_LOCAL) or ".", exist_ok=True)
    with open(ROADMAP_LOCAL, "w") as f:
        json.dump(_fresh, f, indent=2, ensure_ascii=False)
        f.write("\n")
    _api.upload_file(path_or_fileobj=ROADMAP_LOCAL,
                     path_in_repo=ROADMAP_ON_HF, repo_id=HF_ROLLING,
                     repo_type="dataset", token=hf_token)
    with open(_RESET_MARKER, "w") as _mf:
        _mf.write(f"pure-CE ladder archived to {ARCH} at {STAMP}\n")
    print(f"  roadmap reset: 6 phases -> not_started, roadmap_rev "
          f"{_old_rev} -> {_old_rev + 1} (HF + local)", flush=True)
    print(f"\nRESET DONE. Pure-CE run archived at {HF_ROLLING}:{ARCH}/.\n"
          "The reset is one-time: the marker file plus the archive dir on "
          "HF prevent it from ever firing again. Continue with Step 5 "
          "(DATA) and Step 7 (dispatch).", flush=True)

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
# HF_ROLLING / HF_BEST / ROADMAP_ON_HF are defined in Step 4.5 (their
# first use — that cell runs before this one; a duplicate definition here
# would risk the two copies diverging).

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


def assert_adversarial_recipe(statuses):
    """Loud post-run guard: every 'done' phase's manifest must record the
    adversarial recipe (clean_only=False). A phase that completed without
    it means the pure-CE gap recurred — STOP before artifacts accumulate
    (this is the audit that was missing in the 2026-09-25->26 run).

    Colab adaptation: /content is wiped every session, so a done phase's
    manifest is often NOT on local disk — fetch it from the best repo
    (small JSON) before judging. A done phase with the manifest on NEITHER
    local disk NOR HF is the genuine alarm (fail closed).
    """
    bad = []
    for ph, st in sorted(statuses.items()):
        if st != "done":
            continue
        local = f"runs/foundation_{ph}/manifest.json"
        try:
            if not os.path.exists(local):
                _p = hf_hub_download(repo_id=HF_BEST, filename=local,
                                     repo_type="dataset", token=hf_token)
                os.makedirs(os.path.dirname(local), exist_ok=True)
                shutil.copyfile(_p, local)
            man = json.load(open(local))
            rec = man.get("adv_curriculum", {})
            if rec.get("clean_only") is not False:
                bad.append(f"{ph}: clean_only={rec.get('clean_only')!r}")
        except Exception as e:
            bad.append(f"{ph}: manifest unreadable/missing ({e})")
    if bad:
        raise SystemExit(
            "STOP — phase(s) completed WITHOUT the adversarial recipe:\n  "
            + "\n  ".join(bad)
            + "\nThe 2026-09-25->26 pure-CE gap has recurred; do NOT "
              "continue the ladder. Check the checked-out commit.")
    if statuses:
        print("  ✓ adversarial recipe verified in every done phase's "
              "manifest", flush=True)


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

# STALE-STATE STOP (ported from the Kaggle twin, Colab-adapted): the Kaggle
# version keys on MISSING LOCAL manifests, but on Colab a fresh runtime has
# no local manifests even for a healthy ladder. The honest stale signature
# is: HF roadmap says 6/6 done AND no reset archive exists on HF (the one-
# time reset demonstrably never ran — its HF probes failed closed).
if (not DRY_RUN) and len(done) == 6:
    try:
        _names6 = list_repo_files(HF_ROLLING, repo_type="dataset",
                                  token=hf_token)
        if not any(n.startswith("archive/gen1_pure_ce_") for n in _names6):
            raise SystemExit(
                "STOP — HF roadmap says 6/6 done but no reset archive exists "
                "on the rolling repo: this is the STALE pure-CE ladder (the "
                "Step 4.5 auto-reset did not fire — its HF probes failed and "
                "it fails closed). Re-run the notebook: the gate retries and "
                "fires once HF is reachable, or force with J1_RERUN_RESET=1.")
    except SystemExit:
        raise
    except Exception as _e:
        print(f"⚠ stale-state check skipped (HF unreachable: {_e})", flush=True)

# %% [markdown]
# ## Step 6.5: mid-flight durability audit (quota-403 forensics, ported)
#
# The Kaggle twin rescued STRANDED LOCAL files after the 2026-10-01 outage;
# /content is wiped every session so nothing can be stranded HERE. The
# remaining risk is HF-side: a rolling ckpt whose LAST upload 403'd (older
# epoch than the roadmap's recorded best, or a pre-hardening commit id).
# Audit the live HF rolling repo BEFORE dispatch; marker-gated per runtime.

# %%
if not DRY_RUN and not os.path.exists("/content/.j1_rolling_audit_done"):
    try:
        _names = [n for n in list_repo_files(HF_ROLLING, repo_type="dataset",
                                             token=hf_token)
                  if n.startswith("foundation_") and n.endswith("_rolling.pth")]
        # Deep-check ONLY the resume-critical artifact: the rolling ckpt of
        # the phase the machine is about to run/resume. It must record the
        # SAME short commit as this checkout (what resume_commit_ok will
        # enforce) — otherwise the session would abort at the gate after
        # paying for the data bootstrap.
        _cur = subprocess.run("git rev-parse --short HEAD", shell=True,
                              capture_output=True, text=True).stdout.strip()
        _road_p = hf_hub_download(repo_id=HF_ROLLING, filename=ROADMAP_ON_HF,
                                  repo_type="dataset", token=hf_token)
        _cur_ph = _json.load(open(_road_p)).get("current_phase")
        _target = f"foundation_{_cur_ph}_rolling.pth"
        _prob = []
        if _target in _names and _cur:
            _ck = torch.load(
                hf_hub_download(repo_id=HF_ROLLING, filename=_target,
                                repo_type="dataset", token=hf_token),
                map_location="cpu", weights_only=False)
            if _ck.get("code_commit") != _cur:
                _prob.append(f"{_target}: code_commit={_ck.get('code_commit')!r} "
                             f"vs checkout {_cur!r} — the resume gate will "
                             "REFUSE this session; fix the checkout before "
                             "burning runtime on the data bootstrap.")
        for _s in _prob:
            print(f"⚠ {_s}", flush=True)
        print(f"✓ rolling audit: {len(_names)} rolling ckpt(s) on HF; "
              f"current phase {_cur_ph!r} checked against checkout {_cur!r}",
              flush=True)
    except Exception as _e:
        print(f"⚠ rolling audit skipped: {_e}", flush=True)
    open("/content/.j1_rolling_audit_done", "w").write("ok\n")

# %%
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
    assert_adversarial_recipe(statuses)
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
