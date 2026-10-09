#!/usr/bin/env python3
"""
Kaggle Notebook — RHAN-NXA GENERATION 1 FOUNDATION (six phases, J2-era)
=======================================================================

Runs the FULL six-phase foundation ladder
(backbone_only -> recurrence_only -> belief_no_f -> belief_with_f ->
ais_v2_swap -> gen1_core) via training/train_generation1_foundation.py.
Steps 1-4 use the PLACEHOLDER fixed gaze schedule; steps 5-6 run the
AIS-v2 swap (Agent F AISv2GazePolicy) — live in the trainer since the J2
commits, so this notebook needs NO companion file anymore.

This replaces the J1-era 4-phase version (the run that launched 2026-09-25
from the local RTX 4060 has since COMPLETED all six phases; verification
initially false-failed on a verifier data_root bug, fixed 2026-09-29).

DURABILITY MODEL (mirrors Kaggle_NOESIS.py):
  - /kaggle/working is wiped between sessions, so ALL durable state lives
    on HF. The trainer itself owns the two dedicated repos (Gen-0
    contamination is structurally impossible — separate repos):
        FerrariKazu/rhan-nxa-checkpoints          (best ckpts)
        FerrariKazu/rhan-nxa-checkpoints-rolling  (rolling ckpts + roadmap)
    THIS notebook additionally syncs the artifacts the trainer leaves
    host-local (per-phase provenance manifests, eval CSVs/result/verdict
    jsons, compactness reports) to the best repo after each run, so a
    wiped session can still be verified end-to-end.
  - Resume: the trainer calls Agent A's resume_or_abort (local first, then
    the rolling HF repo) + resume_guard; a session dying at ANY point
    resumes by RE-RUNNING THE DISPATCH CELL. Never pass --force-restart.
  - One run per cell execution: the dispatch runs the trainer once (the
    trainer itself executes the machine's next action and exits). Re-run
    the cell (or the notebook) to continue. Phases already done in the
    HF-synced roadmap are skipped.

DATA — fully automated, NO manual dataset attach required:
  scripts/prepare_imagenet100.py downloads the PINNED source
  (clane9/imagenet-100 @ 0519dc2f…), converts to
  <root>/{train,val}/<wnid>/*.jpg, and structurally verifies + fingerprints
  the tree. Resumable per file; ~1-2 h once per runtime (or attach a
  pre-converted Kaggle dataset and point J1_DATA_ROOT at it to skip).
  DEFAULT root is /kaggle/tmp/imagenet100 — writable, and it shares the
  ~57 GB ephemeral scratch pool instead of /kaggle/working's 20 GB
  PERSISTED quota (the dataset alone is ~19 GB; the ladder's checkpoints
  would push a working-dir dataset over quota mid-run). /kaggle/input is
  a read-only mount and cannot be written at all; a read-only ATTACH
  still works when it structurally validates.

USAGE:
  1. Kaggle Notebook, accelerator = GPU (T4 x2 or P100); Internet ON.
  2. Add-ons > Secrets > add 'HF_TOKEN' (key must match exactly).
  3. Run all cells: deps -> clone -> token -> SMOKE PROOF (fast, synthetic)
     -> DATA bootstrap (first run only) -> DISPATCH. When the roadmap
     reaches 6/6, the dispatch runs the Phase-11 completion verifier.
  4. Pre-flight without spending compute: set environment variable
     NOESIS_DRY_RUN=1 (Kaggle: Add-ons > Environment variables) — prints
     the exact launch commands, exercises skip/verify logic against LIVE
     HF state, and never launches training, touches git state, or writes
     to HF (roadmap writes are shielded to a scratch copy).

BATCH/WORKERS: the dispatch mirrors the FROZEN production defaults
(--batch-size 48 --num-workers 4). Override via J1_BATCH / J1_WORKERS env
vars if the runtime struggles — a different batch is a DIFFERENT config
(it enters the provenance hash); never change it mid-ladder.

RERUN (2026-09-29 RECIPE CORRECTION):
The 2026-09-25->26 ladder trained PURE cross-entropy — the trainer had no
adversarial term (confirmed 2026-09-29; see report/GEN1_RESULTS_MASTER.md
on the rolling repo). The corrected trainer (commit 33a180b) ports Gen-0's
TRADES/PGD curriculum and DEFAULTS to it; --clean-only is never passed
here. Step 2 refuses commits that predate the correction, Step 4.5 is the
ONE-TIME audible HF reset for the stale pure-CE state — it AUTO-FIRES
when the HF roadmap still shows the stale pure-CE ladder and no reset
archive exists yet (no environment variable needed: a marker file plus
the archive dir on HF make it fire exactly once across sessions;
J1_RERUN_RESET=1 still force-fires manually), and the dispatch VERIFIES
after every run that each done phase's manifest records the adversarial
recipe — the pure-CE gap cannot silently recur.
"""
# %% [markdown]
# ## Step 1: Environment — fail fast on HF stalls, then deps

# %%
import os, sys, subprocess, json, shutil

# ── PRE-FLIGHT (dry-run) MODE ──────────────────────────────────────────────
# Used by NOESIS_DRY_RUN=1 (Kaggle: Add-ons > Environment variables).
# When set, every run() is a no-op print and every HF/git mutation is skipped
# or shielded, so pre-flight can validate logic against LIVE HF state without
# spending compute or mutating protocol state.
DRY_RUN = os.environ.get("NOESIS_DRY_RUN", "0") == "1"

# The notebook is executed by papermill cell-by-cell. 'training.*' imports are
# reached from this same top-level cell block only AFTER the repo bootstrap cell
# has appended the cloned WORK_DIR to sys.path. To keep 'training.sync_log'
# importable even if a cell ordering/regeneration changes, make the bootstrap code
# set sys.path here as early as possible and keep it idempotent.
_REPO_ROOT_ON_PATH = os.environ.get("REPO_ROOT") or os.environ.get("PROJECT_ROOT")
if _REPO_ROOT_ON_PATH and os.path.isdir(_REPO_ROOT_ON_PATH):
    _p = os.path.abspath(_REPO_ROOT_ON_PATH)
    if _p not in sys.path:
        sys.path.insert(0, _p)
        os.environ.setdefault(
            "PYTHONPATH",
            f"{_p}:{os.environ.get('PYTHONPATH', '')}".strip(":"))
else:
    # Best-effort: if we already have a local './training' directory, use it.
    if os.path.isdir("training"):
        _p = os.path.abspath(".")
        if _p not in sys.path:
            sys.path.insert(0, _p)


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
WORK_DIR = f'/kaggle/working/{REPO_NAME}'

if not DRY_RUN:
    # /kaggle/working is the writable scratch dir (wiped between sessions —
    # hence the HF durability model). Never put the repo on /kaggle/input.
    os.chdir('/kaggle/working')
    if not os.path.exists(WORK_DIR):
        t0 = __import__('time').perf_counter()
        run(f'git clone https://github.com/FerrariKazu/{REPO_NAME}.git', noisy=False)
        print(f"HF sync ok (clone {REPO_NAME}, {__import__('time').perf_counter()-t0:.1f}s)", flush=True)
    os.chdir(WORK_DIR)

    # RHAN-NXA lives on feature/rhan-next. Never reset to origin/main here.
    t0 = __import__('time').perf_counter()
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
          f"{__import__('time').perf_counter()-t0:.1f}s)", flush=True)

    # Make the cloned repo importable before any training.* import.
    # Papermill executes cells top-to-bottom, so 'training.sync_log' is reached
    # while we are still in the repo bootstrap cell; sys.path must already carry
    # WORK_DIR for that import to resolve.
    sys.path.insert(0, WORK_DIR)
    os.environ["PYTHONPATH"] = f"{WORK_DIR}:{os.environ.get('PYTHONPATH', '')}"

    # Sync-log + noisy run() helper only become importable AFTER the repo has
    # been cloned and WORK_DIR added to sys.path (same cell, right here). This
    # is what lets Step 1 itself run before the clone on a fresh Kaggle "Run
    # all" — the sync_log import no longer happens before WORK_DIR exists.
    from training.sync_log import configure_sync_logger, log_sync_event as log_sync

    logger = configure_sync_logger()

    def run(msg: str, *, noisy: bool = True) -> None:
        """run(cmd) wrapper: noisy git/empty-commit/HF-sync chatter goes to
        report/sync.log; only one status line is printed per sync."""
        logger = configure_sync_logger()
        if noisy:
            logger.info(f"RUN START: {msg}")
        try:
            proc = subprocess.run(msg, shell=True, capture_output=True, text=True)
            if noisy:
                if proc.stdout:
                    logger.info(f"RUN STDOUT:\n{proc.stdout}")
                if proc.stderr:
                    logger.info(f"RUN STDERR:\n{proc.stderr}")
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, msg, proc.stderr)
        finally:
            if noisy:
                logger.info(f"RUN END: {msg}")

    log_sync_event(f"checkout feature/rhan-next @ {_sha}", logger)

    if not os.path.exists("training/train_generation1_foundation.py"):
        raise RuntimeError(
            "training/train_generation1_foundation.py not found — the "
            "checked-out commit predates Agent J1. Push/verify the J1 "
            "commit on feature/rhan-next first.")
    # The 2026-09-29 recipe correction (commit 33a180b): the trainer must
    # carry the ported TRADES/PGD curriculum. A stale checkout would launch
    # another pure-CE run — the exact bug this re-run exists to fix.
    _trainer_src = open("training/train_generation1_foundation.py").read()
    if "adv_curriculum" not in _trainer_src or "--clean-only" not in _trainer_src:
        raise RuntimeError(
            "checked-out trainer LACKS the 2026-09-29 curriculum port "
            "(commit 33a180b). git fetch/pull on feature/rhan-next first — "
            "never launch a ladder from pre-correction code.")
    print("✓ trainer carries the 2026-09-29 TRADES/PGD curriculum port",
          flush=True)

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
# One fast end-to-end proof that the SIX-phase chain orchestrates on THIS
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
# ## Step 4.5: RERUN RESET — archive + clear the pure-CE run ON HF (one-time)
#
# The stale pure-CE ladder lives ON HF: a 6/6-done roadmap plus foundation_*
# checkpoints in both repos. Left in place it would poison the re-run (the
# resume gate would restore/refuse against old-code rolling checkpoints;
# the completion verifier would self-heal stale eval CSVs). This cell is
# the AUDIBLE, HF-native reset:
#   1. copies every foundation_* checkpoint on both repos into
#      archive/gen1_pure_ce_<stamp>/checkpoints/ on the ROLLING repo;
#   2. writes an archive README (what the run was, where the results doc
#      lives — the pure-CE arm stays citable as a control);
#   3. DELETES the pure-CE checkpoints, per-phase eval/manifest artifacts,
#      and the stale frozen-launch/completion/supervisor records;
#   4. resets the roadmap to all not_started with roadmap_rev BUMPED, so
#      every host's rev guard treats the reset as the newer state.
#
# AUTO-FIRES EXACTLY ONCE: fires when the HF roadmap still records the
# stale pure-CE ladder AND no reset archive exists yet. A marker file in
# the clone plus the archive dir on HF make it strictly one-time across
# sessions. J1_RERUN_RESET=1 still force-fires (manual override).
# Destructive to the pure-CE working set (archived first); not designed
# to be idempotent past the first run.

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
# gate, which then correctly failed closed).
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
# The account's private LFS quota fills with EVERY overwritten revision of
# an uploaded file (rolling .pth per epoch). When it is full, every
# checkpoint upload 403s while training marches on undurably — the exact
# silent-restart hazard this notebook exists to prevent. Probe BEFORE any
# training starts; squash history first (tip is preserved: archive,
# roadmap, manifests, results). Disable with J1_SKIP_QUOTA_CHECK=1.
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
            f"Archived by Kaggle_J1_FOUNDATION.py reset at {STAMP}.\n\n"
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
# ## Step 5: DATA — automated pinned-source bootstrap (or attach your own)

# %%
# Primary path: scripts/prepare_imagenet100.py pulls the PINNED source
# (clane9/imagenet-100 @ 0519dc2f… — the byte-identity contract of the
# frozen launch manifest), converts to <root>/{train,val}/<wnid>/*.jpg,
# structurally validates (exactly 100 class dirs per split) and writes
# fingerprint.json. Resumable per file: a killed bootstrap resumes by
# re-running this cell. ~1-2 h once per runtime.
#
# Fast path: if DATA_ROOT already contains a valid tree (e.g. you attached
# a pre-converted Kaggle dataset), the bootstrap is skipped.
TRAINER = "training/train_generation1_foundation.py"
CONVERTER = "scripts/prepare_imagenet100.py"
# HF_ROLLING / HF_BEST / ROADMAP_ON_HF are defined in Step 4.5 (their
# first use — that cell runs before this one; a duplicate definition here
# would risk the two copies diverging).

DATA_ROOT = os.environ.get("J1_DATA_ROOT", "/kaggle/tmp/imagenet100")
# NOTE: /kaggle/tmp is writable EPHEMERAL scratch with ~57 GB (the dataset
# is ~19 GB; /kaggle/working's persisted quota is only 20 GB — a dataset
# there risks DiskQuotaExceeded mid-ladder). /kaggle/input is a read-only
# mount: the first re-run attempt died there with `OSError: [Errno 30]
# Read-only file system` because the bootstrap tried to CREATE the tree
# under it. A read-only ATTACH (existing, structurally valid) is still
# fine — the _data_ok branch accepts it before any write is attempted.


def _data_stop(reason: str):
    """The one STOP with the one guidance, for every unwritable-root case."""
    raise SystemExit(
        f"STOP — {reason}\n"
        "The data bootstrap must CREATE/own its root. Pick one:\n"
        "  a) unset J1_DATA_ROOT and use the default /kaggle/tmp/imagenet100\n"
        "     (created on demand; ~57 GB ephemeral scratch);\n"
        "  b) attach a pre-converted dataset and set J1_DATA_ROOT to it\n"
        "     (works as-is when it validates 100/100 classes);\n"
        "  c) copy a read-only attach into scratch first:\n"
        "       !cp -r <read-only-root> /kaggle/tmp/imagenet100\n"
        "     then set J1_DATA_ROOT=/kaggle/tmp/imagenet100.")


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
    # The bootstrap CREATES the tree — it needs a WRITABLE root. Two real
    # failure modes, both hit in this re-run: (1) /kaggle/input is read-only
    # EVEN for creating dirs (EROFS); (2) a MISSING parent makes plain
    # os.access(W_OK) return False (ENOENT != EACCES) — the default
    # /kaggle/tmp/imagenet100 was wrongly rejected on a runtime that had no
    # /kaggle/tmp yet. So probe by DOING: create the root (the converter
    # only writes INSIDE it), then require real writability. One path, one
    # STOP, one guidance (the old exists-but-readonly elif folded in here).
    try:
        os.makedirs(DATA_ROOT, exist_ok=True)
    except OSError as e:
        _data_stop(f"cannot create {DATA_ROOT} ({e}).")
    if not os.access(DATA_ROOT, os.W_OK):
        _data_stop(f"{DATA_ROOT} exists but is NOT writable (read-only "
                   "/kaggle/input mount?).")
    # The converter's hf_hub_download calls use the DEFAULT cache
    # (~/.cache/huggingface, ~19 GB of parquet) — pin it next to the output
    # on the same big ephemeral disk so nothing lands in the container
    # overlay mid-download.
    os.environ.setdefault("HF_HUB_CACHE", "/kaggle/tmp/hf_cache")
    print(f"== DATA bootstrap (pinned source, resumable) -> {DATA_ROOT} ==")
    run(f"python3 {CONVERTER} --root {DATA_ROOT}")
    # The parquet cache is dead weight once the jpg tree + fingerprint exist
    # (resume is keyed on the jpgs, not the cache) — free ~19 GB for
    # checkpoints. Only after FULL success.
    shutil.rmtree(os.environ["HF_HUB_CACHE"], ignore_errors=True)
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
    (this is the audit that was missing in the 2026-09-25->26 run)."""
    bad = []
    for ph, st in sorted(statuses.items()):
        if st != "done":
            continue
        try:
            man = json.load(open(f"runs/foundation_{ph}/manifest.json"))
            rec = man.get("adv_curriculum", {})
            if rec.get("clean_only") is not False:
                bad.append(f"{ph}: clean_only={rec.get('clean_only')!r}")
        except Exception as e:
            bad.append(f"{ph}: manifest unreadable ({e})")
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
# ## Step 6.5: STRANDED-ARTIFACT RESCUE (one-time, 2026-10-01 quota outage)
#
# backbone_only finished DURING the storage-quota 403 window: its best
# (val 0.0588) and epoch-60 rolling were written locally while every HF
# upload failed, and a running dispatch cell cannot be interrupted to
# re-upload them. /kaggle/working PERSISTS across sessions, so this cell —
# run at the start of the next session, BEFORE the dispatch — pushes the
# stranded files to HF from the restored working dir. Marker-gated: at
# most once. If the working dir was wiped instead, this prints what is
# missing; the roadmap on HF (small-file syncs never stopped) keeps the
# recorded best_val_acc as the honest record either way.

# %%
import glob as _glob
_RESCUE_MARKER = "/kaggle/working/.j1_stranded_rescue_done"
if not DRY_RUN and not os.path.exists(_RESCUE_MARKER):
    _JOBS = [
        # (local rel path, HF repo path, repo) — ckpts go to repo ROOT,
        # matching the trainer's upload layout exactly.
        ("checkpoints/foundation_backbone_only_best.pth",
         "foundation_backbone_only_best.pth", HF_BEST),
        ("checkpoints/foundation_backbone_only_rolling.pth",
         "foundation_backbone_only_rolling.pth", HF_ROLLING),
        ("checkpoints/foundation_recurrence_only_best.pth",
         "foundation_recurrence_only_best.pth", HF_BEST),
        ("checkpoints/foundation_recurrence_only_rolling.pth",
         "foundation_recurrence_only_rolling.pth", HF_ROLLING),
    ]
    # Small provenance/eval artifacts for the outage phases (local-only
    # until the end-of-ladder sync would have run).
    _JOBS += [(p, p, HF_BEST) for p in sorted(_glob.glob(
        "runs/foundation_*/manifest.json"))]
    _JOBS += [(p, p, HF_BEST) for p in sorted(_glob.glob(
        "report/foundation_backbone_only_result.json"))]
    _JOBS += [(p, p, HF_BEST) for p in sorted(_glob.glob(
        "report/foundation_backbone_only_compactness.json"))]
    _JOBS += [(p, p, HF_BEST) for p in sorted(_glob.glob(
        "report/foundation_backbone_only_eval/*"))]
    _todo = [(_l, _r, _repo) for (_l, _r, _repo) in _JOBS
             if os.path.exists(_l)]
    if _todo:
        print(f"== RESCUE: uploading {len(_todo)} stranded artifact(s) to "
              "HF (2026-10-01 quota outage) ==", flush=True)
        for _l, _r, _repo in _todo:
            _upload(_l, _r, _repo)
    else:
        print("RESCUE: no stranded artifacts found in /kaggle/working "
              "(working dir was wiped, or no outage artifact existed).",
              flush=True)
    open(_RESCUE_MARKER, "w").write("ok\n")

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

if (not DRY_RUN) and len(done) == 6 and not os.path.exists(
        f"runs/foundation_{done[0]}/manifest.json"):
    # HF says 6/6 done but this clone has no local phase manifests: the
    # STALE pure-CE state. Dispatching now would silently do nothing (the
    # trainer skips done phases) or, worse, let the verifier bless the old
    # run. The corrected re-run must start at Step 4.5.
    raise SystemExit(
        "STOP — HF roadmap says 6/6 done but this clone has NO local phase "
        "manifests, and the Step 4.5 auto-reset did not clear it. Either "
        "the gate's HF probes failed this run (it fails closed and prints "
        "a SKIPPED line above — just re-run the notebook) or a previous "
        "reset completed its archive but failed the roadmap write. Inspect "
        f"{HF_ROLLING}:archive/gen1_pure_ce_*/ and the roadmap JSON; once "
        "the HF state is understood, re-run normally or force with "
        "J1_RERUN_RESET=1.")

if DRY_RUN:
    print(f"[DRY-RUN] would launch: python3 {TRAINER} --data-root {DATA_ROOT} "
          f"--batch-size {BATCH} --num-workers {WORKERS}")
    print("[DRY-RUN] no training launched, no state mutated.",
          flush=True)
elif not _data_ok(DATA_ROOT):
    raise SystemExit(
        f"STOP — ImageNet-100 not validated at {DATA_ROOT}.\n"
        "  Run Step 5 (DATA bootstrap) first, or attach a pre-converted "
        "dataset and point J1_DATA_ROOT at it.\n"
        "  Layout required: <root>/{train,val}/<wnid>/... (100 class dirs "
        "per split). The trainer would also refuse (Agent I structural "
        "validation), but refusing BEFORE the GPU bill is the point.")
else:
    # ONE run per cell execution. The trainer owns: resume gate (local ->
    # HF rolling), provenance manifest, gradient-reach checks, per-epoch
    # rolling + eval artifacts + roadmap sync (per-file HF sync, rev-guarded).
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
# The dispatch cell runs the trainer ONCE per execution (the trainer itself
# executes the machine's next action). To run the full six-phase ladder:
# re-run Step 7 until it prints "ALL SIX FOUNDATION PHASES COMPLETE" and the
# Phase-11 verifier passes. Every re-run is resume-safe:
#
#   - killed mid-training  -> resumes from the HF rolling checkpoint at the
#     last completed epoch (optimizer state + scheduler included);
#   - killed after training, before eval -> the roadmap says trained /
#     eval_pending; the trainer's single-run entry re-enters that phase and
#     completes the remaining substeps;
#   - killed mid-eval -> eval artifacts are rewritten atomically; already-
#     done phases are never re-trained;
#   - session wiped entirely -> fresh clone + HF roadmap restore + (if the
#     bootstrap ran before) HF artifacts re-materialize; the DATA bootstrap
#     re-runs only if no valid tree is present.
#
# ONE-TIME for this re-run: Step 4.5 AUTO-FIRED once the stale pure-CE
# ladder was detected (marker-gated, exactly once), archived it to
# archive/gen1_pure_ce_*/ on the rolling repo and reset the roadmap. The
# dispatch verifies after EVERY run that each done phase's manifest
# carries the adversarial recipe — a silent
# pure-CE recurrence stops the ladder loudly instead of costing 36 GPU-hours.
#
# Monitor: HF repo FerrariKazu/rhan-nxa-checkpoints-rolling carries the
# roadmap (generation1_foundation_roadmap.json) and per-phase rolling
# checkpoints; FerrariKazu/rhan-nxa-checkpoints carries best checkpoints,
# per-phase provenance manifests and the Agent I eval CSVs/summaries.
