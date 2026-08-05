#!/usr/bin/env python3
"""
Colab Notebook — RHAN-Next Stage 1 Execution (Pillar 2: Active Information-Seeking)
====================================================================================
Runs the pre-registered Stage 1 protocol for RHANNext(enable_ais=True,
enable_hpc=False). Training lives HERE (cloud), not locally: the RTX 4060
measures ~2.7-9.2 epochs/hour on this pipeline, so 10-15 epochs is 2-5h and
the full 60-epoch run is 7-20h — far over the 1-hour local budget.

  STEP A — SMOKE (bounded, cheap, catches bugs before commitment):
      train_rhan_next.py --enable-ais --ckpt-name rhan_next_ais_smoke
      --max-epochs 15 (all in phase 1 => epsilon 0.031 ONLY), resuming from
      the SAME base checkpoint used by every prior isolation experiment:
      checkpoints/rhan_stl10_large_pseudolabel_best.pth.
      Every epoch logs the v12 diagnostic block PLUS the two AIS signals:
        - gaze shift distance |Δa| per step + total path;
        - per-sample halting variance (effective evidence steps = sum of the
          soft continuation weights; std > 0 means halting now varies per
          sample, unlike v10/v11's permanently flat steps=4.00);
        - Π_D per class (car/truck must still emerge as highest).
      Machine-readable telemetry is appended to report/rhan_next_ais_smoke_diag.jsonl
      (--diag-json). The HEALTH GATE below reads the last epoch and aborts
      Step B (with reasons) if anything looks degenerate.

  STEP B — FULL VALIDATED RUN:
      Same trainer, --ckpt-name rhan_next_ais, --max-epochs 60. The 3-phase
      curriculum (1-20 @0.031, 21-40 @0.062, 41-60 @0.094) is byte-identical
      to train_rhan_v11.py's — the exact boundaries of the null_ablation_v11
      run that produced 31.56±2.88 @ ε=0.094 — so the result is directly
      comparable. Same base checkpoint, NEVER --force-restart (the trainer's
      mandatory HF resume gate protects against silent restarts).

  STEP C — VALIDATION (5-seed matched eval through the hardened entrypoint):
      python3 phase2_attacks/eval_rhan.py \
          --ckpt-specs rhan_next_ais:checkpoints/rhan_next_ais_best.pth:next \
                       trades_large_baseline:checkpoints/rhan_stl10_large_pseudolabel_best.pth:large \
          --seeds 41 42 43 44 45 --eps-list 0.000 0.094 --n-samples 300
      eval_rhan.py forces norm-space and requires >= 5 seeds (satisfied).
      The verdict is parsed from eval_provenance.json (written automatically
      with git SHA, checkpoint hashes, seed list, timestamp, results, and the
      recomputed Δ>2·σ_combined crossover verdicts) and recorded into
      docs/rhan_next_roadmap.json exactly as Stage 1's acceptance criteria
      specify — a null result is still a valid, reportable outcome.
      RUNTIME: 2 models × 5 seeds × 2 eps × PGD-50 × n=300 = 20 combos,
      ~13-17 min each on a T4 (v12-measured) ≈ 4.5-5.5 GPU-hours. Use a
      Colab Pro/Pro+ runtime or the Kaggle T4x2 (shard_2gpu.py pattern)
      if the session limit is a concern; do not interrupt the eval cell
      once started — eval_provenance.json is only written at the end.

IMPORTANT — BRANCH: this notebook checks out feature/rhan-next from origin.
The branch must be pushed to GitHub before running (git push origin
feature/rhan-next). It intentionally does NOT reset to origin/main — RHANNext
only exists on the branch, and it must not be merged to main until Stage 3's
validated checkbox is checked.

Usage: paste cells into a Colab GPU runtime, set HF_TOKEN in Secrets.
Toggles: DO_STEP_A / DO_STEP_B / DO_STEP_C, SKIP_TRAINING (eval-only),
SMOKE_EPOCHS (10-15), FORCE_STEP_B_OVERRIDE (debug escape — do not use for
publishable numbers).
"""

# %% [markdown]
# # RHAN-Next Stage 1: AIS (enable_ais=True) — Smoke → Full → 5-seed Validation

# %% [markdown]
# ## Step 1: Install Dependencies

# %%
import os, sys, subprocess, time, json

def run(cmd, check=True):
    print(f"\n[RUN]: {cmd}", flush=True)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
    for line in process.stdout:
        print(line, end='', flush=True)
    rc = process.wait()
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return rc

run("pip install --quiet --upgrade pip setuptools wheel")
run("pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121")
run("pip install --quiet huggingface_hub datasets Pillow scipy python-dotenv")

# %% [markdown]
# ## Step 2: Clone and Checkout feature/rhan-next (NOT main!)

# %%
REPO_NAME = 'Adversarial-Cognitive-Model'
WORK_DIR = f'/content/{REPO_NAME}'

if not os.path.exists(WORK_DIR):
    run(f'git clone https://github.com/FerrariKazu/{REPO_NAME}.git')
os.chdir(WORK_DIR)
sys.path.insert(0, WORK_DIR)
sys.path.insert(0, os.path.join(WORK_DIR, 'phase1_training'))

# RHANNext lives on feature/rhan-next. Never reset to origin/main here.
run('git fetch origin')
_branch_ok = subprocess.run(
    'git ls-remote --heads origin feature/rhan-next',
    shell=True, capture_output=True, text=True).stdout.strip()
if not _branch_ok:
    raise RuntimeError(
        "feature/rhan-next is NOT on origin. Push it first:\n"
        "  git push origin feature/rhan-next\n"
        "(RHANNext must not be merged to main until Stage 3 validates.)")
run('git checkout -B feature/rhan-next origin/feature/rhan-next')
run('git reset --hard origin/feature/rhan-next')
print(f"✓ checked out feature/rhan-next @ {subprocess.run('git rev-parse --short HEAD', shell=True, capture_output=True, text=True).stdout.strip()}", flush=True)

# %% [markdown]
# ## Step 3: Set HF_TOKEN and Environment

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
    raise RuntimeError("HF_TOKEN not found. Set it in Colab Secrets (key icon in sidebar).")

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
print(f"✓ HF_TOKEN set for user: {hf_token[:4]}...{hf_token[-4:]}")
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## Step 4: Toggles + Base Checkpoint

# %%
DO_STEP_A   = True    # smoke: 12-15 epochs, ε=0.031 only
DO_STEP_B   = True    # full 60-epoch 3-phase run (auto-gated on Step A health)
DO_STEP_C   = True    # 5-seed matched eval + roadmap verdict recorder
SKIP_TRAINING = False # eval-only mode (requires rhan_next_ais_best.pth on HF/local)
SMOKE_EPOCHS = 15     # 10-15 per protocol
# Step C runtime: 20 combos × ~13-17 min ≈ 4.5-5.5 GPU-hours on a T4 —
# budget the session accordingly (see docstring).
FORCE_STEP_B_OVERRIDE = False  # debug escape — do NOT use for publishable numbers

BASE = "checkpoints/rhan_stl10_large_pseudolabel_best.pth"
os.makedirs("report", exist_ok=True)   # --diag-json / health verdicts live here
if not os.path.exists(BASE):
    print(f"Downloading base checkpoint {BASE} from HF...", flush=True)
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id="FerrariKazu/rhan-checkpoints", repo_type="dataset",
                    filename="rhan_stl10_large_pseudolabel_best.pth",
                    local_dir="checkpoints", token=hf_token)
print(f"✓ base checkpoint present: {BASE} ({os.path.getsize(BASE)/1e6:.0f} MB)", flush=True)

# %% [markdown]
# ## Step 5 — STEP A: SMOKE TEST (10-15 epochs, ε=0.031 single phase)

# %%
print("\n" + "="*70)
print("  STEP A: RHANNext(enable_ais=True) SMOKE — %d epochs, ε=0.031 only" % SMOKE_EPOCHS)
print("="*70)

if DO_STEP_A and not SKIP_TRAINING:
    run(
        f"python3 phase1_training/train_rhan_next.py "
        f"--enable-ais "
        f"--ckpt-name rhan_next_ais_smoke "
        f"--max-epochs {SMOKE_EPOCHS} "
        f"--target-ckpt {BASE} "
        f"--batch-size 16 --accum-steps 16 "
        f"--diag-json report/rhan_next_ais_smoke_diag.jsonl "
        f"--force-single-gpu"
    )
else:
    print("  (Step A skipped: DO_STEP_A=False or SKIP_TRAINING=True)", flush=True)

# %% [markdown]
# ## Step 5b — HEALTH GATE (decides Step A → Step B)

# %%
SMOKE_DIAG = "report/rhan_next_ais_smoke_diag.jsonl"

def load_diag(path):
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows

def health_verdict(rows):
    """Evaluate the last epoch against the pre-registered Step A criteria."""
    if not rows:
        return {"healthy": False,
                "reasons": ["no --diag-json rows found (smoke crashed before "
                            "completing an epoch)"]}
    last = rows[-1]
    reasons, healthy = [], True

    gs = last.get('gaze_shift_total_mean')
    if gs is None or gs < 0.01:
        healthy = False
        reasons.append(f"gaze shift total mean = {gs} (degenerate: fovea is not "
                       f"moving; expect > 0.01)")
    else:
        reasons.append(f"gaze shift total mean = {gs:.4f} (nonzero — Eq. II v12 "
                       f"gradient is moving the fovea)")

    ess = last.get('steps_effective_std')
    fh = last.get('frac_halted_any')
    if ess is None or ess < 0.02 or not fh:
        healthy = False
        reasons.append(f"halting does NOT vary per-sample: effective-steps "
                       f"std={ess}, frac_halted_any={fh} (must be std>0.02 and "
                       f"frac>0 — v10/v11 failure mode is a flat 4.00)")
    else:
        reasons.append(f"halting VARIES per sample: effective-steps "
                       f"std={ess}, frac_halted_any={fh}")

    pd = last.get('pi_d_per_class', {})
    top2 = sorted(pd.items(), key=lambda kv: -kv[1])[:2]
    top2_classes = {k for k, _ in top2}
    if not {'car', 'truck'} <= top2_classes:
        healthy = False
        reasons.append(f"Π_D per-class ordering BROKEN: top-2 = {top2} "
                       f"(car/truck must be highest — reproduced across every "
                       f"RHAN version; stop and debug, do not push through)")
    else:
        reasons.append(f"Π_D ordering reproduced: top-2 = {top2} (car/truck "
                       f"highest — matches every prior RHAN version)")

    return {"healthy": healthy, "reasons": reasons,
            "last_epoch": last.get('epoch'), "summary": last}

rows = load_diag(SMOKE_DIAG)
print(f"\n--- Per-epoch smoke telemetry (report/rhan_next_ais_smoke_diag.jsonl) ---")
for r in rows:
    print(f"  epoch {r['epoch']:>3} | ε={r['eps']:.3f} | gaze_path={r.get('gaze_shift_total_mean')} "
          f"| eff_steps mean={r.get('steps_effective_mean')} std={r.get('steps_effective_std')} "
          f"| frac_halted={r.get('frac_halted_any')}", flush=True)

verdict = health_verdict(rows)
print("\n" + "="*70)
print("  STEP A HEALTH VERDICT:", "HEALTHY — proceed to Step B" if verdict["healthy"]
      else "DEGENERATE — STOP and debug before spending the full run")
print("="*70)
for reason in verdict["reasons"]:
    print(f"    • {reason}", flush=True)
print("="*70)

with open("report/rhan_next_ais_smoke_health.json", "w") as f:
    json.dump(verdict, f, indent=2, sort_keys=True)
print("  Health verdict written to report/rhan_next_ais_smoke_health.json", flush=True)

PROCEED_STEP_B = verdict["healthy"] or FORCE_STEP_B_OVERRIDE
if not PROCEED_STEP_B:
    print("\n  [STOP] Step B will NOT run. Debug the degenerate signal first "
          "(or set FORCE_STEP_B_OVERRIDE=True to override — not for "
          "publishable numbers).", flush=True)

# %% [markdown]
# ## Step 6 — STEP B: FULL 60-EPOCH, 3-PHASE RUN (null_ablation-comparable)

# %%
print("\n" + "="*70)
print("  STEP B: RHANNext(enable_ais=True) FULL — 60 epochs, 0.031→0.062→0.094")
print("="*70)

if DO_STEP_B and not SKIP_TRAINING and PROCEED_STEP_B:
    # Curriculum (1-20 @0.031, 21-40 @0.062, 41-60 @0.094) is identical to
    # train_rhan_v11.py's — the exact boundaries of null_ablation_v11
    # (31.56±2.88 @ ε=0.094). NEVER --force-restart: the trainer's mandatory
    # HF resume gate restores/aborts instead of silently restarting.
    run(
        f"python3 phase1_training/train_rhan_next.py "
        f"--enable-ais "
        f"--ckpt-name rhan_next_ais "
        f"--max-epochs 60 "
        f"--target-ckpt {BASE} "
        f"--batch-size 16 --accum-steps 16 "
        f"--diag-json report/rhan_next_ais_diag.jsonl "
        f"--force-single-gpu"
    )
else:
    print("  (Step B skipped: DO_STEP_B=False, SKIP_TRAINING=True, or health gate blocked)",
          flush=True)

# %% [markdown]
# ## Step 7 — STEP C: 5-SEED MATCHED EVAL (hardened eval_rhan.py)

# %%
def ensure_ckpt(name):
    """Resolve a checkpoint locally, else from HF."""
    p = f"checkpoints/{name}"
    if os.path.exists(p):
        print(f"  Checkpoint present locally: {p}", flush=True)
        return p
    from huggingface_hub import hf_hub_download
    for repo in ("FerrariKazu/rhan-checkpoints", "FerrariKazu/rhan-checkpoints-rolling"):
        try:
            p = hf_hub_download(repo_id=repo, repo_type="dataset",
                                filename=name, local_dir="checkpoints",
                                token=hf_token)
            print(f"  Downloaded {name} from {repo}", flush=True)
            return p
        except Exception:
            continue
    raise RuntimeError(f"No checkpoint found for {name} (local or HF). Train Step B first.")

print("\n" + "="*70)
print("  STEP C: 5-SEED MATCHED EVAL — rhan_next_ais vs TRADES Large baseline")
print("="*70)

if DO_STEP_C:
    # Self-test the eval entrypoint first (structural, against checked-in ref).
    run("python3 phase2_attacks/eval_rhan.py --self-test")

    rhan_ckpt = ensure_ckpt("rhan_next_ais_best.pth")
    bsl_ckpt  = ensure_ckpt("rhan_stl10_large_pseudolabel_best.pth")

    run(
        f"python3 phase2_attacks/eval_rhan.py "
        f"--ckpt-specs rhan_next_ais:{rhan_ckpt}:next "
        f"trades_large_baseline:{bsl_ckpt}:large "
        f"--seeds 41 42 43 44 45 "
        f"--eps-list 0.000 0.094 "
        f"--n-samples 300 "
        f"--batch-size 64 "
        f"--output-dir report/sweep_stage1_ais"
    )
else:
    print("  (Step C skipped: DO_STEP_C=False)", flush=True)

# %% [markdown]
# ## Step 7b — RECORD VERDICT IN docs/rhan_next_roadmap.json

# %%
print("\n" + "="*70)
print("  RECORDING STAGE 1 VERDICT INTO docs/rhan_next_roadmap.json")
print("="*70)

def record_verdict():
    prov_path = "report/sweep_stage1_ais/eval_provenance.json"
    if not os.path.exists(prov_path):
        print("  No eval_provenance.json found — Step C did not complete.", flush=True)
        return
    with open(prov_path) as f:
        prov = json.load(f)

    # Pull the Stage 1 numbers from the provenance (results + verdicts).
    stage1 = {
        "validated": True,
        "validated_date": prov.get("timestamp_utc", "unknown")[:10],
        "git_sha": prov.get("git_sha"),
        "checkpoints": prov.get("checkpoints"),
        "seeds": prov.get("seeds"),
        "eps_list": prov.get("eps_list"),
        "n_samples": prov.get("n_samples"),
        "pgd_steps": prov.get("pgd_steps"),
        "results": prov.get("results"),
        "crossover_verdicts": prov.get("crossover_verdicts"),
        "note": "Stage 1 (AIS) validated via the 5-seed matched protocol. "
                "A null result is still a valid, reportable outcome.",
    }

    roadmap = json.load(open("docs/rhan_next_roadmap.json"))
    roadmap["stages"]["1"]["validated"] = True
    roadmap["stages"]["1"]["validated_date"] = stage1["validated_date"]
    roadmap["stages"]["1"]["validated_note"] = (
        "Stage 1 (AIS) 5-seed matched eval recorded from "
        "report/sweep_stage1_ais/eval_provenance.json — see "
        "roadmap.stages['1'].stage1_verdict. Verdict is what it is, "
        "including a null result.")
    roadmap["stages"]["1"]["stage1_verdict"] = stage1
    with open("docs/rhan_next_roadmap.json", "w") as f:
        json.dump(roadmap, f, indent=2, sort_keys=False)
    print("  ✓ docs/rhan_next_roadmap.json updated with the Stage 1 verdict.", flush=True)
    print("  Verdict summary:", json.dumps(stage1.get("crossover_verdicts"),
                                           indent=2), flush=True)

record_verdict()

# %% [markdown]
# ## Done — next gate

# %%
print("\n" + "="*70)
print("  STAGE 1 EXECUTION COMPLETE")
print("="*70)
print("  - Step A smoke telemetry : report/rhan_next_ais_smoke_diag.jsonl")
print("  - Step A health verdict  : report/rhan_next_ais_smoke_health.json")
print("  - Step B full run        : checkpoints/rhan_next_ais_{best,rolling}.pth")
print("  - Step C eval            : report/sweep_stage1_ais/")
print("  - Verdict recorded       : docs/rhan_next_roadmap.json (stages.1)")
print()
print("  DO NOT begin Stage 2 (HPC) until the Stage 1 verdict is reviewed.")
print("  Stage 2 must keep AIS fixed at whatever Stage 1 validated.")
print("="*70)
