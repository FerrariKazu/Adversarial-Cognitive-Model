#!/usr/bin/env python3
"""
Colab Notebook Pipeline: RHAN-v12 STEP 0 Diagnostics
=====================================================
Twin of cloud_setup/kaggle_v12_step0.py — same diagnostics, Colab setup
(single T4 runtime; the Kaggle T4x2 notebook shards across 2 GPUs).

Runs the mandatory pre-training diagnostics for the RHAN-v12 full run.
Do NOT commit to a 120-epoch run until both diagnostics return and the
step0 report (report/step0_diagnostics.json) is reviewed.

  STEP 0a — mid-training robustness trajectory (null-ablation v11)
    - Recovers the ~epoch-40 mid-training checkpoint from HF revision
      history (rolling checkpoints are uploaded every epoch; the verified
      revision 82b4f6cc98d3 == epoch 41, best_acc 48.89).
    - 3-seed matched sweep (shard_2gpu.py protocol, eps 0.0 / 0.094,
      n=300/seed, PGD-50) vs the known epoch-60 result 31.56±2.88.
    - Verdict: climbing if ep41 eps=0.094 is clearly BELOW 31.56±2.88;
      flat if within noise; declining if above.
    - v12 recon fix: get_reconstruction_loss() now returns a differentiable
      scalar (v11 detached recon_errors, silently zeroing w_recon*L_recon
      gradients). The step0 runs exercise this fixed path.

  STEP 0b — v12 data-mix preference (10-epoch probe)
    - Mix A: 5K real + Sprint 2 synthetic (115K, TRADES-Large pseudo-labels)
      -> train_rhan_v12.py --synthetic-data ... --no-pseudo
    - Mix B: 5K real + pseudo-labels only (default pseudo-label flow)
    - Both: 10 epochs, same curriculum start (phase 1: eps=0.031, lr=3e-3),
      --max-epochs 10, from the TRADES-Large base.
    - 3-seed eval at eps 0.0 / 0.094 for BOTH mixes -> which data mix does
      v12 prefer before committing to the long run?

Decision rules (from the task):
  - Choose the mix with HIGHER eps=0.094 3-seed mean (tie-break: clean).
  - Epoch count: 60 if Step 0a shows climbing, else revisit.

Step 0a provenance caveat (protocol): the reference 31.56±2.88 must come
from the SAME protocol (batch 64, seeds 41/42/43, n=300, PGD-50, norm-space
eps) as this run — batch layout changes the PGD-init RNG stream. If it was
produced with batch 32, the verdict is not directly comparable; re-measure
the final epoch-60 checkpoint in the same batch before trusting the climb
verdict.

CRITICAL provenance caveat (recon gradient): the ep41 checkpoint was trained
with LEGACY v11 code, NOT v12. model_rhan_v11.py appends recon_mse.detach()
to trajectory['recon_errors'] and get_reconstruction_loss() stacks those
DETACHED tensors — so w_recon*L_recon was a gradient no-op in every v11 run
(including the null-ablation run that produced ep41). The Step 0a climbing
trend therefore describes the "TRADES + prior WITHOUT a functioning recon
loss" config ONLY. It is NOT evidence for how v12 (real recon gradients)
behaves over the same epoch range — v12's trajectory must be judged from
Step 0b's own numbers, not from the Step 0a verdict.

Runtime on a single T4: Step 0a (~40 min, optional — already run locally)
+ Mix A 10-epoch train (~1.5 h) + Mix B 10-epoch train (~1.5 h)
+ 3-seed eval of both mixes (~1.5 h) => ~4-5 h total.
Use Toggles to cut scope: DO_STEP0A=0 (already done locally),
DO_STEP0B_MIX_A=1/0, DO_STEP0B_MIX_B=1/0, BATCH_SIZE (default 64).
"""

# %% [markdown]
# # RHAN-v12 STEP 0 Diagnostics (0a: ep41 trajectory | 0b: data-mix probe)

# %% [markdown]
# ## Step 1: Install Dependencies

# %%
import os, sys, subprocess

def run(cmd, check=True):
    print(f"\n[RUN]: {cmd}")
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
run("pip install --quiet opencv-python datasets huggingface_hub Pillow scipy")

DO_STEP0A    = os.environ.get("DO_STEP0A", "1") == "1"
DO_STEP0B_MA = os.environ.get("DO_STEP0B_MIX_A", "1") == "1"
DO_STEP0B_MB = os.environ.get("DO_STEP0B_MIX_B", "1") == "1"
print(f"Toggles: STEP0A={DO_STEP0A} MIX_A={DO_STEP0B_MA} MIX_B={DO_STEP0B_MB}")

# %% [markdown]
# ## Step 2: Clone and Sync Repository

# %%
REPO_NAME = 'Adversarial-Cognitive-Model'
WORK_DIR = f'/content/{REPO_NAME}'

if not os.path.exists(WORK_DIR):
    run(f'git clone https://github.com/FerrariKazu/{REPO_NAME}.git')
os.chdir(WORK_DIR)
run('git fetch origin main && git reset --hard origin/main')
sys.path.insert(0, WORK_DIR)
sys.path.insert(0, os.path.join(WORK_DIR, 'phase1_training'))

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
# ## Step 4: Resolve the base checkpoint

# %%
from huggingface_hub import hf_hub_download

def resolve(filename, repo):
    """Return a local path to filename, local-first with HF fallback."""
    path = f"checkpoints/{filename}"
    if os.path.exists(path):
        print(f"  Present locally: {path}", flush=True)
        return path
    print(f"  Downloading {filename} from {repo}...", flush=True)
    return hf_hub_download(repo_id=repo, repo_type="dataset",
                           filename=filename, local_dir="checkpoints")

_bsl = resolve("rhan_stl10_large_pseudolabel_best.pth", "FerrariKazu/rhan-checkpoints")
print(f"  baseline -> {_bsl}")

# %% [markdown]
# ## Step 5: STEP 0a — mid-training (epoch 41) 3-seed sweep vs 31.56±2.88

# %%
if DO_STEP0A:
    EP41_REV = "82b4f6cc98d3"   # verified == epoch 41 (best_acc 48.89)
    if not os.path.exists("checkpoints/rhan_stl10_v11_ep41.pth"):
        run("python3 phase2_attacks/find_epoch_revision.py "
            "--repo FerrariKazu/rhan-checkpoints-rolling "
            "--file rhan_stl10_v11_rolling.pth "
            "--target-epoch 40 "
            f"--pin-revision {EP41_REV} "
            "--out checkpoints/rhan_stl10_v11_ep41.pth")

    n_gpus = torch.cuda.device_count()
    BATCH = int(os.environ.get("BATCH_SIZE", "64" if n_gpus >= 1 else "32"))
    print(f"\n=== STEP 0a: 3-seed sweep on epoch-41 checkpoint "
          f"(eps 0.0/0.094, n=300/seed, PGD-50) ===")
    run("python3 phase2_attacks/shard_2gpu.py "
        f"--gpus {max(n_gpus, 1)} "
        "--n-samples 300 --seeds 41 42 43 --pgd-steps 50 "
        f"--batch-size {BATCH} "
        "--output-dir report/sweep_step0a_ep41 "
        "--eps-norm-space --eps-list 0.0 0.094 "
        "--baseline-label trades_large_baseline "
        "--ckpt-specs "
        "null_ablation_ep41:checkpoints/rhan_stl10_v11_ep41.pth:v11")

# %% [markdown]
# ## Step 6: STEP 0b Mix A — v12, 10 epochs, 5K real + 115K synthetic

# %%
def download_synthetic_pt():
    """Prefer the pre-built HF .pt; fall back to assembling from shards."""
    path = "synthetic_stl10_115k_tradeslabels.pt"
    if os.path.exists(path):
        return path
    try:
        return hf_hub_download(repo_id="FerrariKazu/stl10-synthetic",
                               filename=path, repo_type="dataset")
    except Exception:
        print("  Pre-built .pt not on HF — assembling from shards "
              "(~20 min)...", flush=True)
        # ── fallback: reproduce kaggle_train_synthetic.py steps 1-3 ──
        import io, tarfile
        import numpy as np
        from PIL import Image
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
        CLASSES = ['airplane','bird','car','cat','deer',
                   'dog','horse','monkey','ship','truck']
        C2I = {c: i for i, c in enumerate(CLASSES)}
        files = api.list_repo_files("FerrariKazu/stl10-synthetic", repo_type="dataset")
        tars = sorted(f for f in files if 'filtered' in f and f.endswith('.tar'))
        os.makedirs("/content/synth_shards", exist_ok=True)
        for f in tars:
            hf_hub_download(repo_id="FerrariKazu/stl10-synthetic", filename=f,
                            repo_type="dataset", local_dir="/content/synth_shards")
        total = 0
        for f in tars:
            with tarfile.open(f"/content/synth_shards/{f}", 'r') as tar:
                total += sum(1 for m in tar.getmembers() if m.name.endswith('.png'))
        imgs = torch.empty((total, 3, 96, 96), dtype=torch.uint8)
        labels = torch.empty(total, dtype=torch.long)
        idx = 0
        for f in tars:
            cls = C2I[f.split('_')[2]]
            with tarfile.open(f"/content/synth_shards/{f}", 'r') as tar:
                for m in tar.getmembers():
                    if not m.name.endswith('.png'):
                        continue
                    img = Image.open(io.BytesIO(tar.extractfile(m).read()))
                    img = img.convert('RGB').resize((96, 96))
                    imgs[idx] = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1)
                    labels[idx] = cls
                    idx += 1
        # pseudo-label with TRADES Large
        from model_rhan_stl10_large import RHANLargeSTL10
        from checkpoint_utils import compat_load
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        m = RHANLargeSTL10().to(dev).eval()
        sd = compat_load("checkpoints/rhan_stl10_large_pseudolabel_best.pth", map_location=dev)
        m.load_state_dict(sd.get('model', sd), strict=False)
        mean = torch.tensor([0.4467,0.4398,0.4066], device=dev).view(1,3,1,1)
        std  = torch.tensor([0.2603,0.2566,0.2713], device=dev).view(1,3,1,1)
        with torch.no_grad():
            for i in range(0, total, 64):
                b = (imgs[i:i+64].float().to(dev)/255.0 - mean) / std
                labels[i:i+64] = m(b).argmax(1).cpu()
        torch.save({'imgs': imgs, 'labels': labels}, path, _use_new_zipfile_serialization=False)
        return path

if DO_STEP0B_MA:
    synth_pt = download_synthetic_pt()
    print(f"\n=== STEP 0b Mix A: v12 x10 epochs, 5K real + 115K synthetic "
          f"(no pseudo) ===")
    run("python3 phase1_training/train_rhan_v12.py "
        f"--synthetic-data {synth_pt} --no-pseudo "
        "--batch-size 16 --accum-steps 16 "
        "--max-epochs 10 --seed 42 --ckpt-name rhan_v12_mixA --force-single-gpu")

# %% [markdown]
# ## Step 7: STEP 0b Mix B — v12, 10 epochs, 5K real + pseudo-labels only

# %%
if DO_STEP0B_MB:
    print(f"\n=== STEP 0b Mix B: v12 x10 epochs, 5K real + pseudo-labels ===")
    run("python3 phase1_training/train_rhan_v12.py "
        "--batch-size 16 --accum-steps 16 "
        "--max-epochs 10 --seed 42 --ckpt-name rhan_v12_mixB --force-single-gpu")

# %% [markdown]
# ## Step 8: 3-seed eval of both mixes at eps 0.0 / 0.094 (arch=v12)

# %%
if DO_STEP0B_MA or DO_STEP0B_MB:
    n_gpus = torch.cuda.device_count()
    BATCH = int(os.environ.get("BATCH_SIZE", "64" if n_gpus >= 1 else "32"))
    specs = []
    if DO_STEP0B_MA:
        specs.append("rhan_v12_mixA:checkpoints/rhan_v12_mixA_best.pth:v12")
    if DO_STEP0B_MB:
        specs.append("rhan_v12_mixB:checkpoints/rhan_v12_mixB_best.pth:v12")
    specs.append(f"trades_large_baseline:{_bsl}:large")
    print(f"\n=== STEP 0b eval: 3-seed protocol on {[s.split(':')[0] for s in specs]} ===")
    run("python3 phase2_attacks/shard_2gpu.py "
        f"--gpus {max(n_gpus, 1)} "
        "--n-samples 300 --seeds 41 42 43 --pgd-steps 50 "
        f"--batch-size {BATCH} "
        "--output-dir report/sweep_step0b_mixes "
        "--eps-norm-space --eps-list 0.0 0.094 "
        "--baseline-label trades_large_baseline "
        "--ckpt-specs " + " ".join(specs))

# %% [markdown]
# ## Step 9: Step-0 decision report

# %%
import csv, json

report = {"schema": "step0_diagnostics_v12", "step0a": {}, "step0b": {}}

def read_agg(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {(r['ckpt_label'], float(r['eps_pixel'])): r
                for r in csv.DictReader(f)}

if DO_STEP0A:
    a = read_agg("report/sweep_step0a_ep41/epsilon_sweep_results.csv")
    for (lab, eps), r in a.items():
        report["step0a"][f"{lab}_eps{eps:.3f}"] = {
            "acc_mean": float(r['acc_mean']), "acc_std": float(r['acc_std']),
            "dprime_mean": float(r['macro_dprime_mean'])}
    report["step0a"]["known_epoch60_eps0.094"] = (
        "31.56 ± 2.88 (from the SAME legacy v11 null-ablation run — trained "
        "with no functioning recon gradient, like ep41)")
    report["step0a"]["provenance"] = (
        "ep41 checkpoint = rhan_stl10_v11_rolling.pth @ HF rev 82b4f6cc98d3 "
        "(epoch 41, best_acc 48.89), from the ORIGINAL null-ablation v11 run "
        "trained with LEGACY train_rhan_v11.py / model_rhan_v11.py. v11's "
        "get_reconstruction_loss() stacks trajectory['recon_errors'], which "
        "were appended as recon_mse.detach() (model_rhan_v11.py:502) — so "
        "w_recon*L_recon was a GRADIENT NO-OP in every v11 run. This "
        "checkpoint was therefore trained with NO functioning reconstruction "
        "loss; it is NOT a v12 artifact.")
    if "null_ablation_ep41_eps0.094" in report["step0a"]:
        m = report["step0a"]["null_ablation_ep41_eps0.094"]["acc_mean"]
        report["step0a"]["verdict"] = (
            "climbing" if m < 31.56 else
            ("flat" if abs(m - 31.56) < 5 else "declining"))
        report["step0a"]["verdict_scope"] = (
            "VALID ONLY for the no-recon-gradient config (TRADES + prior "
            "architecture without a functioning recon loss). NOT evidence for "
            "how v12 (real recon gradients) behaves over the same epoch range; "
            "judge v12 from Step 0b numbers instead.")

if DO_STEP0B_MA or DO_STEP0B_MB:
    a = read_agg("report/sweep_step0b_mixes/epsilon_sweep_results.csv")
    for (lab, eps), r in a.items():
        report["step0b"][f"{lab}_eps{eps:.3f}"] = {
            "acc_mean": float(r['acc_mean']), "acc_std": float(r['acc_std'])}
    keys = [k for k in report["step0b"] if "eps0.094" in k and "baseline" not in k]
    if keys:
        best = max(keys, key=lambda k: report["step0b"][k]["acc_mean"])
        report["step0b"]["preferred_mix"] = best
        report["step0b"]["rule"] = "prefer higher eps=0.094 3-seed mean"

os.makedirs("report", exist_ok=True)
with open("report/step0_diagnostics.json", "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))

print("\n" + "=" * 72)
print("  STEP 0 COMPLETE — review report/step0_diagnostics.json")
print("  Do NOT launch the 120-epoch run until the data-mix + epoch")
print("  decision is made from these numbers.")
print("=" * 72)
