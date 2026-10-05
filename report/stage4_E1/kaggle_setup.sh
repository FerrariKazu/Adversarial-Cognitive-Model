#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Stage 4-E1: Kaggle Setup
#
# Run this at the start of a Kaggle session to:
# 1. Verify HF_TOKEN from Kaggle secrets
# 2. Install dependencies
# 3. Clone/update repo
# 4. Pull existing Stage 4 artifacts from HF
# 5. Resume from exact unfinished state
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

echo "═══════════════════════════════════════════════════════════════"
echo "  Stage 4-E1: Kaggle Setup"
echo "═══════════════════════════════════════════════════════════════"

# ── 1. HF Token ────────────────────────────────────────────────────────
echo ""
echo "─── 1. HuggingFace Authentication ───"
if [ -z "${HF_TOKEN:-}" ]; then
    echo "  Attempting to read HF_TOKEN from Kaggle secrets..."
    python3 -c "
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret('HF_TOKEN')
    import os
    os.environ['HF_TOKEN'] = token
    print(f'  ✓ HF_TOKEN loaded from Kaggle secrets')
except Exception as e:
    print(f'  ✗ HF_TOKEN not found: {e}')
    print(f'    Add it via: Kaggle → Account → Secrets → Add HF_TOKEN')
    exit(1)
" || {
    echo "  FATAL: HF_TOKEN not available. Cannot persist artifacts."
    exit 1
}
fi

# ── 2. Install dependencies ───────────────────────────────────────────
echo ""
echo "─── 2. Dependencies ───"
pip install -q datasets==4.7.0 huggingface_hub torch torchvision scipy

# ── 3. Clone/update repo ──────────────────────────────────────────────
echo ""
echo "─── 3. Repository ───"
REPO_DIR="/kaggle/working/Adversarial_Cognitive_Model"
if [ -d "$REPO_DIR/.git" ]; then
    cd "$REPO_DIR"
    git pull --ff-only 2>/dev/null || echo "  ⚠ git pull failed — using existing checkout"
else
    echo "  Cloning repository..."
    git clone https://github.com/FerrariKazu/Adversarial-Cognitive-Model.git "$REPO_DIR"
    cd "$REPO_DIR"
fi
echo "  ✓ Repository ready: $REPO_DIR"

# ── 4. Pull Stage 4 artifacts from HF ─────────────────────────────────
echo ""
echo "─── 4. Pulling Stage 4 artifacts from HF ───"
python3 - "$REPO_DIR" <<'PYEOF'
import sys, os
from huggingface_hub import HfApi, hf_hub_download

repo = sys.argv[1]
token = os.environ.get("HF_TOKEN")
if not token:
    print("  ⚠ No HF_TOKEN — skipping artifact download")
    sys.exit(0)

api = HfApi(token=token)
artifacts_dir = os.path.join(repo, "stage4_artifacts", "E1")
os.makedirs(artifacts_dir, exist_ok=True)

for hf_repo in ["FerrariKazu/rhan-checkpoints-artifacts", "FerrariKazu/rhan-checkpoints"]:
    try:
        files = api.list_repo_files(repo_id=hf_repo, repo_type="dataset", token=token)
        stage4_files = [f for f in files if f.startswith("stage4_artifacts/E1/")]
        if stage4_files:
            print(f"  Found {len(stage4_files)} Stage 4 artifacts in {hf_repo}")
            for f in stage4_files:
                local_path = os.path.join(repo, f)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                if not os.path.exists(local_path):
                    try:
                        hf_hub_download(repo_id=hf_repo, repo_type="dataset",
                                       filename=f, token=token,
                                       local_dir=repo)
                        print(f"    ✓ {os.path.basename(f)}")
                    except Exception as e:
                        print(f"    ⚠ {os.path.basename(f)}: {e}")
    except Exception as e:
        print(f"  Could not list {hf_repo}: {e}")

# Download checkpoints if not local
ckpt_dir = os.path.join(repo, "checkpoints")
for name in ["rhan_next_ais_hpc_best.pth", "rhan_next_ais_hpc_recon_best.pth"]:
    local = os.path.join(ckpt_dir, name)
    if not os.path.exists(local):
        try:
            hf_hub_download(repo_id="FerrariKazu/rhan-checkpoints",
                           repo_type="dataset", filename=name,
                           token=token, local_dir=repo)
            print(f"  ✓ Downloaded {name}")
        except Exception:
            pass
PYEOF

# ── 5. Resume ──────────────────────────────────────────────────────────
echo ""
echo "─── 5. Resuming experiment ───"
bash report/stage4_E1/resume.sh
