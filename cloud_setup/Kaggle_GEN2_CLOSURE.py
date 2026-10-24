#!/usr/bin/env python3
"""
Kaggle Notebook -- Gen-2 Closure  (Phase 0 housekeeping + Phase 1 mechanism tests)
================================================================================
One file, one session, everything the closure plan asks for.  ~2 GPU-h for
Phase 0 + 1.1 + 1.2; ~6.5 GPU-h more only if the pre-registered gate sends the
run to 1.3 (C4).  Every stage is cached on HuggingFace by (arm, checkpoint
SHA-256, protocol hash) so a session that dies during C4 resumes instead of
re-evaluating.

WHAT THIS RUN DOES
------------------
Phase 0 (housekeeping)
  0.1  ff_56 identity.  Resolve the *true* 56-px direct-Rung-3 checkpoint,
       upload it as `ff_56_direct_rung3.pth`, preserve the stale bytes as
       `ff_56_defective.pth`, and remove the ambiguous legacy name only after
       the replacement bytes are verified present.  Record every SHA-256.
  0.2  Recompute the CKA row *and* column for ff_56 (block-12 CLS, linear CKA,
       same 1,000 frames) and diff it against the stored historical matrix.
  0.3  Finish C3's canonical evaluation, and test C3 vs C1 on 5,000 clean
       with McNemar.  |delta| <= 1 pp and p >= 0.05 => null.
  0.4  Tag decision + export docs/ and report/ to an HF dataset with per-file
       hashes (they are gitignored, so they are not in the clone).

Phase 1
  1.1  T-sweep: recurrence_only at T = 1, 2, 3, 4 -> 5k clean, 1k-slice robust,
       per-sample predictions saved.  Pre-registered gate decides 1.2 vs 1.3.
  1.2  Oracle-halting curve (CPU) from the saved per-T outcomes.  If headroom
       >= 1 pp and the gate says branch A, train the "extra views fixed this"
       probe and compare it against softmax entropy (bar: AUROC 0.837).
  1.3  C4 (4-view pipeline, refinement block off) -- trained only if the gate
       fires.  McNemar vs recurrence_only decides "multi-view" vs "interaction".

PROTOCOL (identical for every arm, fixed before any model is loaded)
  * evaluation/harness.py with the slice-hash gate and the 5k parity gate ON;
  * clean accuracy on all 5,000 val images; robust on the canonical 1,000-slice;
  * one attack set: PGD-50, APGD-CE(100), APGD-DLR(100); worst-case is their
    bitwise intersection;
  * eps = 5.417/255 in [0, 1] pixel space, perturbation budget asserted per
    batch;
  * comparisons by McNemar (exact two-sided binomial primary, Edwards chi2
    reported), never by eyeballing two numbers;
  * a p-value near 0.1 means "not established";
  * "decisive" is only claimed when the parity gates pass.

USAGE ON KAGGLE
---------------
1. Accelerator: GPU T4 or P100 (1x).  Internet: ON.
2. Add-ons > Secrets > 'HF_TOKEN' (write token).
3. Optional but needed for Phase 0.1/0.4 to complete in this environment:
     - attach the true checkpoint folder as a Kaggle input dataset, or set
       FF56_TRUE_PATH=/kaggle/input/<ds>/foundation_ff_56_final.pth
     - attach docs/ + report/ as an input dataset, or set
       RHAN_DOCS_DIR=/kaggle/input/<ds>/docs
   Neither is required for Phase 1: if the true ff_56 is already on HF under a
   fingerprint-matching name it is resolved there.
4. Paste and run.  Everything lands in the HF dataset repo under closure/.

ENV KNOBS (all optional)
  CLOSURE_C4          auto | on | off   (default auto: follows the 1.1 gate)
  CLOSURE_CACHE       1 | 0             (default 1)
  CLOSURE_CONTEXT_ARMS 1 | 0            (default 1: ff_112, g2_gist_only, gen1_core for CKA/5k)
  CLOSURE_SESSION_BUDGET_H  wall-clock hours this session may plan for (default
                         10.5). C4 training starts only if the remaining epochs
                         (from the rolling checkpoint) are estimated to fit.
  CLOSURE_RENAME_LEGACY 1 | 0           (default 1: preserve-then-remove the
                         ambiguous ff_56 legacy name once bytes are verified)
"""

import os
import sys
import json
import time
import math
import shutil
import tempfile
import hashlib
import datetime
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

SESSION_T0 = time.time()


def elapsed_h() -> float:
    return (time.time() - SESSION_T0) / 3600.0


def banner(title: str) -> None:
    print("\n" + "=" * 88)
    print(f"  {title}")
    print("=" * 88, flush=True)


# ============================================================================
# A.  PRE-REGISTRATION -- written down before anything is loaded or run.
# ============================================================================
# These constants and rules are hashed and stored with every result.  Changing
# them invalidates the HF cache, so a result can never be silently produced
# under a different rule than the one written here.

PROTOCOL = {
    "harness": "evaluation/harness.py",
    "clean_set": {"n": 5000, "note": "full val split, path hash asserted"},
    "robust_set": {"n": 1000, "seed": 42,
                   "slice_sha256": "d129534adf0d1f5816a13ebebf20e49e502f935c0f8bd878ce162acefddd6b25"},
    "full_val_path_sha256": "6a78a2fde02bfc1ff93b063eeddb32fb32759d9da4994421adda7353636a3be7",
    "eps_pixel_255": 5.417,
    "attacks": {"pgd50": {"steps": 50, "restarts": 1, "loss": "ce"},
                "apgd_ce": {"steps": 100, "restarts": 1, "loss": "ce"},
                "apgd_dlr": {"steps": 100, "restarts": 1, "loss": "dlr"}},
    "worst_case": "bitwise AND of pgd50, apgd_ce, apgd_dlr",
    "checkpoint_gate_pp": 1.0,
    "mcnemar": "exact two-sided binomial (primary); Edwards continuity chi2 (reported)",
    "parity_gates": [
        "canonical slice SHA-256 asserted (get_canonical_slice assert_hash=True)",
        "full val path SHA-256 asserted",
        "5k clean vs checkpoint training monitor within 1.0 pp (verify_checkpoint_gate)",
        "T-invariance: identical state_dict for T=1..4 (no per-step parameters)",
        "forward parity: module forward == harvested trajectory readout (<1e-5)",
        "two-pipeline parity: 1k clean from the 5k scan == 1k clean from the raw-pixel scan (<=0.4 pp)",
        "L_inf budget asserted on every adversarial batch",
    ],
}

THRESHOLDS = {
    "pp_band": 1.0,          # "within about 1 pp"
    "p_established": 0.01,   # below this and gates pass -> decisive
    "p_ns": 0.05,            # at/above this -> not a significant difference
    "p_not_established_band": 0.20,  # 0.05..0.20 -> "not established"
    "oracle_headroom_pp": 1.0,
    "c3_depth_recovery_frac": 0.70,
    "probe_auc_margin": 0.02,
    "entropy_auc_bar": 0.837,   # own prior measurement: U_t 0.662 vs entropy 0.837
    "cka_delta_flag": 0.05,
}

RULES = {
    "P0.1_ff56_identity": (
        "The true ff_56 is the unique candidate with phase == 'ff_56', epoch == 15, "
        "and val_clean_acc within 0.002 of 0.5904 (the Rung-3 15-epoch monitor in "
        "report/controls/gen2_epoch_log.jsonl). If zero or >1 candidates match, the "
        "identity is UNRESOLVED: nothing is renamed, and no Phase 1 result may be "
        "called decisive."
    ),
    "P0.1_rename": (
        "Preserve-then-remove only: the stale bytes are uploaded as ff_56_defective.pth "
        "and the true bytes as ff_56_direct_rung3.pth, both verified by LFS sha256 in "
        "the same repo, before the legacy name is deleted. If verification fails, the "
        "legacy name is kept and the failure is reported."
    ),
    "P0.2_cka": (
        "Recompute the ff_56 row and column on the identical 1,000 frames "
        "(block-12 CLS, linear CKA, float64). Any |delta| > 0.05 against the stored "
        "matrix means the historical row was produced by a different checkpoint."
    ),
    "P0.3_c3_vs_c1": (
        "C3 vs C1 on 5k clean is a NULL iff |delta| <= 1.0 pp AND McNemar p >= 0.05. "
        "Separately, 'depth explains the clean gain' requires C3 to recover >= 70% of "
        "the recurrence_only-minus-C1 clean gap."
    ),
    "P1.1_gate": (
        "A: |clean5k(T=1) - clean5k(ff_56)| <= 1.0 pp AND McNemar(T=1 vs ff_56) p >= 0.05 "
        "  -> the gain is inference-time multi-view -> run 1.2. "
        "B: clean5k(T=1) exceeds ff_56 by > 1.0 pp -> the gain is in the weights -> run 1.3. "
        "C: clean5k(T=1) is below ff_56 by > 1.0 pp -> multi-view is doing the work; "
        "  run 1.2 (oracle) and 1.3 (C4) because the mechanism question is now multi-view."
    ),
    "P1.2_oracle": (
        "Oracle headroom = oracle_acc(max over T in 1..4) - acc(T=4). Headroom < 1.0 pp "
        "means adaptive compute has nothing to gain at 56 px: stop, do not train a halting head. "
        "Otherwise the probe must beat softmax entropy by >= 0.02 AUROC with a paired "
        "bootstrap CI excluding 0; entropy at 0.837 is the bar to beat."
    ),
    "P1.3_c4": (
        "C4 matches recurrence_only iff McNemar p >= 0.05 on BOTH 5k clean and 1k "
        "worst-case. Match -> the mechanism is multi-view, not extra depth. No match -> "
        "the effect is an interaction; report it as unattributed and spend no more."
    ),
    "global": (
        "A result is 'decisive' only when every parity gate passes. A p-value in "
        "[0.05, 0.20] is reported as 'not established', not as a difference."
    ),
}

PREREG = {
    "id": "gen2-closure/1",
    "created_utc": None,  # filled at runtime, excluded from the hash
    "protocol": PROTOCOL,
    "thresholds": THRESHOLDS,
    "rules": RULES,
}


def _prereg_hash() -> str:
    blob = {k: v for k, v in PREREG.items() if k != "created_utc"}
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode("utf-8")).hexdigest()


PREREG["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
PREREG_HASH = _prereg_hash()


def print_preregistration() -> None:
    banner("A. PRE-REGISTRATION (fixed before any model is loaded)")
    print(f"  protocol id : {PREREG['id']}")
    print(f"  rules hash  : {PREREG_HASH[:32]}...")
    print(f"  eps         : {PROTOCOL['eps_pixel_255']}/255 = {PROTOCOL['eps_pixel_255']/255:.8f} (pixel space)")
    print(f"  clean set   : {PROTOCOL['clean_set']['n']} images (full val)")
    print(f"  robust set  : {PROTOCOL['robust_set']['n']}-image slice, seed {PROTOCOL['robust_set']['seed']}")
    print("  attacks     : PGD-50, APGD-CE(100), APGD-DLR(100); worst = intersection")
    print()
    for k, v in RULES.items():
        print(f"  [{k}] {v}")
    print()


print_preregistration()


# ============================================================================
# B.  ENVIRONMENT
# ============================================================================

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
        "HF_TOKEN not found. Kaggle: Add-ons > Secrets > add label 'HF_TOKEN' "
        "with a HuggingFace WRITE token."
    )
print(f"HF_TOKEN authenticated (starts with: {hf_token[:4]}...)")

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - Kaggle always ships torch
    subprocess.run("pip install -q torch", shell=True, check=True)
    import torch
    import torch.nn as nn

for pkg, mod in [("huggingface_hub", "huggingface_hub"), ("torchattacks", "torchattacks"),
                 ("scipy", "scipy"), ("scikit-learn", "sklearn")]:
    try:
        __import__(mod)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.run(f"pip install -q {pkg}", shell=True, check=True)

import torchattacks
from scipy.stats import chi2, binom
from huggingface_hub import HfApi, hf_hub_download

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB VRAM)")
else:
    print("WARNING: no GPU. This will run, but 1.1 alone will take many hours.")

# ── Repo ─────────────────────────────────────────────────────────────────────
REPO_NAME = "Adversarial-Cognitive-Model"
BRANCH = "feature/gen2-rebuild"
WORK_DIR = f"/kaggle/working/{REPO_NAME}"
TAG_NAME = "gen2-final-pre-closure"

if os.path.exists("/kaggle/working"):
    os.chdir("/kaggle/working")
    if not os.path.exists(WORK_DIR):
        print(f"Cloning FerrariKazu/{REPO_NAME}...")
        subprocess.run(f"git clone https://github.com/FerrariKazu/{REPO_NAME}.git",
                       shell=True, check=True)
    os.chdir(WORK_DIR)
    subprocess.run("git fetch origin", shell=True, check=True)
    subprocess.run(f"git checkout -B {BRANCH} origin/{BRANCH}", shell=True, check=True)
    subprocess.run(f"git reset --hard origin/{BRANCH}", shell=True, check=True)
else:
    print(f"Running locally in {os.getcwd()}")

sys.path.insert(0, os.getcwd())
os.environ["PYTHONPATH"] = f"{os.getcwd()}:{os.environ.get('PYTHONPATH', '')}"

for m in list(sys.modules):
    if any(m.startswith(p) for p in ("training", "gen2_foundation", "noesis_vision", "evaluation")):
        del sys.modules[m]

REPO_SHA = subprocess.check_output("git rev-parse HEAD", shell=True, text=True).strip()
BRANCH_SHA = subprocess.check_output("git rev-parse --short HEAD", shell=True, text=True).strip()
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
print(f"Repo {BRANCH} @ {BRANCH_SHA}")

from gen2_foundation.model import Gen2FoundationModel
from noesis_vision.models.foveation import foveal_sample
from evaluation.harness import (
    get_canonical_slice,
    verify_checkpoint_gate,
    load_model_from_checkpoint,
    build_val_loaders,
    run_pgd_pixel_space,
    NormalizedModel,
    HarnessGateError,
    CANONICAL_VAL_PATH_SHA256,
    DEFAULT_EPS_PIXEL,
)
from training.curriculum_gen2 import IMAGENET_MEAN, IMAGENET_STD

# Normalization helper used by harvest_trajectory.
# NormalizedModel is not available at that call site (base is unwrapped).
_NORM_MEAN = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
_NORM_STD  = torch.tensor(IMAGENET_STD,  dtype=torch.float32).view(1, 3, 1, 1)

def _normalize_pixel(x: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization to a [0, 1] pixel-space tensor."""
    mean = _NORM_MEAN.to(x.device)
    std  = _NORM_STD.to(x.device)
    return (x - mean) / std

# ── Phase 0.4 (tag): local tag always; push only with a credential ───────────
TAG_STATE: Dict[str, Any] = {"name": TAG_NAME, "target": REPO_SHA, "created_local": False,
                             "pushed": False, "note": ""}
subprocess.run(f"git tag -d {TAG_NAME}", shell=True, capture_output=True)
_t = subprocess.run(
    f'git tag -a {TAG_NAME} -m "Gen-2 final pre-closure: protocol, checkpoints and '
    f'docs hashed by Kaggle_GEN2_CLOSURE.py at {STAMP}"',
    shell=True, capture_output=True, text=True)
TAG_STATE["created_local"] = (_t.returncode == 0)
if not TAG_STATE["created_local"]:
    TAG_STATE["note"] = _t.stderr.strip()[:300]

_ght = os.environ.get("GITHUB_TOKEN")
if _ght:
    push_url = f"https://x-access-token:{_ght}@github.com/FerrariKazu/{REPO_NAME}.git"
    _p = subprocess.run(f"git push {push_url} refs/tags/{TAG_NAME}", shell=True, capture_output=True, text=True)
    TAG_STATE["pushed"] = (_p.returncode == 0)
    TAG_STATE["note"] = (_p.stderr.strip()[:300] if _p.returncode else "pushed")
else:
    TAG_STATE["note"] = ("tag created in the session clone only; push requires a "
                         f"credential: git push origin refs/tags/{TAG_NAME}")
print(f"Tag '{TAG_NAME}': local={TAG_STATE['created_local']} pushed={TAG_STATE['pushed']}")

# ── HF repos ─────────────────────────────────────────────────────────────────
HF_BEST = "FerrariKazu/rhan-nxa-g2-checkpoints"
HF_ROLLING = "FerrariKazu/rhan-nxa-g2-checkpoints-rolling"
HF_DOCS = "FerrariKazu/rhan-nxa-docs-report"
api = HfApi(token=hf_token)

for _repo in (HF_BEST, HF_ROLLING, HF_DOCS):
    try:
        api.create_repo(repo_id=_repo, repo_type="dataset", private=True, exist_ok=True)
        print(f"HF repo ready: {_repo}")
    except Exception as e:
        print(f"  Note: could not ensure {_repo}: {e}")

# ============================================================================
# C.  HF PLUMBING -- hashes, never-overwrite uploads, result cache
# ============================================================================
OUT_DIR = Path("report/closure")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

HF_ACTIONS: List[Dict[str, Any]] = []


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def remote_index(repo: str) -> Dict[str, Optional[str]]:
    """filename -> LFS sha256 (None when the file is not LFS)."""
    try:
        info = api.repo_info(repo, repo_type="dataset", files_metadata=True)
        out: Dict[str, Optional[str]] = {}
        for s in info.siblings:
            lfs = getattr(s, "lfs", None) or {}
            out[s.rfilename] = lfs.get("sha256")
        return out
    except Exception as e:
        print(f"  (remote_index {repo}: {e})")
        return {}


_IDX: Dict[str, Dict[str, Optional[str]]] = {}


def idx(repo: str) -> Dict[str, Optional[str]]:
    if repo not in _IDX:
        _IDX[repo] = remote_index(repo)
    return _IDX[repo]


def upload_unique(local_path: str | Path, repo: str, name: str) -> Tuple[str, str]:
    """Upload without ever overwriting a different file.

    Same bytes already present -> no-op.  Different bytes at that name ->
    suffix the name with the content hash instead of clobbering.
    """
    local_path = Path(local_path)
    sha = sha256_file(local_path)
    index = idx(repo)
    name_path = Path(name)
    parent = "" if str(name_path.parent) in (".", "") else f"{name_path.parent.as_posix()}/"
    stem, suffix = name_path.stem, name_path.suffix
    n = 0
    while True:
        candidate = (f"{parent}{stem}{suffix}" if n == 0 else
                     f"{parent}{stem}__{sha[:8]}" + (f"-{n}" if n > 1 else "") + suffix)
        remote = index.get(candidate)
        if remote == sha:
            HF_ACTIONS.append({"action": "skip_same", "repo": repo, "name": candidate, "sha256": sha})
            print(f"  = {repo}/{candidate} already holds these exact bytes; not re-uploading.")
            return candidate, "already_present"
        if remote is None:
            break
        n += 1
    api.upload_file(path_or_fileobj=str(local_path), path_in_repo=candidate,
                    repo_id=repo, repo_type="dataset", token=hf_token)
    index[candidate] = sha
    HF_ACTIONS.append({"action": "upload", "repo": repo, "name": candidate, "sha256": sha,
                       "bytes": local_path.stat().st_size})
    print(f"  + {repo}/{candidate}  ({local_path.stat().st_size/1e6:.1f} MB, sha256 {sha[:12]}...)")
    return candidate, "uploaded"


def download_artifact(repo: str, name: str, dest: Path) -> Optional[Path]:
    try:
        p = hf_hub_download(repo_id=repo, filename=name, repo_type="dataset", token=hf_token)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        return dest
    except Exception:
        return None


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))


USE_CACHE = os.environ.get("CLOSURE_CACHE", "1") == "1"
CACHE_PREFIX = "closure/cache"


def cache_key(label: str, sha: str, size: int, num_glimpses: int) -> str:
    return f"{label}__T{num_glimpses}__{size}px__{sha[:10]}"


def save_result_cache(key: str, record: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> None:
    payload = dict(record)
    payload["prereg_hash"] = PREREG_HASH
    json_dump(OUT_DIR / f"{key}.json", payload)
    np.savez_compressed(OUT_DIR / f"{key}.npz", **arrays)
    upload_unique(OUT_DIR / f"{key}.json", HF_BEST, f"{CACHE_PREFIX}/{key}.json")
    upload_unique(OUT_DIR / f"{key}.npz", HF_BEST, f"{CACHE_PREFIX}/{key}.npz")


def load_result_cache(key: str) -> Optional[Tuple[Dict[str, Any], Dict[str, np.ndarray]]]:
    if not USE_CACHE:
        return None
    j = download_artifact(HF_BEST, f"{CACHE_PREFIX}/{key}.json", OUT_DIR / f"{key}.json")
    if j is None:
        return None
    try:
        rec = json.loads(j.read_text())
    except Exception:
        return None
    if rec.get("prereg_hash") != PREREG_HASH:
        print(f"  (cache {key} was produced under a different protocol hash; ignoring)")
        return None
    n = download_artifact(HF_BEST, f"{CACHE_PREFIX}/{key}.npz", OUT_DIR / f"{key}.npz")
    arrays: Dict[str, np.ndarray] = {}
    if n is not None:
        with np.load(n) as z:
            arrays = {k: z[k] for k in z.files}
    print(f"  (reused cached result {key} from HF -- no GPU spent)")
    return rec, arrays


# ============================================================================
# D.  DATA + canonical slice (gate 1 of 7)
# ============================================================================
DATA_ROOT = os.environ.get(
    "J2_DATA_ROOT",
    "/kaggle/tmp/imagenet100" if os.path.exists("/kaggle/tmp") else
    "/kaggle/working/data/imagenet100" if os.path.exists("/kaggle/working") else
    "data/imagenet100",
)


def _dataset_ready(root: str) -> bool:
    td, vd = os.path.join(root, "train"), os.path.join(root, "val")
    return (os.path.isdir(td) and os.path.isdir(vd)
            and len(os.listdir(td)) == 100 and len(os.listdir(vd)) == 100)


if not _dataset_ready(DATA_ROOT):
    print(f"Bootstrapping ImageNet-100 to {DATA_ROOT} ...")
    os.makedirs(DATA_ROOT, exist_ok=True)
    subprocess.run(f"python3 scripts/prepare_imagenet100.py --root {DATA_ROOT}",
                   shell=True, check=True)

VAL_ROOT = os.path.join(DATA_ROOT, "val")
assert _dataset_ready(DATA_ROOT), f"ImageNet-100 not ready at {DATA_ROOT}"

GATES: List[Dict[str, Any]] = []

# gate #1 + #2: full-val path hash and slice hash, asserted inside the harness
SLICE_INDICES, SLICE_HASH = get_canonical_slice(VAL_ROOT, n=1000, seed=42, assert_hash=True)
GATES.append({"gate": "canonical slice SHA-256", "passed": True, "detail": SLICE_HASH})
GATES.append({"gate": "full val path SHA-256", "passed": True, "detail": CANONICAL_VAL_PATH_SHA256})
print(f"Canonical slice asserted: n={len(SLICE_INDICES)} | sha256 {SLICE_HASH[:16]}...")

EPS = DEFAULT_EPS_PIXEL
assert abs(EPS * 255.0 - PROTOCOL["eps_pixel_255"]) < 1e-9

SLICE_IDX = np.asarray(SLICE_INDICES, dtype=np.int64)

# ============================================================================
# E.  CHECKPOINT INVENTORY -- targeted download, SHA-256, metadata
# ============================================================================
banner("E. CHECKPOINT INVENTORY")

TARGETS = {
    # label: (phase, img_size, [candidate filenames, in preference order])
    "ff_112": (("ff_112",), 112, ["foundation_ff_112_final.pth"]),
    "recurrence_only": (("recurrence_only",), 56, ["foundation_recurrence_only_final.pth"]),
    "c1_gist_only_cont": (("g2_gist_only",), 56, ["foundation_g2_gist_only_cont_final.pth"]),
    "c3_refine_single": (("c3_refine_single",), 56, ["foundation_c3_refine_single_final.pth"]),
    "c4_four_views_no_refine": (("c4_four_views_no_refine",), 56,
                                ["foundation_c4_four_views_no_refine_final.pth"]),
    "g2_gist_only": (("g2_gist_only",), 56,
                     ["foundation_g2_gist_only_final.pth", "foundation_g2_gist_only_final.zip"]),
    "gen1_core": (("gen1_core",), 56, ["foundation_gen1_core_best_clean.pth"]),
}

FF56_CANDIDATES = [
    "ff_56_direct_rung3.pth",
    "foundation_ff_56_final.pth",
    "foundation_ff_56_final.zip",
    "ff_56_defective.pth",
    "foundation_ff_56_defective.pth",
]

# The Rung-3 15-epoch monitor for ff_56 in report/controls/gen2_epoch_log.jsonl.
FF56_MONITOR = 0.5904
FF56_MONITOR_TOL = 0.002


def fetch_from_hf(name: str, required: bool = False) -> Optional[Path]:
    dest = CKPT_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    for repo in (HF_BEST, HF_ROLLING):
        try:
            p = hf_hub_download(repo_id=repo, filename=name, repo_type="dataset", token=hf_token)
            shutil.copyfile(p, dest)
            print(f"  downloaded {name} from {repo}")
            return dest
        except Exception:
            continue
    if required:
        print(f"  ! {name} not found on HF")
    return None


def load_state(path: str | Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model" in obj:
        meta = {k: v for k, v in obj.items() if k not in ("model", "optimizer", "scheduler")}
        return obj["model"], meta
    return obj, {}


INVENTORY: Dict[str, Any] = {}


def register_file(name: str, path: Path, role: str) -> Dict[str, Any]:
    sha = sha256_file(path)
    try:
        _, meta = load_state(path)
    except Exception as e:
        meta = {"_load_error": str(e)[:200]}
    rec = {
        "file": name,
        "sha256": sha,
        "bytes": path.stat().st_size,
        "role": role,
        "phase": meta.get("phase"),
        "epoch": meta.get("epoch"),
        "val_clean_acc": meta.get("val_clean_acc"),
        "val_rob_acc": meta.get("val_rob_acc"),
        "best_clean_acc": meta.get("best_clean_acc"),
        "reported_monitor": bool(meta.get("val_clean_acc") is not None),
    }
    INVENTORY[name] = rec
    return rec


print("\n-- ff_56 candidates --")
FF56_FILES: Dict[str, Dict[str, Any]] = {}
for cand in FF56_CANDIDATES:
    p = fetch_from_hf(cand)
    if p is None:
        continue
    FF56_FILES[cand] = register_file(cand, p, role="ff_56_candidate")
    r = FF56_FILES[cand]
    print(f"  {cand:<42} sha256 {r['sha256'][:12]}  phase={r['phase']!r} "
          f"epoch={r['epoch']} clean={r['val_clean_acc']}")

# A locally attached true checkpoint (Kaggle input dataset) takes precedence.
_attached = os.environ.get("FF56_TRUE_PATH")
if not _attached:
    for pat in ("/kaggle/input/*/foundation_ff_56*.pth", "/kaggle/input/*/*/foundation_ff_56*.pth"):
        hits = sorted(Path("/").glob(pat.lstrip("/")))
        if hits:
            _attached = str(hits[0])
            break
if _attached and Path(_attached).exists():
    dest = CKPT_DIR / "ff_56_attached.pth"
    if sha256_file(_attached) != (sha256_file(dest) if dest.exists() else ""):
        shutil.copyfile(_attached, dest)
    FF56_FILES["ff_56_attached.pth"] = register_file("ff_56_attached.pth", dest, role="ff_56_candidate")
    print(f"  attached: {_attached} -> {FF56_FILES['ff_56_attached.pth']['sha256'][:12]} "
          f"phase={FF56_FILES['ff_56_attached.pth']['phase']!r}")

print("\n-- protocol arms --")
ARM_FILES: Dict[str, Path] = {}
for label, (phases, size, cands) in TARGETS.items():
    got = None
    for cand in cands:
        p = fetch_from_hf(cand, required=False)
        if p is None:
            continue
        rec = register_file(cand, p, role=f"arm:{label}")
        # a final checkpoint that reports its own monitor is the one we want
        if label == "c4_four_views_no_refine" and rec["phase"] not in (None, "c4_four_views_no_refine"):
            print(f"  ! {cand} reports phase {rec['phase']!r}; ignoring for C4")
            continue
        got = p
        break
    if got is None:
        print(f"  - {label}: no checkpoint available (arm will be skipped)")
    else:
        ARM_FILES[label] = got
        r = INVENTORY[got.name]
        print(f"  {label:<26} {got.name:<46} sha256 {r['sha256'][:12]} "
              f"epoch={r['epoch']} monitor={r['val_clean_acc']}")

json_dump(OUT_DIR / "checkpoint_inventory.json",
          {"repo_sha": REPO_SHA, "branch": BRANCH, "files": INVENTORY})


# ============================================================================
# F.  EVALUATION PRIMITIVES
# ============================================================================

def mcnemar(corr_a: np.ndarray, corr_b: np.ndarray) -> Dict[str, Any]:
    """Paired McNemar. Exact two-sided binomial is primary (valid for small b+c)."""
    a = np.asarray(corr_a).astype(bool)
    b = np.asarray(corr_b).astype(bool)
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    n_only_a = int(np.sum(a & ~b))   # A right, B wrong
    n_only_b = int(np.sum(~a & b))   # B right, A wrong
    n = n_only_a + n_only_b
    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "stat": 0.0,
                "p_exact": 1.0, "p_chi2cc": 1.0, "delta_pp": 100.0 * (a.mean() - b.mean())}
    stat = (abs(n_only_a - n_only_b) - 1.0) ** 2 / n
    p_exact = float(min(1.0, 2.0 * binom.cdf(min(n_only_a, n_only_b), n, 0.5)))
    return {
        "b": n_only_a, "c": n_only_b, "n_discordant": n,
        "stat": float(stat),
        "p_exact": p_exact,
        "p_chi2cc": 1.0 if stat == 0.0 else float(chi2.sf(stat, df=1)),
        "delta_pp": 100.0 * (float(a.mean()) - float(b.mean())),
    }


def classify_p(p: float) -> str:
    if p < THRESHOLDS["p_established"]:
        return "established (p<0.01)"
    if p < THRESHOLDS["p_ns"]:
        return "weak, not decisive (0.01<=p<0.05)"
    if p <= THRESHOLDS["p_not_established_band"]:
        return "not established (0.05<=p<=0.20)"
    return "no detectable difference (p>0.20)"


def build_base(phase: str, size: int, num_glimpses: int = 4) -> nn.Module:
    return Gen2FoundationModel(phase=phase, arm="v1", fovea_size=size,
                               num_glimpses=num_glimpses)


def clean_5k_scan(base: nn.Module, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-sample correctness over all 5,000 images (harness transform, AMP)."""
    loader = build_val_loaders(size, VAL_ROOT, batch_size=256, num_workers=2, raw_pixel=False)
    preds, labels = [], []
    base.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                logits = base(x)
            preds.append(logits.argmax(-1).cpu().numpy().astype(np.int16))
            labels.append(y.numpy().astype(np.int16))
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return preds, labels


def gate_checkpoint(base_unwrapped: nn.Module, meta: Dict[str, Any], size: int,
                    label: str) -> Dict[str, Any]:
    """Harness parity gate: 5k clean must match the checkpoint's own monitor."""
    loader = build_val_loaders(size, VAL_ROOT, batch_size=256, num_workers=2, raw_pixel=False)
    try:
        acc5k, monitor = verify_checkpoint_gate(base_unwrapped, meta, loader, DEVICE, tol_pp=1.0)
        passed, detail = True, f"5k={acc5k*100:.2f}% vs monitor {monitor*100:.2f}%"
    except HarnessGateError as e:
        acc5k, monitor, passed, detail = float("nan"), None, False, str(e)[:400]
    except Exception as e:  # missing metadata etc.
        acc5k, monitor, passed, detail = float("nan"), None, False, f"{type(e).__name__}: {e}"[:400]
    GATES.append({"gate": f"checkpoint parity ({label})", "passed": passed, "detail": detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] checkpoint gate: {detail}")
    return {"passed": passed, "acc_5k": acc5k, "monitor": monitor, "detail": detail}


def slice_battery(wrapped: nn.Module, size: int) -> Dict[str, Any]:
    """Clean + PGD-50 + APGD-CE + APGD-DLR on the canonical slice."""
    loader = build_val_loaders(size, VAL_ROOT, batch_size=32, num_workers=2,
                               indices=SLICE_INDICES, raw_pixel=True)
    apgd_ce = torchattacks.APGD(wrapped, norm="Linf", eps=EPS, steps=PROTOCOL["attacks"]["apgd_ce"]["steps"],
                                n_restarts=1, loss="ce", verbose=False)
    apgd_dlr = torchattacks.APGD(wrapped, norm="Linf", eps=EPS, steps=PROTOCOL["attacks"]["apgd_dlr"]["steps"],
                                 n_restarts=1, loss="dlr", verbose=False)

    out: Dict[str, Any] = {"corr": {}, "pred": {}, "logits": {}, "runtime_s": {},
                           "max_linf_255": {}}
    labels_all = None
    for atk in ("clean", "pgd50", "apgd_ce", "apgd_dlr"):
        t0 = time.time()
        preds_l, labels_l, logits_l = [], [], []
        max_linf = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if atk == "clean":
                x_adv = x
            elif atk == "pgd50":
                x_adv = run_pgd_pixel_space(wrapped, x, y, eps=EPS,
                                            steps=PROTOCOL["attacks"]["pgd50"]["steps"], restarts=1)
            elif atk == "apgd_ce":
                x_adv = apgd_ce(x, y)
            else:
                x_adv = apgd_dlr(x, y)

            if atk != "clean":
                batch_linf = float((x_adv - x).abs().max().item())
                max_linf = max(max_linf, batch_linf)
                assert batch_linf <= EPS + 1e-4, \
                    f"L_inf budget violated: {batch_linf*255:.4f}/255 > {EPS*255:.4f}/255"

            with torch.no_grad():
                logits = wrapped(x_adv)
            logits_l.append(logits.to(torch.float16).cpu().numpy())
            preds_l.append(logits.argmax(-1).cpu().numpy().astype(np.int16))
            labels_l.append(y.cpu().numpy().astype(np.int16))

        preds = np.concatenate(preds_l)
        labels = np.concatenate(labels_l)
        labels_all = labels if labels_all is None else labels_all
        out["pred"][atk] = preds
        out["logits"][atk] = np.concatenate(logits_l)
        out["corr"][atk] = (preds == labels).astype(np.int8)
        out["runtime_s"][atk] = time.time() - t0
        out["max_linf_255"][atk] = max_linf * 255.0
        print(f"    {atk:<9} {out['corr'][atk].mean()*100:6.2f}%  "
              f"({out['runtime_s'][atk]:5.1f}s"
              + (f", max L_inf {max_linf*255:.3f}/255" if atk != "clean" else "") + ")")

    out["corr"]["autoattack_core"] = (out["corr"]["apgd_ce"] & out["corr"]["apgd_dlr"]).astype(np.int8)
    out["corr"]["worst_case"] = (out["corr"]["pgd50"] & out["corr"]["apgd_ce"]
                                 & out["corr"]["apgd_dlr"]).astype(np.int8)
    out["labels"] = labels_all
    return out


# ── T-sweep trajectory (mirrors Gen2FoundationModel.forward for recurrence_only)
GAZE_SCHEDULE = [[0.0, 0.0], [0.3, 0.3], [-0.3, 0.3], [0.0, -0.3]]


def harvest_trajectory(base: nn.Module, x: torch.Tensor, t_max: int = 4):
    """Return {t: (z_t, logits_t)} for t = 1..t_max, exactly as forward() computes it.

    recurrence_only: z_0 = mean(gist tokens); z_t = 0.5 z_{t-1} + 0.5 pooled(glimpse_t);
    logits_t = cls_head(z_t).
    """
    B = x.shape[0]
    gist_tokens = base.gist_encoder(x)
    z = gist_tokens.mean(dim=1)
    out: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for t in range(t_max):
        gaze = torch.tensor(GAZE_SCHEDULE[t], device=x.device).view(1, 2).expand(B, -1)
        crop = foveal_sample(x, gaze, fovea_size=base.fovea_size)
        pooled, _ = base.backbone(crop)
        z = 0.5 * z + 0.5 * pooled
        out[t + 1] = (z, base.cls_head(z))
    return out


def parity_t_invariance(sd: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """T is an inference-time knob only if the parameter set is T-invariant."""
    report = {}
    for t in (1, 2, 3, 4):
        m = build_base("recurrence_only", 56, num_glimpses=t)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        report[f"T{t}"] = {"missing": len(missing), "unexpected": len(unexpected)}
        assert not missing and not unexpected, f"T={t} state_dict mismatch: {missing[:3]} {unexpected[:3]}"
    GATES.append({"gate": "T-invariance of parameters (T=1..4)", "passed": True,
                  "detail": "identical state_dict, 0 missing / 0 unexpected"})
    return report


def parity_forward(sd: Dict[str, torch.Tensor], x: torch.Tensor) -> Dict[str, Any]:
    """Module forward vs harvested trajectory readout (one model resident at a time)."""
    report = {}
    passed = True
    with torch.no_grad():
        for t in (1, 2, 3, 4):
            m = build_base("recurrence_only", 56, num_glimpses=t).to(DEVICE)
            m.load_state_dict(sd, strict=False)
            m.eval()
            module_logits = m(x)
            traj = harvest_trajectory(m, x, t_max=t)
            diff = float((module_logits - traj[t][1]).abs().max().item())
            report[f"T{t}"] = diff
            passed = passed and diff < 1e-5
            print(f"    forward parity T={t}: max|module - trajectory| = {diff:.2e}")
            del m
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    GATES.append({"gate": "forward parity (module vs harvested trajectory)", "passed": passed,
                  "detail": ", ".join(f"T{t}:{v:.1e}" for t, v in report.items())})
    return report


# ── CKA ──────────────────────────────────────────────────────────────────────
class Block12Harvester:
    """Capture the block-12 CLS token of the LAST trunk pass in each forward."""

    def __init__(self, base: nn.Module):
        self.act: Optional[np.ndarray] = None
        self.calls = 0
        self._hook = base.backbone.blocks[11].register_forward_hook(self._fn)

    def _fn(self, module, inputs, output):
        self.calls += 1
        self.act = output[:, 0].detach().float().cpu().numpy()

    def remove(self):
        self._hook.remove()


def linear_cka(x1: np.ndarray, x2: np.ndarray) -> float:
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    x1 = x1 - x1.mean(axis=0, keepdims=True)
    x2 = x2 - x2.mean(axis=0, keepdims=True)
    k1, k2 = x1 @ x1.T, x2 @ x2.T
    num = float(np.trace(k1 @ k2))
    den = math.sqrt(max(float(np.trace(k1 @ k1)), 0.0) * max(float(np.trace(k2 @ k2)), 0.0))
    return num / max(den, 1e-12)


def harvest_activations(wrapped: nn.Module, size: int) -> Tuple[np.ndarray, int]:
    harvester = Block12Harvester(wrapped.base_model)
    loader = build_val_loaders(size, VAL_ROOT, batch_size=64, num_workers=2,
                               indices=SLICE_INDICES, raw_pixel=True)
    feats = []
    try:
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(DEVICE)
                wrapped(x)
                feats.append(harvester.act)
    finally:
        calls_per_batch = harvester.calls / max(len(loader), 1)
        harvester.remove()
    return np.concatenate(feats, axis=0), int(round(calls_per_batch))


# ============================================================================
# G.  PHASE 0.1 -- ff_56 identity, then preserve-then-remove rename on HF
# ============================================================================
banner("G. PHASE 0.1 -- ff_56 IDENTITY + HF HOUSEKEEPING")

ff56_identity: Dict[str, Any] = {"candidates": FF56_FILES, "status": "UNRESOLVED",
                                 "true_file": None, "stale_files": [],
                                 "hg_actions": [], "note": ""}
valid = [n for n, r in FF56_FILES.items()
         if r.get("phase") == "ff_56" and r.get("epoch") == 15
         and isinstance(r.get("val_clean_acc"), (int, float))
         and abs(float(r["val_clean_acc"]) - FF56_MONITOR) <= FF56_MONITOR_TOL]

CANONICAL_FF56 = "ff_56_direct_rung3.pth"
if len(valid) == 1:
    ff56_identity["status"] = "RESOLVED"
    ff56_identity["true_file"] = valid[0]
    ff56_identity["aliases"] = []
    ff56_identity["stale_files"] = [n for n in FF56_FILES if n not in valid]
    print(f"  TRUE ff_56 = {valid[0]}  (phase ff_56, epoch 15, clean "
          f"{FF56_FILES[valid[0]]['val_clean_acc']:.4f})")
elif len(valid) > 1 and len({FF56_FILES[n]["sha256"] for n in valid}) == 1:
    # Several names, one byte-identical checkpoint: this is an alias set, not an
    # ambiguity.  The canonical name wins; the rest are renamed away in step 3.
    ff56_identity["status"] = "RESOLVED"
    true_name = CANONICAL_FF56 if CANONICAL_FF56 in valid else valid[0]
    ff56_identity["true_file"] = true_name
    ff56_identity["aliases"] = [n for n in valid if n != true_name]
    ff56_identity["stale_files"] = [n for n in FF56_FILES if n not in valid]
    print(f"  TRUE ff_56 = {true_name} (byte-identical aliases: "
          f"{ff56_identity['aliases']})")
elif len(valid) > 1:
    ff56_identity["note"] = (f"ambiguous: {len(valid)} candidates match the fingerprint with "
                             f"different bytes: {valid}")
    print(f"  ! AMBIGUOUS: {valid}")
elif not FF56_FILES:
    ff56_identity["note"] = (
        "no ff_56 candidate reachable from this environment (HF or attached input). "
        "Attach the local checkpoint folder as a Kaggle input dataset, or set FF56_TRUE_PATH."
    )
    print(f"  ! {ff56_identity['note']}")
else:
    ff56_identity["note"] = ("no candidate carries the Rung-3 15-epoch fingerprint "
                             f"(phase='ff_56', epoch=15, clean~{FF56_MONITOR}).")
    print(f"  ! {ff56_identity['note']}")

# Preserve-then-remove.  Each step only proceeds on verified byte equality.
RENAME = os.environ.get("CLOSURE_RENAME_LEGACY", "1") == "1"


def hf_lfs_sha(name: str) -> Optional[str]:
    return idx(HF_BEST).get(name)


if ff56_identity["status"] == "RESOLVED":
    true_name = ff56_identity["true_file"]
    true_path = CKPT_DIR / true_name
    true_sha = FF56_FILES[true_name]["sha256"]

    # 1. publish the true bytes under a permanent, unambiguous name
    new_name, act = upload_unique(true_path, HF_BEST, "ff_56_direct_rung3.pth")
    ff56_identity["hg_actions"].append({"action": "publish_true", "name": new_name, "sha256": true_sha,
                                        "result": act})
    published_sha = hf_lfs_sha(new_name) or (true_sha if act == "already_present" else None)
    ff56_identity["canonical_name"] = new_name

    # 2. preserve the stale bytes under ff_56_defective.pth
    stale_candidates = list(ff56_identity["stale_files"])
    stale_name = next((s for s in stale_candidates
                       if s.startswith(("ff_56_defective", "foundation_ff_56_defective"))),
                      stale_candidates[0] if stale_candidates else None)
    s_sha: Optional[str] = None
    preserved: Optional[str] = None
    preserved_sha: Optional[str] = None
    if stale_name is not None:
        s_sha = FF56_FILES[stale_name]["sha256"]
        preserved, act = upload_unique(CKPT_DIR / stale_name, HF_BEST, "ff_56_defective.pth")
        ff56_identity["hg_actions"].append({"action": "preserve_stale", "from": stale_name,
                                            "name": preserved, "sha256": s_sha, "result": act})
        preserved_sha = hf_lfs_sha(preserved)
        ff56_identity["stale_preserved_as"] = preserved
    else:
        print("  no stale candidate to preserve")

    # 3. remove the ambiguous legacy name -- only once its bytes are verified
    #    present under a permanent name in the same repo (preserve-then-remove).
    #    Re-read the repo index first: the check below must use HF's own LFS
    #    sha256, not the local hash we just recorded.
    _IDX.pop(HF_BEST, None)
    legacy = "foundation_ff_56_final.pth"
    legacy_sha = hf_lfs_sha(legacy)
    published_sha = hf_lfs_sha(new_name) or published_sha
    if preserved is not None:
        preserved_sha = hf_lfs_sha(preserved) or preserved_sha
    if legacy_sha is None:
        ff56_identity["hg_actions"].append({"action": "rename_noop", "name": legacy,
                                            "reason": "legacy name not present on HF"})
        print(f"  no legacy {legacy} on HF; nothing to rename")
    elif not RENAME:
        ff56_identity["hg_actions"].append({"action": "rename_skipped", "name": legacy,
                                            "reason": "CLOSURE_RENAME_LEGACY=0"})
        print("  legacy name kept (rename disabled by env)")
    elif legacy_sha == true_sha and published_sha == true_sha:
        api.delete_file(path_in_repo=legacy, repo_id=HF_BEST, repo_type="dataset", token=hf_token)
        idx(HF_BEST).pop(legacy, None)
        ff56_identity["hg_actions"].append({"action": "rename_true", "from": legacy,
                                            "to": new_name, "sha256": true_sha})
        print(f"  renamed {legacy} -> {new_name} (bytes verified before removal)")
    elif s_sha is not None and legacy_sha == s_sha and preserved_sha == s_sha:
        api.delete_file(path_in_repo=legacy, repo_id=HF_BEST, repo_type="dataset", token=hf_token)
        idx(HF_BEST).pop(legacy, None)
        ff56_identity["hg_actions"].append({"action": "rename_stale", "from": legacy,
                                            "to": preserved, "sha256": s_sha})
        print(f"  renamed stale {legacy} -> {preserved} (bytes verified before removal)")
    else:
        ff56_identity["hg_actions"].append({
            "action": "rename_blocked", "name": legacy,
            "reason": (f"{legacy} (sha {str(legacy_sha)[:10]}) matches neither the published "
                       f"true bytes ({str(published_sha)[:10]}) nor the preserved stale bytes "
                       f"({str(preserved_sha)[:10]}); left untouched")})
        print("  ! rename blocked: legacy bytes do not match a verified replacement; left untouched")

json_dump(OUT_DIR / "ff56_identity.json", ff56_identity)
print(f"  identity status: {ff56_identity['status']}  true={ff56_identity['true_file']}")

# ============================================================================
# H.  PHASE 0.2 -- CKA row and column for ff_56 (same 1,000 frames)
# ============================================================================
banner("H. PHASE 0.2 -- CKA ROW/COLUMN FOR ff_56")

INCLUDE_CONTEXT = os.environ.get("CLOSURE_CONTEXT_ARMS", "1") == "1"
CKA_ARMS: List[Tuple[str, str, int, Path, int]] = []   # label, phase, size, path, T
if ff56_identity["status"] == "RESOLVED":
    CKA_ARMS.append(("ff_56_direct_rung3", "ff_56", 56,
                     CKPT_DIR / ff56_identity["true_file"], 1))
for lbl, (phases, size, _c) in TARGETS.items():
    if lbl not in ARM_FILES:
        continue
    if lbl in ("ff_112", "g2_gist_only", "gen1_core") and not INCLUDE_CONTEXT:
        continue
    phase = phases[0]
    T = 1 if phase in ("g2_gist_only", "ff_56", "ff_112", "c3_refine_single",
                       "c4_four_views_no_refine") else 4
    CKA_ARMS.append((lbl, phase, size, ARM_FILES[lbl], T))

cka_feats: Dict[str, np.ndarray] = {}
cka_calls: Dict[str, int] = {}
cka_sha: Dict[str, str] = {}
for lbl, phase, size, path, T in CKA_ARMS:
    base = build_base(phase, size, num_glimpses=T).to(DEVICE)
    sd, _meta = load_state(path)
    base.load_state_dict(sd, strict=False)
    base.eval()
    del sd
    wrapped = NormalizedModel(base).to(DEVICE).eval()
    f, calls = harvest_activations(wrapped, size)
    cka_feats[lbl] = f
    cka_calls[lbl] = calls
    cka_sha[lbl] = INVENTORY[path.name]["sha256"]
    print(f"  {lbl:<26} feats {f.shape}  trunk passes/forward={calls}  sha256 {cka_sha[lbl][:10]}")
    del base, wrapped
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

cka_labels = list(cka_feats)
CKA: Dict[str, Dict[str, float]] = {a: {} for a in cka_labels}
for a in cka_labels:
    for b in cka_labels:
        CKA[a][b] = linear_cka(cka_feats[a], cka_feats[b])
print("\n  CKA (block-12 CLS, linear, n=1000):")
print("    " + " " * 26 + "".join(f"{b[:14]:>16}" for b in cka_labels))
for a in cka_labels:
    print(f"    {a:<26}" + "".join(f"{CKA[a][b]:>16.4f}" for b in cka_labels))

# historical matrix (Notebook 3) for the delta
HIST_LABEL = {"ff_56_direct_rung3": "ff_56", "recurrence_only": "recurrence_only",
              "c1_gist_only_cont": "g2_gist_only_cont", "ff_112": "ff_112_s17",
              "g2_gist_only": "g2_gist_only", "gen1_core": "gen1_core"}
hist = download_artifact(HF_BEST, "evidence_harvest/evidence_summary.json",
                         OUT_DIR / "historical_evidence_summary.json")
hist_cka = None
if hist is not None:
    try:
        hist_cka = json.loads(Path(hist).read_text()).get("cka_matrix_block12")
    except Exception as e:
        print(f"  (historical CKA unavailable: {e})")

cka_deltas: Dict[str, Dict[str, float]] = {}
if hist_cka and "ff_56" in hist_cka:
    for ours, theirs in HIST_LABEL.items():
        if theirs not in hist_cka or ours not in CKA:
            continue
        cka_deltas[ours] = {}
        for ours2, theirs2 in HIST_LABEL.items():
            if theirs2 not in hist_cka[theirs] or ours2 not in CKA:
                continue
            cka_deltas[ours][ours2] = CKA[ours][ours2] - hist_cka[theirs][theirs2]

cka_out = {
    "n": 1000,
    "slice_sha256": SLICE_HASH,
    "layer": "backbone.blocks[11] (block 12), CLS token, last trunk pass per forward",
    "estimator": "linear CKA, float64",
    "labels": cka_labels,
    "matrix": CKA,
    "trunk_passes_per_forward": cka_calls,
    "checkpoint_sha256": cka_sha,
    "ff_56_row": {b: CKA["ff_56_direct_rung3"][b] for b in cka_labels} if "ff_56_direct_rung3" in CKA else {},
    "ff_56_column": {a: CKA[a]["ff_56_direct_rung3"] for a in cka_labels} if "ff_56_direct_rung3" in CKA else {},
    "historical_matrix_found": bool(hist_cka),
    "delta_vs_historical": cka_deltas,
}
json_dump(OUT_DIR / "cka_ff56.json", cka_out)

P02: Dict[str, Any] = {"id": "P0.2", "status": "not_applicable",
                       "verdict": "ff_56 identity unresolved; row/column not attributable"}
if "ff_56_direct_rung3" in CKA:
    flagged = {f"{a}|{b}": d for a, row in cka_deltas.items() for b, d in row.items()
               if abs(d) > THRESHOLDS["cka_delta_flag"]}
    if hist_cka is None:
        P02.update({"status": "computed_no_history",
                    "verdict": "row/column recomputed; historical matrix unavailable for diff"})
    elif flagged:
        P02.update({"status": "changed",
                    "verdict": (f"{len(flagged)} entries moved by >"
                                f"{THRESHOLDS['cka_delta_flag']}: the historical ff_56 "
                                "row came from a different checkpoint"),
                    "flagged": flagged})
    else:
        P02.update({"status": "unchanged",
                    "verdict": "row/column reproduces the historical matrix within 0.05"})
print(f"  P0.2 -> {P02['status']}: {P02['verdict']}")

# ============================================================================
# I.  ARM BATTERY -- ff_56, C1, C3, recurrence_only at T=1..4
# ============================================================================
banner("I. ARM BATTERY (5k clean + 1k robust, per-sample saved)")

BATTERY: Dict[str, Dict[str, Any]] = {}
BUNDLE: Dict[str, np.ndarray] = {}


def run_arm(label: str, phase: str, size: int, path: Path, T: int,
            want_gate: bool = True,
            gate_override: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """5k clean scan + canonical-slice attack battery for one (checkpoint, T).

    want_gate runs the harness parity gate for this exact model.  It is False for
    T < 4 because the checkpoint's training monitor was measured at its native T
    (comparing a T=1 scan against a T=4 monitor would fail for a reason that is
    the experiment, not a defect).  gate_override carries the checkpoint-level
    gate result down to those arms so the record is still complete.
    """
    sha = INVENTORY[path.name]["sha256"]
    key = cache_key(label, sha, size, T)
    cached = load_result_cache(key)
    if cached is not None:
        rec, arrays = cached
        if gate_override is not None:
            rec["gate"] = gate_override
        # A cached arm must still contribute its parity gates, otherwise a
        # resumed session could report 'all gates passed' without them.
        _g = rec.get("gate") or {}
        if _g.get("passed") is not None:
            GATES.append({"gate": f"checkpoint parity ({label})", "passed": _g.get("passed"),
                          "detail": (_g.get("detail") or "") + " [replayed from cache]"})
        _p = rec.get("parity") or {}
        if isinstance(_p.get("drift_5k_pp"), (int, float)):
            GATES.append({"gate": f"two-pipeline 5k parity ({label})",
                          "passed": _p["drift_5k_pp"] <= 0.4,
                          "detail": f"drift={_p['drift_5k_pp']:.2f}pp [replayed from cache]"})
        if isinstance(_p.get("drift_1k_pp"), (int, float)):
            GATES.append({"gate": f"two-pipeline 1k parity ({label})",
                          "passed": _p["drift_1k_pp"] <= 0.4,
                          "detail": f"drift={_p['drift_1k_pp']:.2f}pp [replayed from cache]"})
        BATTERY[label] = rec
        for k, v in arrays.items():
            BUNDLE[f"{label}__{k}"] = v
        return rec

    if not path.exists():
        print(f"  - {label}: checkpoint missing, skipped")
        return None

    print(f"\n-- {label} (phase={phase}, {size}px, T={T}, ckpt={path.name}) --")
    base = build_base(phase, size, num_glimpses=T).to(DEVICE)
    sd, meta = load_state(path)
    missing, unexpected = base.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  ! state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        GATES.append({"gate": f"state_dict load ({label})", "passed": False,
                      "detail": f"missing={len(missing)} unexpected={len(unexpected)}"})
    base.eval()

    gate = gate_checkpoint(base, meta, size, label) if want_gate else {"passed": None}
    if gate_override is not None:
        gate = gate_override

    t0 = time.time()
    preds5k, labels5k = clean_5k_scan(base, size)
    acc5k = float((preds5k == labels5k).mean())
    drift = None
    if want_gate and isinstance(gate.get("acc_5k"), float) and not math.isnan(gate["acc_5k"]):
        drift = abs(acc5k - float(gate["acc_5k"])) * 100
        GATES.append({"gate": f"two-pipeline 5k parity ({label})", "passed": drift <= 0.4,
                      "detail": f"gate={gate['acc_5k']*100:.2f}% scan={acc5k*100:.2f}% drift={drift:.2f}pp"})
        if drift > 0.4:
            print(f"  ! two-pipeline 5k drift {drift:.2f} pp (>0.4) -- recorded, gate fails")

    wrapped = NormalizedModel(base).to(DEVICE).eval()
    slice_out = slice_battery(wrapped, size)

    # gate: slice clean from the 5k scan == slice clean from the raw-pixel scan
    clean_from_5k = (preds5k[SLICE_IDX] == labels5k[SLICE_IDX])
    drift_slice = abs(clean_from_5k.mean() - slice_out["corr"]["clean"].mean()) * 100
    GATES.append({"gate": f"two-pipeline 1k parity ({label})", "passed": drift_slice <= 0.4,
                  "detail": f"5k-scan subset={clean_from_5k.mean()*100:.2f}% "
                            f"raw-scan={slice_out['corr']['clean'].mean()*100:.2f}% drift={drift_slice:.2f}pp"})
    if drift_slice > 0.4:
        print(f"  ! two-pipeline 1k drift {drift_slice:.2f} pp (>0.4) -- recorded, gate fails")

    rec = {
        "label": label, "phase": phase, "img_size": size, "num_glimpses": T,
        "checkpoint": path.name, "sha256": sha,
        "meta": {k: meta.get(k) for k in ("phase", "epoch", "val_clean_acc", "val_rob_acc",
                                          "best_clean_acc", "best_rob_acc")},
        "gate": gate,
        "acc_clean_5k": acc5k,
        "slice": {k: float(v.mean()) for k, v in slice_out["corr"].items()},
        "parity": {"drift_5k_pp": drift, "drift_1k_pp": drift_slice},
        "runtime_s": {"clean_5k": time.time() - t0, "slice": slice_out["runtime_s"]},
        "max_linf_255": slice_out["max_linf_255"],
        "cache_key": key,
    }

    arrays = {
        "clean5k_pred": preds5k, "clean5k_label": labels5k,
        "clean5k_corr": (preds5k == labels5k).astype(np.int8),
        "slice_label": slice_out["labels"],
        "slice_clean_pred": slice_out["pred"]["clean"],
        "slice_clean_logits": slice_out["logits"]["clean"],
    }
    for atk in ("clean", "pgd50", "apgd_ce", "apgd_dlr", "autoattack_core", "worst_case"):
        arrays[f"slice_{atk}_corr"] = slice_out["corr"][atk]

    BATTERY[label] = rec
    for k, v in arrays.items():
        BUNDLE[f"{label}__{k}"] = v
    save_result_cache(key, rec, arrays)

    print(f"  => 5k clean {acc5k*100:.2f}% | slice clean {rec['slice']['clean']*100:.2f}% "
          f"| worst {rec['slice']['worst_case']*100:.2f}%")

    del base, wrapped
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return rec


# 1) the reference arms
if ff56_identity["status"] == "RESOLVED":
    run_arm("ff_56_direct_rung3", "ff_56", 56, CKPT_DIR / ff56_identity["true_file"], 1)
for lbl, phase in (("c1_gist_only_cont", "g2_gist_only"), ("c3_refine_single", "c3_refine_single")):
    if lbl in ARM_FILES:
        p = ARM_FILES[lbl]
        run_arm(lbl, phase, 56, p, 1)

# 2) T-sweep on recurrence_only
T_PARITY: Dict[str, Any] = {}
if "recurrence_only" in ARM_FILES:
    rec_path = ARM_FILES["recurrence_only"]
    rec_sha = INVENTORY[rec_path.name]["sha256"]
    sd_ref, meta_ref = load_state(rec_path)

    T_PARITY["t_invariance"] = parity_t_invariance(sd_ref)

    # gate the checkpoint once (T=4 native config), then sweep
    base4 = build_base("recurrence_only", 56, num_glimpses=4).to(DEVICE)
    base4.load_state_dict(sd_ref, strict=False)
    base4.eval()
    gate_rec = gate_checkpoint(base4, meta_ref, 56, "recurrence_only")

    # construction parity: the manual builder must equal the harness loader
    ref_model, _ref_meta = load_model_from_checkpoint(
        "recurrence_only", 56, str(rec_path), DEVICE, wrap_normalized=False)
    _w_ref, _w_mine = ref_model.state_dict(), base4.state_dict()
    _diffs = [float((_w_ref[k].float() - _w_mine[k].float()).abs().max()) for k in _w_ref]
    _cp_ok = max(_diffs) == 0.0 and len(_w_ref) == len(_w_mine)
    GATES.append({"gate": "construction parity (harness loader vs manual builder)",
                  "passed": _cp_ok,
                  "detail": f"{len(_diffs)} tensors, max|delta|={max(_diffs):.1e}"})
    print(f"  [{'PASS' if _cp_ok else 'FAIL'}] construction parity: "
          f"{len(_diffs)} tensors, max|delta|={max(_diffs):.1e}")
    del ref_model, _w_ref, _w_mine

    # forward parity on one real batch (builds one model at a time)
    _xb = next(iter(build_val_loaders(56, VAL_ROOT, batch_size=16, num_workers=0,
                                     raw_pixel=True)))[0].to(DEVICE)
    T_PARITY["forward_parity"] = parity_forward(sd_ref, _xb)
    del base4
    sd_ref = None
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    for t in (1, 2, 3, 4):
        run_arm(f"recurrence_only_T{t}", "recurrence_only", 56, rec_path, t,
                want_gate=False, gate_override=gate_rec)

    # trajectory harvest for the halting probe (clean slice, z_t and logits_t)
    key = cache_key("recurrence_traj", rec_sha, 56, 4)
    cached = load_result_cache(key)
    if cached is not None:
        _rec, traj_arrays = cached
    else:
        print("\n  harvesting clean per-step trajectory (z_t, logits_t) on the slice...")
        base = build_base("recurrence_only", 56, num_glimpses=4).to(DEVICE)
        sd, _meta = load_state(rec_path)
        base.load_state_dict(sd, strict=False)
        base.eval()
        # raw_pixel=True loader produces [0, 1] tensors.
        # harvest_trajectory calls base.gist_encoder and base.backbone directly
        # (no NormalizedModel wrapper), so normalize here to match slice_battery.
        loader = build_val_loaders(56, VAL_ROOT, batch_size=64, num_workers=2,
                                   indices=SLICE_INDICES, raw_pixel=True)
        z_buf = {t: [] for t in (1, 2, 3, 4)}
        l_buf = {t: [] for t in (1, 2, 3, 4)}
        lab_buf = []
        with torch.no_grad():
            for x, y in loader:
                x = _normalize_pixel(x.to(DEVICE))  # normalize before trajectory
                traj = harvest_trajectory(base, x, t_max=4)
                for t in (1, 2, 3, 4):
                    z_buf[t].append(traj[t][0].to(torch.float16).cpu().numpy())
                    l_buf[t].append(traj[t][1].to(torch.float16).cpu().numpy())
                lab_buf.append(y.numpy().astype(np.int16))
        traj_arrays = {f"z_T{t}": np.concatenate(z_buf[t]) for t in (1, 2, 3, 4)}
        traj_arrays.update({f"logits_T{t}": np.concatenate(l_buf[t]) for t in (1, 2, 3, 4)})
        traj_arrays["label"] = np.concatenate(lab_buf)

        # Parity gate: trajectory logits at T must agree with run_arm's slice preds.
        # Both now use the same ImageNet normalization. Agreement must be >= 99.9%.
        # A failure here means the two normalization paths diverged and no T-sweep
        # result should be used until the cause is identified.
        traj_parity_passed = True
        for t in (1, 2, 3, 4):
            same = float((traj_arrays[f"logits_T{t}"].argmax(-1) ==
                          BUNDLE[f"recurrence_only_T{t}__slice_clean_pred"]).mean())
            ok = same >= 0.999
            traj_parity_passed = traj_parity_passed and ok
            print(f"  [{'PASS' if ok else 'FAIL'}] trajectory/slice parity T={t}: "
                  f"{same*100:.2f}% agreement")
        GATES.append({"gate": "trajectory/slice parity (normalized, T=1..4)",
                      "passed": traj_parity_passed,
                      "detail": "harvest_trajectory(_normalize_pixel(x)) vs run_arm slice preds"})
        if not traj_parity_passed:
            print("  ! trajectory parity gate failed -- investigate normalization path "
                  "before using T-sweep results.")
        save_result_cache(key, {"kind": "trajectory", "sha256": rec_sha}, traj_arrays)
        del base
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    for k, v in traj_arrays.items():
        BUNDLE[f"recurrence_traj__{k}"] = v



# ============================================================================
# J.  CHECKPOINT TABLE + PHASE 0.3 -- C3 vs C1
# ============================================================================
banner("J. HEADLINE TABLE + PHASE 0.3 (C3 vs C1)")

HEAD_ORDER = [l for l in ("ff_56_direct_rung3", "c1_gist_only_cont", "c3_refine_single",
                          "recurrence_only_T1", "recurrence_only_T2", "recurrence_only_T3",
                          "recurrence_only_T4") if l in BATTERY]
print(f"  {'arm':<24}{'5k clean':>10}{'1k clean':>10}{'PGD-50':>9}{'APGD-CE':>9}"
      f"{'APGD-DLR':>10}{'worst':>8}")
for lbl in HEAD_ORDER:
    r = BATTERY[lbl]
    s = r["slice"]
    print(f"  {lbl:<24}{r['acc_clean_5k']*100:>9.2f}%{s['clean']*100:>9.2f}%"
          f"{s['pgd50']*100:>8.2f}%{s['apgd_ce']*100:>8.2f}%{s['apgd_dlr']*100:>9.2f}%"
          f"{s['worst_case']*100:>7.2f}%")

DECISIONS: List[Dict[str, Any]] = []
P03: Dict[str, Any] = {"id": "P0.3", "status": "not_evaluable"}
if "c3_refine_single" in BATTERY and "c1_gist_only_cont" in BATTERY:
    c3, c1 = BATTERY["c3_refine_single"], BATTERY["c1_gist_only_cont"]
    d5k = (c3["acc_clean_5k"] - c1["acc_clean_5k"]) * 100.0
    m5k = mcnemar(BUNDLE["c3_refine_single__clean5k_corr"],
                 BUNDLE["c1_gist_only_cont__clean5k_corr"])
    m_worst = mcnemar(BUNDLE["c3_refine_single__slice_worst_case_corr"],
                      BUNDLE["c1_gist_only_cont__slice_worst_case_corr"])
    null = (abs(d5k) <= THRESHOLDS["pp_band"]) and (m5k["p_exact"] >= THRESHOLDS["p_ns"])
    rec_gain = rec_c3_gain = None
    if "recurrence_only_T4" in BATTERY:
        rec_gain = (BATTERY["recurrence_only_T4"]["acc_clean_5k"] - c1["acc_clean_5k"]) * 100.0
        rec_c3_gain = (c3["acc_clean_5k"] - c1["acc_clean_5k"]) * 100.0
    depth = (None if (rec_gain is None or rec_gain <= 0)
             else rec_c3_gain / rec_gain)
    P03 = {
        "id": "P0.3", "status": "evaluated",
        "delta_5k_pp": d5k, "mcnemar_5k": m5k, "mcnemar_worst": m_worst,
        "c3_5k": c3["acc_clean_5k"], "c1_5k": c1["acc_clean_5k"],
        "c3_worst": c3["slice"]["worst_case"], "c1_worst": c1["slice"]["worst_case"],
        "rec_minus_c1_clean_pp": rec_gain,
        "c3_recovery_fraction": depth,
        "is_null": bool(null),
        "depth_attributed": bool(depth is not None and depth >= THRESHOLDS["c3_depth_recovery_frac"]),
        "verdict": ("NULL: C3 and C1 are indistinguishable on 5k clean "
                    f"({d5k:+.2f} pp, p={m5k['p_exact']:.4g}). Depth does not replace the loop."
                    if null else
                    f"C3 differs from C1 on 5k clean ({d5k:+.2f} pp, p={m5k['p_exact']:.4g})."),
    }
    print(f"\n  C3 vs C1 (5k clean): {d5k:+.2f} pp, McNemar b={m5k['b']} c={m5k['c']}, "
          f"p_exact={m5k['p_exact']:.4g} -> {classify_p(m5k['p_exact'])}")
    print(f"  C3 vs C1 (1k worst): {m_worst['delta_pp']:+.2f} pp, "
          f"p_exact={m_worst['p_exact']:.4g} -> {classify_p(m_worst['p_exact'])}")
    if depth is not None:
        print(f"  depth recovery fraction: {depth*100:.1f}% of the recurrence-clean gap "
              f"({rec_gain:+.2f} pp)")
DECISIONS.append(P03)
json_dump(OUT_DIR / "p03_c3_vs_c1.json", P03)

# ============================================================================
# K.  PHASE 1.1 -- T-sweep decision gate
# ============================================================================
banner("K. PHASE 1.1 -- T-SWEEP (pre-registered gate)")

P11: Dict[str, Any] = {"id": "P1.1", "status": "not_evaluable"}
GATE_BRANCH = None
if "recurrence_only_T1" in BATTERY and "recurrence_only_T4" in BATTERY:
    t1 = BATTERY["recurrence_only_T1"]
    t4 = BATTERY["recurrence_only_T4"]
    sweep = {f"T{t}": {"acc_5k": BATTERY[f"recurrence_only_T{t}"]["acc_clean_5k"],
                       "clean_1k": BATTERY[f"recurrence_only_T{t}"]["slice"]["clean"],
                       "worst_1k": BATTERY[f"recurrence_only_T{t}"]["slice"]["worst_case"]}
             for t in (1, 2, 3, 4) if f"recurrence_only_T{t}" in BATTERY}
    print(f"  {'T':<4}{'5k clean':>10}{'1k clean':>10}{'1k worst':>10}")
    for k, v in sweep.items():
        print(f"  {k:<4}{v['acc_5k']*100:>9.2f}%{v['clean_1k']*100:>9.2f}%{v['worst_1k']*100:>9.2f}%")

    P11.update({"sweep": sweep, "status": "evaluated"})

    # monotonicity diagnostic
    accs = [sweep[k]["acc_5k"] for k in ("T1", "T2", "T3", "T4") if k in sweep]
    P11["monotone_clean"] = all(b >= a - 1e-9 for a, b in zip(accs, accs[1:]))
    P11["step_gains_pp"] = [100.0 * (b - a) for a, b in zip(accs, accs[1:])]

    if "ff_56_direct_rung3" in BATTERY:
        f56 = BATTERY["ff_56_direct_rung3"]
        d = (t1["acc_clean_5k"] - f56["acc_clean_5k"]) * 100.0
        m1k = mcnemar(BUNDLE["recurrence_only_T1__clean5k_corr"],
                      BUNDLE["ff_56_direct_rung3__clean5k_corr"])
        m4k = mcnemar(BUNDLE["recurrence_only_T4__clean5k_corr"],
                      BUNDLE["ff_56_direct_rung3__clean5k_corr"])
        m1w = mcnemar(BUNDLE["recurrence_only_T1__slice_worst_case_corr"],
                      BUNDLE["ff_56_direct_rung3__slice_worst_case_corr"])
        if abs(d) <= THRESHOLDS["pp_band"] and m1k["p_exact"] >= THRESHOLDS["p_ns"]:
            GATE_BRANCH = "A"
            why = (f"T=1 is within {THRESHOLDS['pp_band']:.1f} pp of ff_56 ({d:+.2f} pp) and "
                   f"McNemar is not significant (p={m1k['p_exact']:.3g}): the clean gain is "
                   "inference-time multi-view on weights trained for it.")
        elif d > THRESHOLDS["pp_band"]:
            GATE_BRANCH = "B"
            why = (f"T=1 beats ff_56 by {d:+.2f} pp on 5k clean: the gain is in the weights, "
                   "not in the extra views.")
        else:
            GATE_BRANCH = "C"
            why = (f"T=1 is {d:+.2f} pp below ff_56 on 5k clean: the extra views do the work; "
                   "the mechanism question is multi-view.")
        P11.update({
            "vs_ff56_T1": {"delta_5k_pp": d, "mcnemar_clean5k": m1k,
                           "mcnemar_worst_1k": m1w},
            "vs_ff56_T4": {"delta_5k_pp": (t4["acc_clean_5k"] - f56["acc_clean_5k"]) * 100.0,
                           "mcnemar_clean5k": m4k},
            "branch": GATE_BRANCH, "rationale": why,
            "next": {"A": "1.2 oracle + probe; 1.3 not required",
                     "B": "1.3 C4 (weights vs multi-view); 1.2 recorded descriptively",
                     "C": "1.2 oracle + 1.3 C4 (multi-view mechanism)"}[GATE_BRANCH],
        })
        print(f"\n  T=1 vs ff_56: 5k clean {d:+.2f} pp | McNemar b={m1k['b']} c={m1k['c']} "
              f"p_exact={m1k['p_exact']:.4g} ({classify_p(m1k['p_exact'])})")
        print(f"  T=4 vs ff_56: 5k clean {(t4['acc_clean_5k']-f56['acc_clean_5k'])*100:+.2f} pp | "
              f"p_exact={m4k['p_exact']:.4g}")
        print(f"  T=1 vs ff_56 worst-case (1k): {m1w['delta_pp']:+.2f} pp, "
              f"p_exact={m1w['p_exact']:.4g}")
        print(f"\n  GATE -> branch {GATE_BRANCH}: {why}")
    else:
        P11["status"] = "ff56_unresolved"
        P11["gate_applied"] = False
        P11["branch"] = None
        P11["rationale"] = ("ff_56 identity is unresolved, so the pre-registered comparison "
                            "cannot be made. The T-sweep numbers are recorded; the gate is not "
                            "applied and no Phase 1 verdict is decisive. C4 still runs because "
                            "the weights-vs-views question is open either way.")
        print(f"  ! {P11['rationale']}")
    P11.setdefault("gate_applied", GATE_BRANCH is not None)
else:
    print("  ! T-sweep unavailable (recurrence_only checkpoint missing)")
DECISIONS.append(P11)
json_dump(OUT_DIR / "p11_tsweep.json", P11)

# ============================================================================
# L.  PHASE 1.2 -- oracle halting curve (CPU) + conditional probe
# ============================================================================
banner("L. PHASE 1.2 -- ORACLE HALTING")


def oracle_analysis(corr_by_t: Dict[int, np.ndarray]) -> Dict[str, Any]:
    """corr_by_t: {t: bool array}. Oracle stop-at-best-T accuracy and expected views."""
    ts = sorted(corr_by_t)
    stack = np.stack([np.asarray(corr_by_t[t]).astype(bool) for t in ts], axis=0)  # (T, n)
    n = stack.shape[1]
    curve = {f"T<={t}": float(stack[:i + 1].any(axis=0).mean()) for i, t in enumerate(ts)}
    ever = stack.any(axis=0)
    first = np.where(ever, stack.argmax(axis=0) + 1, ts[-1])
    return {
        "n": int(n),
        "accuracy_by_fixed_T": {f"T={t}": float(stack[i].mean()) for i, t in enumerate(ts)},
        "oracle_curve": curve,
        "oracle_all_T": float(ever.mean()),
        "expected_views": float(first.mean()),
        "views_histogram": {int(k): int((first == k).sum()) for k in ts},
        "headroom_vs_last_pp": 100.0 * (float(ever.mean()) - float(stack[-1].mean())),
        "headroom_vs_T4_pp": 100.0 * (float(ever.mean()) - float(stack[ts.index(4)].mean()))
        if 4 in ts else None,
        "samples_fixed_by_extra_views": int(np.sum(stack[1:].any(axis=0) & ~stack[0])),
    }


P12: Dict[str, Any] = {"id": "P1.2", "status": "not_evaluable"}
T_TAGS = [t for t in (1, 2, 3, 4) if f"recurrence_only_T{t}" in BATTERY]
if len(T_TAGS) == 4:
    ORACLE: Dict[str, Any] = {}
    for space, key in (("clean_slice", "slice_clean_corr"), ("worst_slice", "slice_worst_case_corr")):
        corr_by_t = {t: BUNDLE[f"recurrence_only_T{t}__{key}"].astype(bool) for t in T_TAGS}
        ORACLE[space] = oracle_analysis(corr_by_t)
    ORACLE["clean_5k"] = oracle_analysis(
        {t: BUNDLE[f"recurrence_only_T{t}__clean5k_corr"].astype(bool) for t in T_TAGS})

    print(f"  {'space':<14}{'T=1':>9}{'T=2':>9}{'T=3':>9}{'T=4':>9}{'oracle':>9}"
          f"{'headroom':>10}{'E[views]':>10}")
    for space, res in ORACLE.items():
        acc = res["accuracy_by_fixed_T"]
        hr = res["headroom_vs_T4_pp"]
        print(f"  {space:<14}" + "".join(f"{acc[f'T={t}']*100:>8.2f}%" for t in T_TAGS)
              + f"{res['oracle_all_T']*100:>8.2f}%" + (f"{hr:>9.2f}pp" if hr is not None else " " * 9)
              + f"{res['expected_views']:>10.2f}")

    hr_clean5k = ORACLE["clean_5k"]["headroom_vs_T4_pp"]
    P12.update({"oracle": ORACLE, "status": "evaluated",
                "headroom_clean5k_pp": hr_clean5k,
                "headroom_clean1k_pp": ORACLE["clean_slice"]["headroom_vs_T4_pp"],
                "expected_views_clean5k": ORACLE["clean_5k"]["expected_views"]})

    has_headroom = hr_clean5k >= THRESHOLDS["oracle_headroom_pp"]
    P12["adaptive_compute_worthwhile"] = bool(has_headroom)
    print(f"\n  oracle headroom (5k clean, T<=4 vs T=4): {hr_clean5k:+.2f} pp "
          f"-> {'worth attempting' if has_headroom else 'nothing to gain at 56 px'}")

    # Probe only where the pre-registration says it matters (branch A).
    do_probe = has_headroom and GATE_BRANCH == "A" and "recurrence_traj__z_T1" in BUNDLE
    P12["probe_run"] = bool(do_probe)
    if do_probe:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import StratifiedKFold
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import roc_auc_score

            z1 = BUNDLE["recurrence_traj__z_T1"].astype(np.float32)
            lab = BUNDLE["recurrence_traj__label"]
            logits1 = BUNDLE["recurrence_traj__logits_T1"].astype(np.float32)
            logits4 = BUNDLE["recurrence_traj__logits_T4"].astype(np.float32)

            def probs(l):
                l = l - l.max(axis=1, keepdims=True)
                e = np.exp(l)
                return e / e.sum(axis=1, keepdims=True)

            p1, p4 = probs(logits1), probs(logits4)
            ent = -(p1 * np.log(np.clip(p1, 1e-12, 1.0))).sum(axis=1)
            maxp = p1.max(axis=1)
            top2 = np.sort(p1, axis=1)[:, -2:]
            margin = top2[:, 1] - top2[:, 0]

            corr1 = (p1.argmax(1) == lab)
            corr4 = (p4.argmax(1) == lab)
            y = (corr4 & ~corr1).astype(int)
            n_pos = int(y.sum())
            P12["probe_label"] = {"definition": "clean-correct at T=4 and clean-wrong at T=1",
                                  "positives": n_pos, "n": int(len(y))}
            print(f"  probe label: {n_pos}/{len(y)} samples were fixed by extra views")

            if n_pos < 30 or n_pos > len(y) - 30:
                P12["probe_verdict"] = ("not established: too few (or too many) positives "
                                        f"({n_pos}/{len(y)}) to fit a halting head")
                print(f"  ! {P12['probe_verdict']}")
            else:
                feats = {
                    "entropy": ent.reshape(-1, 1),
                    "confidence_only": np.stack([ent, maxp, margin], axis=1),
                    "z_T1": z1,
                    "z_T1+entropy": np.concatenate([z1, ent.reshape(-1, 1), maxp.reshape(-1, 1),
                                                    margin.reshape(-1, 1)], axis=1),
                }
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
                oof: Dict[str, np.ndarray] = {}
                for fname, X in feats.items():
                    scores = np.zeros(len(y))
                    for tr, te in skf.split(X, y):
                        sc = StandardScaler().fit(X[tr])
                        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
                        clf.fit(sc.transform(X[tr]), y[tr])
                        scores[te] = clf.decision_function(sc.transform(X[te]))
                    oof[fname] = scores
                    P12.setdefault("probe_auroc", {})[fname] = float(roc_auc_score(y, scores))

                ent_auc = P12["probe_auroc"]["entropy"]
                probe_auc = P12["probe_auroc"]["z_T1+entropy"]
                rng = np.random.default_rng(0)
                deltas = []
                for _ in range(1000):
                    ii = rng.integers(0, len(y), len(y))
                    if len(np.unique(y[ii])) < 2:
                        continue
                    deltas.append(roc_auc_score(y[ii], oof["z_T1+entropy"][ii])
                                  - roc_auc_score(y[ii], oof["entropy"][ii]))
                deltas = np.asarray(deltas)
                lo, hi = np.percentile(deltas, [2.5, 97.5])
                p_boot = float(np.mean(deltas <= 0))
                beats = (probe_auc - ent_auc >= THRESHOLDS["probe_auc_margin"]
                         and lo > 0)
                P12["probe_delta"] = {"auc_delta": probe_auc - ent_auc,
                                      "ci95": [float(lo), float(hi)],
                                      "p_bootstrap": p_boot,
                                      "beats_entropy": bool(beats)}
                P12["probe_verdict"] = (
                    f"probe beats entropy (+{probe_auc-ent_auc:.3f} AUROC, 95% CI [{lo:.3f}, {hi:.3f}])"
                    if beats else
                    f"entropy suffices: probe {probe_auc:.3f} vs entropy {ent_auc:.3f} "
                    f"(delta {probe_auc-ent_auc:+.3f}, CI [{lo:.3f}, {hi:.3f}])")
                print(f"  AUROC: entropy {ent_auc:.3f} | z_T1 {P12['probe_auroc']['z_T1']:.3f} | "
                      f"z_T1+entropy {probe_auc:.3f} | prior bar {THRESHOLDS['entropy_auc_bar']:.3f}")
                print(f"  {P12['probe_verdict']}")
        except Exception as e:
            P12["probe_run"] = False
            P12["probe_error"] = f"{type(e).__name__}: {e}"[:400]
            print(f"  ! probe failed: {P12['probe_error']}")
    else:
        P12["probe_verdict"] = (
            "not required by the pre-registered gate "
            f"(branch={GATE_BRANCH}, headroom={hr_clean5k:+.2f} pp)"
            if has_headroom else
            f"not attempted: oracle headroom {hr_clean5k:+.2f} pp < "
            f"{THRESHOLDS['oracle_headroom_pp']:.1f} pp")
        print(f"  probe: {P12['probe_verdict']}")
else:
    print("  ! full T=1..4 sweep unavailable")
DECISIONS.append(P12)
json_dump(OUT_DIR / "oracle_halting.json", P12)

# ============================================================================
# M.  PHASE 1.3 -- C4 (conditional): 4 views, refinement block off
# ============================================================================
banner("M. PHASE 1.3 -- C4 (4 views, no refinement)")

C4_MODE = os.environ.get("CLOSURE_C4", "auto").lower()
SESSION_BUDGET_H = float(os.environ.get("CLOSURE_SESSION_BUDGET_H", "10.5"))
C4_H_PER_EPOCH = 6.5 / 15.0   # C1/C3 both ran 15 epochs in ~4-6.5 GPU-h on a T4
P13: Dict[str, Any] = {"id": "P1.3", "status": "not_run", "mode": C4_MODE}

# auto: C4 is required for branches B and C; on branch A the gain was shown to be
# inference-time, so 1.3 is not owed.  If the gate could not be applied (ff_56
# unresolved) C4 still runs -- it is the only arm that separates views from depth.
_gate_needs_c4 = (GATE_BRANCH in ("B", "C")) if P11.get("gate_applied") \
    else ("recurrence_only_T4" in BATTERY)
want_c4 = (C4_MODE == "on") or (C4_MODE == "auto" and _gate_needs_c4)
if not want_c4:
    P13["reason"] = ("pre-registered gate (branch A) says the gain is inference-time; "
                     "1.3 not required" if GATE_BRANCH == "A"
                     else f"CLOSURE_C4={C4_MODE} and gate branch={GATE_BRANCH}")

C4_FINAL = CKPT_DIR / "foundation_c4_four_views_no_refine_final.pth"
C4_ROLLING = CKPT_DIR / "foundation_c4_four_views_no_refine_rolling.pth"

if want_c4:
    fetch_from_hf(C4_FINAL.name)
    fetch_from_hf(C4_ROLLING.name)
    valid_c4 = False
    if C4_FINAL.exists():
        _, _m = load_state(C4_FINAL)
        valid_c4 = (_m.get("phase") == "c4_four_views_no_refine" and _m.get("epoch") == 15)
    if valid_c4:
        print(f"  C4 final already present and verified ({C4_FINAL.name}); skipping training.")
        P13["status"] = "already_trained"
    else:
        # How much training is actually left (a resumed run may have done most of it).
        _start_epoch = 1
        if C4_ROLLING.exists():
            _, _rm = load_state(C4_ROLLING)
            _start_epoch = int(_rm.get("epoch") or 0) + 1
        _remaining = max(0, 15 - (_start_epoch - 1))
        _est_h = _remaining * C4_H_PER_EPOCH
        P13["estimate"] = {"start_epoch": _start_epoch, "remaining_epochs": _remaining,
                           "estimated_gpu_h": round(_est_h, 2),
                           "elapsed_h": round(elapsed_h(), 2),
                           "session_budget_h": SESSION_BUDGET_H}
        print(f"  C4 state: resume at epoch {_start_epoch}/15 ({_remaining} left, ~{_est_h:.1f} h); "
              f"session elapsed {elapsed_h():.2f} h of a {SESSION_BUDGET_H:.1f} h budget")
        if elapsed_h() + _est_h > SESSION_BUDGET_H:
            P13["status"] = "deferred"
            P13["reason"] = (
                f"training the remaining {_remaining} epochs (~{_est_h:.1f} h) would end at "
                f"{elapsed_h() + _est_h:.1f} h, past the {SESSION_BUDGET_H:.1f} h budget, and be "
                "killed mid-run. Re-run this file in a fresh session: it resumes from "
                f"{C4_ROLLING.name} and reuses every cached arm (no GPU spent twice).")
            print(f"  ! C4 deferred: {P13['reason']}")

    if P13["status"] == "not_run":   # neither already trained nor deferred
        print(f"  training C4 for 15 epochs at Rung 3 (elapsed {elapsed_h():.2f} h)")
        DINO_PATH = "checkpoints/dinov2_vits14_pretrain.pth"
        if not os.path.exists(DINO_PATH) or os.path.getsize(DINO_PATH) < 80 * 1024 * 1024:
            subprocess.run(
                "wget -q -c https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/"
                f"dinov2_vits14_pretrain.pth -O {DINO_PATH}", shell=True, check=True)

        def _sync_train_state():
            for p in (C4_ROLLING, C4_FINAL,
                      CKPT_DIR / "foundation_c4_four_views_no_refine_best_clean.pth",
                      CKPT_DIR / "foundation_c4_four_views_no_refine_best_robust.pth"):
                if p.exists():
                    if p.name.endswith("_rolling.pth"):
                        # session state, not an archive: overwrite is the point
                        try:
                            api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name,
                                            repo_id=HF_ROLLING, repo_type="dataset", token=hf_token)
                        except Exception as e:
                            print(f"    (rolling sync failed: {e})")
                    else:
                        upload_unique(p, HF_BEST, p.name)
            for j in OUT_DIR.glob("gen2_epoch_log*.jsonl"):
                upload_unique(j, HF_BEST, f"closure/{STAMP}/{j.name}")

        from training.train_gen2_foundation import run_gen2_phase
        kw = dict(phase="c4_four_views_no_refine", arm="v1", data_root=DATA_ROOT,
                  ckpt_dir="checkpoints", report_dir=str(OUT_DIR), dino_path=DINO_PATH,
                  batch_size=48, num_workers=4, smoke=False, img_size=56, n_epochs=15,
                  epoch_callback=lambda p, e: _sync_train_state())
        try:
            res = run_gen2_phase(**kw)
        except TypeError as e:
            print(f"  run_gen2_phase signature mismatch ({e}); CLI fallback")
            subprocess.run(
                "python3 training/train_gen2_foundation.py --phase c4_four_views_no_refine "
                f"--arm v1 --data-root {DATA_ROOT} --ckpt-dir checkpoints --report-dir {OUT_DIR} "
                f"--dino-path {DINO_PATH} --batch-size 48 --img-size 56 --n-epochs 15",
                shell=True, check=True)
            res = {"status": "cli"}
        _sync_train_state()
        P13["train_result"] = res
        P13["status"] = "trained" if C4_FINAL.exists() else "trained_no_final"
        print(f"  C4 training finished: {res}")

    c4_path = C4_FINAL if C4_FINAL.exists() else (C4_ROLLING if C4_ROLLING.exists() else None)
    if c4_path is None:
        P13["status"] = "no_checkpoint"
        print("  ! no C4 checkpoint available for evaluation")
    else:
        _, _cm = load_state(c4_path)
        P13["checkpoint_epoch"] = _cm.get("epoch")
        if _cm.get("epoch") != 15:
            # a mid-training checkpoint cannot answer the 1.3 question
            if P13["status"] != "deferred":
                P13["status"] = "incomplete"
            P13["checkpoint_complete"] = False
            P13["reason"] = (f"{c4_path.name} is at epoch {_cm.get('epoch')}/15; a mid-training "
                             "checkpoint is not comparable and was not evaluated")
            print(f"  ! {P13['reason']}")
        else:
            register_file(c4_path.name, c4_path, role="arm:c4_four_views_no_refine")
            c4_rec = run_arm("c4_four_views_no_refine", "c4_four_views_no_refine", 56, c4_path, 1)
            if c4_rec is not None and "recurrence_only_T4" in BATTERY:
                m_clean = mcnemar(BUNDLE["c4_four_views_no_refine__clean5k_corr"],
                                  BUNDLE["recurrence_only_T4__clean5k_corr"])
                m_worst = mcnemar(BUNDLE["c4_four_views_no_refine__slice_worst_case_corr"],
                                  BUNDLE["recurrence_only_T4__slice_worst_case_corr"])
                matches = (m_clean["p_exact"] >= THRESHOLDS["p_ns"]
                           and m_worst["p_exact"] >= THRESHOLDS["p_ns"])
                P13.update({
                    "status": "evaluated",
                    "c4_5k": c4_rec["acc_clean_5k"],
                    "c4_clean_1k": c4_rec["slice"]["clean"],
                    "c4_worst_1k": c4_rec["slice"]["worst_case"],
                    "mcnemar_vs_rec_T4_clean5k": m_clean,
                    "mcnemar_vs_rec_T4_worst1k": m_worst,
                    "matches_recurrence": bool(matches),
                    "verdict": ("MULTI-VIEW: C4 matches recurrence_only (p>=0.05 on 5k clean and "
                                "1k worst-case); the mechanism is multi-view, not extra depth."
                                if matches else
                                "INTERACTION: C4 does not match recurrence_only. The effect is an "
                                "interaction between views and depth; report it as unattributed and "
                                "spend no more on it."),
                })
                print("\n  C4 vs recurrence_only(T=4):")
                print(f"    5k clean   {m_clean['delta_pp']:+.2f} pp  p_exact={m_clean['p_exact']:.4g}")
                print(f"    1k worst   {m_worst['delta_pp']:+.2f} pp  p_exact={m_worst['p_exact']:.4g}")
                print(f"    -> {P13['verdict']}")
            P13.setdefault("note", "c4 checkpoint evaluated under the same protocol")
else:
    print(f"  skipped: {P13.get('reason', C4_MODE)}")
DECISIONS.append(P13)
json_dump(OUT_DIR / "p13_c4.json", P13)

# ============================================================================
# N.  PHASE 0.4 -- preserve docs/ and report/ with hashes
# ============================================================================
banner("N. PHASE 0.4 -- PRESERVE docs/ AND report/")


def resolve_docs_dir() -> Optional[Path]:
    cand = os.environ.get("RHAN_DOCS_DIR")
    if cand and Path(cand).is_dir():
        return Path(cand)
    for p in sorted(Path("/").glob("kaggle/input/*/docs")):
        if (p / "rhan_nxa").is_dir():
            return p
    if Path("docs").is_dir() and any(Path("docs").rglob("*.md")):
        return Path("docs")
    return None


P04: Dict[str, Any] = {"id": "P0.4", "tag": TAG_STATE, "status": "unavailable_in_this_environment"}
docs_dir = resolve_docs_dir()
if docs_dir is None:
    P04["reason"] = ("docs/ and report/ are gitignored, so they are absent from the Kaggle clone. "
                     "Attach them as a Kaggle input dataset and set RHAN_DOCS_DIR, or run "
                     "`tar -czf docs_report_export.tar.gz docs report` on the authoring machine "
                     "and upload it to " + HF_DOCS + ".")
    print(f"  ! {P04['reason']}")
else:
    root = docs_dir.parent if docs_dir.name == "docs" else Path(".")
    entries: Dict[str, Dict[str, Any]] = {}
    files: List[Path] = []
    for sub in ("docs", "report"):
        base = (root / sub) if (root / sub).is_dir() else (docs_dir if sub == "docs" else None)
        if base is None or not base.is_dir():
            continue
        files.extend([p for p in base.rglob("*") if p.is_file()])
    for p in files:
        if p.stat().st_size > 400 * 1024 * 1024:
            continue
        entries[str(p)] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    manifest = {"repo_sha": REPO_SHA, "created_utc": PREREG["created_utc"],
                "tag": TAG_STATE, "root": str(root.resolve()), "n_files": len(entries),
                "files": entries,
                "aggregate_sha256": hashlib.sha256(
                    json.dumps(entries, sort_keys=True).encode()).hexdigest()}
    json_dump(OUT_DIR / "docs_report_manifest.json", manifest)
    # The archive must NOT live inside report/ (it would archive itself).
    archive = Path(tempfile.mkdtemp()) / f"docs_report_export_{REPO_SHA[:8]}.tar.gz"
    subprocess.run(f"tar -czf {archive} " + " ".join(
        f"-C {root} {s}" for s in ("docs", "report") if (root / s).is_dir()), shell=True, check=True)
    a_name, _ = upload_unique(archive, HF_DOCS, archive.name)
    m_name, _ = upload_unique(OUT_DIR / "docs_report_manifest.json", HF_DOCS,
                              "docs_report_manifest.json")
    P04.update({"status": "exported", "n_files": len(entries), "archive": a_name,
                "archive_sha256": sha256_file(archive),
                "archive_bytes": archive.stat().st_size, "manifest": m_name,
                "aggregate_sha256": manifest["aggregate_sha256"]})
    if not (Path(root) / "docs" / "rhan_nxa").is_dir():
        P04["warning"] = ("docs/rhan_nxa is absent from this source: only the tracked part of "
                          "docs/ was exported. The handbook sources are gitignored and must be "
                          "attached (RHAN_DOCS_DIR) to be preserved.")
        print(f"  ! {P04['warning']}")
    print(f"  exported {len(entries)} files ({archive.stat().st_size/1e6:.1f} MB) as "
          f"{HF_DOCS}/{a_name}")
    print(f"  aggregate sha256 {manifest['aggregate_sha256'][:16]}...")
DECISIONS.append(P04)

# ============================================================================
# O.  REPORT + FINAL UPLOADS
# ============================================================================
banner("O. FINAL REPORT")

gates_pass = all(g["passed"] is not False for g in GATES)
parity_ok = gates_pass and ff56_identity["status"] == "RESOLVED"
for d in DECISIONS:
    # "decisive" is the conjunction of (a) the analysis ran and (b) every parity
    # gate passed.  A p-value near 0.1 never makes a result decisive.
    if d.get("id") in ("P0.3", "P1.1", "P1.2", "P1.3"):
        d["decisive"] = bool(parity_ok and d.get("status") == "evaluated")
    else:
        d["decisive"] = None

RESULTS = {
    "protocol_id": PREREG["id"], "prereg_hash": PREREG_HASH, "stamp": STAMP,
    "repo_sha": REPO_SHA, "branch": BRANCH, "tag": TAG_STATE,
    "eps_pixel": EPS, "eps_pixel_255": EPS * 255.0,
    "slice_sha256": SLICE_HASH,
    "gates": GATES, "parity_ok": bool(parity_ok),
    "checkpoint_inventory": INVENTORY,
    "ff56_identity": ff56_identity,
    "cka": cka_out,
    "battery": {k: {kk: vv for kk, vv in v.items() if kk != "cache_key"}
                for k, v in BATTERY.items()},
    "t_parity": T_PARITY,
    "decisions": DECISIONS,
    "hf_actions": HF_ACTIONS,
    "session_hours": elapsed_h(),
}
json_dump(OUT_DIR / "closure_results.json", RESULTS)

np.savez_compressed(OUT_DIR / "per_sample_bundle.npz", **BUNDLE)
print(f"  per-sample bundle: {len(BUNDLE)} arrays, "
      f"{(OUT_DIR / 'per_sample_bundle.npz').stat().st_size/1e6:.1f} MB")

# ── markdown report ──────────────────────────────────────────────────────────
md: List[str] = []
md.append(f"# Gen-2 closure run — {STAMP}\n")
md.append(f"- repo `{BRANCH}` @ `{REPO_SHA[:12]}`")
md.append(f"- pre-registration hash `{PREREG_HASH[:32]}` (rules written before the run; see §A)")
md.append(f"- eps = {EPS*255:.3f}/255; clean on 5,000; robust on the canonical 1,000-slice "
          f"(`{SLICE_HASH[:12]}…`)")
md.append(f"- parity gates: **{'PASS' if parity_ok else 'NOT all passed'}** "
          f"({sum(1 for g in GATES if g['passed'] is False)} failed)")
md.append(f"- session: {elapsed_h():.2f} GPU/CPU-hours\n")

md.append("## Checkpoints (SHA-256)\n")
md.append("| file | role | sha256 | phase | epoch | monitor clean |")
md.append("|---|---|---|---|---|---|")
for name, r in INVENTORY.items():
    md.append(f"| `{name}` | {r['role']} | `{r['sha256'][:16]}…` | {r['phase']} | "
              f"{r['epoch']} | {r['val_clean_acc']} |")

md.append("\n## Phase 0\n")
md.append(f"**0.1 ff_56 identity** — `{ff56_identity['status']}`. "
          f"True file: `{ff56_identity['true_file']}`. {ff56_identity.get('note','')}")
for a in ff56_identity.get("hg_actions", []):
    md.append(f"- {a.get('action')}: {a.get('from', a.get('name',''))} -> "
              f"{a.get('to', a.get('name',''))} ({a.get('result', a.get('reason',''))})")
md.append(f"\n**0.2 CKA** — `{P02['status']}`. {P02['verdict']}")
if cka_out["ff_56_row"]:
    md.append("\n| vs | CKA(row) | CKA(col) | Δ vs historical |")
    md.append("|---|---|---|---|")
    for other in cka_labels:
        d = cka_deltas.get("ff_56_direct_rung3", {}).get(other)
        md.append(f"| {other} | {cka_out['ff_56_row'][other]:.4f} | "
                  f"{cka_out['ff_56_column'][other]:.4f} | "
                  f"{'—' if d is None else f'{d:+.4f}'} |")
md.append(f"\n**0.3 C3 vs C1** — `{P03['status']}`. {P03.get('verdict','')}")
if P03["status"] == "evaluated":
    md.append(f"\n- C1 5k clean {P03['c1_5k']*100:.2f}% vs C3 {P03['c3_5k']*100:.2f}% "
              f"({P03['delta_5k_pp']:+.2f} pp), McNemar b={P03['mcnemar_5k']['b']} "
              f"c={P03['mcnemar_5k']['c']} p={P03['mcnemar_5k']['p_exact']:.4g} "
              f"({classify_p(P03['mcnemar_5k']['p_exact'])})")
    if P03.get("c3_recovery_fraction") is not None:
        md.append(f"- C3 recovers {P03['c3_recovery_fraction']*100:.1f}% of the recurrence clean "
                  f"gap ({P03['rec_minus_c1_clean_pp']:+.2f} pp) → depth attributed: "
                  f"{P03['depth_attributed']}")
md.append(f"\n**0.4 preservation** — `{P04['status']}`. "
          f"{P04.get('reason','')}{P04.get('archive','')}")

md.append("\n## Phase 1\n")
md.append("### 1.1 T-sweep\n")
if P11["status"] == "evaluated":
    md.append("| T | 5k clean | 1k clean | 1k worst |")
    md.append("|---|---|---|---|")
    for k, v in P11["sweep"].items():
        md.append(f"| {k} | {v['acc_5k']*100:.2f}% | {v['clean_1k']*100:.2f}% | "
                  f"{v['worst_1k']*100:.2f}% |")
    if "vs_ff56_T1" in P11:
        md.append(f"\n- T=1 vs ff_56: {P11['vs_ff56_T1']['delta_5k_pp']:+.2f} pp, "
                  f"McNemar p={P11['vs_ff56_T1']['mcnemar_clean5k']['p_exact']:.4g}")
    md.append(f"- **gate → branch {P11['branch']}**: {P11['rationale']}")
else:
    md.append(f"`{P11['status']}` — {P11.get('rationale','')}")

md.append("\n### 1.2 Oracle halting\n")
md.append(f"`{P12['status']}`")
if P12["status"] == "evaluated":
    md.append("\n| space | T=1 | T=2 | T=3 | T=4 | oracle | headroom | E[views] |")
    md.append("|---|---|---|---|---|---|---|---|")
    for space, res in P12["oracle"].items():
        acc = res["accuracy_by_fixed_T"]
        hr = res["headroom_vs_T4_pp"]
        md.append(f"| {space} | " + " | ".join(f"{acc[f'T={t}']*100:.2f}%" for t in (1, 2, 3, 4))
                  + f" | {res['oracle_all_T']*100:.2f}% | "
                  + (f"{hr:+.2f} pp" if hr is not None else "—")
                  + f" | {res['expected_views']:.2f} |")
    md.append(f"\n- oracle headroom (5k clean) {P12['headroom_clean5k_pp']:+.2f} pp → adaptive "
              f"compute worthwhile: **{P12['adaptive_compute_worthwhile']}**")
    if P12.get("probe_auroc"):
        md.append("- probe AUROC: " + ", ".join(f"{k} {v:.3f}" for k, v in P12["probe_auroc"].items())
                  + f" (bar: entropy {THRESHOLDS['entropy_auc_bar']:.3f})")
    md.append(f"- {P12.get('probe_verdict','')}")

md.append("\n### 1.3 C4\n")
md.append(f"`{P13['status']}` — {P13.get('reason', P13.get('verdict',''))}")
if P13.get("status") == "evaluated":
    md.append(f"\n- C4 5k clean {P13['c4_5k']*100:.2f}%, 1k worst {P13['c4_worst_1k']*100:.2f}%")
    md.append(f"- vs recurrence_only(T=4): clean {P13['mcnemar_vs_rec_T4_clean5k']['delta_pp']:+.2f} pp "
              f"(p={P13['mcnemar_vs_rec_T4_clean5k']['p_exact']:.4g}), worst "
              f"{P13['mcnemar_vs_rec_T4_worst1k']['delta_pp']:+.2f} pp "
              f"(p={P13['mcnemar_vs_rec_T4_worst1k']['p_exact']:.4g})")
    md.append(f"- {P13['verdict']}")

md.append("\n## Parity gates\n")
md.append("| gate | result | detail |")
md.append("|---|---|---|")
for g in GATES:
    _st = "PASS" if g["passed"] is True else ("n/a" if g["passed"] is None else "FAIL")
    md.append(f"| {g['gate']} | {_st} | {g['detail']} |")

md.append("\n## What was not done\n")
if ff56_identity["status"] != "RESOLVED":
    md.append("- ff_56 identity unresolved: no Phase 1 verdict is decisive.")
if P13["status"] != "evaluated":
    md.append(f"- C4 not evaluated in this session ({P13['status']}): {P13.get('reason','')}")
if P04["status"] != "exported":
    md.append("- docs/report export not performed in this environment (see 0.4).")
if P12.get("probe_run") is False:
    md.append(f"- halting probe not trained: {P12.get('probe_verdict','')}")
md.append("- No claim in this report is made from a p-value in [0.05, 0.20]; those are listed as "
          "'not established'.")

(OUT_DIR / "closure_report.md").write_text("\n".join(md) + "\n")
print(f"\n  report: {OUT_DIR / 'closure_report.md'}")

# ── upload everything (timestamped, never overwriting) ───────────────────────
banner("UPLOAD")
for f in sorted(OUT_DIR.glob("*")):
    if f.is_file() and f.suffix in (".json", ".md", ".npz"):
        upload_unique(f, HF_BEST, f"closure/{STAMP}/{f.name}")

print("\n" + "=" * 88)
print("SUMMARY")
print("=" * 88)
print(f"  parity gates      : {'PASS' if parity_ok else 'NOT all passed'}")
print(f"  ff_56 identity    : {ff56_identity['status']}")
print(f"  P0.2 CKA          : {P02['status']}")
print(f"  P0.3 C3 vs C1     : {P03.get('verdict','n/a')}")
print(f"  P1.1 gate         : branch {P11.get('branch')} — {P11.get('rationale','n/a')}")
if P12["status"] == "evaluated":
    print(f"  P1.2 oracle       : headroom {P12['headroom_clean5k_pp']:+.2f} pp "
          f"({'worthwhile' if P12['adaptive_compute_worthwhile'] else 'nothing to gain'})")
print(f"  P1.3 C4           : {P13['status']} — {P13.get('verdict', P13.get('reason',''))}")
print(f"  P0.4 preservation : {P04['status']}")
print(f"  session           : {elapsed_h():.2f} h")
print(f"\n  artifacts -> {HF_BEST}/closure/{STAMP}/")
print(f"  per-arm cache -> {HF_BEST}/{CACHE_PREFIX}/")
print("\nDONE")
