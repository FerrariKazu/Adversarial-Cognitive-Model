#!/usr/bin/env python3
"""
J1 FOUNDATION GATE — Colab T4 experiment harness (Gen-1/Gen-2 foundation).

 Scientific question (the Foundation Gate):
   Does the real Gen-1/Gen-2 foundation training recipe produce a healthy
   learning signal on real ImageNet-100 data, and how does TRADES compare
   with matched clean CE training in backbone versus classifier parameter
   updates?

 Scope: Experiment A (TRADES, `--phase backbone_only`) and Experiment B
 (matched CE control, `--clean-only`). Everything else is held fixed.
 Until this gate PASSES with real T4 results, the Gen-2 experiment campaign
 (K9->K1->K2->K4->K5->K6->K7->K8->K3) must not begin.

 Relationship to the existing harness:
   cloud/gen1/colab_j1_foundation.py dispatches the FULL six-phase ladder
   (production runs, HF durability). This file does NOT duplicate it: it
   runs the single-phase A/B foundation-gate experiment with the
   J1.0-J1.9 structure, required diagnostics, and a structured
   PASS / FAIL / INCONCLUSIVE decision.

 Modes:
   python3 cloud/gen1/colab_j1_foundation_gate.py             # full run (T4)
   python3 cloud/gen1/colab_j1_foundation_gate.py --dry-run   # pre-flight, no training, no writes
   python3 cloud/gen1/colab_j1_foundation_gate.py --selftest  # CPU test of measurement + decision logic

 Conventions followed from the existing harness (`# %%` cells, fail-fast env
 checks, DRY_RUN shielding). Requires the Gen-1 pipeline sources (training/,
 evaluation/, noesis_vision/) — i.e. a carrier branch checkout
 (feature/rhan-next or stage2/nxa-pipeline-refactor), NOT main.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

PHASE = "backbone_only"
DEFAULT_REPO_URL = "https://github.com/FerrariKazu/Adversarial-Cognitive-Model.git"
DEFAULT_REPO_NAME = "Adversarial-Cognitive-Model"


def _repo_root_from_git() -> str:
    """Repo root reported by git (the true root of the Gen-1 pipeline).

    Discovers the true repository root whether running from a local clone,
    a notebook subfolder, or Google Colab /content. If running in a fresh
    Colab environment where the repository has not been cloned yet,
    automatically clones the repository into /content/Adversarial-Cognitive-Model.
    """
    # 1. Explicit environment variable if provided
    env_root = os.environ.get("REPO_ROOT") or os.environ.get("PROJECT_ROOT")
    if env_root and os.path.isdir(env_root):
        return os.path.abspath(env_root)

    # 2. Determine base directory safely without NameError in notebook cells
    base_dir: Optional[str] = None
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = None

    probe_dirs: List[str] = []
    if base_dir:
        probe_dirs.append(base_dir)
    cwd = os.getcwd()
    if cwd not in probe_dirs:
        probe_dirs.append(cwd)

    # 3. Probe git rev-parse from base_dir and cwd ancestors
    for d in probe_dirs:
        try:
            import subprocess
            out = subprocess.run(
                ["git", "-C", d, "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, check=True,
            )
            root = out.stdout.strip()
            if root and os.path.isdir(root):
                return root
        except Exception:
            pass

    # 4. Upward directory traversal from probe_dirs looking for .git or training/
    for start in probe_dirs:
        candidate = os.path.abspath(start)
        for _ in range(6):
            if os.path.isdir(os.path.join(candidate, ".git")) or os.path.isdir(os.path.join(candidate, "training")):
                return candidate
            parent = os.path.dirname(candidate)
            if parent == candidate:
                break
            candidate = parent

    # 5. Colab/notebook fallback: check subdirectories of cwd or /content
    search_dirs = [cwd]
    if os.path.isdir("/content") and "/content" not in search_dirs:
        search_dirs.append("/content")

    # Priority 5a: look for known repository name with .git
    for parent in search_dirs:
        for name in (DEFAULT_REPO_NAME, "adversarial-cognitive-model", "Adversarial Cognitive Model"):
            cand = os.path.join(parent, name)
            if os.path.isdir(cand) and (os.path.isdir(os.path.join(cand, ".git")) or os.path.isdir(os.path.join(cand, "training"))):
                return os.path.abspath(cand)

    # Priority 5b: check ANY subdirectory containing .git
    for parent in search_dirs:
        try:
            for item in os.listdir(parent):
                candidate = os.path.join(parent, item)
                if os.path.isdir(candidate) and (os.path.isdir(os.path.join(candidate, ".git")) or os.path.isdir(os.path.join(candidate, "training"))):
                    return os.path.abspath(candidate)
        except Exception:
            pass

    # 6. Fresh Colab runtime automatic clone
    if os.path.isdir("/content"):
        target_clone = os.path.join("/content", DEFAULT_REPO_NAME)
        # Clone if missing OR if exists but has no .git (incomplete previous clone)
        needs_clone = not os.path.isdir(target_clone) or not os.path.isdir(os.path.join(target_clone, ".git"))
        if needs_clone:
            repo_url = os.environ.get("RHAN_REPO_URL") or DEFAULT_REPO_URL
            print(f"  == Fresh Colab runtime: cloning {repo_url} -> {target_clone} ==")
            gh_token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
            if not gh_token:
                try:
                    from google.colab import userdata  # type: ignore[import-untyped]
                    gh_token = userdata.get("GH_TOKEN") or userdata.get("GITHUB_TOKEN") or userdata.get("HF_TOKEN")
                except Exception:
                    pass
            clone_target_url = repo_url
            if gh_token and "github.com" in repo_url and "@" not in repo_url:
                clone_target_url = repo_url.replace("https://", f"https://{gh_token}@")
            try:
                import subprocess
                # Remove incomplete clone directory if it exists
                if os.path.isdir(target_clone) and not os.path.isdir(os.path.join(target_clone, ".git")):
                    import shutil as _shutil
                    _shutil.rmtree(target_clone, ignore_errors=True)
                subprocess.run(["git", "clone", clone_target_url, target_clone], check=True)
                return target_clone
            except Exception as e:
                print(f"  ✗ Failed to clone {repo_url}: {e}")
                print("    notice: for private repositories, provide GH_TOKEN in Colab secrets or environment")
        elif os.path.isdir(os.path.join(target_clone, ".git")):
            # Clone exists and is valid — use it
            return target_clone

    return base_dir or cwd


REPO_ROOT = _repo_root_from_git()
if os.path.isdir(REPO_ROOT):
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    # Ensure working directory is REPO_ROOT in notebook / Colab environments
    if os.getcwd() != REPO_ROOT and ("/content" in os.getcwd() or "cloud" in os.getcwd()):
        try:
            os.chdir(REPO_ROOT)
        except Exception:
            pass

REPO_REL_TRAINER = os.path.join("training", "train_generation1_foundation.py")
REQUIRED_PIPELINE_PATHS = [
    os.path.join("training", "train_generation1_foundation.py"),
    os.path.join("training", "adv_curriculum.py"),
    os.path.join("training", "stage_state_machine.py"),
    os.path.join("evaluation", "clean_and_robust.py"),
    os.path.join("evaluation", "imagenet100_loader.py"),
]
REQUIRED_EXTRA_PATHS = [
    os.path.join("evaluation", "compactness_report.py"),
    os.path.join("noesis_vision", "models", "backbone.py"),
    os.path.join("scripts", "prepare_imagenet100.py"),
]
ALL_REQUIRED_PATHS = REQUIRED_PIPELINE_PATHS + REQUIRED_EXTRA_PATHS


def verify_pipeline_imports(repo_root: str) -> Tuple[bool, Optional[str]]:
    """Verify that the trainer and required pipeline modules can be imported."""
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        import training.train_generation1_foundation  # noqa: F401
        import training.adv_curriculum  # noqa: F401
        import training.stage_state_machine  # noqa: F401
        import evaluation.clean_and_robust  # noqa: F401
        import evaluation.imagenet100_loader  # noqa: F401
        return True, None
    except Exception as e:
        return False, str(e)


def _ensure_git_repo(repo_root: str, repo_url: Optional[str] = None) -> str:
    """Ensure repo_root is inside a valid git repository.

    If repo_root has no .git directory, clone the repository there.
    Returns the validated repo_root (may change if cloned to a subdirectory).
    """
    if os.path.isdir(os.path.join(repo_root, ".git")):
        return repo_root

    # Check if we're inside a git worktree (repo_root is a subdirectory)
    try:
        out = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        )
        real_root = out.stdout.strip()
        if real_root and os.path.isdir(os.path.join(real_root, ".git")):
            return real_root
    except Exception:
        pass

    # No git repo at all — clone into repo_root or a subdirectory
    clone_url = repo_url or os.environ.get("RHAN_REPO_URL") or DEFAULT_REPO_URL
    target_clone = repo_root if not os.listdir(repo_root) else os.path.join(repo_root, DEFAULT_REPO_NAME)

    # Skip if target already has a valid .git
    if os.path.isdir(os.path.join(target_clone, ".git")):
        return target_clone

    print(f"  == No git repository at {repo_root} — cloning {clone_url} -> {target_clone} ==")
    gh_token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not gh_token:
        try:
            from google.colab import userdata  # type: ignore[import-untyped]
            gh_token = userdata.get("GH_TOKEN") or userdata.get("GITHUB_TOKEN")
        except Exception:
            pass
    actual_url = clone_url
    if gh_token and "github.com" in clone_url and "@" not in clone_url:
        actual_url = clone_url.replace("https://", f"https://{gh_token}@")
    try:
        subprocess.run(["git", "clone", actual_url, target_clone], check=True,
                       capture_output=True, text=True)
        print(f"  ✓ cloned successfully to {target_clone}")
        return target_clone
    except subprocess.CalledProcessError as e:
        print(f"  ✗ clone failed: {e.stderr.strip() if e.stderr else e}")
        print("    notice: for private repositories, provide GH_TOKEN in Colab secrets or environment")
    return repo_root


def bootstrap_sources(
    repo_root: str,
    source_ref: Optional[str] = None,
    repo_url: Optional[str] = None,
    source_mode: str = "auto",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Safely obtain missing Gen-1 pipeline source files without destructive Git ops.

    Never switches the user's branch, never runs hard-reset or auto-merge.
    Restores only the needed subtrees (training, evaluation, noesis_vision, scripts)
    from a verified carrier or authoritative ref if missing in working directory.

    If repo_root is not inside a git repository, automatically clones the repo first.
    """
    # Ensure we're inside a git repo before attempting git operations
    effective_root = _ensure_git_repo(repo_root, repo_url=repo_url)
    if effective_root != repo_root:
        # Update global REPO_ROOT if the clone changed the effective root
        global REPO_ROOT
        REPO_ROOT = effective_root
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        try:
            os.chdir(REPO_ROOT)
        except Exception:
            pass

    missing = [p for p in ALL_REQUIRED_PATHS if not os.path.exists(os.path.join(effective_root, p))]
    res: Dict[str, Any] = {
        "missing_before": missing,
        "restored": False,
        "ref_used": None,
        "effective_root": effective_root,
        "errors": [],
    }
    if not missing:
        res["status"] = "already_present"
        return res

    if dry_run:
        res["status"] = "dry_run_missing"
        return res

    # Verify we have a .git directory to work with
    if not os.path.isdir(os.path.join(effective_root, ".git")):
        res["status"] = "no_git_repo"
        res["errors"].append(f"no .git directory in {effective_root}; clone may have failed")
        return res

    if source_ref:
        candidate_refs = [source_ref]
    else:
        candidate_refs = [
            "HEAD",
            "main",
            "origin/main",
            "stage2/nxa-pipeline-refactor",
            "origin/stage2/nxa-pipeline-refactor",
            "feature/rhan-next",
            "origin/feature/rhan-next",
            "legacy",
            "origin/legacy",
        ]

    matched_ref = None
    for ref in candidate_refs:
        try:
            check = subprocess.run(
                ["git", "cat-file", "-e", f"{ref}:training/train_generation1_foundation.py"],
                cwd=effective_root, capture_output=True, text=True,
            )
            if check.returncode == 0:
                matched_ref = ref
                break
        except Exception:
            pass

    if not matched_ref and source_mode in ("auto", "fetch"):
        # Determine the actual remote name or URL
        target_remote = "origin"
        if repo_url:
            # If a full URL was provided, use it directly as the remote
            target_remote = repo_url
        fetch_branches = [source_ref] if source_ref else ["main", "stage2/nxa-pipeline-refactor", "feature/rhan-next"]
        for b in fetch_branches:
            try:
                # Fetch to FETCH_HEAD — avoids "refusing to fetch into
                # currently checked out branch" errors
                fetch_run = subprocess.run(
                    ["git", "fetch", target_remote, b],
                    cwd=effective_root, capture_output=True, text=True,
                )
                if fetch_run.returncode == 0:
                    # After plain fetch, the ref is available at FETCH_HEAD
                    # and also updated in remote tracking refs (origin/<b>)
                    for check_ref in ["FETCH_HEAD", f"origin/{b}", b]:
                        check = subprocess.run(
                            ["git", "cat-file", "-e", f"{check_ref}:training/train_generation1_foundation.py"],
                            cwd=effective_root, capture_output=True, text=True,
                        )
                        if check.returncode == 0:
                            matched_ref = check_ref
                            break
                    if matched_ref:
                        break
            except Exception as e:
                res["errors"].append(f"fetch {b} failed: {e}")

    if not matched_ref:
        res["status"] = "ref_not_found"
        return res

    subtrees = ["training", "evaluation", "noesis_vision", "scripts"]
    try:
        cmd = ["git", "checkout", matched_ref, "--"] + subtrees
        cp = subprocess.run(cmd, cwd=effective_root, capture_output=True, text=True)
        if cp.returncode == 0:
            res["restored"] = True
            res["ref_used"] = matched_ref
            res["status"] = "success"
        else:
            res["status"] = "checkout_failed"
            res["errors"].append(cp.stderr.strip() or cp.stdout.strip())
    except Exception as e:
        res["status"] = "error"
        res["errors"].append(str(e))

    res["missing_after"] = [p for p in ALL_REQUIRED_PATHS if not os.path.exists(os.path.join(effective_root, p))]
    return res


# Canonical commands (spec). The ONLY intended difference between arms is
# --clean-only. Do not alter the training objective for convenience.
def canonical_command(data_root: str, clean_only: bool) -> List[str]:
    cmd = [
        "python3", REPO_REL_TRAINER,
        "--phase", PHASE,
        "--data-root", data_root,
        "--batch-size", "64",
        "--num-workers", "0",
        "--device", "cuda",
        "--no-hf",
        "--force-fresh",
    ]
    if clean_only:
        cmd.append("--clean-only")
    return cmd


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def git_json(cwd: str) -> Dict[str, Any]:
    def _run(args: List[str]) -> str:
        try:
            return subprocess.run(["git"] + args, cwd=cwd, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception as e:  # recorded, never faked
            return f"ERROR: {e}"
    return {
        "commit": _run(["rev-parse", "HEAD"]),
        "branch": _run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "status_porcelain": _run(["status", "--porcelain"]),
        "remotes": _run(["remote", "-v"]),
    }


# ── Measurement utilities (generic; unit-tested by --selftest) ───────────────
# The runner already provides: check_gradient_reach (standing gate),
# backbone/classifier optimizer groups, per-phase manifests and eval CSVs.
# The harness adds the per-group NUMBERS the runner does not print.

def group_parameters(model) -> Dict[str, List[Any]]:
    """backbone / classifier split taken from the model's own structure
    (FoundationModel.backbone / FoundationModel.cls_head — the same split the
    trainer's optimizer groups use)."""
    import torch.nn as nn
    if hasattr(model, "backbone") and hasattr(model, "cls_head"):
        return {"backbone": [p for p in model.backbone.parameters()],
                "classifier": [p for p in model.cls_head.parameters()]}
    # generic fallback for selftest/toy models: every parameter in one group
    return {"all": [p for p in model.parameters()]}


def gradient_norms(model, x, y, loss_fn=None) -> Dict[str, float]:
    """CE (or provided) loss; L2 norm of gradients per parameter group."""
    import torch
    import torch.nn.functional as F
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y) if loss_fn else F.cross_entropy(model(x), y)
    loss.backward()
    out = {}
    for name, params in group_parameters(model).items():
        sq = 0.0
        for p in params:
            if p.grad is not None:
                sq += float(p.grad.detach().double().pow(2).sum())
        out[name] = sq ** 0.5
    model.zero_grad(set_to_none=True)
    return {"loss": float(loss.detach()), **{f"{k}_grad_norm": v for k, v in out.items()}}


def group_weight_delta(model_a, model_b) -> Dict[str, float]:
    """|dW| = L2 distance of weights between two states, per group."""
    ga, gb = group_parameters(model_a), group_parameters(model_b)
    out = {}
    for name in ga:
        sq = 0.0
        for pa, pb in zip(ga[name], gb[name]):
            sq += float((pa.detach().double() - pb.detach().double()).pow(2).sum())
        out[f"{name}_dw"] = sq ** 0.5
    return out


def capture_features(model, x, tap: str = "backbone") -> Any:
    """Forward-pass feature tensor at the model's backbone output."""
    import torch
    module = getattr(model, tap)
    holder: Dict[str, Any] = {}

    def _hook(_m, _inp, outp):
        holder["y"] = outp.detach()

    h = module.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model(x)
    finally:
        h.remove()
    return holder.get("y")


def feature_drift(model, x, baseline_features) -> Dict[str, float]:
    feats = capture_features(model, x)
    if feats is None or baseline_features is None:
        return {"feature_drift": None}
    import torch
    a = feats.double()
    b = baseline_features.double()
    if a.shape != b.shape:
        return {"feature_drift": None, "note": f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"}
    return {"feature_drift": float((a - b).norm())}


EPOCH_RE = re.compile(r"epoch\s+(\d+)/(\d+)\s+loss=([0-9.]+)\s+val_acc=([0-9.]+)")


def parse_epoch_log(text: str) -> List[Dict[str, Any]]:
    return [{"epoch": int(m.group(1)), "total": int(m.group(2)),
             "loss": float(m.group(3)), "val_acc": float(m.group(4))}
            for m in EPOCH_RE.finditer(text)]


# ═══════════════════════════════════════════════════════════════════════════
# J1.0 — Environment verification
# ═══════════════════════════════════════════════════════════════════════════
def j10_environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {"timestamp_utc": utc_now(),
                           "python": sys.version.split()[0]}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_compiled"] = torch.version.cuda
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["gpu_count"] = torch.cuda.device_count()
            env["capability"] = torch.cuda.get_device_capability(0)
    except Exception as e:
        env["torch"] = f"ERROR: {e}"
        env["cuda_available"] = False
    env["gpu_ok"] = bool(env.get("cuda_available"))
    print(f"  GPU: {env.get('gpu', 'NONE')} | torch {env.get('torch')} "
          f"| CUDA {env.get('cuda_compiled')} | python {env['python']}")
    return env


# ═══════════════════════════════════════════════════════════════════════════
# J1.1 — Repository verification & source bootstrap
# ═══════════════════════════════════════════════════════════════════════════
def j11_repository(
    cwd: Optional[str] = None,
    source_ref: Optional[str] = None,
    repo_url: Optional[str] = None,
    source_mode: str = "auto",
    bootstrap: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Repository verification and non-destructive source bootstrap.

    The Gen-1 pipeline sources live under the repository root: training/,
    evaluation/, noesis_vision/, scripts/, tests/.
    If pipeline sources are missing, automatically and non-destructively
    retrieves them from the authoritative or carrier ref without branch
    switching or force resets.
    """
    if cwd is not None:
        check_root = cwd
    elif "REPO_ROOT" in globals() and REPO_ROOT and os.path.isdir(REPO_ROOT):
        # Accept REPO_ROOT even without training/ — bootstrap will create it
        check_root = REPO_ROOT
    else:
        repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
        if os.path.isdir(os.path.join(repo_root, ".git")) or os.path.isdir(os.path.join(repo_root, "training")):
            check_root = repo_root
        else:
            check_root = os.getcwd()

    info = git_json(check_root)
    missing_before = [p for p in ALL_REQUIRED_PATHS if not os.path.exists(os.path.join(check_root, p))]

    if missing_before:
        if dry_run:
            print(f"  [DRY-RUN] pipeline sources missing: {missing_before}")
            print(f"  [DRY-RUN] would bootstrap from ref: {source_ref or 'auto'}")
            info["bootstrap_status"] = "dry_run_missing"
        elif bootstrap:
            print(f"  == Bootstrapping missing Gen-1 pipeline sources in {check_root} ==")
            boot_res = bootstrap_sources(
                check_root,
                source_ref=source_ref,
                repo_url=repo_url,
                source_mode=source_mode,
                dry_run=False,
            )
            info["bootstrap_result"] = boot_res
            if boot_res.get("restored"):
                print(f"  ✓ Gen-1 pipeline sources restored from ref '{boot_res.get('ref_used')}'")
                # Update check_root if bootstrap changed the effective root
                if boot_res.get("effective_root") and boot_res["effective_root"] != check_root:
                    check_root = boot_res["effective_root"]
            else:
                print(f"  ✗ Gen-1 pipeline source bootstrap failed: {boot_res.get('status')} ({boot_res.get('errors')})")
                # Update check_root if bootstrap changed the effective root
                if boot_res.get("effective_root") and boot_res["effective_root"] != check_root:
                    check_root = boot_res["effective_root"]
        else:
            info["bootstrap_status"] = "disabled"

    info["missing_pipeline_sources"] = [p for p in REQUIRED_PIPELINE_PATHS
                                        if not os.path.exists(os.path.join(check_root, p))]
    info["missing_extra_sources"] = [p for p in REQUIRED_EXTRA_PATHS
                                     if not os.path.exists(os.path.join(check_root, p))]
    info["pipeline_sources_ok"] = not info["missing_pipeline_sources"] and not info["missing_extra_sources"]

    if not info["pipeline_sources_ok"]:
        print("  ✗ required Gen-1 sources unavailable — checkout a carrier branch "
              "(feature/rhan-next or stage2/nxa-pipeline-refactor) or specify RHAN_SOURCE_REF. "
              f"Missing: {info['missing_pipeline_sources'] + info['missing_extra_sources']}")
        info["import_ok"] = False
        info["repo_ok"] = False
    else:
        # Verify trainer and imports
        imports_ok, import_err = verify_pipeline_imports(check_root)
        info["import_ok"] = imports_ok
        info["import_error"] = import_err
        info["repo_ok"] = bool(info["pipeline_sources_ok"] and info["import_ok"])
        c = info["commit"][:12] if not str(info.get("commit", "")).startswith("ERROR") else "non-git"
        b = info["branch"] if not str(info.get("branch", "")).startswith("ERROR") else "local"
        if imports_ok:
            print(f"  ✓ repo {c} on {b}; pipeline sources verified & imports OK")
        else:
            print(f"  ✗ repo {c} on {b}; imports failed: {import_err}")
    return info


# ═══════════════════════════════════════════════════════════════════════════
# J1.2 — Dataset verification & bootstrap
# ═══════════════════════════════════════════════════════════════════════════
def _resolve_data_root(data_root: Optional[str]) -> str:
    """Resolve the data root to the canonical repo-root data ImageNet-100."""
    if data_root is not None:
        return os.path.abspath(data_root)
    env_root = os.environ.get("J1_DATA_ROOT") or os.environ.get("IMAGENET100_ROOT")
    if env_root:
        return os.path.abspath(env_root)
    # Check known persistent or Colab cache candidates
    candidates = [
        os.path.join(REPO_ROOT, "data", "imagenet100"),
        "/content/drive/MyDrive/imagenet100",
        "/content/imagenet100",
        "/content/data/imagenet100",
    ]
    for cand in candidates:
        if os.path.isdir(cand) and os.path.isdir(os.path.join(cand, "val")):
            return cand
    return os.path.join(REPO_ROOT, "data", "imagenet100")


def _inspect_dataset(root: str) -> Dict[str, Any]:
    """Inspect dataset directory for structural ImageNet-100 compliance."""
    info: Dict[str, Any] = {
        "data_root": root,
        "exists": os.path.isdir(root),
        "valid": False,
        "dataset_source": "clane9/imagenet-100",
        "dataset_revision": "0519dc2f402a3a18c6e57f7913db059215eee25b",
    }
    if not info["exists"]:
        return info

    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        return info

    try:
        from evaluation.imagenet100_loader import validate_imagenet100_root
        n_train = validate_imagenet100_root(root, split="train")
        n_val = validate_imagenet100_root(root, split="val")
        info["train_classes"] = n_train
        info["val_classes"] = n_val
        info["class_count"] = n_val
        if n_train == 100 and n_val == 100:
            info["valid"] = True
    except Exception as e:
        info["validator_error"] = str(e)
        return info

    # Sample counts
    try:
        train_samples = sum(len(files) for _, _, files in os.walk(train_dir) if files)
        val_samples = sum(len(files) for _, _, files in os.walk(val_dir) if files)
        info["train_samples"] = train_samples
        info["val_samples"] = val_samples
    except Exception:
        pass

    fp_path = os.path.join(root, "fingerprint.json")
    if os.path.exists(fp_path):
        try:
            with open(fp_path) as f:
                info["fingerprint"] = json.load(f)
        except Exception:
            pass
    return info


def j12_dataset(
    data_root: Optional[str] = None,
    bootstrap: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Dataset verification at the canonical repo-root-implicit path.

    Restores the canonical clane9/imagenet-100 dataset automatically when
    missing via scripts/prepare_imagenet100.py.
    """
    resolved = _resolve_data_root(data_root)
    info = _inspect_dataset(resolved)

    if info["valid"]:
        print(f"  ✓ dataset pre-flight OK: {resolved} (100 classes in both splits; "
              f"{info.get('train_samples', '?')} train / {info.get('val_samples', '?')} val samples)")
        return info

    # If dataset not valid:
    if dry_run:
        print(f"  [DRY-RUN] canonical ImageNet-100 missing/incomplete: {resolved} — would bootstrap via scripts/prepare_imagenet100.py")
        info["valid"] = False
        return info

    if bootstrap:
        converter = os.path.join(REPO_ROOT, "scripts", "prepare_imagenet100.py")
        if not os.path.exists(converter):
            print(f"  ✗ canonical ImageNet-100 unavailable: converter script missing at {converter}")
            info["valid"] = False
            return info

        print(f"  == ImageNet-100 bootstrap: restoring canonical clane9/imagenet-100 -> {resolved} ==")
        cmd = [sys.executable, converter, "--root", resolved]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=REPO_ROOT
            )
            for line in proc.stdout:  # type: ignore[union-attr]
                if any(k in line for k in ("source:", "rows ->", "train:", "val:", "verify", "error", "STOP")):
                    print(f"    {line.rstrip()}")
            rc = proc.wait()
            if rc == 0:
                info = _inspect_dataset(resolved)
                if info["valid"]:
                    print(f"  ✓ ImageNet-100 bootstrap complete & verified: 100 classes in both splits")
                else:
                    print(f"  ✗ ImageNet-100 bootstrap completed but structural validation failed ({resolved})")
            else:
                print(f"  ✗ canonical ImageNet-100 bootstrap failed (exit {rc})")
                tok = os.environ.get("HF_TOKEN")
                if not tok:
                    print("    notice: if access requires authentication, set HF_TOKEN in environment or Colab secrets")
                info["valid"] = False
        except Exception as e:
            print(f"  ✗ canonical ImageNet-100 bootstrap error: {e}")
            info["valid"] = False
    else:
        print(f"  ✗ dataset root missing or incomplete: {resolved} (bootstrap disabled)")
        info["valid"] = False

    return info


# ═══════════════════════════════════════════════════════════════════════════
# J1.3 — Foundation configuration (both arms, matched)
# ═══════════════════════════════════════════════════════════════════════════
def j13_configuration(data_root: str) -> Dict[str, Any]:
    return {
        "phase": PHASE,
        "data_root": data_root,
        "arms": {
            "trades": canonical_command(data_root, clean_only=False),
            "ce": canonical_command(data_root, clean_only=True),
        },
        "matched_fields": ["dataset", "phase", "model", "batch_size=64",
                           "num_workers=0", "optimizer", "lr", "epochs",
                           "augmentation", "initialization (seeded identically)",
                           "evaluation", "instrumentation", "device"],
        "intended_difference": "objective only: TRADES (w_trades * (CE + beta*KL(adv||clean))) vs clean CE (--clean-only)",
        "documented_unavoidable_differences": [
            "runs execute sequentially on one T4; the second run would overwrite "
            "checkpoints/foundation_backbone_only_{best,rolling}.pth, so each arm is "
            "harvested into its own subdirectory immediately after it finishes",
            "the local ladder roadmap + phase manifest are snapshot before the arms and "
            "restored after harvest: J1 must not advance canonical ladder protocol state",
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# J1.4 — Gradient-reach preflight (+ init snapshot for |dW| / feature drift)
# ═══════════════════════════════════════════════════════════════════════════
def j14_preflight(out_dir: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"timestamp_utc": utc_now()}
    sys.path.insert(0, REPO_ROOT)
    from training.train_generation1_foundation import (  # carrier sources
        FoundationConfig, build_model, seed_everything, check_gradient_reach)
    from evaluation.imagenet100_loader import IMAGENET100_IMG_SIZE, IMAGENET100_NUM_CLASSES
    import torch

    cfg = FoundationConfig()
    result["seed"] = int(cfg.seed)
    result["img_size"] = int(IMAGENET100_IMG_SIZE)

    # Replicate the runner's init exactly: seed_everything(cfg.seed) then
    # build_model(cfg, phase) — the same order the trainer uses (line: seed
    # before build), same process, same torch.
    seed_everything(int(cfg.seed))
    model = build_model(cfg, PHASE)
    model.eval()
    result["param_count"] = sum(p.numel() for p in model.parameters())

    diag = os.path.join(out_dir, "diagnostics")
    os.makedirs(diag, exist_ok=True)
    init_path = os.path.join(diag, "init_state_dict.pt")
    torch.save(model.state_dict(), init_path)
    result["init_state_dict"] = {"path": init_path, "sha256": sha256_file(init_path)}

    # Fixed probe batch — identical inputs for init and trained comparisons.
    g = torch.Generator().manual_seed(int(cfg.seed) + 104)
    x = torch.rand(8, 3, int(IMAGENET100_IMG_SIZE), int(IMAGENET100_IMG_SIZE),
                   generator=g)
    y = torch.randint(0, int(IMAGENET100_NUM_CLASSES), (8,), generator=g)
    torch.save({"x": x, "y": y}, os.path.join(diag, "probe_batch.pt"))

    # The runner's own standing gate (raises -> STOP).
    was_training = model.training
    model.train()
    check_gradient_reach(model, x, y)
    model.train(was_training)
    result["gradient_reach_check"] = "passed (runner's check_gradient_reach)"

    result["init_measurements"] = gradient_norms(model, x, y)
    feats = capture_features(model, x, tap="backbone")
    if feats is not None:
        torch.save(feats, os.path.join(diag, "init_features.pt"))
        result["init_features_shape"] = list(feats.shape)
    write_json(os.path.join(diag, "preflight.json"), result)
    print(f"  ✓ preflight ok: {result['param_count']:,} params, "
          f"backbone grad norm {result['init_measurements'].get('backbone_grad_norm'):.4g}, "
          f"classifier grad norm {result['init_measurements'].get('classifier_grad_norm'):.4g}")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# J1.5 / J1.6 — run one arm and harvest everything it produces
# ═══════════════════════════════════════════════════════════════════════════
HARVEST = [
    ("result",   "report/foundation_backbone_only_result.json"),
    ("compact",  "report/foundation_backbone_only_compactness.json"),
    ("manifest", "runs/foundation_backbone_only/manifest.json"),
    ("ckpt_best", "checkpoints/foundation_backbone_only_best.pth"),
    ("ckpt_rolling", "checkpoints/foundation_backbone_only_rolling.pth"),
]
HARVEST_DIRS = [
    ("eval", "report/foundation_backbone_only_eval"),
]


def run_arm(arm: str, cmd: List[str], out_dir: str, logs: str) -> Dict[str, Any]:
    """Train one arm, tee the log, harvest artifacts into out_dir/<arm>/."""
    summary: Dict[str, Any] = {"arm": arm, "command": cmd,
                               "started_utc": utc_now(), "returncode": None,
                               "artifacts": {}, "missing_artifacts": []}
    arm_dir = os.path.join(out_dir, arm)
    os.makedirs(arm_dir, exist_ok=True)
    log_path = os.path.join(logs, f"{arm}.log")
    os.makedirs(logs, exist_ok=True)

    t0 = time.time()
    with open(log_path, "w") as logf:
        summary["command_str"] = " ".join(cmd)
        print(f"  [{arm}] running: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, cwd=REPO_ROOT)
        for line in proc.stdout:  # type: ignore[union-attr]
            logf.write(line)
            if re.search(r"epoch \d+/\d+|DONE|FAILURE|Error|Traceback", line):
                print(f"    {line.rstrip()}")
        summary["returncode"] = proc.wait()
    summary["runtime_sec"] = round(time.time() - t0, 1)
    summary["finished_utc"] = utc_now()

    with open(log_path) as f:
        epochs = parse_epoch_log(f.read())
    summary["epochs_parsed"] = len(epochs)
    if epochs:
        summary["best_epoch_row"] = max(epochs, key=lambda r: r["val_acc"])
        summary["final_epoch_row"] = epochs[-1]
        summary["training_loss_final"] = epochs[-1]["loss"]

    # harvest (copy — originals get overwritten by the second arm)
    for tag, rel in HARVEST:
        src = os.path.join(REPO_ROOT, rel) if not os.path.isabs(rel) else rel
        if os.path.exists(src):
            dst = os.path.join(arm_dir, os.path.basename(rel))
            shutil.copy2(src, dst)
            summary["artifacts"][tag] = dst
        else:
            summary["missing_artifacts"].append(rel)
    for tag, rel in HARVEST_DIRS:
        src = os.path.join(REPO_ROOT, rel) if not os.path.isabs(rel) else rel
        if os.path.isdir(src):
            dst = os.path.join(arm_dir, os.path.basename(rel))
            shutil.copytree(src, dst, dirs_exist_ok=True)
            summary["artifacts"][tag] = dst
        else:
            summary["missing_artifacts"].append(rel)

    # pull the phase result numbers if present
    res = summary["artifacts"].get("result")
    if res:
        try:
            with open(res) as f:
                r = json.load(f)
            summary["best_val_acc"] = r.get("best_val_acc")
            summary["gaze_scheme"] = r.get("gaze_scheme")
        except Exception as e:
            summary["result_read_error"] = str(e)
    summary["checkpoint_eval_valid"] = bool(summary["artifacts"].get("ckpt_best")
                                            and summary["artifacts"].get("eval")
                                            and summary.get("best_val_acc") is not None)
    write_json(os.path.join(arm_dir, f"{arm}_summary.json"), summary)
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# J1.7 — Diagnostics: |dW|, gradient norms, feature drift, per arm
# ═══════════════════════════════════════════════════════════════════════════
def _load_trained_model(ckpt_path: str):
    import torch
    from training.train_generation1_foundation import FoundationConfig, build_model, seed_everything
    cfg = FoundationConfig()
    seed_everything(int(cfg.seed))          # deterministic build
    model = build_model(cfg, PHASE)
    payload = torch.load(ckpt_path, map_location="cpu")
    sd = payload.get("model", payload) if isinstance(payload, dict) else payload
    sd = {(k[len("module."):] if k.startswith("module.") else k): v
          for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def _load_init_model(out_dir: str):
    import torch
    from training.train_generation1_foundation import FoundationConfig, build_model, seed_everything
    cfg = FoundationConfig()
    seed_everything(int(cfg.seed))
    model = build_model(cfg, PHASE)
    sd = torch.load(os.path.join(out_dir, "diagnostics", "init_state_dict.pt"),
                    map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def j17_diagnostics(out_dir: str, arms: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    import torch
    diag_dir = os.path.join(out_dir, "diagnostics")
    probe = torch.load(os.path.join(diag_dir, "probe_batch.pt"))
    x, y = probe["x"], probe["y"]
    init_feats = torch.load(os.path.join(diag_dir, "init_features.pt"))
    init_model = _load_init_model(out_dir)
    out: Dict[str, Any] = {}
    for arm, summary in arms.items():
        ckpt = summary.get("artifacts", {}).get("ckpt_best")
        entry: Dict[str, Any] = {"arm": arm}
        if not ckpt or not os.path.exists(ckpt):
            entry["error"] = "no harvested best checkpoint — diagnostics impossible"
            out[arm] = entry
            write_json(os.path.join(diag_dir, f"{arm}_diagnostics.json"), entry)
            continue
        trained = _load_trained_model(ckpt)
        entry.update(group_weight_delta(trained, init_model))      # |dW|
        entry.update(gradient_norms(trained, x, y))                # grad norms on probe batch
        entry.update(feature_drift(trained, x, init_feats))        # feature drift vs init
        entry["ckpt"] = ckpt
        out[arm] = entry
        write_json(os.path.join(diag_dir, f"{arm}_diagnostics.json"), entry)
        print(f"  [{arm}] backbone |dW|={entry.get('backbone_dw'):.4g} "
              f"classifier |dW|={entry.get('classifier_dw'):.4g} "
              f"feature_drift={entry.get('feature_drift')} "
              f"backbone_grad={entry.get('backbone_grad_norm'):.4g}")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# J1.8 — Foundation-gate decision (evidence-based; no invented thresholds)
# ═══════════════════════════════════════════════════════════════════════════
HARD_ITEMS = [
    "trades_completed", "ce_completed",
    "ckpt_eval_valid_trades", "ckpt_eval_valid_ce",
    "backbone_grads_trades", "backbone_grads_ce",
    "backbone_moved_trades", "backbone_moved_ce",
    "classifier_grads_trades", "classifier_grads_ce",
    "feature_drift_trades", "feature_drift_ce",
]


def decide(items: Dict[str, Dict[str, Any]], arms_identical: bool = False) -> Dict[str, Any]:
    """PASS / FAIL / INCONCLUSIVE from measured facts only.

    FAIL          — a required fact was MEASURED and is false (training broke,
                    no gradients, parameters did not move, artifacts invalid).
    INCONCLUSIVE  — a required fact is missing/unmeasured, or the two arms are
                    numerically indistinguishable (experiment validity doubt).
    PASS          — every required fact measured true and arms differ.
    """
    reasons: List[str] = []
    false_items = [k for k in HARD_ITEMS
                   if k in items and items[k]["value"] is False]
    missing = [k for k in HARD_ITEMS
               if k not in items or items[k]["value"] is None]
    if false_items:
        decision = "FAIL"
        reasons = [f"measured false: {k} ({items[k].get('evidence', '')})"
                   for k in false_items]
    elif missing:
        decision = "INCONCLUSIVE"
        reasons = [f"not measured: {k}" for k in missing]
    elif arms_identical:
        decision = "INCONCLUSIVE"
        reasons = ["TRADES and CE arms are numerically indistinguishable — "
                   "arm validity in question (clean-only may not have taken effect)"]
    elif all(items.get(k, {}).get("value") is True for k in HARD_ITEMS):
        decision = "PASS"
        reasons = ["all required facts measured true; arms distinguishable"]
    else:
        decision = "INCONCLUSIVE"
        reasons = ["required facts incomplete"]
    return {"decision": decision, "reasons": reasons,
            "thresholds_used": "none — evidence-based per spec "
                               "(no arbitrary numeric thresholds exist in the "
                               "research specification for this gate)",
            "arms_indistinguishable": arms_identical,
            "checklist": items}


def build_checklist(env, repo, dataset, arms, diag) -> Dict[str, Dict[str, Any]]:
    def item(value, evidence):
        return {"value": value, "evidence": evidence}
    ck = {}
    ck["env_ok"] = item(bool(env.get("gpu_ok")),
                        f"cuda_available={env.get('cuda_available')} gpu={env.get('gpu')}")
    ck["repo_ok"] = item(bool(repo.get("pipeline_sources_ok")),
                         f"commit={str(repo.get('commit'))[:12]} missing={repo.get('missing_pipeline_sources')}")
    ck["dataset_ok"] = item(bool(dataset.get("valid")),
                            f"root={dataset.get('data_root')} valid={dataset.get('valid')}")
    for arm in ("trades", "ce"):
        s = arms.get(arm) or {}
        ck[f"{arm}_completed"] = item(
            (s.get("returncode") == 0 and (s.get("epochs_parsed") or 0) > 0)
            if s else None,
            f"rc={s.get('returncode')} epochs_parsed={s.get('epochs_parsed')}" if s else "arm not run")
        ck[f"ckpt_eval_valid_{arm}"] = item(
            s.get("checkpoint_eval_valid") if s else None,
            f"best_val_acc={s.get('best_val_acc')} missing={s.get('missing_artifacts')}" if s else "arm not run")
        d = diag.get(arm) or {}
        for key, name in (("backbone_grad_norm", "backbone_grads"),
                          ("classifier_grad_norm", "classifier_grads"),
                          ("backbone_dw", "backbone_moved"),
                          ("feature_drift", "feature_drift")):
            v = d.get(key)
            ck[f"{name}_{arm}"] = item((v is not None and v > 0) if v is not None else None,
                                       f"{key}={v}" if key in d else "diagnostics unavailable")
    # arms distinguishable? compare recorded numerics
    t, c = arms.get("trades") or {}, arms.get("ce") or {}
    fields = ["training_loss_final", "best_val_acc"]
    vals_t = [t.get(f) for f in fields]
    vals_c = [c.get(f) for f in fields]
    if t and c and all(v is not None for v in vals_t + vals_c):
        identical = all(abs(float(a) - float(b)) < 1e-12 for a, b in zip(vals_t, vals_c))
        ck["trades_vs_ce_differ"] = item(not identical,
                                         f"trades={vals_t} ce={vals_c}")
    else:
        ck["trades_vs_ce_differ"] = item(None, "arms incomplete")
    return ck


# ═══════════════════════════════════════════════════════════════════════════
# J1.9 — Artifact export: comparison.csv, summary.md, manifest.json
# ═══════════════════════════════════════════════════════════════════════════
def j19_export(out_dir: str, ctx: Dict[str, Any]) -> None:
    env, repo, dataset, cfginfo = ctx["env"], ctx["repo"], ctx["dataset"], ctx["config"]
    arms, diag, decision = ctx["arms"], ctx["diagnostics"], ctx["decision"]

    rows = [
        ("epochs_parsed", (arms.get("trades") or {}).get("epochs_parsed"), (arms.get("ce") or {}).get("epochs_parsed")),
        ("training_loss_final", (arms.get("trades") or {}).get("training_loss_final"), (arms.get("ce") or {}).get("training_loss_final")),
        ("best_val_acc", (arms.get("trades") or {}).get("best_val_acc"), (arms.get("ce") or {}).get("best_val_acc")),
        ("runtime_sec", (arms.get("trades") or {}).get("runtime_sec"), (arms.get("ce") or {}).get("runtime_sec")),
        ("backbone_dw", (diag.get("trades") or {}).get("backbone_dw"), (diag.get("ce") or {}).get("backbone_dw")),
        ("classifier_dw", (diag.get("trades") or {}).get("classifier_dw"), (diag.get("ce") or {}).get("classifier_dw")),
        ("backbone_grad_norm", (diag.get("trades") or {}).get("backbone_grad_norm"), (diag.get("ce") or {}).get("backbone_grad_norm")),
        ("classifier_grad_norm", (diag.get("trades") or {}).get("classifier_grad_norm"), (diag.get("ce") or {}).get("classifier_grad_norm")),
        ("feature_drift", (diag.get("trades") or {}).get("feature_drift"), (diag.get("ce") or {}).get("feature_drift")),
    ]
    with open(os.path.join(out_dir, "comparison.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "trades", "ce", "delta"])
        for name, a, b in rows:
            delta = (float(a) - float(b)) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else ""
            w.writerow([name, a, b, delta])

    write_json(os.path.join(out_dir, "decision.json"), decision)
    lines = [
        "# J1 Foundation Gate — run report", "",
        f"- run_id: `{ctx['run_id']}`  ",
        f"- generated: {utc_now()}  ",
        f"- git: `{repo.get('commit')}` ({repo.get('branch')})  ",
        f"- env: GPU={env.get('gpu')} torch={env.get('torch')} CUDA={env.get('cuda_compiled')} python={env.get('python')}  ",
        f"- dataset: `{dataset.get('data_root')}` valid={dataset.get('valid')}  ",
        f"- seed: {ctx.get('seed')} | phase: {PHASE} | batch: 64 | workers: 0  ",
        "", "## Commands", "",
        f"- A (TRADES): `{' '.join(cfginfo['arms']['trades'])}`",
        f"- B (CE):     `{' '.join(cfginfo['arms']['ce'])}`",
        "", "## Measurements", "",
        "| metric | TRADES | CE |", "|---|---|---|",
    ]
    for name, a, b in rows:
        lines.append(f"| {name} | {a} | {b} |")
    lines += ["", "## Gate checklist", "",
              "| item | value | evidence |", "|---|---|---|"]
    for k, v in decision["checklist"].items():
        lines.append(f"| {k} | {v['value']} | {v['evidence']} |")
    lines += ["", f"## DECISION: **{decision['decision']}**", ""]
    lines += [f"- {r}" for r in decision["reasons"]]
    lines += ["", f"Thresholds used: {decision['thresholds_used']}",
              "", "> This gate is OPEN until a PASS is recorded from real T4 "
              "results. No Gen-2 campaign arm may start before the gate passes.",
              "", "## Artifacts", ""]
    for root, _dirs, files in os.walk(out_dir):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            lines.append(f"- `{os.path.relpath(p, out_dir)}`")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    manifest = {"run_id": ctx["run_id"], "generated_utc": utc_now(),
                "git": repo, "environment": env, "dataset": dataset,
                "config": cfginfo, "decision": decision["decision"],
                "files": {}}
    for fn in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, fn)
        if os.path.isfile(p) and fn != "manifest.json":
            manifest["files"][fn] = sha256_file(p)
    write_json(os.path.join(out_dir, "manifest.json"), manifest)


# ═══════════════════════════════════════════════════════════════════════════
# Selftest — CPU, no carrier sources, no dataset required (--selftest)
# ═══════════════════════════════════════════════════════════════════════════
def run_selftest() -> int:
    import copy
    import torch
    import torch.nn as nn
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'} — {name}")
        ok = ok and bool(cond)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8))
            self.cls_head = nn.Linear(8, 3)

        def forward(self, x):
            return self.cls_head(self.backbone(x))

    torch.manual_seed(0)
    toy = Toy()
    x, y = torch.randn(16, 4), torch.randint(0, 3, (16,))

    g = gradient_norms(toy, x, y)
    check("grad norms finite and >0 for both groups",
          g["backbone_grad_norm"] > 0 and g["classifier_grad_norm"] > 0)

    baseline = capture_features(toy, x, tap="backbone")
    trained = copy.deepcopy(toy)
    with torch.no_grad():
        for p in trained.backbone.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    dw = group_weight_delta(trained, toy)
    check("backbone |dW| > 0 after perturbation", dw["backbone_dw"] > 0)
    check("classifier |dW| == 0 when classifier untouched", dw["classifier_dw"] == 0)
    fd = feature_drift(trained, x, baseline)
    check("feature drift > 0 after backbone perturbation",
          fd["feature_drift"] is not None and fd["feature_drift"] > 0)
    fd0 = feature_drift(toy, x, baseline)
    check("feature drift ~0 for identical state", fd0["feature_drift"] < 1e-6)

    log = ("[T] epoch 1/60 loss=4.6012 val_acc=0.0910\n"
           "[T] epoch 2/60 loss=3.9000 val_acc=0.1500\n")
    rows = parse_epoch_log(log)
    check("epoch log parser", len(rows) == 2 and rows[1]["loss"] == 3.9
          and rows[1]["val_acc"] == 0.15)

    A = canonical_command("data/imagenet100", clean_only=False)
    B = canonical_command("data/imagenet100", clean_only=True)
    check("arms matched: B == A + --clean-only only", B == A + ["--clean-only"])

    all_true = {k: {"value": True, "evidence": "t"} for k in HARD_ITEMS}
    check("decide: all true -> PASS", decide(all_true)["decision"] == "PASS")
    bad = dict(all_true); bad["trades_completed"] = {"value": False, "evidence": "rc=1"}
    check("decide: measured false -> FAIL", decide(bad)["decision"] == "FAIL")
    miss = dict(all_true); miss["feature_drift_ce"] = {"value": None, "evidence": ""}
    check("decide: missing -> INCONCLUSIVE", decide(miss)["decision"] == "INCONCLUSIVE")
    check("decide: identical arms -> INCONCLUSIVE",
          decide(all_true, arms_identical=True)["decision"] == "INCONCLUSIVE")
    check("HARD_ITEMS coverage of spec checklist",
          {"trades_completed", "ce_completed", "backbone_grads_trades",
           "backbone_moved_trades", "classifier_grads_trades",
           "feature_drift_trades"} <= set(HARD_ITEMS))
    check("ALL_REQUIRED_PATHS completeness",
          len(ALL_REQUIRED_PATHS) >= 8 and any("train_generation1_foundation.py" in p for p in ALL_REQUIRED_PATHS))
    check("bootstrap_sources dry-run contract",
          bootstrap_sources(REPO_ROOT, dry_run=True).get("status") in ("already_present", "dry_run_missing"))
    check("is_interactive_notebook callable and boolean",
          isinstance(is_interactive_notebook(), bool))
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ═══════════════════════════════════════════════════════════════════════════
# Orchestration: J1.0 -> J1.9
# ═══════════════════════════════════════════════════════════════════════════
def is_interactive_notebook() -> bool:
    """Detect if running inside an interactive notebook (Jupyter / Google Colab)."""
    try:
        from IPython import get_ipython  # type: ignore[import-untyped]
        ip = get_ipython()
        if ip is not None:
            if hasattr(ip, "kernel") or "IPKernelApp" in getattr(ip, "config", {}):
                return True
            if "google.colab" in sys.modules:
                return True
    except Exception:
        pass
    return False


def _collab_kernel_args(argv: Optional[List[str]]) -> List[str]:
    """Strip the JupyterKernelLauncher CLI args before argparse parses.

    Jovian / Colab Kernels run the user script via ipykernel_launcher.py, which
    injects '-f <kernel.json>'. argparse does not know about that flag, so the
    parser would raise SystemExit(2). Drop any '-f' (and its single following
    value) from the argv the user gave us, and hand the cleaned list to
    parse_args. Everything else (--dry-run, --selftest, --data-root,
    --run-id) is untouched and still parsed normally.
    """
    if argv is None:
        argv = sys.argv[1:]
    cleaned: List[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token == "-f":
            skip = True
            continue
        cleaned.append(token)
    return cleaned


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="J1 Foundation Gate harness (Colab T4)")
    ap.add_argument("--dry-run", action="store_true",
                    help="pre-flight only: verify + print, no training, no writes")
    ap.add_argument("--prepare", action="store_true",
                    help="prepare environment, bootstrap sources and dataset, verify dependencies without running training")
    ap.add_argument("--selftest", action="store_true",
                    help="CPU test of measurement + decision logic")
    ap.add_argument("--data-root", default=None,
                    help="path to ImageNet-100 dataset root (default: data/imagenet100)")
    ap.add_argument("--run-id", default=None,
                    help="custom run id for artifact directory")
    ap.add_argument("--source-ref", default=os.environ.get("RHAN_SOURCE_REF"),
                    help="git ref to resolve pipeline sources from (default: RHAN_SOURCE_REF or auto)")
    ap.add_argument("--source-repo", default=os.environ.get("RHAN_REPO_URL"),
                    help="remote git URL for fetching pipeline sources if needed")
    ap.add_argument("--source-mode", default=os.environ.get("RHAN_SOURCE_MODE", "auto"),
                    choices=["auto", "local", "fetch"],
                    help="mode for obtaining pipeline sources: auto, local, fetch")
    ap.add_argument("--no-bootstrap", action="store_true",
                    help="disable automatic source and dataset bootstrapping (check only)")
    args = ap.parse_args(_collab_kernel_args(argv))

    if args.selftest:
        return run_selftest()

    print("J1.0 — Environment verification")
    env = j10_environment()
    print("J1.1 — Repository verification & source bootstrap")
    repo = j11_repository(
        cwd=REPO_ROOT,
        source_ref=args.source_ref,
        repo_url=args.source_repo,
        source_mode=args.source_mode,
        bootstrap=not args.no_bootstrap,
        dry_run=args.dry_run,
    )
    print("J1.2 — Dataset verification & bootstrap")
    dataset = j12_dataset(
        data_root=args.data_root,
        bootstrap=not args.no_bootstrap,
        dry_run=args.dry_run,
    )
    print("J1.3 — Foundation configuration")
    data_root_for_cfg = dataset.get("data_root") or _resolve_data_root(args.data_root)
    config = j13_configuration(data_root_for_cfg)

    # ── Explicit --prepare mode ──────────────────────────────────────────────
    if args.prepare:
        prep_id = args.run_id or datetime.now(timezone.utc).strftime("prepare_%Y%m%dT%H%M%SZ")
        prep_out = os.path.join("runs", "j1_foundation", prep_id)
        if not args.dry_run:
            os.makedirs(prep_out, exist_ok=True)
            write_json(os.path.join(prep_out, "config.json"), config)
            write_json(os.path.join(prep_out, "environment.json"), env)

        if args.dry_run:
            print("\n================================================================================")
            print("PREPARATION REPORT (DRY-RUN)")
            print("================================================================================")
            print(f"Environment       : {'✓' if env.get('gpu_ok') else '✗ MISSING'}")
            print(f"Repository        : {'✓' if repo.get('repo_ok') else '✗ MISSING'}")
            print(f"Pipeline sources  : {'✓' if repo.get('pipeline_sources_ok') else '✗ MISSING'}")
            print(f"Dataset           : {'✓' if dataset.get('valid') else '✗ MISSING'}")
            print(f"Dependencies      : {'✓' if repo.get('import_ok') else '✗ MISSING'}")
            print("\nNo experiment executed.")
            print("No scientific artifacts created.")
            print("================================================================================")
            return 0

        prep_ok = bool(env.get("gpu_ok") and repo.get("pipeline_sources_ok") and repo.get("import_ok") and dataset.get("valid"))
        print("\n================================================================================")
        print(f"PREPARATION STATUS: {'READY' if prep_ok else 'INCOMPLETE'}")
        print("================================================================================")
        print(f"Environment       : {'✓' if env.get('gpu_ok') else '✗ FAIL'}")
        print(f"Repository        : {'✓' if repo.get('repo_ok') else '✗ FAIL'}")
        print(f"Pipeline sources  : {'✓' if repo.get('pipeline_sources_ok') else '✗ FAIL'}")
        print(f"Dataset           : {'✓' if dataset.get('valid') else '✗ FAIL'}")
        print(f"Dependencies      : {'✓' if repo.get('import_ok') else '✗ FAIL'}")
        if prep_ok:
            print("\nThe Colab runtime is fully prepared.")
            print("Run 'python3 cloud/gen1/colab_j1_foundation_gate.py' to launch the Foundation Gate experiment.")
        else:
            print("\nPreparation incomplete. Check logs above for missing prerequisites.")
        print("================================================================================")
        return 0 if prep_ok else 1

    # ── Dry-run mode ─────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n--dry-run: pre-flight only. Would run:")
        print("  A:", " ".join(config["arms"]["trades"]))
        print("  B:", " ".join(config["arms"]["ce"]))
        print("  artifacts -> runs/j1_foundation/<run_id>/ "
              "(config/environment/manifest/trades_summary/ce_summary/"
              "comparison.csv/summary.md/logs/checkpoints/diagnostics)")
        print("PRE-FLIGHT OK — no training launched, no state written.")
        return 0

    run_id = args.run_id or datetime.now(timezone.utc).strftime("j1_%Y%m%dT%H%M%SZ")
    out_dir = os.path.join("runs", "j1_foundation", run_id)
    os.makedirs(out_dir, exist_ok=True)
    write_json(os.path.join(out_dir, "config.json"), config)
    write_json(os.path.join(out_dir, "environment.json"), env)

    ctx: Dict[str, Any] = {"run_id": run_id, "env": env, "repo": repo,
                           "dataset": dataset, "config": config,
                           "arms": {}, "diagnostics": {}}

    shielded = ["report/generation1_foundation_roadmap.json",
                "runs/foundation_backbone_only/manifest.json"]
    backups = {p: (open(p, "rb").read() if os.path.exists(p) else None)
               for p in shielded}
    try:
        can_run = bool(env.get("gpu_ok") and repo.get("pipeline_sources_ok") and repo.get("import_ok") and dataset.get("valid"))
        if not can_run:
            print("  ✗ prerequisites not met — arms will NOT run "
                  f"(gpu_ok={env.get('gpu_ok')}, repo_ok={repo.get('pipeline_sources_ok')}, "
                  f"imports_ok={repo.get('import_ok')}, dataset_ok={dataset.get('valid')})")
            ctx["preflight"] = {"skipped": "prerequisites not met"}
        else:
            print("J1.4 — Gradient-reach preflight")
            ctx["preflight"] = j14_preflight(out_dir)
            print("J1.5 — TRADES backbone-only run")
            ctx["arms"]["trades"] = run_arm("trades", config["arms"]["trades"],
                                            out_dir, os.path.join(out_dir, "logs"))
            print("J1.6 — CE backbone-only control (--clean-only)")
            ctx["arms"]["ce"] = run_arm("ce", config["arms"]["ce"],
                                        out_dir, os.path.join(out_dir, "logs"))
            print("J1.7 — Diagnostics")
            ctx["diagnostics"] = j17_diagnostics(out_dir, ctx["arms"])
    finally:
        # restore canonical ladder protocol state — J1 must not advance it
        for p, data in backups.items():
            if data is None:
                if os.path.exists(p):
                    os.remove(p)
            else:
                with open(p, "wb") as f:
                    f.write(data)
        print("  shielded ladder state (roadmap/manifest) restored")

    print("J1.8 — Foundation-gate decision")
    checklist = build_checklist(env, repo, dataset, ctx["arms"], ctx["diagnostics"])
    t, c = ctx["arms"].get("trades") or {}, ctx["arms"].get("ce") or {}
    identical = bool(t and c and t.get("training_loss_final") is not None
                     and t.get("training_loss_final") == c.get("training_loss_final")
                     and t.get("best_val_acc") == c.get("best_val_acc"))
    ctx["decision"] = decide(checklist, arms_identical=identical)
    print(f"  DECISION: {ctx['decision']['decision']} — "
          f"{'; '.join(ctx['decision']['reasons'])}")
    print("J1.9 — Artifact export")
    ctx["seed"] = (ctx.get("preflight") or {}).get("seed")
    j19_export(out_dir, ctx)
    print(f"  artifacts: {out_dir}/ (summary.md, decision.json, comparison.csv, manifest.json, logs/, diagnostics/, trades/, ce/)")
    print(f"\nFOUNDATION GATE: {ctx['decision']['decision']}  (OPEN until a real T4 PASS)")
    return 0 if ctx["decision"]["decision"] != "FAIL" else 1


if __name__ == "__main__":
    rc = main()
    if is_interactive_notebook():
        if rc != 0:
            raise RuntimeError(f"J1 Foundation Gate failed with exit code: {rc}")
        # Clean completion in interactive notebook without SystemExit(0)
    else:
        sys.exit(rc)
