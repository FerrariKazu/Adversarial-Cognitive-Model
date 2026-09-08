#!/usr/bin/env python3
"""
Comparator-reuse registry — RHAN-NX rule 1b.
================================================================================

NEVER re-run eval on an already-validated checkpoint. D, trades_large_baseline,
rhan_next_ais_v1_halting_only (B), and rhan_next_hpc_only (C) all have
validated per-seed CSVs and provenance JSON on HF and in report/. Every new
stage's comparison table MUST pull these through this module — extending the
existing seed_sweep_comparators.py / REUSE_COMPARATOR_EVAL pattern used for
the E3 sweep. Only the NEW checkpoint (SBR-N, D2, or D3) gets freshly
evaluated in any given stage.

Each registry entry records the checkpoint's sha256 (the file that produced
the rows) so a silently swapped/corrupted checkpoint file can never have its
old numbers cited against it.

`load_comparator(label, requested_seeds)` returns the exact validated rows
(never re-runs PGD). Verification, in order:
  1. source_csv exists locally; if not, downloads from HF first (same
     pattern as regenerate_e1_table.py's HF-with-fallback);
  2. when the checkpoint file exists locally, its sha256 must match the
     registry's recorded hash — catches a silently swapped/corrupted
     checkpoint being cited with old numbers;
  3. requested_seeds must be a SUBSET of validated_seeds — raise loudly if
     the caller asks for a seed the comparator was never evaluated on,
     rather than silently returning fewer rows than expected.
"""
from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Optional, Sequence

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_NAME = "epsilon_sweep_per_seed.csv"

#: HF dataset mirroring report/ (the eval-sweep CSV store; same repo used by
#: seed_sweep_comparators.py).
HF_EVAL_REPO = "FerrariKazu/rhan-eval-sweep"

#: Checkpoint file hashes (sha256 of the .pth) for the four validated
#: comparators. D = rhan_next_ais_hpc (verified locally 2026-09-08:
#: 33144e0a...); B = rhan_next_ais_v1_halting_only (roadmap stage 1 verdict);
#: C = rhan_next_hpc_only (roadmap stage 2 verdict); baseline =
#: rhan_stl10_large_pseudolabel (roadmap stage 1/2 verdicts).
COMPARATOR_REGISTRY: Dict[str, Dict] = {
    "trades_large_baseline": {
        "source_csv": "report/sweep_stage4_e1_d_e1_pgd100/epsilon_sweep_per_seed.csv",
        "hf_path": "sweep_stage4_e1_d_e1_pgd100/epsilon_sweep_per_seed.csv",
        "sha256_checkpoint": "37a4eee0c37b77adc291cb45cc19310de1249c7cd81b0e779b2ebaea6d466afc",
        "checkpoint_path": "checkpoints/rhan_stl10_large_pseudolabel_best.pth",
        "validated_seeds": list(range(41, 57)),     # 16 seeds
        "validated_date": "2026-08-31",             # E1 sweep session
        "note": "DONOR row source (PGD-100 protocol, norm-space eps, n=300).",
    },
    "rhan_next_ais_hpc": {                          # D — FROZEN ground truth
        "source_csv": "report/sweep_stage4_e1_d_e1_pgd100/epsilon_sweep_per_seed.csv",
        "hf_path": "sweep_stage4_e1_d_e1_pgd100/epsilon_sweep_per_seed.csv",
        "sha256_checkpoint": "33144e0acf8fea34736764d6674cae92be5232cabd86d13342f8a51db2c1f3f2",
        "checkpoint_path": "checkpoints/rhan_next_ais_hpc_best.pth",
        "validated_seeds": list(range(41, 57)),     # 16 seeds
        "validated_date": "2026-08-31",
        "note": "DONOR row source (PGD-100 protocol, norm-space eps, n=300). "
                "D is FROZEN read-only ground truth — never retrained, never "
                "re-evaluated for the purpose of improving its stored numbers.",
    },
    "rhan_next_ais_v1_halting_only": {              # B — 8 seeds only
        "source_csv": "report/sweep_stage1_ais_v1_halting_only_merged/epsilon_sweep_per_seed.csv",
        "hf_path": "sweep_stage1_ais_v1_halting_only_merged/epsilon_sweep_per_seed.csv",
        "sha256_checkpoint": "19582ff4b32afdb2a46e88a7089167844a2ac3b24d01d6c34cac79ebbf1118e3",
        "checkpoint_path": "checkpoints/rhan_next_ais_v1_halting_only_best.pth",
        "validated_seeds": list(range(41, 49)),     # 8 seeds — DIFFERENT n
        "validated_date": "2026-08-10",
        "seed_mismatch_caveat": ("B is only 8-seed validated vs D/baseline's "
                                 "16 — any comparison mixing this comparator "
                                 "with a 16-seed fresh checkpoint MUST "
                                 "explicitly flag the seed-count mismatch, "
                                 "never silently average across mismatched n."),
    },
    "rhan_next_hpc_only": {                         # C — 5 seeds only
        "source_csv": "report/sweep_stage2_hpc_only/epsilon_sweep_per_seed.csv",
        "hf_path": "sweep_stage2_hpc_only/epsilon_sweep_per_seed.csv",
        "sha256_checkpoint": "5b7dce1e37e9e26d95a742f8a6dc0ec4e1c895766ca42e0316ec78a90f6f55a0",
        "checkpoint_path": "checkpoints/rhan_next_hpc_only_best.pth",
        "validated_seeds": list(range(41, 46)),     # 5 seeds — DIFFERENT n
        "validated_date": "2026-08-18",
        "seed_mismatch_caveat": ("C is only 5-seed validated vs D/baseline's "
                                 "16 — same explicit-flag rule as B."),
    },
}

#: Clean-only clean-accuracy reference (SBR-1 gate uses D's frozen record).
D_CLEAN_ACC_MEAN = 54.96      # D's clean accuracy, frozen record (16-seed eval)

#: D's eps=0.094 PGD-100 record (frozen, E1 sweep): used by report generators
#: as the single canonical D-vs-new comparison point.
D_PGD100_ACC_MEAN = 34.02
D_PGD100_ACC_STD = 3.24


def _get_hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    for module, attr in (("google.colab", "userdata"),
                         ("kaggle_secrets", "UserSecretsClient")):
        try:
            if module == "google.colab":
                from google.colab import userdata
                return userdata.get("HF_TOKEN") or None
            else:
                from kaggle_secrets import UserSecretsClient
                return UserSecretsClient().get_secret("HF_TOKEN") or None
        except Exception:
            continue
    return None


def _download_csv(entry: Dict, hf_token: Optional[str]) -> bool:
    """Download the comparator's source CSV from HF to its local path.
    Returns True when the local file now exists."""
    local = os.path.join(REPO_ROOT, entry["source_csv"])
    if os.path.exists(local):
        return True
    if not hf_token:
        return False
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(repo_id=HF_EVAL_REPO, repo_type="dataset",
                                     filename=entry["hf_path"], token=hf_token)
        os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
        import shutil
        shutil.copyfile(downloaded, local)
        return os.path.exists(local)
    except Exception as e:
        print(f"  [comparator_registry] WARNING: HF download failed for "
              f"{entry['hf_path']}: {e}", flush=True)
        return False


def _verify_checkpoint_sha(entry: Dict) -> List[str]:
    """Verify the local checkpoint (when present) matches the registry hash.
    Returns a list of problem strings ([] = OK). Missing checkpoint files are
    NOT a problem — the rows' provenance is the registry's record — but a
    PRESENT file with a mismatched hash is a loud failure."""
    problems = []
    ckpt = os.path.join(REPO_ROOT, entry["checkpoint_path"])
    if not os.path.exists(ckpt):
        return problems
    h = hashlib.sha256()
    try:
        with open(ckpt, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError as e:
        return [f"cannot hash {ckpt}: {e}"]
    if h.hexdigest() != entry["sha256_checkpoint"]:
        problems.append(
            f"checkpoint {ckpt} sha256 {h.hexdigest()[:16]}... does NOT match "
            f"the registry's recorded "
            f"{entry['sha256_checkpoint'][:16]}... — a silently "
            f"swapped/corrupted checkpoint must never be cited with its old "
            f"validated rows (rule 1b).")
    return problems


def load_comparator(label: str, requested_seeds: Sequence[int]) -> pd.DataFrame:
    """Load a comparator's validated per-seed rows, verified.

    Args:
        label: a key of COMPARATOR_REGISTRY.
        requested_seeds: the seeds the caller wants rows for — must be a
            SUBSET of the comparator's validated_seeds.

    Returns the exact validated rows (all columns, untouched), restricted to
    `requested_seeds`. Raises (never silently returns fewer rows):
      * KeyError for an unknown label;
      * AssertionError when requested seeds are not a subset of the
        comparator's validated seeds;
      * AssertionError when the source CSV is unavailable locally AND on HF
        (the report cannot cite rows that don't exist);
      * AssertionError when the local checkpoint hash mismatches the registry
        (swapped/corrupted checkpoint with old numbers).
    """
    if label not in COMPARATOR_REGISTRY:
        raise KeyError(
            f"unknown comparator {label!r} — registered: "
            f"{sorted(COMPARATOR_REGISTRY)}")
    entry = COMPARATOR_REGISTRY[label]
    valid = set(entry["validated_seeds"])
    wanted = set(int(s) for s in requested_seeds)
    missing_seeds = wanted - valid
    if missing_seeds:
        raise AssertionError(
            f"comparator {label!r} was validated on seeds "
            f"{sorted(valid)} only; requested seeds {sorted(missing_seeds)} "
            f"were NEVER evaluated on it (validated {entry['validated_date']}). "
            f"Never silently return fewer rows than requested (rule 1b).")

    problems = _verify_checkpoint_sha(entry)
    if problems:
        raise AssertionError("\n".join(problems))

    if not _download_csv(entry, _get_hf_token()):
        raise AssertionError(
            f"comparator {label!r} source CSV not found locally "
            f"({entry['source_csv']}) and could not be restored from HF "
            f"({HF_EVAL_REPO}/{entry['hf_path']}). The rows it would cite do "
            f"not exist — refusing to fabricate a comparison.")

    df = pd.read_csv(os.path.join(REPO_ROOT, entry["source_csv"]))
    rows = df[df["ckpt_label"] == label]
    rows = rows[rows["seed"].isin(wanted)].copy()
    # Seed-subset completeness guard: every requested seed must have every
    # (eps) row the CSV carries for this label.
    eps_counts = rows.groupby("seed")["eps_pixel"].nunique()
    got = set(rows["seed"].astype(int).unique())
    if got != wanted or (len(eps_counts) and eps_counts.nunique() != 1):
        raise AssertionError(
            f"comparator {label!r}: expected {len(wanted)} seeds x uniform eps "
            f"rows, found seeds={sorted(got)} eps-counts="
            f"{dict(eps_counts)} — the source CSV is incomplete; refusing "
            f"to cite partial rows.")
    return rows.reset_index(drop=True)


def donor_note(label: str) -> str:
    """The mandated donor-row caveat string for a comparator label.

    Every merged report MUST state, per comparator row used:
      "DONOR row from <source_csv>, validated <date>, NOT re-evaluated
       this session" — exactly the caveat language established for E3's
      D+baseline donor rows.
    """
    e = COMPARATOR_REGISTRY[label]
    return (f"DONOR row from {e['source_csv']}, validated "
            f"{e['validated_date']}, NOT re-evaluated this session.")


def seed_mismatch_warning(label: str) -> Optional[str]:
    """Explicit seed-count caveat for B (8-seed) / C (5-seed) comparators."""
    e = COMPARATOR_REGISTRY.get(label)
    return e.get("seed_mismatch_caveat") if e else None