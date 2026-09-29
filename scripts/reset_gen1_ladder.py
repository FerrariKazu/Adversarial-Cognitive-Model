#!/usr/bin/env python3
"""
reset_gen1_ladder.py — LOUD protocol reset for the corrected-recipe re-run.
================================================================================

The 2026-09-25->26 Gen-1 foundation run trained PURE cross-entropy (the
frozen trainer had no adversarial term; confirmed 2026-09-29). The
corrected trainer (TRADES/PGD curriculum, training/adv_curriculum.py)
cannot resume from those checkpoints: different recipe = different
experiment = different config hash. This script performs the AUDIBLE
reset the resume gates require:

  1. ARCHIVE (never delete): per-phase rolling/best checkpoints, the
     roadmap, per-phase manifests, and eval artifacts are copied under
     archive/<stamp>/gen1_pure_ce_<stamp>/, then the originals are
     removed from the working tree;
  2. RESET the roadmap: statuses -> not_started, metadata cleared, and
     roadmap_rev BUMPED (max(local, HF) + 1) so every host's
     sync_roadmap_down treats the reset as the newer state (the rev guard
     is what stops a stale all-done HF roadmap from resurrecting the
     completed run on a cloud clone);
  3. REQUIRE explicit opt-in (--execute) — default is a dry run printing
     exactly what would happen.

HF is NOT touched: the Gen-1 repos keep the pure-CE checkpoints as the
historical record (matching the archive). To also mirror the archive to
HF, see --hf-mirror below.

Usage:
    python3 scripts/reset_gen1_ladder.py            # dry run (default)
    python3 scripts/reset_gen1_ladder.py --execute  # performs the reset
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from training.stage_state_machine import (  # noqa: E402
    DEPENDENCIES,
    FOUNDATION_PHASES,
)

ROADMAP = os.path.join(REPO_ROOT, "report",
                       "generation1_foundation_roadmap.json")
CKPT_DIR = os.path.join(REPO_ROOT, "checkpoints")
RUNS_DIR = os.path.join(REPO_ROOT, "runs")
REPORT_DIR = os.path.join(REPO_ROOT, "report")
HF_ROLLING = "FerrariKazu/rhan-nxa-checkpoints-rolling"
HF_BEST = "FerrariKazu/rhan-nxa-checkpoints"
#: rolling + best checkpoints, per phase
CKPT_PATTERNS = [("foundation_{p}_rolling.pth", HF_ROLLING),
                 ("foundation_{p}_best.pth", HF_BEST)]
#: per-phase runtime artifacts
PER_PHASE = [("runs/foundation_{p}/manifest.json", HF_BEST),
             ("report/foundation_{p}_result.json", HF_BEST),
             ("report/foundation_{p}_compactness.json", HF_BEST),
             ("report/foundation_{p}_eval/summary_table.csv", HF_BEST),
             ("report/foundation_{p}_eval/epsilon_sweep_per_seed.csv",
              HF_BEST),
             ("report/foundation_{p}_eval/eval_provenance.json", HF_BEST)]
#: run-wide records
RUN_WIDE = [("report/generation1_foundation_roadmap.json", HF_ROLLING)]


def _hf_rev(token=None) -> int:
    """HF roadmap rev (0 when unreachable — the archive never blocks)."""
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id=HF_ROLLING,
                            filename="generation1_foundation_roadmap.json",
                            repo_type="dataset", token=token)
        with open(p) as f:
            return int(json.load(f).get("roadmap_rev", 0) or 0)
    except Exception:
        return 0


def collect() -> list:
    """All local artifacts belonging to the pure-CE run (existing only)."""
    found = []
    for phase in FOUNDATION_PHASES:
        for pat, _hf in CKPT_PATTERNS:
            p = os.path.join(CKPT_DIR, pat.format(p=phase))
            if os.path.exists(p):
                found.append(p)
        for rel, _hf in PER_PHASE:
            p = os.path.join(REPO_ROOT, rel.format(p=phase))
            if os.path.exists(p):
                found.append(p)
    if os.path.exists(ROADMAP):
        found.append(ROADMAP)
    return found


def archive_and_remove(files: list, stamp: str) -> str:
    arch_root = os.path.join(REPO_ROOT, "archive",
                             f"gen1_pure_ce_{stamp}")
    for src in files:
        rel = os.path.relpath(src, REPO_ROOT)
        dst = os.path.join(arch_root, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    for src in files:
        os.remove(src)
    # Eval directories may now be empty — remove them too.
    for phase in FOUNDATION_PHASES:
        d = os.path.join(REPO_ROOT, "report",
                         f"foundation_{phase}_eval")
        if os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)
    return arch_root


def reset_roadmap(archive_dir: str, token=None) -> int:
    """Write the reset roadmap (fresh scaffold + bumped rev) to the archive
    AND to the live path. Returns the new rev."""
    from training.stage_state_machine import ensure_foundation_state
    with open(ROADMAP) as f:            # pre-reset copy for the rev math
        old = json.load(f)
    local_rev = int(old.get("roadmap_rev", 0) or 0)
    hf_rev = _hf_rev(token)
    new_rev = max(local_rev, hf_rev) + 1

    fresh = {}
    ensure_foundation_state(fresh)
    fresh["roadmap_rev"] = new_rev
    fresh["recipe"] = ("gen1-adv-curriculum-v1 (TRADES/PGD port of Gen-0; "
                       "the 2026-09-25->26 run was pure-CE — see "
                       "archive/gen1_pure_ce_*/README.txt)")
    for phase in FOUNDATION_PHASES:
        fresh["generation1_foundation"]["phases"][phase]["depends_on"] = \
            DEPENDENCIES[phase]

    with open(os.path.join(archive_dir, "reset_roadmap_before.json"), "w") as f:
        json.dump(old, f, indent=2)
    with open(ROADMAP, "w") as f:
        json.dump(fresh, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return new_rev


def write_readme(archive_dir: str, new_rev: int) -> None:
    with open(os.path.join(archive_dir, "README.txt"), "w") as f:
        f.write(
            f"ARCHIVE: Gen-1 foundation pure-CE run (2026-09-25 -> 2026-09-26)\n"
            f"Archived: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n\n"
            f"This run trained with PURE cross-entropy — the frozen trainer\n"
            f"had NO adversarial term (confirmed 2026-09-29 by code read:\n"
            f"train_one_epoch computed only F.cross_entropy). The corrected\n"
            f"trainer adds the ported Gen-0 TRADES/PGD curriculum\n"
            f"(training/adv_curriculum.py: eps 0.031->0.062->0.094, beta\n"
            f"2.0->2.5, PGD-4, w_trades 0.55) — different recipe, therefore a\n"
            f"NEW experiment, therefore the checkpoints could not be resumed\n"
            f"and the ladder was reset audibly by scripts/reset_gen1_ladder.py.\n\n"
            f"roadmap_rev bumped to {new_rev} so no host's rev guard can\n"
            f"resurrect the completed pure-CE roadmap from HF.\n\n"
            f"Results of this run: see HF rolling repo\n"
            f"report/GEN1_RESULTS_MASTER.md (the complete artifact-backed\n"
            f"extraction) and FerrariKazu/rhan-nxa-checkpoints-rolling.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--execute", action="store_true",
                    help="perform the reset (default: dry run)")
    ap.add_argument("--hf-token", default=None)
    args = ap.parse_args()

    files = collect()
    print(f"gen1 pure-CE reset — {'EXECUTE' if args.execute else 'DRY RUN'}")
    print(f"  artifacts found: {len(files)}")
    for p in files[:6]:
        print(f"    {os.path.relpath(p, REPO_ROOT)}")
    if len(files) > 6:
        print(f"    ... and {len(files) - 6} more")

    if not files:
        print("  nothing to reset (ladder already clean)")
        return 0
    if not args.execute:
        print("  dry run only — pass --execute to perform the reset")
        return 0

    stamp = time.strftime("%Y%m%d_%H%M%S")
    arch = archive_and_remove(files, stamp)
    print(f"  archived + removed originals -> {os.path.relpath(arch, REPO_ROOT)}")
    new_rev = reset_roadmap(arch, args.hf_token)
    write_readme(arch, new_rev)
    print(f"  roadmap reset: 6 phases -> not_started, roadmap_rev -> {new_rev}")
    print("  NOTE: HF repos keep the pure-CE checkpoints (historical record).\n"
          "        The trainer will re-sync the reset roadmap on first launch\n"
          "        (rev-guarded: fresh scaffold is newer than the HF copy).\n"
          "        Next: --smoke proof, then the real dispatch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
