#!/usr/bin/env python3
"""Stage 1 verdict-label fix in docs/rhan_next_roadmap.json (+ HF copy).

Applies the 2026-08-10 honest-labeling pass to the RHAN-Next roadmap:

  1. FIXES THE STALE "54.05%" LABEL: the recorded eval_target_note claimed
     "All evaluated artifacts are *_best.pth (peak-val) checkpoints", but the
     2026-08-10 state-dict verification (scratch/verify_best_vs_rolling.py)
     proved rhan_next_ais_v1_halting_only best.pth == rolling.pth — both are
     the FINAL-EPOCH (epoch 60) model. The 54.05% in the rolling checkpoint's
     best_acc metadata is the PEAK-VAL accuracy (weights lost to a session
     wipe), NOT the accuracy of the artifact. Recorded here so no future
     writeup cites 54.05% as this checkpoint's clean accuracy.

  2. RECORDS THE HONEST STAGE 1 HEADLINE (8-seed merged verdict, 2026-08-10):
     +8.5pp positive trend, NOT significant under the pre-registered 2-sigma
     criterion; both models masking-free; no third seed extension.

  3. RECORDS THE STAGE 2 HANDOFF: AIS-v1 (halting-only, no recon-mod) is the
     fixed validated Stage 1 config that Stage 2 (HPC) builds on.

Base = the HF-synced roadmap (the Colab sessions' live source of truth), NOT
the possibly-stale working copy. Writes docs/rhan_next_roadmap.json locally
and pushes it back to HF (same upload path as the notebook's sync_roadmap_up).
Bumps roadmap_rev 2 -> 3 so the version guard treats it as newer.
"""
from __future__ import annotations

import json
import os
import sys

import urllib.request

ROLLING_REPO = "FerrariKazu/rhan-checkpoints-rolling"
LOCAL = "docs/rhan_next_roadmap.json"

STATE_DICT_HASH = "fddc8e09214a0574b42b00df5804bc6747d087445e5bfabe568171a73df6a014"


def _fetch_hf():
    req = urllib.request.Request(
        f"https://huggingface.co/datasets/{ROLLING_REPO}/resolve/main/"
        "rhan_next_roadmap.json",
        headers={"User-Agent": "stage1-label-fix"})
    return json.load(urllib.request.urlopen(req, timeout=90))


def main():
    data = _fetch_hf()
    rev = data.get("roadmap_rev", 0)
    print(f"fetched HF roadmap rev {rev}")

    s1 = data["stages"]["1"]
    sv = s1.setdefault("stage1_verdict", {})

    # ── 1. best-vs-rolling verification + stale-label fix ──────────────────
    verification = {
        "verified": True,
        "verified_date": "2026-08-10",
        "method": ("per-tensor state-dict sha256 of best.pth "
                   "(FerrariKazu/rhan-checkpoints) vs rolling.pth "
                   "(FerrariKazu/rhan-checkpoints-rolling); see "
                   "scratch/verify_best_vs_rolling.py"),
        "state_dict_hash": STATE_DICT_HASH,
        "best_path": "checkpoints/rhan_next_ais_v1_halting_only_best.pth",
        "rolling_path": "checkpoints/rhan_next_ais_v1_halting_only_rolling.pth",
        "rolling_epoch": 60,
        "rolling_best_acc": 54.05,
        "conclusion": ("best.pth and rolling.pth carry state-dict-IDENTICAL "
                       "weights = the FINAL-EPOCH (epoch 60) model. The 54.05% "
                       "figure in the rolling checkpoint's best_acc metadata is "
                       "the PEAK-VAL accuracy reached during training (those "
                       "weights were lost to a session wipe); it is NOT the "
                       "accuracy of this artifact. Do NOT cite 54.05% as this "
                       "checkpoint's clean accuracy."),
    }
    sv["best_vs_rolling_verification"] = verification
    sv["eval_target_note"] = (
        "rhan_next_ais_v1_halting_only: the evaluated artifact is "
        "checkpoints/rhan_next_ais_v1_halting_only_best.pth, VERIFIED "
        "(2026-08-10, state-dict hash " + STATE_DICT_HASH[:16] + "...) to carry "
        "state-dict-IDENTICAL weights to the final-epoch rolling checkpoint "
        "(epoch 60). It is the FINAL-EPOCH model, NOT the peak-val (54.05%) "
        "best — the peak-val weights were lost to a session wipe and the "
        "trainer's finalize fallback wrote the epoch-60 model as the best "
        "artifact. All recorded results (clean 49.4 +/- 3.47, eps=0.094 "
        "32.21 +/- 2.74, 8-seed) correspond to this epoch-final model; the "
        "54.05% peak-val figure must NOT be cited as this artifact's accuracy.")
    s1["eval_target_note"] = sv["eval_target_note"]

    # ── 2. Honest Stage 1 headline (8-seed final verdict) ──────────────────
    sv["stage1_headline"] = {
        "date_utc": "2026-08-10",
        "text": ("AIS-v1 (halting-only variant) shows a consistent positive "
                 "trend (+8.5pp merged, 8 seeds) at eps=0.094 vs the static "
                 "TRADES baseline, comparable in direction and rough magnitude "
                 "to the earlier null_ablation_v11 finding, but does not reach "
                 "significance under the pre-registered 2-sigma criterion at "
                 "this sample size. Both models confirmed masking-free."),
        "do_not_cite": ("The 3-seed-extension-only PGD-100 result must NOT be "
                        "cited as a standalone 'crossover real' finding."),
        "verdict_summary": {
            "eps": 0.094,
            "n_seeds": 8,
            "ais_acc_mean": 32.21, "ais_acc_std": 2.74,
            "baseline_acc_mean": 23.71, "baseline_acc_std": 3.47,
            "diff_pp": 8.5,
            "threshold_2sig_pp": 8.84,
            "verdict": "positive but NOT significant",
            "pgd50_pgd100_gap_pp_ais": 0.04,
            "masking": "GENUINE robustness (no masking) for both models",
        },
        "finality": ("Merged 8-seed result accepted as the FINAL Stage 1 "
                     "verdict (2026-08-10). No third seed extension will be "
                     "run."),
    }
    s1["validated_note"] = (
        "Stage 1 (AIS-v1 (halting-only variant)) 8-seed matched eval recorded "
        "from report/sweep_stage1_ais_v1_halting_only_merged/"
        "eval_provenance.json — see roadmap.stages['1'].stage1_verdict. "
        "Merged 8-seed result is FINAL (2026-08-10): no third seed extension. "
        "Verdict is what it is, including a null result.")

    # ── 3. Stage 2 handoff (gating rule) ───────────────────────────────────
    s1["stage2_handoff"] = {
        "date_utc": "2026-08-10",
        "decision": ("Proceed to Stage 2 (HPC) with AIS-v1 (halting-only, no "
                     "recon-mod) FIXED as the validated Stage 1 config — "
                     "exactly as this roadmap's own gating rule specifies."),
        "note": ("Stage 2 inherits a real, honestly-scoped, "
                 "non-significant-but-promising baseline to build on. recon-mod "
                 "stays DEFERRED until its own validated isolation; HPC must be "
                 "isolated on/off with AIS-v1 (halting-only variant) held fixed."),
    }

    # ── Write local + push to HF (notebook-parity upload path) ─────────────
    data["roadmap_rev"] = 3
    with open(LOCAL, "w") as f:
        json.dump(data, f, indent=2, sort_keys=False)
    print(f"wrote {LOCAL} (rev {data['roadmap_rev']})")

    if os.environ.get("NO_PUSH") == "1":
        print("NO_PUSH=1 — HF copy NOT updated.")
        return
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN", ""))
    api.upload_file(path_or_fileobj=LOCAL, path_in_repo="rhan_next_roadmap.json",
                    repo_id=ROLLING_REPO, repo_type="dataset",
                    token=os.environ.get("HF_TOKEN", ""))
    print(f"pushed rhan_next_roadmap.json (rev {data['roadmap_rev']}) to "
          f"{ROLLING_REPO}")

    print("\nUpdated fields:")
    print(f"  stage1_verdict.eval_target_note  -> {sv['eval_target_note'][:120]}...")
    print(f"  stage1_verdict.best_vs_rolling_verification -> {verification['state_dict_hash'][:16]}...")
    print(f"  stage1_verdict.stage1_headline    -> {sv['stage1_headline']['text'][:100]}...")


if __name__ == "__main__":
    sys.exit(main())
