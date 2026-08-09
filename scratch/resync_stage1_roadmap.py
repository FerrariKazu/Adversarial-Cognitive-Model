#!/usr/bin/env python3
"""One-shot correction of the Stage 1 metadata on the HF roadmap (2026-08-09).

The 2026-08-09 Kaggle resume session clobbered two recorded fields:
  - stages['1'].step_b_resume -> "fresh_from_base" (WRONG: the session
    resumed an already-complete Step B at rolling epoch 60 and trained 0
    epochs — the misleading write is fixed in both notebooks);
  - stages['1'].gate_clear_path/reason -> "healthy_smoke" / "healthy first
    try" (WRONG: the smoke was degenerate (car/airplane); the gate was
    cleared via the 2026-08-07 isolation verdict, and this session merely
    restored that state).

It also stamps stages['1'].eval_target_note: Step C evaluated the FINAL-EPOCH
rolling checkpoint (the *_best.pth never reached HF), so the recorded numbers
are NOT the peak-val model's.

Read-only against the repo (only docs/rhan_next_roadmap.json is rewritten);
the HF copy is updated via the API. roadmap_rev is bumped so future sessions'
sync_roadmap_down version guard accepts the corrected copy.

Usage: python3 scratch/resync_stage1_roadmap.py   (needs HF_TOKEN in env)
"""
import json
import os
import urllib.request

REPO = "FerrariKazu/rhan-checkpoints-rolling"
ROADMAP_PATH = "docs/rhan_next_roadmap.json"


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "stage1-fix"})
    return urllib.request.urlopen(req, timeout=90)


def _eval_target_note(checkpoints):
    notes = []
    for c in checkpoints or []:
        p = str(c.get("path", ""))
        if p.endswith("_rolling.pth"):
            notes.append(f"{c.get('label')}: evaluated on the FINAL-EPOCH "
                         f"rolling checkpoint ({os.path.basename(p)}) — NOT "
                         f"the peak-val best model")
    if notes:
        return ("; ".join(notes) + ". The *_best.pth artifact was missing on "
                "HF (no epoch beat the restored best), so the recorded results "
                "correspond to the epoch-final model, not the peak-val model.")
    return "All evaluated artifacts are *_best.pth (peak-val) checkpoints."


def main():
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise SystemExit("HF_TOKEN not in env — cannot write HF.")

    # Base: the HF copy (authoritative for runtime verdicts).
    try:
        roadmap = json.load(_get(
            f"https://huggingface.co/datasets/{REPO}/resolve/main/"
            "rhan_next_roadmap.json"))
        source = "HF"
    except Exception as e:
        print(f"HF roadmap fetch failed ({e}) — falling back to the local "
              f"committed copy (runtime verdicts may be missing).")
        with open(ROADMAP_PATH) as f:
            roadmap = json.load(f)
        source = "local"

    s1 = roadmap.setdefault("stages", {}).setdefault("1", {})
    sv = s1.setdefault("stage1_verdict", {})
    old = {
        "step_b_resume": s1.get("step_b_resume"),
        "gate_clear_path": s1.get("gate_clear_path"),
        "gate_clear_reason": s1.get("gate_clear_reason"),
    }

    # 1) step_b_resume — truthful record for the resumed/complete run.
    s1["step_b_resume"] = {
        "source": "already_complete_on_hf",
        "note": ("resumed from HF rolling at epoch 60 (already at max 60) — "
                 "0 epochs trained on 2026-08-09. Corrected 2026-08-09: the "
                 "resume session clobbered this field to 'fresh_from_base'; "
                 "the original launch source (per plan: seeded from isoB "
                 "epoch 12, same config) was lost to that overwrite."),
        "seeded": False,
        "original_launch_source": ("unknown (overwritten by the 2026-08-09 "
                                   "resume session); the pre-registered plan "
                                   "was SEED_STEP_B_FROM_ISOB (isoB epoch 12)"),
    }

    # 2) gate_clear — the smoke was DEGENERATE; the isolation verdict cleared
    #    the gate on 2026-08-07.
    s1["gate_clear_path"] = "isolation_verdict"
    s1["gate_clear_reason"] = (
        "isolation_verdict 2026-08-07: recon-mod confirmed driver of the "
        "Pi_D reordering (smoke<->isoB contrast), 'AIS-v1 (halting-only "
        "variant)' selected by the pre-registered decision rule. "
        "[Corrected 2026-08-09: the resume session mislabeled this "
        "'healthy_smoke'/'healthy first try' — the smoke was NOT a healthy "
        "first try.]")

    # 3) eval-target caveat (rolling/epoch-final fallback used by Step C).
    target_note = _eval_target_note(sv.get("checkpoints"))
    s1["eval_target_note"] = target_note
    sv["eval_target_note"] = target_note

    s1["verdict_metadata_fix"] = {
        "date_utc": "2026-08-09",
        "what": ("step_b_resume + gate_clear_path/reason corrected after the "
                 "2026-08-09 resume session mislabeled them; eval_target_note "
                 "added (rolling/epoch-final eval target). Numbers unchanged — "
                 "the 8-seed merged verdict (STEP C2) lands here next."),
    }

    # 4) roadmap_rev bump (max of local/committed and HF, +1) so the version
    #    guard in the notebooks accepts the corrected copy.
    try:
        local_rev = int(json.load(open(ROADMAP_PATH)).get("roadmap_rev", 0))
    except Exception:
        local_rev = 0
    hf_rev = int(roadmap.get("roadmap_rev", 0) or 0)
    roadmap["roadmap_rev"] = max(local_rev, hf_rev) + 1
    print(f"  roadmap_rev: {hf_rev} -> {roadmap['roadmap_rev']} "
          f"(local committed rev was {local_rev})")

    # 5) Write local + HF.
    with open(ROADMAP_PATH, "w") as f:
        json.dump(roadmap, f, indent=2, sort_keys=False)
    print(f"  ✓ {ROADMAP_PATH} updated (from {source})")

    from huggingface_hub import HfApi
    HfApi(token=token).upload_file(
        path_or_fileobj=ROADMAP_PATH, path_in_repo="rhan_next_roadmap.json",
        repo_id=REPO, repo_type="dataset", token=token)
    print("  ✓ rhan_next_roadmap.json synced to HF")

    print("\n  Changed fields:")
    for k, v in old.items():
        print(f"    {k}: {json.dumps(v)[:120] if v else v}")
    print(f"    eval_target_note: {target_note[:200]}")


if __name__ == "__main__":
    main()
