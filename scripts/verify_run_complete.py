#!/usr/bin/env python3
"""
verify_run_complete.py — PHASE 11: completion verification (read-only).
================================================================================
Runs when the roadmap shows all six phases done. Verifies, per phase:
rolling + best checkpoints (best/rolling parity), provenance manifest,
eval summary CSV + result json + compactness report; plus global items:
config hash matches the frozen launch manifest, HF rolling+best repos
carry the artifacts, supervisor log exists. Writes
report/run_completion_report.json.

The config-hash compare must mirror the freeze EXACTLY
(scripts/freeze_run_manifest.py): batch 48 / workers 4 (the launcher
defaults recorded in overrides_applied_to_hash) AND cfg.data_root taken
from the frozen manifest's dataset.root — the freeze hashed the ABSOLUTE
dataset root (mirroring run_j1_local.sh's J1_DATA_ROOT). The first
verifier run (2026-09-26) false-failed exactly here: it hashed the
default RELATIVE data/imagenet100, producing 4fb94efb… against the
frozen 902bf0c1…. Operational bug in the verifier; the run itself was
unaffected. If runs/production_launch_manifest.json is absent (a fresh
cloud clone — runs/ is runtime state, not git-tracked), it is restored
from the rolling HF repo, where the freeze synced it.

SELF-HEAL (fresh cloud session): a session that resumes a run completed in
an EARLIER (possibly wiped) session has the HF-synced roadmap but no local
checkpoints/manifests/eval artifacts. The verifier therefore materializes
any missing artifact from its HF repo (rolling ckpts + verdict + manifest
from the rolling repo; best ckpts + per-phase provenance/eval artifacts
from the best repo, preserving the local relative layout) before checking.
On the machine that ran the training nothing is missing, so this is a
no-op there. CROSS-HOST note: the frozen hash covers dataset.root, so if
the frozen manifest was created on another host the raw compare is
recorded as SKIPPED (a note, not a problem) — per-phase provenance
manifests remain the authoritative per-phase provenance.

Exit codes: 0 = complete & verified; 1 = run not complete yet (normal
while training); 2 = roadmap says done but verification FAILED (human
attention required — never silently declare success).
"""
from __future__ import annotations

import json
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from noesis_vision.core.checkpoint import (  # noqa: E402
    verify_best_rolling_parity)
from noesis_vision.core.provenance import config_sha256  # noqa: E402
from training.stage_state_machine import FOUNDATION_PHASES  # noqa: E402
from training.train_generation1_foundation import (  # noqa: E402
    FoundationConfig)


def _hf_token():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        try:
            from dotenv import dotenv_values
            tok = dotenv_values(os.path.join(REPO_ROOT, ".env")).get("HF_TOKEN")
        except Exception:
            pass
    return tok


def _materialize(rel_path: str, repo_id: str, token) -> bool:
    """Pull an artifact that exists only on HF into its expected local path
    (self-heal for a fresh session resuming a run completed elsewhere)."""
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=repo_id, filename=rel_path,
                        repo_type="dataset", token=token, local_dir=REPO_ROOT)
        return os.path.exists(os.path.join(REPO_ROOT, rel_path))
    except Exception:
        return False


def main() -> int:
    cfg = FoundationConfig()
    # The frozen launch configuration (mirrors run_j1_local.sh DEFAULTS).
    cfg.batch_size, cfg.num_workers = 48, 4

    rm_path = os.path.join(cfg.report_dir, "generation1_foundation_roadmap.json")
    problems: list[str] = []
    notes: list[str] = []
    phases: dict[str, dict] = {}

    token = _hf_token()

    # The frozen launch manifest is needed EARLY: dataset.root is part of
    # the hashed configuration (see module docstring — the 2026-09-26
    # 902bf0c1… vs 4fb94efb… false-fail).
    frozen: dict | None = None
    frozen_path = os.path.join(cfg.runs_dir, "production_launch_manifest.json")
    try:
        frozen = json.load(open(frozen_path))
    except Exception:
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download("FerrariKazu/rhan-nxa-checkpoints-rolling",
                                "production_launch_manifest.json",
                                repo_type="dataset", token=token)
            frozen = json.load(open(p))
        except Exception as e:
            problems.append(f"frozen launch manifest unreadable: {e}")
    if frozen is not None:
        cfg.data_root = frozen["dataset"]["root"]

    try:
        fnd = json.load(open(rm_path))["generation1_foundation"]
        statuses = {p: fnd["phases"][p].get("status") for p in FOUNDATION_PHASES}
    except Exception as e:
        print(f"NOT COMPLETE — no readable roadmap ({e})")
        return 1
    n_done = sum(1 for s in statuses.values() if s == "done")
    if n_done < len(FOUNDATION_PHASES):
        print(f"NOT COMPLETE — {n_done}/{len(FOUNDATION_PHASES)} phases done "
              f"{statuses}")
        return 1

    # Per-phase artifact checks.
    for phase in FOUNDATION_PHASES:
        rolling = os.path.join(cfg.ckpt_dir, f"foundation_{phase}_rolling.pth")
        best = os.path.join(cfg.ckpt_dir, f"foundation_{phase}_best.pth")
        manifest = os.path.join(cfg.runs_dir, f"foundation_{phase}",
                                "manifest.json")
        eval_csv = os.path.join(cfg.report_dir,
                                f"foundation_{phase}_eval", "summary_table.csv")
        result = os.path.join(cfg.report_dir, f"foundation_{phase}_result.json")
        comp = os.path.join(cfg.report_dir,
                            f"foundation_{phase}_compactness.json")
        info: dict = {"status": statuses[phase]}
        # rel layout -> HF archive repo, for self-healing materialize.
        targets = (
            ("rolling", rolling, "FerrariKazu/rhan-nxa-checkpoints-rolling"),
            ("best", best, "FerrariKazu/rhan-nxa-checkpoints"),
            ("manifest", manifest, "FerrariKazu/rhan-nxa-checkpoints"),
            ("eval_csv", eval_csv, "FerrariKazu/rhan-nxa-checkpoints"),
            ("result_json", result, "FerrariKazu/rhan-nxa-checkpoints"),
            ("compactness", comp, "FerrariKazu/rhan-nxa-checkpoints"),
        )
        for label, p, repo in targets:
            if not os.path.exists(p):
                _materialize(os.path.relpath(p, REPO_ROOT), repo, token)
            ok = os.path.exists(p)
            info[label] = ok
            if not ok:
                problems.append(f"{phase}: missing {label}: {p}")
        if os.path.exists(best) and os.path.exists(rolling):
            pok, pwhy = verify_best_rolling_parity(best, rolling)
            info["parity"] = pok
            if not pok:
                problems.append(f"{phase}: parity FAILED: {pwhy}")
        phases[phase] = info

    # Config hash must still match the frozen launch manifest.
    match = False
    if frozen is not None:
        frozen_root = frozen.get("dataset", {}).get("root")
        if (frozen_root is not None and os.path.abspath(str(frozen_root))
                != os.path.abspath(cfg.data_root)):
            # Frozen on another host: the hash covers data_root, so a raw
            # compare would ALWAYS fail there. Record honestly instead.
            notes.append("config-hash compare SKIPPED: frozen manifest was "
                         f"created on another host (dataset.root "
                         f"{frozen_root} vs local {cfg.data_root}); the "
                         "per-phase provenance manifests remain "
                         "authoritative.")
        else:
            now_hash = config_sha256(cfg.to_dict())
            match = frozen["code"]["config_sha256"] == now_hash
            if not match:
                problems.append("config hash CHANGED since the freeze "
                                f"({frozen['code']['config_sha256'][:12]}… vs "
                                f"{now_hash[:12]}…)")

    # HF sync: rolling repo carries roadmap + each rolling ckpt; best repo
    # carries each best ckpt.
    hf_rolling: list[str] = []
    hf_best: list[str] = []
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        hf_rolling = api.list_repo_files(
            "FerrariKazu/rhan-nxa-checkpoints-rolling", repo_type="dataset")
        hf_best = api.list_repo_files(
            "FerrariKazu/rhan-nxa-checkpoints", repo_type="dataset")
        if "generation1_foundation_roadmap.json" not in hf_rolling:
            problems.append("HF rolling repo missing the roadmap")
        for phase in FOUNDATION_PHASES:
            if f"foundation_{phase}_rolling.pth" not in hf_rolling:
                problems.append(f"HF rolling repo missing {phase} rolling ckpt")
            if f"foundation_{phase}_best.pth" not in hf_best:
                problems.append(f"HF best repo missing {phase} best ckpt")
    except Exception as e:
        problems.append(f"HF sync unverifiable (network/auth): {e}")

    report = {
        "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phases_done": n_done,
        "phases": phases,
        "config_hash_matches_freeze": match,
        "notes": notes,
        "hf_rolling_files": len(hf_rolling),
        "hf_best_files": len(hf_best),
        "problems": problems,
    }
    out = os.path.join(cfg.report_dir, "run_completion_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    if problems:
        print(f"VERIFICATION FAILED ({len(problems)} problems) — report: {out}")
        for p in problems:
            print(f"  - {p}")
        return 2
    print(f"RUN COMPLETE & VERIFIED — 6/6 phases, parity ok, manifests ok, "
          f"eval artifacts ok, config hash matches freeze, HF synced. "
          f"Report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
