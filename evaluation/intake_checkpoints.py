#!/usr/bin/env python3
"""
Checkpoint Intake & Verification for Stage 0 Closeout
=====================================================
Asserts:
1. All 5 final checkpoints exist.
2. Every checkpoint has epoch == 15.
3. Computes SHA256 and sha8.
4. Creates canonical seed+sha8 files.
5. Emits checkpoints/intake_manifest.json with zero tolerance for collisions.
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
import torch

CKPT_DIR = Path("checkpoints").resolve()
OUT_MANIFEST = CKPT_DIR / "intake_manifest.json"

MANIFEST = {
    "recurrence_112_seed17": {
        "src": "foundation_recurrence_112_seed17_final.pth",
        "seed": 17,
        "phase": "recurrence_112",
        "canonical_name": "foundation_recurrence_112_seed17_final_636e4b1c.pth",
        "expected_sha8": "636e4b1c",
    },
    "recurrence_112_seed42": {
        "src": "foundation_recurrence_112_seed42_final.pth",
        "seed": 42,
        "phase": "recurrence_112",
        "canonical_name": "foundation_recurrence_112_seed42_final_eb40bf84.pth",
        "expected_sha8": "eb40bf84",
    },
    "ff_112_seed17": {
        "src": "foundation_ff_112_final.pth",
        "seed": 17,
        "phase": "ff_112",
        "canonical_name": "foundation_ff_112_seed17_final_d00fb750.pth",
        "expected_sha8": "d00fb750",
    },
    "ff_112_seed42": {
        "src": "foundation_ff_112_seed42_final.pth",
        "seed": 42,
        "phase": "ff_112",
        "canonical_name": "foundation_ff_112_seed42_final_2920df54.pth",
        "expected_sha8": "2920df54",
    },
    "ff_112_seed99": {
        "src": "foundation_ff_112_seed99_final.pth",
        "seed": 99,
        "phase": "ff_112",
        "canonical_name": "foundation_ff_112_seed99_final_abde98a0.pth",
        "expected_sha8": "abde98a0",
    },
}

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 80)
    print("T1 CHECKPOINT INTAKE & INTEGRITY VERIFICATION")
    print("=" * 80)

    seen_shas = set()
    output_records = {}

    for arm_id, cfg in MANIFEST.items():
        src_path = CKPT_DIR / cfg["src"]
        assert src_path.exists(), f"Missing required checkpoint: {src_path}"

        sha = sha256_file(src_path)
        sha8 = sha[:8]
        assert sha8 == cfg["expected_sha8"], f"Hash mismatch for {cfg['src']}! Expected {cfg['expected_sha8']}, got {sha8}"
        assert sha not in seen_shas, f"Duplicate SHA256 collision detected for {cfg['src']}: {sha}"
        seen_shas.add(sha)

        st = torch.load(src_path, map_location="cpu", weights_only=False)
        ep = st.get("epoch", None)
        assert ep == 15, f"CRITICAL: {cfg['src']} has epoch {ep} != 15!"

        target_path = CKPT_DIR / cfg["canonical_name"]
        if not target_path.exists():
            shutil.copyfile(src_path, target_path)
            print(f"Created canonical file: {target_path.name}")
        else:
            assert sha256_file(target_path) == sha, f"Hash mismatch on existing target: {target_path}"

        best_clean = float(st.get("best_clean_acc", st.get("val_clean_acc", 0.0)))
        best_rob = float(st.get("best_rob_acc", st.get("val_rob_acc", 0.0)))

        record = {
            "arm_id": arm_id,
            "seed": cfg["seed"],
            "phase": cfg["phase"],
            "src_file": cfg["src"],
            "canonical_file": cfg["canonical_name"],
            "sha256": sha,
            "sha8": sha8,
            "epoch": ep,
            "training_monitor_clean": best_clean,
            "training_monitor_robust": best_rob,
        }
        output_records[arm_id] = record
        print(f"[{arm_id}] OK: ep={ep}, sha8={sha8}, clean_mon={best_clean*100:.2f}%, rob_mon={best_rob*100:.2f}%")

    with open(OUT_MANIFEST, "w") as f:
        json.dump(output_records, f, indent=2)

    print(f"\nSaved intake manifest to {OUT_MANIFEST}")
    print("ALL 5 CHECKPOINTS VERIFIED (EPOCH==15, SHA-256 UNIQUE).")

if __name__ == "__main__":
    main()
