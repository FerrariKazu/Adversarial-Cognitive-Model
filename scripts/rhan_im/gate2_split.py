#!/usr/bin/env python3
"""Freeze the Gate-2 train-image holdout used to measure d' without breaking LTT disjointness.

The Gate-2 calibration losses are only independent of the sensor fit if the d' measurement uses
images that the head did not train on.  This tool holds out a fixed prefix of every train class:
16 images for each target class and 32 for each background class.  The head trains on the
complement; d' is measured on target/background scenes built only from the holdout.  The rule is
deterministic, needs no random seed, and is pinned by the hash of the file it writes.

  python scripts/rhan_im/gate2_split.py
  python scripts/rhan_im/gate2_split.py --check

Nothing here reads credentials or touches the network.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import platform
import sys
from typing import Dict, List

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rhan_im import gate1 as g1  # noqa: E402

RULE_ID = "train-prefix-holdout-v1"
TARGET_HOLDOUT_PER_CLASS = 16
BACKGROUND_HOLDOUT_PER_CLASS = 32
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
TOOL = "scripts/rhan_im/gate2_split.py"


def _files_by_class(root: pathlib.Path, classes: List[str]) -> Dict[str, List[str]]:
    out = {}
    for c in classes:
        d = root / c
        if not d.is_dir():
            raise SystemExit(f"missing train class folder: {d}")
        files = sorted(p.name for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise SystemExit(f"train class folder has no images: {d}")
        out[c] = files
    return out


def build(data_root: pathlib.Path, class_split_path: pathlib.Path) -> Dict[str, object]:
    try:
        sp = g1.read_class_split(class_split_path)
    except (ValueError, KeyError, OSError) as e:
        raise SystemExit(str(e))
    train_root = data_root / "train"
    present = sorted(p.name for p in train_root.iterdir() if p.is_dir())
    expected = sorted(sp["targets"] + sp["backgrounds"])
    if present != expected:
        raise SystemExit(
            "the train/ class folders are not exactly the FINAL class split "
            f"(train-only {sorted(set(present) - set(expected))[:5]}, split-only {sorted(set(expected) - set(present))[:5]})"
        )
    files = _files_by_class(train_root, expected)
    target_holdout, background_holdout = {}, {}
    for c in sp["targets"]:
        if len(files[c]) <= TARGET_HOLDOUT_PER_CLASS:
            raise SystemExit(f"target class {c!r} has {len(files[c])} images; need more than {TARGET_HOLDOUT_PER_CLASS}")
        target_holdout[c] = files[c][:TARGET_HOLDOUT_PER_CLASS]
    for c in sp["backgrounds"]:
        if len(files[c]) <= BACKGROUND_HOLDOUT_PER_CLASS:
            raise SystemExit(f"background class {c!r} has {len(files[c])} images; need more than {BACKGROUND_HOLDOUT_PER_CLASS}")
        background_holdout[c] = files[c][:BACKGROUND_HOLDOUT_PER_CLASS]
    holdout = {**target_holdout, **background_holdout}
    target_head = {c: len(files[c]) - TARGET_HOLDOUT_PER_CLASS for c in sp["targets"]}
    background_head = {c: len(files[c]) - BACKGROUND_HOLDOUT_PER_CLASS for c in sp["backgrounds"]}
    total_train = sum(len(v) for v in files.values())
    target_holdout_count = sum(len(v) for v in target_holdout.values())
    background_holdout_count = sum(len(v) for v in background_holdout.values())
    head_count = sum(target_head.values()) + sum(background_head.values())
    if head_count + target_holdout_count + background_holdout_count != total_train:
        raise SystemExit("internal count mismatch")
    holdout_assignment_sha256 = hashlib.sha256(g1.canonical_json_bytes({"rule_id": RULE_ID, "holdout": holdout})).hexdigest()
    return {
        "status": "FINAL",
        "scope": "INTERNAL QUALIFICATION ONLY",
        "rule_id": RULE_ID,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rule": (
            f"per class: sort train file names by code point; hold out the first {TARGET_HOLDOUT_PER_CLASS} images of every "
            f"target class and the first {BACKGROUND_HOLDOUT_PER_CLASS} images of every background class. The head trains on the "
            "remaining train images; d' is measured on scenes built only from the holdout. No train image is used for both."
        ),
        "class_split": sp["class_split_sha256"],
        "data_binding": {"train": g1.listing_hash(train_root)},
        "counts": {
            "train_files": total_train,
            "target_holdout_images": target_holdout_count,
            "background_holdout_images": background_holdout_count,
            "holdout_images": target_holdout_count + background_holdout_count,
            "head_training_images": head_count,
            "target_holdout_per_class": TARGET_HOLDOUT_PER_CLASS,
            "background_holdout_per_class": BACKGROUND_HOLDOUT_PER_CLASS,
            "target_head_training_per_class": target_head,
            "background_head_training_per_class": background_head,
        },
        "target_holdout": target_holdout,
        "background_holdout": background_holdout,
        "hashes": {"holdout_assignment_sha256": holdout_assignment_sha256},
        "tool": {
            "file": TOOL,
            "sha256": g1.file_sha256(ROOT / TOOL),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def check(data_root: pathlib.Path, class_split_path: pathlib.Path, path: pathlib.Path) -> None:
    raw = path.read_bytes()
    obj = json.loads(raw)
    if obj.get("status") != "FINAL":
        raise SystemExit(f"{path}: status is {obj.get('status')!r}, not FINAL")
    built = build(data_root, class_split_path)
    for key in ("rule_id", "class_split", "data_binding", "counts", "target_holdout", "background_holdout"):
        if obj.get(key) != built[key]:
            raise SystemExit(f"{path}: {key} does not match the current data tree / class split")
    if obj.get("hashes", {}).get("holdout_assignment_sha256") != built["hashes"]["holdout_assignment_sha256"]:
        raise SystemExit(f"{path}: holdout_assignment_sha256 does not match the holdout lists")
    print(f"OK {path} sha256={hashlib.sha256(raw).hexdigest()} holdout={built['hashes']['holdout_assignment_sha256']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/imagenet100")
    ap.add_argument("--class-split", default="prereg/class_split.json")
    ap.add_argument("--out", default="prereg/gate2_train_holdout.json")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    data_root, class_split, out = pathlib.Path(args.data_root), pathlib.Path(args.class_split), pathlib.Path(args.out)
    if args.check:
        check(data_root, class_split, out)
        return
    obj = build(data_root, class_split)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2) + "\n")
    print(f"wrote {out} sha256={g1.file_sha256(out)} holdout={obj['hashes']['holdout_assignment_sha256']} "
          f"head_training_images={obj['counts']['head_training_images']}")


if __name__ == "__main__":
    main()
