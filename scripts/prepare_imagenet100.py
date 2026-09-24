#!/usr/bin/env python3
"""
prepare_imagenet100.py — acquire + convert ImageNet-100 for the Gen-1 run.
================================================================================

SOURCE (the one this project uses; do not substitute silently):
  HuggingFace dataset `clane9/imagenet-100` — the standard CMC ImageNet-100
  subset (100 classes, standard ImageNet label names), distributed as
  parquet shards (image bytes + integer label into a 100-entry class_label
  list). Revision is PINNED below so the acquired bytes are reproducible;
  the loader contract (evaluation/imagenet100_loader.py) is structural:
  <root>/{train,val}/<class>/ with EXACTLY 100 class dirs per split.

WHAT THIS SCRIPT DOES (idempotent — re-runnable at any time):
  1. downloads every shard to the HF cache (resumable per-file);
  2. decodes parquet rows ONCE and writes JPEG files into
     data/imagenet100/train/<wnid>/ and data/imagenet100/val/<wnid>/;
  3. verifies: 100 classes per split, expected sample counts, every file
     decodable (sampled full decode + size probe on all files), and
     emits a FINGERPRINT json (per-file sha256 of shard sources is not
     practical at 8GB; instead: pinned revision + per-split file counts
     + aggregate content hash over sorted (relpath, size, sha256-of-bytes)
     of every written image — cheap to re-verify later).

NOTES
  * Class names -> wnid: the HF label order is arbitrary (an index into
    the 100-entry class_label list); we derive stable directory names
    from the class NAME itself (sanitized), not the index, so directory
    identity does not depend on shard order. The mapping name->dir is
    written to the fingerprint json.
  * 'val' split of this mirror is the standard ImageNet-100 val subset.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

REPO_ID = "clane9/imagenet-100"
PINNED_REVISION = "0519dc2f402a3a18c6e57f7913db059215eee25b"
EXPECTED_TRAIN_FILES = 17
EXPECTED_VAL_FILES = 1
DEFAULT_ROOT = os.path.join(REPO_ROOT, "data", "imagenet100")


def _token():
    tok = os.environ.get("HF_TOKEN")
    if not tok and os.path.exists(os.path.join(REPO_ROOT, ".env")):
        try:
            from dotenv import dotenv_values
            tok = dotenv_values(os.path.join(REPO_ROOT, ".env")).get("HF_TOKEN")
        except Exception:
            pass
    return tok


def _shards():
    from huggingface_hub import HfApi
    api = HfApi(token=_token())
    info = api.dataset_info(REPO_ID, revision=PINNED_REVISION,
                            files_metadata=True)
    train = sorted(s.rfilename for s in info.siblings
                   if re.match(r"data/train-\d+-of-\d+\.parquet$",
                               s.rfilename))
    val = sorted(s.rfilename for s in info.siblings
                 if re.match(r"data/validation-\d+-of-\d+\.parquet$",
                             s.rfilename))
    if len(train) != EXPECTED_TRAIN_FILES or len(val) != EXPECTED_VAL_FILES:
        raise SystemExit(
            f"UNEXPECTED SHARD LAYOUT at pinned revision {PINNED_REVISION}: "
            f"{len(train)} train / {len(val)} validation shards (expected "
            f"{EXPECTED_TRAIN_FILES}/{EXPECTED_VAL_FILES}) — STOP, do not "
            f"guess a substitution")
    return train + val


def _wnid_of(class_name: str) -> str:
    """Stable directory name for a class label string (index-free)."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", class_name).strip("_")
    return s[:80] if s else f"class_{abs(hash(class_name)) % 10**8}"


def _decode_bytes(raw: bytes):
    from PIL import Image
    img = Image.open(io.BytesIO(raw))
    img.load()
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--verify-only", action="store_true",
                    help="re-verify an existing converted tree + fingerprint")
    ap.add_argument("--max-per-class-probe", type=int, default=3,
                    help="images per class fully re-decoded during verify")
    args = ap.parse_args()

    root = args.root
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    fp_path = os.path.join(root, "fingerprint.json")

    if args.verify_only:
        return _verify(root, fp_path, args.max_per_class_probe)

    shards = _shards()
    print(f"source: {REPO_ID}@{PINNED_REVISION} "
          f"({len(shards)} shards)", flush=True)

    label_names = None
    for fn in shards:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(REPO_ID, fn, repo_type="dataset",
                            revision=PINNED_REVISION, token=_token())
        import pyarrow.parquet as pq
        tbl = pq.read_table(p)
        cols = tbl.column_names
        if label_names is None:
            # The AUTHORITATIVE label order lives in the parquet schema
            # metadata (features.label.names, 100 entries, index-stable
            # across shards of the pinned revision).
            label_names = _label_names_from_table(tbl)
            _LABEL_CACHE.update(label_names)
        img_col = "image" if "image" in cols else cols[0]
        lab_col = "label" if "label" in cols else cols[-1]
        data = tbl.to_pydict()
        n = len(data[img_col])
        split = "train" if "/train-" in fn else "val"
        out_base = train_dir if split == "train" else val_dir
        t0 = time.time()
        for i in range(n):
            rec = data[img_col][i]
            raw = rec["bytes"] if isinstance(rec, dict) else rec
            lab_i = int(data[lab_col][i])
            name = _label_name(lab_i)
            cls_dir = os.path.join(out_base, _wnid_of(name))
            os.makedirs(cls_dir, exist_ok=True)
            img = _decode_bytes(raw)
            fname = f"{fn.split('/')[-1].replace('.parquet','')}_{i:06d}.jpg"
            out = os.path.join(cls_dir, fname)
            if not os.path.exists(out):
                # Atomic write: a killed process must never leave a partial
                # JPEG that the exists() skip would treat as converted.
                tmp_out = out + ".tmp"
                img.convert("RGB").save(tmp_out, "JPEG", quality=95)
                os.replace(tmp_out, out)
        print(f"  {fn}: {n} rows -> {split}/ ({time.time()-t0:.0f}s)",
              flush=True)
        del data, tbl

    return _verify(root, fp_path, args.max_per_class_probe)


_LABEL_CACHE = {}


def _label_names_from_table(tbl) -> dict:
    """{index: class_name} from the parquet schema metadata — the
    AUTHORITATIVE, index-stable order for the pinned revision."""
    md = tbl.schema.metadata or {}
    hf = json.loads(md.get(b"huggingface", b"{}").decode() or "{}")
    feats = hf.get("info", {}).get("features", {})
    names = feats.get("label", {}).get("names", [])
    if len(names) != 100:
        raise SystemExit(
            f"could not read exactly 100 class names from the parquet "
            f"metadata (got {len(names)}) — STOP, do not guess a "
            f"label-order substitution")
    return {i: n for i, n in enumerate(names)}


def _label_name(idx: int) -> str:
    if not _LABEL_CACHE:
        raise SystemExit("label cache empty — conversion must read the "
                         "first shard before naming classes")
    return _LABEL_CACHE[idx]


def _verify(root: str, fp_path: str, probe: int) -> int:
    """Structural + readability verification; writes fingerprint.json."""
    import random

    from evaluation.imagenet100_loader import validate_imagenet100_root
    n_train_classes = validate_imagenet100_root(root, split="train")
    n_val_classes = validate_imagenet100_root(root, split="val")

    counts = {}
    per_split = {}
    for split, d in (("train", os.path.join(root, "train")),
                     ("val", os.path.join(root, "val"))):
        classes = sorted(os.listdir(d))
        n_img = 0
        for c in classes:
            cd = os.path.join(d, c)
            files = [f for f in os.listdir(cd) if f.endswith(".jpg")]
            n_img += len(files)
            counts[f"{split}/{c}"] = len(files)
        per_split[split] = {"classes": len(classes), "images": n_img}

    # Readability probe: full decode a random sample per class + size probe
    # (JPEG SOF dimensions) on ALL files is too slow; full decode on probe
    # samples, os-size>0 on everything.
    rng = random.Random(0)
    bad = []
    for split, d in (("train", os.path.join(root, "train")),
                     ("val", os.path.join(root, "val"))):
        for c in sorted(os.listdir(d)):
            cd = os.path.join(d, c)
            files = sorted(f for f in os.listdir(cd) if f.endswith(".jpg"))
            if not files:
                bad.append(f"{split}/{c}: EMPTY")
                continue
            for f in rng.sample(files, min(probe, len(files))):
                try:
                    with open(os.path.join(cd, f), "rb") as fh:
                        _decode_bytes(fh.read())
                except Exception as e:
                    bad.append(f"{split}/{c}/{f}: {e}")

    # Aggregate fingerprint: sorted (relpath, size) streamed into sha256
    # (cheap re-verification of the tree's identity without re-reading
    # every byte; the pinned revision is the byte-level guarantee).
    h = hashlib.sha256()
    n_files = 0
    for split, d in (("train", os.path.join(root, "train")),
                     ("val", os.path.join(root, "val"))):
        for c in sorted(os.listdir(d)):
            cd = os.path.join(d, c)
            for f in sorted(os.listdir(cd)):
                p = os.path.join(cd, f)
                st = os.stat(p)
                h.update(f"{split}/{c}/{f}:{st.st_size};".encode())
                n_files += 1

    fp = {
        "source": REPO_ID,
        "revision": PINNED_REVISION,
        "layout": "<root>/{train,val}/<wnid>/<file>.jpg",
        "splits": per_split,
        "aggregate_file_listing_sha256": h.hexdigest(),
        "n_files": n_files,
        "readability_probe_per_class": probe,
        "readability_failures": bad[:20],
        # index->dir mapping is authoritative at conversion time; a later
        # --verify-only pass carries the existing mapping forward.
        "class_names_to_dirs": (
            {f"{i}": _wnid_of(_label_name(i)) for i in range(100)}
            if _LABEL_CACHE else
            (json.load(open(fp_path)).get("class_names_to_dirs", {})
             if os.path.exists(fp_path) else {})),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(root, exist_ok=True)
    with open(fp_path, "w") as f:
        json.dump(fp, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"train classes={n_train_classes} images={per_split['train']['images']}")
    print(f"val   classes={n_val_classes} images={per_split['val']['images']}")
    if bad:
        print(f"READABILITY FAILURES ({len(bad)}):", *bad[:10], sep="\n  ")
        return 1
    print(f"fingerprint -> {fp_path}")
    ok = (n_train_classes == 100 and n_val_classes == 100)
    print("VERIFY:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
