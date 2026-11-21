# Stage 0B — Data and Provenance Binding

**Protocol:** `rhan-stage0b-dinov2-b14-temperature/1` version `1.0.0`  
**Status:** **BOUND — Stage 0B NOT LAUNCHED**  
**Real temperature fitted:** **NO**

---

## 1. Source record

| Field | Value |
|---|---|
| Dataset | ImageNetV2 MatchedFrequency |
| Upstream repository | `vaishaal/ImageNetV2` |
| Pinned revision | `d626240be2538720e83103a0e1178d24aca8b12c` |
| Pinned URL | `https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/d626240be2538720e83103a0e1178d24aca8b12c/imagenetv2-matched-frequency.tar.gz` |
| Local archive | `audit/stage0b_data_gate_v1/work/source/imagenetv2-matched-frequency.tar.gz` |
| Archive size | `1,264,079,360` bytes |
| Archive SHA-256 | `f0c37fdf925916b19ea1323cd9a2208cdb6959ba2c32eef2a7fc393835c9ca7c` |
| Expected size | `1,264,079,360` bytes |
| Expected SHA-256 | `f0c37fdf925916b19ea1323cd9a2208cdb6959ba2c32eef2a7fc393835c9ca7c` |
| Revision evidence | Pinned resolve URL; `HEAD` response `x-repo-commit` equals the pinned revision; downloaded LFS digest equals the pinned digest |
| Container detected | Uncompressed POSIX tar |
| Archive magic | `506178486561` (`PaxHeader`) |
| Extension/container match | **No** — documented source packaging caveat |
| Member integrity | 11,001 members read; 1,244,630,473 regular bytes; 0 size mismatches |
| Extraction | PASS; 10,000 files, 1,001 directories |
| Extracted root | `audit/stage0b_data_gate_v1/work/extracted/imagenetv2-matched-frequency-format-val` |

The `.tar.gz` name is not treated as proof of gzip compression. The pinned bytes, size, and SHA-256 are authoritative, and extraction succeeded. This is recorded as a caveat, not a failure.

---

## 2. Real dataset audit

| Field | Value |
|---|---|
| Audit artifact | `audit/stage0b_data_gate_v1/REAL_DATASET_AUDIT.json` |
| Artifact SHA-256 | `c957ebd2f8fec03d3430fb0a363a24ae32bae4feadb5f70298f058cf55bd2d1b` |
| Status | **PASS** |
| Images | 10,000 |
| Classes | 1,000 |
| Per class | 10 |
| Decoded | 10,000 |
| Decode failures | 0 |
| Formats | JPEG 10,000 |
| Modes | RGB 10,000 |
| Total image bytes | 1,244,630,473 |
| Unexpected files | none |
| Exact duplicate content groups | 0 |
| Distinct dimensions | 1,824 |
| Width range | 68–3,456 |
| Height range | 59–3,872 |
| Symlinks/non-regular entries | rejected by the hardened audit |
| Near-duplicate audit | **NOT PERFORMED**; ImageNetV2 authors' screening is inherited |

Exact duplicate content was checked within the delivered dataset. Near-duplicate overlap with the ImageNet-1k training set was not independently reconstructed. This is a known limitation, not a silent assumption.

---

## 3. Class mapping

Numeric ImageNetV2 folder `f` is parsed as integer `f`. It is **not** mapped by alphabetical `ImageFolder` order.

| Field | Value |
|---|---|
| Mapping artifact | `audit/stage0b_data_gate_v1/CLASS_MAPPING_VERIFICATION.json` |
| Artifact SHA-256 | `271776826d4d797bb01ec11a35400bbeb313bda5503502d658ebbf901248cd5a` |
| Status | **PASS** |
| Numeric folders | 1,000 |
| Canonical ImageNet entries | 1,000 |
| Stage 0A class names | 1,000 |
| Class order matches canonical | yes |
| Stage 0A validation labels | 50,000; 50 per class |
| Class-name SHA-256 | `385e0240499426c022b773be1ae4da780f057b89c9694c7f47199f62746b6edd` |
| Label-array SHA-256 | `bd6a886b061fa64733bc2a4f96e0469b4af8931fdc418a26413b339867535ca8` |
| Mapping SHA-256 | `647b4ce1e14d5f8593355453afe6b4647dc9ced6d3b404b5e713e0b85a79caa7` |
| Canonical index | `audit/stage0b_data_gate_v1/work/meta/imagenet_class_index.json` |
| Canonical index SHA-256 | `b090d218c425be1c056c8cb9dae307b70e349714712013b5b11265bd2cb9d80a` |

---

## 4. Partition

| Field | Value |
|---|---|
| Manifest | `audit/stage0b_data_gate_v1/PARTITION_MANIFEST.json` |
| Schema | `rhan-stage0b-partition-manifest/v1` |
| Status | **PASS** |
| Seed | `20261121` |
| Calibration | 5,000 images; 5 per class |
| Evaluation | 5,000 images; 5 per class |
| Overlap | 0 by path and by content SHA-256 |
| Assignments equal source set | true |
| Every image SHA-256 rematched | true |
| Recomputed twice | true; byte-identical repeats |
| **Canonical manifest SHA-256** | `557e10f1b94c161bc5e57926c137faaa27208410743a0a039b5b43be6da52578` |
| **JSON file SHA-256** | `fa7f23032d6b953664bc573a53d7d90b4d84be3d46fe7faed445e288168135e6` |
| Repeat manifest | `audit/stage0b_data_gate_v1/work/PARTITION_MANIFEST.repeat.json` |

The canonical hash is the hash of the ordered tab-separated assignment rows used by the builder. The JSON file hash is the hash of the serialized manifest. Both are bound so “manifest SHA-256” is no longer ambiguous.

### 4.1 Frozen algorithm

```text
rank = SHA-256(ASCII(seed) || ASCII(WNID) || raw_32_byte_image_content_SHA256)
seed = "20261121"
within each class: sort by rank ascending
positions 0-4  -> calibration
positions 5-9  -> evaluation
```

No filename, path, alphabetical order, or file modification time participates in the ranking. This supersedes the historical path-based proposal.

### 4.2 Immutability

- The calibration/evaluation assignment may not be changed.
- No image may be added or removed.
- No replacement image may be used for a corrupted file.
- Any partition hash mismatch aborts Stage 0B.
- The evaluation split is not used for fitting, selection, implementation tuning, or exploratory analysis.

---

## 5. Rights and handling

Source-code licensing and image-data usage rights are distinct.

Documented operating constraints:

- non-commercial educational/research use only;
- local dataset storage;
- no redistribution of source images;
- no upload of raw images;
- no upload of image-derived features, logits, probabilities, or predictions until their permitted use is established;
- no public Hugging Face dataset repository containing the source data;
- supervisor/institutional confirmation is a launch prerequisite where required.

The ImageNetV2 repository's code is MIT; the images are not MIT and are not redistributable under that code license. No legal assurance beyond the documented terms is made.

Local-only working copies of features, logits, and metrics may be produced during an approved run. They must remain local while artifact rights remain unresolved.

---

## 6. Secondary benchmark

Official ImageNet-1k validation is a secondary, non-independent benchmark:

- it was already evaluated in Stage 0A;
- it may not fit, select, tune, or adjust any temperature;
- it cannot rescue a failed primary decision;
- it may be reported only as “secondary fixed-benchmark replication; not independent”;
- its use is optional and requires a hash-verified pre-existing Stage 0A validation feature cache.

No official-validation image bytes are present locally. Stage 0A image-level replay remains blocked.

---

## 7. Binding summary

The frozen Stage 0B result identity is defined by all of the following together:

1. protocol `rhan-stage0b-dinov2-b14-temperature/1`, version `1.0.0`;
2. the three Stage 0A head hashes in `STAGE0B_PROTOCOL_MANIFEST.json`;
3. the DINOv2 revision and state-dict hash;
4. the ImageNetV2 revision and archive hash;
5. the real-dataset audit artifact hash;
6. the class-mapping artifact hash;
7. both partition hashes;
8. the inference configuration hash;
9. the calibration and evaluation implementation hashes;
10. the runtime fingerprint;
11. the statistical configuration hash.

A result that cannot be tied to all eleven is not a Stage 0B result under this protocol.
