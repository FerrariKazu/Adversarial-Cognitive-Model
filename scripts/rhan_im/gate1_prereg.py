#!/usr/bin/env python3
"""Freeze and evaluate the gate-1 pre-registration (docs/research/RHAN_IM_07_GATE1_PREREG_DRAFT.md).

  freeze     write prereg/rhan_im_gate1_v1.json: the draft thresholds, the planned run configuration, the aggregation rule,
             sha256 of every code file the measurement uses (and of this aggregator), of the checkpoints and of the
             ImageNet-100 file listings, and the git head. Refuses when the git state cannot be read, when the measurement
             code is not committed, and when the primary checkpoint is not the documented ff_112 seed-17 file (unless told
             otherwise on the command line).
  aggregate  read the run JSONs written by gate1_measure_dprime.py, check every run against the frozen file (configuration,
             checkpoint hash, the hashes of the code it ran and of the data listing it read), recompute each run's verdict
             with the FROZEN thresholds (not the ones in the code at analysis time) and apply the frozen aggregation rule.

Fail closed. A protocol deviation in the primary runs (self-test, other checkpoint, other configuration, a missing
configuration key, other code or data, wrong number or set of seeds) makes the aggregate INVALID: the numbers are
reported, the gate is not decided. A required quantity that could not be measured makes it INCONCLUSIVE. Only a run set
without either can PASS or FAIL. Secondary runs are checked the same way; an invalid one is reported and ignored.

Nothing here touches the network or any credential; checkpoints and data are only read.

  python scripts/rhan_im/gate1_prereg.py freeze --primary CKPT_ff112_seed17.pth [--secondary CKPT_42.pth CKPT_99.pth] \
         --data-root data/imagenet100 [--out prereg/rhan_im_gate1_v1.json]
  python scripts/rhan_im/gate1_prereg.py aggregate --prereg prereg/rhan_im_gate1_v1.json \
         --runs run17.json run42.json run99.json [--secondary trunk42.json trunk99.json] [--out verdict.json]
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import pathlib
import statistics
import subprocess
import sys
from typing import Dict, List, Optional

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rhan_im import gate1 as g1  # noqa: E402

# every file whose content can change what the measurement computes, and the aggregator that applies the decision rule
CODE_FILES = list(g1.MEASUREMENT_CODE_FILES) + ["scripts/rhan_im/gate1_prereg.py", "scripts/rhan_im/gate1_slope_heterogeneity.py"]   # C4: the report-only analysis is hashed too
DOCUMENTED_PRIMARY_SHA256 = "d00fb750766b214f69c5f7690772a3373e58b983c6c0cb2420dfd9e558d132bc"   # foundation_ff_112_seed17_final_d00fb750.pth
DATA_KEYS = ("train_path_listing", "val_path_listing")

PLANNED_CONFIG = {
    "canvas": 224, "window": 112, "stride_units": 2, "classes": 50, "sizes": [2, 4, 6], "alphas": [1.0, 0.6], "op_size": 4,
    "p_null": 0.3, "steps": 600, "batch": 48, "lr": 0.002, "n_cal": 3000, "n_test": 6000, "n_pairs": 2000, "chunk_scenes": 200,
}
PRIMARY_SEEDS = [17, 42, 99]
SECONDARY_SEED = 17            # C5: the two secondary checkpoints are measured with this seed; the aggregator now checks it
HEX = set("0123456789abcdef")
AGGREGATION = {
    "n_primary_runs": 3,
    "min_runs_pass_all": 2,
    "min_runs_shared_difficulty": 2,
    "combination_rule": "'sum' only if every primary run says 'sum'; otherwise 'best'",
    "token_evidence": "median over primary runs of vif_null_op (the H_0 block VIF); corrected (z and d divided by sqrt(VIF_null)) iff above vif_independent_max",
    "shared_difficulty": "flagged iff at least min_runs_shared_difficulty primary runs have vif_pos_op / max(vif_null_op, 1) above vif_independent_max; "
                         "never corrected by shrinkage, the remedy is the scale-mixture likelihood with the spread estimated from the VIFs; the median spread is reported; "
                         "a run whose VIFs could not be measured has not assessed it (not the same as absent)",
    "secondary_trunks": "reported only; a secondary trunk that fails marks the result 'trunk-dependent' and restricts the claim to the primary trunk; "
                        "a secondary run that deviates from the frozen protocol is reported as invalid and ignored",
    "invalid_if": "any deviation in the primary runs: self-test, a checkpoint other than the frozen primary, a configuration value other than the frozen one "
                  "or a missing key, code or data listing other than the frozen ones (hashes recorded by each run), a class split or pools file other than the frozen ones, "
                  "a head-initialisation seed other than the run's measurement seed, a missing provenance-sidecar hash, file or runtime record, a provenance linkage hash that is not the hash of the run's own "
                  "protocol fields, a set of seeds other than the frozen one, "
                  "the analysis-time code differing from the frozen hashes; the verdict the rule would have given is reported separately and decides nothing",
    "inconclusive_if": "a primary run could not measure a required quantity (vif_null_op, op_block_tokens, d_native_centre, d_gist and d_gist_full both, null_sd_median, "
                       "zroc_slope_median together with its sd-ratio fallback); one such run is enough, it is not out-voted by the others; re-run, or freeze a new version",
}
NEXT_STEP = {
    "sdt_adequate": "04 section 11 P1: unequal-variance (heteroscedastic) likelihood, or S3; if still outside the band, drop the exact-Bayes claim and keep the risk-control layer",
    "informative": "S3 once (tune the last 2-4 blocks with dense window CE, stop on held-out d', not accuracy), then re-run gate 1 under a NEW pre-registration version; else the switch conditions of 03 section 9",
    "looking_closer_pays": "S3 once as above; if native is no better than gist even then, looking closer does not pay on this trunk: switch conditions of 03 section 9 (temporal looks, fallback)",
    "shared_difficulty": "image-level difficulty is shared across the tokens of a look: use the scale-mixture likelihood (04 section 4, Lemma 1c) in S2 and report E2 stratified by difficulty tercile before any calibrated-confidence claim",
    "invalid": "the runs do not follow the frozen protocol, so the gate is not decided: remove the cause listed under deviations and re-run, or freeze a new version BEFORE looking at any result",
    "inconclusive": "a required quantity was not measured in at least one primary run: re-run those seeds; if it is unmeasurable again the frozen scene counts are too small for it, which is a new protocol version",
}

sha256_file = g1.file_sha256
listing_hash = g1.listing_hash


def git(*args: str, cwd: pathlib.Path = ROOT) -> Optional[str]:
    """stdout of a git command ('' when empty), or None when git could not answer (no repository, no git, failure)."""
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def freeze(primary: pathlib.Path, secondary: List[pathlib.Path], data_root: Optional[pathlib.Path], allow_dirty: bool = False,
           repo_root: pathlib.Path = ROOT, allow_other_primary: bool = False, class_split: Optional[pathlib.Path] = None,
           pools: Optional[pathlib.Path] = None, protocol_version: str = "rhan_im_gate1_v1",
           s3_script: Optional[pathlib.Path] = None, s3_split: Optional[pathlib.Path] = None,
           s3_base_ckpt: Optional[pathlib.Path] = None) -> Dict[str, object]:
    status = git("status", "--porcelain", "--", *CODE_FILES, cwd=repo_root)
    head = git("rev-parse", "HEAD", cwd=repo_root)
    if (status is None or head is None) and not allow_dirty:
        raise SystemExit("refusing to freeze: the git state cannot be read (no repository here, no commit yet, or git is not installed), "
                         "so a clean, committed measurement tree cannot be shown. Run it from a clone of the repository, "
                         "or pass --allow-dirty and accept that the freeze is not tied to a commit")
    if status and not allow_dirty:
        raise SystemExit("refusing to freeze: uncommitted changes in the measurement code:\n" + status + "\n(commit first, or pass --allow-dirty and accept that the freeze is not tied to a commit)")
    if data_root is None:
        raise SystemExit("--data-root is required: the freeze pins the sorted file listing of the train and val splits")
    p_sha = sha256_file(primary)
    if protocol_version == "rhan_im_gate1_v1" and p_sha != DOCUMENTED_PRIMARY_SHA256 and not allow_other_primary:
        raise SystemExit(f"refusing to freeze: the primary checkpoint's sha256 {p_sha[:12]}… is not the documented ff_112 seed-17 file "
                         f"({DOCUMENTED_PRIMARY_SHA256[:12]}…); pass --allow-other-primary if that is deliberate")
    names = [primary.name] + [p.name for p in secondary]
    if len(set(names)) != len(names):
        raise SystemExit("refusing to freeze: checkpoint file names must be distinct (they key the frozen hashes)")
    ckpts = {primary.name: p_sha}
    ckpts.update({p.name: sha256_file(p) for p in secondary})
    data = {f"{split}_path_listing": listing_hash(data_root / split) for split in ("train", "val")}
    if class_split is None or pools is None:
        raise SystemExit("refusing to freeze: --class-split and --pools are required (the freeze pins the class split and the image pools every run must use)")
    try:
        sp = g1.read_class_split(class_split)
        pl = g1.read_pools(pools, sp["class_split_sha256"])
    except (ValueError, KeyError, OSError) as e:
        raise SystemExit(f"refusing to freeze: {e}")
    if pl["val_repo_listing_sha256"] != data["val_path_listing"]["sha256"]:
        raise SystemExit("refusing to freeze: the pools file was made from a different val listing than the one in --data-root")
    s3_provenance = None
    if protocol_version == "rhan_im_gate1_v2":
        if s3_script is None or s3_split is None or s3_base_ckpt is None:
            raise SystemExit("refusing to freeze v2: --s3-script, --s3-split, and --s3-base-ckpt are all required for rhan_im_gate1_v2")
        if not s3_script.is_file():
            raise SystemExit(f"refusing to freeze v2: --s3-script file does not exist: {s3_script}")
        if not s3_split.is_file():
            raise SystemExit(f"refusing to freeze v2: --s3-split file does not exist: {s3_split}")
        if not s3_base_ckpt.is_file():
            raise SystemExit(f"refusing to freeze v2: --s3-base-ckpt file does not exist: {s3_base_ckpt}")
        base_sha = sha256_file(s3_base_ckpt)
        raw_primary = torch.load(primary, map_location="cpu", weights_only=False)
        embedded_base_sha = raw_primary.get("base_checkpoint_sha256") if isinstance(raw_primary, dict) else None
        if embedded_base_sha != base_sha:
            raise SystemExit(
                f"refusing to freeze v2: primary checkpoint's embedded base_checkpoint_sha256 ({embedded_base_sha}) "
                f"does not match --s3-base-ckpt sha256 ({base_sha})"
            )
        s3_provenance = {
            "s3_script": {"file": str(s3_script), "sha256": sha256_file(s3_script)},
            "s3_split": {"file": str(s3_split), "sha256": sha256_file(s3_split)},
            "s3_base_checkpoint": {"file": s3_base_ckpt.name, "sha256": base_sha},
        }
    return {
        "protocol_id": "rhan-im-gate1/1",
        "protocol_version": protocol_version,
        "status": "FROZEN",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo_root),
        "git_head": head,
        "code_dirty_at_freeze": None if status is None else bool(status),         # None: git state unknown (only with --allow-dirty)
        "protocol": {
            "primary_checkpoint": primary.name,
            "primary_is_the_documented_checkpoint": p_sha == DOCUMENTED_PRIMARY_SHA256,
            "secondary_checkpoints": [p.name for p in secondary],
            "primary_seeds": PRIMARY_SEEDS,
            "config": PLANNED_CONFIG,
            "class_split": {"file": class_split.name, "file_sha256": sp["file_sha256"], "class_split_sha256": sp["class_split_sha256"],
                            "class_lists_sha256": sp["class_lists_sha256"], "seed": sp["seed"]},
            "pools": {"file": pools.name, "file_sha256": pl["file_sha256"], "pool_assignment_sha256": pl["pool_assignment_sha256"]},
            "secondary_seed": SECONDARY_SEED,
            "head_init_seed_rule": "config.head_init_seed equals config.seed in every run (explicit, recorded)",
            "original_split_run": "not part of Gate 1 (decision OI-06, 15 Nov 2026): no run on the original sorted-name split is preregistered, measured or aggregated",
            "target_background_split": "the FINAL class-split file pinned in protocol.class_split; every run must report its class_split_sha256 and class_lists_sha256",
            "calibration_test_split": "the pools file pinned in protocol.pools: disjoint halves of the val images (every second file of each class's sorted list); "
                                      "each run checks that its banks draw from exactly those files",
        },
        "rules": {
            "thresholds": {k: list(v) if isinstance(v, tuple) else v for k, v in g1.DEFAULT_THRESHOLDS.items()},
            "aggregation": AGGREGATION,
            "pass_all_means": "sdt_adequate AND informative AND looking_closer_pays at the operating point (size op_size, alpha 1.0)",
            "deviation_policy": "any difference between a run and this file is listed as a deviation; a deviation in the primary runs makes the aggregate verdict INVALID "
                                "(reported, not decided); thresholds may not change after the first run on the real checkpoint",
        },
        "code_and_data_hashes": {**{f: sha256_file(repo_root / f) for f in CODE_FILES}, **data},
        "checkpoint_hashes": ckpts,
        **({"s3_provenance": s3_provenance} if s3_provenance else {}),
    }


SUMMARY_REQUIRED = ("zroc_slope_median", "null_sd_median", "pos_sd_median", "d_native_centre", "d_gist", "rho_null", "vif_null_op", "vif_pos_op", "op_block_tokens")


def _number_or_null(v: object) -> bool:
    return v is None or (isinstance(v, (int, float)) and not isinstance(v, bool))


def run_schema_problems(r: object) -> List[str]:
    """Ways a run JSON does not have the shape gate1_measure_dprime.py writes. A run that cannot be read is not a run: it is a
    deviation (INVALID), never a crash and never a vote."""
    if not isinstance(r, dict):
        return ["the run is not a JSON object"]
    out = [f"'{k}' is missing or not an object" for k in ("config", "trunk", "summary") if not isinstance(r.get(k), dict)]
    if isinstance(r.get("summary"), dict):
        s = r["summary"]
        out += [f"summary lacks {k!r}" for k in SUMMARY_REQUIRED if k not in s]
        out += [f"summary[{k!r}] = {s[k]!r} is neither a number nor null" for k in (*SUMMARY_REQUIRED, "d_gist_full") if k in s and not _number_or_null(s[k])]
    return out


def _unreadable_verdict() -> Dict[str, object]:
    """Stand-in for a run that could not be read: nothing passes and nothing counts as measured or flagged (the aggregate is INVALID anyway)."""
    return {"sdt_adequate": False, "informative": False, "looking_closer_pays": False, "pass_all": False, "combination_rule": "best (run unreadable)",
            "shared_difficulty": False, "shared_difficulty_assessed": False, "unmeasured": []}


def _hex64(x: object) -> bool:
    return isinstance(x, str) and len(x) == 64 and set(x) <= HEX


def run_deviations(r: Dict, prereg: Dict, label: str, expected_sha: Optional[str], what: str, expected_seed: Optional[int] = None) -> List[str]:
    """Every way one run differs from the frozen protocol. A missing piece of evidence counts as a difference: there is no default and no fallback."""
    dev: List[str] = [f"{label}: {p}" for p in run_schema_problems(r)]
    if not isinstance(r, dict):
        return dev
    cfg = r["config"] if isinstance(r.get("config"), dict) else {}
    for k, v in prereg["protocol"]["config"].items():
        if k not in cfg:
            dev.append(f"{label}: config key {k!r} is missing (frozen {v!r})")
        elif cfg[k] != v:
            dev.append(f"{label}: config {k} = {cfg[k]!r}, frozen {v!r}")
    if cfg.get("self_test"):
        dev.append(f"{label}: this is a self-test run (random-init trunk, synthetic data)")
    if expected_seed is not None and cfg.get("seed") != expected_seed:
        dev.append(f"{label}: measurement seed {cfg.get('seed')!r}, frozen {expected_seed!r}")
    hs = cfg.get("head_init_seed")
    if hs is None or isinstance(hs, bool) or hs != cfg.get("seed"):
        dev.append(f"{label}: head_init_seed {hs!r} is not the run's measurement seed {cfg.get('seed')!r}")
    proto = prereg["protocol"]
    cs = r.get("class_split") if isinstance(r.get("class_split"), dict) else {}
    fz = proto.get("class_split")
    if not isinstance(fz, dict) or not isinstance(proto.get("pools"), dict):
        dev.append(f"{label}: the frozen file does not pin a class split and pools")
    else:
        if cs.get("mode") != "class_split_file":
            dev.append(f"{label}: class split mode {cs.get('mode')!r}, frozen 'class_split_file'")
        for k in ("class_split_sha256", "class_lists_sha256"):
            if cs.get(k) != fz.get(k):
                dev.append(f"{label}: {k} {str(cs.get(k))[:12]}… is not the frozen {str(fz.get(k))[:12]}…")
        pl = r.get("pools") if isinstance(r.get("pools"), dict) else {}
        if pl.get("pool_assignment_sha256") != proto["pools"].get("pool_assignment_sha256"):
            dev.append(f"{label}: pool_assignment_sha256 is not the frozen one (or the run used no pools file)")
    prov = r.get("provenance") if isinstance(r.get("provenance"), dict) else {}
    if not (_hex64(prov.get("sidecar_sha256")) and _hex64(prov.get("link_sha256"))):
        dev.append(f"{label}: no per-scene provenance sidecar hash was recorded")
    else:                                                                      # OI-04: the linkage hash must be the one of THIS run's protocol fields
        try:
            plk = r.get("pools") if isinstance(r.get("pools"), dict) else {}
            link = g1.provenance_link(cfg["seed"], cfg["head_init_seed"], (r.get("trunk") or {}).get("sha256"), cs.get("class_lists_sha256"),
                                      plk.get("pool_assignment_sha256"), cfg["n_cal"], cfg["n_test"], min(cfg["n_pairs"], cfg["n_test"]))
            if g1.provenance_link_sha256(link) != prov["link_sha256"]:
                dev.append(f"{label}: the provenance linkage hash is not the hash of this run's seed, head seed, checkpoint, class lists, pools and scene counts")
        except (KeyError, TypeError, ValueError):
            dev.append(f"{label}: the provenance linkage hash could not be recomputed from the run's own fields")
    env = r.get("environment") if isinstance(r.get("environment"), dict) else {}
    if not (env.get("python") and env.get("torch")):
        dev.append(f"{label}: runtime / dependency metadata was not recorded")
    head = r.get("head") if isinstance(r.get("head"), dict) else {}
    if head.get("head_init_seed") != hs or not _hex64(head.get("final_weights_sha256")):
        dev.append(f"{label}: head record missing or inconsistent with config.head_init_seed")
    trunk = r["trunk"] if isinstance(r.get("trunk"), dict) else {}
    if expected_sha is None:
        dev.append(f"{label}: checkpoint {trunk.get('checkpoint')!r} is not in the frozen checkpoint list")
    elif trunk.get("sha256") != expected_sha:
        dev.append(f"{label}: trunk sha256 {str(trunk.get('sha256'))[:12]}… is not {what}")
    frozen = prereg["code_and_data_hashes"]
    ran = r.get("code_sha256")
    if not isinstance(ran, dict):
        dev.append(f"{label}: the run did not record the hashes of the code it ran")
    else:
        for f in g1.MEASUREMENT_CODE_FILES:
            if f in frozen and ran.get(f) != frozen[f]:
                dev.append(f"{label}: ran with a different {f} than the frozen one")
    listing = r.get("data_listing")
    for key in DATA_KEYS:
        if key in frozen and (not isinstance(listing, dict) or listing.get(key) != frozen[key]):
            dev.append(f"{label}: {key} {'was not recorded' if not isinstance(listing, dict) else 'differs from the frozen listing'}")
    return dev


def _seed_of(r: object) -> object:
    return r["config"].get("seed") if (isinstance(r, dict) and isinstance(r.get("config"), dict)) else None


def _ckpt_of(r: object) -> object:
    return r["trunk"].get("checkpoint") if (isinstance(r, dict) and isinstance(r.get("trunk"), dict)) else None


def sidecar_file_problems(r: Dict, run_path: pathlib.Path, label: str) -> List[str]:
    """OI-04 / OI-05: the sidecar file named by the run must sit next to the run JSON and its bytes must hash to the recorded sidecar_sha256
    (a recorded hash without the file is missing provenance)."""
    prov = r.get("provenance") if isinstance(r, dict) and isinstance(r.get("provenance"), dict) else {}
    name = prov.get("file")
    if not isinstance(name, str) or not name or pathlib.PurePath(name).name != name:
        return [f"{label}: the run names no provenance sidecar file next to it"]
    path = pathlib.Path(run_path).with_name(name)
    if not path.is_file():
        return [f"{label}: provenance sidecar file {name} is missing next to the run JSON"]
    if sha256_file(path) != prov.get("sidecar_sha256"):
        return [f"{label}: provenance sidecar file {name} does not hash to the recorded sidecar_sha256"]
    return []


def aggregate(prereg: Dict, runs: List[Dict], secondary: Optional[List[Dict]] = None, repo_root: pathlib.Path = ROOT,
              sidecar_problems: Optional[Dict[str, List[str]]] = None) -> Dict[str, object]:
    """`sidecar_problems`: label -> problems found by sidecar_file_problems() for the run files on disk (the aggregator itself sees only the run JSONs)."""
    sidecar_problems = sidecar_problems or {}
    thr = prereg["rules"]["thresholds"]
    agg = prereg["rules"]["aggregation"]
    primary_name = prereg["protocol"]["primary_checkpoint"]
    frozen = prereg["code_and_data_hashes"]
    deviations: List[str] = []

    for f in CODE_FILES + list(DATA_KEYS):                                           # the frozen file must pin everything this tool checks
        if f not in frozen:
            deviations.append(f"the frozen file does not pin {f} (frozen by an older tool, or without --data-root)")
    for k in ("class_split", "pools", "secondary_seed"):
        if k not in prereg["protocol"]:
            deviations.append(f"the frozen file does not pin protocol.{k}")
    for f in CODE_FILES:                                                             # code at analysis time vs frozen
        p = repo_root / f
        if f in frozen and (not p.exists() or sha256_file(p) != frozen[f]):
            deviations.append(f"code file differs from the frozen hash: {f}")

    per_run = []
    for r in runs:
        readable = not run_schema_problems(r)
        seed = r["config"].get("seed") if (isinstance(r, dict) and isinstance(r.get("config"), dict)) else None
        deviations += run_deviations(r, prereg, f"seed {seed}", prereg["checkpoint_hashes"].get(primary_name), "the frozen primary checkpoint")
        deviations += sidecar_problems.get(f"seed {seed}", [])
        per_run.append({"seed": seed, "verdict": g1.verdict(r["summary"], thr) if readable else _unreadable_verdict(), "summary": r["summary"] if readable else {}})
    seeds, want_seeds = [x["seed"] for x in per_run], list(prereg["protocol"]["primary_seeds"])
    if not (all(isinstance(s, int) and not isinstance(s, bool) for s in seeds) and sorted(seeds) == sorted(want_seeds)):    # integers only: "17" is not 17
        deviations.append(f"primary seeds {seeds}, frozen {want_seeds}")
    if len(per_run) != agg["n_primary_runs"]:
        deviations.append(f"{len(per_run)} primary runs, frozen {agg['n_primary_runs']}")

    n = len(per_run)
    counts = {k: sum(bool(x["verdict"][k]) for x in per_run) for k in ("sdt_adequate", "informative", "looking_closer_pays", "pass_all")}
    need = agg["min_runs_pass_all"]
    failed = [k for k in ("sdt_adequate", "informative", "looking_closer_pays") if counts[k] < need]
    unmeasured = {str(x["seed"]): x["verdict"]["unmeasured"] for x in per_run if x["verdict"]["unmeasured"]}
    vns = [x["summary"].get("vif_null_op") for x in per_run]
    vns = [v for v in vns if v is not None and math.isfinite(v)]
    vif_med = statistics.median(vns) if vns else float("nan")
    correct = bool(vns) and vif_med > thr["vif_independent_max"]
    n_shared = sum(bool(x["verdict"]["shared_difficulty"]) for x in per_run)
    n_assessed = sum(bool(x["verdict"]["shared_difficulty_assessed"]) for x in per_run)
    shared = n_shared >= agg.get("min_runs_shared_difficulty", 2)
    cvs = []
    for x in per_run:
        sm = x["summary"]
        if x["verdict"]["shared_difficulty"] and all(sm.get(k) is not None for k in ("vif_pos_op", "vif_null_op", "op_block_tokens", "d_native_centre")):
            c = g1.difficulty_cv_from_vif(sm["vif_pos_op"], int(round(sm["op_block_tokens"])), sm["d_native_centre"], sm["vif_null_op"])
            if math.isfinite(c):
                cvs.append(c)
    cv_med = statistics.median(cvs) if cvs else float("nan")
    rules = [x["verdict"]["combination_rule"] for x in per_run]
    decided = "PASS" if counts["pass_all"] >= need else "FAIL"
    if deviations:
        gate, steps = "INVALID (PROTOCOL DEVIATIONS)", [NEXT_STEP["invalid"]]
    elif unmeasured:
        gate, steps = "INCONCLUSIVE (A REQUIRED QUANTITY WAS NOT MEASURED)", [NEXT_STEP["inconclusive"]]
    elif counts["pass_all"] >= need:
        gate, steps = "PASS", ["M2 (gate 2) with the combination rule and the token-evidence choice above; every number is re-measured in S2 on a fresh calibration split"]
    else:
        gate = "FAIL"
        steps = [NEXT_STEP[k] for k in failed] or ["inconclusive: fewer runs passed than the frozen rule needs although no single criterion failed in the majority; treat as FAIL"]
    if shared and gate in ("PASS", "FAIL"):
        steps = steps + [NEXT_STEP["shared_difficulty"]]
    out = {
        "gate1": gate, "gate1_if_deviations_and_unmeasured_quantities_were_ignored": decided,
        "runs": n, "pass_counts": counts, "failed_criteria": failed, "unmeasured": unmeasured,
        "combination_rule": "sum" if rules and all(r == "sum" for r in rules) else "best",
        "token_evidence": "correct z and d by 1/sqrt(VIF_null)" if correct else ("independent" if vns else "unmeasured"),
        "vif_null_median": vif_med, "vif_to_apply": vif_med if correct else (1.0 if vns else float("nan")),
        "shared_difficulty": shared, "shared_difficulty_runs": n_shared, "shared_difficulty_assessed_runs": n_assessed, "difficulty_cv_estimate": cv_med,
        "next_step": steps,
        "per_run": per_run, "deviations": deviations,
    }
    if secondary:
        sec = []
        for r in secondary:
            name = r["trunk"].get("checkpoint") if (isinstance(r, dict) and isinstance(r.get("trunk"), dict)) else None
            expected = prereg["checkpoint_hashes"].get(name) if name != primary_name else None
            dev = run_deviations(r, prereg, f"secondary {name}", expected, f"the frozen hash of {name}", expected_seed=prereg["protocol"].get("secondary_seed"))
            dev += sidecar_problems.get(f"secondary {name}", [])
            v = g1.verdict(r["summary"], thr) if not run_schema_problems(r) else _unreadable_verdict()
            sec.append({"checkpoint": name, "valid": not dev and not v["unmeasured"], "deviations": dev, "unmeasured": v["unmeasured"],
                        "pass_all": v["pass_all"], "verdict": v})
        out["secondary_trunks"] = sec
        out["trunk_dependent"] = any(s["valid"] and not s["pass_all"] for s in sec) and counts["pass_all"] >= need
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--primary", required=True, type=pathlib.Path)
    f.add_argument("--secondary", nargs="*", default=[], type=pathlib.Path)
    f.add_argument("--data-root", required=True, type=pathlib.Path)
    f.add_argument("--class-split", required=True, type=pathlib.Path, help="FINAL class-split JSON (gate1_freeze_tools_v2.py split)")
    f.add_argument("--pools", required=True, type=pathlib.Path, help="FINAL pools JSON (gate1_freeze_tools_v2.py pools) for that class split")
    f.add_argument("--allow-dirty", action="store_true")
    f.add_argument("--allow-other-primary", action="store_true", help="the primary checkpoint is not the documented ff_112 seed-17 file")
    f.add_argument("--protocol-version", default="rhan_im_gate1_v1", choices=["rhan_im_gate1_v1", "rhan_im_gate1_v2"],
                   help="protocol version; v2 allows empty secondary list and pins S3 provenance")
    f.add_argument("--s3-script", type=pathlib.Path, default=None, help="(v2 only) S3 training script to pin")
    f.add_argument("--s3-split", type=pathlib.Path, default=None, help="(v2 only) S3 train/heldout partition JSON to pin")
    f.add_argument("--s3-base-ckpt", type=pathlib.Path, default=None, help="(v2 only) base checkpoint that S3 fine-tuned from")
    f.add_argument("--out", type=pathlib.Path, default=ROOT / "prereg" / "rhan_im_gate1_v1.json")
    a = sub.add_parser("aggregate")
    a.add_argument("--prereg", required=True, type=pathlib.Path)
    a.add_argument("--runs", nargs="+", required=True, type=pathlib.Path)
    a.add_argument("--secondary", nargs="*", default=[], type=pathlib.Path)
    a.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()
    if args.cmd == "freeze":
        if args.out.exists():
            raise SystemExit(f"{args.out} exists: a frozen pre-registration is never overwritten (write a new version instead)")
        res = freeze(args.primary, args.secondary, args.data_root, args.allow_dirty, allow_other_primary=args.allow_other_primary,
                     class_split=args.class_split, pools=args.pools, protocol_version=args.protocol_version,
                     s3_script=args.s3_script, s3_split=args.s3_split, s3_base_ckpt=args.s3_base_ckpt)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2) + "\n")
        print("froze", args.out)
        print(json.dumps({k: res[k] for k in ("git_head", "code_dirty_at_freeze", "checkpoint_hashes")}, indent=2))
    else:
        prereg = json.loads(args.prereg.read_text())
        runs = [json.loads(p.read_text()) for p in args.runs]
        sec = [json.loads(p.read_text()) for p in args.secondary]
        problems: Dict[str, List[str]] = {}
        for r, path, label in [(r, p, f"seed {_seed_of(r)}") for r, p in zip(runs, args.runs)] + [(r, p, f"secondary {_ckpt_of(r)}") for r, p in zip(sec, args.secondary)]:
            problems.setdefault(label, []).extend(sidecar_file_problems(r, path, label))
        res = aggregate(prereg, runs, sec, sidecar_problems=problems)
        text = json.dumps(res, indent=2, default=lambda o: None)
        if args.out:
            args.out.write_text(text + "\n")
        print(text)


if __name__ == "__main__":
    main()
