#!/usr/bin/env python3
"""Gate-2 measurement: E2 posterior calibration and E5 LTT risk-controlled stopping.

This script reads the frozen Gate-2 plans, trains the readout head on the complement of the
train holdout, measures the sensor on the holdout, and evaluates the fixed 8-look policy on
the calibration and test pools.  It does not decide the gate; `gate2_aggregate.py` applies the
2-of-3 rule to the three seed runs.

Protocol: prereg/gate2_rules_v1_2.json and docs/research/RHAN_IM_GATE2_ERRATUM_02.md.
No network, no credentials.  The checkpoint is only read.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
import platform
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gen2_foundation.backbone import CompactViTGen2  # noqa: E402
from rhan_im import gate1 as g1  # noqa: E402
from rhan_im import gate2 as g2  # noqa: E402
from rhan_im.layout import UnitLayout  # noqa: E402
from rhan_im.trunk_adapter import ff_token_fn  # noqa: E402


def _load_gate1_script():
    spec = importlib.util.spec_from_file_location("gate1_measure_dprime", ROOT / "scripts" / "rhan_im" / "gate1_measure_dprime.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G1S = _load_gate1_script()


class FileBank:
    def __init__(self, root: pathlib.Path) -> None:
        from PIL import Image

        self.Image = Image
        self.root = pathlib.Path(root)

    def load(self, path: str, size: int) -> torch.Tensor:
        p = pathlib.Path(path)
        p = p if p.is_absolute() else self.root / p
        im = self.Image.open(p).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size), self.Image.BICUBIC)
        return torch.from_numpy(__import__("numpy").asarray(im).copy()).permute(2, 0, 1).float() / 255.0

    def target(self, c: int, size: int, gen: torch.Generator, path: str) -> torch.Tensor:
        return self.load(path, size)

    def background(self, size: int, gen: torch.Generator, path: str) -> torch.Tensor:
        return self.load(path, size)


class ListBank:
    def __init__(self, loader: FileBank, target_classes: Sequence[str], background_classes: Sequence[str],
                 files_by_class: Dict[str, List[str]], seed: int) -> None:
        self.loader = loader
        self.target_classes = list(target_classes)
        self.bg_classes = list(background_classes)
        self.files = files_by_class
        self.n_classes = len(self.target_classes)
        self.rng = random.Random(seed)
        self.last: Optional[Tuple[str, str]] = None

    def target(self, c: int, size: int, gen: torch.Generator) -> torch.Tensor:
        cls = self.target_classes[int(c)]
        path = self.rng.choice(self.files[cls])
        self.last = (cls, pathlib.Path(path).name)
        return self.loader.load(path, size)

    def background(self, size: int, gen: torch.Generator) -> torch.Tensor:
        cls = self.rng.choice(self.bg_classes)
        path = self.rng.choice(self.files[cls])
        self.last = (cls, pathlib.Path(path).name)
        return self.loader.load(path, size)


class SyntheticPlanBank:
    def __init__(self, n_classes: int, seed: int) -> None:
        self.syn = G1S.SyntheticBank(n_classes, seed=seed)
        self.n_classes = n_classes

    def target(self, c: int, size: int, gen: torch.Generator, path: Optional[str] = None) -> torch.Tensor:
        return self.syn.target(int(c), size, gen)

    def background(self, size: int, gen: torch.Generator, path: Optional[str] = None) -> torch.Tensor:
        return self.syn.background(size, gen)


def load_val_split(path: pathlib.Path, class_split_sha256: str, val_listing_sha256: str) -> Dict[str, object]:
    raw = path.read_bytes()
    obj = json.loads(raw)
    if obj.get("status") != "FINAL" or obj.get("class_split") != class_split_sha256:
        raise SystemExit(f"{path}: not a FINAL Gate-2 val split for class_split_sha256 {class_split_sha256[:12]}…")
    if obj.get("listing_binding", {}).get("val", {}).get("repo_listing_sha256") != val_listing_sha256:
        raise SystemExit(f"{path}: the val listing binding does not match the data root")
    for key in ("fit_pool_part_0_of_3", "calibration_pool_part_1_of_3", "test_pool_part_2_of_3"):
        if not isinstance(obj.get(key), dict):
            raise SystemExit(f"{path}: {key} is missing")
    obj["_file_sha256"] = hashlib.sha256(raw).hexdigest()
    return obj


def _prefixed(files_by_class: Dict[str, Sequence[str]], split: str) -> Dict[str, List[str]]:
    return {c: [f"{split}/{c}/{f}" for f in files] for c, files in files_by_class.items()}


def _check_disjoint(pools: Dict[str, Dict[str, Dict[str, Sequence[str]]]]) -> None:
    seen: Dict[Tuple[str, str, str], str] = {}
    for pool, roles in pools.items():
        for role, files in roles.items():
            for cls, names in files.items():
                for name in names:
                    key = (role, cls, name)
                    if key in seen:
                        raise SystemExit(f"image reuse: {role}/{cls}/{name} in both {seen[key]} and {pool}")
                    seen[key] = pool


def _head_training_files(train_root: pathlib.Path, classes: Sequence[str], holdout: Dict[str, Sequence[str]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for c in classes:
        hold = set(holdout.get(c, []))
        files = sorted(p for p in (train_root / c).iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        out[c] = [str(p.relative_to(train_root.parent)) for p in files if p.name not in hold]
        if not out[c]:
            raise SystemExit(f"no head-training images left for class {c!r}")
    return out


def _real_setup(args) -> Dict[str, object]:
    root = pathlib.Path(args.data_root)
    try:
        sp = g1.read_class_split(args.class_split)
        hold = g2.read_train_holdout(args.train_holdout)
    except (ValueError, KeyError, OSError) as e:
        raise SystemExit(str(e))
    if hold.get("class_split") != sp["class_split_sha256"]:
        raise SystemExit("the train holdout is bound to a different class split")
    train_root, val_root = root / "train", root / "val"
    train_listing, val_listing = g1.listing_hash(train_root), g1.listing_hash(val_root)
    if hold.get("counts", {}).get("train_files") != train_listing["n_files"]:
        raise SystemExit("the train holdout was made from a different train listing")
    val = load_val_split(pathlib.Path(args.val_split), sp["class_split_sha256"], val_listing["sha256"])
    tcls, bcls = sp["targets"], sp["backgrounds"]
    holdout_all = {**hold["target_holdout"], **hold["background_holdout"]}
    head_files = _head_training_files(train_root, tcls + bcls, holdout_all)
    for c in tcls + bcls:
        if set(pathlib.Path(p).name for p in head_files[c]) & set(holdout_all[c]):
            raise SystemExit(f"head-training and d'-fit images overlap in class {c!r}")
    pools = {
        "fit": {"background": {c: val["fit_pool_part_0_of_3"][c] for c in bcls}},
        "cal": {"target": {c: val["calibration_pool_part_1_of_3"][c] for c in tcls},
                "background": {c: val["calibration_pool_part_1_of_3"][c] for c in bcls}},
        "test": {"target": {c: val["test_pool_part_2_of_3"][c] for c in tcls},
                 "background": {c: val["test_pool_part_2_of_3"][c] for c in bcls}},
        "dprime": {"target": hold["target_holdout"], "background": hold["background_holdout"]},
    }
    _check_disjoint(pools)
    fit_bg = _prefixed(pools["fit"]["background"], "val")
    cal_t = _prefixed(pools["cal"]["target"], "val")
    cal_b = _prefixed(pools["cal"]["background"], "val")
    test_t = _prefixed(pools["test"]["target"], "val")
    test_b = _prefixed(pools["test"]["background"], "val")
    dp_t = _prefixed(pools["dprime"]["target"], "train")
    dp_b = _prefixed(pools["dprime"]["background"], "train")
    lay = UnitLayout(canvas=args.canvas, patch=14, window=args.window, target_units=args.op_size)
    plans = {
        "fit": g2.build_plan({}, fit_bg, 0, args.fit_null, [args.op_size], lay, args.seed + 101, "fit"),
        "cal": g2.build_plan(cal_t, cal_b, args.n_cal_target, args.n_cal_null, args.sizes, lay, args.seed + 102, "cal"),
        "test": g2.build_plan(test_t, test_b, args.n_test_target, args.n_test_null, args.sizes, lay, args.seed + 103, "test"),
        "dprime": g2.build_plan(dp_t, dp_b, args.n_dprime_target, args.n_dprime_null, [args.op_size], lay, args.seed + 104, "dprime"),
    }
    loader = FileBank(root)
    head_bank = ListBank(loader, tcls, bcls, head_files, seed=args.seed)
    data = {
        "mode": "class_split_file",
        "class_split": {"file": str(args.class_split), "file_sha256": sp["file_sha256"], "class_split_sha256": sp["class_split_sha256"],
                        "class_lists_sha256": sp["class_lists_sha256"], "seed": sp["seed"]},
        "val_split": {"file": str(args.val_split), "file_sha256": val["_file_sha256"],
                      "fit_pool_count": sum(len(v) for v in pools["fit"]["background"].values()),
                      "calibration_pool_count": sum(len(v) for v in pools["cal"]["target"].values()) + sum(len(v) for v in pools["cal"]["background"].values()),
                      "test_pool_count": sum(len(v) for v in pools["test"]["target"].values()) + sum(len(v) for v in pools["test"]["background"].values())},
        "train_holdout": {"file": str(args.train_holdout), "file_sha256": hold["file_sha256"],
                          "holdout_assignment_sha256": hold["holdout_assignment_sha256"], "counts": hold["counts"]},
        "data_listing": {"train_path_listing": train_listing, "val_path_listing": val_listing},
        "head_training_images": sum(len(v) for v in head_files.values()),
    }
    return {"plans": plans, "head_bank": head_bank, "val_bank": loader, "train_bank": loader, "data": data,
            "n_classes": len(tcls), "target_classes": tcls, "background_classes": bcls, "lay": lay}


def _synthetic_setup(args) -> Dict[str, object]:
    C = int(args.classes)
    tcls = [f"t{i:02d}" for i in range(C)]
    bcls = [f"b{i:02d}" for i in range(C)]
    n_files = int(args.synth_files)
    t_files = {c: [f"synth/t/{c}/{j:03d}.png" for j in range(n_files)] for c in tcls}
    b_files = {c: [f"synth/b/{c}/{j:03d}.png" for j in range(n_files)] for c in bcls}
    lay = UnitLayout(canvas=args.canvas, patch=14, window=args.window, target_units=args.op_size)
    plans = {
        "fit": g2.build_plan({}, b_files, 0, args.fit_null, [args.op_size], lay, args.seed + 101, "fit"),
        "cal": g2.build_plan(t_files, b_files, args.n_cal_target, args.n_cal_null, args.sizes, lay, args.seed + 102, "cal"),
        "test": g2.build_plan(t_files, b_files, args.n_test_target, args.n_test_null, args.sizes, lay, args.seed + 103, "test"),
        "dprime": g2.build_plan(t_files, b_files, args.n_dprime_target, args.n_dprime_null, [args.op_size], lay, args.seed + 104, "dprime"),
    }
    return {
        "plans": plans,
        "head_bank": G1S.SyntheticBank(C, seed=args.seed + 1),
        "val_bank": SyntheticPlanBank(C, seed=args.seed + 2),
        "train_bank": SyntheticPlanBank(C, seed=args.seed + 3),
        "data": {"mode": "self_test", "class_split": None, "val_split": None, "train_holdout": None, "data_listing": None,
                 "head_training_images": None},
        "n_classes": C,
        "target_classes": tcls,
        "background_classes": bcls,
        "lay": lay,
    }


def _protocol_guard(args, info: Dict[str, object], setup: Dict[str, object]) -> None:
    """Refuse a non-self-test run that deviates from the frozen v1.2 execution contract."""
    if args.self_test:
        return
    if not args.out:
        raise SystemExit("--out is required for a protocol run (the provenance sidecar must be written)")
    if info.get("sha256") != g2.TRUNK_SHA256:
        raise SystemExit(f"wrong trunk checkpoint: expected sha256 {g2.TRUNK_SHA256}, got {info.get('sha256')}")
    if setup["data"].get("mode") != "class_split_file":
        raise SystemExit("protocol run must use the class-split data path")
    checks = {
        "canvas": 224, "window": 112, "stride_units": 2, "op_size": 4,
        "sizes": [2, 4, 6], "fit_null": 850,
        "n_cal_target": 425, "n_cal_null": 425, "n_test_target": 400, "n_test_null": 400,
        "n_dprime_target": 800, "n_dprime_null": 800,
        "n_cal_subset": 300, "bootstrap": 500, "alpha": 0.10, "delta": 0.10, "k_min": 1, "min_pos": 30,
        "lambdas": list(g2.LTT_LAMBDAS),
        "steps": 600, "batch": 48, "lr": 2e-3,
        "head_sizes": [2, 4, 6], "head_alphas": [1.0, 0.6], "p_null": 0.3,
    }
    for key, expected in checks.items():
        got = getattr(args, key)
        if got != expected:
            raise SystemExit(f"protocol run refuses --{key.replace('_', '-')}={got!r}; expected {expected!r}")
    if int(args.seed) not in (17, 42, 99):
        raise SystemExit(f"protocol run seed must be 17, 42, or 99; got {args.seed}")


def _scores(fn, head, canvas: torch.Tensor, tl: Optional[torch.Tensor], lay: UnitLayout, device: str, chunk: int) -> torch.Tensor:
    tokens = G1S.trunk_tokens(fn, canvas, tl, lay, device, chunk=chunk)
    return head(tokens.to(device)).cpu()


def _collect_scores(plan: Dict[str, object], bank, lay: UnitLayout, cand: torch.Tensor, fn, head, device: str,
                    seed: int, chunk: int, two_windows: bool) -> Dict[str, object]:
    parts = []
    n = int(plan["label"].shape[0])
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        cp = g2.slice_plan(plan, i0, i1)
        sc = g2.make_planned_scenes(cp, bank, lay, seed + i0)
        g = torch.Generator().manual_seed(seed + i0)
        tl = G1S.sample_windows(sc, lay, cand, g)
        native = _scores(fn, head, sc["canvas"], tl, lay, device, chunk)
        gist = _scores(fn, head, sc["canvas"], None, lay, device, chunk)
        cover_native = g1.window_token_overlap(sc["overlap"], tl, lay) > 0
        cover_gist = g1.gist_token_overlap(sc["overlap"], lay) > 0
        part = {"native": native, "gist": gist, "cover_native": cover_native, "cover_gist": cover_gist,
                "label": sc["label"], "tl": tl}
        if two_windows:
            g2gen = torch.Generator().manual_seed(seed + i0 + 10_000)
            tl2 = G1S.sample_windows(sc, lay, cand, g2gen)
            part["native2"] = _scores(fn, head, sc["canvas"], tl2, lay, device, chunk)
            part["tl2"] = tl2
        parts.append(part)
    out = {k: torch.cat([p[k] for p in parts]) for k in ("native", "gist", "cover_native", "cover_gist", "label", "tl")}
    if two_windows:
        out["native2"] = torch.cat([p["native2"] for p in parts])
        out["tl2"] = torch.cat([p["tl2"] for p in parts])
    return out


def _collect_traces(plan: Dict[str, object], bank, lay: UnitLayout, cand: torch.Tensor, fn, head, calib, device: str,
                    mode: str, kappa: float, seed: int, chunk: int) -> Dict[str, torch.Tensor]:
    sensor = g2.Gate2Sensor(fn, head, lay, calib, cand, device)
    parts = []
    n = int(plan["label"].shape[0])
    C = int(head.proj.out_features)
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        cp = g2.slice_plan(plan, i0, i1)
        sc = g2.make_planned_scenes(cp, bank, lay, seed + i0)
        seq = g2.look_sequence(i1 - i0, int(cand.shape[0]), seed, offset=i0)
        z = torch.empty(i1 - i0, g2.K_LOOKS, lay.V, C)
        d = torch.empty(i1 - i0, g2.K_LOOKS, lay.V)
        canvas = sc["canvas"].to(device)
        for t in range(g2.K_LOOKS):
            zt, dt = sensor.look(canvas, seq[:, t])
            z[:, t] = zt.cpu()
            d[:, t] = dt.cpu()
        parts.append(g2.posterior_traces(z, d, sc["eval_size"], sc["label"], lay, mode, kappa, p_null=g2.P_NULL))
    return {k: torch.cat([p[k] for p in parts]) for k in ("conf", "correct", "probs")}


def _sidecar(plans: Dict[str, Dict[str, object]], scores: Dict[str, Dict[str, object]], seed: int, link: Dict[str, object]) -> Dict[str, object]:
    out = {"schema": "rhan-im-gate2-provenance/1", "link": link, "link_sha256": hashlib.sha256(g2.canonical_json_bytes(link)).hexdigest(), "plans": {}}
    for name, plan in plans.items():
        entry = {"plan_sha256": plan["plan_sha256"], **g2.plan_identity(plan)}
        if name in scores:
            entry["native_topleft"] = scores[name]["tl"].tolist()
            if "tl2" in scores[name]:
                entry["native2_topleft"] = scores[name]["tl2"].tolist()
        if name in ("cal", "test"):
            entry["look_sequence"] = g2.look_sequence(int(plan["label"].shape[0]), int(plan["n_native"]) if "n_native" in plan else 25,
                                                       seed + (300 if name == "cal" else 400)).tolist()
        out["plans"][name] = entry
    return out


def runtime_environment(device: str) -> Dict[str, object]:
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__, "cuda_build": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None, "device": device,
           "num_threads": torch.get_num_threads(), "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled())}
    if device.startswith("cuda") and torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(torch.device(device))
    for mod in ("numpy", "PIL"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    return env


def run(args) -> Dict[str, object]:
    t0 = time.time()
    torch.manual_seed(args.seed)
    setup = _real_setup(args) if not args.self_test else _synthetic_setup(args)
    lay, C = setup["lay"], setup["n_classes"]
    cand = lay.candidates(args.stride_units)
    trunk, info = G1S.load_trunk(args.ckpt, args.window, args.device)
    _protocol_guard(args, info, setup)
    fn = ff_token_fn(trunk)
    plans = setup["plans"]
    for name, plan in plans.items():
        n_native = int(cand.shape[0])
        plan["n_native"] = n_native
    gtr = torch.Generator().manual_seed(args.seed + 10)

    def train_batch(step):
        sc = G1S.make_scenes(setup["head_bank"], lay, args.batch, args.head_sizes, args.head_alphas, args.p_null, gtr)
        tl = G1S.sample_windows(sc, lay, cand, gtr)
        tn = G1S.trunk_tokens(fn, sc["canvas"], tl, lay, args.device)
        tg = G1S.trunk_tokens(fn, sc["canvas"], None, lay, args.device)
        cn = g1.window_token_overlap(sc["overlap"], tl, lay)
        cg = g1.gist_token_overlap(sc["overlap"], lay)
        cover = torch.cat([cn >= 0.5, cg >= 0.5])
        weight = torch.cat([torch.ones_like(cn), ((cg == 0) | (cg >= 0.5)).float()])
        return torch.cat([tn, tg]), cover, torch.cat([sc["label"], sc["label"]]), weight

    head, train_log = g1.train_evidence_head(train_batch, 384, C, steps=args.steps, lr=args.lr, seed=args.seed)
    head.eval().to(args.device)
    head_weights_sha256 = hashlib.sha256(torch.cat([p.detach().flatten().cpu() for p in head.parameters()]).numpy().tobytes()).hexdigest()

    fit = _collect_scores(plans["fit"], setup["val_bank"], lay, cand, fn, head, args.device, args.seed + 200, args.chunk_scenes, False)
    dp = _collect_scores(plans["dprime"], setup["train_bank"], lay, cand, fn, head, args.device, args.seed + 210, args.chunk_scenes, True)
    null = dp["label"] < 0
    calib, sensor_report = g2.fit_sensor_from_scores(
        fit["native"], fit["gist"], fit["cover_native"], fit["cover_gist"], fit["label"],
        dp["native"], dp["gist"], dp["cover_native"], dp["cover_gist"], dp["label"],
        dp["native"][null], dp["native2"][null], lay, op_size=args.op_size, min_pos=args.min_pos,
        allow_negative=bool(args.self_test),
    )
    calib.to(args.device)

    cal_traces = _collect_traces(plans["cal"], setup["val_bank"], lay, cand, fn, head, calib, args.device,
                                 sensor_report["combination_mode"], sensor_report["kappa"], args.seed + 300, args.chunk_scenes)
    test_traces = _collect_traces(plans["test"], setup["val_bank"], lay, cand, fn, head, calib, args.device,
                                  sensor_report["combination_mode"], sensor_report["kappa"], args.seed + 400, args.chunk_scenes)
    cal_risk = g2.risk_traces_from_posterior_traces(cal_traces)
    test_risk = g2.risk_traces_from_posterior_traces(test_traces)
    e2 = g2.e2_metrics(test_traces, plans["test"]["label"], plans["test"]["eval_size"])
    e5 = g2.e5_metrics(cal_risk, test_risk, n_cal=args.n_cal_subset, bootstrap=args.bootstrap, alpha=args.alpha,
                       delta=args.delta, k_min=args.k_min, lambdas=args.lambdas, seed=args.seed + 500)

    link = {
        "seed": args.seed,
        "trunk_sha256": info.get("sha256"),
        "class_split_sha256": setup["data"].get("class_split", {}).get("class_split_sha256") if setup["data"].get("class_split") else None,
        "val_split_sha256": setup["data"].get("val_split", {}).get("file_sha256") if setup["data"].get("val_split") else None,
        "train_holdout_sha256": setup["data"].get("train_holdout", {}).get("file_sha256") if setup["data"].get("train_holdout") else None,
        "head_weights_sha256": head_weights_sha256,
        "fit_plan_sha256": plans["fit"]["plan_sha256"],
        "cal_plan_sha256": plans["cal"]["plan_sha256"],
        "test_plan_sha256": plans["test"]["plan_sha256"],
        "dprime_plan_sha256": plans["dprime"]["plan_sha256"],
        "n_fit_null": args.fit_null,
        "n_cal_target": args.n_cal_target,
        "n_cal_null": args.n_cal_null,
        "n_test_target": args.n_test_target,
        "n_test_null": args.n_test_null,
        "n_dprime_target": args.n_dprime_target,
        "n_dprime_null": args.n_dprime_null,
    }
    side = _sidecar(plans, {"fit": fit, "dprime": dp}, args.seed, link)
    side_bytes = g2.canonical_json_bytes(side)
    report = {
        "seed": int(args.seed),
        "protocol": {"protocol_id": "rhan-im-gate2/1.2", "rules_file": "prereg/gate2_rules_v1_2.json",
                     "rules_sha256": g2.file_sha256(ROOT / "prereg/gate2_rules_v1_2.json"),
                     "erratum_file": "docs/research/RHAN_IM_GATE2_ERRATUM_02.md",
                     "erratum_sha256": g2.file_sha256(ROOT / "docs/research/RHAN_IM_GATE2_ERRATUM_02.md")},
        "config": {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()},
        "trunk": info,
        "data": setup["data"],
        "head": {"head_init_seed": args.seed, "final_weights_sha256": head_weights_sha256, "training_log": train_log},
        "sensor": sensor_report,
        "e2": e2,
        "e5": e5,
        "counts": {name: {"n_scenes": int(p["label"].shape[0]), "n_target": int(p["n_target"]), "n_null": int(p["n_null"]),
                          "plan_sha256": p["plan_sha256"]} for name, p in plans.items()},
        "code_sha256": {f: g2.file_sha256(ROOT / f) for f in g2.GATE2_CODE_FILES},
        "environment": runtime_environment(args.device),
        "provenance": {"sidecar_sha256": hashlib.sha256(side_bytes).hexdigest(), "link_sha256": side["link_sha256"], "file": None,
                       "sidecar_bytes": len(side_bytes)},
        "seconds": round(time.time() - t0, 1),
        "_provenance_sidecar": side,
    }
    return report


def _json_safe(x):
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="synthetic textures and random-init trunk; not a protocol run")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data-root", default="data/imagenet100")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--canvas", type=int, default=224)
    ap.add_argument("--window", type=int, default=112)
    ap.add_argument("--stride-units", type=int, default=2)
    ap.add_argument("--classes", type=int, default=50)
    ap.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--op-size", type=int, default=4)
    ap.add_argument("--head-sizes", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--head-alphas", type=float, nargs="+", default=[1.0, 0.6])
    ap.add_argument("--p-null", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--fit-null", type=int, default=850)
    ap.add_argument("--n-cal-target", type=int, default=425)
    ap.add_argument("--n-cal-null", type=int, default=425)
    ap.add_argument("--n-test-target", type=int, default=400)
    ap.add_argument("--n-test-null", type=int, default=400)
    ap.add_argument("--n-dprime-target", type=int, default=800)
    ap.add_argument("--n-dprime-null", type=int, default=800)
    ap.add_argument("--min-pos", type=int, default=30)
    ap.add_argument("--n-cal-subset", type=int, default=300)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--k-min", type=int, default=1)
    ap.add_argument("--lambdas", type=float, nargs="+", default=list(g2.LTT_LAMBDAS))
    ap.add_argument("--chunk-scenes", type=int, default=100)
    ap.add_argument("--synth-files", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--class-split", default="prereg/class_split.json")
    ap.add_argument("--val-split", default="prereg/gate2_val_split.json")
    ap.add_argument("--train-holdout", default="prereg/gate2_train_holdout.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.self_test:
        args.canvas, args.window, args.classes = 112, 56, 6
        args.sizes, args.op_size, args.head_sizes, args.head_alphas = [2, 3], 3, [2, 3], [1.0]
        args.steps, args.batch = 20, 8
        args.fit_null, args.n_cal_target, args.n_cal_null = 120, 30, 30
        args.n_test_target, args.n_test_null = 30, 30
        args.n_dprime_target, args.n_dprime_null = 120, 120
        args.min_pos, args.n_cal_subset, args.bootstrap = 10, 40, 30
        args.chunk_scenes, args.synth_files = 20, 160
        args.seed = 0 if args.seed == 0 else args.seed
    torch.set_num_threads(2)
    res = run(args)
    side = res.pop("_provenance_sidecar")
    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        sp = p.with_name(p.stem + ".provenance.json")
        sp.write_bytes(g2.canonical_json_bytes(side))
        res["provenance"]["file"] = sp.name
        p.write_text(json.dumps(_json_safe(res), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"e2": res["e2"], "e5": res["e5"]["selective"], "sensor": {k: res["sensor"][k] for k in
                     ("d_gist", "vif_tok_raw", "vif_gist_raw", "kappa", "rho_null", "combination_mode")},
                     "seconds": res["seconds"]}, indent=2))


if __name__ == "__main__":
    main()
