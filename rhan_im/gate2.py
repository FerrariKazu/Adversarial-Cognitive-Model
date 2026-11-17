"""Gate-2 measurement: posterior calibration (E2) and LTT risk-controlled stopping (E5).

This module is the pure part of Gate 2.  It builds no-reuse scene plans, turns measured
scores into the Gate-2 sensor calibration, runs the fixed look sequence through the Bayes
state, and computes E2/E5 metrics.  The script that reads images, trains the head, and runs
the trunk is `scripts/rhan_im/gate2_measure.py`; the aggregator is
`scripts/rhan_im/gate2_aggregate.py`.

Protocol: `prereg/gate2_rules_v1_2.json` and `docs/research/RHAN_IM_GATE2_ERRATUM_02.md`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from rhan_im import gate1 as g1
from rhan_im.belief import SearchPosterior
from rhan_im.layout import UnitLayout
from rhan_im.stopping import ltt_fixed_sequence, naive_plugin_choice, stopped_stats, stopped_stats_selective
from rhan_im.trunk_adapter import native_crop
from rhan_im.window_head import EvidenceCalibration

K_LOOKS = 8
P_NULL = 0.5
PROTOCOL_ID = "rhan-im-gate2/1.2"
TRUNK_SHA256 = "d00fb750766b214f69c5f7690772a3373e58b983c6c0cb2420dfd9e558d132bc"
TARGET_SIZES = (2, 4, 6)
OP_SIZE = 4
OP_ALPHA = 1.0
VIF_CORRECTION_MIN = 1.25
RHO_SUM_MAX = 0.2
KAPPA_QUANTILE = 0.999
LTT_ALPHA = 0.10
LTT_DELTA = 0.10
LTT_N_CAL = 300
LTT_BOOTSTRAP = 500
LTT_K_MIN = 1
LTT_LAMBDAS = (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50)
E2_PASS = 0.03
E2_INCONCLUSIVE = 0.05
E5_EFFICIENCY_MIN = 0.25

GATE2_CODE_FILES = (
    "scripts/rhan_im/gate2_measure.py",
    "scripts/rhan_im/gate2_split.py",
    "rhan_im/gate2.py",
    "rhan_im/gate1.py",
    "rhan_im/layout.py",
    "rhan_im/trunk_adapter.py",
    "rhan_im/window_head.py",
    "rhan_im/sdt.py",
    "rhan_im/belief.py",
    "rhan_im/stopping.py",
    "gen2_foundation/backbone.py",
    "noesis_vision/models/recurrent_block.py",
    "noesis_vision/models/foveation.py",
)
AGGREGATION_CODE_FILES = GATE2_CODE_FILES + ("scripts/rhan_im/gate2_aggregate.py",)


def file_sha256(path) -> str:
    return g1.file_sha256(path)


def canonical_json_bytes(obj) -> bytes:
    return g1.canonical_json_bytes(obj)


def _finite(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def read_train_holdout(path) -> Dict[str, object]:
    raw = pathlib.Path(path).read_bytes()
    obj = json.loads(raw)
    if obj.get("status") != "FINAL" or obj.get("rule_id") != "train-prefix-holdout-v1":
        raise ValueError(f"{path}: not a FINAL train-prefix-holdout-v1 file")
    target, background = obj.get("target_holdout"), obj.get("background_holdout")
    if not isinstance(target, dict) or not isinstance(background, dict):
        raise ValueError(f"{path}: target_holdout / background_holdout are missing")
    if set(target) & set(background):
        raise ValueError(f"{path}: a class is in both holdout roles")
    if len(target) != 50 or len(background) != 50:
        raise ValueError(f"{path}: expected 50 target and 50 background classes, got {len(target)} / {len(background)}")
    for c, files in {**target, **background}.items():
        if not isinstance(files, list) or len(set(files)) != len(files) or not files:
            raise ValueError(f"{path}: holdout list for {c!r} is empty or has duplicates")
    assignment = hashlib.sha256(canonical_json_bytes({"rule_id": obj["rule_id"], "holdout": {**target, **background}})).hexdigest()
    if obj.get("hashes", {}).get("holdout_assignment_sha256") != assignment:
        raise ValueError(f"{path}: holdout_assignment_sha256 does not match the holdout lists")
    return {
        "target_holdout": target,
        "background_holdout": background,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "holdout_assignment_sha256": assignment,
        "class_split": obj.get("class_split"),
        "counts": obj.get("counts", {}),
    }


def round_robin(files_by_class: Dict[str, Sequence[str]], n: int, class_order: Sequence[str]) -> List[Tuple[str, str]]:
    """Take `n` (class, file) pairs, cycling classes so every class is represented before any repeats."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return []
    order = list(class_order)
    if not order:
        raise ValueError("class_order is empty")
    cursors = {c: 0 for c in order}
    out: List[Tuple[str, str]] = []
    while len(out) < n:
        progressed = False
        for c in order:
            files = files_by_class.get(c, ())
            i = cursors[c]
            if i < len(files):
                out.append((c, files[i]))
                cursors[c] = i + 1
                progressed = True
                if len(out) == n:
                    break
        if not progressed:
            raise ValueError(f"not enough files for {n} draws from {len(order)} classes")
    return out


def build_plan(
    target_files: Dict[str, Sequence[str]],
    background_files: Dict[str, Sequence[str]],
    n_target: int,
    n_null: int,
    sizes: Sequence[int],
    lay: UnitLayout,
    seed: int,
    name: str,
) -> Dict[str, object]:
    """Build a deterministic no-reuse scene plan for one pool.

    target_files / background_files are class folder -> sorted file names.  A target scene
    consumes one target and one background image; a null scene consumes one background image.
    Classes are cycled round-robin, so the plan is balanced even when the class counts differ.
    """
    sizes = list(sizes)
    if not sizes or any(int(s) < 1 or int(s) > lay.g for s in sizes):
        raise ValueError("sizes must be non-empty and lie inside the canvas")
    target_classes = sorted(target_files)
    background_classes = sorted(background_files)
    if n_target and not target_classes:
        raise ValueError("target scenes requested but target_files is empty")
    if not background_classes:
        raise ValueError("background_files is empty")
    t_pairs = round_robin(target_files, n_target, target_classes) if n_target else []
    b_pairs = round_robin(background_files, n_target + n_null, background_classes)
    n = n_target + n_null
    label = torch.full((n,), -1, dtype=torch.long)
    size_units = torch.zeros(n, dtype=torch.long)
    alpha = torch.full((n,), float(OP_ALPHA))
    top = torch.zeros(n, dtype=torch.long)
    left = torch.zeros(n, dtype=torch.long)
    target_class: List[Optional[str]] = [None] * n
    target_file: List[Optional[str]] = [None] * n
    background_class: List[str] = [""] * n
    background_file: List[str] = [""] * n
    t_index = {c: i for i, c in enumerate(target_classes)}
    for i, (c, f) in enumerate(t_pairs):
        label[i] = t_index[c]
        target_class[i] = c
        target_file[i] = f
        size_units[i] = int(sizes[i % len(sizes)])
    for j, (c, f) in enumerate(b_pairs):
        i = j
        background_class[i] = c
        background_file[i] = f
        if j >= n_target:
            size_units[i] = int(sizes[(j - n_target) % len(sizes)])
    g = torch.Generator().manual_seed(int(seed) * 1_000_003 + sum(ord(ch) for ch in name))
    for i in range(n):
        span = lay.g - int(size_units[i]) + 1
        top[i] = int(torch.randint(0, span, (1,), generator=g).item()) * lay.patch
        left[i] = int(torch.randint(0, span, (1,), generator=g).item()) * lay.patch
    overlap = torch.zeros(n, lay.g, lay.g)
    for s in sorted(set(int(x) for x in size_units.tolist())):
        idx = ((label >= 0) & (size_units == s)).nonzero().squeeze(1)
        if idx.numel():
            overlap[idx] = g1.unit_overlap(top[idx], left[idx], s * lay.patch, lay.g, lay.patch)
    plan = {
        "name": name,
        "seed": int(seed),
        "target_classes": target_classes,
        "background_classes": background_classes,
        "label": label,
        "size_units": size_units,
        "eval_size": size_units.clone(),
        "alpha": alpha,
        "top": top,
        "left": left,
        "overlap": overlap,
        "target_class": target_class,
        "target_file": target_file,
        "background_class": background_class,
        "background_file": background_file,
        "n_target": int(n_target),
        "n_null": int(n_null),
    }
    plan["plan_sha256"] = plan_hash(plan)
    return plan


def plan_identity(plan: Dict[str, object]) -> Dict[str, object]:
    keys = ("name", "seed", "target_classes", "background_classes", "label", "size_units", "eval_size", "alpha", "top", "left",
            "target_class", "target_file", "background_class", "background_file", "n_target", "n_null")
    out = {}
    for k in keys:
        v = plan[k]
        out[k] = v.tolist() if torch.is_tensor(v) else v
    return out


def plan_hash(plan: Dict[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(plan_identity(plan))).hexdigest()


def slice_plan(plan: Dict[str, object], i0: int, i1: int) -> Dict[str, object]:
    n = int(plan["label"].shape[0])
    i0, i1 = max(0, int(i0)), min(n, int(i1))
    out = dict(plan)
    for k in ("label", "size_units", "eval_size", "alpha", "top", "left", "overlap"):
        out[k] = plan[k][i0:i1]
    for k in ("target_class", "target_file", "background_class", "background_file"):
        out[k] = plan[k][i0:i1]
    out["n_target"] = int((out["label"] >= 0).sum())
    out["n_null"] = int((out["label"] < 0).sum())
    out["slice"] = [i0, i1]
    return out


def make_planned_scenes(plan: Dict[str, object], bank, lay: UnitLayout, seed: int) -> Dict[str, object]:
    """Render a plan.  bank.target(c, size_px, gen, path) and bank.background(size_px, gen, path) load pixels."""
    n = int(plan["label"].shape[0])
    g = torch.Generator().manual_seed(int(seed))
    bgs = [bank.background(lay.canvas, g, plan["background_file"][i]) for i in range(n)]
    canvas = torch.stack(bgs)
    for s in sorted(set(int(x) for x in plan["size_units"].tolist())):
        idx = ((plan["label"] >= 0) & (plan["size_units"] == s)).nonzero().squeeze(1)
        if not idx.numel():
            continue
        patches = torch.stack([bank.target(int(plan["label"][int(i)]), s * lay.patch, g, plan["target_file"][int(i)]) for i in idx])
        canvas[idx] = g1.paste_squares(canvas[idx], patches, plan["top"][idx], plan["left"][idx], plan["alpha"][idx])
    return {
        "canvas": canvas,
        "label": plan["label"],
        "size_units": plan["size_units"],
        "eval_size": plan["eval_size"],
        "alpha": plan["alpha"],
        "top": plan["top"],
        "left": plan["left"],
        "overlap": plan["overlap"],
        "bg_src": list(zip(plan["background_class"], plan["background_file"])),
        "tg_src": [None if plan["target_class"][i] is None else (plan["target_class"][i], plan["target_file"][i]) for i in range(n)],
    }


def look_sequence(n: int, n_native: int, seed: int, offset: int = 0) -> torch.Tensor:
    """(n, K) candidate indices: column 0 is the gist (index n_native), the rest are seeded native draws."""
    if n_native < 1:
        raise ValueError("n_native must be positive")
    out = torch.empty(n, K_LOOKS, dtype=torch.long)
    out[:, 0] = n_native
    for i in range(n):
        g = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(offset) + i)
        out[i, 1:] = torch.randint(0, n_native, (K_LOOKS - 1,), generator=g)
    return out


class Gate2Sensor:
    """Device-safe version of WindowSensor for the fixed Gate-2 look policy.

    WindowSensor (rhan_im.trunk_adapter) builds its visibility matrix on the CPU and therefore
    assumes CPU look indices.  Gate 2 runs the trunk on CUDA in the real measurement, so this
    class keeps the same algebra with every sensor tensor on `device`.
    """

    def __init__(self, trunk_fn, head, lay: UnitLayout, calib: EvidenceCalibration, candidates: torch.Tensor, device) -> None:
        self.trunk_fn = trunk_fn
        self.head = head
        self.lay = lay
        self.calib = calib
        self.device = torch.device(device)
        self.candidates = candidates.to(self.device)
        self.n_native = int(candidates.shape[0])
        self.n_candidates = self.n_native + 1
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / self._std

    def vis(self) -> torch.Tensor:
        K, lay = self.n_native, self.lay
        vis = torch.zeros(K + 1, lay.V, device=self.device)
        idx, valid = lay.window_units(self.candidates)
        vals = self.calib.d_tok_eff.reshape(1, -1).expand(K, -1) * valid.to(self.calib.d_tok_eff.dtype)
        vis[:K].scatter_add_(1, idx, vals)
        vis[K, lay.n_fine:] = self.calib.d_gist_eff
        return vis

    def look(self, canvas: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lay, B = self.lay, canvas.shape[0]
        C = self.head.proj.out_features
        z = torch.zeros(B, lay.V, C, dtype=canvas.dtype, device=self.device)
        k = k.to(self.device)
        is_gist = k == self.n_native
        win = ~is_gist
        if bool(win.any()):
            tl = self.candidates[k[win]]
            crop = native_crop(canvas[win], tl * lay.patch, lay.window)
            _, tokens = self.trunk_fn(self._normalise(crop))
            s = self.calib.standardize(self.head(tokens), 0)
            idx, valid = lay.window_units(tl)
            s = s * valid[..., None].to(s.dtype)
            zw = torch.zeros(int(win.sum()), lay.V, C, dtype=s.dtype, device=self.device)
            zw.scatter_add_(1, idx[..., None].expand(-1, -1, C), s)
            z[win] = zw
        if bool(is_gist.any()):
            small = F.interpolate(canvas[is_gist], size=(lay.window, lay.window), mode="area")
            _, tokens = self.trunk_fn(self._normalise(small))
            z[is_gist, lay.n_fine:] = self.calib.standardize(self.head(tokens), 1)
        return z, self.vis()[k]


def fit_sensor_from_scores(
    fit_native: torch.Tensor,
    fit_gist: torch.Tensor,
    fit_cover_native: torch.Tensor,
    fit_cover_gist: torch.Tensor,
    fit_label: torch.Tensor,
    d_native: torch.Tensor,
    d_gist: torch.Tensor,
    d_cover_native: torch.Tensor,
    d_cover_gist: torch.Tensor,
    d_label: torch.Tensor,
    d_null_native1: torch.Tensor,
    d_null_native2: torch.Tensor,
    lay: UnitLayout,
    op_size: int = OP_SIZE,
    min_pos: int = 30,
    allow_negative: bool = False,
) -> Tuple[EvidenceCalibration, Dict[str, object]]:
    """Fit the Gate-2 sensor from fit-pool null scores and train-holdout d' scores."""
    C = int(fit_native.shape[2])
    mu_n, sd_n = g1.fit_standardizer(fit_native, fit_cover_native)
    mu_g, sd_g = g1.fit_standardizer(fit_gist, fit_cover_gist)
    mn = g1.measure_tokens(d_native, d_cover_native, d_label, mu_n, sd_n, lay.tok, min_pos=min_pos)
    mg = g1.measure_tokens(d_gist, d_cover_gist, d_label, mu_g, sd_g, lay.tok, min_pos=min_pos)
    d_tok = mn["d_shift"]
    d_gist_map = mg["d_shift"]
    d_gist_mean = g1.region_means(d_gist_map, mg["n_pos"])["all"]
    if not bool(torch.isfinite(d_tok).all()):
        raise ValueError("d_tok has unmeasured token positions")
    if not _finite(d_gist_mean):
        raise ValueError("d_gist is unmeasured")
    negative_tok = int((d_tok < 0).sum())
    negative_gist = bool(float(d_gist_mean) < 0)
    if (negative_tok or negative_gist) and not allow_negative:
        raise ValueError(f"negative detectability: {negative_tok} token positions, d_gist={float(d_gist_mean):.4f}")
    if allow_negative:
        d_tok = d_tok.clamp_min(0.0)
        d_gist_mean = max(0.0, float(d_gist_mean))
    z_n = (fit_native - mu_n) / sd_n
    z_g = (fit_gist - mu_g) / sd_g
    vif_tok_raw = g1.block_vif(z_n, fit_cover_native, fit_label, lay.tok, block=int(op_size), weights=d_tok)["vif_null"]
    vif_gist_raw = g1.block_vif(z_g, fit_cover_gist, fit_label, lay.tok, block=max(1, int(op_size) // lay.coarse_block),
                                weights=d_gist_map, min_pos=min_pos)["vif_null"]
    vif_tok = float(vif_tok_raw) if _finite(vif_tok_raw) and float(vif_tok_raw) > VIF_CORRECTION_MIN else 1.0
    vif_gist = float(vif_gist_raw) if _finite(vif_gist_raw) and float(vif_gist_raw) > VIF_CORRECTION_MIN else 1.0
    z_n_corr = z_n / math.sqrt(vif_tok)
    z_g_corr = z_g / math.sqrt(vif_gist)
    kappa = float(torch.quantile(torch.cat([z_n_corr.flatten(), z_g_corr.flatten()]).abs(), KAPPA_QUANTILE))
    if not _finite(kappa) or kappa <= 0:
        raise ValueError("kappa is unmeasured")
    rho_null = g1.paired_noise_correlation((d_null_native1 - mu_n) / sd_n, (d_null_native2 - mu_n) / sd_n)
    if not _finite(rho_null):
        raise ValueError("rho_null is unmeasured")
    mode = "sum" if float(rho_null) <= RHO_SUM_MAX else "best"
    calib = EvidenceCalibration(C, lay.tok)
    with torch.no_grad():
        calib.mu0[0].copy_(mu_n)
        calib.mu0[1].copy_(mu_g)
        calib.sd0[0].copy_(sd_n)
        calib.sd0[1].copy_(sd_g)
        calib.d_tok.copy_(d_tok)
        calib.d_gist.fill_(float(d_gist_mean))
        calib.vif_tok.fill_(vif_tok)
        calib.vif_gist.fill_(vif_gist)
    report = {
        "n_fit_scenes": int(fit_native.shape[0]),
        "n_dprime_target_scenes": int((d_label >= 0).sum()),
        "n_dprime_null_scenes": int((d_label < 0).sum()),
        "op_size_units": int(op_size),
        "op_alpha": float(OP_ALPHA),
        "d_tok": d_tok.tolist(),
        "d_tok_regions": g1.region_means(d_tok, mn["n_pos"]),
        "d_tok_n_pos": mn["n_pos"].tolist(),
        "d_gist": float(d_gist_mean),
        "d_gist_raw": float(g1.region_means(d_gist_map, mg["n_pos"])["all"]),
        "d_tok_negative_positions": negative_tok,
        "d_gist_negative": negative_gist,
        "d_gist_map": d_gist_map.tolist(),
        "d_gist_n_pos": mg["n_pos"].tolist(),
        "vif_tok_raw": float(vif_tok_raw) if _finite(vif_tok_raw) else None,
        "vif_gist_raw": float(vif_gist_raw) if _finite(vif_gist_raw) else None,
        "vif_tok_used": vif_tok,
        "vif_gist_used": vif_gist,
        "vif_correction_min": VIF_CORRECTION_MIN,
        "kappa": kappa,
        "kappa_quantile": KAPPA_QUANTILE,
        "rho_null": float(rho_null),
        "combination_mode": mode,
        "min_pos": int(min_pos),
        "unit_cap": None,
    }
    return calib, report


def true_outcome(labels: torch.Tensor) -> torch.Tensor:
    out = labels.clone().long() + 1
    out[labels < 0] = 0
    return out


def _footprint_for(lay: UnitLayout, size_units: int) -> torch.Tensor:
    return dataclasses.replace(lay, target_units=int(size_units)).footprint()


def posterior_traces(
    z: torch.Tensor,
    d: torch.Tensor,
    eval_size: torch.Tensor,
    labels: torch.Tensor,
    lay: UnitLayout,
    mode: str,
    kappa: float,
    p_null: float = P_NULL,
    k_looks: int = K_LOOKS,
) -> Dict[str, torch.Tensor]:
    """Run the fixed look traces through the size-conditioned posterior.

    z (n, K, V, C), d (n, K, V), eval_size (n,), labels (n,).  Returns confidence,
    correctness, and the K-look outcome distribution.  Column 0 is the prior (before any look).
    """
    n, K, V, C = z.shape
    if d.shape != (n, K, V):
        raise ValueError("d must be (n, K, V)")
    if K < k_looks:
        raise ValueError("not enough looks")
    conf = torch.zeros(n, k_looks + 1)
    correct = torch.zeros(n, k_looks + 1, dtype=torch.bool)
    probs = torch.zeros(n, 1 + C)
    truth = true_outcome(labels)
    for s in sorted(set(int(x) for x in eval_size.tolist())):
        idx = (eval_size == s).nonzero().squeeze(1)
        if not idx.numel():
            continue
        post = SearchPosterior(lay.N, C, int(idx.numel()), p_null=p_null, mode=mode, footprint=_footprint_for(lay, s),
                               n_units=V, unit_cap=None)
        dec, p = post.decision()
        conf[idx, 0] = p.cpu()
        correct[idx, 0] = dec.cpu() == truth[idx]
        for t in range(1, k_looks + 1):
            post.update(z[idx, t - 1].cpu(), d[idx, t - 1].cpu(), kappa=kappa)
            dec, p = post.decision()
            conf[idx, t] = p.cpu()
            correct[idx, t] = dec.cpu() == truth[idx]
        probs[idx] = post.outcome_probs().cpu()
    return {"conf": conf, "correct": correct, "probs": probs}


def ece_15(conf: torch.Tensor, correct: torch.Tensor, bins: int = 15) -> float:
    conf = conf.flatten().float()
    correct = correct.flatten().float()
    n = conf.numel()
    if n == 0:
        return float("nan")
    order = torch.argsort(conf, stable=True)
    total = 0.0
    for chunk in torch.tensor_split(order, bins):
        if chunk.numel():
            total += float((conf[chunk].mean() - correct[chunk].mean()).abs()) * chunk.numel() / n
    return total


def nll_brier(probs: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
    truth = true_outcome(labels)
    onehot = F.one_hot(truth, probs.shape[1]).float()
    p_true = probs.gather(1, truth[:, None]).squeeze(1).clamp_min(1e-12)
    return float(-p_true.log().mean()), float(((probs - onehot) ** 2).sum(1).mean())


def e2_metrics(traces: Dict[str, torch.Tensor], labels: torch.Tensor, eval_size: torch.Tensor, k_looks: int = K_LOOKS) -> Dict[str, object]:
    conf, correct, probs = traces["conf"][:, k_looks], traces["correct"][:, k_looks], traces["probs"]
    primary = ece_15(conf, correct)
    nll, brier = nll_brier(probs, labels)
    by_k = [ece_15(traces["conf"][:, t], traces["correct"][:, t]) for t in range(k_looks + 1)]
    by_size = {}
    n_by_size = {}
    for s in sorted(set(int(x) for x in eval_size.tolist())):
        idx = (eval_size == s).nonzero().squeeze(1)
        by_size[str(s)] = ece_15(conf[idx], correct[idx])
        n_by_size[str(s)] = int(idx.numel())
    status = "PASS" if primary <= E2_PASS else ("INCONCLUSIVE" if primary <= E2_INCONCLUSIVE else "FAIL")
    return {
        "ece_15_K8": primary,
        "nll_K8": nll,
        "brier_K8": brier,
        "accuracy_K8": float(correct.float().mean()),
        "mean_confidence_K8": float(conf.mean()),
        "ece_by_K": by_k,
        "ece_by_size": by_size,
        "n_by_size": n_by_size,
        "n_scenes": int(labels.numel()),
        "status": status,
        "thresholds": {"pass_max": E2_PASS, "inconclusive_max": E2_INCONCLUSIVE},
    }


def _e5_variant(
    cal_risk: torch.Tensor,
    cal_correct: torch.Tensor,
    test_risk: torch.Tensor,
    test_correct: torch.Tensor,
    selective: bool,
    n_cal: int,
    bootstrap: int,
    alpha: float,
    delta: float,
    k_min: int,
    lambdas: Sequence[float],
    seed: int,
) -> Dict[str, object]:
    cal_risk, cal_correct = cal_risk.cpu(), cal_correct.cpu()
    test_risk, test_correct = test_risk.cpu(), test_correct.cpu()
    k_looks = test_risk.shape[1] - 1
    losses = []
    for lam in lambdas:
        if selective:
            _, decided, ok = stopped_stats_selective(cal_risk, cal_correct, float(lam), k_min)
            losses.append((decided & ~ok).float())
        else:
            _, ok = stopped_stats(cal_risk, cal_correct, float(lam), k_min)
            losses.append((~ok).float())
    losses = torch.stack(losses, dim=1)
    g = torch.Generator().manual_seed(int(seed))
    fixed_risk = float((~test_correct[:, k_looks]).float().mean())
    violations, risks, coverages, looks, efficiencies, plugin_violations, plugin_risks = [], [], [], [], [], [], []
    chosen_counts = [0] * len(lambdas)
    no_choice = 0
    for _ in range(int(bootstrap)):
        idx = torch.randint(0, losses.shape[0], (int(n_cal),), generator=g)
        choice = ltt_fixed_sequence(losses[idx], alpha, delta)["chosen"]
        plugin = naive_plugin_choice(losses[idx], alpha)
        if choice is None:
            no_choice += 1
            violations.append(True)
            risks.append(1.0)
            coverages.append(0.0)
            looks.append(float(k_looks))
            efficiencies.append(0.0)
        else:
            chosen_counts[int(choice)] += 1
            lam = float(lambdas[int(choice)])
            if selective:
                t_looks, decided, ok = stopped_stats_selective(test_risk, test_correct, lam, k_min)
                risk = float((decided & ~ok).float().mean())
                coverage = float(decided.float().mean())
            else:
                t_looks, ok = stopped_stats(test_risk, test_correct, lam, k_min)
                risk = float((~ok).float().mean())
                coverage = 1.0
            violations.append(risk > alpha)
            risks.append(risk)
            coverages.append(coverage)
            looks.append(float(t_looks.float().mean()))
            efficiencies.append(1.0 - float(t_looks.float().mean()) / k_looks)
        if plugin is None:
            plugin_violations.append(False)
            plugin_risks.append(float("nan"))
        else:
            lam = float(lambdas[int(plugin)])
            if selective:
                _, decided, ok = stopped_stats_selective(test_risk, test_correct, lam, k_min)
                risk = float((decided & ~ok).float().mean())
            else:
                _, ok = stopped_stats(test_risk, test_correct, lam, k_min)
                risk = float((~ok).float().mean())
            plugin_violations.append(risk > alpha)
            plugin_risks.append(risk)
    violation_rate = float(sum(violations) / max(1, len(violations)))
    mean_risk = float(sum(risks) / max(1, len(risks)))
    mean_coverage = float(sum(coverages) / max(1, len(coverages)))
    mean_looks = float(sum(looks) / max(1, len(looks)))
    mean_efficiency = float(sum(efficiencies) / max(1, len(efficiencies)))
    plugin_violation_rate = float(sum(plugin_violations) / max(1, len(plugin_violations)))
    plugin_risk_values = [x for x in plugin_risks if _finite(x)]
    plugin_mean_risk = float(sum(plugin_risk_values) / len(plugin_risk_values)) if plugin_risk_values else float("nan")
    pass_validity = violation_rate <= delta
    pass_efficiency = mean_efficiency >= E5_EFFICIENCY_MIN and mean_risk <= fixed_risk + 1e-12
    return {
        "kind": "selective" if selective else "forced_answer",
        "n_cal_pool": int(losses.shape[0]),
        "n_cal_subset": int(n_cal),
        "bootstrap": int(bootstrap),
        "alpha": float(alpha),
        "delta": float(delta),
        "k_min": int(k_min),
        "lambda_grid": [float(x) for x in lambdas],
        "fixed_horizon_risk": fixed_risk,
        "violation_rate": violation_rate,
        "violation_count": int(sum(violations)),
        "no_choice_count": int(no_choice),
        "mean_test_risk": mean_risk,
        "mean_test_coverage": mean_coverage,
        "mean_looks": mean_looks,
        "mean_looks_saved": k_looks - mean_looks,
        "mean_efficiency": mean_efficiency,
        "chosen_lambda_counts": chosen_counts,
        "pass_validity": bool(pass_validity),
        "pass_efficiency": bool(pass_efficiency),
        "pass_all": bool(pass_validity and pass_efficiency),
        "plugin_violation_rate": plugin_violation_rate,
        "plugin_mean_test_risk": plugin_mean_risk,
        "conservative_control_met": bool(plugin_violation_rate >= 2.0 * delta),
        "efficiency_min": E5_EFFICIENCY_MIN,
    }


def e5_metrics(
    cal_traces: Dict[str, torch.Tensor],
    test_traces: Dict[str, torch.Tensor],
    n_cal: int = LTT_N_CAL,
    bootstrap: int = LTT_BOOTSTRAP,
    alpha: float = LTT_ALPHA,
    delta: float = LTT_DELTA,
    k_min: int = LTT_K_MIN,
    lambdas: Sequence[float] = LTT_LAMBDAS,
    seed: int = 0,
) -> Dict[str, object]:
    """LTT over the calibration traces, evaluated on the full test traces.

    A bootstrap resample that validates no lambda is counted as a violation: a gate cannot pass
    by never certifying a policy.  Selective committed error is primary; forced-answer is reported.
    """
    primary = _e5_variant(cal_traces["risk"], cal_traces["correct"], test_traces["risk"], test_traces["correct"],
                          True, n_cal, bootstrap, alpha, delta, k_min, lambdas, seed)
    forced = _e5_variant(cal_traces["risk"], cal_traces["correct"], test_traces["risk"], test_traces["correct"],
                         False, n_cal, bootstrap, alpha, delta, k_min, lambdas, seed)
    return {"selective": primary, "forced_answer": forced, "primary": "selective"}


def risk_traces_from_posterior_traces(traces: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {"risk": 1.0 - traces["conf"], "correct": traces["correct"]}


def aggregate_runs(runs: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate per-seed Gate-2 runs with the frozen 2-of-3 rule."""
    if not runs:
        return {"status": "INVALID", "reason": "no runs"}
    seeds = [int(r.get("seed")) for r in runs]
    if sorted(seeds) != [17, 42, 99]:
        return {"status": "INVALID", "reason": f"expected seeds 17, 42, 99; got {sorted(seeds)}", "seeds": sorted(seeds)}
    e2 = [r.get("e2", {}) for r in runs]
    e5 = [r.get("e5", {}).get("selective", {}) for r in runs]
    e2_pass = sum(1 for x in e2 if x.get("status") == "PASS")
    e2_inc = sum(1 for x in e2 if x.get("status") == "INCONCLUSIVE")
    if e2_pass >= 2:
        e2_status = "PASS"
    elif e2_pass + e2_inc >= 2:
        e2_status = "INCONCLUSIVE"
    else:
        e2_status = "FAIL"
    validity = sum(1 for x in e5 if x.get("pass_validity") is True)
    efficiency = sum(1 for x in e5 if x.get("pass_efficiency") is True)
    e5_status = "PASS" if (validity >= 2 and efficiency >= 2) else "FAIL"
    if e2_status == "PASS" and e5_status == "PASS":
        gate = "PASS"
    elif e2_status == "FAIL" or e5_status == "FAIL":
        gate = "FAIL"
    else:
        gate = "INCONCLUSIVE"
    return {
        "status": gate,
        "e2_status": e2_status,
        "e5_status": e5_status,
        "seeds": sorted(seeds),
        "e2_pass_count": e2_pass,
        "e2_inconclusive_count": e2_inc,
        "e5_validity_pass_count": validity,
        "e5_efficiency_pass_count": efficiency,
        "conservative_control_count": sum(1 for x in e5 if x.get("conservative_control_met") is True),
        "per_seed": [{"seed": int(r.get("seed")), "e2": r.get("e2", {}).get("status"), "e5_selective_pass": r.get("e5", {}).get("selective", {}).get("pass_all"),
                      "e5_forced_pass": r.get("e5", {}).get("forced_answer", {}).get("pass_all")} for r in runs],
    }
