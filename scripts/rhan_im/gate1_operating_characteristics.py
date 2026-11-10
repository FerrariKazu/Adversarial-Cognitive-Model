#!/usr/bin/env python3
"""Operating characteristics of the gate-1 decision rule at the PLANNED sample sizes.

The real gate-1 run needs the frozen checkpoint and ImageNet-100. Before spending that compute, ask a cheaper question:
if the trunk's token scores had a KNOWN property (equal-variance Gaussian evidence; unequal variance; dependent tokens;
image-to-image difficulty differences; both together; a weak sensor; no resolution advantage), how often would the rule in
`rhan_im.gate1.verdict` return each verdict, given the real scene / window sampling of gate1_measure_dprime.py
(target sizes, positions, windows that contain the target, null fraction) and the planned n_cal / n_test / classes?

Synthetic SCORES (no trunk, no pixels) go through gate1_measure_dprime.analyse(), the same function the real run uses,
so the sampling geometry, the standardiser, the estimators and the verdict are the real ones.

What this does NOT show: whether the real trunk has any of these properties (that is gate 1 itself); the cross-look
correlation (it needs two real windows; the estimator is unit-tested and the rule for it is "measure, then choose the mode").

Usage:  python scripts/rhan_im/gate1_operating_characteristics.py [--replicates 5] [--quick] [--resume] [--out FILE]
(the file is saved after every condition; --resume continues an interrupted run with the same plan, replicates and seed)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pathlib
import sys
import time
from typing import Dict, List

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rhan_im import gate1 as g1  # noqa: E402
from rhan_im.layout import UnitLayout  # noqa: E402

_spec = importlib.util.spec_from_file_location("gate1_script", ROOT / "scripts" / "rhan_im" / "gate1_measure_dprime.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

PLAN = dict(canvas=224, window=112, patch=14, stride_units=2, classes=50, sizes=[2, 4, 6], alphas=[1.0, 0.6], op_size=4,
            p_null=0.3, n_cal=3000, n_test=6000)

# name -> properties of the synthetic token evidence. d_* are in null-sd units; rho_w = correlation between the tokens of one
# look (same scene, same class); s_pos = sd of target-class scores on target tokens relative to the null (1 = equal variance);
# cv_d = coefficient of variation of the per-scene difficulty multiplier on d (a target that is easy at one token is easy at its neighbours).
CONDITIONS: Dict[str, Dict[str, float]] = {
    "ideal_equal_variance": dict(d_centre=2.5, d_edge=1.5, d_gist=1.2, rho_w=0.0, s_pos=1.0, cv_d=0.0),
    "dependent_tokens_rho0.3": dict(d_centre=2.5, d_edge=1.5, d_gist=1.2, rho_w=0.3, s_pos=1.0, cv_d=0.0),
    "image_difficulty_cv0.2": dict(d_centre=2.5, d_edge=1.5, d_gist=1.2, rho_w=0.0, s_pos=1.0, cv_d=0.2),
    "image_difficulty_cv0.4": dict(d_centre=2.5, d_edge=1.5, d_gist=1.2, rho_w=0.0, s_pos=1.0, cv_d=0.4),
    "dependent_tokens_and_difficulty": dict(d_centre=2.5, d_edge=1.5, d_gist=1.2, rho_w=0.3, s_pos=1.0, cv_d=0.4),
    "strongly_dependent_weak_tokens_rho0.8": dict(d_centre=1.7, d_edge=1.0, d_gist=0.8, rho_w=0.8, s_pos=1.0, cv_d=0.0),
    "unequal_variance_s1.15": dict(d_centre=2.5, d_edge=1.5, d_gist=1.2, rho_w=0.0, s_pos=1.15, cv_d=0.0),
    "unequal_variance_s1.25_at_threshold": dict(d_centre=2.5, d_edge=1.5, d_gist=1.2, rho_w=0.0, s_pos=1.25, cv_d=0.0),
    "unequal_variance_s1.4": dict(d_centre=2.5, d_edge=1.5, d_gist=1.2, rho_w=0.0, s_pos=1.4, cv_d=0.0),
    "weak_sensor": dict(d_centre=1.2, d_edge=0.8, d_gist=0.6, rho_w=0.0, s_pos=1.0, cv_d=0.0),
    "no_resolution_advantage": dict(d_centre=2.5, d_edge=1.5, d_gist=2.4, rho_w=0.0, s_pos=1.0, cv_d=0.0),
}


def d_map(lay: UnitLayout, d_centre: float, d_edge: float) -> torch.Tensor:
    """(T,) detectability of every window token: highest at the window centre, falling towards the border."""
    t = lay.tok
    r = torch.arange(t, dtype=torch.float32)
    dist2 = ((r[:, None] - (t - 1) / 2) ** 2 + (r[None, :] - (t - 1) / 2) ** 2).flatten()
    return d_edge + (d_centre - d_edge) * torch.exp(-dist2 / (2 * 2.5 ** 2))


def tokens_noise(n: int, T: int, C: int, rho: float, gen: torch.Generator) -> torch.Tensor:
    e = torch.randn(n, T, C, generator=gen)
    if rho <= 0:
        return e
    return math.sqrt(1 - rho) * e + math.sqrt(rho) * torch.randn(n, 1, C, generator=gen)


def synth_split(n: int, lay: UnitLayout, cand: torch.Tensor, cond: Dict[str, float], gen: torch.Generator, classes: int) -> Dict:
    """Scene geometry and window sampling exactly as in the real script; token scores drawn from the synthetic sensor."""
    geo = G.scene_geometry(lay, n, classes, PLAN["sizes"], PLAN["alphas"], PLAN["p_null"], gen)
    tl = G.sample_windows(geo, lay, cand, gen)
    T = lay.tok ** 2
    label, has = geo["label"], geo["label"] >= 0
    onehot = torch.nn.functional.one_hot(label.clamp_min(0), classes).bool() & has[:, None]       # (n, C)
    kappa = torch.exp(cond["cv_d"] * torch.randn(n, generator=gen) - cond["cv_d"] ** 2 / 2) if cond["cv_d"] > 0 else torch.ones(n)
    out = {"sc": {k: geo[k] for k in ("label", "size_units", "alpha", "top", "left", "overlap")}, "tl": tl}

    cover_n = g1.window_token_overlap(geo["overlap"], tl, lay) > 0
    zn = tokens_noise(n, T, classes, cond["rho_w"], gen)
    hit = cover_n[:, :, None] & onehot[:, None, :]
    shift = d_map(lay, cond["d_centre"], cond["d_edge"])[None, :, None] * kappa[:, None, None]
    out["native"] = torch.where(hit, zn * cond["s_pos"] + shift, zn)

    ov_g = g1.gist_token_overlap(geo["overlap"], lay)
    zg = tokens_noise(n, T, classes, cond["rho_w"], gen)
    hit_g = (ov_g > 0)[:, :, None] & onehot[:, None, :]
    out["gist"] = torch.where(hit_g, zg * cond["s_pos"] + cond["d_gist"] * ov_g[:, :, None] * kappa[:, None, None], zg)

    out["cover_native"] = cover_n
    out["ov_gist"] = ov_g
    out["cover_gist"] = ov_g > 0
    return out


def one_replicate(cond: Dict[str, float], seed: int, plan: Dict) -> Dict:
    lay = UnitLayout(canvas=plan["canvas"], patch=plan["patch"], window=plan["window"], target_units=plan["op_size"])
    cand = lay.candidates(plan["stride_units"])
    gen = torch.Generator().manual_seed(seed)
    cal = synth_split(plan["n_cal"], lay, cand, cond, gen, plan["classes"])
    tst = synth_split(plan["n_test"], lay, cand, cond, gen, plan["classes"])
    tst["native2"] = torch.zeros(0, lay.tok ** 2, plan["classes"])                      # cross-look part is not simulated
    tst["tl2"] = torch.zeros(0, 2, dtype=torch.long)
    return G.analyse(cal, tst, lay, plan["sizes"], plan["alphas"], plan["op_size"], 0)


def finite(o):
    """Recursively replace NaN / inf by None so the saved file is strict JSON."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    return o


KEEP = ("zroc_slope_median", "null_sd_median", "pos_sd_median", "d_native_centre", "d_gist", "d_gist_full", "vif_null_op", "vif_pos_op", "vif_pos_scenes_op", "rho_null",
        "op_block_tokens")


def spread(vals: List[float]) -> Dict[str, float]:
    v = torch.tensor([x for x in vals if x is not None and math.isfinite(x)])
    if v.numel() == 0:
        return {"median": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    return {"median": float(v.median()), "min": float(v.min()), "max": float(v.max()), "n": int(v.numel())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="a third of the planned sample sizes (a smoke test, not the planning numbers)")
    ap.add_argument("--only", default=",".join(CONDITIONS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="keep the conditions already in --out (same plan, replicates and seed) and run only the rest; the file is saved after every condition")
    ap.add_argument("--out", default=str(ROOT / "docs" / "research" / "RHAN_IM_gate1_operating_characteristics.json"))
    a = ap.parse_args()
    torch.set_num_threads(2)
    plan = dict(PLAN)
    if a.quick:
        plan.update(n_cal=1000, n_test=2000)
    res: Dict[str, object] = {"plan": plan, "replicates": a.replicates, "seed": a.seed, "thresholds": {k: list(v) if isinstance(v, tuple) else v for k, v in g1.DEFAULT_THRESHOLDS.items()}, "conditions": {}}
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.resume and out.exists():
        prev = json.loads(out.read_text())
        if (prev.get("plan"), prev.get("replicates"), prev.get("seed")) != (plan, a.replicates, a.seed):
            raise SystemExit("--resume: the existing file was made with a different plan, replicate count or seed; refusing to mix them")
        res["conditions"] = prev["conditions"]
        print("resuming; already done:", ", ".join(res["conditions"]) or "none", flush=True)

    def save() -> None:                                                                  # atomic: a restart can never leave a half-written file
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(finite(res), indent=2, default=lambda o: None, allow_nan=False))
        os.replace(tmp, out)

    op = f"size{plan['op_size']}_alpha{plan['alphas'][0]}"
    for name in a.only.split(","):
        if name in res["conditions"]:
            continue
        cond = CONDITIONS[name]
        t0 = time.time()
        reps = [one_replicate(cond, a.seed + 1000 * r + 7, plan) for r in range(a.replicates)]
        verd = [r["verdict_draft"] for r in reps]
        summ = [r["summary"] for r in reps]
        entry = {
            "properties": cond,
            "rate_sdt_adequate": sum(v["sdt_adequate"] for v in verd) / len(verd),
            "rate_informative": sum(v["informative"] for v in verd) / len(verd),
            "rate_looking_closer_pays": sum(v["looking_closer_pays"] for v in verd) / len(verd),
            "rate_pass_all": sum(v["pass_all"] for v in verd) / len(verd),
            "rate_vif_correction": sum(v["token_evidence"].startswith("correct") for v in verd) / len(verd),
            "rate_shared_difficulty": sum(v["shared_difficulty"] for v in verd) / len(verd),
            "per_replicate": [{"summary": {k: sm.get(k) for k in KEEP}, "verdict": vd} for sm, vd in zip(summ, verd)],
            "stats": {k: spread([s[k] for s in summ]) for k in ("zroc_slope_median", "null_sd_median", "pos_sd_median", "d_native_centre", "d_gist", "d_gist_full", "vif_null_op", "vif_pos_op", "vif_pos_scenes_op")},
            "d_block_effective": spread([v["d_block_effective"] for v in verd]),
            "seconds": round(time.time() - t0, 1),
        }
        if name == "ideal_equal_variance":                                              # sample sizes do not depend on the sensor
            counts = {}
            for s in plan["sizes"]:
                key = f"size{s}_alpha{plan['alphas'][0]}"
                npos = torch.tensor(reps[0]["native"][key]["n_pos"]).flatten()
                counts[key] = {"positives_per_token_median": float(npos.median()), "positives_per_token_min": float(npos.min()),
                               "tokens_with_at_least_30_positives": int((npos >= 30).sum()), "tokens": int(npos.numel())}
            entry["positives_per_token_position"] = counts
            entry["counts"] = reps[0]["counts"]
        res["conditions"][name] = entry
        save()
        print(f"[{name}] pass_all {entry['rate_pass_all']:.2f}  adequate {entry['rate_sdt_adequate']:.2f}  informative {entry['rate_informative']:.2f}  "
              f"pays {entry['rate_looking_closer_pays']:.2f}  vif_corr {entry['rate_vif_correction']:.2f}  shared_diff {entry['rate_shared_difficulty']:.2f}  slope {entry['stats']['zroc_slope_median']['median']:.2f}  "
              f"VIFnull {entry['stats']['vif_null_op']['median']:.2f}  VIFpos {entry['stats']['vif_pos_op']['median']:.2f}  ({entry['seconds']}s)", flush=True)
    save()
    print("wrote", out)


if __name__ == "__main__":
    main()
