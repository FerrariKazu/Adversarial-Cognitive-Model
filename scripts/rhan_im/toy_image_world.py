#!/usr/bin/env python3
"""RHAN-IM on real pixels (small): measured detectability -> Bayes state -> policy -> stopping.

World   48x48 canvas = 6x6 cells of 8 px. 1/f background noise; with probability 0.6 one Gabor target
        (4 orientations) is pasted into a random cell. The "retina" blurs every cell by an amount that
        grows with its distance (in cells) from the fixation; the sensor is a (biorthogonal) template
        matcher. Evidence is DETERMINISTIC given the pixels, so the state combines looks in "best" mode.
What is measured, not assumed
        d'(retinal level) from calibration scenes, the z-ROC slope (is the equal-variance Gaussian model
        a decent description?), the null/positive standard deviations, cross-class leakage.
What this does NOT show
        anything about deep networks. It shows that the *pipeline* (measure -> calibrate -> track the
        ignorance map -> choose looks -> certify stopping) works on non-Gaussian, spatially correlated
        evidence, and where it breaks (the ablations).

Usage: python scripts/rhan_im/toy_image_world.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rhan_im.agent import run_search  # noqa: E402
from rhan_im.geometry import tiling_order  # noqa: E402
from rhan_im.sensors import GaborImageWorld  # noqa: E402
from rhan_im.stopping import (  # noqa: E402
    curtail_certificate,
    ltt_fixed_sequence,
    naive_plugin_choice,
    stopped_stats,
    stopped_stats_selective,
)

torch.set_num_threads(2)
OUT = ROOT / "docs" / "research"
KAPPA = 5.0


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def ece(conf: torch.Tensor, correct: torch.Tensor, bins: int = 15) -> float:
    edges = torch.linspace(0, 1, bins + 1)
    n, e = conf.numel(), 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= edges[0]) & (conf <= edges[1])
        if m.any():
            e += m.sum().item() / n * abs(conf[m].mean().item() - correct[m].float().mean().item())
    return e


def paired(a: torch.Tensor, b: torch.Tensor):
    d = a.float() - b.float()
    m, se = d.mean().item(), d.std().item() / math.sqrt(d.numel())
    return m, (m - 1.96 * se, m + 1.96 * se)


def run_profile(name: str, blur_per_level: float, amplitude: float, quick: bool) -> dict:
    B = 600 if quick else 2000
    B_evoi = 200 if quick else 700
    K = 8
    t0 = time.time()
    world = GaborImageWorld(grid=6, cell_px=8, n_classes=4, amplitude=amplitude, p_null=0.4, n_levels=5, blur_per_level=blur_per_level)
    cal = world.calibrate(n=1200 if quick else 4000, gen=gen(1))
    vis = world.vis()                                                   # (36, 36) measured d'(level(k, b))
    d_fovea = float(world.d_level[0])
    vis_const = torch.full_like(vis, d_fovea)                           # "no psychophysics": same d' everywhere
    scene = world.make_scene(B, gen(2))
    order = tiling_order(6, 6, 3, lattice_only=True)
    res: dict = {"calibration": cal, "B": B, "d_level_measured": [round(x, 3) for x in cal["d_level"]]}

    # ---- policies (state in the correct "best" mode, measured d') ------------------------------
    acc, correct, ign = {}, {}, {}
    for pol in ["evoi", "koopman", "map", "tiling", "random_wor", "random"]:
        if pol == "evoi":
            sc = world.make_scene(B_evoi, gen(2))                       # same first B_evoi scenes (same seed, same stream prefix)
        else:
            sc = scene
        out = run_search(sc, vis, pol, K, world.p_null, mode="best", kappa=KAPPA, generator=gen(7),
                         order=order if pol == "tiling" else None, n_mc=16)
        correct[pol] = out["correct"]
        acc[pol] = out["correct"].float().mean(0).tolist()
        ign[pol] = out["ign"].mean(0).tolist()
    res["accuracy_by_looks"] = acc
    res["mean_ignorance_mass_by_looks"] = ign
    n = correct["evoi"].shape[0]
    res["paired_vs_tiling"] = {}
    for pol in ("koopman", "map", "evoi", "random_wor"):
        res["paired_vs_tiling"][pol] = {
            str(k): dict(zip(("mean", "ci95"), paired(correct[pol][:n, k], correct["tiling"][:n, k]))) for k in (2, 4, 6, 8)
        }

    # ---- ablations on the state (koopman policy, identical scenes) -------------------------------
    abl = {}
    for label, kw in [
        ("best + measured d' (RHAN-IM)", dict(mode="best", vis_believed=vis)),
        ("sum  + measured d' (wrong combination rule)", dict(mode="sum", vis_believed=vis)),
        ("best + constant d' (no eccentricity measurement)", dict(mode="best", vis_believed=vis_const)),
    ]:
        vb = kw.pop("vis_believed")
        out = run_search(scene, vb, "koopman", K, world.p_null, kappa=KAPPA, generator=gen(7), **kw)
        conf = 1.0 - out["risk"]
        abl[label] = {
            str(t): {"accuracy": out["correct"][:, t].float().mean().item(), "mean_confidence": conf[:, t].mean().item(), "ece": ece(conf[:, t], out["correct"][:, t])}
            for t in (2, 4, 8)
        }
    res["ablations_koopman"] = abl

    # ---- lossless curtailment in "best" mode ------------------------------------------------------
    d_cap = vis.max(0).values
    certs = []
    out = run_search(scene, vis, "koopman", K, world.p_null, mode="best", kappa=KAPPA, generator=gen(7),
                     hook=lambda t, post, k, z: certs.append(curtail_certificate(post, d_cap, KAPPA, K - t)[0].clone()))
    cert = torch.stack(certs, 1)
    first = torch.where(cert.any(1), cert.float().argmax(1), torch.full((B,), K))
    dec_exit = out["dec"][torch.arange(B), first]
    res["curtailment_best_mode"] = {
        "mean_looks_used": first.float().mean().item(),
        "budget": K,
        "fraction_exiting_before_budget": (first < K).float().mean().item(),
        "decision_mismatches_vs_full_budget": int((dec_exit != out["dec"][:, -1]).sum().item()),
        "accuracy_at_budget": out["correct"][:, -1].float().mean().item(),
        "certified_fraction_by_look": cert.float().mean(0).tolist(),
    }

    # ---- Learn-then-Test on measured (non-Gaussian) evidence --------------------------------------
    delta = 0.10
    lams = torch.logspace(math.log10(0.002), math.log10(0.45), 20).tolist()
    Bp = 1500 if quick else 4000
    traces = []
    for seed in (11, 12):
        sc = world.make_scene(Bp, gen(seed))
        traces.append(run_search(sc, vis, "koopman", K, world.p_null, mode="best", kappa=KAPPA, generator=gen(seed + 100)))
    ltt = {"delta": delta, "pool": Bp}
    reps, n_cal = (150 if quick else 400), 600
    for variant, alpha in (("forced_answer_error_rate", 0.10), ("selective_committed_error", 0.05)):
        cols, looks, covs = [[], []], [[], []], [[], []]
        for i, o in enumerate(traces):
            for lam in lams:
                if variant.startswith("forced"):
                    t, c = stopped_stats(o["risk"], o["correct"], lam)
                    cols[i].append((~c).float())
                    covs[i].append(torch.ones(Bp))
                else:
                    t, dec, c = stopped_stats_selective(o["risk"], o["correct"], lam)
                    cols[i].append((dec & ~c).float())
                    covs[i].append(dec.float())
                looks[i].append(t.float())
        Lc, Lt = torch.stack(cols[0], 1), torch.stack(cols[1], 1)
        Tt, Ct = torch.stack(looks[1], 1), torch.stack(covs[1], 1)
        Rt, looks_t, cov_t = Lt.mean(0), Tt.mean(0), Ct.mean(0)
        j_nom = max(j for j, l in enumerate(lams) if l <= alpha)
        out = {"alpha": alpha,
               "nominal_rule": {"lambda": lams[j_nom], "true_risk": Rt[j_nom].item(), "mean_looks": looks_t[j_nom].item(), "coverage": cov_t[j_nom].item()},
               "true_risk_by_lambda": dict(zip([f"{l:.4f}" for l in lams], [round(r, 4) for r in Rt.tolist()]))}
        g = gen(900)
        v_l = v_n = ab = 0
        lk_l, lk_n, cv_l = [], [], []
        for _ in range(reps):
            idx = torch.randperm(Bp, generator=g)[:n_cal]
            c = ltt_fixed_sequence(Lc[idx], alpha, delta)["chosen"]
            v = naive_plugin_choice(Lc[idx], alpha)
            if c is None:
                ab += 1
            else:
                v_l += int(Rt[c] > alpha)
                lk_l.append(looks_t[c].item())
                cv_l.append(cov_t[c].item())
            if v is not None:
                v_n += int(Rt[v] > alpha)
                lk_n.append(looks_t[v].item())
        out.update({"n_cal": n_cal, "reps": reps, "ltt_violation_rate": v_l / reps, "ltt_abstain_rate": ab / reps,
                    "ltt_mean_looks": sum(lk_l) / max(1, len(lk_l)), "ltt_mean_coverage": sum(cv_l) / max(1, len(cv_l)),
                    "plugin_violation_rate": v_n / reps, "plugin_mean_looks": sum(lk_n) / max(1, len(lk_n))})
        ltt[variant] = out
    res["learn_then_test"] = ltt
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(OUT / "RHAN_IM_toy_image_world.json"))
    a = ap.parse_args()
    results = {"quick": a.quick, "torch": torch.__version__, "kappa": KAPPA}
    for name, bpl, amp in (
        ("sharp_blur_hard_amp1.1", 1.0, 1.1),
        ("gradual_blur_hard_amp1.1", 0.5, 1.1),
        ("gradual_blur_easier_amp1.8", 0.5, 1.8),
    ):
        results[name] = run_profile(name, bpl, amp, a.quick)
        print(f"[{name}] {results[name]['seconds']}s  d'={results[name]['d_level_measured']}", flush=True)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(results, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
