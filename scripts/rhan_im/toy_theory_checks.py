#!/usr/bin/env python3
"""Behaviour checks of the RHAN-IM mathematics in worlds small enough to be exact.

What this does and does not show
    Every number here comes from a *simulated* world whose generative model is (unless a row
    says otherwise) the model the Bayes update assumes. It verifies that the mathematics is
    implemented correctly and shows how it behaves when its assumptions are broken in known ways.
    It is NOT evidence that a deep-network sensor behaves like this; that is what the gate-1
    measurement (scripts/rhan_im/gate1_measure_dprime.py) is for.

Checks
    calibration   posterior confidence vs frequency (ECE), correct and misspecified worlds
    policies      accuracy vs number of looks for the computed policies and the coverage baselines,
                  paired on identical scenes, two visibility profiles
    ltt           Learn-then-Test vs nominal and plug-in stopping thresholds under misspecification
    patch         patch-certificate coverage and a real evidence-rewriting attack, clipped vs unclipped
    premature     fastest possible confident stop under a strongest-possible attacker vs the bound
    curtail_sum   how much worst-case curtailment can save when looks are independent ("sum" mode)

Usage:  python scripts/rhan_im/toy_theory_checks.py [--quick] [--only calibration,policies,...]
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
from rhan_im.belief import SearchPosterior  # noqa: E402
from rhan_im.geometry import (  # noqa: E402
    eccentricity,
    grid_centers,
    tiling_order,
    visibility_hyperbolic,
    visibility_window,
)
from rhan_im.sensors import ToySDTWorld  # noqa: E402
from rhan_im.stopping import (  # noqa: E402
    curtail_certificate,
    ltt_fixed_sequence,
    min_fixations_to_decide,
    naive_plugin_choice,
    patch_certificate,
    stopped_stats,
)

torch.set_num_threads(2)
OUT = ROOT / "docs" / "research"


# ----------------------------------------------------------------------------- helpers
def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def wilson(k: float, n: int, z: float = 1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def ece(conf: torch.Tensor, correct: torch.Tensor, bins: int = 15) -> float:
    edges = torch.linspace(0, 1, bins + 1)
    n, e = conf.numel(), 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= edges[0]) & (conf <= edges[1])
        if m.any():
            e += m.sum().item() / n * abs(conf[m].mean().item() - correct[m].float().mean().item())
    return e


def grid_footprint(gh: int, gw: int, half: int) -> torch.Tensor:
    c = grid_centers(gh, gw)
    dr = (c[:, None, 0] - c[None, :, 0]).abs()
    dc = (c[:, None, 1] - c[None, :, 1]).abs()
    return ((dr <= half) & (dc <= half)).float()


def make_world(gh, gw, C, profile, d0, p_null=0.4, half=1, e2=1.0, radius=1.5, **kw):
    cells = grid_centers(gh, gw)
    ecc = eccentricity(cells, cells)
    vis = visibility_hyperbolic(ecc, d0, e2) if profile == "hyperbolic" else visibility_window(ecc, d0, radius)
    F = grid_footprint(gh, gw, half) if half is not None else None
    return ToySDTWorld(vis, C, p_null=p_null, footprint=F, **kw), vis, F


def paired(a: torch.Tensor, b: torch.Tensor):
    d = a.float() - b.float()
    m, se = d.mean().item(), d.std().item() / math.sqrt(d.numel())
    return m, (m - 1.96 * se, m + 1.96 * se)


# ----------------------------------------------------------------------------- checks
def check_calibration(quick: bool):
    B, K = (4000 if quick else 20000), 6
    res = {}
    for name, kw in [("well_specified", {}), ("true_d_prime_x0.7", {"true_scale": 0.7}), ("student_t_df4_noise", {"noise_df": 4.0})]:
        world, vis, F = make_world(5, 5, 3, "hyperbolic", 1.6, e2=1.5, **kw)
        out = run_search(world.sample(B, gen(1)), vis, "koopman", K, 0.4, footprint=F, generator=gen(2))
        conf = 1.0 - out["risk"]
        res[name] = [
            {
                "looks": t,
                "accuracy": out["correct"][:, t].float().mean().item(),
                "mean_confidence": conf[:, t].mean().item(),
                "ece": ece(conf[:, t], out["correct"][:, t]),
            }
            for t in (0, 1, 2, 4, 6)
        ]
    return {"B": B, "rows": res}


def check_policies(quick: bool):
    B, K = (600 if quick else 2000), 8
    pols = ["evoi", "koopman", "map", "tiling", "random_wor", "random"]
    res = {}
    for prof in ("hyperbolic", "window"):
        world, vis, F = make_world(6, 6, 3, prof, 1.4, e2=1.0, radius=1.5)
        order = tiling_order(6, 6, 3, lattice_only=True)                       # strongest non-adaptive baseline: cycle the lattice
        correct, acc, looks_to_90 = {}, {}, {}
        t0 = time.time()
        for pol in pols:
            out = run_search(
                world.sample(B, gen(100)), vis, pol, K, 0.4, footprint=F, generator=gen(7),
                order=order if pol == "tiling" else None, n_mc=24,
            )
            correct[pol] = out["correct"]
            acc[pol] = out["correct"].float().mean(0).tolist()
            reach = [k for k, a in enumerate(acc[pol]) if a >= 0.90]
            looks_to_90[pol] = reach[0] if reach else None
        prior_acc = correct["random"][:, 0].float().mean().item()
        diffs = {}
        for a, b in [("evoi", "tiling"), ("koopman", "tiling"), ("map", "tiling"), ("koopman", "map"), ("evoi", "koopman"), ("koopman", "random_wor")]:
            diffs[f"{a}-{b}"] = {
                str(k): dict(zip(("mean", "ci95"), paired(correct[a][:, k], correct[b][:, k]))) for k in (2, 4, 6, 8)
            }
        res[prof] = {"accuracy_by_looks": acc, "looks_to_reach_90pct": looks_to_90, "paired_differences": diffs, "prior_mode_accuracy": prior_acc, "seconds": round(time.time() - t0, 1)}
    return {"B": B, "grid": "6x6", "classes": 3, "footprint": "3x3", "profiles": res}


def check_ltt(quick: bool):
    alpha, delta = 0.10, 0.10
    lams = torch.logspace(math.log10(0.002), math.log10(0.45), 20).tolist()          # conservative -> aggressive
    Bpool, reps = (6000, 300) if quick else (30000, 1000)
    n_cals = (300, 1500)
    res = {}
    for name, kw in [("well_specified", {}), ("true_d_prime_x0.8", {"true_scale": 0.8}), ("true_d_prime_x0.6", {"true_scale": 0.6}), ("student_t_df4", {"noise_df": 4.0})]:
        world, vis, F = make_world(5, 5, 3, "hyperbolic", 1.6, e2=1.5, **kw)
        pools = []
        for seed in (11, 12):                                                         # calibration pool, test pool
            out = run_search(world.sample(Bpool, gen(seed)), vis, "koopman", 8, 0.4, footprint=F, generator=gen(seed + 100))
            cols, looks = [], []
            for lam in lams:
                t, corr = stopped_stats(out["risk"], out["correct"], lam)
                cols.append((~corr).float())
                looks.append(t.float())
            pools.append((torch.stack(cols, 1), torch.stack(looks, 1)))
        (Lcal, _), (Ltest, Ttest) = pools
        R_test, looks_test = Ltest.mean(0), Ttest.mean(0)
        j_nom = max(j for j, l in enumerate(lams) if l <= alpha)
        entry = {
            "nominal_rule": {"lambda": lams[j_nom], "true_risk": R_test[j_nom].item(), "mean_looks": looks_test[j_nom].item(), "violates_alpha": bool(R_test[j_nom] > alpha)},
            "true_risk_by_lambda": dict(zip([f"{l:.4f}" for l in lams], [round(r, 4) for r in R_test.tolist()])),
            "by_n_cal": {},
        }
        for n_cal in n_cals:
            g = gen(500 + n_cal)
            v_ltt = v_nv = ab = 0
            lk_ltt, lk_nv = [], []
            for _ in range(reps):
                idx = torch.randperm(Bpool, generator=g)[:n_cal]
                L = Lcal[idx]
                c = ltt_fixed_sequence(L, alpha, delta)["chosen"]
                v = naive_plugin_choice(L, alpha)
                if c is None:
                    ab += 1
                else:
                    v_ltt += int(R_test[c] > alpha)
                    lk_ltt.append(looks_test[c].item())
                if v is not None:
                    v_nv += int(R_test[v] > alpha)
                    lk_nv.append(looks_test[v].item())
            entry["by_n_cal"][str(n_cal)] = {
                "ltt_violation_rate": v_ltt / reps,
                "ltt_abstain_rate": ab / reps,
                "ltt_mean_looks_when_certified": sum(lk_ltt) / max(1, len(lk_ltt)),
                "plugin_violation_rate": v_nv / reps,
                "plugin_mean_looks": sum(lk_nv) / max(1, len(lk_nv)),
                "reps": reps,
            }
        res[name] = entry
    return {"alpha": alpha, "delta": delta, "pool": Bpool, "K_max": 8, "scenarios": res}


def _replay(vis, N, C, p_null, F, ks, zs, kappa, cap, B):
    post = SearchPosterior(N, C, B, p_null=p_null, mode="sum", footprint=F, unit_cap=cap)
    for k, z in zip(ks, zs):
        post.update(z, vis[k], kappa=kappa)
    return post


def check_patch(quick: bool):
    B, K = (300 if quick else 1500), 6
    gh = gw = 6
    N, C, kappa, cap = gh * gw, 3, 4.0, 8.0
    world, vis, F = make_world(gh, gw, C, "hyperbolic", 3.0, e2=1.5, half=1)
    scene = world.sample(B, gen(31))
    ks, zs = [], []
    run_search(scene, vis, "koopman", K, 0.4, footprint=F, generator=gen(32), hook=lambda t, post, k, z: (ks.append(k), zs.append(z)) if t else None)
    cells = grid_centers(gh, gw)
    rng = gen(33)
    anchor = torch.where(scene.cell >= 0, scene.cell, torch.randint(0, N, (B,), generator=rng))
    order = (cells[anchor][:, None, :] - cells[None, :, :]).norm(dim=-1).argsort(dim=1)      # units nearest the anchor first
    present = scene.cell >= 0
    true_cls = scene.cls
    wrong = (true_cls + 1) % C

    def attacked_posterior(m, value, kappa_, cap_):
        zr = []
        for z in zs:
            z = z.clone()
            for j in range(m):
                u = order[:, j]
                ar = torch.arange(B)
                z[ar, u, wrong] = value                                  # inject the wrong class (or a false alarm)
                z[ar[present], u[present], true_cls[present]] = -value   # erase the true class where there is one
            zr.append(z)
        return _replay(vis, N, C, 0.4, F, ks, zr, kappa_, cap_, B)

    rows = {}
    for label, kappa_, cap_, val in [("clipped (kappa=4) + unit_cap=8", kappa, cap, kappa), ("clipped (kappa=4), no cap", kappa, None, kappa), ("unclipped", None, None, 1000.0)]:
        clean = _replay(vis, N, C, 0.4, F, ks, zs, kappa_, cap_, B)
        d0 = clean.decision()[0]
        ok = d0 == scene.label
        rows[label] = {"clean_accuracy": ok.float().mean().item(), "by_m": {}}
        for m in (1, 2, 3):
            att = attacked_posterior(m, val, kappa_, cap_)
            flipped = (att.decision()[0] != d0) & ok
            entry = {"attack_success_on_clean_correct": flipped.sum().item() / max(1, ok.sum().item())}
            if kappa_ is not None and m <= 2:
                cert, _ = patch_certificate(clean, kappa_, m)
                entry["certified_fraction"] = cert.float().mean().item()
                entry["certified_and_correct"] = (cert & ok).float().mean().item()
                entry["flips_among_certified"] = int((flipped & cert).sum().item())
                entry["attack_success_among_uncertified_clean_correct"] = (flipped & ~cert).sum().item() / max(1, (ok & ~cert).sum().item())
            rows[label]["by_m"][str(m)] = entry
    return {"B": B, "K": K, "world": "6x6 grid, 3 classes, 3x3 footprint, d0=3 (hyperbolic, e2=1.5), independent looks", "rows": rows}


def check_premature(quick: bool):
    lam, rows = 0.05, []
    for kappa, d, C, half in [(3.0, 2.0, 3, 1), (4.0, 3.0, 4, 1), (3.0, 2.0, 3, 0), (1.0, 0.8, 3, 0), (1.5, 1.0, 3, 0), (2.0, 1.0, 3, 1)]:
        world, vis, F = make_world(4, 4, C, "hyperbolic", d, half=half)
        N = 16
        post = SearchPosterior(N, C, 1, p_null=0.4, footprint=F)
        lw = post.outcome_log_weights()[0]
        w = int(F.sum(1).max()) if F is not None else 1
        bound = min_fixations_to_decide((lw[1] - lw[0]).item(), lam, d, kappa, footprint_size=w, rival_is_null=True)
        hit = None
        for t in range(1, 200):
            z = torch.full((1, N, C), -kappa)
            z[:, :, 0] = kappa
            post.update(z, torch.full((1, N), d), kappa=kappa)
            idx, p = post.decision()
            if idx.item() == 1 and 1 - p.item() <= lam:
                hit = t
                break
        rows.append({"kappa": kappa, "d_max": d, "classes": C, "footprint_units": w, "risk_target": lam, "lower_bound_on_looks": bound, "fastest_attack_stop": hit, "bound_respected": hit is not None and hit >= bound})
    return {"rows": rows}


def check_curtail_sum(quick: bool):
    B, K, kappa = (2000 if quick else 8000), 8, 4.0
    world, vis, F = make_world(5, 5, 3, "hyperbolic", 2.0, e2=1.5)
    d_cap = vis.max(0).values
    certs = []
    out = run_search(
        world.sample(B, gen(41)), vis, "koopman", K, 0.4, kappa=kappa, footprint=F, generator=gen(42),
        hook=lambda t, post, k, z: certs.append(curtail_certificate(post, d_cap, kappa, K - t)[0].clone()),
    )
    cert = torch.stack(certs, 1)
    first = torch.where(cert.any(1), cert.float().argmax(1), torch.full((B,), K))
    dec_exit = out["dec"][torch.arange(B), first]
    return {
        "B": B, "K_max": K, "kappa": kappa,
        "mean_looks_used_with_certified_exit": first.float().mean().item(),
        "fraction_exiting_before_budget": (first < K).float().mean().item(),
        "decision_mismatches_vs_full_budget": int((dec_exit != out["dec"][:, -1]).sum().item()),
        "certified_fraction_by_look": cert.float().mean(0).tolist(),
        "accuracy_at_budget": out["correct"][:, -1].float().mean().item(),
    }


CHECKS = {
    "calibration": check_calibration,
    "policies": check_policies,
    "ltt": check_ltt,
    "patch": check_patch,
    "premature": check_premature,
    "curtail_sum": check_curtail_sum,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default=",".join(CHECKS))
    ap.add_argument("--out", default=str(OUT / "RHAN_IM_toy_theory_checks.json"))
    a = ap.parse_args()
    results = {"quick": a.quick, "torch": torch.__version__}
    for name in a.only.split(","):
        t0 = time.time()
        results[name] = CHECKS[name](a.quick)
        print(f"[{name}] done in {time.time() - t0:.1f}s", flush=True)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(results, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
