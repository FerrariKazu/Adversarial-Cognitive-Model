#!/usr/bin/env python3
"""Real parameter / FLOP / memory / shape budget of RHAN-IM-Sensor, measured on the actual
gen2_foundation CompactViTGen2 (random init; CPU only; no weights needed because none of these numbers
depend on the weight values).

What it measures (nothing is assumed):
  * parameters by component of the trunk and of the new evidence head, and what the ff path never uses
  * FLOPs of one look (a 112-px window through prep + 12 blocks + norm) via torch's FlopCounterMode
  * bytes the autograd graph saves for backward in one fp32 look (hooks on saved tensors, de-duplicated,
    parameters excluded), i.e. the activation memory per window that training must hold
  * analytic cost and size of the Bayes state, the update and the Koopman policy for the real layout
  * the tensor shape at every stage

Usage: python scripts/rhan_im/model_budget.py [--classes 100] [--canvas 224] [--out docs/research/RHAN_IM_model_budget.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gen2_foundation.backbone import CompactViTGen2  # noqa: E402
from rhan_im.layout import UnitLayout  # noqa: E402
from rhan_im.trunk_adapter import ff_token_fn  # noqa: E402
from rhan_im.window_head import EvidenceHead  # noqa: E402

torch.set_num_threads(2)


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def saved_activation_bytes(fn, x: torch.Tensor, exclude_ptrs: set) -> int:
    seen, total = set(), 0

    def pack(t: torch.Tensor):
        nonlocal total
        key = (t.data_ptr(), t.numel(), t.dtype)
        if key not in seen and t.data_ptr() not in exclude_ptrs:
            seen.add(key)
            total += t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn(x)[1]                       # patch tokens: what the evidence head reads
        loss = out.float().pow(2).mean()
    loss.backward()
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, default=100)
    ap.add_argument("--canvas", type=int, default=224)
    ap.add_argument("--window", type=int, default=112)
    ap.add_argument("--target-units", type=int, default=4)
    ap.add_argument("--stride-units", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64, help="scenes per batch for the state-size rows")
    ap.add_argument("--out", default=str(ROOT / "docs" / "research" / "RHAN_IM_model_budget.json"))
    a = ap.parse_args()

    torch.manual_seed(0)
    trunk = CompactViTGen2(img_size=a.window, num_refine_iters=2)
    head = EvidenceHead(384, a.classes)
    lay = UnitLayout(canvas=a.canvas, patch=14, window=a.window, target_units=a.target_units)
    cand = lay.candidates(a.stride_units)
    K, C, N, V, B = cand.shape[0] + 1, a.classes, lay.N, lay.V, a.batch      # +1: the gist look

    comp = {
        "patch_embed (Conv2d 3->384, 14x14)": n_params(trunk.patch_embed),
        "cls + register tokens": trunk.cls_token.numel() + trunk.register_token.numel(),
        f"pos_embed ({lay.tok ** 2 + 2} x 384)": trunk.pos_embed.numel(),
        "12 transformer blocks": sum(n_params(b) for b in trunk.blocks),
        "final LayerNorm": n_params(trunk.norm),
    }
    used = sum(comp.values())
    refine = n_params(trunk.refinement)
    report = {
        "config": {"canvas": a.canvas, "window": a.window, "patch": 14, "classes": C, "target_units": a.target_units,
                   "candidates_native": K - 1, "candidates_total_with_gist": K, "N_hypothesis_anchors": N, "V_evidence_units": V,
                   "tokens_per_window": lay.tok ** 2, "gist_tokens": lay.n_coarse, "scenes_per_batch": B},
        "params": {"trunk_used_by_ff_path": comp, "trunk_used_total": used,
                   "tied_refinement_block_unused_by_ff_path": refine,
                   "evidence_head": n_params(head), "rhan_im_sensor_total_trainable_if_trunk_unfrozen": used + n_params(head),
                   "rhan_im_sensor_total_trainable_if_trunk_frozen": n_params(head),
                   "bayes_state_and_policy_learned_parameters": 0},
    }

    fn = ff_token_fn(trunk)
    x1 = torch.randn(1, 3, a.window, a.window)
    from torch.utils.flop_counter import FlopCounterMode
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        cls, tokens = fn(x1)
    look_flops = fc.get_total_flops()
    report["shapes"] = {
        "canvas": [B, 3, a.canvas, a.canvas], "native_window": [B, 3, a.window, a.window],
        "after_patch_embed": [B, 384, lay.tok, lay.tok], "tokens_in": [B, lay.tok ** 2 + 2, 384],
        "trunk_out_cls": [B, cls.shape[-1]], "trunk_out_patch_tokens": [B, *tokens.shape[1:]],
        "evidence_scores_z_per_look": [B, lay.tok ** 2, C], "unit_scores_after_scatter": [B, V, C],
        "detectability_d": [B, V],
        "posterior_u_contrib": [B, V, C], "posterior_log_cells": [B, N, C], "footprint": [N, V],
        "vis_matrix": [K, V], "koopman_gain": [B, K],
    }
    exclude = {p.data_ptr() for p in trunk.parameters()}
    act = saved_activation_bytes(fn, torch.randn(2, 3, a.window, a.window), exclude) / 2
    n_flops = {"one_look_gflops_forward": look_flops / 1e9, "ff_112_forward_gflops (same trunk + 384x100 head)": (look_flops + 2 * 384 * 100) / 1e9}
    fo = float(look_flops)
    K_stop = 8
    n_flops.update({
        f"scene_with_gist_plus_{K_stop - 1}_native_looks_gflops": K_stop * fo / 1e9,
        "ratio_vs_one_ff_112_forward": K_stop,
        "train_step_per_window_fwd_bwd_gflops_approx_3x": 3 * fo / 1e9,
    })
    report["flops"] = n_flops
    report["memory"] = {
        "saved_activation_bytes_per_window_fp32_cpu_path": act,
        "saved_activation_MB_per_window_fp32": act / 2 ** 20,
        "params_fp32_MB_used_trunk": used * 4 / 2 ** 20,
        "params_fp32_MB_with_unused_refinement": (used + refine) * 4 / 2 ** 20,
        "AdamW_state_MB_if_trunk_trained": used * 8 / 2 ** 20,
        "note": "saved activations measured with autograd saved-tensor hooks on the CPU path (math/CPU attention); the GPU flash-attention path saves less",
    }
    f4 = 4
    bayes = {
        "u_contrib_MB": B * V * C * f4 / 2 ** 20, "log_cells_MB": B * N * C * f4 / 2 ** 20,
        "log_prior_MB": B * N * C * f4 / 2 ** 20, "footprint_MB": N * V * f4 / 2 ** 20,
        "update_project_MFLOPs_per_look": 2 * B * N * V * C / 1e6,
        "koopman_policy_MFLOPs_per_look": 2 * B * K * V * N / 1e6,
        "koopman_policy_intermediate_MB": B * K * N * f4 / 2 ** 20,
        "evoi_policy_intermediate_GB_if_n_mc_16_unchunked": B * K * 16 * V * C * f4 / 2 ** 30,
    }
    bayes["update_vs_one_trunk_look_percent"] = 100.0 * (2 * B * N * V * C) / (B * fo)
    bayes["koopman_vs_one_trunk_look_percent"] = 100.0 * (2 * B * K * V * N) / (B * fo)
    report["bayes_state"] = bayes

    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
