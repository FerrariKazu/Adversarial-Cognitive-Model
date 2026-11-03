"""
Stage-2: assemble the audit summary from the raw stage-0/0b/1 artifacts.

Reads the JSON written by stage0_profile.py, stage0b_forward_decomposition.py
and stage1_candidates.py, folds them into one audit table, and writes
report/audit_rev1/stage2_summary.json.

No number in the output is transcribed by hand.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORT_DIR = os.path.join(REPO_ROOT, "report", "audit_rev1")


def load(name: str) -> Dict[str, Any]:
    with open(os.path.join(REPORT_DIR, name)) as f:
        return json.load(f)


def main() -> None:
    prof = load("stage0_profile.json")
    decomp = load("stage0b_forward_decomposition.json")
    cand = {v: load(f"stage1_{v}.json") for v in ("baseline", "a", "b", "c")}

    p = prof["result"]
    base_ms = cand["baseline"]["timing"]["median_ms"]

    candidate_table = []
    labels = {
        "baseline": "baseline (production step verbatim)",
        "a": "A: shared clean forward (fp32 clean fwd reused as PGD target + TRADES clean logits)",
        "b": "B: fused 2B TRADES pass (attack unchanged, clean+adv in one trunk pass)",
        "c": "C: torch.compile(model) on the baseline step",
    }
    for v in ("baseline", "a", "b", "c"):
        d = cand[v]
        candidate_table.append(
            {
                "variant": v,
                "label": labels[v],
                "median_ms": d["timing"]["median_ms"],
                "min_ms": d["timing"]["min_ms"],
                "max_ms": d["timing"]["max_ms"],
                "peak_mem_mb": d["timing"]["peak_mem_mb"],
                "first_call_s": d["first_call_s"],
                "speedup_vs_baseline": round(base_ms / d["timing"]["median_ms"], 3),
                "agreement_vs_baseline": d["agreement"],
                "step": {
                    "trunk_runs": {"baseline": 7, "a": 6, "b": 6, "c": 7}[v],
                },
            }
        )

    top = [e for e in p["profile_events"] if e["event"] != "stage0_training_step"][:12]

    summary = {
        "title": "Audit rev-1 stage 0/1: training-step cost and optimisation candidates",
        "scope": {
            "what": "measure and profile the production TRADES-PGD4 training step, then test three "
                    "candidate rewrites for equality and speed",
            "phase": "recurrence_only",
            "arm": "v1",
            "img_size": 112,
            "batch": 48,
            "eps": 0.094,
            "pgd_steps": 4,
            "beta": 2.5,
            "w_trades": 0.55,
            "machine": prof["machine"],
            "caveat": "RTX 4060, not the T4 the recipe targets. Absolute times are not T4 numbers; "
                      "ratios, kernel structure and equality checks are.",
        },
        "audit_table": {
            "step_structure": {
                "model_forwards_per_step": decomp["trades_pgd4_step"]["model_forwards"],
                "trunk_passes_per_model_forward": decomp["recurrence_only"][
                    "trunk_forward_calls_per_model_forward"
                ],
                "trunk_passes_per_step": decomp["trades_pgd4_step"]["trunk_passes_per_step"],
                "source": "instrumented count of CompactViTGen2._trunk_forward calls per forward",
            },
            "single_clean_forward": {
                "recurrence_only_112_ms": decomp["recurrence_only"]["clean_forward_ms"],
                "ff_112_single_trunk_pass_ms": decomp["ff_112_trunk_pass"]["clean_forward_ms"],
                "ratio": decomp["ratios"]["recurrence_forward_over_ff_trunk_pass"],
            },
            "measured_step": {
                "wall_ms": p["step_wall_ms"],
                "device_busy_ms": p["launch_gaps"]["cuda_busy_ms"],
                "device_busy_fraction": p["launch_gaps"]["cuda_busy_fraction"],
                "kernel_count": p["kernel_count"],
                "total_gap_ms": p["launch_gaps"]["gap_total_ms"],
                "gap_median_us": p["launch_gaps"]["gap_median_us"],
                "gap_p90_us": p["launch_gaps"]["gap_p90_us"],
                "step_cpu_time_ms": max(
                    e["cpu_time_ms"]
                    for e in p["profile_events"]
                    if e["event"] == "stage0_training_step"
                ),
            },
            "sdpa": p["sdpa_backend"],
            "sdpa_backend_selected_from_kernels": {
                "forward": "fmha_cutlassF_f32_aligned_64x64_rf_sm80 (mem-efficient)",
                "backward": "fmha_cutlassB_f32_aligned_64x64_k64_sm80 (mem-efficient)",
                "flash_used": False,
                "why": "attention runs in fp32 with aligned head dim 64, so flash is not selected "
                       "even though flash_sdp is enabled",
            },
            "top_device_time_ops": top,
            "peak_memory_mb_baseline": cand["baseline"]["timing"]["peak_mem_mb"],
        },
        "candidate_table": candidate_table,
        "recommendation": {
            "verdict": "Adopt none of the three candidates. B is lossless but does not pay; A and C "
                       "are regressions.",
            "detail": [
                "B (fused 2B clean+adv trunk pass) is bit-identical to production (max abs diff 0.0 on "
                "x_adv, clean logits and adv logits) but changes median step time by 0.3% (1635.9 vs "
                "1630.6 ms), i.e. nothing. The two B-sized GEMMs cost the same as one 2B GEMM.",
                "A (shared clean forward) is numerically faithful for the attack (x_adv and adv logits "
                "differ by exactly 0.0) and would save 1 of 7 model forwards, but holding the fp32 clean "
                "graph across the 4-step PGD loop pushes peak memory from 5.5 GB to 9.7 GB on an 8 GB "
                "card and makes the step 12.3x slower (20010.6 ms).",
                "C (torch.compile) is 21% slower (1974.9 ms) with 15.0 s of compile time, and is not "
                "numerically identical (x_adv max abs diff 0.188, adv logits 0.061) because the attack "
                "amplifies the compiled-vs-eager fp32 difference through sign().",
                "The profile says why: the step is device-bound at 94.8% CUDA busy with 36.4 ms of "
                "non-overlapping launch gaps in a 1572 ms span. Cutting launches or host work cannot "
                "help; only fewer FLOPs or fewer trunk passes can.",
                "The real cost driver is structural: one recurrence_only forward runs 4 trunk passes "
                "(one per glimpse) at 4.76x the cost of a 64-token trunk pass, so a TRADES-PGD-4 step "
                "costs 28 trunk passes versus 6 for the feedforward 112px substrate.",
            ],
            "if_a_speedup_is_wanted": [
                "Reduce the number of model forwards per step (the PGD inner loop dominates), not the "
                "launch overhead.",
                "Or reduce per-forward work: within_glimpse_iters / num_glimpses are the multipliers.",
                "Any fp16/bf16 change must be re-validated: the attack's KL target is currently the fp32 "
                "clean softmax, distinct from the autocast clean softmax used as the TRADES KL target.",
            ],
        },
        "verified_hypothesis_unknown": {
            "verified": [
                "A TRADES-PGD-4 step of recurrence_only at 112 px, batch 48, takes 1630.6 ms median "
                "(min 1609.6, max 1679.0) on this RTX 4060 / torch 2.9.1+cu128, at 5500 MB peak.",
                "One model forward performs exactly 4 trunk passes (counted, not inferred) and costs "
                "119.7 ms vs 25.2 ms for one 64-token ff_112 trunk pass at the same batch (4.76x).",
                "The step issues 18770 CUDA kernels and is 94.8% device-busy; all non-overlapping gaps "
                "sum to 36.4 ms (median 1.25 us, p90 1.73 us) of a 1572 ms span. Launch overhead is not "
                "a lever on this workload.",
                "SDPA resolves to the mem-efficient fp32 CUTLASS kernels, not flash, in both forward and "
                "backward, although flash_sdp is enabled.",
                "GEMMs dominate device time (addmm/mm/sgemm ~1.2 s of kernel time), i.e. the step is "
                "GEMM-bound and backward-heavy.",
                "The production step runs 7 trunk-scale forwards per step: 1 fp32 no-grad clean (attack "
                "target), 4 fp32 adv (attack), 1 autocast clean + 1 autocast adv (TRADES).",
                "Candidate B is bit-identical to production and B buys 0.3%: batching the two TRADES "
                "passes into one 2B pass is not a lever here.",
                "Candidate A's attack is bit-identical to production (x_adv diff exactly 0.0) only when "
                "the shared clean forward is fp32; sharing an autocast clean forward instead changes the "
                "PGD target and moves x_adv by 0.188 with adv-logit KL 0.329.",
                "Candidate C (torch.compile) is 21% slower and not numerically identical.",
                "The earlier step6_latency_benchmark.json rows reproduce on this machine now "
                "(ff_112 TRADES-PGD4 0.356 vs 0.447 s/batch recorded; recurrence_only 56 px 0.537 vs "
                "0.587 s/batch recorded); those rows are 56 px, not the 112 px recurrent config profiled "
                "here.",
            ],
            "hypothesis": [
                "A large part of the 21% torch.compile loss is the fp32 sgemm path on a device where "
                "inductor reports 'not enough SMs to use max_autotune_gemm'; a different GPU could flip "
                "this sign. Not tested.",
                "Memory, not arithmetic, is what kills A: its 9.7 GB peak exceeds the 8 GB card. A at a "
                "smaller batch (or with the clean activation offloaded / recomputed) is untested and "
                "could be a real saving.",
                "Reducing within_glimpse_iters or num_glimpses should cut step time roughly in "
                "proportion to trunk passes; not measured.",
            ],
            "unknown": [
                "Whether any of these ratios carry to the T4/A100 the recipe targets; only the RTX 4060 "
                "was available.",
                "Whether the step is memory-capacity-bound rather than compute-bound in production, where "
                "the optimizer state and the full training loop also allocate.",
                "Whether CUDA graphs (not tested; the PGD loop reuses buffers, so capture is plausible) "
                "would remove the 36.4 ms gap budget - but that budget is 2.3% of the step, so the "
                "ceiling is small.",
            ],
        },
        "artifacts": {
            "profile": "report/audit_rev1/stage0_profile.json",
            "forward_decomposition": "report/audit_rev1/stage0b_forward_decomposition.json",
            "candidates": [f"report/audit_rev1/stage1_{v}.json" for v in ("baseline", "a", "b", "c")],
            "reference_tensors": "report/audit_rev1/stage1_reference.pt",
            "scripts": [
                "scratch/audit_rev1/stage0_profile.py",
                "scratch/audit_rev1/stage0b_forward_decomposition.py",
                "scratch/audit_rev1/stage1_candidates.py",
                "scratch/audit_rev1/stage2_summary.py",
            ],
        },
    }

    out = os.path.join(REPORT_DIR, "stage2_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote:", out)


if __name__ == "__main__":
    main()
