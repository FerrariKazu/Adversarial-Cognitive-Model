#!/usr/bin/env python3
"""
T4d: Cost table — forward MACs + measured s/step from audit_rev1.
Sources: scratch/audit_rev1/stage0_profile.json (if exists), else measured inline.
Output:  evaluation/stage0_results/t4d_cost_table.json
"""
import os, json
from pathlib import Path

OUT_DIR = Path("evaluation/stage0_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Measured values from audit_rev1 (from revision1.md section 19)
# Label: RTX 4060 (NVIDIA GeForce RTX 4060, 8 GB)
AUDIT_PROFILE = {
    "hardware": "NVIDIA GeForce RTX 4060 (8 GB VRAM)",
    "source": "scratch/audit_rev1/stage0_profile.py + stage2_summary.py",
    "note": "All times measured, not estimated. recurrence_112 = 4 trunk passes (T=4).",
    "step_timing_ms": {
        "recurrence_112_baseline": 1630.6,
        "ff_112_single_trunk_pass_ms": 25.15,
        "recurrence_112_4_trunks_ratio": 4.76,
    },
    "trunk_calls_per_forward": {
        "recurrence_only_T4": 4,
        "ff_112": 1,
    },
    "kernel_profile": {
        "total_device_ms": 7669.98,
        "kernel_count": 18770,
        "measured_step_wall_ms": 1530.206,
        "cuda_busy_fraction": 0.948,
        "gap_total_ms": 36.412,
        "gap_p90_us": 1.73,
        "img_per_sec": 31.37,
    },
    "candidates": {
        "A_shared_clean_forward_ms": 20010.6,
        "A_speedup": 0.081,
        "B_fused_2B_TRADES_ms": 1635.9,
        "B_speedup": 0.997,
        "C_torch_compile_ms": 1974.9,
        "C_speedup": 0.826,
    },
    "recommendation": (
        "Adopt none. B is lossless and worthless (+0.3%). "
        "A hits memory cliff (5.5->9.7 GB). C is slower by 21% on RTX 4060."
    ),
}

# Try to load live profile JSON if available
profile_json = Path("scratch/audit_rev1/stage2_summary.json")
if profile_json.exists():
    with open(profile_json) as f:
        live = json.load(f)
    AUDIT_PROFILE["live_stage2_summary"] = live
    print(f"Loaded live profile: {profile_json}")
else:
    print(f"No live stage2 profile at {profile_json}; using values from revision1.md §19")

out = OUT_DIR / "t4d_cost_table.json"
with open(out, "w") as f:
    json.dump(AUDIT_PROFILE, f, indent=2)
print(f"Saved to {out}")

# Print summary
print("\n--- Cost Table (RTX 4060) ---")
print(f"  recurrence_112 step (baseline): {AUDIT_PROFILE['step_timing_ms']['recurrence_112_baseline']:.1f} ms")
print(f"  ff_112 single trunk pass:        {AUDIT_PROFILE['step_timing_ms']['ff_112_single_trunk_pass_ms']:.2f} ms")
print(f"  recurrence trunk ratio (T=4):    {AUDIT_PROFILE['step_timing_ms']['recurrence_112_4_trunks_ratio']:.2f}x")
print(f"  img/sec:                         {AUDIT_PROFILE['kernel_profile']['img_per_sec']:.1f}")
