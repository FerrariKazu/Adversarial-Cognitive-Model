#!/usr/bin/env python3
"""Extend docs/rhan_next_roadmap.json stages['2'] with the Stage 2 (HPC, matrix
C) execution plan, per the existing stage-tracking schema. Pulls the HF-synced
copy first (source of truth for runtime verdicts), bumps roadmap_rev, and
syncs back to HF — same upload path the notebook's sync_roadmap_up uses.
"""
import json
import os
import urllib.request

ROADMAP = "docs/rhan_next_roadmap.json"
REPO = "FerrariKazu/rhan-checkpoints-rolling"


def _hf_roadmap():
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/rhan_next_roadmap.json"
    req = urllib.request.Request(url, headers={"User-Agent": "stage2-plan"})
    return json.load(urllib.request.urlopen(req, timeout=90))


def main():
    # 1. Restore the HF-synced roadmap (runtime verdicts from prior sessions).
    hf = _hf_roadmap()
    local_rev = json.load(open(ROADMAP)).get("roadmap_rev", 0)
    if hf.get("roadmap_rev", 0) < local_rev:
        print(f"HF roadmap (rev {hf.get('roadmap_rev')}) is OLDER than local "
              f"(rev {local_rev}) — keeping the local copy.")
    else:
        json.dump(hf, open(ROADMAP, "w"), indent=2, sort_keys=False)
        print(f"Restored HF roadmap (rev {hf.get('roadmap_rev')}) over the "
              f"local copy.")

    roadmap = json.load(open(ROADMAP))

    # 2. Extend stages['2'] with the execution plan (same schema as stages['1']).
    st2 = roadmap["stages"]["2"]
    st2["name"] = ("Pillar 1 — Hierarchical Predictive Coding "
                   "(one level at a time; THIS ROUND = matrix C, HPC-only)")
    st2["run_label"] = "C_hpc_only (HPC-only, w_hpc=0.10)"
    st2["validated_note"] = (
        "2026-08-11: Stage 2 infrastructure code-complete + tested (HPCLevel1 "
        "wiring, gradient-flow + AIS-v1 disable backward-compat tests, "
        "ablation matrix A/B/C/D registry + runner, eval_rhan.py "
        "--ablation-matrix flag, SBR feasibility probe built+dormant). THIS "
        "ROUND trains/evaluates ONLY matrix entry C (HPC-only) via the "
        "notebook Stage 2 block. D (AIS+HPC) stays SCAFFOLDED_NOT_RUN — "
        "dormant until the Stage 2 verdict says HPC is worth combining with "
        "AIS. Verdict recorded to stages['2'].stage2_verdict.")
    st2["execution_plan"] = {
        "notebook": "cloud_setup/colab_notebook_noesis.py (twin: Kaggle_NOESIS.py) — Stage 2 block, identical shape to Stage 1",
        "matrix_entry": "C_hpc_only in rhan_core/ablation/matrix.py: RHANNextConfig(enable_ais=False, enable_hpc=True, hpc_num_levels=1, hpc_error_weight=0.10). Training commands are GENERATED from the registry (runner.train_command) so command<->matrix consistency is enforced, not assumed.",
        "step_a_smoke": "train_rhan_next.py --enable-hpc --hpc-num-levels 1 --w-hpc 0.10 --ckpt-name rhan_next_hpc_only_smoke --max-epochs 15 --target-ckpt checkpoints/rhan_next_ais_v1_halting_only_best.pth --batch-size 16 --accum-steps 16 --diag-json report/rhan_next_hpc_only_smoke_diag.jsonl --force-single-gpu (epochs 1-15 all in phase 1 => epsilon 0.031 only; AIS mechanisms OFF per matrix C — do NOT layer AIS onto HPC)",
        "health_gate": "4 checks; any failure STOPS before the 60-epoch run and writes report/rhan_next_hpc_only_smoke_health.json + roadmap.stages['2'].stage2_gate_verdict. (1) HPC gradient flow: tests/test_hpc_gradient_flow.py MUST pass (hard NOT-detached assertion — the Stage-1 lesson as an automated check, not a manual review); (2) HPC prediction-error trend from the diag: final-epoch hpc_error_mean <= 0.9 * first-epoch (>= 10% decrease) AND max over epochs <= 10x the first-epoch value (never explode); (3) AIS-v1 disable backward-compat: tests/test_hpc_disable_backward_compat.py MUST pass (hpc_num_levels=0 reproduces the validated AIS-v1 forward bit-for-bit — the Stage-0 v12-compat pattern, run as an automated smoke-time check); (4) Pi_D per-class car/truck top-2 (same criterion as Stage 1 — if HPC breaks it, directly comparable to the recon-mod finding). Since HPC is a single additive loss term decoupled from AIS internals, a failure implicates HPCLevel1 itself — NO isolation arms needed; the failure writeup CONFIRMS this explicitly rather than assuming it.",
        "step_b_full": "same trainer, --ckpt-name rhan_next_hpc_only, --max-epochs 60, same base (validated AIS-v1 halting-only checkpoint); curriculum (1-20 @0.031, 21-40 @0.062, 41-60 @0.094) byte-identical to train_rhan_v11.py; NEVER --force-restart (mandatory HF resume gate + verify_no_restart); HF rolling+best sync with the best==rolling eval-target verification (_eval_target_note) applied from the start — the 2026-08-08/10 metadata lesson, not rediscovered.",
        "step_c_eval": "THREE-WAY 5-seed matched eval via the new flag: eval_rhan.py --ablation-matrix A_baseline B_ais_only C_hpc_only --seeds 41 42 43 44 45 --eps-list 0.000 0.094 --pgd-steps 50 --n-samples 300 --batch-size 64 --baseline-label trades_large_baseline --output-dir report/sweep_stage2_hpc_only — vs A (static TRADES baseline) AND vs B (AIS-v1 halting-only): this tells us whether HPC adds anything AIS didn't already provide (C-vs-B comparison recorded in the verdict). PGD-100 leg at eps=0.094 for the masking check on all three.",
        "step_c2_rule": "Seed extension to 8 (46-48, eps=0.094, both legs, then scripts/merge_stage1_seed_extension.py) IF AND ONLY IF the 5-seed verdict is BORDERLINE (positive but NOT significant). Cleanly significant or cleanly null at 5 seeds => no extension (extension resolves ambiguity, it does not hunt for significance — the Stage 1 'no third extension' discipline applies at the FIRST extension here too).",
        "verdict_recorded_to": "roadmap.stages['2'].stage2_verdict (parsed from report/sweep_stage2_hpc_only[_merged]/eval_provenance.json by the notebook; three-way table + crossover vs baseline + C-vs-B + masking + eval_target_note; null result = valid reportable outcome).",
        "stage2_headline_rule": "Record whatever the numbers say, honestly. Do NOT cite D or SBR this round; D remains SCAFFOLDED_NOT_RUN and enable_sbr stays locked.",
    }
    st2["ablation_matrix"] = {
        "registry": "rhan_core/ablation/matrix.py (single source of truth for A/B/C/D)",
        "runner": "rhan_core/ablation/runner.py — resolve(key) -> (config, checkpoint, ckpt_name); train_command(key) generates the trainer argv; eval_specs(keys) builds --ckpt-specs",
        "eval_flag": "phase2_attacks/eval_rhan.py --ablation-matrix [keys...] (VALIDATED always; PENDING once its checkpoint exists; SCAFFOLDED_NOT_RUN never)",
        "this_round": "ONLY C_hpc_only trains. A and B already validated. D_ais_plus_hpc = SCAFFOLDED_NOT_RUN (resolves + tested in tests/test_ablation_matrix.py, no training job).",
        "test": "tests/test_ablation_matrix.py — all 4 entries resolve to valid, distinct, non-crashing configs.",
    }
    st2["sbr_feasibility_probe"] = {
        "status": "BUILT + SMOKE-TESTED (2026-08-11) — PAUSED, not interpreted",
        "module": "rhan_core/beliefs/experimental/sbr_feasibility.py (standalone, read-only; explicitly OUTSIDE the config/gating system)",
        "smoke": "runs against the Stage 1 AIS-v1 checkpoint (379/379 keys, stem tap (B,768,12,12)); outputs report/sbr_feasibility/ (probe_summary.json + PNG montages). Slot head UNTRAINED (feasibility of the machinery, not a learned result).",
        "constraints": "never imports/modifies RHANNextConfig (constructs RHANNext(**embedded_config_dict)); never sets enable_sbr=True; never touches the training loop; output never merged into the ablation matrix.",
        "gate": "Interpretation deferred until HPC (Stage 2) validates — per the agreed sequencing, do NOT schedule or cite before then.",
    }
    st2["code_complete_acceptance"] = st2.get("code_complete_acceptance", []) + [
        "HPCLevel1 wiring module (rhan_core/predictive_coding/hpc_level1.py): edge-map target, documented single tap point (foveal_crop), returns (prediction, error, error_map); error NOT detached",
        "w_hpc = 0.10 as a SEPARATE slot from w_recon (config hpc_error_weight + --w-hpc); hpc_num_levels default 0",
        "tests/test_hpc_gradient_flow.py (hard NOT-detached assertion + param grads) and tests/test_hpc_disable_backward_compat.py (hpc_num_levels=0 == AIS-v1 forward) PASS",
        "ablation matrix A/B/C/D registry + runner + eval_rhan.py --ablation-matrix flag; tests/test_ablation_matrix.py PASS",
        "Stage 2 notebook block (colab + Kaggle twins, byte-identical): smoke -> 4-check gate -> 60-epoch -> 3-way eval -> C2 rule -> verdict",
        "SBR feasibility probe built, smoke-tested, PAUSED (not interpreted)"
    ]

    # 3. Fix the §8.1a reference (ARCHITECTURE.md renumbered §8.1a -> §9.1a).
    for key, val in list(roadmap["stages"]["1"]["execution_plan"].items()):
        if isinstance(val, str) and "§8.1a" in val:
            roadmap["stages"]["1"]["execution_plan"][key] = val.replace(
                "§8.1a", "§9.1a")

    # 4. Bump the revision (plan change = rev 4).
    roadmap["roadmap_rev"] = int(roadmap.get("roadmap_rev", 0)) + 1
    with open(ROADMAP, "w") as f:
        json.dump(roadmap, f, indent=2, sort_keys=False)
    print(f"✓ docs/rhan_next_roadmap.json updated (rev {roadmap['roadmap_rev']}): "
          f"stages['2'] execution plan + ablation matrix + SBR probe + gate.")

    # 5. Sync to HF (same path the notebook uses).
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("WARNING: HF_TOKEN unset — local update only, NOT synced to HF.")
        return
    from huggingface_hub import HfApi
    HfApi(token=token).upload_file(
        path_or_fileobj=ROADMAP, path_in_repo="rhan_next_roadmap.json",
        repo_id=REPO, repo_type="dataset", token=token)
    print(f"✓ roadmap synced to HF ({REPO})")


if __name__ == "__main__":
    main()
