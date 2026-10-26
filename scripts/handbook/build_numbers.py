#!/usr/bin/env python3
"""Build the handbook's single numeric source of truth.

Reads only recorded artifacts under ``report/`` and the code constants in
``training/`` and ``gen2_foundation/``; writes ``docs/rhan_nxa/data/``:

    numbers.json          every consequential value + provenance
    numbers.schema.json   the schema the registry is validated against
    claims.json           claim -> metric -> artifact -> section graph
    training_lineage.json checkpoint lineage manifest

Determinism: the same artifacts produce a byte-identical registry.  There is no
timestamp and no absolute path in the output, so a diff means a value or its
provenance changed.

Usage:
    .venv/bin/python scripts/handbook/build_numbers.py           # write
    .venv/bin/python scripts/handbook/build_numbers.py --check   # verify only
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "rhan_nxa" / "data"
REGISTRY_VERSION = "rhan-handbook-v3.1"
SCHEMA_VERSION = "1.0"

# --------------------------------------------------------------------------- #
# reuse the existing verifier rather than duplicating its arithmetic
# --------------------------------------------------------------------------- #
_spec = importlib.util.spec_from_file_location(
    "verify_claims", ROOT / "docs" / "rhan_nxa" / "verify_claims.py")
vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vc)

UNITS = {
    "percent", "percentage_points", "gmac", "ratio", "count", "epochs",
    "steps", "normalized_eps", "pixel_255", "scalar", "mcnemar_p", "t_statistic",
}
STATUSES = {
    "VERIFIED", "SUPPORTED_WITH_LIMITATIONS", "HISTORICAL", "RETRACTED",
    "UNRESOLVED", "NOT_REPRODUCIBLE", "CONCEPTUAL",
}
PROVENANCE_REQUIRED = {"VERIFIED", "SUPPORTED_WITH_LIMITATIONS", "HISTORICAL"}


def fmt(value, unit: str, precision: int, signed: bool = False) -> str:
    """Render a machine value.  The registry stores both; the renderer uses
    ``display`` and the validator recomputes it from ``value``."""
    if unit == "percent":
        return f"{value * 100:.{precision}f} %"
    if unit == "percentage_points":
        return f"{value:+.{precision}f} pp" if signed else f"{value:.{precision}f} pp"
    if unit == "gmac":
        return f"{value:.{precision}f} GMAC"
    if unit == "ratio":
        return f"{value:.{precision}f}×"
    if unit == "count":
        return f"{int(round(value)):,}"
    if unit in {"epochs", "steps"}:
        return f"{int(round(value))}"
    if unit == "normalized_eps":
        return f"{value:.{precision}f}"
    if unit == "pixel_255":
        return f"{value:.{precision}f}/255"
    if unit == "mcnemar_p":
        return f"{value:.{precision}g}"
    if unit == "t_statistic":
        return f"{value:.{precision}f}"
    return f"{value:.{precision}f}"


def metric(mid, value, unit, precision, status, description, *,
           artifacts=(), code=(), derivation=None, limitations=(),
           signed=False, **extra):
    rec = {
        "value": value,
        "unit": unit,
        "display": fmt(value, unit, precision, signed),
        "display_precision": precision,
        "status": status,
        "description": description,
        "source_artifacts": list(artifacts),
        "source_code": list(code),
        "derivation": derivation,
        "limitations": list(limitations),
    }
    if signed:
        rec["signed_display"] = True
    rec.update(extra)
    return mid, rec


def build_metrics() -> dict:
    ff = vc.load(vc.FF)
    ro = vc.load(vc.RO)
    gc = vc.load(vc.GC)
    audit = vc.load(vc.S1)
    d8 = vc.load("report/d8_adaptive_attacks/g2_d8_combined.json")
    pb = vc.load("report/handbook/data_extracts/param_breakdown.json")
    j2 = vc.load("reconciliation/attack_reconciliation_20261021_135827.json")
    ctl = vc.load("report/controls/controls_eval_20261021_202939.json")

    ffs, ros, gcs = ff["sweep_summary"], ro["sweep"], gc["sweep"]
    fin = ff["final_verification"]
    p = vc.paired_stats()
    deg = {d["arm"]: d for d in vc.degradations()}
    macs_ff = vc.analytic_macs(112, 1, refine=0)["total"]
    macs_rec = vc.analytic_macs(56, 4, gist_pass=True, refine=2)["total"]
    ck = audit["checkpoints"]
    paired = audit["paired_comparisons"]

    ff_art = "report/eval/eval_ff_112_20261018_123413.json"
    ro_art = "report/d3_final_eval/g2_d3_recurrence_only_final.json"
    gc_art = "report/d3_final_eval/g2_d3_gen1_core_best_clean.json"
    audit_art = "report/audit_rev1/step1_attack_audit.json"
    d8_art = "report/d8_adaptive_attacks/g2_d8_combined.json"
    pb_art = "report/handbook/data_extracts/param_breakdown.json"

    M = {}
    add = lambda *a, **k: M.__setitem__(*metric(*a, **k))

    # ---- protocol -------------------------------------------------------- #
    add("protocol.eps_norm_train", 0.094, "normalized_eps", 3, "VERIFIED",
        "Training and headline-evaluation L-infinity radius, in ImageNet-normalised "
        "pixel coordinates.",
        artifacts=[audit_art, ff_art],
        code=["training/curriculum_gen2.py::norm_to_pixel_eps",
              "training/train_gen2_foundation.py::run_pgd_ce_eval"])
    add("protocol.eps_pixel_255", 0.094 * vc.MEAN_STD * 255.0, "pixel_255", 2,
        "VERIFIED", "The training radius expressed in 8-bit pixel units.",
        artifacts=[audit_art], derivation="eps_norm * MEAN_STD * 255, MEAN_STD = 0.226")
    add("protocol.eps_norm_8_255", (8 / 255) / vc.MEAN_STD, "normalized_eps", 4,
        "VERIFIED", "8/255 pixel converted to normalised units.",
        code=["training/eval_gen2_foundation.py::run_final_checkpoint_eval"],
        derivation="(8/255) / 0.226")
    add("protocol.pgd_inner_steps", 4, "steps", 0, "VERIFIED",
        "Inner-loop PGD steps used by the TRADES training attack.",
        code=["training/train_gen2_foundation.py::run_pgd_kl_attack"])
    add("protocol.trades_w", 0.55, "scalar", 2, "VERIFIED",
        "Global multiplier applied to the TRADES objective.",
        code=["training/train_gen2_foundation.py::compute_trades_loss"])
    add("protocol.trades_beta", 2.5, "scalar", 1, "VERIFIED",
        "Final KL weight beta; the schedule ramps 2.0 -> 2.5.",
        code=["training/curriculum_gen2.py"])
    add("protocol.eval_n_audit", 500, "count", 0, "VERIFIED",
        "Images in the matched-epsilon attack audit.", artifacts=[audit_art])
    add("protocol.eval_n_sweep", 300, "count", 0, "VERIFIED",
        "Images per validation subset in the 8-seed sweeps.",
        artifacts=[ff_art, ro_art, gc_art])
    add("protocol.eval_n_d8", 300, "count", 0, "VERIFIED",
        "Images in the adaptive-attack (D8) evaluation.", artifacts=[d8_art])
    add("protocol.eval_seeds", 8, "count", 0, "VERIFIED",
        "Independent validation subsets behind each sweep mean.",
        artifacts=[ff_art, ro_art, gc_art])

    # ---- 8-seed sweeps (PGD-10, corrected alpha) ------------------------- #
    sweeps = {
        "ff_112": ("ff112", ffs, "clean (eps=0.0)", "norm_eps=0.094", ff_art),
        "recurrence_only": ("rec", ros, "clean", "norm_eps_0094", ro_art),
        "gen1_core": ("gen1", gcs, "clean", "norm_eps_0094", gc_art),
    }
    for arm, (prefix, art, kclean, kadv, path) in sweeps.items():
        add(f"sweep.{prefix}.clean", art[kclean]["mean"], "percent", 2, "VERIFIED",
            f"{arm} clean accuracy, mean of 8 x 300-image validation subsets.",
            artifacts=[path], code=["training/eval_gen2_foundation.py"])
        add(f"sweep.{prefix}.pgd10_0094", art[kadv]["mean"], "percent", 2, "VERIFIED",
            f"{arm} robust accuracy under PGD-10 at eps_norm 0.094, alpha = 2.5*eps/10.",
            artifacts=[path], code=["training/train_gen2_foundation.py::run_pgd_ce_eval"])
        add(f"sweep.{prefix}.drop_ppg", deg[arm]["drop_pp"], "percentage_points", 2,
            "VERIFIED", f"Clean-minus-robust drop for {arm} at eps_norm 0.094.",
            artifacts=[path], derivation="clean - pgd10_0094, in pp")
        add(f"sweep.{prefix}.retention", deg[arm]["retained"], "ratio", 3, "VERIFIED",
            f"{arm} robust/clean retention at eps_norm 0.094 (PGD-10).",
            artifacts=[path], derivation="pgd10_0094 / clean",
            limitations=["Numerator and denominator from the same sweep, so n and "
                         "protocol match; PGD-10 is not the strongest attack."])
        add(f"sweep.{prefix}.clean_sd", art[kclean]["std"], "scalar", 4, "VERIFIED",
            f"Standard deviation of {arm} clean accuracy across the 8 subsets.",
            artifacts=[path])

    add("sweep.clean_gap_ff_minus_rec",
        (ffs["clean (eps=0.0)"]["mean"] - ros["clean"]["mean"]) * 100,
        "percentage_points", 1, "VERIFIED",
        "ff_112 clean accuracy minus recurrence_only clean accuracy, 8-seed sweeps.",
        artifacts=[ff_art, ro_art], derivation="clean_ff - clean_rec, in pp",
        limitations=["Means from separate 8-subset sweeps, so this is a difference of "
                     "means, not a paired difference."])
    add("paired.ff_minus_rec_ppg", p["mean_pp"], "percentage_points", 2, "VERIFIED",
        "Paired ff_112 minus recurrence_only advantage at eps_norm 0.094 (PGD-10), "
        "over 8 seeds.",
        artifacts=[ff_art, ro_art], derivation="mean of per-seed differences, pp",
        signed=True,
        limitations=["PGD-10 only; the advantage is not replicated by the library "
                     "APGD-CE audit, which does not include ff_112."])
    add("paired.ci95_low", p["ci95_low"], "percentage_points", 2, "VERIFIED",
        "Lower bound of the paired 95 % interval for the same difference.",
        artifacts=[ff_art, ro_art], derivation="paired t(7) interval")
    add("paired.ci95_high", p["ci95_high"], "percentage_points", 2, "VERIFIED",
        "Upper bound of the paired 95 % interval.",
        artifacts=[ff_art, ro_art], derivation="paired t(7) interval")
    add("paired.t_statistic", p["t_paired"], "t_statistic", 2, "VERIFIED",
        "Paired t statistic for the 8-seed difference.",
        artifacts=[ff_art, ro_art], derivation="mean / (sd / sqrt(n)) on per-seed diffs")
    add("paired.df", p["df"], "count", 0, "VERIFIED",
        "Degrees of freedom for the paired test.", artifacts=[ff_art, ro_art])

    # ---- matched-epsilon audit (n = 500) --------------------------------- #
    for key, short in (("g2_gist_only_final", "gist"),
                       ("recurrence_only_final", "rec"),
                       ("gen1_core_best_clean", "gen1")):
        a = ck[key]["accuracies"]
        for atk, mid in (("clean", "clean"), ("inhouse_pgd_50", "pgd50"),
                         ("lib_apgd_ce_100", "apgd_ce")):
            add(f"audit.{short}.{mid}", a[atk], "percent", 2, "VERIFIED",
                f"{key} accuracy under {atk} at eps_norm 0.094, n = 500, seed 42.",
                artifacts=[audit_art], code=["scratch/audit_rev1/step1_attack_audit.py"].copy())
    add("audit.rec.pgd50_corrected", ck["recurrence_only_final"]["accuracies"]["pgd50_alpha_2_5x"],
        "percent", 2, "VERIFIED",
        "recurrence_only under PGD-50 with the corrected step rule alpha = 2.5*eps/50.",
        artifacts=[audit_art], code=["training/train_gen2_foundation.py::run_pgd_ce_eval"])
    add("audit.rec.pgd50_corrected_diff_vs_apgd",
        (ck["recurrence_only_final"]["accuracies"]["pgd50_alpha_2_5x"]
         - ck["recurrence_only_final"]["accuracies"]["lib_apgd_ce_100"]) * 100,
        "percentage_points", 1, "VERIFIED",
        "Residual gap between the corrected in-house PGD-50 and library APGD-CE-100.",
        artifacts=[audit_art], derivation="pgd50_corrected - apgd_ce, in pp", signed=True)
    add("audit.paired_rec_minus_gen1_apgd", paired["lib_apgd_ce_100"]["diff"] * 100,
        "percentage_points", 1, "SUPPORTED_WITH_LIMITATIONS",
        "recurrence_only minus gen1_core under APGD-CE-100 at eps 0.094; not "
        "significant at n = 500.",
        artifacts=[audit_art], signed=True,
        limitations=[f"McNemar p = {paired['lib_apgd_ce_100']['p_value']:.3g}; the "
                     "sample cannot distinguish this from zero."])

    # ---- adaptive attacks (D8, n = 300) ---------------------------------- #
    add("d8.rec.apgd_ce", d8["recurrence_only_final"]["attacks"]["apgd_ce_100"]["acc"],
        "percent", 2, "VERIFIED",
        "recurrence_only under torchattacks APGD-CE-100 at eps 0.094, n = 300.",
        artifacts=[d8_art], code=["eval_scripts/g2_d8_adaptive_attacks.py"])
    add("d8.gen1.apgd_ce", d8["gen1_core_best_clean"]["attacks"]["apgd_ce_100"]["acc"],
        "percent", 2, "VERIFIED",
        "gen1_core under the same D8 APGD-CE-100 protocol.",
        artifacts=[d8_art], code=["eval_scripts/g2_d8_adaptive_attacks.py"])
    add("d8.rec.mi_fgsm", d8["recurrence_only_final"]["attacks"]["mi_fgsm_100"]["acc"],
        "percent", 2, "VERIFIED", "recurrence_only under MI-FGSM-100.",
        artifacts=[d8_art])
    add("d8.gen1.mi_fgsm", d8["gen1_core_best_clean"]["attacks"]["mi_fgsm_100"]["acc"],
        "percent", 2, "VERIFIED", "gen1_core under MI-FGSM-100.", artifacts=[d8_art])
    add("d8.rec.diversity_pgd", d8["recurrence_only_final"]["attacks"]["diversity_pgd_5x20"]["acc"],
        "percent", 2, "VERIFIED", "recurrence_only under 5-restart diversity PGD.",
        artifacts=[d8_art])
    add("d8.gen1.diversity_pgd", d8["gen1_core_best_clean"]["attacks"]["diversity_pgd_5x20"]["acc"],
        "percent", 2, "VERIFIED", "gen1_core under 5-restart diversity PGD.",
        artifacts=[d8_art])

    # ---- ff_112 final verification -------------------------------------- #
    add("final.ff.clean", fin["final_clean_acc"], "percent", 1, "VERIFIED",
        "ff_112 clean accuracy on the 500-image final-verification slice.",
        artifacts=[ff_art])
    add("final.ff.pgd50", fin["final_pgd50_acc"], "percent", 1, "VERIFIED",
        "ff_112 under PGD-50 at eps_norm 0.094, alpha = 2.5*eps/50, fp32.",
        artifacts=[ff_art], code=["training/train_gen2_foundation.py::run_pgd_ce_eval"])
    add("final.ff.eot", fin["final_eot_acc"], "percent", 1, "SUPPORTED_WITH_LIMITATIONS",
        "ff_112 under EOT-PGD (n_eot = 5, 20 steps) at eps_norm 0.094.",
        artifacts=[ff_art], code=["gen2_foundation/eot.py::eot_pgd_attack"],
        limitations=["Below the registered EOT strength (n_eot = 20, 50 steps)."])
    add("final.ff.retention_pgd50", fin["final_pgd50_acc"] / fin["final_clean_acc"],
        "ratio", 3, "VERIFIED",
        "ff_112 robust/clean retention under PGD-50 on the same 500 images.",
        artifacts=[ff_art], derivation="pgd50 / clean")
    add("final.rec.retention_pgd50",
        ck["recurrence_only_final"]["accuracies"]["pgd50_alpha_2_5x"]
        / ck["recurrence_only_final"]["accuracies"]["clean"],
        "ratio", 3, "VERIFIED",
        "recurrence_only robust/clean retention under the corrected PGD-50 on the "
        "same 500-image protocol.",
        artifacts=[audit_art], derivation="pgd50_corrected / clean")

    # The invalid pair.  Status RETRACTED, with the code-level reason.
    add("final.ff.apgd_ce_invalid", fin["final_apgd_ce_acc"], "percent", 1, "RETRACTED",
        "Stored flat key final_apgd_ce_acc. Not a valid measurement at any radius.",
        artifacts=[ff_art],
        code=["training/eval_gen2_foundation.py@938ada3^ (eps=0.031 passed on "
              "normalised tensors)", "torchattacks.APGD clamps to [0,1]"],
        limitations=[
            "git show 938ada3^:training/eval_gen2_foundation.py line 262 passes "
            "eps=0.031 to torchattacks on already-normalised inputs, i.e. a radius "
            "one third of the training radius 0.094, not the 8/255 (0.1389) reading "
            "recorded in the v3.0 handbook.",
            "torchattacks clamps the perturbed normalised tensor to [0,1]. A valid "
            "normalised ImageNet input spans roughly [-2.12, 2.64], so the clamp "
            "destroys the input rather than projecting it onto the threat set.",
            "The artifact stores no eps field for this block, so the radius cannot be "
            "recovered from the artifact; it is recovered from the code revision that "
            "produced it.",
            "Therefore the value supports neither an ff_112-at-0.094 claim nor an "
            "ff_112-at-8/255 claim."])
    add("final.ff.apgd_t_invalid", fin["final_apgd_t_acc"], "percent", 1, "RETRACTED",
        "Stored final_apgd_t_acc, produced by the same pre-fix harness as "
        "final_apgd_ce_acc.", artifacts=[ff_art],
        limitations=["Same two defects: eps 0.031 in normalised units, and a [0,1] "
                     "clamp applied to normalised tensors."])

    # ---- J2 reconciled battery (n = 1000, seed 42, 5.417/255) ------------ #
    # The four arms the J2 reconciliation evaluated under one pixel-space
    # harness.  Every row was produced by the same loader, the same budget and
    # the same wrapper; the realised Linf is asserted in the script.
    J2_ARM = {
        "ff112": "ff_112", "rec": "recurrence_only",
        "gist": "g2_gist_only", "gen1": "gen1_core",
    }
    J2_ATTACK = {
        "clean": ("clean_acc", "clean accuracy, no perturbation"),
        "pgd50": ("pgd50_acc", "PGD-50, alpha = 2.5*eps/50, one random start"),
        "apgd_ce_fp32": ("apgd_ce_fp32_acc", "torchattacks APGD-CE, 100 steps, fp32"),
        "apgd_ce_amp": ("apgd_ce_amp_acc", "same attack through the autocast wrapper"),
        "pgd200_3r": ("pgd200_3restarts_acc", "PGD-200, alpha = eps/10, three restarts"),
        "apgd_dlr": ("apgd_dlr_acc", "torchattacks APGD-DLR, 100 steps"),
    }
    for short, arm in J2_ARM.items():
        r = j2["results"][arm]
        for mid, (key, what) in J2_ATTACK.items():
            add(f"j2.{short}.{mid}", r[key], "percent", 2, "VERIFIED",
                f"{arm}: {what} on the J2 reconciliation slice.",
                artifacts=["reconciliation/attack_reconciliation_20261021_135827.json"],
                code=["cloud_setup/Kaggle_ATTACK_RECONCILIATION.py"],
                derivation="per-sample correct / 1000")
        add(f"j2.{short}.aa_core", r["autoattack_core_acc"], "percent", 2,
            "SUPPORTED_WITH_LIMITATIONS",
            f"{arm}: samples correct under both APGD-CE and APGD-DLR (the script's "
            "'AutoAttack core').",
            artifacts=["reconciliation/attack_reconciliation_20261021_135827.json"],
            code=["cloud_setup/Kaggle_ATTACK_RECONCILIATION.py"],
            derivation="apgd_ce_fp32 AND apgd_dlr, per sample",
            limitations=["This is the intersection of two APGD runs, not the "
                         "AutoAttack ensemble (APGD-CE + APGD-DLR + FAB + square). It "
                         "must not be quoted as AutoAttack."])
        add(f"j2.{short}.worst", r["worst_case_all_acc"], "percent", 2, "VERIFIED",
            f"{arm}: worst-case robust accuracy, correct under every listed attack.",
            artifacts=["reconciliation/attack_reconciliation_20261021_135827.json"],
            code=["cloud_setup/Kaggle_ATTACK_RECONCILIATION.py"],
            derivation="AND over PGD-50, APGD-CE fp32, APGD-CE autocast, PGD-200x3, "
                       "APGD-DLR (AA-core is their intersection and adds nothing)",
            limitations=["Worst case over the six attacks listed; not a certified "
                         "radius and not the full AutoAttack suite."])
        add(f"j2.{short}.retention", r["worst_case_all_acc"] / r["clean_acc"],
            "ratio", 3, "VERIFIED",
            f"{arm}: worst-case robust accuracy as a share of clean accuracy.",
            artifacts=["reconciliation/attack_reconciliation_20261021_135827.json"],
            derivation="j2.%s.worst / j2.%s.clean" % (short, short),
            limitations=["A ratio of two accuracies on one sample, not a certified "
                         "robustness ratio."])
        if "transfer_from_ff112_acc" in r:
            add(f"j2.{short}.transfer", r["transfer_from_ff112_acc"], "percent", 2,
                "VERIFIED",
                f"{arm}: accuracy on ff_112 PGD-50 adversarial examples after bilinear "
                "resize to the arm's input resolution.",
                artifacts=["reconciliation/attack_reconciliation_20261021_135827.json"],
                code=["cloud_setup/Kaggle_ATTACK_RECONCILIATION.py::Transfer Attack "
                      "Evaluation"],
                limitations=["Transfer is asymmetric: only ff_112 serves as surrogate. "
                             "The resize changes the perturbation, so this is not a "
                             "budget-matched transfer claim."])
    add("j2.protocol.n", j2["eval_n"], "count", 0, "VERIFIED",
        "Images in the J2 reconciliation slice (first 1000 of the shuffled validation "
        "split, seed 42).",
        artifacts=["reconciliation/attack_reconciliation_20261021_135827.json"],
        code=["cloud_setup/Kaggle_ATTACK_RECONCILIATION.py::build_raw_loader"])
    add("j2.protocol.eps_pixel_255", j2["target_eps_pixel_255"], "pixel_255", 3,
        "VERIFIED", "Budget of every J2 attack, in [0,1] pixel units x255.",
        artifacts=["reconciliation/attack_reconciliation_20261021_135827.json"],
        code=["cloud_setup/Kaggle_ATTACK_RECONCILIATION.py::EPS_PIXEL"])

    # ---- 56-px controls (C1 epoch-matched, C2 single-pass) --------------- #
    cir = ctl["results"]
    add("ctrl.n", ctl["eval_n"], "count", 0, "VERIFIED",
        "Images in the controls evaluation; the same slice as the J2 battery.",
        artifacts=["report/controls/controls_eval_20261021_202939.json"],
        code=["cloud_setup/Kaggle_CONTROLS_56PX.py"])
    add("ctrl.c1.epochs_extra", 15, "epochs", 0, "VERIFIED",
        "Epochs C1 (g2_gist_only_cont) trained on top of g2_gist_only_final, at rung 3.",
        code=["cloud_setup/Kaggle_CONTROLS_56PX.py::CONTROL 1"])
    add("ctrl.c1.epochs_total", 45, "epochs", 0, "VERIFIED",
        "Cumulative epochs behind C1: 30 inherited from g2_gist_only plus 15.",
        derivation="lineage.epochs_gist + ctrl.c1.epochs_extra",
        code=["cloud_setup/Kaggle_CONTROLS_56PX.py::CONTROL 1"])
    for short, key in (("c1", "g2_gist_only_cont"), ("c2", "ff_56"),
                       ("rec", "recurrence_only")):
        r = cir[key]
        for mid in ("clean", "pgd50", "apgd_ce", "apgd_dlr", "worst"):
            kk = "worst_case_acc" if mid == "worst" else f"{mid}_acc"
            add(f"ctrl.{short}.{mid}", r[kk], "percent", 2, "VERIFIED",
                f"{key}: {mid} accuracy on the controls slice (n=1000, seed 42, "
                "5.417/255).",
                artifacts=["report/controls/controls_eval_20261021_202939.json"],
                code=["cloud_setup/Kaggle_CONTROLS_56PX.py"],
                derivation="per-sample correct / 1000")
    for short, key in (("c1", "g2_gist_only_cont"), ("c2", "ff_56"),
                       ("rec", "recurrence_only")):
        add(f"ctrl.{short}.retention",
            cir[key]["worst_case_acc"] / cir[key]["clean_acc"], "ratio", 3, "VERIFIED",
            f"{key}: worst-case accuracy as a share of clean accuracy.",
            artifacts=["report/controls/controls_eval_20261021_202939.json"],
            derivation=f"ctrl.{short}.worst / ctrl.{short}.clean")
    add("ctrl.rec_minus_c1_clean", (cir["recurrence_only"]["clean_acc"]
        - cir["g2_gist_only_cont"]["clean_acc"]) * 100, "percentage_points", 2, "VERIFIED",
        "recurrence_only minus its epoch-matched control on clean accuracy, same 1000 "
        "images.",
        artifacts=["report/controls/controls_eval_20261021_202939.json"],
        derivation="ctrl.rec.clean - ctrl.c1.clean", signed=True)
    add("ctrl.rec_minus_c1_worst", (cir["recurrence_only"]["worst_case_acc"]
        - cir["g2_gist_only_cont"]["worst_case_acc"]) * 100, "percentage_points", 2,
        "VERIFIED", "recurrence_only minus its epoch-matched control on worst-case "
        "robust accuracy, same 1000 images and the same three attacks (PGD-50, "
        "APGD-CE, APGD-DLR).",
        artifacts=["report/controls/controls_eval_20261021_202939.json"],
        derivation="ctrl.rec.worst - ctrl.c1.worst", signed=True,
        limitations=["Paired over images but no per-sample outcomes were persisted, so "
                     "no McNemar test and no confidence interval can be computed for "
                     "this contrast from the stored artifact."])
    add("ctrl.c2_minus_rec_worst", (cir["ff_56"]["worst_case_acc"]
        - cir["recurrence_only"]["worst_case_acc"]) * 100, "percentage_points", 2, "VERIFIED",
        "ff_56 minus recurrence_only on worst-case robust accuracy, same slice.",
        artifacts=["report/controls/controls_eval_20261021_202939.json"],
        derivation="ctrl.c2.worst - ctrl.rec.worst", signed=True,
        limitations=["ff_56 trains at rung 3 from a DINOv2 start without the clean "
                     "warm-up, so its collapse confounds recipe with architecture."])
    add("j2.ff112_minus_rec_worst", (j2["results"]["ff_112"]["worst_case_all_acc"]
        - j2["results"]["recurrence_only"]["worst_case_all_acc"]) * 100,
        "percentage_points", 2, "VERIFIED",
        "ff_112 minus recurrence_only on worst-case accuracy over six attacks, same "
        "1000 images.",
        artifacts=["reconciliation/attack_reconciliation_20261021_135827.json"],
        derivation="j2.ff112.worst - j2.rec.worst", signed=True)
    add("j2.ff112_minus_rec_clean", (j2["results"]["ff_112"]["clean_acc"]
        - j2["results"]["recurrence_only"]["clean_acc"]) * 100,
        "percentage_points", 2, "VERIFIED",
        "ff_112 minus recurrence_only on clean accuracy, same 1000 images.",
        artifacts=["reconciliation/attack_reconciliation_20261021_135827.json"],
        derivation="j2.ff112.clean - j2.rec.clean", signed=True)

    # ---- closure suite: one environment, six arms ----------------------- #
    # Everything in this block comes from a single transcription artifact,
    # report/closure/closure_run_2461bc2.json, because the run's own output
    # was supplied by the repository owner rather than exported into the tree.
    # The status is therefore SUPPORTED_WITH_LIMITATIONS, not VERIFIED, and
    # every cell that disagrees with a locally recorded artifact is listed in
    # that artifact's divergences block.
    cl = json.loads((ROOT / "report/closure/closure_run_2461bc2.json")
                    .read_text(encoding="utf-8"))
    cl_art = "report/closure/closure_run_2461bc2.json"
    CL_LIM = ["Transcribed from the closure run's printed output (environment "
              "2461bc2); the raw Kaggle export is not stored in this repository, so "
              "the value is not re-derivable from a persisted artifact."]
    CL_CELL = {
        "clean_5k": "clean accuracy, 5,000-image validation set",
        "clean_1k": "clean accuracy, canonical 1,000-image slice",
        "pgd50": "PGD-50 robust accuracy, 1,000-image slice",
        "apgd_ce": "APGD-CE (100 steps) robust accuracy, 1,000-image slice",
        "apgd_dlr": "APGD-DLR (100 steps) robust accuracy, 1,000-image slice",
        "worst": "worst-case robust accuracy over the four attacks, 1,000-image slice",
    }
    div = {(d["arm"], d["cell"]): d
           for d in cl["verification_in_repo"]["divergences"]}

    add("closure.protocol.n_clean", cl["protocol"]["clean_5k_n"], "count", 0,
        "SUPPORTED_WITH_LIMITATIONS",
        "Images in the closure suite clean evaluation.", artifacts=[cl_art],
        limitations=CL_LIM)
    add("closure.protocol.n_robust", cl["protocol"]["robust_n"], "count", 0,
        "SUPPORTED_WITH_LIMITATIONS",
        "Images in the closure suite adversarial slice.", artifacts=[cl_art],
        limitations=CL_LIM)
    add("closure.protocol.eps_pixel_255", cl["protocol"]["eps_pixel_255"],
        "pixel_255", 3, "SUPPORTED_WITH_LIMITATIONS",
        "Budget of every closure suite attack, in [0, 1] pixel units x255.",
        artifacts=[cl_art], limitations=CL_LIM,
        derivation="the same 5.417/255 radius the J2 battery used")

    for a in cl["headline"]:
        for cell, what in CL_CELL.items():
            lims = list(CL_LIM)
            d = div.get((a["id"], cell))
            if d:
                lims.append(
                    f"Differs from {'/'.join(f'{v:.4g}' for v in d['recorded'])} in "
                    f"canonical_headline_preflight.json or controls_eval by "
                    f"{d['delta_pp']:+.2f} pp; both values are printed in Ch. 26.10 "
                    f"rather than reconciled.")
            add(f"closure.{a['id']}.{cell}", a[cell], "percent", 2,
                "SUPPORTED_WITH_LIMITATIONS",
                f"{a['label']}: {what}, closure suite (n and protocol per cell).",
                artifacts=[cl_art], limitations=lims,
                derivation="per-sample correct / evaluated set, printed by the run")

    for row in cl["mcnemar"]:
        rid = row["id"]
        n_set = 5000 if "5k" in rid else 1000
        add(f"closure.mcnemar.{rid}.delta_pp", row["delta_pp"],
            "percentage_points", 2, "SUPPORTED_WITH_LIMITATIONS",
            f"{row['a']} minus {row['b']} on the {row['sample']} set.",
            artifacts=[cl_art], signed=True, limitations=CL_LIM,
            derivation="(a_only - b_only) / n x 100, printed by the run")
        add(f"closure.mcnemar.{rid}.a_only", row["a_only"], "count", 0,
            "SUPPORTED_WITH_LIMITATIONS",
            f"{row['a']} correct and {row['b']} wrong, {row['sample']} set.",
            artifacts=[cl_art], limitations=CL_LIM)
        add(f"closure.mcnemar.{rid}.b_only", row["b_only"], "count", 0,
            "SUPPORTED_WITH_LIMITATIONS",
            f"{row['b']} correct and {row['a']} wrong, {row['sample']} set.",
            artifacts=[cl_art], limitations=CL_LIM)
        add(f"closure.mcnemar.{rid}.n_disc", row["n_discordant"], "count", 0,
            "SUPPORTED_WITH_LIMITATIONS",
            f"Discordant pairs behind the {row['sample']} comparison.",
            artifacts=[cl_art], limitations=CL_LIM,
            derivation="a_only + b_only")
        add(f"closure.mcnemar.{rid}.chi2_cc", row["chi2_cc"], "scalar", 2,
            "SUPPORTED_WITH_LIMITATIONS",
            f"Continuity-corrected chi-square for the {row['sample']} comparison.",
            artifacts=[cl_art], limitations=CL_LIM + [
                "Secondary to the exact test; recomputed from the printed b and c it "
                "agrees to 0.05 or better in every row."],
            derivation="(|a_only - b_only| - 1)^2 / (a_only + b_only)")
        add(f"closure.mcnemar.{rid}.p", row["p_exact"], "mcnemar_p", 3,
            "SUPPORTED_WITH_LIMITATIONS",
            f"Exact two-sided binomial p-value for the {row['sample']} comparison "
            f"({row['decision']}).",
            artifacts=[cl_art], limitations=CL_LIM)

    for step in cl["waterfall"]["steps"]:
        add(f"closure.waterfall.{step['id']}", step["delta_pp"],
            "percentage_points", 2, "SUPPORTED_WITH_LIMITATIONS",
            f"Waterfall step '{step['label']}' in worst-case robust accuracy: "
            f"{step['status']}.",
            artifacts=[cl_art], signed=True, limitations=CL_LIM + [
                "The steps are measured against different baselines (the recipe and "
                "refinement steps against C1, the multi-view step against the loop, "
                "the resolution step against C1), so they are not additive."])

    for series, vals in cl["t_sweep"]["series"].items():
        for i, v in enumerate(vals, 1):
            add(f"closure.tsweep.t{i}.{series}", v, "percent", 2,
                "SUPPORTED_WITH_LIMITATIONS",
                f"recurrence_only at T = {i}: {CL_CELL[series]}, closure T-sweep.",
                artifacts=[cl_art], limitations=CL_LIM)
        add(f"closure.tsweep.delta.{series}", round((vals[3] - vals[0]) * 100, 2),
            "percentage_points", 2, "SUPPORTED_WITH_LIMITATIONS",
            f"T = 1 to T = 4 change in {CL_CELL[series]}.",
            artifacts=[cl_art], signed=True, limitations=CL_LIM,
            derivation=f"closure.tsweep.t4.{series} - closure.tsweep.t1.{series}")

    OR_LABEL = {"clean_5k": "5,000-image clean set",
                "clean_1k": "canonical 1,000-image clean slice",
                "worst_1k": "1,000-image worst-case slice"}
    for key, blob in cl["oracle"].items():
        if key == "exit_histogram_clean_5k":
            continue
        lab = OR_LABEL[key]
        for field, mid, unit, prec, what in (
                ("fixed_t4", "fixed_t4", "percent", 2, "accuracy at a fixed T = 4"),
                ("le1", "le1", "percent", 2, "accuracy exiting at step 1 or earlier"),
                ("le2", "le2", "percent", 2, "accuracy exiting at step 2 or earlier"),
                ("le3", "le3", "percent", 2, "accuracy exiting at step 3 or earlier"),
                ("all", "all", "percent", 2, "accuracy with the best possible exit per image"),
                ("headroom_pp", "headroom_pp", "percentage_points", 2,
                 "headroom of the oracle over fixed T = 4"),
                ("expected_views", "expected_views", "scalar", 3,
                 "mean number of passes under the oracle policy"),
                ("fixed_by_extra", "fixed_by_extra", "count", 0,
                 "images the extra passes recover")):
            add(f"closure.oracle.{key}.{mid}", blob[field], unit, prec,
                "SUPPORTED_WITH_LIMITATIONS",
                f"Oracle halting, {lab}: {what}.",
                artifacts=[cl_art], limitations=CL_LIM + [
                    "An oracle exit is not an executable policy: it uses the label to "
                    "decide when to stop."],
                signed=(field == "headroom_pp"))

    for row in cl["oracle"]["exit_histogram_clean_5k"]:
        add(f"closure.exit.t{row['exit_step']}.count", row["count"], "count", 0,
            "SUPPORTED_WITH_LIMITATIONS",
            f"Images whose earliest correct prediction is at step {row['exit_step']} "
            "(5,000-image clean set, oracle exit).",
            artifacts=[cl_art], limitations=CL_LIM)
        add(f"closure.exit.t{row['exit_step']}.share", row["share"], "percent", 1,
            "SUPPORTED_WITH_LIMITATIONS",
            f"Share of the 5,000-image clean set exiting at step {row['exit_step']}.",
            artifacts=[cl_art], limitations=CL_LIM)

    cka = cl["cka"]
    for i, a_id in enumerate(cka["labels"]):
        for j in range(i + 1, len(cka["labels"])):
            add(f"closure.cka.{a_id}_{cka['labels'][j]}",
                cka["matrix"][i][j], "scalar", 4, "SUPPORTED_WITH_LIMITATIONS",
                f"Linear CKA between the final block-12 CLS representations of {a_id} "
                f"and {cka['labels'][j]} on the canonical 1,000 frames.",
                artifacts=[cl_art], limitations=CL_LIM + [
                    "Similarity of representations, not of behaviour: a high value "
                    "does not imply the same decisions."])

    add("closure.r1_apgd_restoration",
        cl["retractions"][0]["magnitude_pp"], "percentage_points", 1,
        "SUPPORTED_WITH_LIMITATIONS",
        "Robust accuracy ff_112 recovers when APGD is run inside NormalizedModel "
        "instead of on raw [0, 1] pixels.",
        artifacts=[cl_art], signed=True, limitations=CL_LIM + [
            "The 'before' value (15.60 %) is the RETRACTED reading, not a measurement "
            "of the current harness; the difference is therefore a defect magnitude, "
            "not an accuracy gain."],
        derivation="closure.ff112.apgd_ce - 15.60, in pp")

    # ---- cost and capacity ---------------------------------------------- #
    add("cost.macs_ff", macs_ff / 1e9, "gmac", 3, "VERIFIED",
        "Analytic multiply-accumulates for one ff_112 forward pass, attention included.",
        code=["docs/rhan_nxa/verify_claims.py::analytic_macs"],
        derivation="conv + 12 transformer blocks, 66 tokens; no refinement block runs")
    add("cost.macs_rec", macs_rec / 1e9, "gmac", 3, "VERIFIED",
        "Analytic multiply-accumulates for one recurrence_only forward pass: four "
        "56-px passes plus the gist pass.",
        code=["docs/rhan_nxa/verify_claims.py::analytic_macs"],
        derivation="4 x (14 blocks, 18 tokens) + gist conv")
    add("cost.macs_ratio", macs_ff / macs_rec, "ratio", 3, "VERIFIED",
        "ff_112 MACs as a fraction of recurrence_only MACs.",
        code=["docs/rhan_nxa/verify_claims.py::analytic_macs"],
        derivation="macs_ff / macs_rec", limitations=[
            "Analytic count, not measured latency; the two arms have different "
            "memory-access patterns."])
    add("cost.params_ff", pb["per_phase"]["ff_112"]["unique_params"], "count", 0,
        "VERIFIED", "Unique parameters in the ff_112 checkpoint (state-dict tensors "
        "minus the twice-counted shared patch embedder).",
        artifacts=[pb_art], code=["gen2_foundation/backbone.py"])
    add("cost.params_rec", pb["per_phase"]["recurrence_only"]["unique_params"], "count", 0,
        "VERIFIED", "Unique parameters in the recurrence_only checkpoint.",
        artifacts=[pb_art], code=["gen2_foundation/backbone.py"])
    add("cost.effective_trainable_ff", pb["effective_capacity_delta"]["ff_112"], "count", 0,
        "VERIFIED", "Parameters that received a gradient in ff_112: unique parameters "
        "minus the tied refinement block and the gist gate.",
        artifacts=[pb_art], code=["gen2_foundation/model.py::forward"],
        derivation="optimizer state entries x 2 + 177 step scalars = 21,594,340")
    add("cost.effective_trainable_rec", pb["effective_capacity_delta"]["recurrence_only"],
        "count", 0, "VERIFIED",
        "Parameters that received a gradient in recurrence_only (only the gist gate "
        "is dead).", artifacts=[pb_art])
    add("cost.effective_delta", pb["effective_capacity_delta"]["rec_minus_ff"], "count", 0,
        "VERIFIED",
        "Effective-trainable difference, recurrent minus feedforward. Positive: the "
        "nominal +18,432 in favour of ff_112 inverts once dead weights are removed.",
        artifacts=[pb_art], derivation="effective_trainable_rec - effective_trainable_ff")

    # ---- training exposure ---------------------------------------------- #
    add("lineage.epochs_gist", 30, "epochs", 0, "VERIFIED",
        "Epochs in g2_gist_only, the phase recurrence_only inherits.",
        code=["training/train_gen2_foundation.py::PHASE_TOTAL_EPOCHS"])
    add("lineage.epochs_rec_phase", 15, "epochs", 0, "VERIFIED",
        "Epochs trained in the recurrence_only phase itself.",
        code=["training/train_gen2_foundation.py::PHASE_TOTAL_EPOCHS"])
    add("lineage.epochs_rec_total", 45, "epochs", 0, "VERIFIED",
        "Cumulative epochs behind the recurrence_only checkpoint.",
        derivation="30 inherited + 15 phase epochs",
        code=["training/train_gen2_foundation.py::run_gen2_phase (inherit FINAL weights)"])
    add("lineage.epochs_ff", 15, "epochs", 0, "VERIFIED",
        "Epochs behind the ff_112 checkpoint; warm-started from DINOv2, not from the "
        "Gen-2 ladder.",
        code=["training/train_gen2_foundation.py::PHASE_TOTAL_EPOCHS",
              "training/train_gen2_foundation.py::run_gen2_phase (DINOv2 warm start)"])

    # ---- sanity: derived values must equal their components -------------- #
    assert abs(M["paired.ff_minus_rec_ppg"]["value"]
               - (ffs["norm_eps=0.094"]["mean"] - ros["norm_eps_0094"]["mean"]) * 100) < 1e-4
    assert M["lineage.epochs_rec_total"]["value"] == (
        M["lineage.epochs_gist"]["value"] + M["lineage.epochs_rec_phase"]["value"])
    assert abs(M["cost.macs_ratio"]["value"] - macs_ff / macs_rec) < 1e-12
    for arm in J2_ARM:
        assert M[f"j2.{arm}.worst"]["value"] <= min(
            M[f"j2.{arm}.{a}"]["value"] for a in J2_ATTACK if a != "clean") + 1e-9, arm
    assert abs(M["ctrl.rec_minus_c1_clean"]["value"]
               - 100 * (M["ctrl.rec.clean"]["value"] - M["ctrl.c1.clean"]["value"])) < 1e-6
    assert abs(M["ctrl.rec_minus_c1_worst"]["value"]
               - 100 * (M["ctrl.rec.worst"]["value"] - M["ctrl.c1.worst"]["value"])) < 1e-6
    assert abs(M["j2.ff112_minus_rec_worst"]["value"]
               - 100 * (M["j2.ff112.worst"]["value"] - M["j2.rec.worst"]["value"])) < 1e-6

    # The transcribed closure tables must close against themselves. A discrepancy
    # here means a cell was mistyped, so the build fails rather than shipping it.
    for row in cl["mcnemar"]:
        rid = row["id"]
        n_set = 5000 if "5k" in rid else 1000
        assert row["a_only"] + row["b_only"] == row["n_discordant"], rid
        assert abs((row["a_only"] - row["b_only"]) / n_set * 100
                   - row["delta_pp"]) < 0.006, rid
        formula = (abs(row["a_only"] - row["b_only"]) - 1) ** 2 / row["n_discordant"]
        assert abs(formula - row["chi2_cc"]) < 0.06, (rid, formula)
    assert abs(M["closure.waterfall.loop"]["value"]
               - 100 * (M["closure.rec.worst"]["value"]
                        - M["closure.c1.worst"]["value"])) < 0.01
    assert abs(M["closure.waterfall.multiview"]["value"]
               - 100 * (M["closure.c4.worst"]["value"]
                        - M["closure.rec.worst"]["value"])) < 0.01
    assert abs(M["closure.waterfall.resolution"]["value"]
               - 100 * (M["closure.ff112.worst"]["value"]
                        - M["closure.c1.worst"]["value"])) < 0.01
    assert abs(M["closure.waterfall.recipe"]["value"]) < 1e-9
    assert abs(M["closure.waterfall.refinement"]["value"]) < 1e-9
    for series, vals in cl["t_sweep"]["series"].items():
        assert abs(M[f"closure.tsweep.delta.{series}"]["value"]
                   - 100 * (vals[3] - vals[0])) < 0.01, series
    for key, blob in cl["oracle"].items():
        if key == "exit_histogram_clean_5k":
            continue
        skey = "worst" if key == "worst_1k" else key
        assert abs(blob["all"] - blob["fixed_t4"] - blob["headroom_pp"] / 100) < 1e-9, key
        assert abs(blob["all"] - (blob["le1"]
                   + blob["fixed_by_extra"] / blob["n"])) < 1e-9, key
        assert abs(blob["fixed_t4"]
                   - M[f"closure.tsweep.t4.{skey}"]["value"]) < 1e-9, key
        assert abs(blob["le1"] - M[f"closure.tsweep.t1.{skey}"]["value"]) < 1e-9, key
    hist = cl["oracle"]["exit_histogram_clean_5k"]
    assert sum(r["count"] for r in hist) == cl["protocol"]["clean_5k_n"]
    assert abs(sum(r["share"] for r in hist) - 1.0) < 1e-9
    for i, a_id in enumerate(cka["labels"]):
        for j in range(i + 1, len(cka["labels"])):
            assert abs(cka["matrix"][i][j] - cka["matrix"][j][i]) < 1e-12, (a_id, j)
    return M


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
def schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://rhan.local/numbers.schema.json",
        "title": "RHAN handbook numeric registry",
        "type": "object",
        "required": ["schema_version", "registry_version", "project", "metrics"],
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "registry_version": {"const": REGISTRY_VERSION},
            "project": {"const": "RHAN"},
            "generated_at": {"type": ["string", "null"]},
            "code_revision": {"type": ["string", "null"]},
            "metrics": {
                "type": "object",
                "minProperties": 1,
                "patternProperties": {
                    "^[a-z0-9]+(\\.[a-z0-9_]+)+$": {
                        "type": "object",
                        "required": ["value", "unit", "display", "display_precision",
                                     "status", "description", "source_artifacts",
                                     "source_code", "derivation", "limitations"],
                        "additionalProperties": False,
                        "properties": {
                            "value": {"type": ["number", "integer"]},
                            "unit": {"enum": sorted(UNITS)},
                            "display": {"type": "string", "minLength": 1},
                            "display_precision": {"type": "integer", "minimum": 0,
                                                  "maximum": 6},
                            "signed_display": {"type": "boolean"},
                            "status": {"enum": sorted(STATUSES)},
                            "description": {"type": "string", "minLength": 10},
                            "source_artifacts": {"type": "array",
                                                 "items": {"type": "string"}},
                            "source_code": {"type": "array",
                                            "items": {"type": "string"}},
                            "derivation": {"type": ["string", "null"]},
                            "limitations": {"type": "array",
                                            "items": {"type": "string"}},
                        },
                    }
                },
                "additionalProperties": False,
            },
        },
    }


def validate(doc: dict) -> list[str]:
    """Dependency-free validation of the constraints the schema encodes, plus
    the semantic rules a schema cannot express."""
    errs: list[str] = []
    for key in ("schema_version", "registry_version", "project", "metrics"):
        if key not in doc:
            errs.append(f"missing top-level key {key}")
    if doc.get("schema_version") != SCHEMA_VERSION:
        errs.append("schema_version mismatch")
    if not doc.get("metrics"):
        errs.append("no metrics")
        return errs

    seen_display = {}
    for mid, m in doc["metrics"].items():
        parts = mid.split(".")
        if len(parts) < 2 or not all(p and p == p.lower() for p in parts):
            errs.append(f"{mid}: id must be lower-case dot-separated segments")
        if m["unit"] not in UNITS:
            errs.append(f"{mid}: unknown unit {m['unit']}")
        if m["status"] not in STATUSES:
            errs.append(f"{mid}: unknown status {m['status']}")
        if not isinstance(m["value"], (int, float)):
            errs.append(f"{mid}: value must be numeric, not {type(m['value']).__name__}")
            continue
        if m["status"] in PROVENANCE_REQUIRED and not (m["source_artifacts"] or m["source_code"]):
            errs.append(f"{mid}: {m['status']} without provenance")
        if m["status"] in {"RETRACTED", "UNRESOLVED", "NOT_REPRODUCIBLE"} and not m["limitations"]:
            errs.append(f"{mid}: {m['status']} must state why in limitations")
        recomputed = fmt(m["value"], m["unit"], m["display_precision"],
                         bool(m.get("signed_display")))
        if recomputed != m["display"]:
            errs.append(f"{mid}: display {m['display']!r} != formatted value {recomputed!r}")
        if m["display"] in seen_display:
            pass  # distinct metrics may legitimately print the same string
        seen_display[m["display"]] = mid
        for art in m["source_artifacts"]:
            if art.startswith("/") or os.path.isabs(art):
                errs.append(f"{mid}: absolute artifact path {art}")
            elif not (ROOT / art).exists():
                errs.append(f"{mid}: artifact missing on disk: {art}")
    # derived values that must close
    def get(k):
        return doc["metrics"][k]["value"]
    checks = [
        ("sweep.ff112.drop_ppg",
         (get("sweep.ff112.clean") - get("sweep.ff112.pgd10_0094")) * 100, 0.02),
        ("sweep.rec.drop_ppg",
         (get("sweep.rec.clean") - get("sweep.rec.pgd10_0094")) * 100, 0.02),
        ("sweep.gen1.drop_ppg",
         (get("sweep.gen1.clean") - get("sweep.gen1.pgd10_0094")) * 100, 0.02),
        ("cost.effective_delta",
         get("cost.effective_trainable_rec") - get("cost.effective_trainable_ff"), 0.5),
        ("lineage.epochs_rec_total",
         get("lineage.epochs_gist") + get("lineage.epochs_rec_phase"), 0.5),
    ]
    for mid, expect, tol in checks:
        if mid in doc["metrics"] and abs(get(mid) - expect) > tol:
            errs.append(f"{mid}: {get(mid)} does not close against its components "
                        f"({expect})")
    return errs


# --------------------------------------------------------------------------- #
# claims and lineage
# --------------------------------------------------------------------------- #
def build_claims() -> dict:
    C = []
    def claim(cid, text, section, metrics, artifacts, status, note):
        C.append({"id": cid, "claim": text, "locations": section,
                  "metric_ids": metrics, "artifacts": artifacts,
                  "status": status, "notes": note})

    claim("C-RES-01", "One 112-px pass is 5.29 pp more robust than four 56-px "
          "glimpses at eps 0.094 under PGD-10.",
          ["00_front_matter/01_executive_overview.md",
           "04_results/26_results_narrative.md", "04_results/27_recurrence_versus_resolution.md",
           "03_evaluation/22_adversarial_evaluation.md"],
          ["paired.ff_minus_rec_ppg", "paired.ci95_low", "paired.ci95_high"],
          ["report/eval/eval_ff_112_20261018_123413.json",
           "report/d3_final_eval/g2_d3_recurrence_only_final.json"],
          "SUPPORTED_WITH_LIMITATIONS",
          "PGD-10 only. No ff_112 row exists in the library-APGD audit, so the "
          "advantage is not established under the strongest attack.")
    claim("C-RES-02", "The in-house PGD-50 was under-stepped; the corrected rule "
          "lands within 1.2 pp of library APGD-CE.",
          ["03_evaluation/22_adversarial_evaluation.md"],
          ["audit.rec.pgd50_corrected", "audit.rec.apgd_ce",
           "audit.rec.pgd50_corrected_diff_vs_apgd"],
          ["report/audit_rev1/step1_attack_audit.json"], "VERIFIED",
          "Step-size-only variation on identical samples.")
    claim("C-RES-03", "The stored ff_112 APGD figures (15.6 %, 16.8 %) are invalid "
          "and must not be compared with any arm.",
          ["00_front_matter/01_executive_overview.md",
           "03_evaluation/22_adversarial_evaluation.md",
           "06_appendices/D_per_seed_results.md",
           "06_appendices/F_evidence_registry.md"],
          ["final.ff.apgd_ce_invalid", "final.ff.apgd_t_invalid"],
          ["report/eval/eval_ff_112_20261018_123413.json"], "RETRACTED",
          "Radius 0.031 in normalised units (git 938ada3^) and a [0,1] clamp applied "
          "to normalised tensors. Neither 0.094 nor 8/255.")
    claim("C-RES-04", "recurrence_only beats gen1_core by 2.4 pp under APGD-CE and "
          "this is not significant at n = 500.",
          ["03_evaluation/22_adversarial_evaluation.md",
           "04_results/26_results_narrative.md"],
          ["audit.paired_rec_minus_gen1_apgd", "audit.rec.apgd_ce", "audit.gen1.apgd_ce"],
          ["report/audit_rev1/step1_attack_audit.json"],
          "SUPPORTED_WITH_LIMITATIONS", "McNemar p = 0.0896.")
    claim("C-COST-01", "ff_112 costs 1.456 GMAC against 1.8156 GMAC for the loop, "
          "0.802x.",
          ["03_evaluation/24_computational_cost.md", "04_results/27_recurrence_versus_resolution.md",
           "05_limits_future/29_limitations.md"],
          ["cost.macs_ff", "cost.macs_rec", "cost.macs_ratio"],
          ["docs/rhan_nxa/verify_claims.py"], "VERIFIED",
          "ff_112 never executes its tied refinement block, so the block must not be "
          "counted.")
    claim("C-COST-02", "The nominal parameter advantage of ff_112 (+18,432) inverts "
          "once dead weights are excluded.",
          ["03_evaluation/23_ablations.md", "06_appendices/B_configuration_reference.md"],
          ["cost.params_ff", "cost.params_rec", "cost.effective_trainable_ff",
           "cost.effective_trainable_rec", "cost.effective_delta"],
          ["report/handbook/data_extracts/param_breakdown.json"], "VERIFIED",
          "Measured from optimizer state, not from a nominal count.")
    claim("C-LIN-01", "recurrence_only carries 45 cumulative epochs against ff_112's 15.",
          ["04_results/27_recurrence_versus_resolution.md",
           "05_limits_future/29_limitations.md"],
          ["lineage.epochs_rec_total", "lineage.epochs_ff"],
          ["training/train_gen2_foundation.py"], "VERIFIED",
          "Inheritance is a code behaviour (FINAL weights of the previous phase), not "
          "a filename inference. Architecture-only attribution is therefore limited.")
    claim("C-J2-01", "At n = 1000 under one pixel-space harness, ff_112 is 12.5 pp "
          "more robust in the worst case than recurrence_only (36.2 vs 23.7 %).",
          ["00_front_matter/01_executive_overview.md",
           "03_evaluation/22_adversarial_evaluation.md",
           "04_results/26_results_narrative.md"],
          ["j2.ff112.worst", "j2.rec.worst", "j2.ff112_minus_rec_worst",
           "j2.ff112.clean", "j2.rec.clean"],
          ["reconciliation/attack_reconciliation_20261021_135827.json"],
          "SUPPORTED_WITH_LIMITATIONS",
          "One checkpoint per arm, one image sample, and the worst case is over six "
          "attacks rather than a certified radius. No per-sample outcomes were "
          "persisted, so no paired test accompanies this contrast.")
    claim("C-J2-02", "Against its epoch-matched control C1, recurrence buys +5.4 pp "
          "clean and +1.3 pp worst-case robust accuracy at 56 px.",
          ["00_front_matter/01_executive_overview.md",
           "04_results/27_recurrence_versus_resolution.md",
           "03_evaluation/23_ablations.md"],
          ["ctrl.rec_minus_c1_clean", "ctrl.rec_minus_c1_worst",
           "ctrl.rec.clean", "ctrl.c1.clean", "ctrl.rec.worst", "ctrl.c1.worst"],
          ["report/controls/controls_eval_20261021_202939.json"],
          "SUPPORTED_WITH_LIMITATIONS",
          "Both checkpoints have 45 cumulative epochs. The difference is measured, "
          "not tested: the artifact stores accuracies, not per-sample outcomes.")
    claim("C-J2-03", "recurrence_only beats the 56-px feedforward control C2 by 17.7 pp "
          "worst-case, but C2's recipe differs (rung 3 from a DINOv2 start, no clean "
          "warm-up).",
          ["03_evaluation/23_ablations.md", "04_results/27_recurrence_versus_resolution.md"],
          ["ctrl.c2.worst", "ctrl.rec.worst", "ctrl.c2_minus_rec_worst"],
          ["report/controls/controls_eval_20261021_202939.json"],
          "SUPPORTED_WITH_LIMITATIONS",
          "C2 confounds architecture with curriculum: its collapse cannot be read as "
          "a pure resolution effect.")
    claim("C-J2-04", "No full AutoAttack run exists; the J2 'AutoAttack core' is the "
          "intersection of APGD-CE and APGD-DLR.",
          ["03_evaluation/22_adversarial_evaluation.md",
           "06_appendices/E_attack_protocols.md"],
          ["j2.ff112.aa_core", "j2.rec.aa_core"],
          ["reconciliation/attack_reconciliation_20261021_135827.json"],
          "SUPPORTED_WITH_LIMITATIONS",
          "APGD-CE and APGD-DLR only; FAB, square and the full ensemble were not run.")
    claim("C-J2-05", "The J2 battery realises its budget: max Linf is 5.417/255 for "
          "every attack on every arm.",
          ["03_evaluation/22_adversarial_evaluation.md",
           "06_appendices/E_attack_protocols.md"],
          ["j2.protocol.eps_pixel_255"],
          ["reconciliation/attack_reconciliation_20261021_135827.json"], "VERIFIED",
          "The script asserts the bound per batch; the artifact records the maximum "
          "per attack.")
    claim("C-STAT-01", "The 8-seed paired difference has t(7) = 9.31.",
          ["00_front_matter/01_executive_overview.md", "03_evaluation/22_adversarial_evaluation.md"],
          ["paired.t_statistic", "paired.df"], ["docs/rhan_nxa/verify_claims.py"],
          "VERIFIED", "Computed from per-seed arrays, not inferred from a CI.")
    return {"schema": "rhan.handbook.claims/1", "claims": C}


def build_experiments() -> dict:
    """Manifest of the recorded artifacts, with content hashes.

    A hash is what makes staleness detectable: if an artifact is re-run, its
    digest changes and the validator refuses the registry until the numbers are
    rebuilt.
    """
    import hashlib

    rows = [
        ("AUDIT-ATTACKS", "report/audit_rev1/step1_attack_audit.json",
         "13-attack matched-eps sweep, per-sample outcomes, McNemar",
         "500", "42", "0.094", "in-house PGD family, APGD-CE/DLR, MI-FGSM", "complete"),
        ("AUDIT-CALIBRATION", "report/audit_rev1/step2_calibration_audit.json",
         "NLL, Brier, ECE, AUROC, U_t", "500", "42 / 101", "0.094", "PGD-10", "complete"),
        ("AUDIT-RESOLUTION", "report/audit_rev1/step3_resolution_audit.json",
         "foveal trace, 56 vs 112 probe, CKA", "-", "-", "-", "-", "complete"),
        ("AUDIT-RECURRENT", "report/audit_rev1/step4_recurrent_audit.json",
         "contradiction/order probe, surprise probe", "50 pairs", "42", "-", "-", "complete"),
        ("AUDIT-RES-PROBE", "report/audit_rev1/step5_full_resolution_probe.json",
         "56/112/126 probes, 5 epochs", "5000", "-", "0", "-", "complete"),
        ("AUDIT-LATENCY", "report/audit_rev1/step6_latency_benchmark.json",
         "s/batch, epoch projection", "126689", "-", "-", "-", "complete"),
        ("FF112-SWEEP", "report/eval/eval_ff_112_20261018_123413.json",
         "8-seed sweep + final verification", "8x300 / 500", "101-108 / 42", "0-0.094",
         "PGD-10, PGD-50, APGD-CE, APGD-T, EOT", "complete; APGD block invalid"),
        ("D3-RECURRENCE", "report/d3_final_eval/g2_d3_recurrence_only_final.json",
         "sweep + final", "8x300 / 500", "101-108 / 42", "0-0.094", "PGD-10, PGD-50, EOT",
         "complete"),
        ("D3-GEN1", "report/d3_final_eval/g2_d3_gen1_core_best_clean.json",
         "sweep + final", "8x300 / 500", "101-108 / 42", "0-0.094", "PGD-10, PGD-50, EOT",
         "complete"),
        ("D8-ADAPTIVE", "report/d8_adaptive_attacks/g2_d8_combined.json",
         "MI-FGSM, APGD-CE, diversity-PGD", "300", "42", "0.094", "3 attacks", "complete"),
        ("J2-RECONCILIATION", "reconciliation/attack_reconciliation_20261021_135827.json",
         "six-attack matched-eps battery on four arms, transfer from ff_112",
         "1000", "42", "5.417/255",
         "PGD-50, APGD-CE fp32 and autocast, PGD-200 x3, APGD-DLR", "complete"),
        ("CONTROLS-56", "report/controls/controls_eval_20261021_202939.json",
         "epoch-matched 56-px control and 56-px feedforward control",
         "1000", "42", "5.417/255", "PGD-50, APGD-CE, APGD-DLR", "complete"),
        ("CONTROLS-TRAINLOG", "report/controls/gen2_epoch_log.jsonl",
         "per-epoch clean/robust validation accuracy and losses for C1",
         "val subset", "-", "0.094", "PGD-4 inner loop", "complete"),
    ]
    out = []
    for rid, rel, contents, n, seed, eps, attacks, status in rows:
        path = ROOT / rel
        entry = {"id": rid, "path": rel, "contents": contents, "n": n,
                 "seed": seed, "eps": eps, "attacks": attacks, "status": status}
        if path.exists():
            blob = path.read_bytes()
            entry["sha256"] = hashlib.sha256(blob).hexdigest()
            entry["bytes"] = len(blob)
        else:
            entry["sha256"] = None
        out.append(entry)
    return {"schema": "rhan.handbook.experiment_manifest/1",
            "note": "Hashes pin the artifacts the registry was built from. A changed hash "
                    "means the numbers must be rebuilt before they are published.",
            "experiments": out}


def build_lineage() -> dict:
    return {
        "schema": "rhan.handbook.training_lineage/1",
        "source": "training/train_gen2_foundation.py (PHASE_LADDER, "
                  "PHASE_TOTAL_EPOCHS, run_gen2_phase weight-loading rule)",
        "nodes": [
            {"id": "dinov2_vits14", "kind": "pretrained", "epochs": 0,
             "artifact": "checkpoints/dinov2_vits14_pretrain.pth", "confidence": "verified"},
            {"id": "g2_gist_only", "kind": "phase", "epochs": 30,
             "artifact": "checkpoints/foundation_g2_gist_only_final.zip",
             "confidence": "verified"},
            {"id": "recurrence_only", "kind": "phase", "epochs": 15,
             "artifact": "checkpoints/foundation_recurrence_only_final.pth",
             "confidence": "verified"},
            {"id": "ff_112", "kind": "phase", "epochs": 15,
             "artifact": "checkpoints/foundation_ff_112_final.pth", "confidence": "verified"},
            {"id": "ff_56", "kind": "phase", "epochs": 15,
             "artifact": "checkpoints/foundation_ff_56_final.pth",
             "confidence": "unresolved",
             "note": "Artifact present and evaluated, but its embedded phase reads "
                     "g2_gist_only, so its identity is unresolved: it is a renamed copy, "
                     "not an independently identifiable 56-px feedforward run."},
            {"id": "g2_gist_only_cont", "kind": "phase", "epochs": 15,
             "artifact": "checkpoints/foundation_g2_gist_only_cont_final.pth",
             "confidence": "verified",
             "note": "Epoch-matched control (C1), trained and evaluated: "
                     "report/controls/controls_eval_20261021_202939.json."},
        ],
        "edges": [
            {"from": "dinov2_vits14", "to": "g2_gist_only", "type": "warm_start",
             "confidence": "verified"},
            {"from": "g2_gist_only", "to": "recurrence_only", "type": "inherit_final",
             "confidence": "verified"},
            {"from": "dinov2_vits14", "to": "ff_112", "type": "warm_start",
             "confidence": "verified"},
            {"from": "g2_gist_only", "to": "ff_56", "type": "same_recipe_as",
             "confidence": "unresolved"},
            {"from": "g2_gist_only", "to": "g2_gist_only_cont", "type": "continue",
             "confidence": "verified"},
        ],
        "cumulative_epochs": {"ff_112": 15, "recurrence_only": 45,
                              "g2_gist_only": 30, "ff_56": 30,
                              "g2_gist_only_cont": 45},
        "limitations": [
            "Epochs are optimisation passes over the same 126,689-image training "
            "split; they are not equal sample exposures once phases differ in "
            "augmentation or adversarial mix.",
            "The 30 inherited epochs already include the g2_gist_only adversarial "
            "schedule, so the asymmetry is in total exposure, not only in phase count.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="validate the existing registry instead of rewriting it")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": SCHEMA_VERSION,
        "registry_version": REGISTRY_VERSION,
        "project": "RHAN",
        "generated_at": None,
        "code_revision": None,
        "metrics": build_metrics(),
    }
    errs = validate(doc)
    if errs:
        print("REGISTRY INVALID")
        for e in errs:
            print("  -", e)
        return 1

    files = {
        "numbers.json": doc,
        "numbers.schema.json": schema(),
        "claims.json": build_claims(),
        "training_lineage.json": build_lineage(),
        "experiment_manifest.json": build_experiments(),
    }
    for name, payload in files.items():
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        path = OUT / name
        if args.check:
            if not path.exists():
                print(f"MISSING {path}")
                return 1
            if path.read_text(encoding="utf-8") != text:
                print(f"STALE {path.relative_to(ROOT)} — regenerate with "
                      f"scripts/handbook/build_numbers.py")
                return 1
        else:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)}")

    n = len(doc["metrics"])
    by_status = {}
    for m in doc["metrics"].values():
        by_status[m["status"]] = by_status.get(m["status"], 0) + 1
    print(f"{n} metrics; " + ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
