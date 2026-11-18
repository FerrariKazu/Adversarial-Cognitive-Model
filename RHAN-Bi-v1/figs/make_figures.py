"""Generate the dossier figures: five schematics and a simulator scenario plate.

    python figures/make_figures.py

All figures are drawn from the code's own constants where possible (costs, gate criteria are text;
the scenario plate renders real simulator frames). PNG only: the repository root gitignores *.pdf.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rhan_bi.config import default_config  # noqa: E402
from rhan_bi.simulator import make_scene, mutate, render  # noqa: E402

OUT = Path(__file__).resolve().parent
INK = "#1b1b1b"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GREY = "#666666"
LIGHT = "#F3F3F3"


def _box(ax, x, y, w, h, text, fc="white", ec=INK, fs=7.6, bold=False, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.03",
                                linewidth=0.8, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def _arrow(ax, x0, y0, x1, y1, color=INK, lw=0.8, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=7,
                                 linewidth=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


def fig_loop():
    stages = [
        ("visual environment", "camera · light rig · plate"),
        ("observation", "image + action metadata"),
        ("perceptual encoder", "features + anomaly maps"),
        ("evidence builder", "quality · LLR · pair evidence"),
        ("evidence ledger", "immutable records · dependence"),
        ("belief state", "posterior · debt · conflicts"),
        ("observation planner", "falsification · EIG · cost"),
        ("new observation", "view · light · zoom"),
        ("belief revision", "recorded · attributable"),
        ("commit / abstain", "with coverage reported"),
    ]
    fig, ax = plt.subplots(figsize=(4.6, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.09)
    ax.axis("off")
    h, gap = 0.074, 0.026
    top = 1.02
    for i, (title, sub) in enumerate(stages):
        y = top - i * (h + gap) - h
        fc = LIGHT if i not in (4, 5, 6) else "#E8F1F8"
        _box(ax, 0.10, y, 0.78, h, "", fc=fc)
        ax.text(0.49, y + h * 0.66, title, ha="center", va="center", fontsize=7.8,
                fontweight="bold", color=INK)
        ax.text(0.49, y + h * 0.30, sub, ha="center", va="center", fontsize=6.5, color=GREY)
        if i < len(stages) - 1:
            _arrow(ax, 0.49, y, 0.49, y - gap + 0.004)
    # the repeat is a property of the loop, not a second set of arrows
    ax.add_patch(FancyBboxPatch((0.015, top - 7 * (h + gap) - h), 0.012, 2 * (h + gap) + h,
                                boxstyle="round,pad=0.004,rounding_size=0.01",
                                linewidth=0.0, facecolor=ORANGE, alpha=0.35))
    ax.text(0.043, top - 7 * (h + gap) - h / 2, "repeat while\nevidence is\ninsufficient",
            fontsize=5.6, color=ORANGE, va="center", ha="left")
    ax.text(0.10, 0.004, "the planner, not the encoder, decides what the next observation is",
            fontsize=6.2, color=GREY)
    ax.set_title("The evidence-integrity loop", fontsize=9, loc="left", color=INK)
    fig.savefig(OUT / "fig01_evidence_loop.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def fig_architecture():
    comps = [
        ("visual foundation", "image → features,\nmaps, patch tokens", BLUE),
        ("evidence builder", "quality, LLR,\npair comparison", BLUE),
        ("evidence ledger", "content-addressed\nrecords + graph", GREEN),
        ("belief state", "posterior, debt,\nconflicts, revisions", GREEN),
        ("observation planner", "expected displacement,\nrisk, cost, redundancy", ORANGE),
        ("revision + stopping", "commit / abstain /\nlook again", ORANGE),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w, h = 0.27, 0.26
    positions = [(0.02, 0.62), (0.355, 0.62), (0.69, 0.62),
                 (0.69, 0.18), (0.355, 0.18), (0.02, 0.18)]
    for (x, y), (title, sub, col) in zip(positions, comps):
        _box(ax, x, y, w, h, "", ec=col)
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=7.6,
                fontweight="bold", color=col)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=6.4, color=INK)
    _arrow(ax, 0.29, 0.75, 0.355, 0.75)
    _arrow(ax, 0.625, 0.75, 0.69, 0.75)
    _arrow(ax, 0.825, 0.62, 0.825, 0.44)
    _arrow(ax, 0.69, 0.31, 0.625, 0.31)
    _arrow(ax, 0.355, 0.31, 0.29, 0.31)
    _arrow(ax, 0.155, 0.44, 0.155, 0.62, color=ORANGE)
    ax.text(0.165, 0.52, "next action", fontsize=6.2, color=ORANGE, rotation=90, va="center")
    ax.text(0.02, 0.95, "six components, one loop", fontsize=9, color=INK)
    ax.text(0.02, 0.035, "interfaces are serialisable; no component reads ground truth",
            fontsize=6.4, color=GREY)
    fig.savefig(OUT / "fig02_architecture.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def fig_ledger():
    fields = [
        "record_id        sha256(canonical payload)",
        "episode_id        scene + episode",
        "step              observation index",
        "action            kind · view · light · roi · cost",
        "image_sha256      exact frame hash",
        "features          standardised 8-vector",
        "quality           score · flags · components",
        "llr               defect/reflection/shadow/none",
        "pair              persistence · bright_shift",
        "dependence        relations · weight",
        "provenance        code · model · camera",
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.08), 0.56, 0.84, boxstyle="round,pad=0.012",
                                linewidth=0.9, edgecolor=BLUE, facecolor="#F7FBFF"))
    ax.text(0.05, 0.85, "EvidenceRecord", fontsize=8.5, fontweight="bold", color=BLUE)
    for i, line in enumerate(fields):
        ax.text(0.05, 0.77 - i * 0.066, line, fontsize=6.3, family="monospace", color=INK)
    ax.text(0.64, 0.86, "dependence graph", fontsize=8.2, fontweight="bold", color=GREEN)
    nodes = {"r0\nbase": (0.70, 0.66), "r1\nlight": (0.86, 0.50), "r2\nzoom": (0.71, 0.33),
             "r3\nlight": (0.90, 0.20)}
    for label, (x, y) in nodes.items():
        ax.add_patch(FancyBboxPatch((x - 0.055, y - 0.05), 0.11, 0.10,
                                    boxstyle="round,pad=0.008,rounding_size=0.02",
                                    linewidth=0.8, edgecolor=GREEN, facecolor="white"))
        ax.text(x, y, label, ha="center", va="center", fontsize=6.2, color=INK)
    for a, b, rad, lab in [("r0\nbase", "r1\nlight", -0.2, "0.55"),
                           ("r0\nbase", "r2\nzoom", 0.25, "0.45"),
                           ("r1\nlight", "r3\nlight", 0.25, "0.60")]:
        (x0, y0), (x1, y1) = nodes[a], nodes[b]
        _arrow(ax, x0, y0, x1, y1, color=GREEN, lw=0.7, rad=rad)
        ax.text((x0 + x1) / 2 + 0.01, (y0 + y1) / 2 + 0.02, lab, fontsize=5.8, color=GREEN)
    ax.text(0.64, 0.09, "weight = 1 / max(1, 1 + Σρ) × (0.5 + 0.5q)", fontsize=6.4, color=GREY)
    ax.text(0.64, 0.035, "a duplicate can only reduce influence", fontsize=6.4, color=GREY)
    ax.set_title("The evidence ledger", fontsize=9, loc="left", color=INK)
    fig.savefig(OUT / "fig03_ledger.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def fig_gates():
    gates = [
        ("A", "foundation", "accuracy ≥ 0.80 · balanced ≥ 0.75 ·\ndefect/reflection in [0.55, 0.80]", BLUE),
        ("B", "spatial evidence", "IoU@0.3 ≥ 0.25 · pointing ≥ 0.85 ·\nlocalization-conditioned ≥ 0.90", BLUE),
        ("C", "reliability", "ECE ≤ 0.08 · selective risk@0.5 ≤ 0.25 ·\ncoverage ≥ 0.40 at risk 0.20", GREEN),
        ("D", "belief revision", "revision accuracy ≥ pooled + 0.05 ·\nledger ECE ≤ pooled · ablations reported", GREEN),
        ("E", "adaptive selection", "B = 3 · paired error, 95% CI · McNemar,\nHolm · oracle headroom reported", ORANGE),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.06, 1.02)
    ax.axis("off")
    for i, (gid, name, crit, col) in enumerate(gates):
        y = 0.86 - i * 0.175
        _box(ax, 0.02, y - 0.12, 0.07, 0.12, gid, fc=col, ec=col, fs=9, bold=True, color="white")
        ax.text(0.11, y - 0.035, name, fontsize=7.8, fontweight="bold", color=INK, va="center")
        ax.text(0.36, y - 0.055, crit, fontsize=6.5, color=INK, va="center")
        if i < len(gates) - 1:
            _arrow(ax, 0.055, y - 0.12, 0.055, y - 0.175, color=GREY, lw=0.7)
    ax.text(0.02, 0.97, "Five gates, each with a named failure", fontsize=9, color=INK)
    ax.text(0.02, -0.035, "a failed gate is reported, not skipped; Gates A–C are prerequisites for D–E",
            fontsize=6.4, color=GREY)
    fig.savefig(OUT / "fig04_gates.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def fig_planner():
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rows = [("action", "changes", "cost"), ("light", "illumination azimuth", "1.0"),
            ("zoom", "crop factor", "1.0"), ("view", "camera azimuth", "2.0")]
    ax.text(0.02, 0.86, "action space", fontsize=8.2, fontweight="bold", color=ORANGE)
    for r, (a, b, c) in enumerate(rows):
        y = 0.78 - r * 0.13
        weight = "bold" if r == 0 else "normal"
        ax.text(0.02, y, a, fontsize=7.0, fontweight=weight, color=INK)
        ax.text(0.17, y, b, fontsize=7.0, fontweight=weight, color=INK)
        ax.text(0.52, y, c, fontsize=7.0, fontweight=weight, color=INK)
    ax.plot([0.02, 0.55], [0.84, 0.84], color=GREY, lw=0.6)
    ax.text(0.62, 0.86, "U(a) =  w_f · E[1 − p_leader(after a)]", fontsize=7.4, color=INK)
    ax.text(0.62, 0.73, "        + w_r · E[risk before − risk after]", fontsize=7.4, color=INK)
    ax.text(0.62, 0.60, "        − λ_c · cost − λ_r · ρ(a) − λ_u · (1 − q(a))", fontsize=7.4, color=INK)
    ax.text(0.62, 0.44, "defaults: w_f 1.0 · w_r 0.5 · λc 0.05 · λr 0.15 · λu 0.10",
            fontsize=6.3, color=GREY)
    ax.text(0.62, 0.30, "baselines: EIG · confidence · random · light-first",
            fontsize=6.3, color=GREY)
    ax.text(0.62, 0.17, "a decision score, not a calibrated utility", fontsize=6.3, color=ORANGE)
    ax.text(0.02, 0.97, "Falsification-directed selection", fontsize=9, color=INK)
    fig.savefig(OUT / "fig05_planner.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def fig_scenarios():
    cfg = default_config()
    panels = []

    def add(scene, view, light, title, zoom=None):
        panels.append((render(scene, view, light, cfg.sim, zoom_box=zoom), title))

    s1_defect = make_scene("train", 0, cfg.sim)
    add(s1_defect, 0, 1, "S1 defect · neutral light")
    add(s1_defect, 0, 2, "S1 defect · light changed\n(stays put)")
    s1_refl = make_scene("train", 1, cfg.sim)
    add(s1_refl, 0, 1, "S1 reflection · neutral light")
    add(s1_refl, 0, 2, "S1 reflection · light changed\n(moves)")
    s2 = next(make_scene("train", i, cfg.sim) for i in range(3, 40)
              if make_scene("train", i, cfg.sim).occluder is not None)
    add(s2, 0, 1, "S2 occlusion · missing evidence")
    s3 = make_scene("train", 2, cfg.sim)
    add(s3, 0, 1, "S3 shadow · soft, low-frequency")
    add(s3, 0, 2, "S3 shadow · light changed")
    s4 = next(make_scene("train", i, cfg.sim) for i in range(3, 40)
              if make_scene("train", i, cfg.sim).cause == "none")
    add(mutate(s4, "introduce_defect", cfg.sim), 0, 1, "S4 state change\ndefect introduced")

    fig, axes = plt.subplots(2, 4, figsize=(7.4, 4.0))
    for ax, (img, title) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=6.4, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GREY)
            spine.set_linewidth(0.6)
    fig.suptitle("The inspection laboratory: four controlled scenarios", fontsize=8.6, x=0.02,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig06_scenarios.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_loop()
    fig_architecture()
    fig_ledger()
    fig_gates()
    fig_planner()
    fig_scenarios()
    print("wrote 6 figures to", OUT)
