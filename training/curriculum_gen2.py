"""Gen-2 Adversarial Curriculum — training/curriculum_gen2.py
================================================================================

Implements the S4 curriculum specification:
- Deviation D1: Phase g2_gist_only starts with 10 clean epochs (w_trades=0.0),
  followed by a 20-epoch TRADES ramp reusing frozen Gen-0 constants:
    * Rung 1 (epochs 11-17, 7 eps): eps=0.031, beta=2.0, w_trades=0.55, PGD-4
      At epoch 17: GATE R check (pause if clean val < 31.3%).
    * Rung 2 (epochs 18-24, 7 eps): eps=0.062, beta=2.0, w_trades=0.55, PGD-4
    * Rung 3 (epochs 25-30, 6 eps): eps=0.094, beta=2.5, w_trades=0.55, PGD-4
- At epoch 10: GATE C check (clean val must exceed Gate W probe accuracy).
- Downstream phases (recurrence_only, belief_no_f, belief_with_f, ais_v2_swap,
  gen1_core) run 15 epochs at the top rung (eps=0.094, beta=2.5), inheriting
  the previous phase's FINAL-epoch weights (not best.pth).
- Prints norm-space and pixel-space epsilon equivalents in the header.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
MEAN_STD: float = sum(IMAGENET_STD) / len(IMAGENET_STD)  # ~0.226


@dataclass(frozen=True)
class Gen2CurriculumPoint:
    phase: str
    epoch: int
    total_epochs: int
    eps_norm: float
    eps_pixel: float
    beta: float
    pgd_steps: int
    w_trades: float
    adversarial: bool
    is_warmup: bool
    gate_trigger: Optional[str] = None

    def banner(self) -> str:
        adv_str = f"eps_norm={self.eps_norm:.4f} (pix={self.eps_pixel:.4f}) beta={self.beta:.1f} pgd={self.pgd_steps} w_trades={self.w_trades:.2f}" if self.adversarial else "clean_warmup w_trades=0.0"
        gate_str = f" [{self.gate_trigger}]" if self.gate_trigger else ""
        return f"[Curriculum] {self.phase} ep {self.epoch}/{self.total_epochs} -> {adv_str}{gate_str}"


def norm_to_pixel_eps(eps_norm: float) -> float:
    """Convert norm-space epsilon to approximate [0, 1] pixel space: eps_pixel = eps_norm * mean(std)."""
    return float(eps_norm * MEAN_STD)


def print_curriculum_header() -> None:
    print("=" * 70)
    print("Gen-2 Curriculum Mapping (Normalized Space -> Pixel Space Equivalence):")
    print(f"  ImageNet std: {IMAGENET_STD} (mean std: {MEAN_STD:.4f})")
    for eps in (0.0, 0.031, 0.062, 0.094):
        pix = norm_to_pixel_eps(eps)
        pix_255 = pix * 255.0
        print(f"  eps_norm={eps:.3f}  ->  eps_pixel={pix:.4f}  (~{pix_255:.1f}/255)")
    print("=" * 70)


def curriculum_for_phase_epoch(phase: str, epoch: int) -> Gen2CurriculumPoint:
    """Return the authoritative curriculum point for a given phase and epoch."""
    if phase == "g2_gist_only":
        total_epochs = 30
        if epoch <= 10:
            # 10 clean adaptation epochs
            gate = "GATE_C" if epoch == 10 else None
            return Gen2CurriculumPoint(
                phase=phase,
                epoch=epoch,
                total_epochs=total_epochs,
                eps_norm=0.0,
                eps_pixel=0.0,
                beta=0.0,
                pgd_steps=0,
                w_trades=0.0,
                adversarial=False,
                is_warmup=(epoch <= 3),
                gate_trigger=gate,
            )
        elif epoch <= 17:
            # Rung 1: 7 epochs @ eps 0.031, beta 2.0
            gate = "GATE_R" if epoch == 17 else None
            return Gen2CurriculumPoint(
                phase=phase,
                epoch=epoch,
                total_epochs=total_epochs,
                eps_norm=0.031,
                eps_pixel=norm_to_pixel_eps(0.031),
                beta=2.0,
                pgd_steps=4,
                w_trades=0.55,
                adversarial=True,
                is_warmup=False,
                gate_trigger=gate,
            )
        elif epoch <= 24:
            # Rung 2: 7 epochs @ eps 0.062, beta 2.0
            return Gen2CurriculumPoint(
                phase=phase,
                epoch=epoch,
                total_epochs=total_epochs,
                eps_norm=0.062,
                eps_pixel=norm_to_pixel_eps(0.062),
                beta=2.0,
                pgd_steps=4,
                w_trades=0.55,
                adversarial=True,
                is_warmup=False,
                gate_trigger=None,
            )
        else:
            # Rung 3: 6 epochs @ eps 0.094, beta 2.5
            return Gen2CurriculumPoint(
                phase=phase,
                epoch=epoch,
                total_epochs=total_epochs,
                eps_norm=0.094,
                eps_pixel=norm_to_pixel_eps(0.094),
                beta=2.5,
                pgd_steps=4,
                w_trades=0.55,
                adversarial=True,
                is_warmup=False,
                gate_trigger=None,
            )
    else:
        # Downstream phases: 15 epochs @ top rung (eps 0.094, beta 2.5)
        total_epochs = 15
        is_warmup = (epoch <= 2)
        return Gen2CurriculumPoint(
            phase=phase,
            epoch=epoch,
            total_epochs=total_epochs,
            eps_norm=0.094,
            eps_pixel=norm_to_pixel_eps(0.094),
            beta=2.5,
            pgd_steps=4,
            w_trades=0.55,
            adversarial=True,
            is_warmup=is_warmup,
            gate_trigger=None,
        )
