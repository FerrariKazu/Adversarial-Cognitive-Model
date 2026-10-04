"""Production halt guard — Generation-1 foundation run CANCELLED 2026-10-03.
================================================================================

Directive: "# RHAN-NXA — PRODUCTION RUN CANCELLATION + FORENSIC TRAINING
FAILURE DIAGNOSIS" (§1, §20). The current production iteration is stopped
formally; continuing it would produce little useful scientific information.

This guard is:
  * LOUD      — it prints the reason and cites the cancellation notice;
  * FAIL-CLOSED — real-data training aborts unless explicitly overridden;
  * REVERSIBLE — only by an explicit, documented action:
        J1_ALLOW_TRAINING=1  (and a written authorization note in
        docs/CANCELLATION_NOTICE_2026-10-03.md) — never by editing code
        silently.

Smoke mode (--smoke: synthetic loaders, no HF writes, quarantined
artifacts) is NOT blocked: it is the orchestration proof, costs seconds,
and cannot produce a citable result.
"""
from __future__ import annotations

import os

CANCELLATION_NOTICE = "docs/CANCELLATION_NOTICE_2026-10-03.md"

_BANNER = r"""
==============================================================================
  PRODUCTION RUN CANCELLED (2026-10-03) — training is HALTED.

  The Generation-1 foundation run was formally cancelled by directive:
  the adversarial-recipe iteration showed a fundamental learning failure
  (belief_with_f plateaued at ~13.6% val_acc / loss ~2.12 from epoch 4;
  backbone/recurrence phases finished at 5.9%/5.8%).

  Refusing to train. See: docs/CANCELLATION_NOTICE_2026-10-03.md
  Forensic work lives on branch diagnosis/nxa-forensic-2026-10-03.

  To override (requires written authorization recorded in the notice):
      export J1_ALLOW_TRAINING=1
==============================================================================
"""


def enforce_cancellation(smoke: bool) -> None:
    """Abort real-data training while the cancellation stands.

    Args:
        smoke: True for --smoke (synthetic, quarantined) — never blocked.
    Raises:
        SystemExit: when real-data training is attempted.
    """
    if smoke:
        return
    if os.environ.get("J1_ALLOW_TRAINING") == "1":
        print("[halt-guard] WARNING: J1_ALLOW_TRAINING=1 — cancellation "
              "override is ACTIVE. This must be backed by a written "
              "authorization in " + CANCELLATION_NOTICE, flush=True)
        return
    print(_BANNER, flush=True)
    raise SystemExit(
        "STOP — Generation-1 production run is cancelled (2026-10-03); "
        f"see {CANCELLATION_NOTICE}. Set J1_ALLOW_TRAINING=1 ONLY with "
        "written authorization.")
