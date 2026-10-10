# Gen-1 Foundation Quarantine Notice

**Date:** 2026-10-10  
**Status:** Quarantined — Historical Artifact Only

## Finding
Gen-1 foundation results prior to 2026-10-10 were effectively clean-trained (TRADES gradient never reached the optimizer during the from-scratch CompactViT run, and evaluation on epoch 49 confirmed 5.82% clean top-1 and 0.00% robust top-1 under PGD-10 $\epsilon=0.094$).

## Quarantine Actions
1. Checkpoints from commit `fef50f3` (`foundation_backbone_only_rolling.pth` and `foundation_backbone_only_best.pth`) are archived under `archive/invalid_parent/` with `status: invalid_parent`.
2. Downstream foundation phases (`recurrence_only`, `belief_no_f`, etc.) must not inherit weights from this parent.
3. Hugging Face storage repos for Gen-2 are partitioned into dedicated repos (`FerrariKazu/rhan-nxa-g2-checkpoints` and `FerrariKazu/rhan-nxa-g2-checkpoints-rolling`) with code-identity guards to prevent silent resumption or cross-contamination.
