# Quarantined Checkpoints: invalid_parent

- `foundation_backbone_only_best.pth` (metric: 0.059 clean val top-1)
- `foundation_backbone_only_rolling.pth` (epoch 49, commit `fef50f3`)

These checkpoints are quarantined as `invalid_parent` per S0 quarantine specification. Downstream phases must never inherit weights from these checkpoints.
