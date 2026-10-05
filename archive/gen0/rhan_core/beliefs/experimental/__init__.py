"""
rhan_core/beliefs/experimental — explicitly OUTSIDE the main config/gating
system.

Anything under this package is a standalone, read-only probe or experiment —
never imported by the training graph, never wired into RHANNextConfig, never
merged into the ablation matrix. Importing from here has ZERO side effects on
the model, the loss, or the roadmap. (Pillar 3 / SBR remains scaffold-only:
enable_sbr stays locked at False and RHANNextConfig.validate() keeps
rejecting it.)
"""
