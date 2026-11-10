# Output of scripts/rhan_im/forensic_checks.py

Plain output of the audit script (random-init instantiation of the official ff_112 and recurrence_112 configurations; no weights, no data). Referenced from RHAN_IM_01_FORENSIC_AUDIT.md.

```
[ff_112] state_dict numel=23,595,749  unique-storage numel=23,369,573  parameters() numel=23,369,573  requires_grad numel=23,369,573
   aliased groups: [['backbone.patch_embed.proj.weight', 'gist_encoder.patch_embed.proj.weight'], ['backbone.patch_embed.proj.bias', 'gist_encoder.patch_embed.proj.bias']]
   diff state_dict - parameters() = 226,176  (patch-embed weight+bias = 226,176)
[ff_112] params with NO gradient after fwd+bwd: 15 tensors, 1,775,233 values
      backbone.refinement: 1,775,232
      gist_encoder.gate: 1
   refinement block total = 1,775,232 values
[recurrence_112 (arch_phase=recurrence_only)] state_dict numel=23,595,749  unique-storage numel=23,369,573  parameters() numel=23,369,573  requires_grad numel=23,369,573
   aliased groups: [['backbone.patch_embed.proj.weight', 'gist_encoder.patch_embed.proj.weight'], ['backbone.patch_embed.proj.bias', 'gist_encoder.patch_embed.proj.bias']]
   diff state_dict - parameters() = 226,176  (patch-embed weight+bias = 226,176)
[recurrence_112] params with NO gradient after fwd+bwd: 1 tensors, 1 values
      gist_encoder.gate: 1
   refinement block total = 1,775,232 values

--- foveal_sample at H=112, fovea=112 (the flagship configuration)
gaze (0,0): max|crop - frame| = 4.410743713378906e-05
gaze (0.3, 0.3): shift = 16.8px,16.8px -> 27.8% of the crop is border-replicated; no pixel outside the original 112x112 frame is ever sampled
gaze (-0.3, 0.3): shift = 16.8px,16.8px -> 27.8% of the crop is border-replicated; no pixel outside the original 112x112 frame is ever sampled
gaze (0.0, -0.3): shift = 0.0px,16.8px -> 15.0% of the crop is border-replicated; no pixel outside the original 112x112 frame is ever sampled

--- contrast: canvas 224, window 112 at centre: max|crop - native 112 centre crop| = 5.078315734863281e-05 (native-resolution crop of a larger frame)
```
