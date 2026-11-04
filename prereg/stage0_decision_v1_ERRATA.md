# Errata: Stage 0 Decision Prereg Batch Fields

**Affects:** `prereg/stage0_decision_v1.json` lines 13-15
**Status:** correction only; the original prereg file is not edited.

The fields

- `effective_batch_size: 64`
- `micro_batch_size: 16`
- `grad_accum_steps: 4`

do not describe any Stage 0 run. The Gen-2 training code has no gradient accumulation; the loader batch size is the effective batch size per optimizer step.

Actual executions:

- `ff_112` ran at loader batch 48: `126,689 // 48 = 2,639` steps per epoch. The final AdamW step counters are `39,566` (seed 17), `39,568` (seed 42), and `39,567` (seed 99).
- The flagship `recurrence_112` ran at loader batch 64: `126,689 // 64 = 1,979` steps per epoch. Seed 42's final AdamW step counter is `29,668`.

This errata does not change the pre-registered decision rule, the verdict, or any result file. `prereg/stage0_decision_v1.json` remains unchanged.
