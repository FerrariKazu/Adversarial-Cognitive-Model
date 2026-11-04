# Prereg Batch/Step Audit (Part 13)

**Scope:** read-only. No prereg or docs file was edited.
**Reference points:** `ff_112` = loader batch 48 => `126,689 // 48 = 2,639` steps/epoch; flagship = loader batch 64 => `126,689 // 64 = 1,979` steps/epoch. The Gen-2 training code has no gradient accumulation; the loader batch size is the effective batch size per optimizer step.

**Search command:**

```bash
rg -n -i 'batch[ _-]?size|micro[ _-]?batch|gradient accumulation|grad[ _-]?accum|accumulation|steps[ _-]?per[ _-]?epoch|effective_batch|steps/epoch' prereg/*.json docs/*.md
```

---

## 1. Prereg files in scope

| File:line | Statement | Correct? | Assessment |
| :--- | :--- | :---: | :--- |
| `prereg/stage2_lever1_epochs_v1_2.json:8` | `"effective_batch_size_prereg": 64, "micro_batch_size_prereg": 16, "grad_accum_steps_prereg": 4, "code_reality": "no gradient accumulation ... (126689//64 steps/epoch)"` | **No (partly)** | The `code_reality` sentence is correct: there is no gradient accumulation in the Gen-2 code. The three batch fields are not a description of the executed `ff_112` recipe, which used batch 48. The protocol describes itself as "ff_112 recipe unchanged except 30 epochs", so switching the effective batch to 64 is not "unchanged". |
| `prereg/stage2_lever1_epochs_v1_2.json:9` | `"steps_per_epoch": 1979` | **Only for batch 64** | `126,689 // 64 = 1,979`. The existing 15-epoch `ff_112` baselines ran at batch 48 and `2,639` steps/epoch, so this is not the baseline recipe. |
| `prereg/stage0_decision_v1.json:13` | `"effective_batch_size": 64` | **No as a universal field** | The flagship used batch 64, but the `ff_112` baselines used batch 48. The field describes the flagship only. |
| `prereg/stage0_decision_v1.json:14` | `"micro_batch_size": 16` | **No** | No Stage 0 run used micro-batch 16. The flagship used micro 64; the feedforward baselines used micro 48. |
| `prereg/stage0_decision_v1.json:15` | `"grad_accum_steps": 4` | **No** | No Stage 0 run used gradient accumulation. The code has no accumulation path. |
| `prereg/stage3_gate0_v1_3.json` | No batch size, micro-batch, accumulation, or steps-per-epoch statement found. | **N/A** | Measurement-only protocol; no training batch is specified. No new version is required on these grounds. |
| `prereg/multiview_control_prereg_v1_4.json` | No batch size, micro-batch, accumulation, or steps-per-epoch statement found. | **N/A** | Evaluation-only protocol; no training batch is specified. No new version is required on these grounds. |

### Files needing a new version

1. **`prereg/stage0_decision_v1.json`** — lines 13-15 misdescribe every Stage 0 run. A new version should record the actual two batch sizes: flagship 64 (`1,979` steps/epoch) and `ff_112` 48 (`2,639` steps/epoch), and should delete the micro-16 / accumulation-4 claim.
2. **`prereg/stage2_lever1_epochs_v1_2.json`** — lines 8-9. If the 30-epoch `ff_112` extension is meant to be recipe-matched to the 15-epoch baselines, it must use batch 48 and `2,639` steps/epoch, or it must explicitly declare a batch change and stop calling the recipe unchanged. A new version (`v1.3`) is needed before launch.

No new version is required for `stage3_gate0_v1_3.json` or `multiview_control_prereg_v1_4.json` on batch/step grounds.

---

## 2. `docs/*.md` statements found by the same search

| File:line | Statement | Correct for Stage 0? | Assessment |
| :--- | :--- | :---: | :--- |
| `docs/stage3_preregistration.md:70` | `Batch size 32` | **Out of scope** | This is a separate Stage 3 protocol, not the Stage 0 112px runs. No Stage 0 action. If it is meant to describe current 112px training, it is stale and needs its own revision. |
| `docs/REPOSITORY_MAP.md:724` | CLI default `--batch-size 48` | **Yes** | Matches the `ff_112` default and the actual Stage 0 feedforward batch. |
| `docs/ARCHITECTURE.md:492` | Example `--batch-size 16 --accum-steps 16` | **Out of scope** | This is a Stage 1 diagnostic command for a different trainer path, not the Gen-2 `run_gen2_phase` path. |
| `docs/ARCHITECTURE.md:618`, `:625`, `:630` | Examples `--batch-size 64` | **Out of scope** | Stage 1 sweep examples, not Stage 0 112px training. |
| `docs/Manual Documentation.md:49` | `evaluate.py` default batch size 64 | **Out of scope** | Evaluation utility, not training. |

No Stage 0 doc edit is required from this audit; the only Stage 0 corrections belong in the two prereg JSON files listed above.
