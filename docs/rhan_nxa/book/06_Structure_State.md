# Chapter 06 — Structure State S_t: The None-Contract and Slot Dynamics

## 1. In one sentence
$S_t$ is the slot-based structured/relational component of the belief state, currently **locked to None** in the Generation-1 core build, with re-entry deferred to post-Step-6 validation at 2–4 slots only.

## 2. Intuition
While $z_t$ is a dense global vector that summarizes the entire scene as one continuous representation, $S_t$ was designed to hold *structured*, object-level or relational representations—a small set of "slots," each encoding one meaningful scene element (e.g., "object at left carrying a pattern," "background region").

The idea draws on the Slot Attention literature: rather than pooling all visual information into one vector, the system maintains a small set of discrete slot vectors that compete for explaining different parts of the scene. In principle, this enables more compositional reasoning and more targeted attention.

## 3. Why it is locked to None in Gen-1

Generation-0 implemented a 16-slot `SlotAttention` bottleneck. **This specific implementation is rejected outright** for two reasons documented in `schema.py`:

1. The 16-slot architecture introduced severe gradient conflicts with the recurrent update path.
2. The SBR confound (`_nx_trainer` bug) means the isolated contribution of the slot mechanism can never be cleanly attributed from Gen-0 data.

The research plan mandates that the core Generation-1 loop be validated at $S_t = \text{None}$ **first**—establishing a clean, unconfounded baseline. Only after Step-6 produces a validated ImageNet-100 result will $S_t$ re-enter the system, at 2–4 slots and with an explicit structural isolation experiment.

## 4. Implementation status

| Status | Meaning |
|:---|:---|
| **REJECTED** | 16-slot Gen-0 implementation. Never re-introduced as a fallback without an explicit plan ruling. |
| **LOCKED (None)** | Current Gen-1 core build default. `enable_sbr = False` in `RHANNXAConfig`. |
| **DEFERRED** | 2–4 slot concept pending Step-6 validation. Gated by `step6_validated_result`. |

The gate is enforced in `RHANNXAConfig.__post_init__`:

```python
if getattr(self, "enable_sbr") and not getattr(self, "step6_validated_result"):
    raise ValueError(
        "enable_sbr=True requires step6_validated_result ..."
    )
```

Setting `enable_sbr = True` without a recorded `step6_validated_result` is a hard runtime error—not a warning, not a soft override.

## 5. Mathematical formulation (for documentation completeness)
When and if $S_t$ re-enters the system, the belief state becomes:

$$B_t = (z_t, S_t, U_t, E_t, A_t), \quad S_t \in \mathbb{R}^{B \times K \times D_s}$$

where $K \in \{2, 3, 4\}$ is the number of slots and $D_s$ is the slot dimension. The `as_tensor()` readout, which currently returns $z_t$ directly, would then require an explicit fusion mechanism—a `NotImplementedError` is deliberately raised in `VectorBeliefState.as_tensor()` for this path to prevent silent improvisation.

## 6. None-safety invariant (mandatory)
**Every consumer of a `BeliefState` must have an explicit branch for `S_t is None`.** The interfaces.py header states: *"this is not a hypothetical: it is the actual default configuration."*

This means:
- Do not write code that silently assumes `S_t` is a tensor.
- Do not write code that silently assumes `S_t` is None (which would break the future re-entry).
- Always test both cases independently.

## 7. Scientific status
- **16-slot Slot Attention (Gen-0)**: **REJECTED OUTRIGHT**.
- **$S_t = \text{None}$ in Gen-1 core**: **LOCKED**.
- **2–4 slot concept**: **EXPERIMENTAL CANDIDATE**, **DEFERRED** pending Step-6 result.

## 8. Source references
- [`schema.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/core/schema.py): `enable_sbr`, `sbr_num_slots`, `REJECTED_OUTRIGHT`, `GATED_FLAGS`.
- [`interfaces.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/beliefs/interfaces.py): `BeliefState.s` property; None-safety documentation.
- [`vector_belief.py`](file:///home/ferrarikazu/Adversarial%20Cognitive%20Model/noesis_vision/beliefs/vector_belief.py): `VectorBeliefState.as_tensor()` with explicit `S_t is None` branch.

## 9. Related components
- Chapter 04 (Belief State): The container that holds $S_t$.
- Chapter 21 (Decision Records): DR-001 through DR-004 document the original slot rejection rationale.
- Chapter 17 (Experimental DAG): Step-6 is the gating condition for $S_t$ re-entry.
