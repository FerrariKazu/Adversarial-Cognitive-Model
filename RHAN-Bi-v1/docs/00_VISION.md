# 00 — Vision

> **RHAN is a reliability-governed visual perception system that constructs traceable beliefs from
> measured evidence, actively seeks observations that challenge competing interpretations, and
> revises its understanding according to the quality, independence, and consistency of the evidence
> acquired.**

## 1. The problem with confident perception

A conventional vision system maps an image to a label and a score. The score is usually a softmax
value, which measures the model's internal certainty and not the adequacy of the evidence. Two
failures follow:

1. **Unjustified confidence.** A system can be 96 % confident on an image whose relevant region is
   occluded, blurred, or lit in a way that makes a reflection look like a defect. Nothing in the
   architecture represents the difference between "the evidence is strong" and "the model is
   confident".
2. **Redundant looking.** When such a system is allowed to take more observations, it tends to take
   more of the same observation — another crop at the same angle under the same light. Because the
   new pixels are highly correlated with the old ones, confidence rises while information does not.
   RHAN-IM measured exactly this: a variance inflation factor of ≈11.6 under its repeated-look
   protocol.

The second failure is the one that makes "look again" a weak idea unless the system also knows *what
kind of look* would be informative.

## 2. Three principles

### P1 — Every interpretation carries an evidence basis

The system maintains, for each hypothesis it is considering, the observations that support it, the
observations that contradict it, the quality of those observations, and their dependence on one
another. This is not an explanation layer bolted onto a classifier: it is the state the classifier's
output is read from. The ledger is the system's memory, not a report about its memory.

### P2 — Uncertainty governs action

Posterior uncertainty, evidence debt and expected value of information decide whether the system
looks again, what it looks at, and when it stops. A system that always looks once is not adaptive; a
system that always looks until the budget is exhausted is not deciding. Both are baselines here.

### P3 — The system seeks disconfirmation

Given a leading interpretation and its strongest alternative, the planner prefers the observation
whose expected outcome would most change the leading interpretation if that interpretation is wrong.
This is deliberately *not* the same as maximizing expected information gain: information gain is
maximized by any observation that resolves any uncertainty, including uncertainty the decision does
not depend on. Falsification-directed selection asks a narrower and more useful question: *which
observation would tell me that I am wrong?*

## 3. The one question

> **What evidence would prove my current interpretation wrong, and what is the most efficient
> reliable way to obtain it?**

Every design decision below is judged against that question. If a component does not help answer it,
it is not in v1.

## 4. What RHAN-Bi is not

| Not this | Why |
|---|---|
| a bigger classifier | accuracy is necessary but not the contribution; Stage A exists to establish a competent encoder, not to be the thesis |
| a multi-agent system | more agents do not produce better evidence; v1 has one loop and six measurable components |
| an LLM-centred system | an LLM may later phrase hypotheses or coordinate investigation, but visual truth, reliability and revision are computed, not narrated |
| a JEPA-scale training project | predictive representation learning is interesting, but reproducing ViT-Huge pretraining is not a prerequisite for testing evidence governance |
| a simulator paper | the simulator is the laboratory. Synthetic competence is not evidence of real-world competence, and the documents say so at every gate |
| a causal inference engine | CEE is the destination. v1 ends at Gate D; Gate E is specified, not claimed |

## 5. The scientific identity

**Reliable Adaptive Visual Perception through Evidence Acquisition, Evidence Evaluation, and Belief
Revision.**

The candidate contribution is not "a model that can look again". Active perception, object-centric
representation, uncertainty estimation and selective prediction all exist. The candidate contribution
is that **the integrity of the evidence is part of the perception-and-action problem**: the system's
decision about what to do next is computed from a state that records how good, how independent and how
contradictory its evidence is, and every belief change is traceable to the records that caused it.

Whether that combination is novel is an empirical question about the literature, not a claim in this
document. `docs/14_LITERATURE_AND_NOVELTY.md` specifies the audit that must be completed before any
novelty claim is written, and `docs/09_EXPERIMENTS_AND_GATES.md` specifies what has to be true in the
experiments before the contribution is asserted.

## 6. Scope of v1

In scope:

- a deterministic inspection simulator with ground truth and controlled interventions (S1–S4);
- an evidence ledger with content hashing, provenance and dependence relations;
- a measured reliability model (quality, dependence, conflict kept separate);
- an explicit belief state with a documented update rule, contradiction detection and evidence debt;
- a planner with five policies, including the falsification-directed one, compared at matched budget;
- a stopping rule evaluated for coverage as well as risk;
- training stages A–E at a scale that runs on one 8 GB consumer GPU and, for the demo, on CPU.

Out of scope in v1 (specified but deferred):

- Unreal Engine rendering (the API contract is in `docs/18_UNREAL_HANDOFF.md`);
- real industrial data beyond a transfer check on MVTec AD 2 (Gate C/D secondary);
- LLM hypothesis generation and natural-language explanation;
- causal experimentation and CEE (Gate E);
- joint end-to-end training of the whole loop (deliberately: see `docs/07_TRAINING_PROGRAM.md` §2).

## 7. What success would look like

A successful v1 does not prove that RHAN understands visual reality. It establishes four narrower
things:

1. a single observation has a measurable ceiling on this task, and the ceiling is caused by
   identifiability rather than by model capacity;
2. an evidence-aware belief state revises correctly when a disambiguating observation arrives, and
   its confidence tracks its accuracy;
3. dependence-aware updating avoids the overconfidence that repeated correlated looks produce in an
   ablated system;
4. at the same number of observations, falsification-directed selection reduces decision error
   relative to named baselines, with a confidence interval that excludes no effect.

If (4) fails while (1)–(3) hold, the programme still has a result: evidence governance without a
better planner is a calibration and auditing contribution, not an active-perception one. That is a
legitimate outcome and will be reported as such.
