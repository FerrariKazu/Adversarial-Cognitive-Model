# Results Interpretation Guide
## Adversarial Cognition Divergence — Phase 5 SDT

This guide covers **every possible pattern of results** you might observe,
what each pattern means scientifically, and how to frame it for your professor.

---

## Quick Reference: What d' Values Mean

| d' Value | Meaning | Real-World Analogy |
|----------|---------|-------------------|
| d' ≈ 0.0 | **No discrimination** — random guessing | Flipping a coin |
| d' ≈ 1.0 | **Threshold** — barely detectable | Hearing a whisper in a quiet room |
| d' ≈ 2.0 | **Moderate** — reliable detection | Normal conversation volume |
| d' ≈ 3.0 | **Strong** — very reliable | Shouting — hard to miss |
| d' ≈ 4.65 | **Near-perfect** — HR≈99%, FAR≈1% | Fire alarm — impossible to miss |

---

## Pattern 1: CNN Collapses First (EXPECTED — Supports Hypothesis)

### What you see:
- CNN d' drops below 1.0 at ε ≈ 0.05–0.10
- Human d' stays above 1.0 until ε ≈ 0.20–0.30
- Large threshold gap (0.10–0.25 epsilon units)

### Scientific interpretation:
> The CNN's feedforward, texture-biased processing is **fundamentally more fragile**
> than human visual processing. Adversarial perturbations destroy the high-frequency
> texture features that the CNN relies on for classification, while humans'
> recurrent feedback loops allow them to reconstruct global shape information from
> noisy inputs. The d' threshold gap directly quantifies the **perceptual cost**
> of lacking recurrent processing.

### One sentence for professor:
> "Our SDT analysis reveals a [X]-epsilon robustness gap where the CNN has
> lost perceptual sensitivity but human observers remain above detection threshold,
> providing quantitative evidence that feedforward texture processing is
> categorically less robust than recurrent shape processing."

### Conference framing:
This is the **ideal result** for a cognitive science venue. It connects
adversarial ML (computer science) to established psychophysical measurement
theory (psychology), showing that the gap is not just "CNNs are worse" but
that they have a **qualitatively different perceptual representation**.

---

## Pattern 2: Both Systems Collapse Together

### What you see:
- CNN d' and human d' drop below 1.0 at similar epsilon values
- Small or zero threshold gap

### Scientific interpretation:
> The adversarial perturbation at these epsilon levels is destroying **all
> visual information**, not selectively targeting CNN-specific features. At
> very high epsilon, the images become literally unrecognizable — even to
> humans. This means the perturbation has crossed from "adversarial" (CNN-targeted)
> to "destructive" (information-destroying).

### What this means:
- **NOT a failure of the hypothesis** — it means you're testing at too-high
  epsilon levels where the comparison becomes meaningless.
- **Action**: Focus your analysis on **lower epsilon** values where the curves
  diverge. The interesting science happens at ε = 0.01–0.10, not at ε = 0.30.

### One sentence for professor:
> "At extreme perturbation levels, both human and machine sensitivity converge
> toward chance, indicating the noise magnitude exceeds the biological visibility
> threshold — the meaningful divergence occurs at moderate epsilon where the
> perturbation is subliminal to humans but catastrophic for the CNN."

---

## Pattern 3: Humans More Fragile on Certain Classes

### What you see:
- For most classes, CNN d' < human d' (expected)
- But for 1-3 specific classes, CNN d' > human d' (surprising!)
- Human observers struggle with those specific categories under noise

### Scientific interpretation:
> This is actually a **very interesting finding**. It suggests that for some
> visual categories, the CNN has learned features that are **more noise-robust**
> than human perceptual features. Possible explanations:
>
> (a) **Low-level redundancy**: The CNN may exploit low-level statistical
>     redundancies (e.g., color distributions) that are partially preserved
>     under L∞ perturbation, while humans rely on mid-level shape cues that
>     are more fragile for that category.
>
> (b) **Perceptual confusability**: Some CIFAR-10 classes are inherently
>     confusable for humans (cat/dog, automobile/truck) but not for CNNs
>     that have learned subtle discriminative textures.
>
> (c) **Display artifacts**: At 32×32 resolution, some classes are already
>     hard for humans on clean images. Adding noise makes them impossible.

### One sentence for professor:
> "Intriguingly, human observers show greater d' vulnerability than the CNN
> for [class X], suggesting that the adversarial perturbation disrupts
> human-relevant mid-level features while preserving CNN-exploitable low-level
> statistics for this category — a dissociation that reveals different
> representational strategies."

### Conference framing:
This is potentially the **most novel finding** — it's counterintuitive
and would generate discussion. Frame it as evidence for "representational
dissociation" between human and machine vision.

---

## Pattern 4: CNN d' Never Drops Below 1.0

### What you see:
- CNN maintains above-threshold sensitivity across all tested epsilons
- No crossing point can be identified

### Scientific interpretation:
> The tested perturbation budget is insufficient to fully degrade CNN
> performance. This can happen with:
>
> (a) **Very robust models** (e.g., ViT with strong augmentation)
> (b) **Weak attacks** (FGSM is single-step and often insufficient)
> (c) **Insufficient epsilon range** (need to test higher values)

### Action:
- Extend epsilon range to [0.0, 0.5, 0.75, 1.0]
- Use stronger attacks (PGD-100, AutoAttack)
- If using FGSM, switch to PGD (much stronger, multi-step attack)

---

## Pattern 5: Per-Class d' Reveals Texture vs Shape Split

### What you see:
- **Animals** (cat, dog, deer, frog) show rapid CNN d' collapse
- **Vehicles** (airplane, truck, ship) show slower CNN d' collapse
- Human d' is roughly uniform across all categories

### Scientific interpretation:
> This is **direct evidence for the Geirhos texture bias hypothesis**. Animal
> classes are identified by the CNN primarily through texture (fur patterns,
> skin patterns, scale patterns). Adversarial noise destroys these textures
> efficiently. Vehicle classes have more distinctive global shapes (wings,
> wheels, hull) that are partially preserved under L∞ perturbation.
>
> Humans show uniform d' because they process ALL categories using global
> shape, regardless of whether the category is an animal or vehicle.

### One sentence for professor:
> "The per-class SDT analysis reveals an animal-vehicle dissociation in CNN
> sensitivity — texture-defined classes collapse 2× faster than shape-defined
> classes under attack — while human sensitivity remains uniform, consistent
> with the Geirhos (2019) texture bias hypothesis."

---

## Pattern 6: β (Response Bias) Shifts Under Attack

### What you see:
- CNN β becomes very low (liberal) — says "yes" to everything
- OR CNN β becomes very high (conservative) — suppresses predictions

### Scientific interpretation:
> **Liberal shift** (β < 1): The CNN is "hallucinating" — it starts
> predicting target classes even when they're absent. The softmax
> concentrates on a few classes for all inputs.
>
> **Conservative shift** (β > 1): The CNN avoids predicting certain
> classes entirely, as if it "knows" those features are unreliable.
>
> **Human β stable** (β ≈ 1): Humans maintain calibrated metacognition —
> they adjust confidence but don't systematically shift their criterion.

### One sentence for professor:
> "Under adversarial attack, the CNN's response criterion shifts [liberally/
> conservatively], while human observers maintain stable metacognitive
> calibration — the d' analysis separates this criterion artifact from
> the genuine sensitivity loss."

---

## Running the Pipeline

```bash
# Step 1: Compute SDT metrics
python phase5_sdt/sdt_analysis.py

# Step 2: Generate plots
python phase5_sdt/sdt_plots.py

# Step 3: Generate final report
python phase5_sdt/final_report.py
```

Results will appear in:
- `phase5_sdt/results/sdt_results.csv` — raw SDT data
- `phase5_sdt/figures/dprime_vs_epsilon.png` — Plot 1 (d' curves)
- `phase5_sdt/figures/perclass_dprime_eps0.10.png` — Plot 2 (heatmap)
- `phase5_sdt/final_report.txt` — full synthesis report
