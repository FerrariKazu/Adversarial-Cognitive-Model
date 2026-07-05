# RHAN Scientific Evidence Figure Suite (v2)

Hey! Welcome to the v2 visualization suite. The figures in this directory are designed to serve as the core empirical evidence for the **Robust Hierarchical Attention Network (RHAN)** paper. While the v1 figures layout the network's structural wiring, this suite provides the quantitative and qualitative plots that reviewers look for to understand *why* the model behaves robustly under severe noise where traditional vision models collapse.

---

## 📂 Export Options & Theme Variants

For every single figure registry below, we generate five separate files in the directory to make paper formatting and slide design as easy as possible:
1. **`*_light.pdf` & `*_light.svg`**: Clean vector files using standard high-contrast scientific palettes and pure white backgrounds. Embed these directly into your LaTeX drafts to ensure text remains crisp at any zoom level.
2. **`*_light.png`**: A high-resolution (300 DPI) raster version with a clean white background.
3. **`*_dark.png`**: PowerPoint/Keynote optimized slide versions. They feature a dark slate background (`#0d1117`), gold titles, and bright, high-contrast curves that look incredibly sharp under presentation projectors.
4. **`*_transparent.png`**: Background-free versions with high-contrast elements. Use these if you want to overlay plots directly onto custom slide layouts, shapes, or multi-panel grids.

---

## 🖼️ Detailed Figure Registry & Reviewer Insights

### 🏛️ Group 1: Representation Geometry (`figures_v2/representation/`)

#### 🔹 Figure A1: UMAP Feature Space (`figure_a1_umap_feature_space`)
* **What it shows:** A 2D projection comparing the latent clusters of three classes (Airplane, Bird, Ship) under clean and adversarial conditions.
* **Why it matters:** Standard models like ResNet-18 and ViT show beautifully separated clusters on clean data, but under a PGD attack, their latent features scatter and bleed into each other, destroying classification boundaries. RHAN's features, however, remain tightly clustered and separated even under attack, proving representation stability.
* **Writing Tip:** Use this in your *Representation Analysis* section. Point out how the boundary margins of the classes in RHAN shrink only slightly under noise, whereas ResNet's boundaries collapse completely.

#### 🔹 Figure A2: t-SNE Representation Drift (`figure_a2_tsne_representation_drift`)
* **What it shows:** Individual clean samples plotted alongside their corresponding adversarial perturbers, with colored arrows indicating the direction and magnitude of the displacement vector ($z_{clean} \to z_{adv}$).
* **Why it matters:** It visualizes the *path* of the attack. In ResNet and ViT, the arrows are long and point erratically across class boundaries. In RHAN, the drift vectors are tiny and remain well within the original class boundaries, proving the model restricts the adversary's capability to manipulate latent codes.
* **Writing Tip:** Highlight that the mean vector displacement is bounded by the spherical classification head, preventing the network from exploding represented differences.

#### 🔹 Figure A3: Representation Drift Distance Histogram (`figure_a3_representation_drift_histogram`)
* **What it shows:** A frequency histogram of clean-to-adversarial representation distance ($||z_{clean} - z_{adv}||_2$) for ResNet-18, ViT-Small, and RHAN over 1000 test samples.
* **Why it matters:** It provides statistical proof. RHAN has a narrow, tightly concentrated distribution peaking near `0.18`, while ResNet-18 peaks near `0.85` with a wide variance. This shows that representation stability is a system-wide property of RHAN, not just a fluke on a few samples.

#### 🔹 Figure H1: 2D Decision Boundary Slices (`figure_h1_decision_boundary_slices`)
* **What it shows:** A 2D slice of the classification boundary around a clean sample, showing the path a PGD attack takes to cross into a wrong class region.
* **Why it matters:** Standard models display highly fragmented, thin, and complex decision boundaries—allowing an adversary to find a backdoor with a tiny nudge. RHAN's boundary is smooth and wide, requiring a massive perturbation to push the sample out of its correct class zone.

#### 🔹 Figure H2: Adversarial Loss Landscape (`figure_h2_loss_landscape`)
* **What it shows:** A 1D cross-section of the loss function along the direction of adversarial perturbation.
* **Why it matters:** Standard architectures display a "sharp minimum"—a narrow valley surrounded by steep loss spikes. This makes optimization highly sensitive to noise. RHAN exhibits a "flat minimum basin" where the loss remains low and stable, demonstrating robustness to local shifts.

---

### 👁️ Group 2: Attention Analysis (`figures_v2/attention/`)

#### 🔹 Figure B1: Attention Target Focus Overlay (`figure_b1_attention_overlay`)
* **What it shows:** Transformer self-attention maps overlayed on clean vs. attacked images of a target object (e.g., a bird).
* **Why it matters:** Traditional ViTs lose their focus under attack, shifting their attention to random high-frequency background noise generated by the PGD perturber. RHAN's top-down feedback loops filter out this noise, keeping the attention maps focused on the actual shape of the bird.
* **Writing Tip:** Use this to visually demonstrate the benefit of the recurrent feedback path—it acts as an active visual attention gate.

#### 🔹 Figure B2: Attention Focus Evolution (`figure_b2_attention_evolution`)
* **What it shows:** A four-panel panel tracking attention maps across recurrent steps (Step 1 $\to$ Step 2 $\to$ Step 3 $\to$ Final).
* **Why it matters:** Proves that the attention is not static. It starts diffuse (first forward pass) and progressively sharpens and concentrates on the object's foreground features as the predictive coding loop converges.
* **Writing Tip:** Frame this as evidence of *temporal refinement* in neural representations, mimicking how the human brain spends 100–150ms to "fill in" object details in noisy scenes.

#### 🔹 Figure B3: Feedback Correction Flow (`figure_b3_feedback_correction`)
* **What it shows:** Subplots showing intermediate representations: the top-down prediction prior, the prediction error map, the frequency gating weights, and the resulting clean feature map.
* **Why it matters:** Shows the inner workings of the predictive coding layer. It visually explains *how* the gating filter ($g(e^t)$) selectively suppresses the high-frequency components of the error map to reconstruct a clean feature output.

---

### 📶 Group 3: Frequency Gating Dynamics (`figures_v2/frequency/`)

#### 🔹 Figure C1: Low-Frequency Gate ($w_L$) vs. $\varepsilon$ (`figure_c1_low_frequency_gate`)
* **What it shows:** The transmission factor of the low-frequency gate plotted against noise level epsilon.
* **Why it matters:** Proves that the model continues to pass coarse, low-frequency semantic information (shapes, layouts) regardless of how noisy the input becomes. The weight remains high ($\sim 0.95$).

#### 🔹 Figure C2: High-Frequency Gate ($w_H$) vs. $\varepsilon$ (`figure_c2_high_frequency_gate`)
* **What it shows:** The transmission factor of the high-frequency gate plotted against noise level epsilon.
* **Why it matters:** Shows that the model dynamically shuts down the high-frequency channels (from `0.8` down to `0.05`) as noise increases, protecting the deep layers from adversarial perturbations.

#### 🔹 Figure C3: Gate Coefficients Throughout Training (`figure_c3_gate_weights_training`)
* **What it shows:** The training history of $w_L$ and $w_H$ across the three stages of training.
* **Why it matters:** Demonstrates that the gating mechanism adapts during training: $w_L$ is learned early, while $w_H$ becomes increasingly selective as the training curriculum increases the epsilon perturbation budget.

---

### 📐 Group 4: Prototype Geometry (`figures_v2/geometry/`)

#### 🔹 Figure D1: Spherical Prototypes Hypersphere (`figure_d1_spherical_prototypes`)
* **What it shows:** Feature vectors and class prototypes projected onto a 3D unit sphere.
* **Why it matters:** Visualizes the angular classification head. Standard linear classifiers separate space with unbounded hyperplanes, leaving them vulnerable to feature scaling attacks. RHAN constrains all representations to $S^{D-1}$, making classification invariant to scaling.

#### 🔹 Figure D2: Angular Decision Boundaries Schematic (`figure_d2_angular_decision_boundaries`)
* **What it shows:** Comparison of unbounded flat boundaries in standard linear classification vs. bounded conical angular sectors in spherical classification.
* **Why it matters:** Explains the math visually. The spherical head forces features to group into tight, class-specific angular cones, leaving no vast "empty zones" for an adversary to exploit.

#### 🔹 Figure D3: Feature-to-Prototype Angular Distribution (`figure_d3_angular_margin_distribution`)
* **What it shows:** Histogram of the angle (in degrees) between feature representations and their target prototypes.
* **Why it matters:** Proves that correct classifications are tightly grouped near $0^\circ-20^\circ$ (high cosine similarity), while incorrect/noisy classifications are spread out, providing a natural confidence calibration.

---

### 📊 Group 5: Robustness Performance Sweeps (`figures_v2/evaluation/`)

#### 🔹 Figure E1: Accuracy vs. Epsilon Curves (`figure_e1_accuracy_vs_epsilon`)
* **What it shows:** Robust decay curves under PGD sweeps comparing all baseline models (ResNet, ViT, RHAN base, RHAN Large) with the Human Visual ceiling.
* **Why it matters:** Ranks the model. Standard models drop to 0% accuracy by $\varepsilon = 0.03$. RHAN Large stays robust, showing a slow, natural decay profile similar to human observers.

#### 🔹 Figure E2: Sensitivity $d'$ vs. Epsilon Curves (`figure_e2_dprime_vs_epsilon`)
* **What it shows:** Sensitivity index ($d'$) decay curves compared to actual human visual study data.
* **Why it matters:** This is the ultimate metric of the paper. It shows that RHAN is the first model to match the human sensitivity curve shape, maintaining $d' > 1.0$ at high noise levels where feedforward networks fall below the detection threshold.

#### 🔹 Figure E3: Robustness Threshold Comparison (`figure_e3_robustness_threshold_comparison`)
* **What it shows:** Horizontal bar chart comparing the boundary threshold $\varepsilon_{thresh}$ (the noise level where $d'$ drops to 1.0) for every model.
* **Why it matters:** Shows the quantitative improvement. RHAN Large pushes the threshold boundary to `0.250`, coming very close to the human boundary of `0.300` and far exceeding ResNet's `0.030`.

#### 🔹 Figure E4: Class Robustness Heatmap (`figure_e4_class_robustness_heatmap`)
* **What it shows:** Grid mapping each of the 10 STL-10 classes (y-axis) against epsilon levels (x-axis), color-coded by accuracy.
* **Why it matters:** Shows class-wise stability. It reveals that classes like *Ship* and *Airplane* maintain high robustness longer due to strong geometric shapes, while *Cat* and *Dog* degrade earlier, matching biological observations.

---

### 🧠 Group 6: Biological Recurrence Dynamics (`figures_v2/biology/`)

#### 🔹 Figure F1: Predictive Coding Convergence (`figure_f1_predictive_coding_convergence`)
* **What it shows:** Line plot showing the prediction error ($e^t$) decreasing monotonically over 5 recurrent iterations.
* **Why it matters:** Proves stability. It shows that the recurrent feedback loop converges cleanly to a stable minimum rather than oscillating or exploding.

#### 🔹 Figure F2: ACT Pondering Steps vs. Difficulty (`figure_f2_act_pondering`)
* **What it shows:** Scatter plot of logits entropy (image difficulty) vs. the number of recurrent steps utilized by the model.
* **Why it matters:** Proves computational efficiency. The model uses only 1–2 steps for clear, easy images, but dynamically scales up to 4–5 steps for noisy, hard-to-classify inputs—demonstrating adaptive resource allocation.

#### 🔹 Figure F3: Recurrence Utilization Distribution (`figure_f3_recurrence_utilization`)
* **What it shows:** Histogram of the recurrence steps utilized across the STL-10 dataset.
* **Why it matters:** Shows that on average, the model does not waste computation; the bulk of clean images are processed quickly, saving power.

---

### 🔄 Group 7: Training Dynamics (`figures_v2/training/`)

#### 🔹 Figure G1: Multi-Objective Loss Curves (`figure_g1_loss_curves`)
* **What it shows:** Convergence history of clean classification, TRADES consistency, feature alignment, and frequency gating losses over 120 epochs.
* **Why it matters:** Proves training stability. Shows that despite optimizing 4 competing losses, the training schedule converges smoothly without gradients exploding.

#### 🔹 Figure G2: Learning Rate Schedule (`figure_g2_learning_rate_schedule`)
* **What it shows:** Step-decay profile of the learning rate over epochs.
* **Why it matters:** Documents the schedule used during calibration and stage shifts.

#### 🔹 Figure G3: Gradient Norm Evolution (`figure_g3_gradient_norm_evolution`)
* **What it shows:** L2 norm of model gradients per epoch.
* **Why it matters:** Confirms that the training process does not suffer from vanishing or exploding gradients.

#### 🔹 Figure G4: Parameter Update Magnitude (`figure_g4_parameter_update_magnitude`)
* **What it shows:** Mean relative parameter change per epoch.
* **Why it matters:** Proves stable convergence in the final stage of curriculum training.

---

### 🩺 Group 8: Diagnostics (`figures_v2/diagnostics/`)

#### 🔹 Figure I1: Network Diagnostics Dashboard (`figure_i1_diagnostics_dashboard`)
* **What it shows:** A 2x3 panel of training diagnostics (feature variance, attention entropy, prototype norms, gate saturation, convergence steps).
* **Why it matters:** Serves as a diagnostic report. Reviewers appreciate seeing that features maintain high variance (no representation collapse) and attention entropy stabilizes.

---

### 🔍 Group 9: Explainability (`figures_v2/explainability/`)

#### 🔹 Figure J1: Grad-CAM Saliency Drift (`figure_j1_explainability_gradcam`)
* **What it shows:** Grad-CAM heatmaps for ResNet, ViT, and RHAN under clean and attacked inputs.
* **Why it matters:** Shows that under attack, ResNet's focus shifts completely away from the object. RHAN's Grad-CAM remains centered on the target, demonstrating that its classification decisions are still based on correct semantic regions.
