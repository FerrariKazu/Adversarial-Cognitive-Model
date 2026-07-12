# RHAN Publication-Ready Scientific Visualization Suite (v2)

This folder contains the complete, publication-quality visual proof suite (v2) for the **Robust Hierarchical Attention Network (RHAN)**. 

Every figure represents a quantitative or qualitative comparison demonstrating *why* and *how* the RHAN model matches human visual robustness.

## 📂 Directory Structure

For every figure in each folder, there are 5 files exported:
1. **`*_light.svg`**: Publication-ready vector format (LaTeX/paper draft compatible).
2. **`*_light.pdf`**: Publication-ready vector PDF.
3. **`*_light.png`**: High-DPI (300) clean raster image (white background).
4. **`*_dark.png`**: High-DPI (300) dark-themed presentation slide version (slate background `#0d1117`, gold titles).
5. **`*_transparent.png`**: High-DPI (300) transparent background version (suitable for custom layout templates).

---

## 📊 Figure Registry

### 🏛️ 1. Representation Geometry (`representation/`)
* **Figure A1 — UMAP Feature Space (`figure_a1_umap_feature_space`)**: Compares class cluster compactness under Clean and PGD conditions. Shows that ResNet/ViT collapse into messy overlapping clusters, while RHAN preserves cluster separation.
* **Figure A2 — t-SNE Representation Drift (`figure_a2_tsne_representation_drift`)**: Visualizes sample displacement vectors (arrows) showing that adversarial perturbations cause large boundary crosses in ResNet/ViT, but are tightly bounded in RHAN.
* **Figure A3 — Representation Drift Histogram (`figure_a3_representation_drift_histogram`)**: Quantifies clean-to-adv distance distribution ($||z_{clean} - z_{adv}||$). RHAN displays significantly narrower and lower drift.
* **Figure H1 — 2D Boundary Slices (`figure_h1_decision_boundary_slices`)**: Maps the decision boundary surrounding clean points, illustrating ResNet's fragmented and narrow boundary compared to RHAN's wide, robust boundary.
* **Figure H2 — Loss Landscape (`figure_h2_loss_landscape`)**: Renders a cross-section of the loss landscape, highlighting the sharp vulnerable spike of traditional CNNs vs. the flat robust basin of RHAN.

### 👁️ 2. Attention Analysis (`attention/`)
* **Figure B1 — Attention Overlay (`figure_b1_attention_overlay`)**: Self-attention heatmaps overlayed on clean vs. PGD-attacked inputs. Demonstrates that RHAN attention remains focused on the target object, whereas standard ViT attention shifts to random background noise.
* **Figure B2 — Attention Evolution (`figure_b2_attention_evolution`)**: Panel tracking ventral transformer attention maps over recurrent steps, showing focus sharpening over time (Step 1 $	o$ 2 $	o$ 3 $	o$ Final).
* **Figure B3 — Feedback Correction (`figure_b3_feedback_correction`)**: Multi-panel flow diagram mapping: Prediction $	o$ Prediction Error $	o$ Feedback Gate $	o$ Corrected feature map.

### 📶 3. Frequency Pathway Analysis (`frequency/`)
* **Figure C1 — Low-Frequency Gate Weight ($w_L$) vs. $arepsilon$ (`figure_c1_low_frequency_gate`)**: Curve showing wL stays high across noise levels, ensuring semantic layout transmission.
* **Figure C2 — High-Frequency Gate Weight ($w_H$) vs. $arepsilon$ (`figure_c2_high_frequency_gate`)**: Curve showing wH decreases exponentially as epsilon grows, suppressing high-frequency noise.
* **Figure C3 — Gate Weights Throughout Training (`figure_c3_gate_weights_training`)**: Epoch training history showing the adaptive gating curriculum across the three training stages.

### 📐 4. Prototype Geometry (`geometry/`)
* **Figure D1 — Spherical Prototypes (`figure_d1_spherical_prototypes`)**: 3D unit hypersphere projection of feature vectors and class prototype vectors, displaying bounded spherical separation.
* **Figure D2 — Angular Decision Boundaries (`figure_d2_angular_decision_boundaries`)**: Schematic comparing standard linear classifier unbounded boundaries with prototype angular cone boundaries.
* **Figure D3 — Angular Margin Distribution (`figure_d3_angular_margin_distribution`)**: Histogram showing the distribution of angles between feature vectors and their corresponding class prototypes.

### 📊 5. Robustness Sweeps (`evaluation/`)
* **Figure E1 — Accuracy vs. $arepsilon$ (`figure_e1_accuracy_vs_epsilon`)**: Robustness decay comparison curves under PGD sweeps. Places the RHAN Large model at the top of AI models, near the Human Visual ceiling.
* **Figure E2 — $d'$ vs. $arepsilon$ (`figure_e2_dprime_vs_epsilon`)**: Sensitivity decay curves ($d'$) compared directly to human psychophysics control.
* **Figure E3 — Robustness Threshold Comparison (`figure_e3_robustness_threshold_comparison`)**: Horizontal bar plot showing the perturbation threshold boundary $arepsilon_{thresh}$ at $d'=1.0$.
* **Figure E4 — Class Robustness Heatmap (`figure_e4_class_robustness_heatmap`)**: Detailed grid of per-class classification accuracy across epsilon levels.

### 🧠 6. Biological Analysis (`biology/`)
* **Figure F1 — Predictive Coding Convergence (`figure_f1_predictive_coding_convergence`)**: Curve of monotonically decreasing prediction error ($e^t$) over recurrent iterations.
* **Figure F2 — ACT Pondering (`figure_f2_act_pondering`)**: Image difficulty entropy vs. pondering steps used, showing adaptive processing time.
* **Figure F3 — Recurrence Step Distribution (`figure_f3_recurrence_utilization`)**: Histogram showing the recurrence steps utilized across the STL-10 dataset.

### 🔄 7. Training Dynamics (`training/`)
* **Figure G1 — Loss Curves (`figure_g1_loss_curves`)**: Loss convergence history for Clean classification, Robust TRADES consistency, Feature alignment, and Gating objectives.
* **Figure G2 — Learning Rate Schedule (`figure_g2_learning_rate_schedule`)**: Step-decay lr schedule over epochs.
* **Figure G3 — Gradient Norm Evolution (`figure_g3_gradient_norm_evolution`)**: Layer gradient norm stabilization.
* **Figure G4 — Parameter Update Magnitude (`figure_g4_parameter_update_magnitude`)**: Average relative weight updates ($||\Delta W||/||W||$) per epoch.

### 🩺 8. Network Diagnostics (`diagnostics/`)
* **Figure I1 — Diagnostics Dashboard (`figure_i1_diagnostics_dashboard`)**: 2x3 panel of training health diagnostics (variance, entropy, prototype norms, gate saturation).

### 🔍 9. Explainability (`explainability/`)
* **Figure J1 — Grad-CAM Comparison (`figure_j1_explainability_gradcam`)**: Comparison of saliency maps under Clean and PGD-attacked inputs between ResNet-18, ViT-Small, and RHAN.
