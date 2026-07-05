# RHAN Architecture & Flow Visualization Suite (v1)

This folder contains the baseline publication-quality visualizations for the **Robust Hierarchical Attention Network (RHAN)**. These figures focus on structural topology, computational graphs, and stage-wise training flow.

---

## 📂 File Formats
Each figure is exported in three formats:
* **`.svg`**: Infinitely scalable vector graphics (best for digital papers and zoom-in).
* **`.pdf`**: High-quality vector PDF (perfect for direct LaTeX embedding without resolution loss).
* **`.png`**: Raster image rendered at 300+ DPI (best for presentations and web pages).

---

## 🖼️ Figure Catalog

### 🏛️ 1. Full RHAN Architecture
* **File Paths:**
  - `architecture/figure_1_rhan_architecture.svg`
  - `architecture/figure_1_rhan_architecture.pdf`
  - `architecture/figure_1_rhan_architecture.png`
* **Visual Description:** A multi-panel layout showing the volumetric 3D traced layer blocks (top) generated via VisualTorch and a corresponding detailed information flow diagram (bottom) with labeled blocks and connecting data pathways.
* **Core Concept:** Illustrates how clean or perturbed images pass through local feature extraction, split into parallel Ventral (semantics) and Dorsal (geometry) streams, integrate in the predictive coding module, and map to angular prototype classes. It shows the active top-down recurrent feedback pathway returning predictions to modulate the input stem.
* **Quantitative Details:** Mapped dimensions include: input tensor ($3 \times 96 \times 96$), WideSEConvStem output ($64 \times 96 \times 96$), PatchTokenizer output ($128 \times 48 \times 48$), Dual-stream embeddings ($768$), and class logits ($10$). Total parameters: **55.62 Million**.
* **Paper Location Recommendation:** Place on **Page 1 (Teaser Figure)** to grab reviewer attention immediately.

### 🧱 2. Layer-by-Layer Architecture Stack
* **File Paths:**
  - `architecture/figure_2_layer_by_layer.svg`
  - `architecture/figure_2_layer_by_layer.pdf`
  - `architecture/figure_2_layer_by_layer.png`
* **Visual Description:** An isometric block projection showing every sub-layer stack in the network, color-coded by layer type (Convolutions in pink, Normalizations in orange, Linear layers in purple, and Transformers in pink/cyan).
* **Core Concept:** Provides structural transparency. It traces the actual instantiated PyTorch modules (`WideSEConvStem`, `PatchTokeniserLarge`, and the dual stream transformers) to display structural depth and channel scaling.
* **Paper Location Recommendation:** Place in **Section 3: Methodology (Architecture Details)** or in the Appendix.

### 🕸️ 3. Computational Trace Graph
* **File Paths:**
  - `architecture/figure_3_computational_graph.svg`
  - `architecture/figure_3_computational_graph.pdf`
  - `architecture/figure_3_computational_graph.png`
* **Visual Description:** A high-resolution node connection graph traced directly from the model's JIT execution graph.
* **Core Concept:** Shows the operational nodes and residual routing pathways (skip connections, transformer self-attention blocks, and the feedback loop) showing how gradients and data flow through the parallel streams.
* **Paper Location Recommendation:** Place in the **Appendix (Model Diagnostics)**.

### 🌳 4. Module Hierarchy tree
* **File Paths:**
  - `architecture/figure_4_module_hierarchy.svg`
  - `architecture/figure_4_module_hierarchy.pdf`
  - `architecture/figure_4_module_hierarchy.png`
* **Visual Description:** A clean hierarchical tree chart detailing the structural nesting of sub-modules within the main `RHANLargeSTL10` class.
* **Core Concept:** Outlines how child modules inherit parameters and split computation. Useful for explaining module boundaries and weight sharing.
* **Quantitative Details:** Annotates parameters per module: WideSEConvStem (1.45M), PatchTokenizer (0.12M), Ventral Transformer (26.85M), Dorsal Transformer (26.85M), Predictive Gating (0.15M), Spherical Head (0.20M).
* **Paper Location Recommendation:** Place in **Section 3: Architecture Implementation**.

### 🥞 5. Parameter Distribution Treemap
* **File Paths:**
  - `geometry/figure_5_parameter_distribution.svg`
  - `geometry/figure_5_parameter_distribution.pdf`
  - `geometry/figure_5_parameter_distribution.png`
* **Visual Description:** A proportional block treemap representing parameter capacity allocation.
* **Core Concept:** Proves that the computational power is heavily concentrated in the dual stream transformers (representing 96.6% of the network capacity), matching biological visual systems where recurrent cortical pathways hold the bulk of processing cells.
* **Paper Location Recommendation:** Place in **Section 4: Experiments (Model Efficiency)**.

### ➡️ 6. Activation Flow and Dimensions
* **File Paths:**
  - `architecture/figure_6_activation_flow.svg`
  - `architecture/figure_6_activation_flow.pdf`
  - `architecture/figure_6_activation_flow.png`
* **Visual Description:** A horizontal sequence of dimension boxes showing batch and feature dimensions at core junctions.
* **Core Concept:** Traces the transformation of spatial dimensions (height/width) into token sequences and flat embedding vectors.
* **Paper Location Recommendation:** Place in **Methodology (Data Flow)**.

### 📉 7. Tensor Shape Evolution Flowchart
* **File Paths:**
  - `architecture/figure_7_tensor_shape_evolution.svg`
  - `architecture/figure_7_tensor_shape_evolution.pdf`
  - `architecture/figure_7_tensor_shape_evolution.png`
* **Visual Description:** A vertical flowchart representing step-by-step tensor operations.
* **Core Concept:** Details the mathematical transformations (e.g., unfolding patches, adding CLS tokens, and cosine classification projections).
* **Paper Location Recommendation:** Place in **Methodology (Implementation Details)**.

### 💾 8. Memory Footprint Profile
* **File Paths:**
  - `evaluation/figure_8_memory_consumption.svg`
  - `evaluation/figure_8_memory_consumption.pdf`
  - `evaluation/figure_8_memory_consumption.png`
* **Visual Description:** Double bar chart comparing VRAM usage with vs. without gradient checkpointing across memory categories.
* **Core Concept:** Proves the necessity of gradient checkpointing inside the deep dual-stream transformers during curriculum training to prevent GPU Out-of-Memory (OOM) failures.
* **Quantitative Details:** Shows peak activation VRAM drops from **5800 MB** to **1200 MB** (an 80% reduction) with checkpointing enabled.
* **Paper Location Recommendation:** Place in **Section 4: Computational Cost & VRAM Optimization**.

### ⚡ 9. GFLOPs Compute Allocation
* **File Paths:**
  - `evaluation/figure_9_flops_distribution.svg`
  - `evaluation/figure_9_flops_distribution.pdf`
  - `evaluation/figure_9_flops_distribution.png`
* **Visual Description:** Side-by-side pie chart and horizontal bar plot representing forward pass computational complexity in GFLOPs.
* **Core Concept:** Quantifies the floating-point operations. The Dual Transformer takes up **85.5%** (42.8 GFLOPs) of the forward workload, proving it represents the core processing engine.
* **Paper Location Recommendation:** Place in **Section 4: Computational Performance**.

### 🧠 10. Biological Correspondence Mapping
* **File Paths:**
  - `biological/figure_10_biological_mapping.svg`
  - `biological/figure_10_biological_mapping.pdf`
  - `biological/figure_10_biological_mapping.png`
* **Visual Description:** An aligned 3D block comparison showing the human visual pathway (left) linked via cross-connectors to corresponding RHAN computational modules (right).
* **Core Concept:** Maps cortical layers: Retina $\to$ Input, LGN $\to$ ConvStem, V1/V2 $\to$ Tokenizer, Ventral/Dorsal streams $\to$ Parallel transformers, IT Cortex $\to$ Prototype Head.
* **Paper Location Recommendation:** Place in **Section 1: Introduction or Section 2: Related Work**.

### 🔁 11. Predictive Coding Loop
* **File Paths:**
  - `training/figure_11_predictive_coding_loop.svg`
  - `training/figure_11_predictive_coding_loop.pdf`
  - `training/figure_11_predictive_coding_loop.png`
* **Visual Description:** Side-by-side panel showing target layers (left) and the feedback math flow (right).
* **Core Concept:** Explains the recurrence update formula: $f^{t+1} = f_{stem} + g(e^t) \odot e^t$. Shows how prediction error and gating interact to filter out adversarial perturbers.
* **Paper Location Recommendation:** Place in **Section 3: Methodology (Recurrent Predictive Coding)**.

### 🛡️ 12. SAIL Invariance Pipeline
* **File Paths:**
  - `training/figure_12_sail_pipeline.svg`
  - `training/figure_12_sail_pipeline.pdf`
  - `training/figure_12_sail_pipeline.png`
* **Visual Description:** Flowchart mapping Clean and Adversarial images through encoders to optimize InfoNCE loss.
* **Core Concept:** Explains how Self-supervised Adversarial Invariance Learning (SAIL) aligns perturbed representations in latent space.
* **Paper Location Recommendation:** Place in **Section 3: Methodology (SAIL pretraining)**.

### 🎞️ 13. TDV Spatiotemporal Pipeline
* **File Paths:**
  - `training/figure_13_tdv_pipeline.svg`
  - `training/figure_13_tdv_pipeline.pdf`
  - `training/figure_13_tdv_pipeline.png`
* **Visual Description:** Video temporal flowchart tracking Frame $t$ and Frame $t+1$ to compute motion vectors and check temporal consistency.
* **Core Concept:** Outlines Temporal Difference Vision (TDV) consistency learning, enforcing representation stability across natural video changes.
* **Paper Location Recommendation:** Place in **Section 3: Methodology (Temporal Consistency)**.

### 🌐 14. Radial Chart Overview
* **File Paths:**
  - `training/figure_14_rhan_ecosystem.svg`
  - `training/figure_14_rhan_ecosystem.pdf`
  - `training/figure_14_rhan_ecosystem.png`
* **Visual Description:** A radial circular diagram linking the core RHAN backbone to its 8 surrounding algorithmic mechanisms.
* **Core Concept:** Serves as a comprehensive conceptual overview of how SAIL, TDV, CLIP supervision, CORnet cortical teachers, TRADES, and Prototype heads integrate into the ecosystem.
* **Paper Location Recommendation:** Place in the **Methodology Overview**.

### 🗓️ 15. Three-Stage Training Pipeline Timeline
* **File Paths:**
  - `training/figure_15_training_pipeline.svg`
  - `training/figure_15_training_pipeline.pdf`
  - `training/figure_15_training_pipeline.png`
* **Visual Description:** Horizontal timeline flowchart indicating Stage 1 (SAIL), Stage 2 (TRADES), and Stage 3 (TDV) epochs.
* **Core Concept:** Summarizes the training schedule from unsupervised representation building to adversarial alignment and video consistency.
* **Paper Location Recommendation:** Place in **Section 4: Experiments (Training Details)**.

### 📊 16. Performance Evaluation Dashboard
* **File Paths:**
  - `evaluation/figure_16_evaluation_dashboard.svg`
  - `evaluation/figure_16_evaluation_dashboard.pdf`
  - `evaluation/figure_16_evaluation_dashboard.png`
* **Visual Description:** A 2x2 dashboard containing: clean vs. robust accuracy curves (top-left), sensitivity $d'$ decay (top-right), class-wise AutoAttack scores (bottom-left), and a summary table (bottom-right).
* **Core Concept:** Compiles all key benchmark results (clean/robust accuracies, VRAM, speed, and sensitivity indicators) to give reviewers a complete metric overview in a single figure.
* **Paper Location Recommendation:** Place in **Section 4: Experimental Results (Main Benchmark)**.

### 📉 Auxiliary 1: Loss Landscape cross-section
* **File Paths:**
  - `losses/figure_loss_landscape.svg`
  - `losses/figure_loss_landscape.pdf`
  - `losses/figure_loss_landscape.png`
* **Visual Description:** Plot comparing standard CNN sharp loss spikes with RHAN's flat, wide stable loss basin.
* **Core Concept:** Explains why RHAN generalization is stable under perturbations (gradients are smooth and margins are flat).
* **Paper Location Recommendation:** Place in **Section 5: Analysis (Loss Landscapes)**.

### 📐 Auxiliary 2: Spherical Prototype Classifier
* **File Paths:**
  - `geometry/figure_spherical_prototype.svg`
  - `geometry/figure_spherical_prototype.pdf`
  - `geometry/figure_spherical_prototype.png`
* **Visual Description:** Vector projection showing normalized feature vectors $z$ and prototype vectors $p_c$ on a circle.
* **Core Concept:** Explains the angular classifier formula, showing how normalizing inputs and prototype weights eliminates scale-based gradient exploitation.
* **Paper Location Recommendation:** Place in **Methodology (Classification Head)**.
