#!/usr/bin/env python3
"""
Generate an exhaustive, highly-detailed, non-generic repository map for README.md.
"""
import subprocess
import os

raw_files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "HEAD"], text=True).splitlines()

class Node:
    def __init__(self, name, is_file=False, path=""):
        self.name = name
        self.is_file = is_file
        self.path = path
        self.children = {}

root = Node("")
for f in raw_files:
    parts = f.split("/")
    curr = root
    for i, p in enumerate(parts):
        is_last = (i == len(parts) - 1)
        subpath = "/".join(parts[:i+1])
        if p not in curr.children:
            curr.children[p] = Node(p, is_file=is_last, path=subpath)
        curr = curr.children[p]

# Comprehensive directory dictionary
dir_comments = {
    "gen2_foundation": "Gen-2 foundation package: DINOv2-S trunk, corrected TRADES-PGD4, GRU belief update",
    "gen2_foundation/tests": "Unit tests for Gen-2 foundation modules (EMA, EOT, Gist parity, UpdateNet v2)",
    "noesis_vision": "NOESIS cognitive vision engine implementing active inference and the biological perceptual loop",
    "noesis_vision/RHAN_NXA": "RHAN-NXA architecture documentation, master plans, and locked contracts",
    "noesis_vision/RHAN_NXA/docs": "32-chapter exhaustive technical specification handbook for RHAN-NXA",
    "noesis_vision/core": "Core cognitive orchestrator: configuration schemas, atomic checkpointing, provenance manifests",
    "noesis_vision/models": "Neural backbones: CompactViT, differentiable foveation grid sampling, tied recurrence",
    "noesis_vision/gaze": "Adaptive Information Seeking (AIS-v2) gaze policy, candidate sampling, and center bias",
    "noesis_vision/beliefs": "Latent belief state representations (VectorBeliefState) and drift regularization",
    "noesis_vision/predictive_coding": "Top-down feature prediction, precision weighting, and spatial error pooling",
    "noesis_vision/uncertainty": "Dirichlet evidential deep learning (EDL) heads for epistemic uncertainty quantification",
    "training": "Training drivers, adversarial curricula (TRADES/PGD), health monitors, and cloud sync",
    "evaluation": "Hardened evaluation harness: clean/robust accuracy, Geirhos cue-conflict, compactness accounting",
    "scripts": "Operational tooling: ImageNet-100 data prep, promotion gates, sweep runners, report generators",
    "tests": "Comprehensive test suite (406+ assertions: invariants, gradient flow, regression, numerical stability)",
    "docs": "Technical manuals, Stage 3 preregistrations, research records, and compiled handbooks",
    "docs/rhan_nxa": "RHAN-NXA unified handbook, QA reports, and figure gallery",
    "docs/rhan_nxa/book": "RHAN-NXA comprehensive textbook chapters (00-26) detailing the complete architecture",
    "docs/rhan_nxa/appendices": "Technical appendices covering tensor dimensions, ABC interfaces, and port tables",
    "docs/rhan_nxa/figure_gallery": "Standalone HTML figure gallery and high-res diagram assets",
    "docs/rhan_nxa/figures": "Vector SVG publication diagrams and architecture schematics",
    "docs/rhan_nxa/pdf_qa_pages": "Rendered visual QA inspection pages for compiled handbook",
    "docs/historical": "Archived stage reports, early proposal drafts, and pre-registrations",
    "docs/research": "Research references, citation audits, and empirical seed tables",
    "report": "Master experimental results tables, empirical JSON databases, and scientific publication reports",
    "report/assets": "Plot assets and visual diagrams for empirical reports",
    "report/stage3_D": "Stage 3-D Pareto trade-off analysis, figures, and provenance audits",
    "report/stage3_D/figures": "Visual plots comparing Stage 3-D Pareto trade-offs and seed variance",
    "report/stage4_E1": "Stage 4-E1 lens introspection, pre-registration scripts, and runners",
    "report/sbr_feasibility": "Spatial belief representation (SBR) slot mask visual feasibility checks",
    "report/lens_e1_analysis": "Comprehensive audit report for Lens E1 introspection experiment",
    "cloud": "Cloud deployment configurations and Slurm cluster specs",
    "cloud/canonical": "Canonical production cloud dispatchers for Kaggle",
    "cloud/gen0": "Historical Gen-0 cloud dispatchers for Kaggle and Colab",
    "cloud/gen1": "Generation-1 cloud dispatchers and shell wrappers",
    "cloud/misc": "Historical Colab/Kaggle setup scripts and experimental runners",
    "diagnostics": "Runtime profiling artifacts and isolation test scripts",
    "diagnostics/diagnosis_artifacts": "Diagnostic scripts isolating autograd, AMP, and loss landscapes",
    "diagnostics/diagnosis_artifacts/out": "JSON diagnostic reports on dataset forensics and overfit checks",
    "diagnostics/diagnosis_artifacts/hf_rolling_adv": "Adversarial roadmap diagnostics synced from rolling checkpoint repo",
    "checkpoints": "Model weights directory (tracked JSON manifests, local cache, and symlinks)",
    "config": "Central YAML configuration files for models, training, and attack bounds",
    "runs": "Experiment output runs containing TensorBoard event logs and metrics",
    "archive": "Archived legacy stages and previous experimental pipelines",
    "archive/gen-1-cifar12": "Archived May 2026 CIFAR-10 7-model + human psychophysics comparison",
    "archive/gen-1-cifar12/phase1_training": "CIFAR-10 model training definitions (ResNet, ViT, EfficientNet, etc.)",
    "archive/gen0": "Archived Gen-0 (RHAN-Next) codebase and world model prototypes",
    "archive/gen0/rhan_core": "Gen-0 core package: ablation, beliefs, gaze, precision, predictive coding",
    "archive/gen0/rhan_core/ablation": "Ablation modules isolating recurrence, gating, and foveation",
    "archive/gen0/rhan_core/artifacts": "Reference evaluation outputs for Gen-0 models",
    "archive/gen0/rhan_core/beliefs": "Gen-0 belief state implementations and experimental dynamics",
    "archive/gen0/rhan_core/beliefs/experimental": "Experimental continuous-discrete belief representations",
    "archive/gen0/rhan_core/config": "Configuration dataclasses for Gen-0 models",
    "archive/gen0/rhan_core/gaze": "Gen-0 fixation sampling policies and spatial grids",
    "archive/gen0/rhan_core/lens": "Introspection lens extracting interpretable state tokens",
    "archive/gen0/rhan_core/optim": "Optimizer parameter group constructors for Gen-0",
    "archive/gen0/rhan_core/precision": "Dynamic precision weighting mechanisms for Gen-0",
    "archive/gen0/rhan_core/predictive_coding": "Gen-0 top-down error computation and prediction heads",
    "archive/gen0/rhan_core/world_model": "World state transition models predicting future states",
    "archive/gen3-human": "Archived human psychophysics baseline experiments and trial records",
    "archive/gen3-human/phase3_human_study": "Human trial logs, response templates, and participant data",
    "archive/gen4-analysis": "Archived divergence analysis curves, heatmaps, and attention figures",
    "archive/gen4-analysis/phase4_analysis": "Cross-model divergence scripts and representation similarity metrics",
    "archive/gen4-analysis/phase4_analysis/figures": "Attention and Grad-CAM visualizations across 7 architectures",
    "archive/gen5-sdt": "Archived Signal Detection Theory calculations and ROC curves",
    "archive/gen5-sdt/phase5_sdt": "Signal Detection Theory (SDT) sensitivity d-prime and criterion c calculators",
    "archive/historical-report": "Prior LaTeX manuscripts, draft papers, and workshop submissions",
    "archive/historical-report/RHANv12": "Draft manuscripts and LaTeX figures for RHAN v12",
    "archive/historical-report/RHANv12/scripts": "Script utilities for generating figures for RHAN v12 paper",
    "archive/legacy-evals": "Archived evaluation benchmarks, attack scripts, and verification sweeps",
    "archive/legacy-evals/cognitive_vision_lab": "Streamlit cognitive vision laboratory interactive app",
    "archive/pkg-rhan-math": "Mathematical derivations, proofs, and vector calculus reports",
    "archive/pkg-rhan-math/rhan_math": "LaTeX source files for mathematical formulation paper",
    "archive/pkg-tier1": "Archived Tier-1 multi-seed reproducibility benchmark suites",
    "archive/pkg-tier1/tier1": "Tier-1 evaluation harness and multi-seed result tables",
    "archive/rhan-v1-v7": "Early RHAN recurrent prototypes (v1 through v7)",
    "archive/stl10-scaleup": "STL-10 self-supervised pretraining and pseudo-labeling experiments",
    "archive/working-scratch": "Archived scratch scripts and prototype renderers",
    "figures": "Publication figures v1 (architectures, loss landscapes, information flows)",
    "figures/architecture": "Architectural diagrams for foveation, gist encoder, and predictive loop",
    "figures/biological": "Comparative schematics illustrating primate ventral stream parallels",
    "figures/evaluation": "Robustness curves, attack comparison bar plots, and error rate tables",
    "figures/explainability": "Grad-CAM saliency maps and attention rollouts under adversarial attacks",
    "figures/geometry": "Latent space geometry visualizations and decision boundary plots",
    "figures/losses": "Loss surface contour plots and convergence trajectory curves",
    "figures/training": "Training loss trajectories, learning rate curves, and gradient norm plots",
    "figures_v2": "Publication figures v2 (attention maps, biology, geometry, frequency analysis)",
    "figures_v3": "Publication figures v3 (multi-theme light/dark publication-ready SVG/PNG)",
    ".verify_samples": "Ground-truth reference validation images across 10 object classes",
}

# Exhaustive dictionary of specific file descriptions
file_comments = {
    # Root
    ".gitattributes": "Git line endings and binary diff configuration",
    ".gitignore": "Ignored patterns (virtual environments, PyTorch weights, cache)",
    ".mailmap": "Developer name and email address canonicalization",
    "CONTRIBUTING.md": "Contributor instructions, branch policy, and pull request workflow",
    "README.md": "Master repository overview, empirical benchmark findings, and exhaustive repo map",
    "RHANarch.pdf": "Architectural vector diagram of Recurrent Hierarchical Attention Network",
    "ROADMAP.md": "Multi-generation research roadmap (Gen-1 through Gen-4)",
    "T4x2.ipynb": "Dual-T4 multi-GPU training and experimentation notebook",
    "TRAINING_HEALTH_REPORT.md": "Diagnostic criteria for loss stability, gradient norms, and NaN guards",
    "app.py": "Interactive web dashboard for visualizing model predictions and attacks",
    "confusion_matrix.png": "Confusion matrix visualization for baseline ResNet-18",
    "confusion_matrix_efficientnet.png": "Confusion matrix visualization for EfficientNet-B0",
    "confusion_matrix_vit.png": "Confusion matrix visualization for Vision Transformer (ViT-Small)",
    "requirements.txt": "Python dependencies (PyTorch, torchvision, timm, torchattacks, scipy)",

    # gen2_foundation
    "gen2_foundation/__init__.py": "Package exports and public API for Gen-2 foundation stack",
    "gen2_foundation/flags.py": "Experiment registry feature flags gating independent architectural modules",
    "gen2_foundation/gist.py": "Global gist encoder: DINOv2-S patch projection, antialiased pos-embed, and CLS prior",
    "gen2_foundation/precision.py": "Calculates dynamic observation precision weights (gamma) from belief uncertainty",
    "gen2_foundation/recipe.py": "Builds multi-group AdamW optimizer parameter groups with layer-wise LR decay",
    "gen2_foundation/ema.py": "Tracks exponential moving average (EMA) shadow model weights to mitigate overfitting",
    "gen2_foundation/eot.py": "Expectation Over Transformation (EOT-PGD) attack for evaluating stochastic defenses",
    "gen2_foundation/update_net_v2.py": "Bounded recurrent belief update network (GRU) integrating error and glimpse features",
    "gen2_foundation/AUDIT.md": "Package forensic audit report verifying 52 passing tests and zero gradient leakage",
    "gen2_foundation/EXPERIMENT_REGISTRY.md": "Formal preregistered ledger of experimental arms, baselines, and gate criteria",
    "gen2_foundation/tests/__init__.py": "Unit test package initialization for gen2_foundation",
    "gen2_foundation/tests/test_ema.py": "Verifies EMA shadow weight updates, decay schedules, and eval weight swapping",
    "gen2_foundation/tests/test_eot.py": "Tests EOT-PGD loss gradient averaging across spatial transformation distributions",
    "gen2_foundation/tests/test_flags_gating.py": "Ensures feature flags cleanly toggle modules without dead parameter residual leaks",
    "gen2_foundation/tests/test_gist_parity.py": "Verifies that Gist encoder outputs bit-exact CLS representations at gate-0",
    "gen2_foundation/tests/test_precision.py": "Tests precision weight output dynamic range and positive boundedness",
    "gen2_foundation/tests/test_recipe.py": "Verifies optimizer parameter group separation and learning rate allocations",
    "gen2_foundation/tests/test_update_net_v2.py": "Tests UpdateNet v2 gradient flow, hidden state dimensions, and residual steps",

    # noesis_vision/RHAN_NXA
    "noesis_vision/RHAN_NXA/MASTER_PLAN.md": "Master plan defining locked contracts and experimental roadmap for Gen-1",
    "noesis_vision/RHAN_NXA/docs/00_README.md": "Index and architectural overview of the 31-chapter RHAN-NXA specification handbook",
    "noesis_vision/RHAN_NXA/docs/01_What_Is_RHAN_NXA.md": "Theoretical definition of the active inference perceptual loop",
    "noesis_vision/RHAN_NXA/docs/02_Why_RHAN_NXA_Exists.md": "Scientific rationale: why passive feedforward vision fails under adversarial attacks",
    "noesis_vision/RHAN_NXA/docs/03_Perception_As_Investigation.md": "Formulating visual recognition as sequential hypothesis testing and evidence seeking",
    "noesis_vision/RHAN_NXA/docs/04_Belief_State.md": "Mathematical formulation of internal belief representation B_t = (z_t, S_t)",
    "noesis_vision/RHAN_NXA/docs/05_z_State.md": "Continuous latent feature vector dynamics representing the semantic hypothesis space",
    "noesis_vision/RHAN_NXA/docs/06_Structure_State.md": "Structured symbolic representation of object parts and spatial relationships",
    "noesis_vision/RHAN_NXA/docs/07_Uncertainty.md": "Epistemic vs aleatoric uncertainty quantification using Dirichlet distributions",
    "noesis_vision/RHAN_NXA/docs/08_Prediction_Error.md": "Top-down generative feature prediction and error residual computation",
    "noesis_vision/RHAN_NXA/docs/09_Recurrence.md": "Recurrent refinement dynamics: state updating vs multiview aggregation",
    "noesis_vision/RHAN_NXA/docs/10_AIS_v2.md": "Adaptive Information Seeking: expected uncertainty reduction gaze policy",
    "noesis_vision/RHAN_NXA/docs/11_Complete_Perceptual_Loop.md": "Step-by-step tensor tracing through the full perceptual cycle",
    "noesis_vision/RHAN_NXA/docs/12_Architecture_Data_Flow.md": "Detailed module-by-module data flow and interface boundaries",
    "noesis_vision/RHAN_NXA/docs/13_Gradient_Flow.md": "Backpropagation paths, detached copies, and gradient isolation boundaries",
    "noesis_vision/RHAN_NXA/docs/14_Tensor_Shape_Reference.md": "Complete tensor dimension reference table across all operational steps",
    "noesis_vision/RHAN_NXA/docs/15_Status_And_Decision_System.md": "Experimental confidence classification (VERIFIED, HYPOTHESIS, REJECTED)",
    "noesis_vision/RHAN_NXA/docs/16_Gen0_Evidence_And_Confounds.md": "Analysis of Gen-0 empirical results and confounding factors",
    "noesis_vision/RHAN_NXA/docs/17_Common_Misunderstandings.md": "Conceptual clarifications regarding foveation, recurrence, and precision",
    "noesis_vision/RHAN_NXA/docs/18_Glossary.md": "Formal mathematical definitions and project vocabulary",
    "noesis_vision/RHAN_NXA/docs/19_Decision_Records.md": "Architecture Decision Records (ADRs) establishing locked design choices",
    "noesis_vision/RHAN_NXA/docs/20_Scope_Boundaries.md": "Explicit definitions of in-scope vs out-of-scope research questions",
    "noesis_vision/RHAN_NXA/docs/21_L_stab_Stability.md": "Numerical stability constraints and belief state variance bounds",
    "noesis_vision/RHAN_NXA/docs/22_Memory.md": "Memory taxonomy (M0 weights, M1 working state, M2 glimpse buffer, M3 spatial bank)",
    "noesis_vision/RHAN_NXA/docs/23_V1_Frontend.md": "Primary visual cortex (V1) multi-scale Gabor filter frontend ablation",
    "noesis_vision/RHAN_NXA/docs/24_Training_Phase_DAG.md": "Directed Acyclic Graph defining phase dependencies and promotion gates",
    "noesis_vision/RHAN_NXA/docs/25_Ablation_Matrix.md": "Systematic ablation matrix isolating each cognitive component",
    "noesis_vision/RHAN_NXA/docs/26_Agent_Organization.md": "Decoupled multi-agent architecture and interface contracts",
    "noesis_vision/RHAN_NXA/docs/27_Infrastructure_Port_Table.md": "Port mapping table between legacy codebase and canonical modules",
    "noesis_vision/RHAN_NXA/docs/28_Gates_and_Compute_Accounting.md": "Strict promotion gates (Gate B budget, Gate S search, Gate A robustness)",
    "noesis_vision/RHAN_NXA/docs/29_Literature_Classification.md": "Positioning relative to neuroscience and computer vision literature",
    "noesis_vision/RHAN_NXA/docs/30_Agent0_Interface_Contract.md": "Root interface contract defining inputs, outputs, and invariants",
    "noesis_vision/RHAN_NXA/docs/Proposed_Plan.md": "Multi-stage master roadmap for Gen-2 and Gen-3 development",

    # noesis_vision/core
    "noesis_vision/core/checkpoint.py": "Atomic checkpoint serialization, model state dict extraction, and integrity checking",
    "noesis_vision/core/consistency_assert.py": "Runtime invariant assertions for tensor dimensions and state continuity",
    "noesis_vision/core/experiment_registry.py": "Registry tracking active experiment flags, seeds, and component variants",
    "noesis_vision/core/multi_group_optimizer.py": "Multi-parameter-group AdamW optimizer configuring layer-wise learning rates",
    "noesis_vision/core/provenance.py": "Captures git commit SHA, dependency hashes, and environment metadata",
    "noesis_vision/core/schema.py": "Data class schemas for configurations, hyperparameters, and experiment records",
    "noesis_vision/core/seed_management.py": "Deterministic RNG seeding across PyTorch, NumPy, and Python standard library",
    "noesis_vision/core/agent_contract_template.md": "Interface specification contract template between decoupled agents",
    "noesis_vision/core/dependency_graph.md": "Mathematical dependency graph between perceptual modules",

    # noesis_vision/models
    "noesis_vision/models/backbone.py": "Vision transformer backbone wrapper with positional embedding interpolation",
    "noesis_vision/models/foveation.py": "Differentiable affine grid transformation and spatial foveated glimpse extractor",
    "noesis_vision/models/recurrent_block.py": "Recurrent processing block tying latent state across temporal glimpse steps",

    # noesis_vision/gaze
    "noesis_vision/gaze/ais_v2_policy.py": "Adaptive Information Seeking (AIS) policy ranking candidate gaze fixations",
    "noesis_vision/gaze/candidate_sampler.py": "Spatial candidate sampler generating prospective fixation bounding coordinates",
    "noesis_vision/gaze/center_bias.py": "Foveal center-bias measurement and spatial entropy calculation",
    "noesis_vision/gaze/gaze_state.py": "Canonical data structure encapsulating fixation history and trajectory state",

    # noesis_vision/beliefs
    "noesis_vision/beliefs/drift.py": "Belief state drift tracking measuring divergence across recurrent steps",
    "noesis_vision/beliefs/factory.py": "Factory constructor instantiating concrete belief state representations",
    "noesis_vision/beliefs/interfaces.py": "Abstract Base Class defining the belief state interface contract",
    "noesis_vision/beliefs/vector_belief.py": "Dense continuous vector representation of the agent's internal world belief",

    # noesis_vision/predictive_coding
    "noesis_vision/predictive_coding/ema_predictor.py": "Exponential moving average target encoder providing stable prediction targets",
    "noesis_vision/predictive_coding/glimpse_predictor.py": "Top-down generative predictor forecasting incoming glimpse features",
    "noesis_vision/predictive_coding/interfaces.py": "Abstract Base Class defining predictive coding predictor/error interfaces",
    "noesis_vision/predictive_coding/precision.py": "Spatial and channel-wise precision weighting of incoming sensory error",
    "noesis_vision/predictive_coding/spatial_error_pool.py": "Spatial pooling layer aggregating localized prediction error residuals",
    "noesis_vision/predictive_coding/update_net.py": "Update network computing belief innovation updates from prediction error",

    # noesis_vision/uncertainty
    "noesis_vision/uncertainty/evidential_head.py": "Dirichlet Evidential Deep Learning (EDL) head estimating epistemic uncertainty",

    # training
    "training/HF_PERSISTENCE_README.md": "Documentation for HuggingFace automated backup and recovery",
    "training/adv_curriculum.py": "Adversarial curriculum scheduler controlling epsilon ramp and step sizes",
    "training/adv_curriculum_freeze.py": "Curriculum state freezing and deterministic trajectory reproduction",
    "training/epoch_summary.py": "Formats and computes per-epoch training metrics and loss breakdowns",
    "training/hf_persistence.py": "HuggingFace automated checkpoint sync, resume, and model upload handlers",
    "training/measure_training_health.py": "Monitors gradient norms, loss landscape sharpness, and numerical stability",
    "training/production_halt.py": "Safety guard halting training upon NaN detection or diverging loss",
    "training/run_report.py": "Generates human-readable experimental reports at training completion",
    "training/run_report_jsonl.py": "Streams structured telemetry metrics to JSONL log files per epoch",
    "training/stage2_pipeline.py": "Multi-phase pipeline state machine coordinating ladder progressions",
    "training/stage_state_machine.py": "Manages discrete stage transitions and gate criteria checks",
    "training/summarize_run.py": "Generates summary tables and comparative charts across completed runs",
    "training/sync_log.py": "Utility syncing local training event logs with remote telemetry sinks",
    "training/sync_utils.py": "Low-level retry logic and file locking helpers for cloud synchronization",
    "training/test_hf_persistence.py": "Self-test verifying HuggingFace upload, download, and resume mechanisms",
    "training/train_generation1_foundation.py": "Training pipeline driver for the 6-phase Gen-1 foundation ladder",
    "training/verify_hf_checkpoint_chain.py": "Validates cryptographic integrity and lineage of checkpoint chains",

    # evaluation
    "evaluation/clean_and_robust.py": "Standardized evaluation runner for clean and Linf PGD robustness",
    "evaluation/compactness_report.py": "Profiles model parameter counts, FLOPs, memory, and inference latency",
    "evaluation/geirhos_loader.py": "Dataloader for the Geirhos cue-conflict dataset (shape vs. texture bias)",
    "evaluation/imagenet100_loader.py": "ImageNet-100 dataset loader with normalization, transforms, and caching",
    "evaluation/imagenet_c_loader.py": "ImageNet-C corruption benchmark loader across 15 synthetic noise types",
    "evaluation/shape_texture_bias.py": "Signal Detection Theory (d') script for shape vs texture classification",

    # scripts
    "scripts/build_noesis_pdf.py": "Compiles NOESIS markdown architecture documentation into PDF format",
    "scripts/build_rhan_nx_report.py": "Generates formal experimental summary report in PDF format",
    "scripts/build_rhan_nxa_handbook.py": "Assembles the complete RHAN-NXA Technical Handbook from book chapters",
    "scripts/comparator_registry.py": "Registry and baseline comparator loader for paired hypothesis tests",
    "scripts/consistency_assert.py": "Assertions verifying architectural and tensor dimension consistency",
    "scripts/eval_ais_v2_gate.py": "Gating assertion evaluating AIS-v2 gaze policy against random/fixed gaze",
    "scripts/freeze_run_manifest.py": "Generates cryptographic SHA-256 manifest of model weights and configs",
    "scripts/generate_full_sweep.py": "Automated script running full hyperparameter and attack evaluation sweeps",
    "scripts/measure_group_dw.py": "Measures parameter update magnitudes (||dW||) across optimizer groups",
    "scripts/merge_stage1_seed_extension.py": "Aggregates multi-seed stage 1 evaluation outputs into master dataset",
    "scripts/prepare_imagenet100.py": "Automated download, unpack, and validation for ImageNet-100 dataset",
    "scripts/reset_gen1_ladder.py": "Utility to reset training state machine to restart foundation ladder",
    "scripts/sbr0_gate.py": "Automated gate checker validating Stage SBR-0 criteria before promotion",
    "scripts/stage_state_machine.py": "State machine script managing transitions between training stages",
    "scripts/verify_run_complete.py": "Verifies that all epochs and validation passes finished successfully",

    # tests
    "tests/conftest.py": "Global pytest fixtures, synthetic batch generators, and device initialization",
    "tests/test_ablation_matrix.py": "Verifies that every cell in the 9-point ablation matrix executes without error",
    "tests/test_adv_curriculum.py": "Asserts TRADES epsilon schedule bounds, step-size alpha ratios, and ramp timing",
    "tests/test_adv_curriculum_integration.py": "Integration test validating PGD-KL execution during training step",
    "tests/test_ais_ablation_flags.py": "Verifies that disabling AIS flags reverts gaze policy to deterministic fixed grid",
    "tests/test_ais_v2_gaze_policy.py": "Tests AIS candidate ranking, temperature softmax scaling, and entropy reduction",
    "tests/test_ais_v2_gradient_flow.py": "Asserts non-zero gradient backpropagation to AIS policy parameters",
    "tests/test_backbone_param_count.py": "Asserts CompactViT parameter count remains within the strict [20M, 25M] bounds",
    "tests/test_belief_state_composition.py": "Tests latent belief state tensor concatenation, initialization, and shapes",
    "tests/test_center_bias_measurement.py": "Calculates spatial fixation entropy to detect and prevent foveal center collapse",
    "tests/test_checkpoint_never_silently_restarts.py": "Asserts training immediately aborts if a mandatory parent checkpoint is missing",
    "tests/test_comparator_reuse_integrity.py": "Verifies comparator baselines use identical evaluation seeds and data slices",
    "tests/test_config_backward_compat.py": "Ensures configuration schema changes maintain full backward compatibility",
    "tests/test_consistency_assert_catches_corruption.py": "Tests that table-CSV integrity assertions flag injected data discrepancies",
    "tests/test_detached_copy_isolation.py": "Verifies belief state detachment prevents unintended BPTT memory leaks",
    "tests/test_diagnostics_next.py": "Unit tests for diagnostic profiling scripts and memory trackers",
    "tests/test_drift_to_gradient_flow.py": "Verifies gradient flow through belief state drift regularization targets",
    "tests/test_eval_rhan_protocol.py": "Verifies that evaluation harness strictly enforces the registered attack protocol",
    "tests/test_eval_structural_consistency.py": "Asserts that evaluation data loaders and model heads match class labels",
    "tests/test_evidential_gradient_flow.py": "Verifies non-zero gradient propagation through Dirichlet evidential heads",
    "tests/test_evidential_shapes.py": "Asserts Dirichlet concentration parameters (alpha) have correct positive output bounds",
    "tests/test_foveation_differentiable.py": "Verifies differentiability of spatial grid sampling during glimpse extraction",
    "tests/test_gaze_state_canonical.py": "Verifies canonical data structure formatting for fixation history trajectories",
    "tests/test_gen0_recipe_freeze.py": "Ensures historical Gen-0 training recipe hyperparameters remain immutably frozen",
    "tests/test_gradient_flow.py": "Comprehensive gradient backpropagation check across all trainable parameter groups",
    "tests/test_hpc_belief_gradient_flow.py": "Asserts gradient flow through hierarchical predictive coding belief pathways",
    "tests/test_hpc_diag_baseline.py": "Baseline diagnostic checks for predictive error computation modules",
    "tests/test_hpc_disable_backward_compat.py": "Verifies that disabling predictive coding cleanly recovers feedforward behavior",
    "tests/test_hpc_gradient_flow.py": "Verifies gradients reach predictive error projection weights",
    "tests/test_hpc_isolated.py": "Standalone unit tests for hierarchical predictive coding components in isolation",
    "tests/test_hpc_optimizer_group_resume.py": "Verifies optimizer state restoration when resuming predictive coding phases",
    "tests/test_interface_imports.py": "Verifies clean importability of all public abstract base classes and interfaces",
    "tests/test_j2_ais_v2_integration.py": "End-to-end integration test of AIS-v2 gaze policy in Job J2 dispatch pipeline",
    "tests/test_lens_introspection.py": "Tests feature capture hooks and activation recording in the introspection lens",
    "tests/test_live_perception_determinism.py": "Asserts deterministic evaluation outputs given pinned RNG seeds",
    "tests/test_multi_group_optimizer.py": "Verifies parameter allocation, weight decay grouping, and learning rate scaling",
    "tests/test_multi_group_optimizer_isolation.py": "Ensures learning rate updates to one group do not affect other parameter groups",
    "tests/test_numerical_stability.py": "Tests model behavior under extreme input perturbations, overflow, and underflow",
    "tests/test_pillar_scaffold_import.py": "Verifies clean instantiation of modular cognitive pillars (HPC, AIS, SBR, IWM)",
    "tests/test_precision_own_optimizer_group.py": "Asserts that precision weighting heads occupy an isolated optimizer group",
    "tests/test_predictor_ema_target.py": "Verifies exponential moving average shadow updates for prediction target latents",
    "tests/test_predictor_is_cheap.py": "Asserts next-glimpse predictor FLOPs and parameter count remain below compute budget caps",
    "tests/test_preflight_dw.py": "Pre-flight test checking parameter update magnitudes (||dW||) before committing runs",
    "tests/test_provenance_manifest.py": "Verifies git SHA, environment hashes, and config fingerprints in run manifests",
    "tests/test_recurrent_tied_weights.py": "Asserts parameter count remains constant regardless of temporal glimpse steps T",
    "tests/test_resume_commit_guard.py": "Asserts that resuming runs requires matching git commits to prevent drift",
    "tests/test_run_report.py": "Unit tests validating the formatting and numerical accuracy of generated run reports",
    "tests/test_sbr0_gate_criteria.py": "Unit test asserting Stage SBR-0 gate promotion requirements",
    "tests/test_sbr_gradient_flow.py": "Verifies gradient backpropagation through spatial belief representation slot masks",
    "tests/test_schema_version.py": "Asserts configuration dataclass schema versioning contracts",
    "tests/test_spatial_error_pool.py": "Verifies spatial pooling layer shapes and reduction operations on error maps",
    "tests/test_stage2_dag_artifacts.py": "Validates existence and integrity of Stage 2 DAG dependency artifacts",
    "tests/test_stage2_mechanism_seam.py": "Tests interface seams between belief update and predictive error modules",
    "tests/test_stage_state_machine.py": "Unit test verifying state machine transitions across foundation training stages",
    "tests/test_stage_state_machine_resume.py": "Tests that the state machine resumes from the correct stage upon interruption",
    "tests/test_sweep_resume.py": "Verifies automatic resumption of hyperparameter evaluation sweeps",
    "tests/test_t0_produces_exact_zero_error.py": "Asserts that initial step T=0 produces bit-exact zero prediction error",
    "tests/test_uncertainty_monotonic.py": "Asserts that epistemic uncertainty decreases monotonically as observations accumulate",
    "tests/test_update_net_gradient_flow.py": "Verifies gradient flow through UpdateNet belief innovation projections",
    "tests/test_warm_start_key_match.py": "Verifies key matching and positional interpolation when loading DINOv2 warm-starts",

    # docs
    "docs/REPOSITORY_MAP.md": "Comprehensive 1,600-line descriptive inventory of the entire repository",
    "docs/ARCHITECTURE.md": "Mathematical architecture specification: perceptual loop, objectives, state equations",
    "docs/FORENSIC_REPORT_NXA_GENERATION1.md": "Forensic audit report analyzing Gen-1 mechanisms and performance anomalies",
    "docs/NOESIS_FOUNDATION.md": "Biological visual cognition foundations, ventral stream models, and active inference",
    "docs/STAGE2_REFACTOR_PLAN.md": "Refactoring specification for Stage 2 modularization and agent decoupling",
    "docs/Stage_E1-3_Analysis.md": "Detailed empirical analysis across experimental stages E1 through E3",
    "docs/git_branch_audit_before_cleanup.md": "Branch audit documentation detailing historical branch lineage",
    "docs/git_branch_cleanup_report.md": "Cleanup report recording merged, archived, and pruned git branches",
    "docs/repository_reorganization.md": "Documentation of repository folder reorganization and canonical paths",
    "docs/stage3_environment.json": "Environment fingerprint and hardware specification for Stage 3 runs",
    "docs/stage3_preregistration.md": "Formal preregistered protocol and hypotheses for Stage 3 experiments",
    "docs/rhan_next_roadmap.json": "Machine-readable roadmap tracking stage milestones and completion gates",

    # report
    "report/GEN1_RESULTS_MASTER.md": "Master experimental results table for Generation-1 foundation phases",
    "report/Gen0.md": "Historical summary of Generation-0 STL-10 baseline experiments",
    "report/IEEEtran.cls": "IEEE Transactions LaTeX document class for manuscript publication",
    "report/empirical_report.json": "Master JSON database of clean and robust accuracy across all models",
    "report/final_sweep_results_stl10.json": "Benchmark sweep results across STL-10 recurrent architectures",
    "report/generate_figures.py": "Generates vector publication figures from empirical JSON reports",
    "report/prior_results.json": "Historical reference benchmarks and baseline comparison records",
    "report/rhan_nx_generation1_report.md": "Comprehensive experimental audit report for Generation-1",
    "report/rhan_nx_sbr0_cosine_t0.json": "Telemetry data for Stage SBR-0 cosine similarity checks at step T=0",
    "report/rhan_v10_scientific_report.tex": "LaTeX source for the RHAN v10 comprehensive scientific report",
    "report/rhan_v10_scientific_report_body.tex": "Body sections and theorem proofs for RHAN v10 report",
    "report/rhan_v10_scientific_report.pdf": "Compiled scientific publication manuscript for RHAN v10",
    "report/stage2_hpc_run_log.md": "Execution logs for Stage 2 high-performance computing (HPC) training",
    "report/stage3_trainer_audit.md": "Audit findings and gradient verification for Stage 3 trainer",
    "report/supervisor.log": "Log trace emitted by the autonomous experiment supervisor process",
}

def get_comment(path, name, is_file):
    if not is_file:
        return dir_comments.get(path, f"Subsystem directory: {name}/")
    
    if path in file_comments:
        return file_comments[path]
    if name in file_comments:
        return file_comments[name]
        
    # Heuristics with rich descriptions
    if path.startswith("tests/"):
        n = name.replace("test_", "").replace(".py", "").replace("_", " ")
        return f"Unit test asserting {n} behavior and invariant constraints"
    if path.startswith("noesis_vision/RHAN_NXA/docs/"):
        title = name.replace(".md", "").replace("_", " ")
        return f"Technical specification chapter: {title}"
    if path.startswith("docs/rhan_nxa/book/"):
        title = name.replace(".md", "").replace("_", " ")
        return f"Handbook textbook chapter: {title}"
    if path.startswith("docs/rhan_nxa/appendices/"):
        title = name.replace(".md", "").replace("_", " ")
        return f"Handbook appendix: {title}"
    if path.startswith("cloud/"):
        return f"Cloud execution launcher / configuration script: {name}"
    if path.startswith("diagnostics/"):
        return f"Diagnostic isolation test / profiling script: {name}"
    if path.startswith("report/"):
        return f"Report artifact or empirical analysis tool: {name}"
    if path.startswith("runs/"):
        run_name = os.path.dirname(path).split("/")[-1]
        return f"TensorBoard telemetry event log for experiment run: {run_name}"
    if path.startswith("archive/rhan-v1-v7/"):
        return f"Early prototype source: {name}"
    if path.startswith("archive/stl10-scaleup/"):
        return f"STL-10 pretraining / pseudo-labeling experiment script: {name}"
    if path.startswith("archive/gen0/"):
        return f"Historical Gen-0 (RHAN-Next) component: {name}"
    if path.startswith("archive/gen-1-cifar12/"):
        return f"Historical CIFAR-10 baseline comparison file: {name}"
    if path.startswith("archive/gen3-human/"):
        return f"Human psychophysics trial log and empirical response data: {name}"
    if path.startswith("archive/gen4-analysis/"):
        return f"Representation analysis script and divergence plot: {name}"
    if path.startswith("archive/gen5-sdt/"):
        return f"Signal Detection Theory sensitivity (d-prime) and criterion script: {name}"
    if path.startswith("archive/pkg-rhan-math/"):
        return f"Mathematical derivation and formal proof manuscript: {name}"
    if path.startswith("archive/legacy-evals/"):
        return f"Archived evaluation harness script / benchmark tool: {name}"
    if path.startswith("archive/"):
        return f"Archived project asset: {name}"
    if path.startswith("figures/"):
        return f"Publication figure asset: {name}"
    if path.startswith("figures_v2/"):
        return f"Extended figure asset: {name}"
    if path.startswith("figures_v3/"):
        return f"Publication figure v3 asset: {name}"
    if path.startswith(".verify_samples/"):
        cls_name = path.split("/")[1]
        return f"Pinned evaluation validation image for class '{cls_name}': {name}"
    if ":Zone.Identifier" in name:
        base = name.split(":")[0]
        return f"Windows filesystem download zone identifier metadata for {base}"
    return f"Source component: {name}"

def render_tree(node, prefix=""):
    lines = []
    keys = sorted(node.children.keys())
    for i, k in enumerate(keys):
        child = node.children[k]
        is_last = (i == len(keys) - 1)
        connector = "└── " if is_last else "├── "
        sub_prefix = "    " if is_last else "│   "
        
        comment = get_comment(child.path, child.name, child.is_file)
        display_name = child.name if child.is_file else f"{child.name}/"
        line_str = f"{prefix}{connector}{display_name:<44} # {comment}"
        lines.append(line_str)
        
        if not child.is_file:
            lines.extend(render_tree(child, prefix + sub_prefix))
    return lines

tree_lines = render_tree(root)
print(f"Rendered total lines: {len(tree_lines)}")

# Update README.md
with open("README.md", "r") as f:
    content = f.read()

start_marker = "## Repository Map & Directory Layout"
end_marker = "## References"

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx == -1 or end_idx == -1:
    raise ValueError("Markers not found in README.md!")

header = """## Repository Map & Directory Layout

The codebase has evolved from the initial CIFAR-10 baseline study into the **RHAN-NXA** (Recurrent Hierarchical Attention Network — Noesis eXperimental Architecture) cognitive vision system. Below is a complete, exhaustive structural map of all directories and files across the repository, annotated with their functional role:

```text
.
"""

new_section = header + "\n".join(tree_lines) + "\n```\n\n"
updated_content = content[:start_idx] + new_section + content[end_idx:]

with open("README.md", "w") as f:
    f.write(updated_content)

print("Updated README.md successfully!")
