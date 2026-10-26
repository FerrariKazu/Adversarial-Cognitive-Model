#!/usr/bin/env python3
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

def get_comment(path, name, is_file):
    if not is_file:
        dir_comments = {
            "gen2_foundation": "Generation-2 foundation model package (DINOv2, TRADES-PGD4, GRU belief)",
            "gen2_foundation/tests": "Unit test suite for gen2_foundation components",
            "noesis_vision": "NOESIS cognitive vision engine and biological perceptual loop",
            "noesis_vision/RHAN_NXA": "RHAN-NXA architecture documentation and master specifications",
            "noesis_vision/RHAN_NXA/docs": "32-chapter comprehensive technical specification handbook",
            "noesis_vision/core": "Core cognitive orchestrator, schemas, and optimizer abstractions",
            "noesis_vision/models": "Neural backbones, foveation mechanisms, and vision transformers",
            "noesis_vision/gaze": "Adaptive Information Seeking (AIS) candidate generator and gaze policy",
            "noesis_vision/beliefs": "Latent belief state representations (z_t) and spatial memory",
            "noesis_vision/predictive_coding": "Top-down predictive coding, predictor heads, and error pooling",
            "noesis_vision/uncertainty": "Dirichlet evidential heads, softmax entropy, and calibration modules",
            "training": "Training orchestration, adversarial curricula, and cloud persistence",
            "evaluation": "Evaluation datasets, corruptions, and benchmark protocols",
            "scripts": "Operational automation, data preparation, gate verification, and reports",
            "tests": "Comprehensive pytest test suite (invariants, regression, gradient flows)",
            "docs": "Technical manuals, design plans, and forensic reports",
            "docs/rhan_nxa": "RHAN-NXA unified handbook, QA reports, and figure gallery",
            "docs/rhan_nxa/book": "RHAN-NXA comprehensive textbook chapters (00-26)",
            "docs/rhan_nxa/appendices": "Technical appendices (tensors, ABC interfaces, port tables)",
            "docs/rhan_nxa/figure_gallery": "Standalone HTML figure gallery and high-res diagram assets",
            "docs/rhan_nxa/figures": "Vector SVG publication diagrams and architecture schematics",
            "docs/rhan_nxa/pdf_qa_pages": "Rendered visual QA inspection pages for compiled handbook",
            "docs/historical": "Archived stage reports, early proposal drafts, and pre-registrations",
            "docs/research": "Research references, citation audits, and empirical seed tables",
            "report": "Experimental results, master tables, and scientific audit reports",
            "report/assets": "Plot assets and visual diagrams for empirical reports",
            "report/stage3_D": "Stage 3-D Pareto trade-off analysis, figures, and provenance audits",
            "report/stage3_D/figures": "Visual plots comparing Stage 3-D Pareto trade-offs and seed variance",
            "report/stage4_E1": "Stage 4-E1 lens introspection, pre-registration scripts, and runners",
            "report/sbr_feasibility": "Spatial belief representation (SBR) slot mask visual feasibility checks",
            "report/lens_e1_analysis": "Comprehensive audit report for Lens E1 introspection experiment",
            "cloud": "Cloud deployment configurations and Slurm cluster specs",
            "cloud/canonical": "Canonical production cloud dispatchers",
            "cloud/gen0": "Historical Gen-0 cloud dispatchers",
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
        return dir_comments.get(path, f"Directory: {name}/")
    
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
        "gen2_foundation/__init__.py": "Package exports and initialization for Gen-2 foundation",
        "gen2_foundation/flags.py": "Feature flags, experimental toggles, and runtime configuration switches",
        "gen2_foundation/gist.py": "Global gist encoder using DINOv2-small patch embedding and CLS projection",
        "gen2_foundation/precision.py": "Observation reliability scoring and dynamic precision weighting",
        "gen2_foundation/recipe.py": "Layer-wise parameter grouping and AdamW optimizer schedule builder",
        "gen2_foundation/ema.py": "Exponential Moving Average (EMA) shadow weights tracker for robust evaluation",
        "gen2_foundation/eot.py": "Expectation Over Transformation (EOT-PGD) implementation for stochastic defense",
        "gen2_foundation/update_net_v2.py": "Bounded recurrent belief update network with residual state dynamics",
        "gen2_foundation/AUDIT.md": "Package forensic audit ledger, test results, and integrity verification",
        "gen2_foundation/EXPERIMENT_REGISTRY.md": "Preregistered Gen-2 hypothesis arms, baseline comparisons, and metrics",

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

        # config
        "config/train_config.yaml": "YAML configuration defining learning rates, attack radii, and curriculum stages",

        # checkpoints
        "checkpoints/generation1_foundation_roadmap.json": "Stage tracking and checkpoint state metadata for Gen-1 ladder",
        "checkpoints/training_complete.json": "Sentinel JSON indicating completion status of training pipeline",

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
    
    if path in file_comments:
        return file_comments[path]
    if name in file_comments:
        return file_comments[name]
        
    if path.startswith("tests/"):
        n = name.replace("test_", "").replace(".py", "").replace("_", " ")
        return f"Unit test asserting {n}"
    if path.startswith("gen2_foundation/tests/"):
        n = name.replace("test_", "").replace(".py", "").replace("_", " ")
        return f"Gen-2 unit test asserting {n}"
    if path.startswith("noesis_vision/RHAN_NXA/docs/"):
        return f"RHAN-NXA technical specification chapter: {name.replace('.md', '').replace('_', ' ')}"
    if path.startswith("docs/rhan_nxa/book/"):
        return f"Handbook textbook chapter: {name.replace('.md', '').replace('_', ' ')}"
    if path.startswith("docs/rhan_nxa/appendices/"):
        return f"Handbook appendix: {name.replace('.md', '').replace('_', ' ')}"
    if path.startswith("cloud/"):
        return f"Cloud execution script / configuration: {name}"
    if path.startswith("diagnostics/"):
        return f"Diagnostic isolation test / trace: {name}"
    if path.startswith("report/"):
        return f"Report artifact or analysis tool: {name}"
    if path.startswith("runs/"):
        return f"TensorBoard telemetry event log: {name}"
    if path.startswith("archive/"):
        return f"Archived project asset: {name}"
    if path.startswith("figures/"):
        return f"Figure diagram asset: {name}"
    if path.startswith("figures_v2/"):
        return f"Extended figure asset: {name}"
    if path.startswith("figures_v3/"):
        return f"Publication figure v3 asset: {name}"
    if path.startswith(".verify_samples/"):
        return f"Validation sample image: {name}"
    if ":Zone.Identifier" in name:
        return f"Windows filesystem metadata for {name.split(':')[0]}"
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
with open("scratch/full_rendered_tree.txt", "w") as f:
    f.write("\n".join(tree_lines))
print("Done!")
