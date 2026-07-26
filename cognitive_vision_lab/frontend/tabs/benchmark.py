import json
import time
from pathlib import Path

import streamlit as st
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

from cognitive_vision_lab.backend.model_registry import list_available_models, load_model
from cognitive_vision_lab.backend.attacks import pgd_attack, compute_accuracy
from cognitive_vision_lab.backend.metrics import accuracy_to_dprime, find_ethresh
from cognitive_vision_lab.config import STL10_MEAN, STL10_STD


def render():
    st.subheader("Performance Benchmark")
    st.caption(
        "Run standardized robustness benchmarks on selected models. "
        "Results are computed live and not cached."
    )

    models = list_available_models()
    model_names = [m["id"] for m in models]
    selected = st.multiselect("Select models to benchmark", model_names, default=model_names[:2])

    col1, col2, col3 = st.columns(3)
    with col1:
        eps_grid = st.text_input("Epsilon grid (comma-separated)", "0.0, 0.008, 0.0313")
    with col2:
        n_samples = st.number_input("Test samples per model", 50, 500, 100, step=50)
    with col3:
        pgd_steps = st.number_input("PGD steps", 10, 100, 40)

    use_cpu = st.checkbox("Force CPU (slow but works on any machine)", value=True)
    run_btn = st.button("Run Benchmark", type="primary")

    if run_btn and selected:
        epsilons = [float(x.strip()) for x in eps_grid.split(",")]

        st.markdown("---")
        st.markdown("### Results")

        progress_bar = st.progress(0)
        status_text = st.empty()

        results = []
        for i, model_id in enumerate(selected):
            status_text.text(f"Benchmarking {model_id}... ({i+1}/{len(selected)})")

            try:
                model, transform, is_stl10 = load_model(model_id, use_cpu=use_cpu)
                device = next(model.parameters()).device

                t0 = time.time()

                if is_stl10:
                    dataset = torchvision.datasets.STL10(
                        root="./data/stl10", split="test", download=True,
                        transform=transform,
                    )
                    n_classes = 10
                else:
                    dataset = torchvision.datasets.ImageNet(
                        root="./data/imagenet", split="val",
                        transform=transform,
                    )
                    n_classes = 1000

                indices = list(range(min(n_samples, len(dataset))))
                subset = Subset(dataset, indices)
                loader = DataLoader(subset, batch_size=32, num_workers=0)

                clean_acc = compute_accuracy(model, loader, device)
                ethresh = None
                rob_results = {}

                for eps in epsilons:
                    if eps == 0:
                        continue
                    correct = total = 0
                    model.eval()
                    for images, labels in loader:
                        images, labels = images.to(device), labels.to(device)
                        for b in range(images.size(0)):
                            adv = pgd_attack(
                                model, images[b], labels[b].item(),
                                eps=eps, steps=pgd_steps,
                            )
                            with torch.no_grad():
                                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                                    out = model(adv.unsqueeze(0))
                                    if isinstance(out, (tuple, list)):
                                        out = out[0]
                                correct += out.argmax(1).eq(labels[b]).sum().item()
                                total += 1
                    acc = 100.0 * correct / max(total, 1)
                    rob_results[f"ε={eps}"] = acc
                    if acc < 50:
                        dp = accuracy_to_dprime(acc, n_classes)
                        if dp < 1.0 and ethresh is None:
                            ethresh = eps

                inference_ms = (time.time() - t0) / max(len(indices), 1) * 1000
                params_m = sum(p.numel() for p in model.parameters()) / 1e6

                results.append({
                    "Model": model_id,
                    "Clean Acc %": round(clean_acc, 2),
                    **{f"Rob@ε={eps}": round(rob_results.get(f"ε={eps}", 0), 2) for eps in epsilons if eps > 0},
                    "εthresh": ethresh if ethresh else ">max",
                    "Params (M)": round(params_m, 1),
                    "Inference (ms/img)": round(inference_ms, 1),
                })

            except Exception as e:
                st.error(f"Error benchmarking {model_id}: {e}")
                results.append({"Model": model_id, "Error": str(e)})

            progress_bar.progress((i + 1) / len(selected))

        progress_bar.empty()
        status_text.text("Benchmark complete.")

        if results:
            st.dataframe(results, use_container_width=True)

            import pandas as pd
            df = pd.DataFrame(results)
            csv = df.to_csv(index=False)
            st.download_button(
                "Download results as CSV", csv, "benchmark_results.csv", "text/csv",
            )
