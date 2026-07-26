"""Batch sweep generator: loads every discoverable checkpoint and generates d' data."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase1_training"))

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from cognitive_vision_lab.config import (
    SWEEP_PATH, CHECKPOINTS_DIR, CHECKPOINTS_TIER2_DIR,
    STL10_MEAN, STL10_STD,
)
from cognitive_vision_lab.backend.model_registry import (
    get_all_checkpoint_models, load_model, predict,
    ARCHITECTURE_LOOKUP,
)
from cognitive_vision_lab.backend.attacks import pgd_attack, compute_accuracy
from cognitive_vision_lab.backend.metrics import accuracy_to_dprime, find_ethresh

EPSILONS = [0.0, 0.002, 0.004, 0.006, 0.008, 0.016, 0.024, 0.0313]
N_SAMPLES = 100
BATCH_SIZE = 32
N_CLASSES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _sweep_key(ckpt_name: str) -> str:
    key = ckpt_name.replace(".pth", "").replace(":Zone.Identifier", "")
    for suffix in ["_best", "_final", "_rolling", "_checkpoint"]:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def build_stl10_loader(transform, n_samples=N_SAMPLES):
    full = torchvision.datasets.STL10(
        root="./data/stl10", split="test", download=True,
        transform=transform,
    )
    indices = list(range(min(n_samples, len(full))))
    subset = Subset(full, indices)
    return DataLoader(subset, batch_size=BATCH_SIZE, num_workers=0)


def compute_sweep(model, loader, device=DEVICE):
    epsilons = EPSILONS
    clean_acc = compute_accuracy(model, loader, device)
    macro_dprimes = []
    pooled_dprimes = []
    accuracy_list = [clean_acc]

    for eps in epsilons[1:]:
        correct = total = 0
        model.eval()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            for b in range(images.size(0)):
                adv = pgd_attack(
                    model, images[b], labels[b].item(),
                    eps=eps, steps=40,
                )
                with torch.no_grad():
                    out = model(adv.unsqueeze(0))
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    correct += out.argmax(1).eq(labels[b]).sum().item()
                    total += 1
        acc = 100.0 * correct / max(total, 1)
        accuracy_list.append(acc)

    for acc in accuracy_list:
        dp = accuracy_to_dprime(acc, N_CLASSES)
        macro_dprimes.append(dp.get("macro", 0.0) if isinstance(dp, dict) else dp)
        pooled_dprimes.append(dp.get("pooled", 0.0) if isinstance(dp, dict) else dp)

    thresh_macro = find_ethresh(epsilons, macro_dprimes, target=1.0)
    thresh_pooled = find_ethresh(epsilons, pooled_dprimes, target=1.0)

    return {
        "epsilons": epsilons,
        "accuracy": [round(a, 4) for a in accuracy_list],
        "macro_dprime": [round(d, 4) for d in macro_dprimes],
        "pooled_dprime": [round(d, 4) for d in pooled_dprimes],
        "thresh_dprime_1_macro": thresh_macro,
        "thresh_dprime_1_pooled": thresh_pooled,
    }


def main():
    transform = T.Compose([
        T.Resize((96, 96)),
        T.ToTensor(),
        T.Normalize(STL10_MEAN, STL10_STD),
    ])

    try:
        loader = build_stl10_loader(transform)
    except Exception as e:
        print(f"Failed to load STL-10: {e}")
        sys.exit(1)

    # Load existing data
    if SWEEP_PATH.exists():
        with open(SWEEP_PATH) as f:
            results = json.load(f)
    else:
        results = {}

    # Also check for Tier2 duplicates
    tier2_ckpts = set()
    if CHECKPOINTS_TIER2_DIR.exists():
        for fpath in CHECKPOINTS_TIER2_DIR.iterdir():
            if fpath.name.endswith(".pth") and ":Zone.Identifier" not in fpath.name:
                tier2_ckpts.add(fpath.name)

    discovered = get_all_checkpoint_models()
    total = len(discovered)
    print(f"Found {total} checkpoints to benchmark")

    for idx, (ckpt_name, entry) in enumerate(discovered.items()):
        sk = _sweep_key(ckpt_name)
        if sk in results:
            print(f"  [{idx+1}/{total}] {ckpt_name} — already in sweep data, skipping")
            continue

        if ckpt_name.startswith("rhan_stl10_large_pseudolabel") and ckpt_name in tier2_ckpts:
            # Check if the main dir version was already processed
            main_version = ckpt_name  # same name in both dirs, process once
            if ckpt_name in {e["checkpoint"] for e in discovered.values()}:
                pass

        print(f"  [{idx+1}/{total}] Loading {ckpt_name}... ", end="", flush=True)
        t0 = time.time()

        # Need to use _sweep_key as model_id since list_available uses the filename
        model_id = ckpt_name

        try:
            model, _, _ = load_model(model_id)
            load_time = time.time() - t0
            print(f"loaded in {load_time:.1f}s ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")

            print(f"         Running sweep ({N_SAMPLES} samples)... ", end="", flush=True)
            t1 = time.time()
            sweep_data = compute_sweep(model, loader)
            sweep_time = time.time() - t1
            print(f"done in {sweep_time:.1f}s")

            results[sk] = sweep_data

            # Save incrementally
            with open(SWEEP_PATH, "w") as f:
                json.dump(results, f, indent=2)
            print(f"         Saved to {SWEEP_PATH}")

        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nDone. Sweep data for {len(results)} models saved to {SWEEP_PATH}")


if __name__ == "__main__":
    main()
