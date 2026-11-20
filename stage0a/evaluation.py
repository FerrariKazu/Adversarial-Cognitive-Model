"""Official ImageNet-1k evaluation and 3-seed aggregation for Stage 0A."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from stage0a.features import FeatureCache


def _ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(conf)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf > lo) & (conf <= hi)
        if not np.any(mask):
            continue
        acc = float(correct[mask].mean())
        avg_conf = float(conf[mask].mean())
        ece += (mask.sum() / total) * abs(acc - avg_conf)
    return float(ece)


@torch.no_grad()
def evaluate_head(
    head: torch.nn.Module,
    cache: FeatureCache,
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate a linear head on a pooled-feature cache.

    Returns (metrics, labels, predictions).  The official val split is passed
    exactly once by the trainer; this function has no selection side effects.
    """
    head.eval()
    features = cache.store()
    labels = np.load(cache.labels, mmap_mode="r")
    n = len(labels)
    preds = np.empty(n, dtype=np.int64)
    top5s = np.empty((n, 5), dtype=np.int64)
    nll_sum = 0.0
    correct = np.empty(n, dtype=bool)
    confidence = np.empty(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        x = torch.from_numpy(np.array(features[start:end], copy=True)).float().to(device)
        logits = head(x)
        probs = F.softmax(logits.float(), dim=-1)
        conf, pred = probs.max(dim=-1)
        top5 = probs.topk(5, dim=-1).indices
        y = torch.from_numpy(np.array(labels[start:end], copy=True)).long().to(device)
        nll_sum += float(F.cross_entropy(logits.float(), y, reduction="sum").item())
        preds[start:end] = pred.cpu().numpy()
        top5s[start:end] = top5.cpu().numpy()
        confidence[start:end] = conf.cpu().numpy()
        correct[start:end] = (pred == y).cpu().numpy()
    y = np.asarray(labels, dtype=np.int64)
    top1 = float(correct.mean())
    top5 = float(np.mean([y[i] in top5s[i] for i in range(n)]))
    nll = float(nll_sum / max(n, 1))
    ece = _ece(confidence, correct)
    per_class = []
    for c in range(int(y.max()) + 1):
        m = y == c
        if np.any(m):
            per_class.append(float(correct[m].mean()))
    metrics = {
        "n": int(n),
        "top1": top1,
        "top5": top5,
        "nll": nll,
        "ece_15bin": ece,
        "macro_top1": float(np.mean(per_class)) if per_class else None,
        "per_class_count": len(per_class),
        "confidence_mean": float(confidence.mean()),
        "correct_count": int(correct.sum()),
    }
    return metrics, y, preds


def save_eval_artifacts(
    run_dir: str | Path,
    *,
    metrics: Dict[str, Any],
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: Optional[list[str]] = None,
) -> Dict[str, str]:
    """Persist predictions and a compact per-class accuracy CSV."""
    import csv

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run_dir / "official_val_predictions.npz",
        labels=labels.astype(np.int16),
        predictions=preds.astype(np.int16),
    )
    per_class_path = run_dir / "official_val_per_class.csv"
    with open(per_class_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_index", "class_name", "correct", "total", "accuracy"])
        for c in range(int(labels.max()) + 1):
            m = labels == c
            total = int(m.sum())
            if total:
                w.writerow([c, class_names[c] if class_names else c, int((preds[m] == c).sum()), total, float((preds[m] == c).mean())])
    report_path = run_dir / "official_val_metrics.json"
    report_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "metrics": str(report_path),
        "predictions": str(run_dir / "official_val_predictions.npz"),
        "per_class": str(per_class_path),
    }


def aggregate_seed_reports(
    reports: list[Dict[str, Any]],
    *,
    backbone: str,
    target_top1: float = 0.80,
    seed_floor_top1: float = 0.79,
) -> Dict[str, Any]:
    top1 = np.asarray([r["top1"] for r in reports], dtype=np.float64)
    top5 = np.asarray([r["top5"] for r in reports], dtype=np.float64)
    nll = np.asarray([r["nll"] for r in reports], dtype=np.float64)
    ece = np.asarray([r["ece_15bin"] for r in reports], dtype=np.float64)
    result = {
        "backbone": backbone,
        "n_seeds": len(reports),
        "seeds": [r.get("seed") for r in reports],
        "top1_mean": float(top1.mean()),
        "top1_std": float(top1.std(ddof=1)) if len(top1) > 1 else 0.0,
        "top1_min": float(top1.min()),
        "top1_max": float(top1.max()),
        "top5_mean": float(top5.mean()),
        "nll_mean": float(nll.mean()),
        "ece_mean": float(ece.mean()),
        "target_top1": float(target_top1),
        "seed_floor_top1": float(seed_floor_top1),
        "pass_mean_target": bool(top1.mean() >= target_top1),
        "pass_all_seed_floor": bool((top1 >= seed_floor_top1).all()),
        "verdict": "PASS" if (top1.mean() >= target_top1 and (top1 >= seed_floor_top1).all()) else "FAIL",
        "per_seed": sorted(reports, key=lambda r: r.get("seed", 0)),
    }
    return result
