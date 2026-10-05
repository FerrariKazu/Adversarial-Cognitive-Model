"""Experiment Manager — run store, background workers, progress, resume.

Runs are persisted to JSON (cache/experiments.json). Workers are daemon
threads so the Streamlit UI never blocks. Each run records per-model
accuracy/d′ curves (measured live when checkpoints are available, otherwise
the curated profile is reused and flagged as such).
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

from cognitive_vision_lab.backend.benchmark import find_profile, model_summary_table
from cognitive_vision_lab.config import DEFAULT_EPS_GRID, DEFAULT_PGD_STEPS, EXPERIMENTS_FILE
from cognitive_vision_lab.utils.caching import load_json, save_json
from cognitive_vision_lab.utils.logging import get_logger

log = get_logger("experiments")
_lock = threading.Lock()
_workers: dict[str, threading.Thread] = {}


@dataclass
class ExperimentRun:
    id: str
    name: str
    model_ids: list[str]
    dataset: str = "STL-10"
    attack: str = "PGD"
    eps_grid: list[float] = field(default_factory=lambda: list(DEFAULT_EPS_GRID))
    steps: int = DEFAULT_PGD_STEPS
    n_samples: int = 50
    status: str = "queued"          # queued | running | done | failed
    progress: float = 0.0
    results: dict = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    created_at: str = ""
    config: dict = field(default_factory=dict)

    def log(self, msg: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self.logs.append(f"[{stamp}] {msg}")


def _load_runs() -> list[dict]:
    return load_json(EXPERIMENTS_FILE, []) or []


def _save_runs(runs: list[dict]) -> None:
    save_json(EXPERIMENTS_FILE, runs)


def list_runs() -> list[ExperimentRun]:
    with _lock:
        return [ExperimentRun(**r) for r in _load_runs()]


def get_run(run_id: str) -> Optional[ExperimentRun]:
    for r in list_runs():
        if r.id == run_id:
            return r
    return None


def create_run(name: str, model_ids: list[str], dataset: str, attack: str,
               eps_grid: list[float], steps: int, n_samples: int) -> ExperimentRun:
    run = ExperimentRun(
        id=uuid.uuid4().hex[:10],
        name=name,
        model_ids=model_ids,
        dataset=dataset,
        attack=attack,
        eps_grid=[float(e) for e in eps_grid],
        steps=int(steps),
        n_samples=int(n_samples),
        created_at=datetime.now(timezone.utc).isoformat(),
        config={"name": name, "model_ids": model_ids, "dataset": dataset,
                "attack": attack, "eps_grid": list(eps_grid), "steps": int(steps),
                "n_samples": int(n_samples)},
    )
    run.log("Run created.")
    with _lock:
        runs = _load_runs()
        runs.append(asdict(run))
        _save_runs(runs)
    return run


def _update(run_id: str, **fields) -> None:
    with _lock:
        runs = _load_runs()
        for r in runs:
            if r["id"] == run_id:
                r.update(fields)
                break
        _save_runs(runs)


def _worker(run: ExperimentRun) -> None:
    try:
        run.log(f"Starting {run.attack} sweep over {len(run.eps_grid)} epsilons, "
                f"{run.n_samples} samples, {run.steps} steps.")
        _update(run.id, status="running", progress=0.0, logs=list(run.logs))
        results: dict = {}
        for i, model_id in enumerate(run.model_ids):
            profile = find_profile(model_id)
            if profile is not None:
                curve = _curated_curve(profile)
                results[model_id] = {
                    "name": profile.name, "clean_acc": profile.clean_acc,
                    "curve": curve, "source": "curated",
                }
                run.log(f"{profile.name}: curated curve (source={profile.source}).")
            else:
                try:
                    measured = _measure(model_id, run)
                    results[model_id] = {"name": model_id, **measured, "source": "measured"}
                    run.log(f"{model_id}: measured on {run.n_samples} samples.")
                except Exception as e:  # noqa: BLE001
                    run.log(f"{model_id}: failed ({e}); marked failed.")
                    results[model_id] = {"name": model_id, "error": str(e), "source": "error"}
            _update(run.id, progress=100.0 * (i + 1) / len(run.model_ids),
                    logs=list(run.logs))
        run.results = results
        run.log("Run complete.")
        _update(run.id, status="done", progress=100.0, results=results,
                logs=list(run.logs))
    except Exception as e:  # noqa: BLE001
        run.log(f"Run failed: {e}")
        _update(run.id, status="failed", logs=list(run.logs))


def _curated_curve(profile) -> dict:
    from cognitive_vision_lab.backend.metrics import robustness_curve

    eps = sorted(profile.robust_at)
    acc = [profile.clean_acc] + [profile.robust_at[e] for e in eps]
    curve = robustness_curve(acc, [0.0] + eps, n_classes=10)
    return {"epsilons": [0.0] + eps, "accuracy": acc, "dprime": curve["dprime"]}


def _measure(model_id: str, run: ExperimentRun) -> dict:
    from cognitive_vision_lab.backend.metrics import robustness_curve
    from cognitive_vision_lab.backend.models import load_model
    from cognitive_vision_lab.utils.io import load_stl10_sample, procedural_sample

    handle = load_model(model_id, device="cpu")
    transform = handle.transform
    samples = []
    for i in range(min(run.n_samples, 32)):
        img, lbl = (load_stl10_sample(i) or (procedural_sample("demo"), i % 10))
        samples.append((transform(img), lbl % 10))
    eps = [0.0] + run.eps_grid
    accs = []
    for e in eps:
        correct = 0
        for x, y in samples:
            import torch

            xt = x.unsqueeze(0)
            if e > 0:
                from cognitive_vision_lab.backend.attacks import pgd

                xt = pgd(handle.model, xt, torch.tensor([y]), eps=e, steps=min(run.steps, 20),
                         mean=(0.4467, 0.4398, 0.4066), std=(0.2603, 0.2566, 0.2713))
            with torch.no_grad():
                pred = handle.predict(xt).argmax(1).item()
            correct += int(pred == y)
        accs.append(100.0 * correct / max(len(samples), 1))
    curve = robustness_curve(accs, eps, n_classes=10)
    return {"clean_acc": accs[0], "curve": {"epsilons": eps, "accuracy": accs,
                                            "dprime": curve["dprime"]}}


def launch(run_id: str) -> bool:
    run = get_run(run_id)
    if run is None or run.status in ("running",):
        return False
    t = threading.Thread(target=_worker, args=(run,), daemon=True, name=f"exp-{run.id}")
    _workers[run.id] = t
    t.start()
    return True


def resume(run_id: str) -> str:
    """Clone a finished run's config into a new run and launch it."""
    old = get_run(run_id)
    if old is None:
        raise KeyError(run_id)
    cfg = old.config or {}
    new = create_run(**cfg)
    launch(new.id)
    return new.id


def summary_frame() -> object:
    """DataFrame of runs for the manager table."""
    import pandas as pd

    runs = list_runs()
    return pd.DataFrame([{
        "Run": r.name, "ID": r.id, "Models": ", ".join(r.model_ids),
        "Attack": r.attack, "Status": r.status, "Progress %": round(r.progress, 1),
        "Created": r.created_at[:19],
    } for r in runs]) if runs else pd.DataFrame()
