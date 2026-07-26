"""
Optional PostgreSQL storage for benchmark results.

Schema:
  - benchmark_runs: top-level evaluation sessions
  - model_results: per-model accuracy + d' at each epsilon
  - inference_logs: individual prediction records

Configure via DATABASE_URL env var. Falls back to JSON if unset.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from cognitive_vision_lab.config import DATABASE_URL, DATA_DIR

SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT NOW(),
    label VARCHAR(255),
    attack_type VARCHAR(50) DEFAULT 'PGD-50',
    eps_grid FLOAT[],
    n_samples INTEGER,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS model_results (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES benchmark_runs(id) ON DELETE CASCADE,
    model_id VARCHAR(255),
    clean_acc FLOAT,
    rob_acc_at_eps JSONB,
    ethresh FLOAT,
    macro_dprime_curve FLOAT[],
    pooled_dprime_curve FLOAT[],
    params_million FLOAT,
    inference_ms FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    result_id INTEGER REFERENCES model_results(id) ON DELETE CASCADE,
    image_hash VARCHAR(64),
    epsilon FLOAT,
    predicted_class VARCHAR(255),
    predicted_idx INTEGER,
    confidence FLOAT,
    correct BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_results_run ON model_results(run_id);
CREATE INDEX IF NOT EXISTS idx_model_results_model ON model_results(model_id);
"""


class BenchmarkDB:
    def __init__(self):
        self.conn = None
        self._connect()

    def _connect(self):
        if not DATABASE_URL:
            return
        try:
            import psycopg2
            self.conn = psycopg2.connect(DATABASE_URL)
            self.conn.autocommit = True
            with self.conn.cursor() as cur:
                cur.execute(SQL_SCHEMA)
        except Exception as e:
            print(f"Database connection failed (falling back to JSON): {e}")
            self.conn = None

    @property
    def available(self):
        return self.conn is not None

    def save_run(self, label: str, attack_type: str, eps_grid: list[float],
                 n_samples: int, notes: str = "") -> Optional[int]:
        if not self.available:
            return None
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO benchmark_runs (label, attack_type, eps_grid, n_samples, notes) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (label, attack_type, eps_grid, n_samples, notes),
            )
            return cur.fetchone()[0]

    def save_model_result(self, run_id: int, model_id: str, clean_acc: float,
                          rob_acc: dict, ethresh: float,
                          macro_dprime: list, pooled_dprime: list,
                          params_m: float, inference_ms: float):
        if not self.available:
            self._json_fallback(run_id, model_id, clean_acc, rob_acc, ethresh)
            return
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO model_results "
                "(run_id, model_id, clean_acc, rob_acc_at_eps, ethresh, "
                "macro_dprime_curve, pooled_dprime_curve, params_million, inference_ms) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (run_id, model_id, clean_acc, json.dumps(rob_acc), ethresh,
                 macro_dprime, pooled_dprime, params_m, inference_ms),
            )

    def _json_fallback(self, run_id, model_id, clean_acc, rob_acc, ethresh):
        fallback_path = DATA_DIR / "benchmark_fallback.json"
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        records = []
        if fallback_path.exists():
            records = json.loads(fallback_path.read_text())
        records.append({
            "run_id": run_id,
            "model_id": model_id,
            "clean_acc": clean_acc,
            "rob_acc": rob_acc,
            "ethresh": ethresh,
            "timestamp": datetime.now().isoformat(),
        })
        fallback_path.write_text(json.dumps(records, indent=2))

    def get_recent_runs(self, limit: int = 10) -> list:
        if not self.available:
            return []
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id, created_at, label, attack_type, n_samples "
                "FROM benchmark_runs ORDER BY created_at DESC LIMIT %s", (limit,)
            )
            return cur.fetchall()

    def close(self):
        if self.conn:
            self.conn.close()


db = BenchmarkDB()
