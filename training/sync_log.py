"""Noise control for Kaggle/Colab orchestration chatter (Step 4).

Routes git-clone/empty-commit/HF-sync chatter to report/sync.log and prints
only one line per sync: "HF sync ok (<n> files, <sec>s)" or a loud failure.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

_REPORT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "report"
)


def sync_log_path() -> str:
    return os.path.join(_REPORT_DIR, "sync.log")


def configure_sync_logger() -> logging.Logger:
    """Create/return the logger that writes noisy orchestration chatter."""
    os.makedirs(_REPORT_DIR, exist_ok=True)
    logger = logging.getLogger("hf_sync")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        h = logging.FileHandler(sync_log_path(), mode="a", encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(h)
    return logger


def log_sync_event(message: str,
                   logger: Optional[logging.Logger] = None) -> None:
    logger = logger or configure_sync_logger()
    logger.info(message)


# Compatibility alias so existing callers that do
#   from training.sync_log import log_sync
# continue to work.
log_sync = log_sync_event


def print_sync_status(prefix: str, n_files: int, seconds: float,
                      ok: bool,
                      logger: Optional[logging.Logger] = None,
                      extra: Optional[str] = None) -> None:
    """Print ONE line per sync to stdout; full chatter goes to sync.log."""
    logger = logger or configure_sync_logger()
    if ok:
        line = f"{prefix} HF sync ok ({n_files} files, {seconds:.1f}s)"
    else:
        line = f"{prefix} HF sync FAILED ({n_files} files, {seconds:.1f}s)"
    if extra:
        line += f" — {extra}"
    print(line, flush=True)
    log_sync_event(line, logger=logger)


def summarize_sync_log(path: Optional[str] = None) -> str:
    p = path or sync_log_path()
    if not os.path.exists(p):
        return f"no sync.log at {p}"
    lines = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return "\n".join(lines[-200:]) if lines else "sync.log empty"
