"""Structured logging for the lab."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from cognitive_vision_lab.config import CACHE_DIR

_LOGGER_NAME = "cvl"


def get_logger(name: str = "cvl") -> logging.Logger:
    """Return the shared lab logger, configured once."""
    logger = logging.getLogger(_LOGGER_NAME if name == "cvl" else f"{_LOGGER_NAME}.{name}")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        log_file = CACHE_DIR / "cvl.log"
        try:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except OSError:
            pass
        logger.propagate = False
    return logger
