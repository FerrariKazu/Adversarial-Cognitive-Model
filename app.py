#!/usr/bin/env python3
"""
Cognitive Vision Lab — Intelligent Benchmarking Platform
========================================================
Research dashboard for comparing human vision, CNNs, ViTs, and
RHAN-inspired models under adversarial conditions.

Run with:
    streamlit run app.py

Deploy with:
    cd cognitive_vision_lab && docker-compose up
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cognitive_vision_lab.app import main

if __name__ == "__main__":
    main()
