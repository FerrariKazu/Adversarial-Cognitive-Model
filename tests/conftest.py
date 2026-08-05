"""Pytest setup: make the repo root and phase1_training importable."""
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (_ROOT, os.path.join(_ROOT, "phase1_training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _cuda_or_cpu():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pytest_addoption(parser):
    parser.addoption(
        "--device", action="store", default=None,
        help="Device for heavy model tests (default: cuda if available)")


def pytest_configure(config):
    dev = config.getoption("--device")
    config.option.device = dev or _cuda_or_cpu()


def pytest_generate_tests(metafunc):
    pass
