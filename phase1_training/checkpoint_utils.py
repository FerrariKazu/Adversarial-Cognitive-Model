import sys
import types
import torch


def compat_load(path, map_location=None, **kwargs):
    """Load a PyTorch checkpoint with compatibility for removed modules.

    PyTorch >= 2.5 removed ``torch.utils.serialization``, but the
    internals of ``torch.serialization`` still reference it at
    runtime (e.g. ``from torch.utils.serialization import config``).
    We inject a proxy module into ``sys.modules`` to prevent the
    resulting ``ModuleNotFoundError``.
    """
    if 'torch.utils.serialization' not in sys.modules:
        _ensure_serialization_proxy()

    return torch.load(path, map_location=map_location, weights_only=False, **kwargs)


def _ensure_serialization_proxy():
    class _LoadConfig:
        calculate_storage_offsets = True
        mmap = False
        mmap_flags = 0
        endianness = 'little'

    config_mod = types.ModuleType('torch.utils.serialization.config')
    config_mod.__path__ = []
    config_mod.__package__ = 'torch.utils.serialization'
    config_mod.load = _LoadConfig()
    sys.modules['torch.utils.serialization.config'] = config_mod

    parent_mod = types.ModuleType('torch.utils.serialization')
    parent_mod.__path__ = []
    parent_mod.__package__ = 'torch.utils'
    parent_mod.config = config_mod
    sys.modules['torch.utils.serialization'] = parent_mod
