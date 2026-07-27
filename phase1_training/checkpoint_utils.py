import sys
import types
import torch


def compat_load(path, map_location=None, **kwargs):
    _ensure_serialization_proxy()
    return torch.load(path, map_location=map_location, weights_only=False, **kwargs)


def compat_save(obj, path, **kwargs):
    _ensure_serialization_proxy()
    return torch.save(obj, path, **kwargs)


def _ensure_serialization_proxy():
    if 'torch.utils.serialization' in sys.modules:
        return

    class _Attrs:
        pass

    load = _Attrs()
    load.calculate_storage_offsets = True
    load.mmap = False
    load.mmap_flags = 0
    load.endianness = 'little'

    save = _Attrs()
    save.compute_crc32 = False
    save.storage_alignment = 64
    save.use_pinned_memory_for_d2h = False

    config_mod = types.ModuleType('torch.utils.serialization.config')
    config_mod.__path__ = []
    config_mod.__package__ = 'torch.utils.serialization'
    config_mod.load = load
    config_mod.save = save
    sys.modules['torch.utils.serialization.config'] = config_mod

    parent_mod = types.ModuleType('torch.utils.serialization')
    parent_mod.__path__ = []
    parent_mod.__package__ = 'torch.utils'
    parent_mod.config = config_mod
    sys.modules['torch.utils.serialization'] = parent_mod
