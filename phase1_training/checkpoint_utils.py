import pickle
import torch
import io


class _CompatUnpickler(pickle.Unpickler):
    """Redirect old torch.utils.serialization -> torch.serialization."""
    def find_class(self, module, name):
        if module.startswith('torch.utils.serialization'):
            module = 'torch.serialization' + module[len('torch.utils.serialization'):]
        return super().find_class(module, name)


def _compat_load(f, **kwargs):
    return _CompatUnpickler(f, **kwargs).load()


class _CompatPickle:
    Unpickler = _CompatUnpickler
    load = staticmethod(_compat_load)
    loads = staticmethod(pickle.loads)
    PicklingError = pickle.PicklingError
    UnpicklingError = pickle.UnpicklingError
    HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL


def compat_load(path, map_location=None, **kwargs):
    return torch.load(
        path,
        map_location=map_location,
        pickle_module=_CompatPickle,
        weights_only=False,
        **kwargs
    )
