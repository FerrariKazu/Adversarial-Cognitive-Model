import pickle
import torch

class _CompatUnpickler(pickle.Unpickler):
    """Redirect old torch.utils.serialization → torch.serialization."""
    def find_class(self, module, name):
        if module.startswith('torch.utils.serialization'):
            module = 'torch.serialization' + module[len('torch.utils.serialization'):]
        return super().find_class(module, name)

class _CompatPickle:
    Unpickler = _CompatUnpickler
    load = pickle.load
    loads = pickle.loads
    PicklingError = pickle.PicklingError
    UnpicklingError = pickle.UnpicklingError
    HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL


def compat_load(path, map_location=None, **kwargs):
    """Load a PyTorch checkpoint with old-format pickle compatibility.

    Uses :func:`torch.load` with a custom *pickle_module* that redirects
    the removed ``torch.utils.serialization`` to ``torch.serialization``.
    """
    return torch.load(
        path,
        map_location=map_location,
        pickle_module=_CompatPickle,
        weights_only=False,
        **kwargs
    )
