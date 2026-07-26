import torch
import torch.nn.functional as F
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.gradients = None
        self.activations = None
        self._target_layer = None
        self._register_hooks(target_layer)

    def _resolve_model(self):
        m = self.model
        if hasattr(m, 'module'):
            m = m.module
        for attr in ['model', 'backbone', 'encoder', 'net']:
            if hasattr(m, attr):
                candidate = getattr(m, attr)
                if hasattr(candidate, 'named_modules'):
                    m = candidate
                    break
        return m

    def _register_hooks(self, target_layer):
        m = self._resolve_model()

        def forward_hook(module, input, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            go = grad_output
            if isinstance(go, (tuple, list)):
                go = go[0]
            if isinstance(go, (tuple, list)):
                go = go[0]
            self.gradients = go.detach()

        target = self._find_target_layer(m, target_layer)
        if target is not None:
            self._target_layer = target
            target.register_forward_hook(forward_hook)
            target.register_full_backward_hook(backward_hook)

    def _find_target_layer(self, root, target_layer=None):
        if target_layer is not None:
            for name, module in root.named_modules():
                if name == target_layer:
                    return module
        for name, module in root.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                return module
        for name, module in root.named_modules():
            if isinstance(module, (torch.nn.Linear,)):
                return module
        return None

    def generate(self, image_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(image_tensor.unsqueeze(0))
        if isinstance(output, (tuple, list)):
            output = output[0]
        if class_idx is None:
            class_idx = output.argmax(1).item()
        target = output[0, class_idx]
        target.backward(retain_graph=True)
        if self.gradients is None or self.activations is None:
            return np.zeros((image_tensor.shape[1], image_tensor.shape[2]), dtype=np.float32)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze(0)
        cam = F.relu(cam)
        h, w = image_tensor.shape[1:]
        if cam.ndim == 0:
            cam = cam.unsqueeze(0).unsqueeze(0)
        if cam.ndim == 1:
            cam = cam.unsqueeze(0).unsqueeze(0)
        if cam.ndim == 2:
            cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(
            cam.unsqueeze(0) if cam.ndim == 3 else cam,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu().numpy()
        cam = np.maximum(cam, 0)
        cmin, cmax = cam.min(), cam.max()
        if cmax > cmin:
            cam = (cam - cmin) / (cmax - cmin)
        return cam


def extract_attention_maps(model, image_tensor):
    attention_maps = {}
    hooks = []

    def hook_fn(name):
        def fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            if output.ndim == 3:
                attention_maps[name] = output.detach()
            elif output.ndim == 4 and output.shape[1] > 1:
                attention_maps[name] = output.detach()
        return fn

    for name, module in model.named_modules():
        if "attn" in name.lower() or "attention" in name.lower():
            hooks.append(module.register_forward_hook(hook_fn(name)))

    with torch.no_grad():
        _ = model(image_tensor.unsqueeze(0))

    for h in hooks:
        h.remove()

    return attention_maps


def compute_representation(model, image_tensor, layer_name=None):
    representation = {}
    hooks = []

    if layer_name is not None:
        def hook_fn(name):
            def fn(module, input, output):
                if isinstance(output, (tuple, list)):
                    representation[name] = output[0].detach()
                else:
                    representation[name] = output.detach()
            return fn

        for name, module in model.named_modules():
            if layer_name in name:
                hooks.append(module.register_forward_hook(hook_fn(name)))
                break
    else:
        def hook_fn(name):
            def fn(module, input, output):
                if isinstance(output, torch.Tensor) and output.ndim >= 2:
                    representation[name] = output.detach()
            return fn

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and module.out_features > 100:
                hooks.append(module.register_forward_hook(hook_fn(name)))
                if len(representation) >= 1:
                    break

    with torch.no_grad():
        _ = model(image_tensor.unsqueeze(0))

    for h in hooks:
        h.remove()

    return representation
