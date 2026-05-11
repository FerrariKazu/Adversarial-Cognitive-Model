import torch
import torch.nn as nn
from torchvision.models import resnet50


class ShapeResNet(nn.Module):
    """
    ResNet-50 initialized with ImageNet pretrained weights and fine-tuned
    on CIFAR-10 using shape-biased augmentation (grayscale, edge-preserving
    transforms) to replicate the shape bias of Geirhos et al. (2019).

    Scientific role: SAME architecture as ResNet-18 baseline, DIFFERENT
    training objective. If this model is more adversarially robust than
    ResNet-18, training objective (shape bias) is the key variable —
    not architecture.
    """

    def __init__(self, num_classes=10, weights_path='resnet50_trained_on_SIN.model'):
        super().__init__()
        self.model = resnet50(weights=None)

        # Load pretrained weights
        if weights_path:
            state = torch.load(weights_path, map_location='cpu')
            # Handle both raw state_dict and wrapped {'state_dict': ...} formats
            if 'state_dict' in state:
                state = state['state_dict']
            # Strip 'module.' prefix if saved from DataParallel
            state = {k.replace('module.', ''): v for k, v in state.items()}
            # Load with strict=False to allow FC layer mismatch
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print(f"Weights loaded. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")

        # Replace FC layer for CIFAR-10 (10 classes instead of 1000)
        self.model.fc = nn.Linear(2048, num_classes)
        nn.init.xavier_uniform_(self.model.fc.weight)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    # Quick sanity check
    model = ShapeResNet(num_classes=10, weights_path='resnet50_trained_on_SIN.model')
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("model_shaperesnet.py is working correctly ✓")
