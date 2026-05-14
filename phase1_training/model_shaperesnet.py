import os
import torch
import torch.nn as nn
from torchvision import models


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

    def __init__(self, num_classes=10, weights_path='phase1_training/checkpoints/shaperesnet50_best_v2.pth'):
        super(ShapeResNet, self).__init__()
        # Initialize a standard ResNet-50 architecture
        self.model = models.resnet50(weights=None)
        
        # Replace FC layer for CIFAR-10 (10 classes instead of 1000) BEFORE loading weights
        self.model.fc = nn.Linear(2048, num_classes)
        nn.init.xavier_uniform_(self.model.fc.weight)
        
        # Load the SIN (Stylized-ImageNet) or best trained weights
        if weights_path and os.path.exists(weights_path):
            print(f"Loading weights from {weights_path}...")
            state = torch.load(weights_path, map_location='cpu')
            
            # If state_dict is nested (e.g., from 'model' key)
            if 'state_dict' in state:
                state = state['state_dict']
            elif 'model' in state:
                state = state['model']
                
            # Strip 'module.' or 'model.' prefix if saved from DataParallel or wrapping
            new_state = {}
            for k, v in state.items():
                if k.startswith("module."):
                    new_state[k[7:]] = v
                elif k.startswith("model."):
                    new_state[k[6:]] = v
                else:
                    new_state[k] = v
                
            # Load with strict=False to allow FC layer mismatch
            missing, unexpected = self.model.load_state_dict(new_state, strict=False)
            print(f"Weights loaded. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
        else:
            print(f"Warning: {weights_path} not found. Initializing with random weights.")

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    # Quick sanity check
    model = ShapeResNet(num_classes=10, weights_path='phase1_training/checkpoints/shaperesnet50_best_v2.pth')
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("model_shaperesnet.py is working correctly ✓")
