# EfficientNet-B0 — Owner: Youssef
# torchvision.models.efficientnet_b0 with pretrained ImageNet weights
# Requires transforms.Resize(224) in dataloader.
# Target clean accuracy: 90-93%
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class CIFAREfficientNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)