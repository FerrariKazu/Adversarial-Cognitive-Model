# BagNet-33 — Owner: Eyad
# Classifies using only 33x33 local patches — zero global context.
# Install: pip install git+https://github.com/wielandbrendel/bag-of-local-features-models.git
# Requires transforms.Resize(64) minimum in dataloader.
# Target clean accuracy: 75-82% — lower accuracy is EXPECTED and correct.
"""
What BagNet-33 is:
BagNet-33 is a bag-of-local-features model that classifies images using only
33x33 pixel patches with zero global spatial context. It processes each local
patch independently and aggregates evidence via average pooling.

Why BagNet is scientifically important for our study:
BagNet represents the extreme local-processing baseline on our model spectrum.
By comparing its adversarial robustness to architectures with increasing global
integration (ResNet → ViT → Human), we can quantify how much spatial context
contributes to perceptual robustness. BagNet's expected fragility under PGD
provides the lower bound for our spectrum.
"""
import torch.nn as nn
import bagnets.pytorchnet as bagnets

class CIFARBagNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = bagnets.bagnet33(pretrained=True)
        # Replace final FC layer for CIFAR-10
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
