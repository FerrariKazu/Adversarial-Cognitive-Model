# BagNet-33 — Owner: Eyad
# Classifies using only 33x33 local patches — zero global context.
# Install: pip install git+https://github.com/wielandbrendel/bag-of-local-features-models.git
# Requires transforms.Resize(64) minimum in dataloader.
# Target clean accuracy: 75-82% — lower accuracy is EXPECTED and correct.
import torch.nn as nn
import bagnets.pytorchnet as bn

class CIFARBagNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = bn.bagnet33(pretrained=False)
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.model(x)