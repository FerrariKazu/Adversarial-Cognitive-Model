# pip install git+https://github.com/dicarlolab/CORnet.git
# CORnet-S
# cornet.cornet_s with pretrained ImageNet weights
# Requires transforms.Resize(224) in dataloader.
"""
What CORnet-S is:
CORnet-S is a recurrent neural network model of the primate visual cortex.

Why recurrence is scientifically important:
Recurrence is scientifically important for our study because it mimics the core functional 
architecture of the primate ventral visual stream. By comparing it to purely feedforward 
architectures like ResNet or EfficientNet, we can analyze the critical role of recurrent 
feedback processing in maintaining robustness against adversarial perturbations.

The four areas:
It explicitly models four distinct anatomical areas of the visual pathway: V1, V2, V4, and IT.
"""
import torch.nn as nn
import cornet

class CIFARCORnet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.model = cornet.cornet_s(pretrained=True)
        
        # Replace decoder.linear with nn.Linear(512, 10) for CIFAR-10
        self.model.decoder.linear = nn.Linear(512, num_classes)
        
    def forward(self, x):
        return self.model(x)
