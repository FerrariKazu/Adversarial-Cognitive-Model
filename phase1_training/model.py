import torch
import torch.nn as nn
import torchvision.models as models

# =============================================================================
# CIFAR-10 ResNet-18 Wrapper
# =============================================================================
# 1. WHAT: This class wraps the standard torchvision ResNet-18 model.
# 2. WHY: We need to adapt the standard ImageNet ResNet-18 to handle CIFAR-10's 
#         32x32 pixel images and 10 classes instead of ImageNet's 224x224 images
#         and 1000 classes. We start with ImageNet weights to speed up convergence.
# 3. OBSERVE: When initialized, it downloads the pretrained weights (if not cached).
# =============================================================================
class CIFARResNet(nn.Module):
    def __init__(self):
        super(CIFARResNet, self).__init__()
        # Load the pretrained ResNet-18 model
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # ---------------------------------------------------------------------
        # Modify the first Convolutional Layer
        # ---------------------------------------------------------------------
        # 1. WHAT: Replacing the original 7x7 conv (stride=2, padding=3) with a 
        #          3x3 conv (stride=1, padding=1).
        # 2. WHY: CIFAR-10 images are very small (32x32). The original 7x7 conv 
        #         with stride 2 downsamples the image to 16x16 immediately, destroying
        #         too much spatial information. A 3x3 conv with stride 1 preserves 
        #         the 32x32 resolution.
        # 3. OBSERVE: The first layer's output shape will be exactly 32x32.
        # ---------------------------------------------------------------------
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        
        # ---------------------------------------------------------------------
        # Remove the MaxPool Layer
        # ---------------------------------------------------------------------
        # 1. WHAT: Replacing the MaxPool2d layer with an Identity layer (no-op).
        # 2. WHY: The original MaxPool downsamples the image again by a factor of 2.
        #         For 32x32 images, two immediate downsamplings leave us with too 
        #         little spatial dimension for the rest of the deep network. Identity
        #         keeps the resolution at 32x32 passing into the residual blocks.
        # 3. OBSERVE: The network parameter count stays the same, but the activation
        #         maps remain larger, consuming more VRAM than default ResNet18.
        # ---------------------------------------------------------------------
        self.resnet.maxpool = nn.Identity()
        
        # ---------------------------------------------------------------------
        # Modify the Fully Connected (FC) Layer
        # ---------------------------------------------------------------------
        # 1. WHAT: Replacing the final Linear layer to output 10 logits instead of 1000.
        # 2. WHY: CIFAR-10 has exactly 10 classes (airplane, automobile, bird, etc.).
        # 3. OBSERVE: The final output tensor shape will be [batch_size, 10].
        # ---------------------------------------------------------------------
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        # 1. WHAT: The standard forward pass through the modified network.
        # 2. WHY: Required by PyTorch to define the computational graph.
        # 3. OBSERVE: Returns raw unnormalized logits (not softmax probabilities).
        return self.resnet(x)

    def get_feature_vector(self, x):
        # ---------------------------------------------------------------------
        # Feature Extraction Method
        # ---------------------------------------------------------------------
        # 1. WHAT: Passes the input through all layers EXCEPT the final FC layer.
        # 2. WHY: For downstream tasks like Grad-CAM, clustering, or analyzing 
        #         the latent space under adversarial attacks, we need the 512-dim 
        #         embeddings before they are collapsed into 10 class logits.
        # 3. OBSERVE: Returns a tensor of shape [batch_size, 512].
        # ---------------------------------------------------------------------
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x
