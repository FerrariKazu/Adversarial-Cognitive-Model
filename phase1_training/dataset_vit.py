"""
CIFAR-10 Dataloader for Vision Transformer (ViT)
=================================================

WHY A SEPARATE DATALOADER?
    ViT requires 224×224 input images. CIFAR-10 images are natively 32×32.
    The standard dataset.py (used by ResNet) does NOT resize because ResNet's
    modified conv1 layer (kernel=3, stride=1) is specifically designed for 32×32.

    ViT, however, uses patch_size=16. The number of patches is:
        patches = (img_size / patch_size)² = (224 / 16)² = 14² = 196

    If we fed 32×32 images: (32/16)² = 4 patches — only 4 tokens for
    self-attention, which is far too few for the model to learn spatial
    relationships. That's why we MUST resize to 224.

TRANSFORM DIFFERENCES FROM RESNET (dataset.py):

    ResNet (32×32):                    ViT (224×224):
    ────────────────                   ──────────────
    [no resize]                        Resize(224)          ← NEW
    RandomCrop(32, padding=4)          RandomCrop(224, padding=28)  ← scaled
    RandomHorizontalFlip()             RandomHorizontalFlip()
    ToTensor()                         ToTensor()
    Normalize(...)                     Normalize(...)

WHY PADDING SCALES PROPORTIONALLY:
    ResNet uses padding=4 on a 32×32 image → 4/32 = 12.5% padding ratio.
    To maintain the same augmentation intensity, ViT needs:
        224 × 0.125 = 28 pixels of padding
    This ensures the RandomCrop augmentation has the same relative "wiggle room"
    to shift the image, preventing the model from overfitting to exact
    center-aligned inputs.

NORMALIZATION VALUES:
    We use the same CIFAR-10 normalization as ResNet:
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010)
    These are the channel-wise mean and std of the CIFAR-10 training set.
    Even though we resize the images, the pixel value distribution remains
    the same (resize is spatial, not intensity-based), so the same
    normalization applies.
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from PIL import Image

# =============================================================================
# CIFAR-10 Constants (identical to dataset.py)
# =============================================================================
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# =============================================================================
# ViT Input Resolution
# =============================================================================
# 1. WHAT: The target image size for ViT-Small.
# 2. WHY: ViT-Small was pretrained at 224×224. The positional embeddings
#          encode positions for exactly 196 patches (14×14 grid at patch_size=16).
#          Using a different resolution would require interpolating positional
#          embeddings, which degrades performance.
# 3. OBSERVE: All transforms below use this constant.
# =============================================================================
VIT_IMG_SIZE = 224


# =============================================================================
# HuggingFace Dataset Wrapper (identical to dataset.py)
# =============================================================================
class HFCIFAR10(Dataset):
    """Wraps HuggingFace CIFAR-10 into a PyTorch Dataset."""
    def __init__(self, split='train', transform=None):
        self.dataset = load_dataset('cifar10', split=split)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item['img'].convert('RGB')
        label = item['label']
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloaders_vit(batch_size=64, num_workers=4, data_dir='data', **kwargs):
    """
    Create CIFAR-10 dataloaders with 224×224 resizing for ViT.

    KEY DIFFERENCES FROM get_dataloaders() in dataset.py:
        1. Resize(224) is added FIRST in the transform pipeline.
           This is done BEFORE RandomCrop and ToTensor because:
           - PIL.Image.resize uses bilinear interpolation (smooth upscaling)
           - RandomCrop must operate on the already-resized image
           - ToTensor() converts PIL → Tensor, so resize must happen before

        2. RandomCrop padding is 28 instead of 4 (proportional scaling).

        3. batch_size defaults to 64 instead of 128 because ViT uses ~2×
           more VRAM per image than ResNet (attention is O(n²) in sequence
           length, and 196 patches × 384 dim × 12 heads is expensive).
    """
    # =========================================================================
    # Training Transforms
    # =========================================================================
    # Order matters:
    #   1. Resize(224): Upscale 32×32 PIL image to 224×224 using bilinear interp
    #   2. RandomCrop(224, padding=28): Add 28px black border, then randomly
    #      crop back to 224×224. This shifts the image randomly by up to 28px
    #      in any direction, preventing the model from overfitting to exact
    #      center alignment. 28/224 = 12.5% — same ratio as ResNet's 4/32.
    #   3. RandomHorizontalFlip: 50% chance to mirror the image (standard aug)
    #   4. ToTensor: Convert PIL (H,W,C) uint8 [0,255] → Tensor (C,H,W) float [0,1]
    #   5. Normalize: Shift and scale each channel to zero-mean unit-variance
    #      using CIFAR-10 training set statistics.
    # =========================================================================
    transform_train = transforms.Compose([
        transforms.Resize(VIT_IMG_SIZE),                          # 32×32 → 224×224
        transforms.RandomCrop(VIT_IMG_SIZE, padding=28),          # Augmentation
        transforms.RandomHorizontalFlip(),                        # Standard aug
        transforms.ToTensor(),                                    # PIL → Tensor
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),                             # CIFAR-10 mean
            (0.2023, 0.1994, 0.2010),                             # CIFAR-10 std
        ),
    ])

    # =========================================================================
    # Test Transforms
    # =========================================================================
    # No augmentation — just resize, convert, normalize.
    # The test set must be processed deterministically for fair evaluation.
    # =========================================================================
    transform_test = transforms.Compose([
        transforms.Resize(VIT_IMG_SIZE),                          # 32×32 → 224×224
        transforms.ToTensor(),                                    # PIL → Tensor
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])

    # =========================================================================
    # Dataset Loading (HuggingFace — same as dataset.py)
    # =========================================================================
    trainset = HFCIFAR10(split='train', transform=transform_train)
    testset = HFCIFAR10(split='test', transform=transform_test)

    # =========================================================================
    # DataLoaders
    # =========================================================================
    # NOTE: batch_size=64 is the practical max for ViT-Small on an RTX 4060 (8GB).
    # At batch_size=128, CUDA will OOM because:
    #   - Input: 128 × 3 × 224 × 224 = 128 × 150,528 = ~73MB (vs ~1.2MB at 32×32)
    #   - Attention: 128 × 12 heads × 197 × 197 × 4 bytes ≈ 237MB
    #   - Total with gradients ≈ 6-7GB (barely fits in 8GB)
    # =========================================================================
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True)

    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True)

    return trainloader, testloader
