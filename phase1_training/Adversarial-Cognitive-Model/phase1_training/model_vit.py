# ViT-Small — Owner: Mina (FerrariKazu)
# Vision Transformer, patch_size=16, img_size=224
# Uses timm library. Requires transforms.Resize(224) in dataloader.
# Target clean accuracy: 88-92%
"""
ViT-Small (Vision Transformer) for CIFAR-10
============================================

WHAT IS A VISION TRANSFORMER?
    Unlike ResNet (which slides convolutional filters across the image in a
    fixed local pattern), ViT treats the image as a sequence of patches and
    processes them with self-attention — the same mechanism that powers GPT
    and BERT in language models.

    Steps:
    1. The 224×224 image is divided into a grid of 16×16-pixel patches.
       224 ÷ 16 = 14 patches per row → 14 × 14 = 196 patches total.
    2. Each patch is flattened and linearly projected into a 384-dim embedding.
    3. A special [CLS] token (class token) is prepended → 197 tokens total.
    4. Positional embeddings are added so the model knows WHERE each patch was.
    5. The 197 tokens pass through 12 transformer encoder layers (self-attention
       + feedforward network).
    6. The output [CLS] token is fed through a linear classifier → 10 logits.

WHY patch_size=16?
    16 divides 224 evenly: 224 / 16 = 14 patches per axis.
    16 does NOT divide 32 (CIFAR-10's native resolution): 32 / 16 = 2, giving
    only 2×2 = 4 patches — far too few for self-attention to learn meaningful
    relationships. That's why we MUST resize CIFAR-10 images to 224×224.

    More patches = more tokens = richer attention patterns = better accuracy,
    but also = quadratic memory cost (O(n²) attention). 196 tokens is the
    sweet spot for ViT-Small.

WHY PRETRAINED MATTERS:
    ViT has NO inductive bias for images. Unlike CNNs, which have built-in
    locality (convolution) and translation equivariance, a ViT starts with
    NO knowledge about spatial structure. Training from scratch on a small
    dataset like CIFAR-10 (50k images) leads to terrible results (~30-40%
    accuracy). ImageNet pretraining teaches the model spatial priors first,
    then we fine-tune on CIFAR-10.

    This is the fundamental difference from ResNet:
    - ResNet can train from scratch on CIFAR-10 because convolutions encode
      spatial priors by design.
    - ViT must learn spatial priors from data, so it needs massive pretraining.

WHY ViT FOR THIS PROJECT:
    ViT processes the image GLOBALLY via self-attention — every patch can
    attend to every other patch. This is the closest architecture to
    "global shape processing." Our hypothesis predicts that ViT should be
    the MOST adversarially robust model (closest to human), because:
    - Adversarial perturbations target local texture features
    - ViT's attention can integrate information across the entire image
    - This is analogous to the recurrent feedback loops in the human visual
      cortex that allow humans to reconstruct global shape from noisy inputs

    If ViT is indeed more robust than ResNet, it supports the theory that
    GLOBAL processing (not just architecture type) drives robustness.

CLS TOKEN EXPLAINED:
    The [CLS] token is a learnable 384-dim vector prepended to the patch
    sequence. It has no spatial location — it doesn't correspond to any
    patch of the image. Instead, it serves as a "collector" that aggregates
    information from ALL patches through self-attention.

    After 12 transformer layers, the CLS token has attended to every patch
    and accumulated a holistic representation of the entire image. This is
    why it's used for classification — it's the model's global summary.

    For our project, get_feature_vector() returns this CLS embedding,
    which we use for:
    - Grad-CAM-like attention analysis
    - Latent space clustering under adversarial attack
    - Comparing how the global representation degrades vs ResNet's local one
"""

import torch
import torch.nn as nn
import timm


class CIFARViT(nn.Module):
    """
    Vision Transformer (ViT-Small) adapted for CIFAR-10 classification.

    Architecture:
        - Backbone: ViT-Small (patch_size=16, embed_dim=384, 12 heads, 12 layers)
        - Input: 224×224×3 images (CIFAR-10 resized from 32×32)
        - Output: 10 logits (one per CIFAR-10 class)
        - Parameters: ~22M (vs ResNet-18's ~11M)

    Memory note:
        ViT-Small uses ~2× the parameters of ResNet-18 and the attention
        mechanism requires O(n²) memory where n=196 patches. On an RTX 4060
        with 8GB VRAM, batch_size=64 is the practical maximum.
    """

    def __init__(self):
        super(CIFARViT, self).__init__()

        # =====================================================================
        # Load Pretrained ViT-Small from timm
        # =====================================================================
        # 1. WHAT: Loading a ViT-Small model pretrained on ImageNet-1K.
        # 2. WHY: ViT has no inductive bias — without pretraining, it cannot
        #         learn spatial structure from CIFAR-10's 50k images alone.
        #         ImageNet pretraining teaches patch relationships, positional
        #         encoding meaning, and basic visual features BEFORE we fine-tune.
        # 3. OBSERVE: The model downloads ~87MB of pretrained weights on first run.
        #
        # KEY PARAMETERS:
        #   - 'vit_small_patch16_224': Small variant, 16×16 patches, 224px input
        #   - pretrained=True: Load ImageNet-1K weights (CRITICAL for convergence)
        #   - num_classes=10: Replace the ImageNet 1000-class head with a 10-class
        #     linear layer for CIFAR-10
        #   - img_size=224: Explicitly set input resolution (must match dataloader)
        # =====================================================================
        self.vit = timm.create_model(
            'vit_small_patch16_224',
            pretrained=True,
            num_classes=10,
            img_size=224,
        )

    def forward(self, x):
        """
        Standard forward pass.

        1. WHAT: Passes the input through all ViT layers and returns class logits.
        2. WHY: Required by PyTorch to define the computational graph.
        3. OBSERVE: Returns raw unnormalized logits (shape [batch_size, 10]),
                    same as CIFARResNet.forward().

        Input:  x of shape [B, 3, 224, 224]
        Output: logits of shape [B, 10]
        """
        return self.vit(x)

    def get_feature_vector(self, x):
        """
        Extract the [CLS] token embedding BEFORE the classification head.

        1. WHAT: Runs the input through all transformer layers and returns the
                 384-dimensional CLS token output — the model's global summary
                 of the entire image.
        2. WHY: For Phase 4 analysis (Grad-CAM, latent space clustering,
                 representation degradation under attack), we need the embedding
                 space — not the final logits. This is analogous to
                 CIFARResNet.get_feature_vector() which returns the 512-dim
                 vector from the avgpool layer.
        3. OBSERVE: Returns a tensor of shape [batch_size, 384].
                    (ResNet returns [batch_size, 512] — different dim but same role.)

        WHY 384 DIMENSIONS?
            ViT-Small uses embed_dim=384. This is the dimensionality of each
            token (including CLS) throughout all transformer layers. It's smaller
            than ViT-Base (768) to reduce computation, but large enough for
            CIFAR-10's 10-class problem.
        """
        # forward_features() runs the full ViT encoder (patch embed → position
        # embed → transformer blocks → layer norm) but STOPS before the
        # classification head. Returns the CLS token embedding.
        return self.vit.forward_features(x)
