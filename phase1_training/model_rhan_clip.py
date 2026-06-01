import os
import sys
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan import RHAN

class RHANWithCLIP(nn.Module):
    """
    RHAN + CLIP Semantic Grounding (Trial 1)
    
    Motivation: RHAN-adv achieves d'=0.076 implementing 3/4 principles:
    1. Local convolutional smoothing ✓
    2. Global transformer attention ✓  
    3. Recurrent feedback ✓
    4. Semantic grounding ← THIS TRIAL ADDS THIS
    
    Instead of learning "this texture pattern = cat",
    the model learns "this image is semantically close to the 
    concept of cat" using CLIP text embeddings as anchors.
    
    Hypothesis: language-grounded representations are inherently 
    less texture-dependent and more robust to adversarial attacks
    that corrupt texture statistics.
    """
    
    CIFAR10_PROMPTS = [
        "a photo of an airplane",
        "a photo of an automobile",  
        "a photo of a bird",
        "a photo of a cat",
        "a photo of a deer",
        "a photo of a dog",
        "a photo of a frog",
        "a photo of a horse",
        "a photo of a ship",
        "a photo of a truck",
    ]
    
    def __init__(self, rhan_checkpoint_path, device='cuda'):
        super().__init__()
        
        # Load pretrained RHAN-adv as the visual backbone
        # We keep all of RHAN's architecture: stem, transformer, feedback
        # We ONLY replace the final classification head
        self.rhan_backbone = RHAN()
        state = torch.load(rhan_checkpoint_path, map_location=device)
        
        # Load all weights except the classifier head (head in RHAN)
        backbone_state = {k: v for k, v in state.items() 
                         if not (k.startswith('classifier') or k.startswith('head'))}
        
        self.rhan_backbone.load_state_dict(backbone_state, strict=False)
        print(f"[RHANWithCLIP] Successfully loaded RHAN backbone from {rhan_checkpoint_path}")
        
        # Freeze CLIP — we only use it for text embeddings
        self.clip_model, _ = clip.load('ViT-B/32', device=device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        print("[RHANWithCLIP] Successfully loaded and froze CLIP ViT-B/32 model")
        
        # Projection head: map RHAN 512-dim features to CLIP 512-dim space
        self.projection = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
        )
        
        # Learnable temperature for contrastive scaling
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052)  # log(100)
        
        # Pre-compute and cache text embeddings
        self.register_buffer('text_features', 
                            self._encode_text_prompts(device))
    
    def _encode_text_prompts(self, device):
        tokens = clip.tokenize(self.CIFAR10_PROMPTS).to(device)
        with torch.no_grad():
            text_feat = self.clip_model.encode_text(tokens)
            text_feat = F.normalize(text_feat.float(), dim=-1)
        return text_feat  # (10, 512)
    
    def forward(self, x):
        # Get projected features directly
        projected = self.get_feature_vector(x)
        
        # Cosine similarity with text embeddings
        scale = self.logit_scale.exp().clamp(max=100)
        logits = scale * projected @ self.text_features.T  # (B, 10)
        return logits
    
    def get_feature_vector(self, x):
        visual_features = self.rhan_backbone.get_feature_vector(x)
        projected = self.projection(visual_features)
        return F.normalize(projected, dim=-1)
