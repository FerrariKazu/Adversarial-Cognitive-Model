import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_rhan import ConvStem, PatchTokeniser, GlobalAttention

class AdaptiveRHAN(nn.Module):
    """
    RHAN with Adaptive Computation Time (Trial 2)
    
    Motivation: Human visual cortex uses more recurrent processing 
    cycles for degraded inputs (measurable as longer reaction times).
    Current RHAN uses fixed 2 feedback steps regardless of difficulty.
    
    This model learns to decide HOW MANY recurrent steps to use:
    - Clean images (easy): 1-2 steps, fast inference
    - Low epsilon (slightly degraded): 2-3 steps
    - High epsilon (heavily degraded): 4-6 steps
    
    The halting mechanism:
    At each step t, the model computes a halting probability h_t.
    It keeps running recurrent feedback until:
      sum(h_1 + h_2 + ... + h_t) > 1 - epsilon_halt
    Or until max_steps is reached.
    
    This is inspired by Graves (2016) Adaptive Computation Time.
    """
    
    def __init__(self, num_classes=10, embed_dim=512, num_heads=8,
                 ff_dim=2048, dropout=0.1, num_transformer_layers=3,
                 max_steps=6, epsilon_halt=0.01):
        super().__init__()
        self.max_steps = max_steps
        self.epsilon_halt = epsilon_halt
        
        # Stage 1: Convolutional stem — local smoothing
        self.stem = ConvStem()
        
        # Stage 2: Patch tokenisation
        self.tokeniser = PatchTokeniser(embed_dim=embed_dim, num_patches=64)
        
        # Stage 3: Global transformer attention
        self.transformer = GlobalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            num_layers=num_transformer_layers,
        )
        
        # Stage 4: Recurrent feedback convolution & gate
        self.feedback_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        
        self.gate_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=True)
        
        # Stage 5: Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes),
        )
        
        # NEW: halting network
        # Takes current spatial feature map, outputs halt probability
        self.halting_network = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # global average pool spatial features
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # output in [0, 1]
        )
        
    def tokenise(self, stem_features):
        return self.tokeniser(stem_features)
        
    def tokens_to_spatial(self, tokens):
        spatial_tokens = tokens[:, 1:, :]  # Remove CLS token
        B, N, C = spatial_tokens.shape
        return spatial_tokens.transpose(1, 2).reshape(B, C, 8, 8)
        
    def load_from_rhan_adv(self, rhan_checkpoint_path, device='cuda'):
        """
        Maps standard RHAN checkpoints into AdaptiveRHAN.
        """
        state = torch.load(rhan_checkpoint_path, map_location=device)
        
        adapted_state = {}
        for k, v in state.items():
            if k.startswith('feedback.feedback_conv.'):
                adapted_state[k.replace('feedback.feedback_conv.', 'feedback_conv.')] = v
            elif k.startswith('feedback.gate.0.'):
                adapted_state[k.replace('feedback.gate.0.', 'gate_conv.')] = v
            elif k.startswith('head.'):
                adapted_state[k.replace('head.', 'classifier.')] = v
            else:
                adapted_state[k] = v
                
        # Load the mapped weights (halting_network will be initialized randomly)
        msg = self.load_state_dict(adapted_state, strict=False)
        print(f"[AdaptiveRHAN] Mapped checkpoint weights loaded with message: {msg}")
        
    def forward(self, x):
        # Stage 1: Conv stem
        stem_features = self.stem(x)  # (B, 512, 8, 8)
        
        # Stage 2: Tokenise
        tokens = self.tokenise(stem_features)  # (B, 65, 512)
        
        # Stage 3: Adaptive recurrent feedback
        B = x.size(0)
        cumulative_halt = torch.zeros(B, device=x.device)
        remainder = torch.ones(B, device=x.device)
        weighted_output = torch.zeros_like(tokens)
        steps_used = torch.zeros(B, device=x.device)
        
        for t in range(self.max_steps):
            # Transformer forward pass
            transformer_out = self.transformer(tokens)
            
            # Compute halting probability for this step
            spatial = self.tokens_to_spatial(transformer_out)
            h_t = self.halting_network(spatial).squeeze(-1)  # (B,)
            
            # Adaptive halting logic (Graves 2016)
            # Still running: cumulative_halt < 1 - epsilon_halt
            still_running = (cumulative_halt < 1 - self.epsilon_halt).float()
            
            # New cumulative halt
            new_cumulative = cumulative_halt + h_t * still_running
            
            # Weight for this step's output
            # If this is the last step, use remainder
            exceeds = (new_cumulative > 1 - self.epsilon_halt).float()
            weight = (exceeds * remainder + (1 - exceeds) * h_t) * still_running
            
            # Accumulate weighted output
            weighted_output += weight.view(B, 1, 1) * transformer_out
            
            # Update state
            remainder -= weight * still_running
            cumulative_halt = new_cumulative
            steps_used += still_running
            
            # Recurrent feedback for next step
            spatial = self.tokens_to_spatial(transformer_out)
            feedback = self.feedback_conv(spatial)
            gate = torch.sigmoid(self.gate_conv(spatial))
            stem_features = stem_features + gate * feedback
            tokens = self.tokenise(stem_features)
            
            # Early exit if all samples in batch have halted
            if still_running.sum() == 0:
                break
        
        # Classify from weighted accumulated output
        cls_token = weighted_output[:, 0, :]  # (B, 512)
        logits = self.classifier(cls_token)
        
        return logits, steps_used, cumulative_halt
        
    def forward_eval(self, x):
        """Returns logits, steps_used for analysis"""
        logits, steps_used, _ = self.forward(x)
        return logits, steps_used
