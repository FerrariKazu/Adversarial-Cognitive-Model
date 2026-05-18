#!/usr/bin/env python3
"""
Adversarial Cognition Divergence — Live Interactive Explorer
============================================================
Developed by Antigravity, 2026.
Serves a self-contained local web server demonstrating our RHAN-adv model,
dynamic PGD-20 attacks, prediction shifts, and recurrent spatial gating maps.
"""

import os
import sys
import io
import json
import base64
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import Image
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver

# Set path priorities
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phase1_training'))

from phase1_training.model_rhan import RHAN
from phase1_training.model import CIFARResNet
from phase1_training.dataset import CLASSES
from phase2_attacks.pgd import pgd_attack

# Global state
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEMPLATES = {}
MODELS = {}

# Constants
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

def denormalize(img_tensor):
    """(3, H, W) normalized -> (H, W, 3) in [0, 1]"""
    img = img_tensor.cpu() * STD + MEAN
    img = torch.clamp(img, 0, 1)
    return img.permute(1, 2, 0).numpy()

def tensor_to_base64(img_tensor):
    """Convert (3, H, W) normalized tensor to base64 PNG URL."""
    img_np = denormalize(img_tensor)
    pil_img = Image.fromarray((img_np * 255).astype(np.uint8))
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"

def gate_to_base64_overlay(img_tensor, gate_map):
    """Overlay gate_map (8, 8) onto img_tensor (3, 32, 32) using viridis colormap."""
    img_np = denormalize(img_tensor)
    g_np = gate_map.detach().cpu().numpy()
    
    # Scale to [0, 1]
    g_min, g_max = g_np.min(), g_np.max()
    g_np = (g_np - g_min) / (g_max - g_min + 1e-12)
    
    # Bilinear upscale to 32x32
    g_img = Image.fromarray((g_np * 255).astype(np.uint8)).resize((32, 32), Image.BILINEAR)
    g_resized = np.array(g_img) / 255.0
    
    # Map to viridis colors
    cmap = plt.get_cmap('viridis')
    rgba_heatmap = cmap(g_resized)
    
    # Alpha blend
    alpha = 0.55
    blended = (1 - alpha) * img_np + alpha * rgba_heatmap[:, :, :3]
    blended = np.clip(blended, 0, 1)
    
    pil_blended = Image.fromarray((blended * 255).astype(np.uint8))
    buffered = io.BytesIO()
    pil_blended.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"

def get_model(model_name):
    """Get or load a model dynamically to optimize RAM."""
    if model_name in MODELS:
        return MODELS[model_name]
        
    print(f"Loading {model_name} on device {DEVICE}...")
    if model_name == 'rhan-adv':
        model = RHAN(num_classes=10, head_type='linear').to(DEVICE)
        ckpt = os.path.join(os.path.dirname(__file__), 'checkpoints', 'rhan_adv_best.pth')
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    elif model_name == 'rhan-clean':
        model = RHAN(num_classes=10, head_type='linear').to(DEVICE)
        ckpt = os.path.join(os.path.dirname(__file__), 'checkpoints', 'rhan_v2_best.pth')
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    elif model_name == 'resnet-18':
        model = CIFARResNet().to(DEVICE)
        ckpt = os.path.join(os.path.dirname(__file__), 'phase1_training', 'checkpoints', 'best.pth')
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    elif model_name == 'vit':
        from phase1_training.model_vit import CIFARViT
        model = CIFARViT().to(DEVICE)
        ckpt = os.path.join(os.path.dirname(__file__), 'phase1_training', 'checkpoints', 'vit_small_best.pth')
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    elif model_name == 'efficientnet':
        from phase1_training.model_efficientnet import CIFAREfficientNet
        model = CIFAREfficientNet().to(DEVICE)
        ckpt = os.path.join(os.path.dirname(__file__), 'phase1_training', 'checkpoints', 'efficientnet_best.pth')
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    elif model_name == 'shaperesnet':
        from phase1_training.model_shaperesnet import ShapeResNet
        ckpt = os.path.join(os.path.dirname(__file__), 'phase1_training', 'checkpoints', 'shaperesnet50_best_v2.pth')
        model = ShapeResNet(num_classes=10, weights_path=ckpt).to(DEVICE)
    elif model_name == 'cornets':
        from phase1_training.model_cornets import CIFARCORnet
        model = CIFARCORnet().to(DEVICE)
        ckpt = os.path.join(os.path.dirname(__file__), 'phase1_training', 'checkpoints', 'cornets_best.pth')
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    elif model_name == 'bagnet':
        from phase1_training.model_bagnet import CIFARBagNet
        model = CIFARBagNet().to(DEVICE)
        ckpt = os.path.join(os.path.dirname(__file__), 'phase1_training', 'checkpoints', 'bagnet_best.pth')
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    MODELS[model_name] = model
    return model

def forward_with_gates(model, x):
    """Forward pass extracting intermediate recurrent feedback gating maps."""
    # Stage 1: Conv Stem
    stem_features = model.stem(x)  # (B, 512, 8, 8)
    
    # Stage 2: Tokeniser
    tokens = model.tokeniser(stem_features)  # (B, 65, 512)
    
    # Stage 3: Transformer
    attended = model.transformer(tokens)  # (B, 65, 512)
    
    # Stage 4: Recurrent Feedback
    gates = []
    current = attended
    for t in range(model.feedback.num_recurrent_steps):
        cls_token = current[:, :1, :]
        spatial = model.feedback.tokens_to_spatial(current)
        feedback = model.feedback.feedback_conv(spatial)
        g = model.feedback.gate(feedback)  # (B, 512, 8, 8)
        gates.append(g.detach().cpu())
        
        modulated = stem_features + g * feedback
        modulated_tokens = model.feedback.spatial_to_tokens(modulated, cls_token)
        current = model.transformer(modulated_tokens)
        
    refined = current
    cls_output = refined[:, 0, :]
    logits = model.head(cls_output)
    
    return logits, gates

def compute_entropy(gate_map):
    """Compute Shannon entropy of the spatial gating map distribution."""
    p = gate_map.flatten()
    p = p / (p.sum() + 1e-12)
    entropy = -torch.sum(p * torch.log(p + 1e-12)).item()
    return entropy

def load_templates():
    """Load cached template images directly from CIFAR-10 test split."""
    global TEMPLATES
    print("Pre-fetching standard template images from CIFAR-10 Split...")
    from phase1_training.dataset import get_dataloaders
    _, testloader = get_dataloaders(batch_size=128, num_workers=0, model_name='resnet')
    
    for images, labels in testloader:
        for img, lbl in zip(images, labels):
            c = lbl.item()
            if c not in TEMPLATES:
                TEMPLATES[c] = (img, c)
            if len(TEMPLATES) == 10:
                break
        if len(TEMPLATES) == 10:
            break
    print(f"Loaded {len(TEMPLATES)} templates successfully.")

# =============================================================================
# HTML FRONTEND (Sleek Glassmorphic Dark Mode Dashboard)
# =============================================================================
FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Adversarial Cognition Divergence — Live Explorer</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #070a13;
            --bg-card: rgba(18, 25, 41, 0.7);
            --bg-card-hover: rgba(26, 36, 59, 0.85);
            --border-color: rgba(255, 255, 255, 0.08);
            --color-primary: #7c3aed;
            --color-primary-glow: rgba(124, 58, 237, 0.35);
            --color-success: #10b981;
            --color-danger: #ef4444;
            --color-text-main: #f3f4f6;
            --color-text-sub: #9ca3af;
            --color-text-muted: #6b7280;
            --font-sans: 'Outfit', sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
            --transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        * {
            margin: 0; padding: 0; box-sizing: border-box;
        }
        
        body {
            background-color: var(--bg-base);
            color: var(--color-text-main);
            font-family: var(--font-sans);
            min-height: 100vh;
            padding: 2.5rem 1.5rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(124, 58, 237, 0.06) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(16, 185, 129, 0.04) 0%, transparent 40%);
        }
        
        .container {
            max-width: 1200px;
            width: 100%;
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }
        
        header {
            text-align: center;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        
        header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #fff 30%, var(--color-primary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
        }
        
        header p {
            color: var(--color-text-sub);
            font-size: 1.1rem;
            font-weight: 300;
        }
        
        .grid-layout {
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 2rem;
        }
        
        @media (max-width: 900px) {
            .grid-layout {
                grid-template-columns: 1fr;
            }
        }
        
        .glass-panel {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.75rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            transition: var(--transition);
        }
        
        .glass-panel:hover {
            border-color: rgba(255, 255, 255, 0.12);
        }
        
        .section-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1.25rem;
            display: flex;
            align-items: center;
            gap: 8px;
            color: #fff;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
        }
        
        /* Form elements */
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
        }
        
        .form-label {
            font-size: 0.9rem;
            font-weight: 500;
            color: var(--color-text-sub);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Model selector */
        .pill-selector {
            display: grid;
            grid-template-columns: 1fr;
            gap: 8px;
        }
        
        .pill-selector::-webkit-scrollbar {
            width: 5px;
        }
        .pill-selector::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.01);
            border-radius: 4px;
        }
        .pill-selector::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }
        .pill-selector::-webkit-scrollbar-thumb:hover {
            background: var(--color-primary);
        }
        
        .pill-btn {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 12px;
            color: var(--color-text-sub);
            cursor: pointer;
            font-family: var(--font-sans);
            font-weight: 500;
            text-align: left;
            display: flex;
            flex-direction: column;
            gap: 4px;
            transition: var(--transition);
        }
        
        .pill-btn:hover {
            background: rgba(255, 255, 255, 0.06);
            border-color: rgba(255, 255, 255, 0.15);
        }
        
        .pill-btn.active {
            background: rgba(124, 58, 237, 0.12);
            border-color: var(--color-primary);
            color: #fff;
            box-shadow: 0 0 12px var(--color-primary-glow);
        }
        
        .pill-title {
            font-size: 0.95rem;
            font-weight: 600;
        }
        
        .pill-desc {
            font-size: 0.75rem;
            color: var(--color-text-muted);
        }
        
        /* Slider styling */
        .slider-container {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .slider-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .slider-val {
            font-family: var(--font-mono);
            font-weight: 600;
            font-size: 1.1rem;
            color: var(--color-primary);
        }
        
        .range-slider {
            -webkit-appearance: none;
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            outline: none;
            transition: var(--transition);
        }
        
        .range-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: var(--color-primary);
            cursor: pointer;
            box-shadow: 0 0 8px var(--color-primary-glow);
            transition: var(--transition);
        }
        
        .range-slider::-webkit-slider-thumb:hover {
            transform: scale(1.25);
        }
        
        .tick-labels {
            display: flex;
            justify-content: space-between;
            font-size: 0.75rem;
            color: var(--color-text-muted);
            font-family: var(--font-mono);
        }
        
        /* Uploader & templates */
        .upload-area {
            border: 2px dashed var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            cursor: pointer;
            transition: var(--transition);
            background: rgba(255, 255, 255, 0.01);
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
        }
        
        .upload-area:hover {
            border-color: var(--color-primary);
            background: rgba(124, 58, 237, 0.02);
        }
        
        .upload-icon {
            font-size: 1.8rem;
            color: var(--color-primary);
        }
        
        .upload-text {
            font-size: 0.85rem;
            color: var(--color-text-sub);
        }
        
        .template-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 6px;
            margin-top: 8px;
        }
        
        .temp-btn {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 8px 4px;
            font-size: 0.7rem;
            color: var(--color-text-sub);
            cursor: pointer;
            text-transform: capitalize;
            transition: var(--transition);
            text-align: center;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        .temp-btn:hover, .temp-btn.active {
            border-color: var(--color-primary);
            color: #fff;
            background: rgba(124, 58, 237, 0.1);
        }
        
        /* Run button */
        .submit-btn {
            background: linear-gradient(135deg, var(--color-primary) 0%, #6d28d9 100%);
            border: none;
            border-radius: 12px;
            color: #fff;
            padding: 14px;
            font-family: var(--font-sans);
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            transition: var(--transition);
            box-shadow: 0 4px 16px var(--color-primary-glow);
            width: 100%;
            margin-top: 1rem;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 8px;
        }
        
        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(124, 58, 237, 0.55);
        }
        
        .submit-btn:active {
            transform: translateY(0);
        }
        
        .submit-btn:disabled {
            background: var(--color-text-muted);
            box-shadow: none;
            cursor: not-allowed;
        }
        
        /* Right panel results styling */
        .results-wrapper {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            min-height: 500px;
        }
        
        .placeholder-state {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100%;
            color: var(--color-text-muted);
            gap: 12px;
            text-align: center;
            padding: 4rem 2rem;
        }
        
        .placeholder-icon {
            font-size: 3rem;
            opacity: 0.3;
        }
        
        /* HUD Stats row */
        .hud-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
        }
        
        @media (max-width: 600px) {
            .hud-row {
                grid-template-columns: 1fr 1fr;
            }
        }
        
        .hud-stat {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        
        .hud-label {
            font-size: 0.75rem;
            color: var(--color-text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .hud-val {
            font-size: 1.25rem;
            font-weight: 700;
            font-family: var(--font-mono);
            color: #fff;
        }
        
        /* Visual cards */
        .visual-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
        }
        
        @media (max-width: 1000px) {
            .visual-grid {
                grid-template-columns: 1fr 1fr;
            }
        }
        
        @media (max-width: 500px) {
            .visual-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .visual-card {
            background: rgba(255, 255, 255, 0.01);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 10px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            align-items: center;
            transition: var(--transition);
        }
        
        .visual-card:hover {
            border-color: rgba(255, 255, 255, 0.15);
            background: rgba(255, 255, 255, 0.03);
        }
        
        .visual-title {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--color-text-sub);
            text-transform: uppercase;
            letter-spacing: 0.03em;
            text-align: center;
        }
        
        .visual-frame {
            width: 100%;
            aspect-ratio: 1;
            background: #111827;
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
            border: 1px solid rgba(255, 255, 255, 0.05);
            position: relative;
        }
        
        .visual-frame img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            image-rendering: pixelated; /* Shows clean upscaled pixels */
        }
        
        .visual-frame .na-badge {
            color: var(--color-text-muted);
            font-size: 0.8rem;
            text-align: center;
            padding: 8px;
            font-weight: 500;
        }
        
        /* Prediction bars */
        .prob-wrapper {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .prob-row {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 0.85rem;
        }
        
        .prob-label {
            width: 90px;
            text-transform: capitalize;
            color: var(--color-text-sub);
            font-weight: 500;
        }
        
        .prob-track {
            flex: 1;
            height: 8px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .prob-fill {
            height: 100%;
            border-radius: 4px;
            background: var(--color-primary);
            transition: width 0.4s ease-out;
        }
        
        .prob-val {
            width: 50px;
            text-align: right;
            font-family: var(--font-mono);
            font-weight: 600;
            color: var(--color-text-main);
        }
        
        /* Status highlights */
        .status-shield {
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        
        .status-shield.fooled {
            background: rgba(239, 68, 68, 0.12);
            color: var(--color-danger);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }
        
        .status-shield.robust {
            background: rgba(16, 185, 129, 0.12);
            color: var(--color-success);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        
        /* Spinner */
        .spinner {
            border: 2px solid rgba(255,255,255,0.1);
            border-radius: 50%;
            border-top: 2px solid #fff;
            width: 18px;
            height: 18px;
            animation: spin 0.8s linear infinite;
            display: none;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Adversarial Cognition Divergence</h1>
            <p>Live Interactive Perceptual Robustness Explorer</p>
        </header>
        
        <div class="grid-layout">
            <!-- Sidebar controls -->
            <div class="glass-panel" style="display: flex; flex-direction: column; gap: 1.25rem;">
                <div class="section-title">
                    <span>1. Configuration</span>
                </div>
                
                <!-- Model select -->
                <div class="form-group">
                    <label class="form-label">Select Architecture</label>
                    <div class="pill-selector" style="max-height: 380px; overflow-y: auto; padding-right: 4px; display: flex; flex-direction: column; gap: 6px;">
                        <button class="pill-btn active" data-model="rhan-adv" onclick="selectModel('rhan-adv')">
                            <span class="pill-title">🔮 RHAN-adv (Ours)</span>
                            <span class="pill-desc">Recurrent Feedback + Robust Training</span>
                        </button>
                        <button class="pill-btn" data-model="rhan-clean" onclick="selectModel('rhan-clean')">
                            <span class="pill-title">🌀 RHAN-clean</span>
                            <span class="pill-desc">Recurrent Feedback (Clean weights)</span>
                        </button>
                        <button class="pill-btn" data-model="resnet-18" onclick="selectModel('resnet-18')">
                            <span class="pill-title">📈 ResNet-18</span>
                            <span class="pill-desc">Standard Feedforward Baseline (32x32)</span>
                        </button>
                        <button class="pill-btn" data-model="vit" onclick="selectModel('vit')">
                            <span class="pill-title">🧩 ViT-Small</span>
                            <span class="pill-desc">Vision Transformer (224x224)</span>
                        </button>
                        <button class="pill-btn" data-model="cornets" onclick="selectModel('cornets')">
                            <span class="pill-title">🐵 CORnet-S</span>
                            <span class="pill-desc">Biologically Recurrent CNN (224x224)</span>
                        </button>
                        <button class="pill-btn" data-model="shaperesnet" onclick="selectModel('shaperesnet')">
                            <span class="pill-title">🎨 Shape-ResNet-50</span>
                            <span class="pill-desc">Shape-Biased Augmentation (224x224)</span>
                        </button>
                        <button class="pill-btn" data-model="bagnet" onclick="selectModel('bagnet')">
                            <span class="pill-title">🛍️ BagNet-33</span>
                            <span class="pill-desc">Extreme Local Patch Bag (64x64)</span>
                        </button>
                        <button class="pill-btn" data-model="efficientnet" onclick="selectModel('efficientnet')">
                            <span class="pill-title">⚡ EfficientNet-B0</span>
                            <span class="pill-desc">High-Efficiency Modern CNN (224x224)</span>
                        </button>
                    </div>
                </div>
                
                <!-- Epsilon budget -->
                <div class="form-group">
                    <div class="slider-header">
                        <label class="form-label">PGD Epsilon budget (ε)</label>
                        <span class="slider-val" id="val-epsilon">0.05</span>
                    </div>
                    <input type="range" class="range-slider" min="0" max="5" step="1" value="2" id="input-epsilon" oninput="updateEpsilon(this.value)">
                    <div class="tick-labels">
                        <span>0.00</span>
                        <span>0.01</span>
                        <span>0.05</span>
                        <span>0.10</span>
                        <span>0.20</span>
                        <span>0.30</span>
                    </div>
                </div>
                
                <!-- Image Input -->
                <div class="form-group">
                    <label class="form-label">Input Stimulus</label>
                    <div class="upload-area" onclick="document.getElementById('file-input').click()">
                        <span class="upload-icon">📸</span>
                        <span class="upload-text">Upload Custom Image</span>
                        <span class="upload-text" style="font-size: 0.7rem; color: var(--color-text-muted);">Auto-resized to 32x32</span>
                        <input type="file" id="file-input" style="display: none;" accept="image/*" onchange="handleUpload(event)">
                    </div>
                    
                    <div style="margin-top: 8px;">
                        <span class="form-label" style="font-size: 0.75rem; color: var(--color-text-muted);">Or Choose CIFAR-10 Template:</span>
                        <div class="template-grid" id="templates-container">
                            <!-- Template pills inject here -->
                        </div>
                    </div>
                </div>
                
                <button class="submit-btn" id="btn-submit" onclick="runAnalysis()">
                    <span class="spinner" id="btn-spinner"></span>
                    <span id="btn-text">Run Adversarial Analysis</span>
                </button>
            </div>
            
            <!-- Main results area -->
            <div class="glass-panel" id="panel-results">
                <div class="placeholder-state" id="results-placeholder">
                    <span class="placeholder-icon">👁️</span>
                    <h3>Perceptual Analysis Pending</h3>
                    <p>Select a model, adjust epsilon noise level, and select a stimulus image to run prediction and visualize recurrent spatial feedback loops.</p>
                </div>
                
                <div class="results-wrapper" id="results-content" style="display: none;">
                    <div class="section-title" style="justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                        <span>2. Empirical & Computational Output</span>
                        <div id="robust-badge"></div>
                    </div>
                    
                    <!-- HUD metrics -->
                    <div class="hud-row">
                        <div class="hud-stat">
                            <span class="hud-label">Ground Truth</span>
                            <span class="hud-val" id="stat-truth" style="text-transform: capitalize;">—</span>
                        </div>
                        <div class="hud-stat">
                            <span class="hud-label">Adversarial Pred.</span>
                            <span class="hud-val" id="stat-pred" style="text-transform: capitalize;">—</span>
                        </div>
                        <div class="hud-stat">
                            <span class="hud-label">Confidence</span>
                            <span class="hud-val" id="stat-conf">—</span>
                        </div>
                        <div class="hud-stat">
                            <span class="hud-label">Gating Entropy</span>
                            <span class="hud-val" id="stat-entropy">—</span>
                        </div>
                    </div>
                    
                    <!-- Side-by-side Visuals -->
                    <div class="visual-grid">
                        <div class="visual-card">
                            <span class="visual-title">Clean Input</span>
                            <div class="visual-frame">
                                <img id="img-clean" src="" alt="Clean image">
                            </div>
                        </div>
                        
                        <div class="visual-card">
                            <span class="visual-title">Adversarial Input</span>
                            <div class="visual-frame">
                                <img id="img-adv" src="" alt="Adversarial image">
                            </div>
                        </div>
                        
                        <div class="visual-card">
                            <span class="visual-title">Recurrent Gate (t=1)</span>
                            <div class="visual-frame" id="frame-gate1">
                                <img id="img-gate1" src="" alt="Gate step 1">
                            </div>
                        </div>
                        
                        <div class="visual-card">
                            <span class="visual-title">Recurrent Gate (t=2)</span>
                            <div class="visual-frame" id="frame-gate2">
                                <img id="img-gate2" src="" alt="Gate step 2">
                            </div>
                        </div>
                    </div>
                    
                    <!-- Confidence Distributions -->
                    <div style="margin-top: 1rem;">
                        <span class="form-label" style="margin-bottom: 8px; display: block;">Adversarial Prediction Distribution</span>
                        <div class="prob-wrapper" id="distribution-bars">
                            <!-- Prob rows inject here -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const EPS_VALS = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30];
        let currentModel = 'rhan-adv';
        let currentEpsilon = 0.05;
        let selectedTemplateIdx = 3; // Default 'cat'
        let uploadedImageBase64 = null;
        
        // Fetch templates on load
        window.addEventListener('DOMContentLoaded', () => {
            fetch('/api/templates')
                .then(r => r.json())
                .then(data => {
                    const grid = document.getElementById('templates-container');
                    grid.innerHTML = data.map(t => {
                        const active = t.index === selectedTemplateIdx ? 'active' : '';
                        return `<button class="temp-btn ${active}" id="temp-${t.index}" onclick="selectTemplate(${t.index})">${t.name}</button>`;
                    }).join('');
                });
        });
        
        function selectModel(model) {
            currentModel = model;
            document.querySelectorAll('.pill-btn').forEach(btn => {
                btn.classList.toggle('active', btn.getAttribute('data-model') === model);
            });
        }
        
        function updateEpsilon(sliderVal) {
            const idx = parseInt(sliderVal);
            currentEpsilon = EPS_VALS[idx];
            document.getElementById('val-epsilon').textContent = currentEpsilon.toFixed(2);
        }
        
        function selectTemplate(idx) {
            selectedTemplateIdx = idx;
            uploadedImageBase64 = null;
            document.querySelectorAll('.temp-btn').forEach(btn => {
                btn.classList.toggle('active', btn.getAttribute('id') === `temp-${idx}`);
            });
            document.querySelector('.upload-area').style.borderColor = 'rgba(255, 255, 255, 0.08)';
            document.querySelector('.upload-area').style.background = 'rgba(255, 255, 255, 0.01)';
        }
        
        function handleUpload(event) {
            const file = event.target.files[0];
            if (!file) return;
            
            const reader = new FileReader();
            reader.onload = function(e) {
                uploadedImageBase64 = e.target.result;
                selectedTemplateIdx = null;
                document.querySelectorAll('.temp-btn').forEach(btn => btn.classList.remove('active'));
                
                // Highlight upload block
                const uploadArea = document.querySelector('.upload-area');
                uploadArea.style.borderColor = 'var(--color-primary)';
                uploadArea.style.background = 'rgba(124, 58, 237, 0.08)';
            };
            reader.readAsDataURL(file);
        }
        
        function runAnalysis() {
            const btn = document.getElementById('btn-submit');
            const spinner = document.getElementById('btn-spinner');
            const btnText = document.getElementById('btn-text');
            
            btn.disabled = true;
            spinner.style.display = 'block';
            btnText.textContent = 'Computing PGD Path...';
            
            const payload = {
                model: currentModel,
                epsilon: currentEpsilon,
                template_idx: selectedTemplateIdx,
                image_base64: uploadedImageBase64
            };
            
            fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })
            .then(r => r.json())
            .then(data => {
                // Hide placeholder, show content
                document.getElementById('results-placeholder').style.display = 'none';
                document.getElementById('results-content').style.display = 'flex';
                
                // Set text stats
                document.getElementById('stat-truth').textContent = data.ground_truth;
                document.getElementById('stat-pred').textContent = data.predicted_class;
                document.getElementById('stat-conf').textContent = data.confidence.toFixed(1) + '%';
                
                const entropyEl = document.getElementById('stat-entropy');
                if (data.entropy !== null) {
                    entropyEl.textContent = data.entropy.toFixed(3) + ' nats';
                } else {
                    entropyEl.textContent = 'N/A';
                }
                
                // Set images
                document.getElementById('img-clean').src = data.clean_image;
                document.getElementById('img-adv').src = data.adv_image;
                
                // Gating visual handlers
                const frame1 = document.getElementById('frame-gate1');
                const frame2 = document.getElementById('frame-gate2');
                
                if (data.gate_step1) {
                    frame1.innerHTML = `<img id="img-gate1" src="${data.gate_step1}" alt="Gate Step 1">`;
                } else {
                    let desc = "Feedforward Baseline";
                    if (currentModel === 'cornets') desc = "Recurrence Internal (CORnet-S)";
                    else if (currentModel === 'vit') desc = "Global Self-Attention (ViT)";
                    frame1.innerHTML = `<span class="na-badge">Recurrence N/A<br><span style="font-size:0.7rem;opacity:0.6;">${desc}</span></span>`;
                }
                
                if (data.gate_step2) {
                    frame2.innerHTML = `<img id="img-gate2" src="${data.gate_step2}" alt="Gate Step 2">`;
                } else {
                    let desc = "Feedforward Baseline";
                    if (currentModel === 'cornets') desc = "Recurrence Internal (CORnet-S)";
                    else if (currentModel === 'vit') desc = "Global Self-Attention (ViT)";
                    frame2.innerHTML = `<span class="na-badge">Recurrence N/A<br><span style="font-size:0.7rem;opacity:0.6;">${desc}</span></span>`;
                }
                
                // Set robust badge status
                const badge = document.getElementById('robust-badge');
                if (data.is_correct) {
                    badge.innerHTML = `<span class="status-shield robust">✓ ROBUST SHIELDED</span>`;
                } else {
                    badge.innerHTML = `<span class="status-shield fooled">✗ MODEL FOOLED</span>`;
                }
                
                // Render confidence list
                const barsContainer = document.getElementById('distribution-bars');
                barsContainer.innerHTML = data.probabilities.map(p => {
                    const fillStyle = p.name === data.predicted_class 
                        ? (data.is_correct ? 'var(--color-success)' : 'var(--color-danger)')
                        : 'var(--color-primary)';
                    return `
                        <div class="prob-row">
                            <span class="prob-label">${p.name}</span>
                            <div class="prob-track">
                                <div class="prob-fill" style="width: ${p.confidence}%; background: ${fillStyle};"></div>
                            </div>
                            <span class="prob-val">${p.confidence.toFixed(1)}%</span>
                        </div>
                    `;
                }).join('');
            })
            .catch(err => {
                alert('Analysis failed: ' + err);
            })
            .finally(() => {
                btn.disabled = false;
                spinner.style.display = 'none';
                btnText.textContent = 'Run Adversarial Analysis';
            });
        }
    </script>
</body>
</html>
"""

# =============================================================================
# HTTP BACKEND API HANDLER
# =============================================================================
class APIHandler(BaseHTTPRequestHandler):
    
    def log_message(self, format, *args):
        # Silence default terminal logs to keep interface clean
        pass
        
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(FRONTEND_HTML.encode('utf-8'))
            
        elif self.path == '/api/templates':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            # Pack templates details (names and raw pixel representation)
            res = []
            for idx, (img_tensor, label) in TEMPLATES.items():
                res.append({
                    "index": idx,
                    "name": CLASSES[label]
                })
            self.wfile.write(json.dumps(res).encode('utf-8'))
        else:
            self.send_error(404, "Not Found")
            
    def do_POST(self):
        if self.path == '/api/analyze':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            payload = json.loads(post_data.decode('utf-8'))
            
            model_name = payload.get('model', 'rhan-adv')
            epsilon = float(payload.get('epsilon', 0.05))
            template_idx = payload.get('template_idx')
            image_base64 = payload.get('image_base64')
            
            # 1. Resolve image source
            if image_base64 is not None:
                # Custom upload
                clean_tensor = process_uploaded_image(image_base64).to(DEVICE)
                label_idx = 0  # Dummy label for custom images
            else:
                # Template index
                idx = int(template_idx) if template_idx is not None else 3
                clean_tensor, label_idx = TEMPLATES[idx]
                clean_tensor = clean_tensor.to(DEVICE)
                
            # 2. Get active model
            model = get_model(model_name)
            
            # 3. Generate adversarial image under PGD-20
            # Steps are set to 20 for interactive performance (<0.3s runtime)
            cifar_min = torch.tensor([-2.4291, -2.4181, -2.2194]).view(1, 3, 1, 1).to(DEVICE)
            cifar_max = torch.tensor([2.6400, 2.6210, 2.7615]).view(1, 3, 1, 1).to(DEVICE)
            
            # Resolve input size for the model
            if model_name in ['vit', 'efficientnet', 'shaperesnet', 'cornets']:
                input_size = 224
            elif model_name == 'bagnet':
                input_size = 64
            else:
                input_size = 32
                
            # Resize clean tensor to input size if necessary
            if input_size != 32:
                img_unsqueezed = F.interpolate(clean_tensor.unsqueeze(0), size=(input_size, input_size), mode='bilinear', align_corners=False)
            else:
                img_unsqueezed = clean_tensor.unsqueeze(0)
                
            lbl_tensor = torch.tensor([label_idx]).to(DEVICE)
            
            if epsilon > 0:
                steps = 20
                alpha = max(epsilon / steps, 0.005)
                adv_tensor, _ = pgd_attack(
                    model, img_unsqueezed, lbl_tensor,
                    epsilon=epsilon, alpha=alpha, steps=steps,
                    device=DEVICE, clip_min=cifar_min, clip_max=cifar_max,
                    random_start=True
                )
            else:
                adv_tensor = img_unsqueezed.clone()
                
            # 4. Run prediction and gate extraction
            gate_step1_b64 = None
            gate_step2_b64 = None
            mean_entropy = None
            
            if 'rhan' in model_name:
                # Extract gates via custom forward path
                with torch.no_grad():
                    logits, gates = forward_with_gates(model, adv_tensor)
                    
                map_t1 = gates[0][0].mean(dim=0)
                map_t2 = gates[1][0].mean(dim=0)
                
                gate_step1_b64 = gate_to_base64_overlay(adv_tensor[0], map_t1)
                gate_step2_b64 = gate_to_base64_overlay(adv_tensor[0], map_t2)
                mean_entropy = (compute_entropy(map_t1) + compute_entropy(map_t2)) / 2
            else:
                # Standard feedforward ResNet/ViT/CORnetS/BagNet/EfficientNet
                with torch.no_grad():
                    logits = model(adv_tensor)
                    
            # Compute probabilities
            probs = F.softmax(logits, dim=1)[0]
            top_probs, top_classes = torch.sort(probs, descending=True)
            
            predicted_class = CLASSES[top_classes[0].item()]
            confidence = top_probs[0].item() * 100
            is_correct = top_classes[0].item() == label_idx
            
            # Build probability distribution output
            probs_output = []
            for i in range(10):
                c_idx = top_classes[i].item()
                probs_output.append({
                    "name": CLASSES[c_idx],
                    "confidence": probs[c_idx].item() * 100
                })
                
            # Create base64 clean and adv representations
            clean_b64 = tensor_to_base64(img_unsqueezed[0])
            adv_b64 = tensor_to_base64(adv_tensor[0])
            
            # Send JSON response
            response_data = {
                "ground_truth": CLASSES[label_idx],
                "predicted_class": predicted_class,
                "confidence": confidence,
                "is_correct": is_correct,
                "entropy": mean_entropy,
                "clean_image": clean_b64,
                "adv_image": adv_b64,
                "gate_step1": gate_step1_b64,
                "gate_step2": gate_step2_b64,
                "probabilities": probs_output
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
        else:
            self.send_error(404, "Not Found")

def process_uploaded_image(base64_str):
    """Parse base64 uploaded canvas back into a standard CIFAR tensor."""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_bytes = base64.b64decode(base64_str)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    # Scale to CIFAR dimensions
    pil_img = pil_img.resize((32, 32), Image.BILINEAR)
    
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return transform(pil_img)

# =============================================================================
# MAIN INVOCATION ENTRYPOINT
# =============================================================================
def main():
    PORT = 8000
    
    # Load sample templates
    try:
        load_templates()
    except Exception as e:
        print(f"Error loading templates directly: {e}")
        print("Creating fallback mock templates...")
        # Fallback dummy tensors if HuggingFace/local split fails
        for i in range(10):
            TEMPLATES[i] = (torch.randn(3, 32, 32), i)
            
    # Pre-load RHAN-adv as default to optimize first request response
    try:
        get_model('rhan-adv')
    except Exception as e:
        print(f"Warning pre-loading RHAN-adv: {e}")
        
    class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
        pass

    server = ThreadingHTTPServer(('0.0.0.0', PORT), APIHandler)
    print("\n" + "="*70)
    print("  ACD RESEARCH DASHBOARD — LIVE INTERACTIVE WEB EXPLORER")
    print("="*70)
    print(f"  ✓ Live web server successfully started on port {PORT}")
    print(f"  ✓ Point your browser to: http://localhost:{PORT}")
    print(f"  ✓ Active Models: RHAN-adv, RHAN-clean, ResNet-18, ViT-Small, CORnet-S, ShapeResNet-50, BagNet-33, EfficientNet-B0")
    print("="*70)
    print("  Press Ctrl+C to stop the local server.\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local research server... Done.")

if __name__ == '__main__':
    main()
