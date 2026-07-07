#!/usr/bin/env python3
"""
RHAN Scientific Visualization Suite
===================================
Generates 13 publication-quality figures (Nature / NeurIPS style)
for the RHAN project, exporting both SVG (vector) and PNG (high-res).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

# Setup figures directory
os.makedirs("figures", exist_ok=True)

# Modern scientific color palette (NeurIPS / Nature style)
LIGHT_PALETTE = {
    'bg': '#FFFFFF',
    'box_bg_default': '#F8F9FA',
    'box_bg_ventral': '#FFF0F5', # Soft rose for ventral
    'box_bg_dorsal': '#E0FFFF',  # Soft cyan for dorsal
    'box_border': '#1F77B4',
    'text': '#111111',
    'forward': '#1F77B4',        # Solid blue for forward
    'feedback': '#D32F2F',       # Dashed red for feedback
    'annotation': '#555555',
    'grid': '#F1F3F5',
    'highlight': '#FF9800',
    'cluster_a': '#FFCDD2',
    'cluster_b': '#C8E6C9'
}

DARK_PALETTE = {
    'bg': '#0D1117',
    'box_bg_default': '#161B22',
    'box_bg_ventral': '#2A1F2D', # Dark rose
    'box_bg_dorsal': '#1F2E35',  # Dark cyan
    'box_border': '#58A6FF',
    'text': '#FFFFFF',
    'forward': '#58A6FF',
    'feedback': '#FF7B72',
    'annotation': '#8B949E',
    'grid': '#21262D',
    'highlight': '#D29922',
    'cluster_a': '#3D1B1E',
    'cluster_b': '#1E3E20'
}

PALETTE = LIGHT_PALETTE

# Typography helper
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'

def draw_box(ax, x, y, w, h, text, subtitle=None, bg=None, border=None, lw=1.5):
    if bg is None:
        bg = PALETTE['box_bg_default']
    if border is None:
        border = PALETTE['box_border']
    # Rectangle with clean border
    rect = patches.Rectangle(
        (x - w/2, y - h/2), w, h,
        facecolor=bg, edgecolor=border, linewidth=lw, joinstyle='round'
    )
    ax.add_patch(rect)
    
    # Text
    if subtitle:
        ax.text(x, y + 0.15*h, text, ha='center', va='center', fontsize=9, fontweight='bold', color=PALETTE['text'])
        ax.text(x, y - 0.2*h, subtitle, ha='center', va='center', fontsize=7.5, style='italic', color=PALETTE['annotation'])
    else:
        ax.text(x, y, text, ha='center', va='center', fontsize=9.5, fontweight='bold', color=PALETTE['text'])

def draw_arrow(ax, x1, y1, x2, y2, color=None, style='solid', lw=1.5):
    if color is None:
        color = PALETTE['forward']
    linestyle = '-' if style == 'solid' else '--'
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=linestyle, mutation_scale=12)
    )

def save_and_close(filename):
    fig = plt.gcf()
    fig.patch.set_facecolor(PALETTE['bg'])
    for ax in fig.axes:
        ax.set_facecolor(PALETTE['bg'])
        ax.xaxis.label.set_color(PALETTE['text'])
        ax.yaxis.label.set_color(PALETTE['text'])
        ax.title.set_color(PALETTE['text'])
        for text in ax.texts:
            if PALETTE == DARK_PALETTE:
                if text.get_color() in ['#111111', 'black', '#111']:
                    text.set_color('white')
            else:
                if text.get_color() in ['white', '#ffffff', '#fff']:
                    text.set_color('#111111')
    plt.tight_layout()
    theme_suffix = "_light" if PALETTE == LIGHT_PALETTE else "_dark"
    base_dir = "figures_v3"
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"{filename}{theme_suffix}")
    
    if PALETTE == LIGHT_PALETTE:
        plt.savefig(f"{path}.svg", format='svg', bbox_inches='tight', facecolor=PALETTE['bg'])
        plt.savefig(f"{path}.pdf", format='pdf', bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.savefig(f"{path}.png", format='png', dpi=300, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close()
    print(f"Exported: {path}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 1: Complete RHAN Architecture
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def gen_figure_1():
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(1, 9)
    ax.axis('off')
    
    ax.text(7, 8.5, "Complete RHAN Architecture & Information Flow", ha='center', fontsize=14, fontweight='bold')
    
    # Draw blocks
    draw_box(ax, 1.0, 6.0, 1.4, 0.8, "Input Image", "STL-10 (96x96)")
    draw_box(ax, 3.2, 6.0, 1.8, 0.9, "Convolutional Stem", "V1/Retinal Processing")
    draw_box(ax, 5.5, 6.0, 1.6, 0.8, "Patch Tokenizer", "Dimensionality Reduction")
    
    # Dual streams
    draw_box(ax, 8.0, 7.2, 2.0, 0.9, "Ventral Stream", "What: Object Semantics", bg=PALETTE['box_bg_ventral'])
    draw_box(ax, 8.0, 4.8, 2.0, 0.9, "Dorsal Stream", "Where: Spatial Geometry", bg=PALETTE['box_bg_dorsal'])
    
    draw_box(ax, 10.8, 6.0, 2.0, 0.8, "Global Spatial Map", "Feature Aggregator")
    draw_box(ax, 13.0, 6.0, 1.4, 0.8, "Prototype Head", "Angular Classifier")
    
    # Feedback loop modules (bottom)
    draw_box(ax, 10.8, 2.8, 1.8, 0.8, "Predictor Module", "Temporal/Spatial Predictor")
    draw_box(ax, 8.0, 2.8, 1.8, 0.8, "Prediction Error", "Residual Computation")
    draw_box(ax, 5.5, 2.8, 1.6, 0.8, "Error Gate", "Frequency Filter")
    draw_box(ax, 3.2, 2.8, 1.8, 0.8, "Recurrent Feedback", "Active Inference Gating")
    
    # Draw forward arrows (solid blue)
    draw_arrow(ax, 1.7, 6.0, 2.3, 6.0)
    draw_arrow(ax, 4.1, 6.0, 4.7, 6.0)
    
    # Split paths
    draw_arrow(ax, 6.3, 6.0, 7.0, 7.2)
    draw_arrow(ax, 6.3, 6.0, 7.0, 4.8)
    
    # Merge paths
    draw_arrow(ax, 9.0, 7.2, 9.8, 6.0)
    draw_arrow(ax, 9.0, 4.8, 9.8, 6.0)
    
    draw_arrow(ax, 11.8, 6.0, 12.3, 6.0)
    
    # Feedback path activation
    draw_arrow(ax, 10.8, 5.6, 10.8, 3.2)
    draw_arrow(ax, 9.9, 2.8, 8.9, 2.8)
    draw_arrow(ax, 7.1, 2.8, 6.3, 2.8)
    draw_arrow(ax, 4.7, 2.8, 4.1, 2.8)
    
    # Loop back up to Stem (dashed red)
    draw_arrow(ax, 3.2, 3.2, 3.2, 5.55, color=PALETTE['feedback'], style='dashed')
    
    save_and_close("figure_1_rhan_architecture")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 2: Biological Inspiration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def gen_figure_2():
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    ax.text(5, 6.5, "Biological Inspiration Behind RHAN", ha='center', fontsize=14, fontweight='bold')
    
    # Headers
    ax.text(2.5, 5.8, "Human Visual System", ha='center', fontsize=11, fontweight='bold', color=PALETTE['forward'])
    ax.text(7.5, 5.8, "RHAN Architecture", ha='center', fontsize=11, fontweight='bold', color=PALETTE['box_border'])
    
    # Human nodes
    draw_box(ax, 2.5, 5.0, 1.8, 0.5, "Retina")
    draw_box(ax, 2.5, 4.1, 1.8, 0.5, "LGN")
    draw_box(ax, 2.5, 3.2, 1.8, 0.5, "Primary Visual Cortex (V1)")
    draw_box(ax, 1.5, 2.3, 1.4, 0.5, "Ventral Path (What)", bg=PALETTE['box_bg_ventral'])
    draw_box(ax, 3.5, 2.3, 1.4, 0.5, "Dorsal Path (Where)", bg=PALETTE['box_bg_dorsal'])
    draw_box(ax, 2.5, 1.4, 1.8, 0.5, "IT Cortex")
    draw_box(ax, 2.5, 0.5, 1.8, 0.5, "Perception & Decision")
    
    # Connect human nodes
    draw_arrow(ax, 2.5, 4.75, 2.5, 4.35)
    draw_arrow(ax, 2.5, 3.85, 2.5, 3.45)
    draw_arrow(ax, 2.5, 2.95, 1.5, 2.55)
    draw_arrow(ax, 2.5, 2.95, 3.5, 2.55)
    draw_arrow(ax, 1.5, 2.05, 2.5, 1.65)
    draw_arrow(ax, 3.5, 2.05, 2.5, 1.65)
    draw_arrow(ax, 2.5, 1.15, 2.5, 0.75)
    
    # Top-down cortical feedback
    draw_arrow(ax, 1.8, 1.4, 1.8, 4.1, color=PALETTE['feedback'], style='dashed')
    ax.text(1.3, 2.8, "Top-down\nfeedback", color=PALETTE['feedback'], fontsize=8, ha='center')
    
    # RHAN nodes
    draw_box(ax, 7.5, 5.0, 2.0, 0.5, "Input Tensors")
    draw_box(ax, 7.5, 4.1, 2.0, 0.5, "Convolutional Stem")
    draw_box(ax, 7.5, 3.2, 2.0, 0.5, "Patch Tokenizer")
    draw_box(ax, 6.5, 2.3, 1.5, 0.5, "Ventral Transformer", bg=PALETTE['box_bg_ventral'])
    draw_box(ax, 8.5, 2.3, 1.5, 0.5, "Dorsal Transformer", bg=PALETTE['box_bg_dorsal'])
    draw_box(ax, 7.5, 1.4, 2.0, 0.5, "Recurrent Feedback Loop")
    draw_box(ax, 7.5, 0.5, 2.0, 0.5, "Prototype Head Prediction")
    
    # Connect RHAN nodes
    draw_arrow(ax, 7.5, 4.75, 7.5, 4.35)
    draw_arrow(ax, 7.5, 3.85, 7.5, 3.45)
    draw_arrow(ax, 7.5, 2.95, 6.5, 2.55)
    draw_arrow(ax, 7.5, 2.95, 8.5, 2.55)
    draw_arrow(ax, 6.5, 2.05, 7.5, 1.65)
    draw_arrow(ax, 8.5, 2.05, 7.5, 1.65)
    draw_arrow(ax, 7.5, 1.15, 7.5, 0.75)
    
    # Cross-connections (mapping)
    for y in [5.0, 4.1, 3.2, 1.4, 0.5]:
        draw_arrow(ax, 3.5, y, 6.4, y, color='#78909C', style='dashed', lw=1.0)
    
    save_and_close("figure_2_biological_inspiration")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 3: Predictive Coding Loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def gen_figure_3():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(-1, 9)
    ax.set_ylim(-0.5, 5.5)
    ax.axis('off')
    
    ax.text(4, 5.1, "Recurrent Predictive Coding Loop Over Time Steps", ha='center', fontsize=13, fontweight='bold')
    
    # Iterations columns
    steps = [0, 1, 2]
    for i, t in enumerate(steps):
        x = i * 3.2
        ax.text(x + 0.8, 4.5, f"Iteration t = {t}", ha='center', fontsize=11, fontweight='bold', color=PALETTE['forward'])
        
        # Draw boxes
        draw_box(ax, x + 0.8, 3.8, 1.8, 0.5, f"Stem Features f({t})")
        draw_box(ax, x + 0.8, 2.8, 1.8, 0.5, f"Transformer Output s({t})")
        draw_box(ax, x + 0.8, 1.8, 1.8, 0.5, f"Prediction Error e({t})")
        draw_box(ax, x + 0.8, 0.8, 1.8, 0.5, f"Frequency Gate g(e)")
        
        # Connections inside step
        draw_arrow(ax, x + 0.8, 3.55, x + 0.8, 3.05)
        draw_arrow(ax, x + 0.8, 2.55, x + 0.8, 2.05)
        draw_arrow(ax, x + 0.8, 1.55, x + 0.8, 1.05)
        
        # Loop to next step
        if t < 2:
            draw_arrow(ax, x + 1.7, 0.8, x + 2.3, 3.8, color=PALETTE['feedback'], style='dashed')
            ax.text(x + 2.0, 2.5, f"f({t+1}) Update", color=PALETTE['feedback'], fontsize=8, ha='center', rotation=-40)
            
    # Equations box
    ax.text(-0.8, 2.5, "Equations:\n\n$e^t = f_{stem} - P(s^t)$\n\n$g(e^t) = \sigma(Conv(e^t))$\n\n$f^{t+1} = f_{stem} + g(e^t) \odot e^t$", 
            ha='left', va='center', fontsize=9.5, bbox=dict(facecolor='#ECEFF1', edgecolor='#90A4AE', boxstyle='round,pad=0.6'))
            
    save_and_close("figure_3_predictive_coding_loop")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 4: Dual Transformer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def gen_figure_4():
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    ax.text(5, 5.5, "Dual-Stream Attention Token Processing", ha='center', fontsize=13, fontweight='bold')
    
    # Input
    draw_box(ax, 5.0, 4.7, 2.4, 0.5, "Input Patch Tokens", "Sequence: [Seq_Len, Embed_Dim]")
    draw_box(ax, 5.0, 3.8, 2.4, 0.5, "Positional Encoding", "Sinusoidal / Learnable Add")
    draw_arrow(ax, 5.0, 4.45, 5.0, 4.05)
    
    # Split
    draw_box(ax, 5.0, 2.9, 1.8, 0.4, "Channel Split (50/50)", "Split along embedding dimension")
    draw_arrow(ax, 5.0, 3.55, 5.0, 3.1)
    
    # Dual pathways
    draw_box(ax, 2.5, 2.0, 2.2, 0.7, "Ventral Transformer\n(What Pathway)", "Identity / Texture (0.5 * Dim)", bg=PALETTE['box_bg_ventral'])
    draw_box(ax, 7.5, 2.0, 2.2, 0.7, "Dorsal Transformer\n(Where Pathway)", "Spatial Layout (0.5 * Dim)", bg=PALETTE['box_bg_dorsal'])
    
    draw_arrow(ax, 4.1, 2.9, 2.5, 2.35)
    draw_arrow(ax, 5.9, 2.9, 7.5, 2.35)
    
    # Output and merge
    draw_box(ax, 5.0, 1.0, 2.2, 0.5, "Concatenate & Project", "Re-assemble [Seq_Len, Embed_Dim]")
    draw_arrow(ax, 2.5, 1.65, 3.9, 1.2)
    draw_arrow(ax, 7.5, 1.65, 6.1, 1.2)
    
    draw_box(ax, 5.0, 0.3, 2.6, 0.4, "Output Shared Representation")
    draw_arrow(ax, 5.0, 0.75, 5.0, 0.5)
    
    save_and_close("figure_4_dual_transformer")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 5: Spherical Prototype Head
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def gen_figure_5():
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(0, 1.3, "Spherical Prototype Head (Angular Classification)", ha='center', fontsize=12, fontweight='bold')
    
    # Draw unit circle
    circle = plt.Circle((0, 0), 1.0, fill=False, color='#90A4AE', ls='--', lw=1.5)
    ax.add_patch(circle)
    
    # Coordinates axes
    ax.axhline(0, color='#CFD8DC', zorder=1, ls=':')
    ax.axvline(0, color='#CFD8DC', zorder=1, ls=':')
    
    # Vectors
    # Feature vector z
    z_angle = np.deg2rad(55)
    z_x, z_y = np.cos(z_angle), np.sin(z_angle)
    ax.quiver(0, 0, z_x, z_y, angles='xy', scale_units='xy', scale=1, color='#1F77B4', zorder=3, width=0.015, label='Feature Vector $z$')
    ax.text(z_x*1.1, z_y*1.1, "$z$ (Normalized)", fontsize=10, color='#1F77B4', fontweight='bold')
    
    # Prototypes
    p_angles = [20, 85, 150]
    colors = ['#4CAF50', '#FF9800', '#9C27B0']
    for idx, angle in enumerate(p_angles):
        rad = np.deg2rad(angle)
        px, py = np.cos(rad), np.sin(rad)
        ax.quiver(0, 0, px, py, angles='xy', scale_units='xy', scale=1, color=colors[idx], zorder=3, width=0.012)
        ax.text(px*1.15, py*1.15, f"$p_{idx+1}$", fontsize=10, color=colors[idx], fontweight='bold')
        
    # Decision boundaries
    boundaries = [52.5, 117.5]
    for b in boundaries:
        rad = np.deg2rad(b)
        ax.plot([0, np.cos(rad)*1.3], [0, np.sin(rad)*1.3], color='#E0E0E0', ls='-', lw=1.5, zorder=2)
        
    # Boundary label
    ax.text(0.65, 0.85, "Decision Boundary", fontsize=8, color='#9E9E9E', rotation=-35)
    
    # Math annotation
    ax.text(0, -1.3, r"$P(y = c) = \text{softmax}\left( \exp(\alpha) \cdot \frac{z \cdot p_c}{\|z\| \|p_c\|} \right)$", 
            ha='center', fontsize=12, bbox=dict(facecolor='#F5F5F7', edgecolor='#B0BEC5', boxstyle='round,pad=0.5'))
            
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.6, 1.5)
    
    save_and_close("figure_5_spherical_prototype")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 6: SAIL Algorithm Flowchart
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def gen_figure_6():
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    ax.text(5, 6.5, "Self-Supervised Adversarial Invariance Learning (SAIL)", ha='center', fontsize=13, fontweight='bold')
    
    # Left flow (Clean)
    draw_box(ax, 2.5, 5.5, 1.8, 0.6, "Clean Image (x)", "Original STL-10 Sample")
    draw_box(ax, 2.5, 4.0, 1.8, 0.6, "Shared Encoder\n(RHAN Backbone)")
    draw_arrow(ax, 2.5, 5.2, 2.5, 4.3)
    
    # Right flow (Adv)
    draw_box(ax, 7.5, 5.5, 1.8, 0.6, "Adversarial Image (x_adv)", "Generated via PGD-10")
    draw_box(ax, 7.5, 4.0, 1.8, 0.6, "Shared Encoder\n(RHAN Backbone)")
    draw_arrow(ax, 7.5, 5.2, 7.5, 4.3)
    
    # Representations
    draw_box(ax, 2.5, 2.6, 1.8, 0.5, "Clean Representation (z)")
    draw_box(ax, 7.5, 2.6, 1.8, 0.5, "Adv Representation (z_adv)")
    draw_arrow(ax, 2.5, 3.7, 2.5, 2.85)
    draw_arrow(ax, 7.5, 3.7, 7.5, 2.85)
    
    # InfoNCE loss
    draw_box(ax, 5.0, 1.6, 2.4, 0.6, "InfoNCE Loss (Positive Pair)", "Attracts (z, z_adv)\nRepels negative samples")
    draw_arrow(ax, 2.5, 2.35, 3.8, 1.8)
    draw_arrow(ax, 7.5, 2.35, 6.2, 1.8)
    
    # Output and downstream
    draw_box(ax, 5.0, 0.6, 2.6, 0.5, "Invariant Representation Space", "Robust features before supervision")
    draw_arrow(ax, 5.0, 1.3, 5.0, 0.85)
    
    save_and_close("figure_6_sail_flowchart")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 7: Geometric Interpretation of SAIL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def gen_figure_7():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))
    
    # Panel A: Traditional
    ax1.set_title("Panel A: Traditional Network", fontsize=11, fontweight='bold')
    ax1.set_xlim(-2, 6)
    ax1.set_ylim(-2, 6)
    ax1.axis('off')
    
    # Draw decision boundary
    ax1.plot([-2, 6], [2.5, 2.5], color='#D32F2F', ls='--', lw=2)
    ax1.text(-1.5, 2.7, "Decision Boundary", color='#D32F2F', fontsize=8, fontweight='bold')
    
    # Draw clusters
    cluster_clean = patches.Ellipse((2, 1), 3, 1.5, angle=15, color=PALETTE['cluster_a'], alpha=0.6, label='Automobile Cluster')
    cluster_truck = patches.Ellipse((2, 4), 3, 1.5, angle=15, color=PALETTE['cluster_b'], alpha=0.6, label='Truck Cluster')
    ax1.add_patch(cluster_clean)
    ax1.add_patch(cluster_truck)
    ax1.text(2, 1, "Automobile Class", ha='center', va='center', color='#5D4037', fontsize=9, fontweight='bold')
    ax1.text(2, 4, "Truck Class", ha='center', va='center', color='#1B5E20', fontsize=9, fontweight='bold')
    
    # Vectors
    ax1.scatter([1.5], [1.2], color='#1F77B4', s=80, zorder=3)
    ax1.text(1.5, 0.8, "Clean $x$", color='#1F77B4', fontsize=9, ha='center', fontweight='bold')
    
    ax1.scatter([2.5], [3.8], color='#E53935', s=80, zorder=3)
    ax1.text(2.5, 4.2, "Adv $x_{adv}$ (Fools model)", color='#E53935', fontsize=9, ha='center', fontweight='bold')
    
    # Draw arrow of adversarial drift
    ax1.annotate("", xy=(2.4, 3.6), xytext=(1.6, 1.4), arrowprops=dict(arrowstyle="->", color='#F44336', lw=1.5, ls=':'))
    ax1.text(1.7, 2.4, "Adversarial Shift\n(Crosses Boundary)", color='#F44336', fontsize=8, rotation=70)
    
    # Panel B: SAIL
    ax2.set_title("Panel B: SAIL Representation", fontsize=11, fontweight='bold')
    ax2.set_xlim(-2, 6)
    ax2.set_ylim(-2, 6)
    ax2.axis('off')
    
    # Decision boundary (wider margin)
    ax2.plot([-2, 6], [2.8, 2.8], color='#4CAF50', ls='-', lw=2.5)
    ax2.text(-1.5, 3.0, "Robust Decision Margin", color='#4CAF50', fontsize=8, fontweight='bold')
    
    # Shared manifolds
    manifold_auto = patches.Ellipse((2, 1.2), 3.2, 2.0, angle=15, color=PALETTE['cluster_a'], alpha=0.7)
    manifold_truck = patches.Ellipse((2, 4.5), 3.0, 1.4, angle=15, color=PALETTE['cluster_b'], alpha=0.7)
    ax2.add_patch(manifold_auto)
    ax2.add_patch(manifold_truck)
    ax2.text(2, 1.2, "Automobile Manifold", ha='center', va='center', color='#5D4037', fontsize=9, fontweight='bold')
    ax2.text(2, 4.5, "Truck Manifold", ha='center', va='center', color='#1B5E20', fontsize=9, fontweight='bold')
    
    # Clean and Adv both mapped to the same manifold
    ax2.scatter([1.2], [1.2], color='#1F77B4', s=80, zorder=3)
    ax2.text(1.2, 0.8, "Clean $z$", color='#1F77B4', fontsize=9, ha='center', fontweight='bold')
    
    ax2.scatter([2.2], [1.4], color='#E53935', s=80, zorder=3)
    ax2.text(2.5, 1.7, "Adv $z_{adv}$", color='#E53935', fontsize=9, ha='center', fontweight='bold')
    
    # Small local arrow (no manifold crossing)
    ax2.annotate("", xy=(2.0, 1.35), xytext=(1.4, 1.22), arrowprops=dict(arrowstyle="->", color='#4CAF50', lw=1.5))
    ax2.text(1.7, 1.5, "Invariant Map", color='#4CAF50', fontsize=8)
    
    save_and_close("figure_7_geometric_sail")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 8: TDV Framework
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def gen_figure_8():
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis('off')
    
    ax.text(5, 5.8, "Temporal Difference Vision (TDV) Self-Supervised Learning", ha='center', fontsize=13, fontweight='bold')
    
    # Inputs
    draw_box(ax, 2.0, 4.8, 1.8, 0.6, "Video Frame (t)", "STL-10 / UCF-101")
    draw_box(ax, 8.0, 4.8, 1.8, 0.6, "Video Frame (t+1)", "Consecutive frame")
    
    # Encoders
    draw_box(ax, 2.0, 3.4, 1.8, 0.6, "Spatial Encoder\n(RHAN Stem)")
    draw_box(ax, 8.0, 3.4, 1.8, 0.6, "Spatial Encoder\n(RHAN Stem)")
    draw_arrow(ax, 2.0, 4.5, 2.0, 3.7)
    draw_arrow(ax, 8.0, 4.5, 8.0, 3.7)
    
    # Features
    draw_box(ax, 2.0, 2.2, 1.8, 0.5, "Representation z(t)")
    draw_box(ax, 8.0, 2.2, 1.8, 0.5, "Representation z(t+1)")
    draw_arrow(ax, 2.0, 3.1, 2.0, 2.45)
    draw_arrow(ax, 8.0, 3.1, 8.0, 2.45)
    
    # Motion Encoder
    draw_box(ax, 5.0, 3.4, 1.8, 0.6, "Motion Encoder\n(Temporal Diff)", bg=PALETTE['box_bg_dorsal'])
    draw_arrow(ax, 2.0, 4.8, 4.1, 3.7)
    draw_arrow(ax, 8.0, 4.8, 5.9, 3.7)
    
    # Motion vector
    draw_box(ax, 5.0, 2.2, 1.8, 0.5, "Motion Vector (m)", "Dynamics descriptor")
    draw_arrow(ax, 5.0, 3.1, 5.0, 2.45)
    
    # Prediction loss
    draw_box(ax, 5.0, 1.1, 2.2, 0.5, "Prediction Loss (MSE)", "z(t) + m = z(t+1)")
    draw_arrow(ax, 2.0, 2.2, 3.9, 1.1)
    draw_arrow(ax, 5.0, 1.95, 5.0, 1.35)
    draw_arrow(ax, 8.0, 2.2, 6.1, 1.1)
    
    # Additional VICReg losses
    ax.text(5.0, 0.3, "Variance Regularization (VICReg)  |  Covariance Decorrelation (VICReg)", 
            ha='center', fontsize=8.5, fontweight='bold', color=PALETTE['annotation'])
            
    save_and_close("figure_8_tdv_framework")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 9: Adversarial TDV
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def gen_figure_9():
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis('off')
    
    ax.text(5, 5.8, "Adversarial Temporal Consistency & Causality", ha='center', fontsize=13, fontweight='bold')
    
    # Step 1: Clean frame -> PGD -> Adv frame
    draw_box(ax, 2.0, 4.8, 1.8, 0.6, "Clean Frame (t)")
    draw_box(ax, 2.0, 3.4, 1.8, 0.5, "Adversarial Frame (t)", "Noised: x + delta", bg=PALETTE['box_bg_ventral'], border=PALETTE['feedback'])
    draw_arrow(ax, 2.0, 4.5, 2.0, 3.65, color=PALETTE['feedback'])
    ax.text(2.35, 4.1, "PGD-10\nAttack", color=PALETTE['feedback'], fontsize=8)
    
    # Next Clean Frame
    draw_box(ax, 8.0, 4.8, 1.8, 0.6, "Clean Frame (t+1)")
    
    # Encoders
    draw_box(ax, 2.0, 2.2, 1.8, 0.5, "Adv Representation z_adv")
    draw_box(ax, 8.0, 2.2, 1.8, 0.5, "Clean Representation z_next")
    draw_arrow(ax, 2.0, 3.15, 2.0, 2.45)
    draw_arrow(ax, 8.0, 4.5, 8.0, 2.45)
    
    # Motion Encoder
    draw_box(ax, 5.0, 4.8, 1.8, 0.6, "Motion Encoder\n(Temporal Diff)", bg=PALETTE['box_bg_dorsal'])
    draw_arrow(ax, 2.9, 4.8, 4.1, 4.8)
    draw_arrow(ax, 7.1, 4.8, 5.9, 4.8)
    
    draw_box(ax, 5.0, 3.4, 1.8, 0.5, "Motion Vector (m)")
    draw_arrow(ax, 5.0, 4.5, 5.0, 3.65)
    
    # Consistency Constraint
    draw_box(ax, 5.0, 1.4, 2.4, 0.6, "Adversarial TDV Constraint", "$z_{adv} + m \\approx z_{next}$")
    draw_arrow(ax, 2.0, 1.95, 3.8, 1.4)
    draw_arrow(ax, 5.0, 3.15, 5.0, 1.7)
    draw_arrow(ax, 8.0, 1.95, 6.2, 1.4)
    
    # Conclusion
    ax.text(5.0, 0.5, "Implication: Adversarial perturbations cannot break temporal causality boundaries.", 
            ha='center', fontsize=9.5, fontweight='bold', color=PALETTE['box_border'])
            
    save_and_close("figure_9_adversarial_tdv")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 10: Complete Training Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def gen_figure_10():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0.5, 5)
    ax.axis('off')
    
    ax.text(6.5, 4.6, "3-Stage RHAN Training Pipeline", ha='center', fontsize=13, fontweight='bold')
    
    # Stage 1: SAIL
    draw_box(ax, 2.0, 3.0, 2.4, 1.6, "Stage 1: SAIL\n(Pretraining)", 
             "Self-Supervised Invariance\nInfoNCE Loss (Clean/Adv Pairs)\nBackbone Only (50 Epochs)")
    
    # Stage 2: Head Calibration
    draw_box(ax, 5.5, 3.0, 2.4, 1.6, "Stage 2: Calibration\n(Classifier Tuning)", 
             "Frozen Backbone\nWarm-up classification head\nAdam Optimizer (10 Epochs)")
             
    # Stage 3: TRADES Fine-Tuning
    draw_box(ax, 9.0, 3.0, 2.4, 1.6, "Stage 3: TRADES\n(Adversarial Fine-tuning)", 
             "Unfrozen Backbone\nCurriculum Epsilon Scaling\nSGD + Cosine Scheduler (120 Ep)")
             
    # Connect stages
    draw_arrow(ax, 3.2, 3.0, 4.3, 3.0)
    draw_arrow(ax, 6.7, 3.0, 7.8, 3.0)
    
    # Evaluation
    draw_box(ax, 12.0, 3.0, 1.6, 1.6, "Evaluation", "PGD-20\nAutoAttack (AA)\nRobustness Curve")
    draw_arrow(ax, 10.2, 3.0, 11.2, 3.0)
    
    save_and_close("figure_10_training_pipeline")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 11: Loss Landscape
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def gen_figure_11():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    x = np.linspace(-2, 2, 200)
    
    # Panel A: Standard CNN
    y_sharp = x**4 + 3 * x**2
    ax1.plot(x, y_sharp, color='#D32F2F', lw=2.5, label='Loss Landscape')
    ax1.fill_between(x, y_sharp, color='#FFEBEE', alpha=0.5)
    ax1.set_title("Standard CNN Landscape (Sharp Minima)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Parameter Space / Perturbation Direction")
    ax1.set_ylabel("Loss")
    ax1.axvline(0, color='#555555', ls=':')
    
    # Perturbation range showing spike
    ax1.axvspan(-0.3, 0.3, color='#B0BEC5', alpha=0.3, label=r'Perturbation Range $\varepsilon$')
    ax1.annotate("Sharp Spike\n(Vulnerable)", xy=(0.3, 0.5), xytext=(0.8, 6.0),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6))
    ax1.legend(loc='upper left')
    
    # Panel B: RHAN
    y_flat = 0.5 * x**2
    ax2.plot(x, y_flat, color='#388E3C', lw=2.5, label='Loss Landscape')
    ax2.fill_between(x, y_flat, color='#E8F5E9', alpha=0.5)
    ax2.set_title("RHAN Landscape (Flat Minima)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Parameter Space / Perturbation Direction")
    ax2.set_ylabel("Loss")
    ax2.axvline(0, color='#555555', ls=':')
    
    ax2.axvspan(-0.3, 0.3, color='#B0BEC5', alpha=0.3, label=r'Perturbation Range $\varepsilon$')
    ax2.annotate("Flat Basin\n(Robust)", xy=(0.25, 0.05), xytext=(0.8, 1.2),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6))
    ax2.legend(loc='upper left')
    
    save_and_close("figure_11_loss_landscape")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 12: Human vs RHAN Information Flow
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def gen_figure_12():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.5, 5.5)
    ax.axis('off')
    
    ax.text(5, 5.2, "Human Visual Pathway vs. RHAN Component Mapping", ha='center', fontsize=13, fontweight='bold')
    
    # Colors for matches
    col_retina = '#E3F2FD'
    col_lgn = '#E8F5E9'
    col_stream = '#FFF3E0'
    col_feedback = '#FFEBEE'
    
    border_retina = '#1E88E5'
    border_lgn = '#43A047'
    border_stream = '#FB8C00'
    border_feedback = '#E53935'
    
    # Left: Human
    ax.text(2.5, 4.5, "Human Pathway", ha='center', fontsize=11, fontweight='bold', color=PALETTE['annotation'])
    draw_box(ax, 2.5, 3.8, 2.0, 0.5, "Retina / Photoreceptors", "Image capture", bg=col_retina, border=border_retina)
    draw_box(ax, 2.5, 2.9, 2.0, 0.5, "LGN / V1 V2 Cortex", "Local frequency mapping", bg=col_lgn, border=border_lgn)
    draw_box(ax, 2.5, 2.0, 2.0, 0.5, "Ventral & Dorsal Streams", "What & Where extraction", bg=col_stream, border=border_stream)
    draw_box(ax, 2.5, 1.1, 2.0, 0.5, "Synaptic Feedback Loops", "Active predictive coding", bg=col_feedback, border=border_feedback)
    
    draw_arrow(ax, 2.5, 3.55, 2.5, 3.15)
    draw_arrow(ax, 2.5, 2.65, 2.5, 2.25)
    draw_arrow(ax, 2.5, 1.75, 2.5, 1.35)
    
    # Right: RHAN
    ax.text(7.5, 4.5, "RHAN Model Mapping", ha='center', fontsize=11, fontweight='bold', color=PALETTE['annotation'])
    draw_box(ax, 7.5, 3.8, 2.2, 0.5, "Input Tensors", "Image matrices", bg=col_retina, border=border_retina)
    draw_box(ax, 7.5, 2.9, 2.2, 0.5, "Convolutional Stem", "Local feature kernels", bg=col_lgn, border=border_lgn)
    draw_box(ax, 7.5, 2.0, 2.2, 0.5, "Dual Transformers", "Separate semantic/layout weights", bg=col_stream, border=border_stream)
    draw_box(ax, 7.5, 1.1, 2.2, 0.5, "Error Gating & Recurrence", "Feedback iteration loop", bg=col_feedback, border=border_feedback)
    
    draw_arrow(ax, 7.5, 3.55, 7.5, 3.15)
    draw_arrow(ax, 7.5, 2.65, 7.5, 2.25)
    draw_arrow(ax, 7.5, 1.75, 7.5, 1.35)
    
    # Matching links
    for y in [3.8, 2.9, 2.0, 1.1]:
        draw_arrow(ax, 3.6, y, 6.3, y, color='#90A4AE', style='dashed', lw=1.0)
        
    save_and_close("figure_12_information_flow")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FIGURE 13: RHAN Ecosystem
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def gen_figure_13():
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.text(0, 1.45, "The RHAN Robustness Ecosystem", ha='center', fontsize=14, fontweight='bold')
    
    # Center circle
    center_circle = plt.Circle((0, 0), 0.38, facecolor='#E3F2FD', edgecolor='#1F77B4', lw=3.0, zorder=3)
    ax.add_patch(center_circle)
    ax.text(0, 0, "RHAN\nBackbone", ha='center', va='center', fontsize=11, fontweight='bold', color='#111111', zorder=4)
    
    # Surrounding components (Polar layout)
    components = [
        ("SAIL Invariance", "Self-supervised feature mapping", 90),
        ("TDV Consistency", "Video temporal difference", 50),
        ("CLIP Supervision", "Semantic feature anchoring", 10),
        ("CORnet Teacher", "Cortical brain alignment", 330),
        ("Curriculum TRADES", "Curriculum epsilon scaling", 290),
        ("Prototype Head", "Spherical classification", 250),
        ("Frequency Gating", "Frequency filtering", 210),
        ("Dual Stream", "What vs Where pathway", 170),
        ("Predictive Feedback", "Active prediction error", 130)
    ]
    
    for name, desc, angle in components:
        rad = np.deg2rad(angle)
        cx, cy = np.cos(rad)*1.0, np.sin(rad)*1.0
        
        # Draw box
        draw_box(ax, cx, cy, 0.72, 0.36, name, bg='#F8F9FA', border='#90A4AE', lw=1.2)
        ax.text(cx, cy - 0.22, desc, ha='center', fontsize=6.0, style='italic', color=PALETTE['annotation'])
        
        # Connect to center
        # Arrow points from box boundary to center boundary
        # Box boundary approx 0.8 * unit. Center boundary 0.38.
        x_start, y_start = np.cos(rad)*0.78, np.sin(rad)*0.78
        x_end, y_end = np.cos(rad)*0.39, np.sin(rad)*0.39
        draw_arrow(ax, x_start, y_start, x_end, y_end, color='#78909C', lw=1.2)
        
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.65)
    
    save_and_close("figure_13_rhan_ecosystem")

if __name__ == "__main__":
    funcs = [
        ("figure_1_rhan_architecture", gen_figure_1),
        ("figure_2_biological_inspiration", gen_figure_2),
        ("figure_3_predictive_coding_loop", gen_figure_3),
        ("figure_4_dual_transformer", gen_figure_4),
        ("figure_5_spherical_prototype", gen_figure_5),
        ("figure_6_sail_flowchart", gen_figure_6),
        ("figure_7_geometric_sail", gen_figure_7),
        ("figure_8_tdv_framework", gen_figure_8),
        ("figure_9_adversarial_tdv", gen_figure_9),
        ("figure_10_training_pipeline", gen_figure_10),
        ("figure_11_loss_landscape", gen_figure_11),
        ("figure_12_information_flow", gen_figure_12),
        ("figure_13_rhan_ecosystem", gen_figure_13)
    ]
    
    print("Generating all scientific figures in Light Theme...")
    PALETTE = LIGHT_PALETTE
    for _, func in funcs:
        func()
        
    print("\nGenerating all scientific figures in Dark Theme...")
    PALETTE = DARK_PALETTE
    for _, func in funcs:
        func()
        
    print("\nGeneration complete. Figures are available in figures_v3/ folder.")
