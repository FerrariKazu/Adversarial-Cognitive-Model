import os
import sys
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import psutil

# --- Memory Monitor ---
ram_gb = psutil.virtual_memory().available / 1e9
if ram_gb < 4.0:
    print(f"WARNING: Only {ram_gb:.1f}GB RAM available. High crash risk.")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1_training'))

from phase2_attacks.generate_adv_all_models import MODELS
from phase1_training.dataset import CLASSES

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'figures', 'combined', 'latent_space')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_SAMPLES_PER_CLASS = 50 # Total 500

def get_embeddings(model, images_np, labels_np, device):
    """Extracts normalized feature vectors for given images."""
    model.eval()
    embeddings = []
    
    # We want 50 images per class
    indices = []
    for c in range(10):
        idx = np.where(labels_np == c)[0][:NUM_SAMPLES_PER_CLASS]
        indices.extend(idx)
    
    indices = np.array(indices)
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_imgs = torch.tensor(images_np[batch_idx], device=device)
            # get_feature_vector returns [B, dim]
            features = model.get_feature_vector(batch_imgs)
            # Flatten if necessary (ResNet might return [B, 512, 1, 1])
            features = features.view(features.size(0), -1)
            embeddings.append(features.cpu().numpy())
            del batch_imgs, features
            
    embeddings = np.concatenate(embeddings)
    # L2 Normalization for fair distance/t-SNE
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-12)
    
    return embeddings, labels_np[indices], indices

def plot_tsne(axes, embeddings_2d, labels, title, show_arrows=None, clean_2d=None):
    """Scatter plot of t-SNE embeddings."""
    scatter = axes.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                           c=labels, cmap='tab10', alpha=0.6, s=30)
    
    # Class labels at centroids
    for c in range(10):
        mask = labels == c
        centroid = np.mean(embeddings_2d[mask], axis=0)
        axes.text(centroid[0], centroid[1], CLASSES[c].upper(), 
                  fontsize=10, fontweight='bold', ha='center', va='center',
                  bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    if show_arrows is not None and clean_2d is not None:
        # Show arrows for 10 representative images (1 per class)
        for c in range(10):
            idx = np.where(labels == c)[0][0] # First image of each class
            start = clean_2d[idx]
            end = embeddings_2d[idx]
            axes.annotate("", xy=end, xytext=start,
                          arrowprops=dict(arrowstyle="->", color='black', alpha=0.8, lw=1.5))

    axes.set_title(title, fontsize=14)
    axes.set_xticks([]); axes.set_yticks([])
    return scatter

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results = {}
    
    for m_name in ['resnet', 'vit']:
        print(f"\nProcessing Latent Space for {m_name.upper()}...")
        cfg = MODELS[m_name]
        
        # Load Model
        model = cfg['class']().to(DEVICE)
        model.load_state_dict(torch.load(cfg['ckpt'], map_location=DEVICE))
        
        # Load Data
        img_dir = cfg['out']
        labels_all = np.load(os.path.join(img_dir, 'labels.npy'))
        clean_all = np.load(os.path.join(img_dir, 'pgd_eps0.00_images.npy'), mmap_mode='r')
        adv_all = np.load(os.path.join(img_dir, 'pgd_eps0.10_images.npy'), mmap_mode='r')
        
        # Extract Embeddings
        print("  Extracting feature vectors...")
        clean_emb, labels, _ = get_embeddings(model, clean_all, labels_all, DEVICE)
        adv_emb, _, _ = get_embeddings(model, adv_all, labels_all, DEVICE)
        
        # Compute High-Dim Displacement (L2)
        displacement = np.linalg.norm(adv_emb - clean_emb, axis=1)
        mean_disp = np.mean(displacement)
        print(f"  Mean high-dim displacement: {mean_disp:.4f}")
        results[m_name] = mean_disp
        
        # Run t-SNE on combined set for unified layout
        print("  Running t-SNE (1000 samples)...")
        combined = np.concatenate([clean_emb, adv_emb], axis=0)
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
        combined_2d = tsne.fit_transform(combined)
        
        clean_2d = combined_2d[:500]
        adv_2d = combined_2d[500:]
        
        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        plot_tsne(axes[0], clean_2d, labels, f"{m_name.upper()} Feature Space — Clean")
        plot_tsne(axes[1], adv_2d, labels, f"{m_name.upper()} Feature Space — PGD ε=0.10", 
                  show_arrows=True, clean_2d=clean_2d)
        
        plt.suptitle(f"Latent Space Topology Analysis: {m_name.upper()}", fontsize=20, y=1.02)
        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, f"{m_name}_tsne.png")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved t-SNE to {out_path}")
        
        del model
        torch.cuda.empty_cache()

    # --- Side-by-Side Comparison ---
    plt.figure(figsize=(8, 6), dpi=150)
    names = list(results.keys())
    disps = [results[n] for n in names]
    plt.bar([n.upper() for n in names], disps, color=['#E94560', '#7C3AED'])
    plt.title('Mean Latent Space Displacement (Clean → ε=0.10)', fontsize=14)
    plt.ylabel('Mean L2 Distance (Normalized Features)')
    plt.grid(axis='y', alpha=0.3)
    
    comp_path = os.path.join(OUTPUT_DIR, 'displacement_comparison.bar.png')
    plt.savefig(comp_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved displacement comparison to {comp_path}")

    # --- SCIENTIFIC COMMENTARY ---
    # 1. t-SNE (t-Distributed Stochastic Neighbor Embedding)
    #    t-SNE is a non-linear dimensionality reduction technique that excels at 
    #    preserving local neighborhood structure. If images cluster well, the 
    #    model has learned distinct semantic "concepts" for the classes.
    #
    # 2. FEATURE SPACE DISPLACEMENT
    #    The "arrows" in the adversarial plots show how each image moves in the 
    #    latent space under attack. If an image moves far and enters a different 
    #    cluster, it is "deeply fooled"—the model's internal concept of that 
    #    image has completely shifted.
    #
    # 3. ResNet vs ViT
    #    We expect ViT adversarial examples to stay closer to their correct 
    #    clusters (smaller displacement) than ResNet. This is because ViT's 
    #    global attention is harder to completely disrupt with local noise; 
    #    the model might classify it wrong at the surface level, but its 
    #    latent representation remains somewhat anchored to the global shape.
    #
    # 4. COGNITIVE ANALOG
    #    This maps to the difference between a "perceptual error" (seeing a 
    #    puddle as a hole) and a "categorical breakdown" (forgetting what a 
    #    hole is). High displacement means the model's fundamental "concept" 
    #    of the object has been hijacked.

if __name__ == '__main__':
    main()
