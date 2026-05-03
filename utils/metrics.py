"""
Shared Metrics & Data Loading Utilities
=======================================

PURPOSE:
    Centralized functions for computing accuracy, confidence, and loading the
    adversarial datasets. Used extensively across Phase 4 scripts.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Path to the directory where Phase 2 saved the .npy adversarial datasets
ADV_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'phase2_attacks', 'adv_images')


def accuracy(preds, labels):
    """
    Compute binary accuracy over a batch or dataset.

    1. WHAT: Calculates the percentage of predictions that exactly match true labels.
    2. WHY: Standard performance metric. For human vs CNN comparison, this tells
       us the top-1 hit rate.
    3. OBSERVE: Returns a float between 0.0 and 100.0.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
        
    correct = (preds == labels).sum()
    total = len(labels)
    return (correct / total) * 100.0 if total > 0 else 0.0


def per_class_accuracy(preds, labels, num_classes=10):
    """
    Compute accuracy independently for each class.

    1. WHAT: Breaks down overall accuracy into a 10-element array.
    2. WHY: Helps identify "Texture Bias". If a CNN relies on textures (e.g.,
       fur for cats) while humans rely on shapes, adversarial attacks might
       destroy specific classes much faster for CNNs than humans.
    3. OBSERVE: Returns a numpy array of shape (num_classes,) with values 0-100.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
        
    accs = np.zeros(num_classes)
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            class_correct = (preds[mask] == labels[mask]).sum()
            accs[c] = (class_correct / mask.sum()) * 100.0
            
    return accs


def confidence_from_logits(logits):
    """
    Convert raw model outputs to confidence scores.

    1. WHAT: Applies Softmax to logits, then takes the maximum probability per image.
    2. WHY: Raw logits are unscaled. Softmax converts them to a probability
       distribution (0.0 to 1.0). The max value represents the model's certainty
       in its chosen answer. This serves as the CNN equivalent to the human
       1-10 confidence rating.
    3. OBSERVE: Returns a numpy array of shape (N,) with values in [0.1, 1.0].
    """
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
        
    probs = F.softmax(logits, dim=1)
    conf, _ = torch.max(probs, dim=1)
    return conf.cpu().numpy()


def load_adv_batch(attack, epsilon, return_tensor=True):
    """
    Load an adversarial dataset and its true labels from disk.

    1. WHAT: Reads the .npy files generated in Phase 2.
    2. WHY: Allows Phase 4 analysis scripts to evaluate the CNN precisely on
       the same images that were exported for the human study.
    3. OBSERVE: epsilon='auto' is used for C&W.

    Returns:
        (images, labels) — As torch.Tensors (float32, int64) or numpy arrays.
    """
    if attack.lower() == 'cw':
        img_filename = f"{attack}_images.npy"
    else:
        # epsilon might be passed as float or string
        eps_str = f"{float(epsilon):.2f}"
        img_filename = f"{attack}_eps{eps_str}_images.npy"
        
    img_path = os.path.join(ADV_DATA_DIR, img_filename)
    lbl_path = os.path.join(ADV_DATA_DIR, 'labels.npy')
    
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Missing stimulus file: {img_path}. Did Phase 2 finish?")
        
    images = np.load(img_path)
    labels = np.load(lbl_path)
    
    if return_tensor:
        images = torch.from_numpy(images)
        labels = torch.from_numpy(labels).long()
        
    return images, labels
