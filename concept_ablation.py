#!/usr/bin/env python3
"""
Concept Activation Quality Ablation: Phase B vs Phase C RHAN Checkpoints
Compares concept representation quality between curriculum phases for RHAN-CBM preparation.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add project paths
script_dir = os.path.dirname(__file__)
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(script_dir, '..'))

from phase1_training.model_rhan_v5 import RHANv5
from phase1_training.dataset import get_dataloaders

def load_rhan_checkpoint(checkpoint_path, device, head_type='cosine'):
    """Load RHAN model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    model = RHANv5(head_type=head_type).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def extract_features(model, dataloader, device):
    """Extract CLS token features (post recurrent feedback) from RHANv5 model."""
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass to get features
            cls_token = model.get_feature_vector(data)  # (B, 512)
            
            features_list.append(cls_token.cpu().numpy())
            labels_list.append(target.cpu().numpy())
            
            if batch_idx % 10 == 0:
                print(f"  Processed {batch_idx * len(data)} samples...")
    
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    print(f"Extracted features shape: {features.shape}")
    return features, labels

def train_linear_concept_probe(features, labels, concept_labels, concept_name):
    """Train a linear probe to predict a binary concept from features."""
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Train logistic regression
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(features_scaled, concept_labels)
    
    # Predict and evaluate
    concept_pred = clf.predict(features_scaled)
    concept_acc = accuracy_score(concept_labels, concept_pred)
    
    return {
        'accuracy': concept_acc,
        'classifier': clf,
        'scaler': scaler,
        'predictions': concept_pred
    }

def evaluate_concept_quality(phase_b_features, phase_b_labels, 
                           phase_c_features, phase_c_labels,
                           concept_definitions):
    """Evaluate concept representation quality for both phases."""
    results = {}
    
    print("\n" + "="*60)
    print("CONCEPT ACTIVATION QUALITY ABLATION")
    print("="*60)
    
    for concept_name, concept_def_fn in concept_definitions.items():
        print(f"\nEvaluating concept: {concept_name}")
        
        # Get concept labels for full dataset
        phase_b_concept_labels = np.array([concept_def_fn(label) for label in phase_b_labels])
        phase_c_concept_labels = np.array([concept_def_fn(label) for label in phase_c_labels])
        
        # Check class balance
        b_pos = np.sum(phase_b_concept_labels)
        b_neg = len(phase_b_concept_labels) - b_pos
        c_pos = np.sum(phase_c_concept_labels)
        c_neg = len(phase_c_concept_labels) - c_pos
        
        print(f"  Phase B - Positives: {b_pos}/{len(phase_b_concept_labels)} ({100*b_pos/len(phase_b_concept_labels):.1f}%)")
        print(f"  Phase C - Positives: {c_pos}/{len(phase_c_concept_labels)} ({100*c_pos/len(phase_c_concept_labels):.1f}%)")
        
        # Skip if too imbalanced
        if b_pos < 10 or b_neg < 10 or c_pos < 10 or c_neg < 10:
            print(f"  Skipping {concept_name} - insufficient samples for balanced evaluation")
            continue
        
        # Train probes for both phases
        print(f"  Training Phase B probe...")
        b_result = train_linear_concept_probe(
            phase_b_features, phase_b_labels, 
            phase_b_concept_labels, concept_name
        )
        
        print(f"  Training Phase C probe...")
        c_result = train_linear_concept_probe(
            phase_c_features, phase_c_labels, 
            phase_c_concept_labels, concept_name
        )
        
        # Store results
        results[concept_name] = {
            'phase_b': {
                'accuracy': b_result['accuracy'],
                'classifier': b_result['classifier'],
                'scaler': b_result['scaler']
            },
            'phase_c': {
                'accuracy': c_result['accuracy'],
                'classifier': c_result['classifier'],
                'scaler': c_result['scaler']
            },
            'improvement': c_result['accuracy'] - b_result['accuracy']
        }
        
        print(f"  Phase B concept accuracy: {b_result['accuracy']:.3f}")
        print(f"  Phase C concept accuracy: {c_result['accuracy']:.3f}")
        print(f"  Improvement: {c_result['accuracy'] - b_result['accuracy']:+.3f}")
    
    return results

def print_summary(results):
    """Print formatted summary of results."""
    print("\n" + "="*60)
    print("SUMMARY: CONCEPT REPRESENTATION QUALITY")
    print("="*60)
    
    # Sort by improvement
    sorted_concepts = sorted(results.items(), 
                           key=lambda x: x[1]['improvement'], 
                           reverse=True)
    
    print(f"{'Concept':<20} {'Phase B':<10} {'Phase C':<10} {'Improvement':<12}")
    print("-" * 60)
    
    for concept_name, result in sorted_concepts:
        b_acc = result['phase_b']['accuracy']
        c_acc = result['phase_c']['accuracy']
        impr = result['improvement']
        print(f"{concept_name:<20} {b_acc:<10.3f} {c_acc:<10.3f} {impr:+.3f}")
    
    # Overall statistics
    improvements = [result['improvement'] for result in results.values()]
    print("-" * 60)
    print(f"Mean improvement: {np.mean(improvements):+.3f}")
    print(f"Std improvement:  {np.std(improvements):.3f}")
    print(f"Concepts improved: {sum(1 for imp in improvements if imp > 0)}/{len(improvements)}")
    
    # Key findings for CBM decision
    print("\n" + "="*60)
    print("KEY FINDINGS FOR RHAN-CBM DECISION")
    print("="*60)
    
    better_phase_c = sum(1 for result in results.values() if result['improvement'] > 0)
    total_concepts = len(results)
    
    if better_phase_c > total_concepts * 0.6:
        print("✅ RECOMMENDATION: Phase C features show better concept representation")
        print("   → Use rhan_trades_phase_c_final.pth for RHAN-CBM")
        print("   → Higher robust accuracy correlates with better concept learning")
    elif better_phase_c < total_concepts * 0.4:
        print("✅ RECOMMENDATION: Phase B features show better concept representation") 
        print("   → Use rhan_trades_phase_b_final.pth for RHAN-CBM")
        print("   → Better generalization/concept stability outweighs robustness gain")
    else:
        print("⚠️  INCONCLUSIVE: Mixed results between phases")
        print("   → Consider ensemble features or further ablation")
        print("   → Check if specific concept types favor one phase")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths to checkpoints
    phase_b_ckpt = os.path.join(script_dir, 'checkpoints', 'rhan_trades_phase_b_final.pth')
    phase_c_ckpt = os.path.join(script_dir, 'checkpoints', 'rhan_trades_phase_c_final.pth')
    
    # Verify checkpoints exist
    if not os.path.exists(phase_b_ckpt):
        print(f"ERROR: Phase B checkpoint not found at {phase_b_ckpt}")
        return
    if not os.path.exists(phase_c_ckpt):
        print(f"ERROR: Phase C checkpoint not found at {phase_c_ckpt}")
        return
    
    # Load models
    print("\nLoading models...")
    model_b = load_rhan_checkpoint(phase_b_ckpt, device)
    model_c = load_rhan_checkpoint(phase_c_ckpt, device)
    
    # Load dataset (use same settings as training)
    print("\nLoading CIFAR-10 dataset...")
    _, testloader_raw = get_dataloaders(batch_size=128, num_workers=4, model_name='resnet')
    testloader = DataLoader(testloader_raw.dataset, batch_size=128, shuffle=False,
                           num_workers=4, pin_memory=True, persistent_workers=False)
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Define concept set focused on the geometric overlap problem
    # These concepts directly address why automobile/horse/truck collapse
    concept_definitions = {
        # Vehicle-specific concepts
        'has_wheels': lambda label: 1 if label in [1, 9] else 0,  # automobile, truck
        'has_cab': lambda label: 1 if label in [1, 9] else 0,     # automobile, truck (has cabin)
        'vehicle_body': lambda label: 1 if label in [1, 9] else 0, # automobile, truck
        
        # Animal-specific concepts  
        'quadrupedal': lambda label: 1 if label in [2, 3, 4, 5, 6, 7] else 0,  # bird(2), cat(3), deer(4), dog(5), frog(6), horse(7)
        'has_tail': lambda label: 1 if label in [2, 3, 4, 5, 6, 7] else 0,     # most animals except maybe frog?
        'pointed_ears': lambda label: 1 if label in [3, 4, 5, 7] else 0,        # cat, deer, dog, horse
        
        # Shape/proportion concepts (addressing the geometric overlap)
        'elongated_body': lambda label: 1 if label in [1, 2, 4, 5, 6, 7, 9] else 0,  # most except bird, frog, ship?
        'tall_profile': lambda label: 1 if label in [3, 4, 5, 7] else 0,              # cat, deer, dog, horse
        'wide_profile': lambda label: 1 if label in [0, 1, 6, 8, 9] else 0,           # airplane, automobile, frog, ship, truck
        
        # Additional discriminative concepts
        'ground_vehicle': lambda label: 1 if label in [1, 9] else 0,                  # automobile, truck
        'flying': lambda label: 1 if label in [0, 8] else 0,                          # airplane, ship
        'water_vehicle': lambda label: 1 if label == 8 else 0,                        # ship
    }
    
    print(f"Defined {len(concept_definitions)} concepts for evaluation")
    
    # Extract features from both models (using subset for faster iteration)
    print("\nExtracting features from Phase B model...")
    # Use subset for reasonable timing - full 10k takes too long for iterative dev
    subset_indices = list(range(0, min(2000, len(testloader.dataset))))  # First 2k samples
    subset_testloader = DataLoader(
        Subset(testloader.dataset, subset_indices),
        batch_size=128, shuffle=False, num_workers=2
    )
    
    phase_b_features, phase_b_labels = extract_features(model_b, subset_testloader, device)
    print("Extracting features from Phase C model...")
    phase_c_features, phase_c_labels = extract_features(model_c, subset_testloader, device)
    
    # Verify labels match
    assert np.array_equal(phase_b_labels, phase_c_labels), "Label mismatch between phases!"
    print(f"✓ Label verification passed - {len(phase_b_labels)} samples")
    
    # Run concept quality evaluation
    results = evaluate_concept_quality(
        phase_b_features, phase_b_labels,
        phase_c_features, phase_c_labels,
        concept_definitions
    )
    
    # Print summary and recommendations
    print_summary(results)
    
    # Save results for later reference
    results_path = os.path.join(script_dir, 'concept_ablation_results.npz')
    # Convert results to savable format
    save_dict = {}
    for concept_name, result in results.items():
        save_dict[f'{concept_name}_phase_b_acc'] = result['phase_b']['accuracy']
        save_dict[f'{concept_name}_phase_c_acc'] = result['phase_c']['accuracy']
        save_dict[f'{concept_name}_improvement'] = result['improvement']
    
    np.savez(results_path, **save_dict)
    print(f"\n💾 Results saved to: {results_path}")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Review which concepts show clear phase differences")
    print("2. If Phase C wins: proceed with rhan_trades_phase_c_final.pth for CBM") 
    print("3. If Phase B wins: proceed with rhan_trades_phase_b_final.pth for CBM")
    print("4. If mixed: consider which concepts matter most for your CBM goal")
    print("5. Next: Build actual CBM head using winning feature extractor")

if __name__ == "__main__":
    main()