import sys
import os
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    print("Testing model compilation...")
    from phase1_training.model_rhan_v6 import RHANv6
    model = RHANv6(head_type='cosine')
    x = torch.randn(4, 3, 32, 32)
    logits, steps = model(x)
    print(f"  ✓ forward() succeeded: logits {logits.shape}, steps {steps.item():.2f}")
    
    logits, feats, steps = model.forward_with_features(x)
    print(f"  ✓ forward_with_features() succeeded: logits {logits.shape}, feats {feats.shape}, steps {steps.item():.2f}")
    
    # Check separate frequencies method
    x_low, x_high = model.separate_frequencies(x)
    print(f"  ✓ separate_frequencies() succeeded: low {x_low.shape}, high {x_high.shape}")
    
    print("\nTesting pretrain scripts compilation...")
    import py_compile
    py_compile.compile("phase1_training/pretrain_rhan_v6_clip.py", doraise=True)
    print("  ✓ pretrain_rhan_v6_clip.py compiled cleanly.")
    py_compile.compile("phase1_training/train_rhan_v6.py", doraise=True)
    print("  ✓ train_rhan_v6.py compiled cleanly.")
    py_compile.compile("phase1_training/pretrain_noise_estimator.py", doraise=True)
    print("  ✓ pretrain_noise_estimator.py compiled cleanly.")
    
    print("\n=== COMPILATION CHECKS PASSED SUCCESSFULLY ===")
except Exception as e:
    print(f"\n❌ Verification failed with error: {e}")
    sys.exit(1)
