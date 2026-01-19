"""
Script to process SAVEE dataset for Vocal Tone model training.

This script processes all WAV files and extracts MFCC features.
Run this to prepare training data: X (features) and y (labels).
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory (backend) to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from run_models import process_savee_dataset_for_training


def main():
    """Main function to process the dataset."""
    print("=" * 80)
    print("SAVEE Dataset Processor for Vocal Tone Training")
    print("=" * 80)
    print()
    
    try:
        # Process dataset
        print("Processing dataset...")
        X, y, labels_map = process_savee_dataset_for_training(
            dataset_path=None,  # Auto-detect path
            target_sr=16000,
            target_duration_sec=3.0,
            n_mfcc=40,
            show_progress=True
        )
        
        print("\n" + "=" * 80)
        print("RESULTS:")
        print("=" * 80)
        print(f"✅ X shape: {X.shape} (#samples, #features)")
        print(f"✅ y shape: {y.shape} (#samples,)")
        print(f"✅ Number of features per sample: {X.shape[1]} (40 mean + 40 std)")
        print(f"✅ Number of classes: {len(labels_map)}")
        print()
        
        print("Class distribution:")
        for label_idx in sorted(labels_map.keys()):
            label_name = labels_map[label_idx]
            count = np.sum(y == label_idx)
            percentage = (count / len(y)) * 100
            print(f"  {label_name:20s}: {count:4d} samples ({percentage:5.1f}%)")
        
        print()
        print("Feature statistics (first 5 features):")
        print(f"  Mean: {X[:, :5].mean(axis=0)}")
        print(f"  Std:  {X[:, :5].std(axis=0)}")
        print(f"  Min:  {X[:, :5].min(axis=0)}")
        print(f"  Max:  {X[:, :5].max(axis=0)}")
        
        print()
        print("=" * 80)
        print("Dataset ready for training!")
        print("=" * 80)
        print()
        print("You can now use X and y to train your model:")
        print("  X: numpy array of shape", X.shape, "- features matrix")
        print("  y: numpy array of shape", y.shape, "- labels vector")
        print("  labels_map:", labels_map)
        
        return X, y, labels_map
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == '__main__':
    X, y, labels_map = main()

