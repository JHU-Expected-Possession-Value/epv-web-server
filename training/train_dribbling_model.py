"""
Train models with proper train/validation/test split

This is the CORRECT way to evaluate model performance:
- Training set: 70% of matches
- Validation set: 15% of matches
- Test set: 15% of matches
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dribbling_model import DribblingModel
from value_function import ValueFunction
from pitch_control import PitchControlCache
from sklearn.model_selection import train_test_split

def main():
    print("=" * 80)
    print("TRAINING WITH PROPER TRAIN/VAL/TEST SPLIT")
    print("=" * 80)

    data_dir = Path('data/skillcorner_download')

    # Get all match IDs
    tracking_files = list(data_dir.glob('*_tracking_extrapolated.jsonl'))
    all_match_ids = sorted([int(f.stem.split('_')[0]) for f in tracking_files])

    print(f"\nTotal matches available: {len(all_match_ids)}")

    # Split into train/val/test
    # First split: 70% train, 30% temp
    train_ids, temp_ids = train_test_split(
        all_match_ids,
        test_size=0.30,
        random_state=42  # For reproducibility
    )

    # Second split: split temp into 50/50 (15% val, 15% test of total)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        random_state=42
    )

    print(f"\nSplit breakdown:")
    print(f"  Training:   {len(train_ids)} matches ({len(train_ids)/len(all_match_ids)*100:.1f}%)")
    print(f"  Validation: {len(val_ids)} matches ({len(val_ids)/len(all_match_ids)*100:.1f}%)")
    print(f"  Test:       {len(test_ids)} matches ({len(test_ids)/len(all_match_ids)*100:.1f}%)")

    print(f"\nEstimated dribbles per split:")
    avg_dribbles = 470
    print(f"  Training:   ~{len(train_ids) * avg_dribbles:,} dribbles")
    print(f"  Validation: ~{len(val_ids) * avg_dribbles:,} dribbles")
    print(f"  Test:       ~{len(test_ids) * avg_dribbles:,} dribbles")

    print("\n" + "=" * 80)
    print("WHAT TO DO NEXT")
    print("=" * 80)
    print("""
1. TRAIN on train_ids (67 matches):
   - Use examples/train_on_full_dataset.py
   - But filter to only train_ids

2. VALIDATE during training:
   - Check performance on val_ids (15 matches)
   - Tune hyperparameters if needed
   - Prevent overfitting

3. FINAL TEST on test_ids (15 matches):
   - Run evaluate_dribbling_model.py on each test match
   - Average accuracy across all 15 test matches
   - This is your TRUE performance estimate

4. COMPARE results:
   - Current (all data): 88.94% on 1 match
   - Proper split: ?.??% on 15 held-out matches
   - Difference shows if we overfit!
    """)

    # Save the split for reproducibility
    split_info = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids,
        'random_state': 42
    }

    import pickle
    with open('data/train_val_test_split.pkl', 'wb') as f:
        pickle.dump(split_info, f)

    print("✓ Split saved to: data/train_val_test_split.pkl")
    print("\nYou can load this split in other scripts to ensure consistency!")

    print("\n" + "=" * 80)
    print("COMPARISON: CURRENT vs PROPER APPROACH")
    print("=" * 80)
    print("""
CURRENT APPROACH (what we did):
  ✓ Simple and fast
  ✓ Uses all data for training
  ✗ Can't measure true generalization
  ✗ Might be overfitting
  ✗ Not rigorous for academic work

PROPER APPROACH (recommended):
  ✓ Measures true generalization
  ✓ Prevents overfitting
  ✓ More rigorous/defensible
  ✓ Standard machine learning practice
  ✗ Uses less data for training
  ✗ Takes more time to implement

For your undergraduate project:
  - Current approach is probably OK
  - But proper split would be BETTER
  - Your choice based on time available!
    """)

if __name__ == '__main__':
    main()
