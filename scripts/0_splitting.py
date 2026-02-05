"""
Stage 0: Data Splitting Script

Splits COCO dataset into train and validation sets with stratification.

Features:
- Stratified split by object count (default)
- Stratified split by class distribution
- K-Fold cross-validation support
- Detailed split statistics

Usage:
    # Simple train/val split
    python scripts/0_splitting.py --coco_json data/coco_data/merged_coco.json --output_dir data/splits

    # K-Fold split
    python scripts/0_splitting.py --coco_json data/coco_data/merged_coco.json --output_dir data/splits --kfold 5

    # Custom train ratio
    python scripts/0_splitting.py --coco_json data/coco_data/merged_coco.json --output_dir data/splits --train_ratio 0.85
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.coco_utils import load_coco_json
from src.data.split_utils import (
    kfold_split,
    print_split_statistics,
    save_split_info,
    stratified_split_by_class,
    stratified_split_by_object_count,
)
from src.utils import set_seed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Split COCO dataset into train/val sets with stratification"
    )
    
    # Input/Output
    parser.add_argument(
        '--coco_json',
        type=str,
        required=True,
        help='Path to merged COCO JSON file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/splits',
        help='Output directory for split files (default: data/splits)'
    )
    
    # Split parameters
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Training set ratio (default: 0.8)'
    )
    parser.add_argument(
        '--stratify_by',
        type=str,
        choices=['object_count', 'class'],
        default='object_count',
        help='Stratification method (default: object_count)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # K-Fold parameters
    parser.add_argument(
        '--kfold',
        type=int,
        default=None,
        help='Number of folds for K-Fold cross-validation (default: None, simple split)'
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=None,
        help='Which fold to use as validation (0-indexed, requires --kfold)'
    )
    
    # Additional options
    parser.add_argument(
        '--min_objects',
        type=int,
        default=1,
        help='Minimum objects per image (default: 1)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed statistics'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    print(f"\n{'='*80}")
    print("STAGE 0: DATA SPLITTING")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    print(f"  COCO JSON: {args.coco_json}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Train ratio: {args.train_ratio}")
    print(f"  Stratify by: {args.stratify_by}")
    print(f"  Seed: {args.seed}")
    print(f"  Min objects: {args.min_objects}")
    
    # Load COCO data
    print(f"\nLoading COCO data from {args.coco_json}...")
    coco_data = load_coco_json(args.coco_json)
    print(f"  Total images: {len(coco_data['images'])}")
    print(f"  Total annotations: {len(coco_data['annotations'])}")
    print(f"  Total categories: {len(coco_data['categories'])}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Perform split
    if args.kfold:
        print(f"\n{'='*80}")
        print(f"K-FOLD CROSS-VALIDATION (k={args.kfold})")
        print(f"{'='*80}")
        
        # Generate all folds
        folds = kfold_split(
            coco_data,
            n_folds=args.kfold,
            seed=args.seed,
            stratify_by=args.stratify_by
        )
        
        # Save all folds
        for fold_idx, (train_ids, val_ids) in enumerate(folds):
            fold_dir = output_dir / f"fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            
            split_path = fold_dir / "split_info.json"
            save_split_info(
                train_ids,
                val_ids,
                str(split_path),
                metadata={
                    'fold': fold_idx,
                    'total_folds': args.kfold,
                    'stratify_by': args.stratify_by,
                    'seed': args.seed
                }
            )
            
            print(f"\nFold {fold_idx}:")
            print(f"  Train: {len(train_ids)} images")
            print(f"  Val:   {len(val_ids)} images")
            print(f"  Saved to: {split_path}")
            
            if args.verbose or (args.fold is not None and fold_idx == args.fold):
                print_split_statistics(coco_data, train_ids, val_ids)
        
        # If specific fold requested, also save to main split directory
        if args.fold is not None:
            if args.fold >= args.kfold:
                raise ValueError(f"Fold {args.fold} out of range (max: {args.kfold - 1})")
            
            train_ids, val_ids = folds[args.fold]
            split_path = output_dir / "split_info.json"
            save_split_info(
                train_ids,
                val_ids,
                str(split_path),
                metadata={
                    'fold': args.fold,
                    'total_folds': args.kfold,
                    'stratify_by': args.stratify_by,
                    'seed': args.seed
                }
            )
            print(f"\n{'='*80}")
            print(f"Using Fold {args.fold} as default split")
            print(f"{'='*80}")
            print(f"  Saved to: {split_path}")
    
    else:
        print(f"\n{'='*80}")
        print(f"TRAIN/VAL SPLIT (ratio={args.train_ratio})")
        print(f"{'='*80}")
        
        # Perform stratified split
        if args.stratify_by == 'object_count':
            train_ids, val_ids = stratified_split_by_object_count(
                coco_data,
                train_ratio=args.train_ratio,
                seed=args.seed,
                min_objects_per_split=args.min_objects
            )
        else:  # class
            train_ids, val_ids = stratified_split_by_class(
                coco_data,
                train_ratio=args.train_ratio,
                seed=args.seed
            )
        
        # Save split info
        split_path = output_dir / "split_info.json"
        save_split_info(
            train_ids,
            val_ids,
            str(split_path),
            metadata={
                'train_ratio': args.train_ratio,
                'stratify_by': args.stratify_by,
                'seed': args.seed,
                'min_objects': args.min_objects
            }
        )
        
        print(f"\nSplit completed:")
        print(f"  Train: {len(train_ids)} images")
        print(f"  Val:   {len(val_ids)} images")
        print(f"  Saved to: {split_path}")
        
        # Print statistics
        if args.verbose:
            print_split_statistics(coco_data, train_ids, val_ids)
    
    print(f"\n{'='*80}")
    print("✓ STAGE 0 COMPLETED")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
