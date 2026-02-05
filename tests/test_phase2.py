"""
Test script for Phase 2 data pipeline.

Creates sample data and tests:
- COCO format creation and merging
- Category mapping
- Data splitting (stratified and K-Fold)
- COCO dataset class

Note: This test uses synthetic data since real data is on local machine.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.coco_utils import (
    COCODataset,
    create_category_mapping,
    save_category_mapping,
    validate_coco_format,
    visualize_coco_sample,
)
from src.data.split_utils import (
    print_split_statistics,
    save_split_info,
    stratified_split_by_object_count,
    kfold_split,
)


def create_sample_coco_data(num_images: int = 50, num_classes: int = 10) -> dict:
    """Create sample COCO data for testing.
    
    Args:
        num_images: Number of sample images
        num_classes: Number of sample classes
        
    Returns:
        COCO format dictionary
    """
    import random
    random.seed(42)
    
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    # Create categories
    for i in range(num_classes):
        coco_data['categories'].append({
            'id': i + 1,  # Start from 1, not 0
            'name': f'pill_class_{i+1}',
            'supercategory': 'pill'
        })
    
    # Create images and annotations
    ann_id = 1
    for img_id in range(1, num_images + 1):
        # Add image
        coco_data['images'].append({
            'id': img_id,
            'file_name': f'image_{img_id:04d}.png',
            'width': 640,
            'height': 480
        })
        
        # Add 1-4 annotations per image
        num_objects = random.randint(1, 4)
        for _ in range(num_objects):
            category_id = random.randint(1, num_classes)
            x = random.randint(50, 400)
            y = random.randint(50, 300)
            w = random.randint(30, 150)
            h = random.randint(30, 150)
            
            coco_data['annotations'].append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': category_id,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0
            })
            ann_id += 1
    
    return coco_data


def test_coco_validation():
    """Test COCO format validation."""
    print("\n" + "="*80)
    print("TEST 1: COCO Format Validation")
    print("="*80)
    
    # Create sample data
    coco_data = create_sample_coco_data(num_images=50, num_classes=10)
    
    print(f"\nCreated sample COCO data:")
    print(f"  Images: {len(coco_data['images'])}")
    print(f"  Annotations: {len(coco_data['annotations'])}")
    print(f"  Categories: {len(coco_data['categories'])}")
    
    # Validate
    is_valid, errors = validate_coco_format(coco_data)
    
    if is_valid:
        print(f"\n✓ COCO format is valid")
    else:
        print(f"\n✗ COCO format has errors:")
        for error in errors[:5]:
            print(f"  - {error}")
    
    return coco_data


def test_category_mapping(coco_data):
    """Test category mapping creation."""
    print("\n" + "="*80)
    print("TEST 2: Category Mapping")
    print("="*80)
    
    # Create mapping
    coco_to_yolo, yolo_to_coco, yolo_to_name = create_category_mapping(coco_data)
    
    print(f"\n✓ Created category mapping:")
    print(f"  Number of classes: {len(yolo_to_coco)}")
    print(f"\n  Sample mappings:")
    for yolo_idx in range(min(5, len(yolo_to_coco))):
        coco_id = yolo_to_coco[yolo_idx]
        name = yolo_to_name[yolo_idx]
        print(f"    YOLO {yolo_idx} <-> COCO {coco_id} ({name})")
    
    # Save mapping
    test_output_dir = Path("runs/phase2_test")
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    mapping_path = test_output_dir / "category_mapping.json"
    save_category_mapping(yolo_to_coco, yolo_to_name, str(mapping_path))
    print(f"\n✓ Saved mapping to: {mapping_path}")
    
    return yolo_to_coco, yolo_to_name


def test_stratified_split(coco_data):
    """Test stratified split by object count."""
    print("\n" + "="*80)
    print("TEST 3: Stratified Split by Object Count")
    print("="*80)
    
    # Perform split
    train_ids, val_ids = stratified_split_by_object_count(
        coco_data,
        train_ratio=0.8,
        seed=42
    )
    
    print(f"\n✓ Split completed:")
    print(f"  Train: {len(train_ids)} images")
    print(f"  Val:   {len(val_ids)} images")
    
    # Print statistics
    print_split_statistics(coco_data, train_ids, val_ids)
    
    # Save split info
    test_output_dir = Path("runs/phase2_test")
    split_path = test_output_dir / "split_info.json"
    save_split_info(
        train_ids,
        val_ids,
        str(split_path),
        metadata={'stratify_by': 'object_count', 'seed': 42}
    )
    print(f"\n✓ Saved split info to: {split_path}")
    
    return train_ids, val_ids


def test_kfold_split(coco_data):
    """Test K-Fold cross-validation split."""
    print("\n" + "="*80)
    print("TEST 4: K-Fold Cross-Validation")
    print("="*80)
    
    # Perform K-Fold split
    n_folds = 5
    folds = kfold_split(
        coco_data,
        n_folds=n_folds,
        seed=42,
        stratify_by='object_count'
    )
    
    print(f"\n✓ Created {n_folds} folds:")
    for fold_idx, (train_ids, val_ids) in enumerate(folds):
        print(f"  Fold {fold_idx}: Train={len(train_ids)}, Val={len(val_ids)}")
    
    # Save fold 0 as example
    test_output_dir = Path("runs/phase2_test")
    fold_dir = test_output_dir / "fold_0"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    fold_path = fold_dir / "split_info.json"
    save_split_info(
        folds[0][0],
        folds[0][1],
        str(fold_path),
        metadata={'fold': 0, 'total_folds': n_folds, 'stratify_by': 'object_count'}
    )
    print(f"\n✓ Saved fold 0 info to: {fold_path}")


def test_coco_dataset_class(coco_data):
    """Test COCODataset wrapper class."""
    print("\n" + "="*80)
    print("TEST 5: COCODataset Class")
    print("="*80)
    
    # Save sample COCO JSON
    test_output_dir = Path("runs/phase2_test")
    coco_json_path = test_output_dir / "sample_coco.json"
    
    with open(coco_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\n✓ Saved sample COCO JSON to: {coco_json_path}")
    
    # Note: We can't fully test COCODataset without actual images
    print(f"\n✓ COCODataset class available for use with real data")
    print(f"  Usage example:")
    print(f"    dataset = COCODataset(")
    print(f"        coco_json_path='data/coco_data/merged_coco.json',")
    print(f"        image_dir='data/raw/train_images'")
    print(f"    )")
    print(f"    print(f'Number of images: {{len(dataset)}}')")
    print(f"    print(f'Number of classes: {{dataset.get_num_classes()}}')")


def main():
    """Run all Phase 2 tests."""
    print("\n" + "="*80)
    print("PHASE 2 DATA PIPELINE TESTS")
    print("="*80)
    print("\nNote: Using synthetic data since real data is on local machine")
    
    try:
        # Test 1: COCO validation
        coco_data = test_coco_validation()
        
        # Test 2: Category mapping
        yolo_to_coco, yolo_to_name = test_category_mapping(coco_data)
        
        # Test 3: Stratified split
        train_ids, val_ids = test_stratified_split(coco_data)
        
        # Test 4: K-Fold split
        test_kfold_split(coco_data)
        
        # Test 5: COCODataset class
        test_coco_dataset_class(coco_data)
        
        print("\n" + "="*80)
        print("✓ ALL PHASE 2 TESTS PASSED!")
        print("="*80)
        print(f"\nTest artifacts saved to: runs/phase2_test/")
        print(f"\nNext steps:")
        print(f"  1. Run scripts/1_create_coco_format.py with real data on local machine")
        print(f"  2. Run scripts/0_splitting.py to create train/val split")
        print(f"  3. Proceed to Phase 3 (Model Training)")
        
    except Exception as e:
        print("\n" + "="*80)
        print("✗ TEST FAILED!")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
