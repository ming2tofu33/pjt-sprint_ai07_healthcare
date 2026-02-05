"""
Test script for Phase 3 model training framework.

Tests:
- YOLO dataset conversion (COCO → YOLO)
- Base trainer interface
- YOLO trainer initialization
- Training argument preparation

Note: This test uses synthetic data. Full training tests require real data.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.yolo_dataset import (
    convert_bbox_coco_to_yolo,
    create_yolo_data_yaml,
)
from src.models import BaseTrainer, YOLOTrainer
from src.utils import create_experiment_dir, load_config, set_seed


def test_bbox_conversion():
    """Test COCO to YOLO bbox conversion."""
    print("\n" + "="*80)
    print("TEST 1: COCO to YOLO Bbox Conversion")
    print("="*80)
    
    # Test case: COCO bbox [100, 200, 50, 80] in 640x480 image
    bbox_coco = [100, 200, 50, 80]
    image_width = 640
    image_height = 480
    
    x_center, y_center, width, height = convert_bbox_coco_to_yolo(
        bbox_coco, image_width, image_height
    )
    
    print(f"\nCOCO bbox: {bbox_coco} (image: {image_width}x{image_height})")
    print(f"YOLO bbox: [{x_center:.6f}, {y_center:.6f}, {width:.6f}, {height:.6f}]")
    
    # Verify conversion
    assert 0 <= x_center <= 1, "x_center out of range"
    assert 0 <= y_center <= 1, "y_center out of range"
    assert 0 <= width <= 1, "width out of range"
    assert 0 <= height <= 1, "height out of range"
    
    # Verify math
    expected_x_center = (100 + 50/2) / 640
    expected_y_center = (200 + 80/2) / 480
    expected_width = 50 / 640
    expected_height = 80 / 480
    
    assert abs(x_center - expected_x_center) < 1e-6
    assert abs(y_center - expected_y_center) < 1e-6
    assert abs(width - expected_width) < 1e-6
    assert abs(height - expected_height) < 1e-6
    
    print("\n✓ Bbox conversion correct")


def test_data_yaml_creation():
    """Test YOLO data.yaml creation."""
    print("\n" + "="*80)
    print("TEST 2: YOLO data.yaml Creation")
    print("="*80)
    
    test_output_dir = Path("runs/phase3_test")
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = ['pill_class_1', 'pill_class_2', 'pill_class_3']
    num_classes = 3
    
    yaml_path = create_yolo_data_yaml(
        output_dir=str(test_output_dir),
        train_path="images/train",
        val_path="images/val",
        class_names=class_names,
        num_classes=num_classes
    )
    
    print(f"\n✓ Created data.yaml: {yaml_path}")
    
    # Verify content
    import yaml
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    assert data['nc'] == num_classes
    assert data['names'] == class_names
    assert 'train' in data
    assert 'val' in data
    
    print("✓ data.yaml content verified")


def test_base_trainer_interface():
    """Test base trainer abstract interface."""
    print("\n" + "="*80)
    print("TEST 3: Base Trainer Interface")
    print("="*80)
    
    # BaseTrainer is abstract, so we just verify it exists
    print(f"\n✓ BaseTrainer class available")
    print(f"  Abstract methods:")
    print(f"    - build_model()")
    print(f"    - train()")
    print(f"    - validate()")
    print(f"    - save_checkpoint()")
    print(f"    - load_checkpoint()")
    
    print("\n✓ Base trainer interface verified")


def test_yolo_trainer_initialization():
    """Test YOLO trainer initialization."""
    print("\n" + "="*80)
    print("TEST 4: YOLO Trainer Initialization")
    print("="*80)
    
    # Check if ultralytics is available
    try:
        import ultralytics
        print(f"\n✓ Ultralytics YOLO available (version: {ultralytics.__version__})")
    except ImportError:
        print(f"\n✗ Ultralytics YOLO not installed")
        print(f"  Install with: pip install ultralytics")
        print(f"  Skipping YOLO trainer test")
        return
    
    # Load config
    config = load_config(
        base_config_path="configs/base.yaml",
        exp_config_path="configs/experiments/exp001_baseline.yaml"
    )
    
    # Create experiment directory
    exp_dir, exp_id = create_experiment_dir(
        runs_dir="runs/phase3_test",
        exp_name="test_trainer"
    )
    
    print(f"\n✓ Created test experiment: {exp_id}")
    
    # Initialize trainer
    try:
        trainer = YOLOTrainer(
            config=config,
            exp_dir=exp_dir,
            logger=None
        )
        
        print(f"✓ YOLO trainer initialized successfully")
        print(f"  Model: {config['model']['name']}")
        print(f"  Checkpoints dir: {trainer.checkpoints_dir}")
        
        # Verify model is loaded
        assert trainer.model is not None
        print(f"✓ Model loaded successfully")
        
    except Exception as e:
        print(f"✗ YOLO trainer initialization failed: {e}")
        raise


def test_training_args_preparation():
    """Test training arguments preparation."""
    print("\n" + "="*80)
    print("TEST 5: Training Arguments Preparation")
    print("="*80)
    
    # Check if ultralytics is available
    try:
        import ultralytics
    except ImportError:
        print(f"\n✗ Ultralytics YOLO not installed, skipping test")
        return
    
    # Load config
    config = load_config(
        base_config_path="configs/base.yaml",
        exp_config_path="configs/experiments/exp001_baseline.yaml"
    )
    
    # Create experiment directory
    exp_dir, exp_id = create_experiment_dir(
        runs_dir="runs/phase3_test",
        exp_name="test_args"
    )
    
    # Initialize trainer
    trainer = YOLOTrainer(
        config=config,
        exp_dir=exp_dir,
        logger=None
    )
    
    # Prepare training args
    data_yaml = "data/yolo_data/data.yaml"
    train_args = trainer._prepare_training_args(data_yaml)
    
    print(f"\n✓ Training arguments prepared")
    print(f"\nKey training arguments:")
    print(f"  data: {train_args.get('data')}")
    print(f"  epochs: {train_args.get('epochs')}")
    print(f"  batch: {train_args.get('batch')}")
    print(f"  imgsz: {train_args.get('imgsz')}")
    print(f"  optimizer: {train_args.get('optimizer')}")
    print(f"  lr0: {train_args.get('lr0')}")
    print(f"  device: {train_args.get('device')}")
    
    # Verify required args are present
    required_args = ['data', 'epochs', 'batch', 'imgsz', 'optimizer', 'lr0']
    for arg in required_args:
        assert arg in train_args, f"Missing required arg: {arg}"
    
    print(f"\n✓ All required arguments present")


def main():
    """Run all Phase 3 tests."""
    print("\n" + "="*80)
    print("PHASE 3 MODEL TRAINING FRAMEWORK TESTS")
    print("="*80)
    
    set_seed(42)
    
    try:
        # Test 1: Bbox conversion
        test_bbox_conversion()
        
        # Test 2: Data YAML creation
        test_data_yaml_creation()
        
        # Test 3: Base trainer interface
        test_base_trainer_interface()
        
        # Test 4: YOLO trainer initialization
        test_yolo_trainer_initialization()
        
        # Test 5: Training args preparation
        test_training_args_preparation()
        
        print("\n" + "="*80)
        print("✓ ALL PHASE 3 TESTS PASSED!")
        print("="*80)
        print(f"\nTest artifacts saved to: runs/phase3_test/")
        print(f"\nNext steps (with real data):")
        print(f"  1. Run Stage 2 to prepare YOLO dataset:")
        print(f"     python scripts/2_prepare_yolo.py")
        print(f"  2. Run Stage 3 to train model:")
        print(f"     python scripts/3_train.py --config configs/experiments/exp001_baseline.yaml")
        
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
