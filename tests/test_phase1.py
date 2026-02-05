"""
Test script for Phase 1 utilities.

Tests:
- Config loading and merging
- Experiment directory creation
- Logger initialization
- Seed setting
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import (
    create_experiment_dir,
    get_next_experiment_number,
    load_config,
    print_config,
    set_seed,
    ExperimentLogger,
    save_config,
)


def test_config_loading():
    """Test config loading and merging."""
    print("\n" + "="*80)
    print("TEST 1: Config Loading and Merging")
    print("="*80)
    
    # Load base config
    base_config_path = "configs/base.yaml"
    exp_config_path = "configs/experiments/exp001_baseline.yaml"
    
    config = load_config(base_config_path, exp_config_path)
    
    print("\n✓ Successfully loaded and merged configs")
    print(f"  - Project name: {config['project']['name']}")
    print(f"  - Model: {config['model']['name']}")
    print(f"  - Epochs: {config['training']['epochs']}")
    print(f"  - Batch size: {config['training']['batch_size']}")
    print(f"  - Seed: {config['seed']}")
    
    # Print full config
    print("\nFull Configuration:")
    print_config(config)
    
    return config


def test_experiment_creation():
    """Test experiment directory creation."""
    print("\n" + "="*80)
    print("TEST 2: Experiment Directory Creation")
    print("="*80)
    
    # Get next experiment number
    next_exp = get_next_experiment_number("runs")
    print(f"\n✓ Next experiment number: {next_exp}")
    
    # Create experiment directory
    exp_dir, exp_id = create_experiment_dir(
        runs_dir="runs",
        exp_name="test_baseline"
    )
    
    print(f"✓ Created experiment directory:")
    print(f"  - ID: {exp_id}")
    print(f"  - Path: {exp_dir}")
    print(f"  - Subdirectories: {[d.name for d in exp_dir.iterdir() if d.is_dir()]}")
    
    return exp_dir, exp_id


def test_seed_setting():
    """Test seed setting for reproducibility."""
    print("\n" + "="*80)
    print("TEST 3: Seed Setting")
    print("="*80)
    
    seed = 42
    set_seed(seed, deterministic=True)
    
    print(f"\n✓ Set seed to {seed}")
    print("✓ Configured for deterministic behavior")
    
    # Test random number generation
    import numpy as np
    import random
    
    try:
        import torch
        torch_rand = torch.rand(3)
        print(f"\n  Sample random numbers (should be reproducible):")
        print(f"  - PyTorch: {torch_rand.tolist()}")
    except ImportError:
        print(f"\n  PyTorch not available, skipping torch random test")
    
    np_rand = np.random.rand(3)
    py_rand = [random.random() for _ in range(3)]
    
    print(f"  - NumPy: {np_rand.tolist()}")
    print(f"  - Python: {py_rand}")


def test_logger(exp_dir, config):
    """Test logger initialization."""
    print("\n" + "="*80)
    print("TEST 4: Logger Initialization")
    print("="*80)
    
    log_dir = exp_dir / "logs"
    
    logger = ExperimentLogger(
        log_dir=str(log_dir),
        experiment_name="test_experiment",
        config=config,
        use_wandb=False,  # Disable W&B for testing
        use_tensorboard=False  # Disable TensorBoard for testing
    )
    
    print(f"\n✓ Initialized logger")
    print(f"  - Log directory: {log_dir}")
    
    # Test logging
    logger.info("This is a test info message")
    logger.warning("This is a test warning message")
    
    # Test metric logging
    test_metrics = {
        'train/loss': 0.5,
        'train/mAP': 0.75,
        'val/loss': 0.6,
        'val/mAP': 0.72
    }
    logger.log_metrics(test_metrics, step=1)
    
    print("✓ Logged test messages and metrics")
    
    # Close logger
    logger.close()
    print("✓ Closed logger")
    
    return logger


def test_config_save(exp_dir, config):
    """Test saving config snapshot."""
    print("\n" + "="*80)
    print("TEST 5: Config Snapshot Save")
    print("="*80)
    
    config_snapshot_path = exp_dir / "config_snapshot.yaml"
    save_config(config, str(config_snapshot_path))
    
    print(f"\n✓ Saved config snapshot to: {config_snapshot_path}")
    print(f"  - File size: {config_snapshot_path.stat().st_size} bytes")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PHASE 1 UTILITY TESTS")
    print("="*80)
    
    try:
        # Test 1: Config loading
        config = test_config_loading()
        
        # Test 2: Experiment creation
        exp_dir, exp_id = test_experiment_creation()
        
        # Test 3: Seed setting
        test_seed_setting()
        
        # Test 4: Logger
        logger = test_logger(exp_dir, config)
        
        # Test 5: Config save
        test_config_save(exp_dir, config)
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        print(f"\nTest artifacts saved to: {exp_dir}")
        
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
