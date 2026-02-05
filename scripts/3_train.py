"""
Stage 3: Model Training Script

Trains YOLO model with config-based settings.

Features:
- Config + CLI args integration
- Automatic experiment directory creation
- W&B and TensorBoard logging (optional)
- Checkpoint management
- Comprehensive logging

Usage:
    # Train with experiment config
    python scripts/3_train.py --config configs/experiments/exp001_baseline.yaml

    # Override config values
    python scripts/3_train.py \\
        --config configs/experiments/exp001_baseline.yaml \\
        --epochs 100 \\
        --batch_size 32

    # Resume from checkpoint
    python scripts/3_train.py \\
        --config configs/experiments/exp001_baseline.yaml \\
        --resume runs/exp001_baseline_*/checkpoints/last.pt
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import YOLOTrainer
from src.utils import (
    ExperimentLogger,
    create_experiment_dir,
    load_config,
    save_config,
    save_experiment_metadata,
    set_seed,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLO model for pill detection"
    )
    
    # Config file
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment config YAML file'
    )
    parser.add_argument(
        '--base_config',
        type=str,
        default='configs/base.yaml',
        help='Path to base config YAML file (default: configs/base.yaml)'
    )
    
    # Data paths
    parser.add_argument(
        '--data_yaml',
        type=str,
        default='data/yolo_data/data.yaml',
        help='Path to YOLO data.yaml file (default: data/yolo_data/data.yaml)'
    )
    
    # Training overrides
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=None,
        help='Input image size (overrides config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    
    # Model overrides
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name (yolov8n, yolov8s, yolov8m, etc.) (overrides config)'
    )
    parser.add_argument(
        '--pretrained',
        type=bool,
        default=None,
        help='Use pretrained weights (overrides config)'
    )
    
    # Experiment management
    parser.add_argument(
        '--exp_name',
        type=str,
        default=None,
        help='Experiment name (default: from config or "experiment")'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )
    
    # Logging
    parser.add_argument(
        '--use_wandb',
        action='store_true',
        help='Enable W&B logging (overrides config)'
    )
    parser.add_argument(
        '--use_tensorboard',
        action='store_true',
        help='Enable TensorBoard logging (overrides config)'
    )
    
    return parser.parse_args()


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply command-line argument overrides to config.
    
    Args:
        config: Configuration dictionary
        args: Parsed command-line arguments
        
    Returns:
        Updated configuration dictionary
    """
    # Training overrides
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.imgsz is not None:
        config['training']['imgsz'] = args.imgsz
    if args.lr is not None:
        config['training']['optimizer']['lr'] = args.lr
    
    # Model overrides
    if args.model is not None:
        config['model']['name'] = args.model
    if args.pretrained is not None:
        config['model']['yolo']['pretrained'] = args.pretrained
    
    # Seed override
    if args.seed is not None:
        config['seed'] = args.seed
    
    # Logging overrides
    if args.use_wandb:
        config['logging']['wandb']['enabled'] = True
    if args.use_tensorboard:
        config['logging']['tensorboard']['enabled'] = True
    
    return config


def main():
    """Main training function."""
    args = parse_args()
    
    print(f"\n{'='*80}")
    print("STAGE 3: MODEL TRAINING")
    print(f"{'='*80}")
    
    # Load configuration
    print(f"\nLoading configuration...")
    print(f"  Base config: {args.base_config}")
    print(f"  Experiment config: {args.config}")
    
    config = load_config(
        base_config_path=args.base_config,
        exp_config_path=args.config
    )
    
    # Apply CLI overrides
    config = apply_cli_overrides(config, args)
    
    # Set seed for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed, deterministic=True)
    print(f"\n✓ Set random seed: {seed}")
    
    # Get experiment name
    exp_name = args.exp_name
    if exp_name is None:
        exp_name = config.get('experiment', {}).get('name', 'experiment')
    
    # Create experiment directory
    print(f"\nCreating experiment directory...")
    exp_dir, exp_id = create_experiment_dir(
        runs_dir=config.get('output', {}).get('runs_dir', 'runs'),
        exp_name=exp_name
    )
    print(f"  Experiment ID: {exp_id}")
    print(f"  Experiment directory: {exp_dir}")
    
    # Save experiment metadata
    save_experiment_metadata(exp_dir, config, exp_id, exp_name)
    print(f"  ✓ Saved metadata")
    
    # Save config snapshot
    config_snapshot_path = exp_dir / "config_snapshot.yaml"
    save_config(config, str(config_snapshot_path))
    print(f"  ✓ Saved config snapshot: {config_snapshot_path}")
    
    # Initialize logger
    print(f"\nInitializing logger...")
    log_dir = exp_dir / "logs"
    logger = ExperimentLogger(
        log_dir=str(log_dir),
        experiment_name=exp_id,
        config=config,
        use_wandb=config.get('logging', {}).get('wandb', {}).get('enabled', False),
        use_tensorboard=config.get('logging', {}).get('tensorboard', {}).get('enabled', False),
        wandb_project=config.get('logging', {}).get('wandb', {}).get('project'),
        wandb_entity=config.get('logging', {}).get('wandb', {}).get('entity')
    )
    logger.info("Logger initialized")
    
    # Print configuration summary
    logger.info(f"\n{'='*80}")
    logger.info("CONFIGURATION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"\nExperiment: {exp_id} ({exp_name})")
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Epochs: {config['training']['epochs']}")
    logger.info(f"Batch size: {config['training']['batch_size']}")
    logger.info(f"Image size: {config['training']['imgsz']}")
    logger.info(f"Learning rate: {config['training']['optimizer']['lr']}")
    logger.info(f"Optimizer: {config['training']['optimizer']['name']}")
    logger.info(f"Device: {config['training']['device']}")
    logger.info(f"Seed: {seed}")
    
    # Validate data.yaml
    data_yaml_path = Path(args.data_yaml)
    if not data_yaml_path.exists():
        logger.error(f"\nData YAML not found: {data_yaml_path}")
        logger.error("Please run Stage 2 first:")
        logger.error("  python scripts/2_prepare_yolo.py")
        sys.exit(1)
    
    logger.info(f"\nData YAML: {data_yaml_path}")
    
    # Initialize trainer
    logger.info(f"\n{'='*80}")
    logger.info("INITIALIZING TRAINER")
    logger.info(f"{'='*80}")
    
    trainer = YOLOTrainer(
        config=config,
        exp_dir=exp_dir,
        logger=logger
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train model
    try:
        results = trainer.train(data_yaml=str(data_yaml_path))
        
        # Log final results
        logger.info(f"\n{'='*80}")
        logger.info("TRAINING RESULTS")
        logger.info(f"{'='*80}")
        
        # Run validation on best model
        val_metrics = trainer.validate(data_yaml=str(data_yaml_path))
        
        # Log final metrics
        logger.log_metrics(val_metrics)
        
        # Save final info
        best_checkpoint = trainer.get_best_checkpoint_path()
        if best_checkpoint:
            logger.info(f"\nBest checkpoint: {best_checkpoint}")
        
        logger.info(f"\n{'='*80}")
        logger.info("✓ TRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"{'='*80}")
        logger.info(f"\nExperiment directory: {exp_dir}")
        logger.info(f"Checkpoints: {exp_dir / 'checkpoints'}")
        logger.info(f"Logs: {exp_dir / 'logs'}")
        
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Evaluate model:")
        logger.info(f"     python scripts/4_evaluate.py --checkpoint {best_checkpoint}")
        logger.info(f"  2. Create submission:")
        logger.info(f"     python scripts/5_submission.py --checkpoint {best_checkpoint}")
        
    except Exception as e:
        logger.error(f"\n{'='*80}")
        logger.error("✗ TRAINING FAILED")
        logger.error(f"{'='*80}")
        logger.error(f"\nError: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    finally:
        # Close logger
        logger.close()


if __name__ == "__main__":
    main()
