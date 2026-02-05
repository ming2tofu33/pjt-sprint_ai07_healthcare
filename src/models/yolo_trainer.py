"""
YOLO trainer implementation using Ultralytics YOLO.

Supports:
- YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x
- Config-based training
- Custom callbacks
- W&B and TensorBoard integration (through Ultralytics)
"""

import shutil
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from .base_trainer import BaseTrainer


class YOLOTrainer(BaseTrainer):
    """YOLO model trainer using Ultralytics."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        exp_dir: Path,
        logger: Optional[Any] = None
    ):
        """Initialize YOLO trainer.
        
        Args:
            config: Configuration dictionary
            exp_dir: Experiment directory
            logger: Logger instance (optional)
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "Ultralytics YOLO is not installed. "
                "Install with: pip install ultralytics"
            )
        
        super().__init__(config, exp_dir, logger)
        
        # Extract YOLO-specific config
        self.model_config = config.get('model', {})
        self.training_config = config.get('training', {})
        self.augmentation_config = config.get('augmentation', {})
        
        # Build model
        self.build_model()
    
    def build_model(self) -> None:
        """Build YOLO model from config."""
        model_name = self.model_config.get('name', 'yolov8n')
        pretrained = self.model_config.get('yolo', {}).get('pretrained', True)
        pretrained_weights = self.model_config.get('yolo', {}).get('pretrained_weights')
        
        self.log_info(f"\nBuilding YOLO model: {model_name}")
        self.log_info(f"  Pretrained: {pretrained}")
        
        # Load model
        if pretrained_weights:
            # Load custom pretrained weights
            self.log_info(f"  Loading custom weights: {pretrained_weights}")
            self.model = YOLO(pretrained_weights)
        elif pretrained:
            # Load official pretrained weights (e.g., yolov8n.pt)
            model_file = f"{model_name}.pt"
            self.log_info(f"  Loading pretrained weights: {model_file}")
            self.model = YOLO(model_file)
        else:
            # Load model architecture only (random initialization)
            model_yaml = f"{model_name}.yaml"
            self.log_info(f"  Loading model architecture: {model_yaml}")
            self.model = YOLO(model_yaml)
        
        self.log_info("✓ Model built successfully")
    
    def _prepare_training_args(self, data_yaml: str) -> Dict[str, Any]:
        """Prepare training arguments from config.
        
        Args:
            data_yaml: Path to data.yaml file
            
        Returns:
            Dictionary of training arguments for Ultralytics
        """
        args = {
            # Data
            'data': data_yaml,
            
            # Training duration
            'epochs': self.training_config.get('epochs', 50),
            'batch': self.training_config.get('batch_size', 16),
            'imgsz': self.training_config.get('imgsz', 640),
            
            # Optimizer
            'optimizer': self.training_config.get('optimizer', {}).get('name', 'AdamW'),
            'lr0': self.training_config.get('optimizer', {}).get('lr', 0.001),
            'lrf': self.training_config.get('scheduler', {}).get('min_lr', 0.00001) / 
                   self.training_config.get('optimizer', {}).get('lr', 0.001),
            'momentum': self.training_config.get('optimizer', {}).get('momentum', 0.9),
            'weight_decay': self.training_config.get('optimizer', {}).get('weight_decay', 0.0001),
            
            # Scheduler
            'warmup_epochs': self.training_config.get('scheduler', {}).get('warmup_epochs', 3),
            
            # Device
            'device': self.training_config.get('device', 'cuda'),
            'workers': self.training_config.get('num_workers', 4),
            
            # Mixed precision
            'amp': self.training_config.get('amp', True),
            
            # Augmentation
            'hsv_h': self.augmentation_config.get('hsv_h', 0.015),
            'hsv_s': self.augmentation_config.get('hsv_s', 0.7),
            'hsv_v': self.augmentation_config.get('hsv_v', 0.4),
            'degrees': self.augmentation_config.get('degrees', 0.0),
            'translate': self.augmentation_config.get('translate', 0.1),
            'scale': self.augmentation_config.get('scale', 0.5),
            'shear': self.augmentation_config.get('shear', 0.0),
            'perspective': self.augmentation_config.get('perspective', 0.0),
            'flipud': self.augmentation_config.get('flipud', 0.0),
            'fliplr': self.augmentation_config.get('fliplr', 0.5),
            'mosaic': self.augmentation_config.get('mosaic', 1.0),
            'mixup': self.augmentation_config.get('mixup', 0.0),
            
            # Validation
            'val': True,
            'save': True,
            'save_period': self.training_config.get('save_period', 10),
            
            # Checkpointing
            'project': str(self.exp_dir.parent),
            'name': self.exp_dir.name,
            'exist_ok': True,
            
            # Logging
            'verbose': True,
            'plots': True,
        }
        
        # Early stopping
        early_stopping_config = self.training_config.get('early_stopping', {})
        if early_stopping_config.get('enabled', True):
            args['patience'] = early_stopping_config.get('patience', 15)
        
        return args
    
    def train(self, data_yaml: str) -> Dict[str, Any]:
        """Train YOLO model.
        
        Args:
            data_yaml: Path to data.yaml file
            
        Returns:
            Dictionary containing training results
        """
        self.log_info(f"\n{'='*80}")
        self.log_info("STARTING TRAINING")
        self.log_info(f"{'='*80}")
        
        # Prepare training arguments
        train_args = self._prepare_training_args(data_yaml)
        
        # Log training configuration
        self.log_info("\nTraining configuration:")
        for key, value in train_args.items():
            if key not in ['data', 'project', 'name']:
                self.log_info(f"  {key}: {value}")
        
        # Train model
        self.log_info(f"\nTraining with data: {data_yaml}")
        results = self.model.train(**train_args)
        
        # Copy best and last checkpoints to standard location
        # Ultralytics saves to project/name/weights/
        ultralytics_weights_dir = self.exp_dir / "weights"
        
        if (ultralytics_weights_dir / "best.pt").exists():
            shutil.copy2(
                ultralytics_weights_dir / "best.pt",
                self.checkpoints_dir / "best.pt"
            )
            self.log_info(f"\n✓ Copied best checkpoint to: {self.checkpoints_dir / 'best.pt'}")
        
        if (ultralytics_weights_dir / "last.pt").exists():
            shutil.copy2(
                ultralytics_weights_dir / "last.pt",
                self.checkpoints_dir / "last.pt"
            )
            self.log_info(f"✓ Copied last checkpoint to: {self.checkpoints_dir / 'last.pt'}")
        
        self.log_info(f"\n{'='*80}")
        self.log_info("✓ TRAINING COMPLETED")
        self.log_info(f"{'='*80}")
        
        return results
    
    def validate(self, data_yaml: Optional[str] = None) -> Dict[str, float]:
        """Run validation on best model.
        
        Args:
            data_yaml: Path to data.yaml file (optional, uses training data if not provided)
            
        Returns:
            Dictionary of validation metrics
        """
        self.log_info(f"\n{'='*80}")
        self.log_info("RUNNING VALIDATION")
        self.log_info(f"{'='*80}")
        
        # Use best checkpoint if available
        best_checkpoint = self.get_best_checkpoint_path()
        if best_checkpoint:
            self.log_info(f"\nUsing checkpoint: {best_checkpoint}")
            model = YOLO(str(best_checkpoint))
        else:
            self.log_info("\nUsing current model (no checkpoint found)")
            model = self.model
        
        # Run validation
        val_args = {
            'verbose': True,
            'plots': True,
        }
        
        if data_yaml:
            val_args['data'] = data_yaml
        
        results = model.val(**val_args)
        
        # Extract metrics
        metrics = {
            'mAP_0.50': float(results.box.map50),
            'mAP_0.50-0.95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
        }
        
        self.log_info("\nValidation metrics:")
        for key, value in metrics.items():
            self.log_info(f"  {key}: {value:.4f}")
        
        self.log_info(f"\n{'='*80}")
        self.log_info("✓ VALIDATION COMPLETED")
        self.log_info(f"{'='*80}")
        
        return metrics
    
    def save_checkpoint(self, path: str, **kwargs) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            **kwargs: Additional metadata (not used for YOLO)
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # YOLO models are automatically saved during training
        # This method is for manual saving if needed
        self.model.save(path)
        self.log_info(f"✓ Saved checkpoint to: {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        self.log_info(f"\nLoading checkpoint: {path}")
        self.model = YOLO(path)
        self.log_info("✓ Checkpoint loaded successfully")
    
    def export(self, format: str = 'onnx', **kwargs) -> str:
        """Export model to different formats.
        
        Args:
            format: Export format (onnx, torchscript, coreml, etc.)
            **kwargs: Additional export arguments
            
        Returns:
            Path to exported model
        """
        self.log_info(f"\nExporting model to {format} format...")
        
        # Use best checkpoint if available
        best_checkpoint = self.get_best_checkpoint_path()
        if best_checkpoint:
            model = YOLO(str(best_checkpoint))
        else:
            model = self.model
        
        export_path = model.export(format=format, **kwargs)
        self.log_info(f"✓ Model exported to: {export_path}")
        
        return export_path
