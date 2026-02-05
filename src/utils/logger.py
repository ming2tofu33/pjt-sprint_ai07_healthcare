"""
Logging utilities for experiment tracking.

Supports:
- Structured logging to console and file
- Experiment metadata tracking
- Weights & Biases (W&B) integration (optional)
- TensorBoard integration (optional)
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ExperimentLogger:
    """Unified logger for experiments with optional W&B/TensorBoard support."""
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        config: Dict[str, Any],
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
    ):
        """Initialize experiment logger.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name of the experiment
            config: Configuration dictionary
            use_wandb: Whether to use Weights & Biases
            use_tensorboard: Whether to use TensorBoard
            wandb_project: W&B project name
            wandb_entity: W&B entity (username/team)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.config = config
        
        # Setup standard Python logger
        self.logger = self._setup_logger()
        
        # Setup W&B if requested
        self.use_wandb = use_wandb
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project or config.get('project', {}).get('name', 'yolo-experiment'),
                    entity=wandb_entity,
                    name=experiment_name,
                    config=config,
                    dir=str(self.log_dir)
                )
                self.logger.info(f"W&B initialized: {self.wandb_run.url}")
            except ImportError:
                self.logger.warning("W&B requested but not installed. Install with: pip install wandb")
                self.use_wandb = False
            except Exception as e:
                self.logger.warning(f"W&B initialization failed: {e}")
                self.use_wandb = False
        
        # Setup TensorBoard if requested
        self.use_tensorboard = use_tensorboard
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.log_dir / "tensorboard"
                self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
                self.logger.info(f"TensorBoard log directory: {tb_dir}")
            except ImportError:
                self.logger.warning("TensorBoard requested but not installed. Install with: pip install tensorboard")
                self.use_tensorboard = False
    
    def _setup_logger(self) -> logging.Logger:
        """Setup Python logger with file and console handlers."""
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = self.log_dir / f"{self.experiment_name}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
        return logger
    
    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to all enabled backends.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Training step/epoch number
        """
        # Log to console
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        step_str = f"Step {step} - " if step is not None else ""
        self.logger.info(f"{step_str}{metrics_str}")
        
        # Log to W&B
        if self.use_wandb and self.wandb_run:
            self.wandb_run.log(metrics, step=step)
        
        # Log to TensorBoard
        if self.use_tensorboard and self.tb_writer and step is not None:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)
    
    def log_image(self, tag: str, image: Any, step: Optional[int] = None) -> None:
        """Log image to enabled backends.
        
        Args:
            tag: Image tag/name
            image: Image tensor or array
            step: Training step/epoch number
        """
        if self.use_wandb and self.wandb_run:
            import wandb
            self.wandb_run.log({tag: wandb.Image(image)}, step=step)
        
        if self.use_tensorboard and self.tb_writer and step is not None:
            self.tb_writer.add_image(tag, image, step)
    
    def log_model_summary(self, model: Any) -> None:
        """Log model architecture summary.
        
        Args:
            model: PyTorch model
        """
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, skipping model summary")
            return
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model Summary:")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        
        if self.use_wandb and self.wandb_run:
            self.wandb_run.config.update({
                'model_total_params': total_params,
                'model_trainable_params': trainable_params
            })
    
    def save_checkpoint_info(self, checkpoint_path: str, metrics: Dict[str, float]) -> None:
        """Log checkpoint save event.
        
        Args:
            checkpoint_path: Path to saved checkpoint
            metrics: Metrics at checkpoint time
        """
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        self.logger.info(f"  Metrics: {metrics}")
        
        if self.use_wandb and self.wandb_run:
            import wandb
            self.wandb_run.log_artifact(checkpoint_path, type='model')
    
    def close(self) -> None:
        """Close all logging backends."""
        if self.use_wandb and self.wandb_run:
            self.wandb_run.finish()
        
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.close()
        
        # Close file handlers
        for handler in self.logger.handlers:
            handler.close()
