"""
Base trainer abstract class for extensibility.

Allows easy integration of different frameworks:
- YOLO (Ultralytics)
- Detectron2
- MMDetection
- Custom models
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class BaseTrainer(ABC):
    """Abstract base class for model trainers."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        exp_dir: Path,
        logger: Optional[Any] = None
    ):
        """Initialize base trainer.
        
        Args:
            config: Configuration dictionary
            exp_dir: Experiment directory
            logger: Logger instance (optional)
        """
        self.config = config
        self.exp_dir = Path(exp_dir)
        self.logger = logger
        
        # Create subdirectories
        self.checkpoints_dir = self.exp_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.best_metric = None
    
    @abstractmethod
    def build_model(self) -> None:
        """Build model from config.
        
        This method should:
        1. Load pretrained weights if specified
        2. Configure model architecture
        3. Set up optimizer and scheduler
        """
        pass
    
    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Train model.
        
        Returns:
            Dictionary containing training results and metrics
        """
        pass
    
    @abstractmethod
    def validate(self) -> Dict[str, float]:
        """Run validation.
        
        Returns:
            Dictionary of validation metrics
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str, **kwargs) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            **kwargs: Additional metadata to save
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        pass
    
    def log_info(self, message: str) -> None:
        """Log info message.
        
        Args:
            message: Message to log
        """
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Training step (optional)
        """
        if self.logger:
            self.logger.log_metrics(metrics, step=step)
        else:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            step_str = f"Step {step} - " if step is not None else ""
            print(f"{step_str}{metrics_str}")
    
    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to best checkpoint.
        
        Returns:
            Path to best checkpoint or None if not found
        """
        best_checkpoint = self.checkpoints_dir / "best.pt"
        if best_checkpoint.exists():
            return best_checkpoint
        return None
    
    def get_last_checkpoint_path(self) -> Optional[Path]:
        """Get path to last checkpoint.
        
        Returns:
            Path to last checkpoint or None if not found
        """
        last_checkpoint = self.checkpoints_dir / "last.pt"
        if last_checkpoint.exists():
            return last_checkpoint
        return None
