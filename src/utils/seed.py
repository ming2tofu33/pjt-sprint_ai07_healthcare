"""
Reproducibility utilities for setting random seeds.

Ensures deterministic behavior across:
- Python's random module
- NumPy
- PyTorch (CPU and CUDA)
"""

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (slower but reproducible)
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch (if available)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        
        # PyTorch deterministic behavior
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # For PyTorch >= 1.8
            if hasattr(torch, 'use_deterministic_algorithms'):
                torch.use_deterministic_algorithms(True)
        else:
            torch.backends.cudnn.benchmark = True
    
    # Environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_seed_from_config(config: dict) -> int:
    """Extract seed from config with fallback to default.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Seed value
    """
    return config.get('seed', 42)


def worker_init_fn(worker_id: int, seed: Optional[int] = None) -> None:
    """Initialize DataLoader worker with proper seed.
    
    Use this function as worker_init_fn in DataLoader to ensure
    each worker has a different but reproducible seed.
    
    Args:
        worker_id: Worker ID assigned by DataLoader
        seed: Base seed value (if None, uses current random state)
    """
    if not TORCH_AVAILABLE:
        # Fallback to random seed generation
        if seed is None:
            seed = 42
        worker_seed = seed + worker_id
    else:
        if seed is None:
            seed = torch.initial_seed() % 2**32
        worker_seed = seed + worker_id
    
    np.random.seed(worker_seed)
    random.seed(worker_seed)
