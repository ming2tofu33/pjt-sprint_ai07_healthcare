"""
Experiment management utilities.

Handles:
- Automatic experiment numbering (exp001, exp002, ...)
- Run directory creation
- Experiment metadata tracking
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple


def get_next_experiment_number(runs_dir: str = "runs") -> int:
    """Get next experiment number by scanning existing run directories.
    
    Args:
        runs_dir: Directory containing experiment runs
        
    Returns:
        Next available experiment number
    """
    runs_path = Path(runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)
    
    # Find all existing experiment directories (exp001, exp002, etc.)
    exp_pattern = re.compile(r"^exp(\d{3}).*")
    existing_numbers = []
    
    for item in runs_path.iterdir():
        if item.is_dir():
            match = exp_pattern.match(item.name)
            if match:
                existing_numbers.append(int(match.group(1)))
    
    # Return next number
    if existing_numbers:
        return max(existing_numbers) + 1
    else:
        return 1


def create_experiment_dir(
    runs_dir: str = "runs",
    exp_name: Optional[str] = None,
    exp_number: Optional[int] = None
) -> Tuple[Path, str]:
    """Create experiment directory with automatic numbering.
    
    Creates directory structure:
    runs/
    └── exp001_baseline_20250205_143022/
        ├── checkpoints/
        ├── logs/
        ├── visualizations/
        └── config_snapshot.yaml
    
    Args:
        runs_dir: Base directory for runs
        exp_name: Optional experiment name (e.g., 'baseline', 'advanced')
        exp_number: Optional explicit experiment number (otherwise auto-increment)
        
    Returns:
        Tuple of (experiment directory path, experiment ID)
    """
    runs_path = Path(runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)
    
    # Get experiment number
    if exp_number is None:
        exp_number = get_next_experiment_number(runs_dir)
    
    # Create experiment ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"exp{exp_number:03d}"
    
    if exp_name:
        exp_dir_name = f"{exp_id}_{exp_name}_{timestamp}"
    else:
        exp_dir_name = f"{exp_id}_{timestamp}"
    
    # Create directory structure
    exp_dir = runs_path / exp_dir_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "visualizations").mkdir(exist_ok=True)
    
    return exp_dir, exp_id


def save_experiment_metadata(
    exp_dir: Path,
    config: Dict,
    exp_id: str,
    exp_name: Optional[str] = None
) -> None:
    """Save experiment metadata for reproducibility.
    
    Args:
        exp_dir: Experiment directory path
        config: Configuration dictionary
        exp_id: Experiment ID (e.g., 'exp001')
        exp_name: Optional experiment name
    """
    metadata = {
        'experiment_id': exp_id,
        'experiment_name': exp_name,
        'created_at': datetime.now().isoformat(),
        'config': config
    }
    
    metadata_path = exp_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def find_experiment_dir(exp_id: str, runs_dir: str = "runs") -> Optional[Path]:
    """Find experiment directory by experiment ID.
    
    Args:
        exp_id: Experiment ID (e.g., 'exp001')
        runs_dir: Base directory for runs
        
    Returns:
        Path to experiment directory or None if not found
    """
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return None
    
    # Search for directory starting with exp_id
    for item in runs_path.iterdir():
        if item.is_dir() and item.name.startswith(exp_id):
            return item
    
    return None


def load_experiment_metadata(exp_dir: Path) -> Optional[Dict]:
    """Load experiment metadata from directory.
    
    Args:
        exp_dir: Experiment directory path
        
    Returns:
        Metadata dictionary or None if not found
    """
    metadata_path = exp_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return metadata


def list_experiments(runs_dir: str = "runs") -> list[Dict]:
    """List all experiments with their metadata.
    
    Args:
        runs_dir: Base directory for runs
        
    Returns:
        List of experiment metadata dictionaries
    """
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return []
    
    experiments = []
    exp_pattern = re.compile(r"^(exp\d{3}).*")
    
    for item in sorted(runs_path.iterdir()):
        if item.is_dir():
            match = exp_pattern.match(item.name)
            if match:
                metadata = load_experiment_metadata(item)
                if metadata:
                    metadata['directory'] = str(item)
                    experiments.append(metadata)
                else:
                    # Create minimal metadata if not found
                    experiments.append({
                        'experiment_id': match.group(1),
                        'directory': str(item),
                        'created_at': datetime.fromtimestamp(item.stat().st_ctime).isoformat()
                    })
    
    return experiments
