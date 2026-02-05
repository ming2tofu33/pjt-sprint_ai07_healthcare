"""
Configuration management for YOLO experiments.

Supports:
- YAML-based config files
- CLI argument override
- Config validation
- Experiment-specific settings
"""

import argparse
import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigDict(dict):
    """Dict subclass that allows attribute-style access."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")


def load_yaml(yaml_path: str) -> Dict[str, Any]:
    """Load YAML file and return as dictionary.
    
    Args:
        yaml_path: Path to YAML file
        
    Returns:
        Dictionary containing YAML contents
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config or {}


def merge_configs(base_config: Dict[str, Any], exp_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge experiment config into base config (deep merge).
    
    Args:
        base_config: Base configuration dictionary
        exp_config: Experiment-specific configuration
        
    Returns:
        Merged configuration dictionary
    """
    merged = copy.deepcopy(base_config)
    
    def _deep_merge(base: Dict, update: Dict) -> None:
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                _deep_merge(base[key], value)
            else:
                base[key] = value
    
    _deep_merge(merged, exp_config)
    return merged


def override_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Override config values with command-line arguments.
    
    Args:
        config: Configuration dictionary
        args: Parsed command-line arguments
        
    Returns:
        Updated configuration dictionary
    """
    args_dict = vars(args)
    
    # Override with non-None CLI arguments
    for key, value in args_dict.items():
        if value is not None and key in config:
            # Handle nested keys (e.g., 'model.name')
            if '.' in key:
                keys = key.split('.')
                target = config
                for k in keys[:-1]:
                    target = target.setdefault(k, {})
                target[keys[-1]] = value
            else:
                config[key] = value
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration values.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = ['project', 'data', 'model', 'training']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")
    
    # Validate data paths
    data_config = config.get('data', {})
    if 'raw_dir' in data_config:
        raw_dir = Path(data_config['raw_dir'])
        if not raw_dir.exists():
            raise ValueError(f"Data directory does not exist: {raw_dir}")
    
    # Validate training parameters
    training_config = config.get('training', {})
    if 'epochs' in training_config:
        if training_config['epochs'] <= 0:
            raise ValueError("epochs must be positive")
    if 'batch_size' in training_config:
        if training_config['batch_size'] <= 0:
            raise ValueError("batch_size must be positive")


def load_config(
    base_config_path: str,
    exp_config_path: Optional[str] = None,
    args: Optional[argparse.Namespace] = None
) -> ConfigDict:
    """Load and merge configuration from multiple sources.
    
    Priority (low to high):
    1. Base config (base.yaml)
    2. Experiment config (exp00X.yaml)
    3. CLI arguments
    
    Args:
        base_config_path: Path to base configuration file
        exp_config_path: Path to experiment configuration file (optional)
        args: Command-line arguments (optional)
        
    Returns:
        Merged configuration as ConfigDict
    """
    # Load base config
    base_config = load_yaml(base_config_path)
    
    # Merge with experiment config if provided
    if exp_config_path:
        exp_config = load_yaml(exp_config_path)
        config = merge_configs(base_config, exp_config)
    else:
        config = base_config
    
    # Override with CLI args if provided
    if args:
        config = override_config_with_args(config, args)
    
    # Validate final config
    validate_config(config)
    
    # Convert to ConfigDict for attribute access
    return ConfigDict(config)


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(dict(config), f, default_flow_style=False, sort_keys=False)


def print_config(config: Dict[str, Any], indent: int = 0) -> None:
    """Pretty print configuration.
    
    Args:
        config: Configuration dictionary
        indent: Current indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")
