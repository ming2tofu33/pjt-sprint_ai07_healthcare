# Phase 1: Core Infrastructure - Completed ✅

## Overview
Phase 1 implements the foundational utilities for experiment management, configuration, logging, and reproducibility.

## Implemented Components

### 1. Configuration System (`src/utils/config.py`)
- **YAML-based configuration** with deep merging support
- **CLI argument override** for flexible experimentation
- **Config validation** to catch errors early
- **Attribute-style access** via `ConfigDict` class

**Usage:**
```python
from src.utils import load_config

config = load_config(
    base_config_path="configs/base.yaml",
    exp_config_path="configs/experiments/exp001_baseline.yaml",
    args=args  # Optional CLI arguments
)

# Access config values
print(config.model.name)  # yolov8n
print(config.training.epochs)  # 50
```

### 2. Experiment Management (`src/utils/experiment.py`)
- **Automatic experiment numbering** (exp001, exp002, ...)
- **Timestamp-based directory naming** for easy tracking
- **Structured directory creation** (checkpoints/, logs/, visualizations/)
- **Metadata tracking** for reproducibility

**Usage:**
```python
from src.utils import create_experiment_dir, save_experiment_metadata

# Create experiment directory
exp_dir, exp_id = create_experiment_dir(
    runs_dir="runs",
    exp_name="baseline"
)
# Creates: runs/exp001_baseline_20260205_171409/

# Save metadata
save_experiment_metadata(exp_dir, config, exp_id, exp_name="baseline")
```

### 3. Logging System (`src/utils/logger.py`)
- **Unified logging interface** for console and file
- **Optional W&B integration** (enable in config)
- **Optional TensorBoard integration** (enable in config)
- **Structured metric logging** with automatic formatting

**Usage:**
```python
from src.utils import ExperimentLogger

logger = ExperimentLogger(
    log_dir="runs/exp001_baseline/logs",
    experiment_name="exp001",
    config=config,
    use_wandb=config.logging.wandb.enabled,
    use_tensorboard=config.logging.tensorboard.enabled
)

# Log messages
logger.info("Training started")

# Log metrics
logger.log_metrics({
    'train/loss': 0.5,
    'val/mAP': 0.75
}, step=100)

# Close logger
logger.close()
```

### 4. Reproducibility (`src/utils/seed.py`)
- **Deterministic seed setting** across Python, NumPy, PyTorch
- **DataLoader worker initialization** for multi-process reproducibility
- **Optional deterministic mode** (slower but fully reproducible)

**Usage:**
```python
from src.utils import set_seed, worker_init_fn

# Set seed for reproducibility
set_seed(seed=42, deterministic=True)

# Use in DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(
    dataset,
    batch_size=16,
    worker_init_fn=lambda worker_id: worker_init_fn(worker_id, seed=42)
)
```

## Configuration Files

### Base Configuration (`configs/base.yaml`)
Default settings shared across all experiments:
- Project metadata
- Data paths and split settings
- Model configuration (framework, name, pretrained weights)
- Training hyperparameters (epochs, batch size, optimizer, scheduler)
- Data augmentation settings
- Evaluation metrics and thresholds
- Submission format settings
- Logging configuration (W&B, TensorBoard)

### Experiment Configurations
Override base config for specific experiments:
- **exp001_baseline.yaml**: YOLOv8n baseline with 50 epochs
- **exp002_yolov8s_extended.yaml**: YOLOv8s with 100 epochs and strong augmentation
- **exp003_yolov8m_highres.yaml**: YOLOv8m with 1280px images for small objects

## Directory Structure
```
runs/
└── exp001_baseline_20260205_171409/
    ├── checkpoints/        # Model checkpoints
    ├── logs/              # Log files
    │   └── exp001.log
    ├── visualizations/    # Visualization outputs
    ├── config_snapshot.yaml  # Config used for this run
    └── metadata.json      # Experiment metadata
```

## Testing
Run Phase 1 tests to verify all utilities:
```bash
python tests/test_phase1.py
```

**Test Output:**
```
✓ TEST 1: Config Loading and Merging
✓ TEST 2: Experiment Directory Creation  
✓ TEST 3: Seed Setting
✓ TEST 4: Logger Initialization
✓ TEST 5: Config Snapshot Save

✓ ALL TESTS PASSED!
```

## Key Features

### 1. **Flexible Configuration Priority**
```
Base Config → Experiment Config → CLI Args
(lowest)                            (highest)
```

### 2. **Automatic Experiment Numbering**
- Scans `runs/` directory for existing experiments
- Auto-increments experiment number (exp001 → exp002 → ...)
- Adds timestamp for uniqueness

### 3. **Multi-Backend Logging**
- **Console**: Real-time output with colored formatting
- **File**: Persistent logs with timestamps
- **W&B**: Web dashboard for team collaboration (optional)
- **TensorBoard**: Local visualization (optional)

### 4. **Deterministic Reproducibility**
- Seeds: Python, NumPy, PyTorch (CPU & CUDA)
- Deterministic algorithms (optional, slower)
- Worker seeds for DataLoader parallelism

## Configuration Options

### Enable W&B Logging
```yaml
# In configs/experiments/exp00X.yaml
logging:
  wandb:
    enabled: true
    project: "pill-detection"
    entity: "your-username"
    tags: ["baseline", "exp001"]
```

### Enable TensorBoard
```yaml
logging:
  tensorboard:
    enabled: true
```

### Class Filtering (for experiments)
```yaml
data:
  class_filter:
    enabled: true
    include_classes: [1, 2, 3, 5, 7]  # Only use these classes
    # OR
    exclude_classes: [10, 15, 20]  # Exclude these classes
```

### K-Fold Cross-Validation
```yaml
data:
  split:
    method: "kfold"
    n_folds: 5
    current_fold: 0  # 0-4
```

## Next Steps (Phase 2)
- [ ] Data splitting script (`scripts/0_splitting.py`)
- [ ] COCO format conversion (`scripts/1_create_coco_format.py`)
- [ ] Category ID mapping for submission
- [ ] Dataset validation and statistics

---

**Status:** ✅ Completed and Tested  
**Date:** 2026-02-05
