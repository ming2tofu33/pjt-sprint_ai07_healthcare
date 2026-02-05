# Phase 3: Model Training Framework - Completed ✅

## Overview
Phase 3 implements a flexible, extensible model training framework with YOLO as the primary implementation.

## Implemented Components

### 1. Base Trainer (`src/models/base_trainer.py`)
Abstract base class for easy framework extension.

**Features:**
- ✅ Abstract interface for any detection framework
- ✅ Extensible to Detectron2, MMDetection, etc.
- ✅ Standard checkpoint management
- ✅ Logging integration

**Abstract Methods:**
```python
class BaseTrainer(ABC):
    @abstractmethod
    def build_model(self) -> None
    
    @abstractmethod
    def train(self) -> Dict[str, Any]
    
    @abstractmethod
    def validate(self) -> Dict[str, float]
    
    @abstractmethod
    def save_checkpoint(self, path: str, **kwargs) -> None
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> None
```

### 2. YOLO Trainer (`src/models/yolo_trainer.py`)
Ultralytics YOLO wrapper with config integration.

**Features:**
- ✅ Config-based training
- ✅ Automatic checkpoint management
- ✅ Built-in augmentation support
- ✅ W&B and TensorBoard integration (via Ultralytics)
- ✅ Model export support (ONNX, TorchScript, etc.)

**Usage:**
```python
from src.models import YOLOTrainer

trainer = YOLOTrainer(
    config=config,
    exp_dir=exp_dir,
    logger=logger
)

# Train
results = trainer.train(data_yaml="data/yolo_data/data.yaml")

# Validate
metrics = trainer.validate()

# Export
onnx_path = trainer.export(format='onnx')
```

### 3. YOLO Dataset Utilities (`src/data/yolo_dataset.py`)
COCO to YOLO format conversion.

**Features:**
- ✅ Bbox format conversion (COCO → YOLO normalized)
- ✅ Dataset structure creation
- ✅ data.yaml generation
- ✅ Image symlinking (fast) or copying (portable)

**Key Functions:**
```python
# Convert bbox format
x_center, y_center, w, h = convert_bbox_coco_to_yolo(
    bbox_coco=[100, 200, 50, 80],
    image_width=640,
    image_height=480
)

# Prepare complete YOLO dataset
data_yaml_path = prepare_yolo_dataset_from_coco(
    coco_train_json="data/coco_data/train_coco.json",
    coco_val_json="data/coco_data/val_coco.json",
    train_images_dir="data/raw/train_images",
    val_images_dir="data/raw/train_images",
    output_dir="data/yolo_data",
    category_mapping_path="data/coco_data/category_mapping.json"
)
```

### 4. Stage 2: YOLO Dataset Preparation (`scripts/2_prepare_yolo.py`)
Prepares YOLO dataset from COCO format.

**Features:**
- ✅ Split-aware conversion
- ✅ Automatic data.yaml creation
- ✅ Image management (symlink/copy)

**Usage:**
```bash
python scripts/2_prepare_yolo.py \
    --coco_dir data/coco_data \
    --images_dir data/raw/train_images \
    --splits_dir data/splits \
    --output_dir data/yolo_data

# Options:
#   --copy_images    Copy images instead of symlinks (portable but slower)
```

**Output:**
```
data/yolo_data/
├── data.yaml              # YOLO config
├── images/
│   ├── train/            # Training images (symlinks or copies)
│   └── val/              # Validation images
└── labels/
    ├── train/            # Training labels (.txt files)
    └── val/              # Validation labels
```

### 5. Stage 3: Model Training (`scripts/3_train.py`)
Main training script with full config integration.

**Features:**
- ✅ Config + CLI args override
- ✅ Automatic experiment management
- ✅ Checkpoint management
- ✅ Comprehensive logging
- ✅ Resume training support

**Usage:**
```bash
# Basic training
python scripts/3_train.py \
    --config configs/experiments/exp001_baseline.yaml

# With overrides
python scripts/3_train.py \
    --config configs/experiments/exp001_baseline.yaml \
    --epochs 100 \
    --batch_size 32 \
    --model yolov8m

# Resume training
python scripts/3_train.py \
    --config configs/experiments/exp001_baseline.yaml \
    --resume runs/exp001_*/checkpoints/last.pt

# Enable logging
python scripts/3_train.py \
    --config configs/experiments/exp001_baseline.yaml \
    --use_wandb \
    --use_tensorboard
```

## Complete Training Pipeline

### Step 1: Prepare COCO Dataset (From Phase 2)
```bash
# Create COCO format
python scripts/1_create_coco_format.py \
    --train_images data/raw/train_images \
    --train_annotations data/raw/train_annotations \
    --output_dir data/coco_data

# Create splits
python scripts/0_splitting.py \
    --coco_json data/coco_data/merged_coco.json \
    --output_dir data/splits \
    --train_ratio 0.8
```

### Step 2: Prepare YOLO Dataset
```bash
python scripts/2_prepare_yolo.py \
    --coco_dir data/coco_data \
    --images_dir data/raw/train_images \
    --splits_dir data/splits \
    --output_dir data/yolo_data
```

### Step 3: Train Model
```bash
python scripts/3_train.py \
    --config configs/experiments/exp001_baseline.yaml
```

### Expected Output:
```
runs/
└── exp001_baseline_20260205_173000/
    ├── checkpoints/
    │   ├── best.pt           # Best model
    │   └── last.pt           # Last checkpoint
    ├── logs/
    │   └── exp001.log        # Training logs
    ├── weights/              # Ultralytics weights dir
    ├── config_snapshot.yaml  # Config used
    └── metadata.json         # Experiment metadata
```

## Configuration System

### Training Config Structure
```yaml
model:
  name: yolov8n                # Model architecture
  yolo:
    pretrained: true          # Use COCO pretrained weights
    pretrained_weights: null  # Or path to custom weights

training:
  epochs: 50
  batch_size: 16
  imgsz: 640
  device: cuda
  
  optimizer:
    name: AdamW
    lr: 0.001
    weight_decay: 0.0001
  
  scheduler:
    name: CosineAnnealingLR
    warmup_epochs: 3
    min_lr: 0.00001
  
  early_stopping:
    enabled: true
    patience: 15

augmentation:
  mosaic: 1.0
  mixup: 0.0
  fliplr: 0.5
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
```

### CLI Override Examples
```bash
# Override epochs and batch size
--epochs 100 --batch_size 32

# Override model
--model yolov8m

# Override learning rate
--lr 0.0005

# Enable logging
--use_wandb --use_tensorboard
```

## Key Features

### 1. **Framework Extensibility**
Easy to add new frameworks:
```python
class CustomTrainer(BaseTrainer):
    def build_model(self):
        # Your model initialization
        pass
    
    def train(self):
        # Your training loop
        pass
    
    # ... implement other abstract methods
```

### 2. **Bbox Format Conversion**
```
COCO: [x, y, width, height]      # Absolute pixels
YOLO: [x_center, y_center, w, h] # Normalized 0-1
```

Example:
```python
# COCO bbox: [100, 200, 50, 80] in 640x480 image
# YOLO bbox: [0.195312, 0.500000, 0.078125, 0.166667]
```

### 3. **Automatic Experiment Management**
- ✅ Auto-incrementing experiment numbers (exp001, exp002, ...)
- ✅ Timestamp-based directory names
- ✅ Config snapshots for reproducibility
- ✅ Comprehensive logging

### 4. **Checkpoint Management**
- ✅ Best checkpoint (by validation metric)
- ✅ Last checkpoint (for resuming)
- ✅ Periodic checkpoints (every N epochs)
- ✅ Automatic checkpoint copying from Ultralytics structure

## Testing

Run Phase 3 tests:
```bash
python tests/test_phase3.py
```

**Test Output:**
```
✓ TEST 1: COCO to YOLO Bbox Conversion
✓ TEST 2: YOLO data.yaml Creation
✓ TEST 3: Base Trainer Interface
✓ TEST 4: YOLO Trainer Initialization (requires ultralytics)
✓ TEST 5: Training Arguments Preparation (requires ultralytics)

✓ ALL PHASE 3 TESTS PASSED!
```

## Dependencies

**Required:**
- PyTorch (with CUDA)
- ultralytics (YOLOv8)
- PyYAML
- Pillow

**Optional:**
- wandb (for W&B logging)
- tensorboard (for TensorBoard logging)

**Install:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics
pip install wandb tensorboard  # Optional
```

## Troubleshooting

### Issue: "Ultralytics not installed"
```bash
pip install ultralytics
```

### Issue: "CUDA out of memory"
```bash
# Reduce batch size
--batch_size 8

# Or reduce image size
--imgsz 320
```

### Issue: "Data YAML not found"
```bash
# Run Stage 2 first
python scripts/2_prepare_yolo.py
```

### Issue: Training is slow
```bash
# Check GPU usage
nvidia-smi

# Reduce num_workers if CPU bottleneck
# Edit config: training.num_workers: 2
```

## Performance Tips

### 1. **Batch Size**
- Start with batch_size=16 for yolov8n/s
- Reduce to 8 for yolov8m/l/x
- Increase with more GPU memory

### 2. **Image Size**
- 640: Fast, good for large objects
- 1280: Better for small objects, slower

### 3. **Augmentation**
- Strong augmentation for small datasets
- Reduce augmentation if overfitting on validation

### 4. **Learning Rate**
- Default 0.001 works well for most cases
- Reduce to 0.0005 for larger models
- Use warmup_epochs=3-5

## Next Steps (Phase 4)
- [ ] Implement evaluation script with mAP@0.75-0.95
- [ ] Add per-class metrics
- [ ] Create visualization tools
- [ ] Add model comparison utilities

---

**Status:** ✅ Completed and Tested  
**Date:** 2026-02-05  
**Ready for:** Local training with real data
