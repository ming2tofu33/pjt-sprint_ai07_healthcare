# 🎯 Complete YOLOv8 Baseline Implementation - All Phases Complete ✅

**Project:** Kaggle AI07 Pill Detection Challenge  
**Branch:** `feat/DM-refactor`  
**Status:** ✅ All 5 Phases Implemented and Ready for Use  
**Date:** 2026-02-05

---

## 📋 Implementation Overview

This document provides a comprehensive summary of the complete YOLOv8 baseline implementation, covering all 5 phases from infrastructure setup to Kaggle submission generation.

### ✅ Phase Completion Status

| Phase | Component | Status | Key Deliverables |
|-------|-----------|--------|------------------|
| **Phase 1** | Core Infrastructure | ✅ Complete | Config system, Logging, Seed management, Experiment tracking |
| **Phase 2** | Data Pipeline | ✅ Complete | COCO format conversion, Data splitting, Validation |
| **Phase 3** | Model Training | ✅ Complete | YOLO trainer, Training scripts, Dataset preparation |
| **Phase 4** | Evaluation | ✅ Complete | Metrics calculation, Visualization tools |
| **Phase 5** | Submission | ✅ Complete | Kaggle CSV generation, Format validation |

---

## 🚀 Quick Start Guide

### Prerequisites
```bash
# Ensure you have PyTorch and CUDA installed
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Install dependencies
pip install -r requirements.txt
```

### Complete Pipeline (All Stages)

```bash
# Navigate to project directory
cd /home/user/webapp

# ========================================
# STAGE 1: Create COCO Format
# ========================================
python scripts/1_create_coco_format.py \
    --train_images data/raw/train_images \
    --train_annotations data/raw/train_annotations \
    --output_dir data/coco_data \
    --validate \
    --verbose

# Output: data/coco_data/merged_coco.json
#         data/coco_data/category_mapping.json (Critical for submission!)

# ========================================
# STAGE 0: Split Data
# ========================================
python scripts/0_splitting.py \
    --coco_json data/coco_data/merged_coco.json \
    --output_dir data/splits \
    --train_ratio 0.8 \
    --stratify_by object_count \
    --seed 42 \
    --verbose

# Output: data/splits/split_info.json
#         data/splits/train_split/
#         data/splits/val_split/

# ========================================
# STAGE 2: Prepare YOLO Dataset
# ========================================
python scripts/2_prepare_yolo.py \
    --coco_dir data/coco_data \
    --images_dir data/raw/train_images \
    --splits_dir data/splits \
    --output_dir data/yolo_data \
    --symlink \
    --verbose

# Output: data/yolo_data/data.yaml
#         data/yolo_data/images/train/
#         data/yolo_data/images/val/
#         data/yolo_data/labels/train/
#         data/yolo_data/labels/val/

# ========================================
# STAGE 3: Train Model
# ========================================
python scripts/3_train.py \
    --config configs/experiments/exp001_baseline.yaml \
    --epochs 100 \
    --batch_size 16 \
    --verbose

# Output: runs/exp001_baseline_YYYYMMDD_HHMMSS/
#         ├── checkpoints/best.pt
#         ├── checkpoints/last.pt
#         ├── logs/exp001.log
#         └── config_snapshot.yaml

# ========================================
# STAGE 4: Evaluate Model
# ========================================
python scripts/4_evaluate.py \
    --checkpoint runs/exp001_*/checkpoints/best.pt \
    --data_yaml data/yolo_data/data.yaml \
    --output_dir evaluation_results \
    --verbose

# Output: evaluation_results/evaluation_results.json
#         evaluation_results/summary.txt

# ========================================
# STAGE 5: Generate Kaggle Submission
# ========================================
python scripts/5_submission.py \
    --checkpoint runs/exp001_*/checkpoints/best.pt \
    --test_images data/raw/test_images \
    --category_mapping data/coco_data/category_mapping.json \
    --output_dir submissions \
    --conf_threshold 0.25 \
    --iou_nms 0.45 \
    --max_det 4 \
    --verbose

# Output: submissions/submission_exp001_baseline_YYYYMMDD_HHMMSS.csv
```

---

## 📁 Project Structure

```
/home/user/webapp/
├── configs/
│   ├── base.yaml                           # Base configuration
│   └── experiments/
│       ├── exp001_baseline.yaml            # YOLOv8n baseline
│       ├── exp002_yolov8s_extended.yaml    # YOLOv8s with augmentation
│       └── exp003_yolov8m_highres.yaml     # YOLOv8m high-resolution
│
├── src/
│   ├── utils/
│   │   ├── config.py                       # Config loading & merging
│   │   ├── seed.py                         # Reproducibility utilities
│   │   ├── logger.py                       # Unified logging (W&B, TensorBoard)
│   │   └── experiment.py                   # Experiment management
│   │
│   ├── data/
│   │   ├── coco_utils.py                   # COCO format utilities
│   │   ├── split_utils.py                  # Data splitting strategies
│   │   └── yolo_dataset.py                 # COCO → YOLO conversion
│   │
│   ├── models/
│   │   ├── base_trainer.py                 # Abstract trainer base class
│   │   └── yolo_trainer.py                 # YOLO trainer implementation
│   │
│   └── evaluation/
│       ├── metrics.py                      # mAP, IoU calculations
│       └── visualizer.py                   # Visualization tools
│
├── scripts/
│   ├── 0_splitting.py                      # Data splitting
│   ├── 1_create_coco_format.py             # COCO format creation
│   ├── 2_prepare_yolo.py                   # YOLO dataset preparation
│   ├── 3_train.py                          # Model training
│   ├── 4_evaluate.py                       # Model evaluation
│   └── 5_submission.py                     # Kaggle submission generation
│
├── data/
│   ├── raw/
│   │   ├── train_images/                   # 232 PNG images
│   │   ├── train_annotations/              # 763 JSON files (114 folders)
│   │   └── test_images/                    # 842 PNG images
│   │
│   ├── coco_data/
│   │   ├── merged_coco.json                # Merged COCO format
│   │   └── category_mapping.json           # YOLO ↔ COCO ID mapping
│   │
│   ├── splits/
│   │   └── split_info.json                 # Train/Val split metadata
│   │
│   └── yolo_data/
│       ├── data.yaml                       # YOLO dataset config
│       ├── images/                         # Train/Val images
│       └── labels/                         # Train/Val labels
│
├── runs/
│   └── exp001_baseline_*/                  # Experiment outputs
│       ├── checkpoints/
│       │   ├── best.pt                     # Best model weights
│       │   └── last.pt                     # Last epoch weights
│       ├── logs/
│       │   └── exp001.log                  # Training logs
│       └── config_snapshot.yaml            # Config used for training
│
├── submissions/
│   └── submission_*.csv                    # Kaggle submission files
│
├── evaluation_results/
│   ├── evaluation_results.json             # Detailed metrics
│   └── summary.txt                         # Human-readable summary
│
├── tests/
│   ├── test_phase1.py                      # Phase 1 tests
│   ├── test_phase2.py                      # Phase 2 tests
│   └── test_phase3.py                      # Phase 3 tests
│
└── docs/
    ├── PHASE1_COMPLETE.md                  # Phase 1 documentation
    ├── PHASE2_COMPLETE.md                  # Phase 2 documentation
    ├── PHASE3_COMPLETE.md                  # Phase 3 documentation
    ├── PHASE4_5_COMPLETE.md                # Phase 4 & 5 documentation
    └── IMPLEMENTATION_COMPLETE.md          # This file
```

---

## 🔧 Configuration System

### Config Priority (Highest to Lowest)
```
CLI Arguments > Experiment Config > Base Config > Defaults
```

### Base Configuration (`configs/base.yaml`)
Core settings shared across all experiments:
- Project metadata
- Data paths
- Model defaults
- Training hyperparameters
- Augmentation settings
- Evaluation metrics
- Logging configuration

### Experiment Configurations
Override base settings for specific experiments:

**exp001_baseline.yaml** - YOLOv8n baseline
```yaml
experiment:
  name: "baseline"
  model_variant: "yolov8n"
  epochs: 50
  description: "Minimal baseline with YOLOv8n"
```

**exp002_yolov8s_extended.yaml** - Enhanced with augmentation
```yaml
experiment:
  name: "yolov8s_extended"
  model_variant: "yolov8s"
  epochs: 100
  augmentation_level: "medium"
```

**exp003_yolov8m_highres.yaml** - High-resolution training
```yaml
experiment:
  name: "yolov8m_highres"
  model_variant: "yolov8m"
  image_size: 1280
  batch_size: 8
```

### CLI Override Examples
```bash
# Override epochs
python scripts/3_train.py --config configs/experiments/exp001_baseline.yaml --epochs 200

# Override batch size and learning rate
python scripts/3_train.py --config configs/experiments/exp002_yolov8s_extended.yaml \
    --batch_size 32 --lr 0.0005

# Override image size
python scripts/3_train.py --config configs/experiments/exp003_yolov8m_highres.yaml \
    --image_size 1280
```

---

## 📊 Data Pipeline Details

### Stage 1: COCO Format Creation
**Purpose:** Merge 763 JSON files into a single COCO format file

**Key Features:**
- ✅ Validates all JSON files
- ✅ Extracts image metadata from filenames
- ✅ Assigns sequential image IDs
- ✅ Maps original category IDs to 0-based indices
- ✅ Validates bounding boxes
- ✅ Generates category mapping file (**Critical for submission!**)

**Output Files:**
- `merged_coco.json`: Single COCO format file
- `category_mapping.json`: YOLO index ↔ COCO category_id mapping

**Category Mapping Example:**
```json
{
  "yolo_to_coco": {
    "0": 1,
    "1": 11,
    "2": 24,
    "3": 69
  },
  "coco_to_yolo": {
    "1": 0,
    "11": 1,
    "24": 2,
    "69": 3
  }
}
```

### Stage 0: Data Splitting
**Purpose:** Split data into train/val sets

**Strategies:**
1. **Object Count Stratification** (Default)
   - Groups images by number of objects (1, 2, 3, 4)
   - Ensures balanced distribution in train/val

2. **Class Distribution Stratification**
   - Considers class frequencies
   - Maintains class balance across splits

3. **K-Fold Cross-Validation**
   - Generates K folds for robust evaluation
   - Each fold has separate train/val split

**Configuration:**
```bash
# Basic split (80/20)
python scripts/0_splitting.py \
    --coco_json data/coco_data/merged_coco.json \
    --train_ratio 0.8 \
    --stratify_by object_count

# 5-Fold CV
python scripts/0_splitting.py \
    --coco_json data/coco_data/merged_coco.json \
    --kfold 5
```

### Stage 2: YOLO Dataset Preparation
**Purpose:** Convert COCO format to YOLO format

**Bbox Format Conversion:**
```
COCO: [x_min, y_min, width, height]  (absolute pixels)
  ↓
YOLO: [center_x, center_y, width, height]  (normalized 0-1)
```

**Output Structure:**
```
data/yolo_data/
├── data.yaml
├── images/
│   ├── train/  (symlinks to original images)
│   └── val/
└── labels/
    ├── train/  (.txt files with YOLO format)
    └── val/
```

**data.yaml Example:**
```yaml
path: /home/user/webapp/data/yolo_data
train: images/train
val: images/val

nc: 40  # Number of classes
names:
  0: class_1
  1: class_11
  2: class_24
  ...
```

---

## 🎓 Model Training

### Supported Models
- **YOLOv8n**: Fastest, smallest (3.2M params)
- **YOLOv8s**: Balanced (11.2M params)
- **YOLOv8m**: Higher accuracy (25.9M params)
- **YOLOv8l**: Large model (43.7M params)
- **YOLOv8x**: Largest, best accuracy (68.2M params)

### Training Features
- ✅ Config-driven training
- ✅ Automatic experiment management
- ✅ W&B and TensorBoard logging
- ✅ Checkpoint management (best.pt, last.pt)
- ✅ Training resumption
- ✅ Mixed precision training
- ✅ Model export (ONNX, TorchScript)

### Training Script Usage
```bash
# Basic training
python scripts/3_train.py --config configs/experiments/exp001_baseline.yaml

# Resume training
python scripts/3_train.py \
    --config configs/experiments/exp001_baseline.yaml \
    --resume runs/exp001_*/checkpoints/last.pt

# Training with custom settings
python scripts/3_train.py \
    --config configs/experiments/exp002_yolov8s_extended.yaml \
    --epochs 150 \
    --batch_size 32 \
    --lr 0.0005 \
    --warmup_epochs 5
```

### Hyperparameter Recommendations

| Model | Batch Size | Image Size | Learning Rate | Epochs |
|-------|-----------|------------|---------------|--------|
| YOLOv8n | 16-32 | 640 | 0.001 | 50-100 |
| YOLOv8s | 16-32 | 640-1280 | 0.001 | 100-150 |
| YOLOv8m | 8-16 | 1280 | 0.0005 | 150-200 |
| YOLOv8l | 4-8 | 1280 | 0.0005 | 150-200 |
| YOLOv8x | 4-8 | 1280 | 0.0003 | 200+ |

### Data Augmentation

**Built-in YOLO Augmentations:**
- Mosaic (combines 4 images)
- Copy-Paste
- Random HSV adjustments
- Random horizontal flip
- Mixup

**Custom Albumentations (Optional):**
```yaml
augmentation:
  enabled: true
  level: "medium"  # low, medium, high
  
  transforms:
    - RandomBrightnessContrast: {p: 0.3}
    - HueSaturationValue: {p: 0.3}
    - RandomRotate90: {p: 0.2}
    - HorizontalFlip: {p: 0.5}
    - ShiftScaleRotate: {p: 0.3}
```

---

## 📈 Evaluation & Metrics

### Primary Metric: mAP@[0.75:0.95]
**Calculation:**
```
mAP@[0.75:0.95] = (mAP@0.75 + mAP@0.80 + mAP@0.85 + mAP@0.90 + mAP@0.95) / 5
```

**Interpretation:**
- **> 0.30**: Baseline achieved
- **> 0.50**: Competitive score
- **> 0.70**: Top-tier performance

### Secondary Metrics
- **mAP@0.50**: More lenient IoU threshold
- **mAP@0.75**: Moderate threshold
- **Precision**: Percentage of correct predictions
- **Recall**: Percentage of ground truths detected
- **F1-Score**: Harmonic mean of precision and recall

### Per-Class Metrics
- AP per class
- Precision/Recall per class
- Confusion matrix

### Evaluation Script
```bash
# Basic evaluation
python scripts/4_evaluate.py \
    --checkpoint runs/exp001_*/checkpoints/best.pt \
    --data_yaml data/yolo_data/data.yaml

# With custom thresholds
python scripts/4_evaluate.py \
    --checkpoint runs/exp001_*/checkpoints/best.pt \
    --data_yaml data/yolo_data/data.yaml \
    --conf_threshold 0.001 \
    --iou_nms 0.6
```

**Output:**
```json
{
  "mAP_0.50": 0.65,
  "mAP_0.50-0.95": 0.45,
  "mAP_0.75-0.95": 0.38,
  "precision": 0.72,
  "recall": 0.68,
  "per_class_ap": {
    "class_1": 0.55,
    "class_11": 0.48,
    ...
  }
}
```

---

## 📤 Kaggle Submission

### Critical: Format Conversion

**⚠️ Most Important Step:**
```python
# YOLO trains with 0-based indices
yolo_prediction = 3  # Model output

# Must convert to original COCO category_id
coco_category_id = yolo_to_coco[yolo_prediction]  # 3 → 69

# Wrong mapping = 0 score on leaderboard!
```

### CSV Format Requirements
```csv
annotation_id,image_id,category_id,bbox_x,bbox_y,bbox_w,bbox_h,score
1,1,15,100.5,200.3,50.2,80.1,0.95
2,1,23,300.1,150.2,60.3,70.4,0.88
3,2,7,50.0,100.0,40.0,50.0,0.92
```

**Column Specifications:**
- `annotation_id`: Unique ID (1, 2, 3, ...)
- `image_id`: Extracted from filename (`test_001.png` → `1`)
- `category_id`: Original COCO ID (not YOLO index!)
- `bbox_x, bbox_y, bbox_w, bbox_h`: Absolute pixel coordinates
- `score`: Confidence score [0, 1]

### Submission Script Usage
```bash
# Basic submission
python scripts/5_submission.py \
    --checkpoint runs/exp001_*/checkpoints/best.pt \
    --test_images data/raw/test_images \
    --category_mapping data/coco_data/category_mapping.json

# With custom thresholds
python scripts/5_submission.py \
    --checkpoint runs/exp001_*/checkpoints/best.pt \
    --test_images data/raw/test_images \
    --category_mapping data/coco_data/category_mapping.json \
    --conf_threshold 0.25 \
    --iou_nms 0.45 \
    --max_det 4

# With Test Time Augmentation (better score, slower)
python scripts/5_submission.py \
    --checkpoint runs/exp001_*/checkpoints/best.pt \
    --test_images data/raw/test_images \
    --category_mapping data/coco_data/category_mapping.json \
    --tta
```

### Threshold Tuning Guide

**Confidence Threshold:**
```bash
# More predictions (higher recall)
--conf_threshold 0.15

# Balanced (default)
--conf_threshold 0.25

# Fewer but confident (higher precision)
--conf_threshold 0.35
```

**NMS IoU Threshold:**
```bash
# Aggressive suppression
--iou_nms 0.40

# Moderate (default)
--iou_nms 0.45

# Keep more overlapping boxes
--iou_nms 0.50
```

### Submission Validation

The script automatically checks:
- ✅ All required columns present
- ✅ No duplicate annotation_ids
- ✅ Positive bbox dimensions (w > 0, h > 0)
- ✅ Non-negative coordinates
- ✅ Max 4 detections per image
- ✅ Valid score range [0, 1]
- ✅ Valid category_ids

---

## 🔍 Troubleshooting

### Issue: Low mAP on Kaggle

**Possible Causes & Fixes:**

1. **❌ Wrong category_id mapping**
   ```bash
   # Verify category_mapping.json is used
   python -c "
   import json
   with open('data/coco_data/category_mapping.json') as f:
       mapping = json.load(f)
   print('YOLO → COCO:', mapping['yolo_to_coco'])
   "
   ```

2. **❌ Wrong bbox format**
   ```python
   # Bboxes should be absolute pixels, not normalized
   # Good: [100.5, 200.3, 50.2, 80.1]
   # Bad:  [0.15, 0.25, 0.08, 0.10]
   ```

3. **❌ Wrong image_id extraction**
   ```python
   # Verify image_id mapping
   import pandas as pd
   df = pd.read_csv('submissions/submission.csv')
   print(df[['image_id']].head(20))
   # Should match: test_001.png → 1, test_042.png → 42
   ```

### Issue: Training not converging

**Solutions:**
1. Reduce learning rate
2. Increase warmup epochs
3. Check data augmentation strength
4. Verify label quality
5. Try different model size

### Issue: Out of memory during training

**Solutions:**
```bash
# Reduce batch size
--batch_size 8

# Reduce image size
--image_size 640

# Use smaller model
--model_variant yolov8n
```

---

## 🎯 Performance Optimization Tips

### 1. Model Selection
- Start with YOLOv8n for quick experiments
- Use YOLOv8s or YOLOv8m for better accuracy
- Reserve YOLOv8l/x for final submission

### 2. Image Size
- 640: Fast training, good for prototyping
- 1280: Better for small objects (2-4x slower)

### 3. Augmentation Strategy
- Start with light augmentation
- Gradually increase strength based on validation metrics
- Monitor training/validation gap

### 4. Training Duration
- YOLOv8n: 50-100 epochs sufficient
- YOLOv8s/m: 100-150 epochs
- YOLOv8l/x: 150-200+ epochs

### 5. Test Time Augmentation
```bash
# TTA improves score by ~2-5% but 4x slower
--tta
```

### 6. Ensemble (Manual)
Train multiple models and combine predictions:
```bash
# Train different variants
python scripts/3_train.py --config configs/experiments/exp001_baseline.yaml
python scripts/3_train.py --config configs/experiments/exp002_yolov8s_extended.yaml

# Generate predictions
python scripts/5_submission.py --checkpoint runs/exp001_*/checkpoints/best.pt ...
python scripts/5_submission.py --checkpoint runs/exp002_*/checkpoints/best.pt ...

# Combine manually (weighted averaging, WBF, etc.)
```

---

## 📝 Experiment Tracking

### Experiment Naming Convention
```
exp001_baseline_20260205_171409
│    │     │         │       └── Time (HHMMSS)
│    │     │         └── Date (YYYYMMDD)
│    │     └── Experiment name
│    └── Experiment number (auto-increment)
└── Prefix
```

### Experiment Metadata
Each experiment automatically saves:
- `config_snapshot.yaml`: Exact config used
- `metadata.json`: Start time, model info, dataset stats
- `logs/exp001.log`: Detailed training logs
- `checkpoints/best.pt`: Best model weights
- `checkpoints/last.pt`: Latest checkpoint (for resumption)

### W&B Integration
```yaml
# Enable in config
logging:
  wandb:
    enabled: true
    project: "pill-detection"
    entity: "your-team"
```

### TensorBoard Integration
```yaml
# Enable in config
logging:
  tensorboard:
    enabled: true
```

**View logs:**
```bash
tensorboard --logdir runs/
```

---

## 🧪 Testing

### Run All Tests
```bash
# Phase 1: Infrastructure
python tests/test_phase1.py

# Phase 2: Data pipeline
python tests/test_phase2.py

# Phase 3: Model training
python tests/test_phase3.py
```

### Manual Testing
```bash
# Test data pipeline
python scripts/1_create_coco_format.py --validate
python scripts/0_splitting.py --verbose

# Test training (1 epoch)
python scripts/3_train.py --config configs/experiments/exp001_baseline.yaml --epochs 1

# Test evaluation
python scripts/4_evaluate.py --checkpoint runs/exp001_*/checkpoints/last.pt

# Test submission generation
python scripts/5_submission.py --checkpoint runs/exp001_*/checkpoints/last.pt --verbose
```

---

## 📦 Deployment Checklist

### Before Final Submission
- [ ] All data processed correctly (232 train, 842 test images)
- [ ] COCO format validated
- [ ] Train/val split created
- [ ] Model trained to convergence
- [ ] Validation mAP@0.75-0.95 > 0.30
- [ ] Evaluation report generated
- [ ] Submission CSV created
- [ ] CSV validation passed
- [ ] category_mapping.json verified
- [ ] Bbox format checked (absolute coordinates)
- [ ] image_id extraction verified
- [ ] Unique images in submission = 842
- [ ] Average detections per image ≈ 2-3
- [ ] Ready to upload to Kaggle! 🚀

---

## 📚 Documentation

- **Phase 1:** [PHASE1_COMPLETE.md](./PHASE1_COMPLETE.md) - Infrastructure setup
- **Phase 2:** [PHASE2_COMPLETE.md](./PHASE2_COMPLETE.md) - Data pipeline
- **Phase 3:** [PHASE3_COMPLETE.md](./PHASE3_COMPLETE.md) - Model training
- **Phase 4 & 5:** [PHASE4_5_COMPLETE.md](./PHASE4_5_COMPLETE.md) - Evaluation & submission

---

## 🤝 Team Collaboration

### For Team Members

**1. Clone and Setup:**
```bash
git clone <repo-url>
cd pjt-sprint_ai07_healthcare
git checkout feat/DM-refactor
pip install -r requirements.txt
```

**2. Run Your Experiment:**
```bash
# Create your own experiment config
cp configs/experiments/exp001_baseline.yaml configs/experiments/exp_yourname.yaml

# Edit your config
vim configs/experiments/exp_yourname.yaml

# Run training
python scripts/3_train.py --config configs/experiments/exp_yourname.yaml
```

**3. Share Results:**
- Experiment runs are saved in `runs/exp00X_*/`
- Share `config_snapshot.yaml` and `evaluation_results.json`
- Compare metrics in W&B dashboard

### Git Workflow
```bash
# Pull latest changes
git pull origin feat/DM-refactor

# Create your experiment config (don't modify existing ones)
# Add your config to git
git add configs/experiments/exp_yourname.yaml
git commit -m "feat: Add experiment config for [your-experiment]"
git push origin feat/DM-refactor
```

---

## 🎓 Key Learnings & Best Practices

### 1. **Always Verify Category Mapping**
- YOLO uses 0-based indices
- Kaggle expects original COCO category_ids
- Wrong mapping = 0 score!

### 2. **Bbox Format is Critical**
- Training: YOLO format (normalized)
- Submission: COCO format (absolute pixels)
- Conversion must be exact

### 3. **Start Simple, Iterate Fast**
- Begin with YOLOv8n for quick experiments
- Validate pipeline before training large models
- Use small epoch count for testing

### 4. **Monitor Both Training and Validation**
- Large gap → overfitting → increase augmentation
- Both poor → increase model capacity or train longer

### 5. **Threshold Tuning is Important**
- Spend time tuning confidence and NMS thresholds
- Can improve score by 5-10% without retraining

### 6. **Test Time Augmentation Works**
- TTA typically adds 2-5% to score
- Worth the 4x inference time for final submission

---

## 🚀 Next Steps

### Immediate Actions
1. ✅ Verify data is in correct location
2. ✅ Run Stage 1 to create COCO format
3. ✅ Run Stage 0 to split data
4. ✅ Run Stage 2 to prepare YOLO dataset
5. ⏳ Run Stage 3 to train baseline model
6. ⏳ Run Stage 4 to evaluate model
7. ⏳ Run Stage 5 to generate submission
8. ⏳ Submit to Kaggle and check leaderboard

### Optimization Experiments
- [ ] Try different model sizes (YOLOv8s, YOLOv8m)
- [ ] Experiment with image size (640 vs 1280)
- [ ] Test different augmentation strategies
- [ ] Tune confidence and NMS thresholds
- [ ] Apply Test Time Augmentation
- [ ] Train ensemble of models

### Advanced Techniques
- [ ] Implement custom loss function (Focal Loss)
- [ ] Try different architectures (RT-DETR, YOLO-NAS)
- [ ] Implement Weighted Box Fusion (WBF) for ensemble
- [ ] Add CutMix/MixUp augmentation
- [ ] Try self-distillation

---

## 📞 Support & Contact

For questions or issues:
1. Check documentation in `docs/`
2. Review error messages in `logs/`
3. Verify configuration in `config_snapshot.yaml`
4. Contact team lead

---

**Status:** ✅ All Phases Complete - Ready for Production Use  
**Last Updated:** 2026-02-05  
**Branch:** `feat/DM-refactor`  
**Maintainer:** GenSpark AI Team

---

**Good luck with the competition! 🏆**
