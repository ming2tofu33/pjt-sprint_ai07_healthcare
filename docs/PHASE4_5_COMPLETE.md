# Phase 4 & 5: Evaluation and Submission - Completed ✅

## Overview
Phase 4 implements comprehensive evaluation metrics and visualization tools.
Phase 5 implements Kaggle submission file generation with proper format conversion.

## Phase 4: Evaluation & Visualization

### Implemented Components

#### 1. Evaluation Metrics (`src/evaluation/metrics.py`)
Comprehensive object detection metrics implementation.

**Features:**
- ✅ IoU (Intersection over Union) calculation
- ✅ Average Precision (AP) at specific IoU threshold
- ✅ mAP@[0.75:0.95] (Primary Kaggle metric)
- ✅ mAP@0.50, mAP@0.75 (Reference metrics)
- ✅ Per-class AP calculation
- ✅ Confusion matrix computation

**Key Functions:**
```python
# Calculate mAP over IoU range (primary metric)
average_map, map_per_iou, per_class_ap = calculate_map_range(
    predictions,
    ground_truths,
    iou_thresholds=[0.75, 0.80, 0.85, 0.90, 0.95]
)

# Comprehensive evaluation
results = evaluate_detections(
    predictions,
    ground_truths,
    num_classes=40,
    iou_thresholds=[0.75, 0.80, 0.85, 0.90, 0.95]
)
```

#### 2. Visualization Tools (`src/evaluation/visualizer.py`)
Professional visualization for evaluation results.

**Features:**
- ✅ Prediction visualization on images (GT in green, Pred in red)
- ✅ Precision-Recall curves
- ✅ Confusion matrix heatmaps
- ✅ mAP per IoU threshold plot
- ✅ Per-class AP bar charts
- ✅ Comprehensive evaluation report generation

**Key Functions:**
```python
# Visualize predictions
visualize_predictions(
    image_path="image.png",
    predictions=preds,
    ground_truths=gts,
    class_names=class_names,
    save_path="pred_viz.png"
)

# Create complete evaluation report
create_evaluation_report(
    results=evaluation_results,
    class_names=class_names,
    output_dir="evaluation_results"
)
```

#### 3. Stage 4: Evaluation Script (`scripts/4_evaluate.py`)
Comprehensive model evaluation with YOLO validation.

**Features:**
- ✅ YOLO built-in validation metrics
- ✅ mAP@0.50, mAP@0.50-0.95 calculation
- ✅ mAP@0.75-0.95 approximation
- ✅ Precision, Recall reporting
- ✅ Results saved to JSON and text summary
- ✅ Kaggle submission readiness assessment

**Usage:**
```bash
# Basic evaluation
python scripts/4_evaluate.py \
    --checkpoint runs/exp001_*/checkpoints/best.pt \
    --data_yaml data/yolo_data/data.yaml \
    --output_dir evaluation_results

# With custom thresholds
python scripts/4_evaluate.py \
    --checkpoint runs/exp001_*/checkpoints/best.pt \
    --data_yaml data/yolo_data/data.yaml \
    --conf_threshold 0.001 \
    --iou_nms 0.6
```

**Output:**
```
evaluation_results/
├── evaluation_results.json    # Detailed metrics
└── summary.txt                # Human-readable summary
```

---

## Phase 5: Kaggle Submission

### Implemented Components

#### 1. Stage 5: Submission Script (`scripts/5_submission.py`)
Complete Kaggle submission file generation.

**Features:**
- ✅ Test image inference (842 images)
- ✅ **YOLO index → COCO category_id conversion** (Critical!)
- ✅ Image ID extraction from filename (`test_001.png` → `1`)
- ✅ Top-4 predictions per image (score-based)
- ✅ NMS and confidence thresholding
- ✅ CSV format validation
- ✅ Optional TTA (Test Time Augmentation)
- ✅ Comprehensive submission summary

**Usage:**
```bash
# Basic submission
python scripts/5_submission.py \
    --checkpoint runs/exp001_*/checkpoints/best.pt \
    --test_images data/raw/test_images \
    --category_mapping data/coco_data/category_mapping.json \
    --output_dir submissions

# With custom thresholds
python scripts/5_submission.py \
    --checkpoint runs/exp001_*/checkpoints/best.pt \
    --test_images data/raw/test_images \
    --category_mapping data/coco_data/category_mapping.json \
    --conf_threshold 0.25 \
    --iou_nms 0.45 \
    --max_det 4

# With TTA (improves score but slower)
python scripts/5_submission.py \
    --checkpoint runs/exp001_*/checkpoints/best.pt \
    --test_images data/raw/test_images \
    --category_mapping data/coco_data/category_mapping.json \
    --tta
```

**Output:**
```
submissions/
└── submission_exp001_baseline_20260205_180000.csv
```

**CSV Format:**
```csv
annotation_id,image_id,category_id,bbox_x,bbox_y,bbox_w,bbox_h,score
1,1,15,100.5,200.3,50.2,80.1,0.95
2,1,23,300.1,150.2,60.3,70.4,0.88
3,2,7,50.0,100.0,40.0,50.0,0.92
...
```

### Critical Conversion: YOLO → COCO

**⚠️ Most Important Step:**
```python
# Training: YOLO uses 0-based class indices
yolo_prediction = 3  # Model predicts class 3

# Submission: Must convert to original COCO category_id
coco_category_id = yolo_to_coco[yolo_prediction]  # e.g., 3 → 24

# This mapping is loaded from category_mapping.json
# Created during Stage 1 (COCO format creation)
```

**Why this matters:**
- YOLO trains with indices: 0, 1, 2, ..., N-1
- Original COCO data has IDs: 1, 11, 24, 69, ...
- Kaggle expects original COCO category_ids
- **Wrong mapping = 0 score on leaderboard!**

### Submission Validation

The script automatically validates:
- ✅ Required columns present
- ✅ No duplicate annotation_ids
- ✅ Bbox dimensions positive (w > 0, h > 0)
- ✅ Bbox coordinates non-negative
- ✅ Max 4 detections per image
- ✅ Score in valid range [0, 1]

### Image ID Extraction

```python
# Automatic extraction from filename
"test_001.png" → image_id = 1
"test_042.png" → image_id = 42
"test_842.png" → image_id = 842

# Uses regex to find numbers in filename
# Fallback to hash if no numbers found
```

---

## Complete Pipeline (Phases 1-5)

### Full Workflow:
```bash
# Phase 1: Already set up (Config, Logger, Seed)
# Phase 2: Data preparation
python scripts/1_create_coco_format.py \
    --train_images data/raw/train_images \
    --train_annotations data/raw/train_annotations \
    --output_dir data/coco_data

python scripts/0_splitting.py \
    --coco_json data/coco_data/merged_coco.json \
    --output_dir data/splits

# Phase 3: Model training
python scripts/2_prepare_yolo.py \
    --coco_dir data/coco_data \
    --images_dir data/raw/train_images \
    --splits_dir data/splits \
    --output_dir data/yolo_data

python scripts/3_train.py \
    --config configs/experiments/exp001_baseline.yaml

# Phase 4: Evaluation
python scripts/4_evaluate.py \
    --checkpoint runs/exp001_*/checkpoints/best.pt \
    --data_yaml data/yolo_data/data.yaml \
    --output_dir evaluation_results

# Phase 5: Kaggle submission
python scripts/5_submission.py \
    --checkpoint runs/exp001_*/checkpoints/best.pt \
    --test_images data/raw/test_images \
    --category_mapping data/coco_data/category_mapping.json \
    --output_dir submissions
```

---

## Evaluation Metrics Explanation

### Primary Metric: mAP@[0.75:0.95]
```
mAP@[0.75:0.95] = Average of mAP@0.75, mAP@0.80, mAP@0.85, mAP@0.90, mAP@0.95

This is stricter than COCO's mAP@[0.50:0.95]
Requires more precise bounding boxes
```

### Target Scores:
- **Baseline**: mAP@[0.75:0.95] > 0.30
- **Competitive**: mAP@[0.75:0.95] > 0.50
- **Top performers**: mAP@[0.75:0.95] > 0.70

### Secondary Metrics (Reference):
- mAP@0.50: More lenient, easier to achieve
- mAP@0.75: Good middle ground
- Precision: What % of predictions are correct
- Recall: What % of ground truths are detected

---

## Submission Best Practices

### 1. **Confidence Threshold Tuning**
```bash
# Try different thresholds
--conf_threshold 0.15  # More predictions (higher recall)
--conf_threshold 0.25  # Balanced (default)
--conf_threshold 0.35  # Fewer but more confident (higher precision)
```

### 2. **NMS Threshold Tuning**
```bash
# Lower = more aggressive suppression
--iou_nms 0.40  # Aggressive

# Higher = keep more overlapping boxes
--iou_nms 0.50  # Moderate (default: 0.45)
```

### 3. **Test Time Augmentation (TTA)**
```bash
# Improves score but slower (4x inference time)
--tta

# TTA applies horizontal flip and averages predictions
```

### 4. **Model Ensemble (Manual)**
```bash
# Generate submissions from multiple models
python scripts/5_submission.py --checkpoint runs/exp001_*/checkpoints/best.pt ...
python scripts/5_submission.py --checkpoint runs/exp002_*/checkpoints/best.pt ...

# Then manually combine predictions (weighted averaging, WBF, etc.)
```

---

## Troubleshooting

### Issue: "Low mAP score on Kaggle"
**Possible causes:**
1. ❌ Wrong category_id mapping
   - **Fix**: Verify `category_mapping.json` is used correctly
   - Check: YOLO index → COCO category_id conversion

2. ❌ Wrong bbox format
   - **Fix**: Ensure absolute coordinates (pixels), not normalized
   - Check: Bbox values should be >> 1 (not in 0-1 range)

3. ❌ Wrong image_id extraction
   - **Fix**: Verify image_id matches Kaggle's expected format
   - Check: Print image_id for a few samples

### Issue: "CSV validation errors"
```bash
# Check for issues
python -c "
import pandas as pd
df = pd.read_csv('submissions/submission.csv')
print(df.head())
print(df.describe())
print(f'Unique images: {df['image_id'].nunique()}')
print(f'Max detections per image: {df.groupby('image_id').size().max()}')
"
```

### Issue: "Too few/many detections"
```bash
# Adjust confidence threshold
--conf_threshold 0.15  # More detections
--conf_threshold 0.35  # Fewer detections

# Check average detections per image
# Should be around 2-3 for this competition
```

---

## Testing

Tests for Phase 4 & 5 are integrated into the scripts (validation checks).

**Manual testing:**
```bash
# Test evaluation
python scripts/4_evaluate.py \
    --checkpoint runs/exp001_*/checkpoints/best.pt \
    --data_yaml data/yolo_data/data.yaml

# Test submission generation
python scripts/5_submission.py \
    --checkpoint runs/exp001_*/checkpoints/best.pt \
    --test_images data/raw/test_images \
    --category_mapping data/coco_data/category_mapping.json \
    --verbose
```

---

## Final Checklist Before Kaggle Submission

- [ ] Model trained with good validation mAP
- [ ] Evaluation shows mAP@0.75-0.95 > 0.30
- [ ] Submission CSV generated successfully
- [ ] CSV validation passed (no errors)
- [ ] Verified category_mapping.json is correct
- [ ] Checked bbox format (absolute coordinates)
- [ ] Verified image_id extraction is correct
- [ ] Submission file size reasonable (<10MB)
- [ ] Unique images count = 842 (test set size)
- [ ] Average detections per image ≈ 2-3
- [ ] Ready to upload to Kaggle!

---

**Status:** ✅ Completed and Tested  
**Date:** 2026-02-05  
**Ready for:** Kaggle Submission! 🚀
