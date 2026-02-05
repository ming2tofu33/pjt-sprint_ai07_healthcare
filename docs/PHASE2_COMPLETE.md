# Phase 2: Data Pipeline - Completed ✅

## Overview
Phase 2 implements the complete data processing pipeline for COCO format conversion, validation, and train/val splitting.

## Implemented Components

### 1. COCO Utilities (`src/data/coco_utils.py`)
- **COCO JSON loading and saving**
- **Format validation** with comprehensive error checking
- **Category mapping** (YOLO index ↔ COCO category_id)
  - ⚠️ **Critical for submission**: YOLO uses 0-indexed classes, COCO uses arbitrary IDs
- **COCODataset wrapper class** for easy data access
- **Visualization utilities**

**Key Functions:**
```python
# Load and validate COCO data
coco_data = load_coco_json("data/coco_data/merged_coco.json")
is_valid, errors = validate_coco_format(coco_data)

# Create category mapping (essential for submission!)
coco_to_yolo, yolo_to_coco, yolo_to_name = create_category_mapping(coco_data)
save_category_mapping(yolo_to_coco, yolo_to_name, "data/coco_data/category_mapping.json")

# Use COCODataset wrapper
dataset = COCODataset(
    coco_json_path="data/coco_data/merged_coco.json",
    image_dir="data/raw/train_images"
)
print(f"Classes: {dataset.get_class_names()}")
```

### 2. Split Utilities (`src/data/split_utils.py`)
- **Stratified split by object count** (1, 2, 3, 4 objects per image)
- **Stratified split by class distribution**
- **K-Fold cross-validation** with stratification
- **Detailed split statistics** (object counts, class distribution)

**Key Functions:**
```python
# Stratified split by object count (recommended)
train_ids, val_ids = stratified_split_by_object_count(
    coco_data,
    train_ratio=0.8,
    seed=42
)

# K-Fold split
folds = kfold_split(
    coco_data,
    n_folds=5,
    seed=42,
    stratify_by='object_count'
)

# Save split info
save_split_info(train_ids, val_ids, "data/splits/split_info.json")

# Print statistics
print_split_statistics(coco_data, train_ids, val_ids)
```

### 3. Stage 1: COCO Format Creation (`scripts/1_create_coco_format.py`)
Merges 763 JSON files (from 114 directories) into single COCO JSON.

**Features:**
- ✅ Automatic image filename extraction from JSON path
- ✅ Image existence validation
- ✅ Bbox validation and clipping to image boundaries
- ✅ Optional class filtering (`--include_classes` / `--exclude_classes`)
- ✅ Category mapping generation
- ✅ Comprehensive statistics

**Usage:**
```bash
# Basic merge
python scripts/1_create_coco_format.py \
    --train_images data/raw/train_images \
    --train_annotations data/raw/train_annotations \
    --output_dir data/coco_data \
    --verbose

# With class filtering (if needed)
python scripts/1_create_coco_format.py \
    --train_images data/raw/train_images \
    --train_annotations data/raw/train_annotations \
    --output_dir data/coco_data \
    --include_classes 1,2,3,5,7,11 \
    --validate

# Output files:
#   data/coco_data/merged_coco.json          # Main COCO JSON
#   data/coco_data/category_mapping.json     # YOLO <-> COCO mapping
```

### 4. Stage 0: Data Splitting (`scripts/0_splitting.py`)
Creates train/val splits with stratification.

**Features:**
- ✅ Stratified by object count (ensures balanced 1/2/3/4 object distribution)
- ✅ Stratified by class distribution
- ✅ K-Fold cross-validation support
- ✅ Detailed split statistics

**Usage:**
```bash
# Simple train/val split (80/20)
python scripts/0_splitting.py \
    --coco_json data/coco_data/merged_coco.json \
    --output_dir data/splits \
    --train_ratio 0.8 \
    --stratify_by object_count \
    --verbose

# K-Fold split
python scripts/0_splitting.py \
    --coco_json data/coco_data/merged_coco.json \
    --output_dir data/splits \
    --kfold 5 \
    --stratify_by object_count

# Use specific fold as default
python scripts/0_splitting.py \
    --coco_json data/coco_data/merged_coco.json \
    --output_dir data/splits \
    --kfold 5 \
    --fold 0

# Output files:
#   data/splits/split_info.json              # Train/val image IDs
#   data/splits/fold_0/split_info.json       # K-Fold splits (if --kfold)
```

## Data Pipeline Workflow

### Step 1: Merge Annotations to COCO Format
```bash
python scripts/1_create_coco_format.py \
    --train_images data/raw/train_images \
    --train_annotations data/raw/train_annotations \
    --output_dir data/coco_data \
    --validate \
    --verbose
```

**Expected Output:**
```
data/coco_data/
├── merged_coco.json          # 232 images, ~N annotations
└── category_mapping.json     # YOLO index <-> COCO category_id
```

### Step 2: Create Train/Val Split
```bash
python scripts/0_splitting.py \
    --coco_json data/coco_data/merged_coco.json \
    --output_dir data/splits \
    --train_ratio 0.8 \
    --stratify_by object_count \
    --seed 42 \
    --verbose
```

**Expected Output:**
```
data/splits/
└── split_info.json           # {"train_ids": [...], "val_ids": [...]}
```

### Step 3: Verify Split Statistics
The script automatically prints:
- Total images (train/val counts)
- Object count distribution per split
- Class distribution per split

## Key Features

### 1. **Category ID Mapping (Critical for Submission!)**
```
YOLO Training:  class_index = 0, 1, 2, ..., N-1
COCO Original:  category_id = 1, 11, 24, 69, ...
```

**Why this matters:**
- YOLO model predicts class indices (0-based)
- Submission requires original COCO category_ids
- **Must use `yolo_to_coco` mapping when creating submission CSV**

**Example:**
```python
# Load mapping
with open('data/coco_data/category_mapping.json', 'r') as f:
    mapping = json.load(f)
yolo_to_coco = {int(k): v for k, v in mapping['yolo_to_coco'].items()}

# During submission
yolo_prediction = 3  # YOLO predicted class 3
coco_category_id = yolo_to_coco[yolo_prediction]  # Convert to COCO ID
```

### 2. **Stratified Splitting**
Ensures both train and val sets have similar distributions:
- **Object count**: Images with 1, 2, 3, or 4 objects
- **Class distribution**: Each class appears in both splits

**Example Distribution:**
```
Objects    Train           Val
1            35 (35.0%)      8 (32.0%)
2            20 (20.0%)      5 (20.0%)
3            30 (30.0%)      8 (32.0%)
4            15 (15.0%)      4 (16.0%)
```

### 3. **COCO Format Validation**
Automatically checks:
- ✅ Required keys (images, annotations, categories)
- ✅ Unique IDs (no duplicates)
- ✅ Valid bbox format [x, y, w, h]
- ✅ Bbox boundaries (w > 0, h > 0, within image)
- ✅ Valid references (image_id, category_id exist)

### 4. **Class Filtering (Optional)**
Filter specific classes during COCO creation:
```bash
# Include only specific classes
--include_classes 1,2,3,5,7,11

# Exclude problematic classes
--exclude_classes 99,100
```

## Testing

Run Phase 2 tests with synthetic data:
```bash
python tests/test_phase2.py
```

**Test Output:**
```
✓ TEST 1: COCO Format Validation
✓ TEST 2: Category Mapping
✓ TEST 3: Stratified Split by Object Count
✓ TEST 4: K-Fold Cross-Validation
✓ TEST 5: COCODataset Class

✓ ALL PHASE 2 TESTS PASSED!
```

## Directory Structure After Phase 2
```
data/
├── raw/
│   ├── train_images/          # 232 PNG images
│   ├── train_annotations/     # 763 JSON files (114 directories)
│   └── test_images/           # 842 PNG images
├── coco_data/
│   ├── merged_coco.json       # Single COCO JSON
│   └── category_mapping.json  # YOLO <-> COCO mapping
└── splits/
    ├── split_info.json        # Train/val split
    └── fold_0/                # K-Fold splits (if used)
        └── split_info.json
```

## Important Notes

### ⚠️ Critical for Kaggle Submission
1. **Always use `category_mapping.json`** to convert YOLO predictions to COCO category_ids
2. **Image ID extraction**: Must match Kaggle's expected format
   - Example: `test_001.png` → `image_id = 1`
3. **Bbox format**: Use absolute coordinates [x, y, w, h], NOT normalized

### 📊 Data Statistics (Expected for Real Data)
Based on notebook analysis:
- **Train images**: 232 PNG files
- **Train annotations**: 763 JSON files in 114 directories
- **Test images**: 842 PNG files
- **Classes**: Variable (some classes only in train, ~40 in test)
- **Objects per image**: 1-4 (maximum 4 per image)

### 🔧 Troubleshooting

**Issue: "Image not found" errors**
```bash
# Verify image paths
ls data/raw/train_images/*.png | wc -l  # Should be 232

# Check first few files
ls data/raw/train_images/ | head -5
```

**Issue: Invalid bbox warnings**
- Script automatically clips bboxes to image boundaries
- Invalid bboxes (w≤0 or h≤0) are skipped
- Check `stats['invalid_bbox_count']` in output

**Issue: Class imbalance in splits**
- Use `--stratify_by class` instead of `object_count`
- Or use K-Fold for better class coverage

## Next Steps (Phase 3)
- [ ] Implement model training framework
- [ ] Create YOLO trainer wrapper
- [ ] Add training script with config support
- [ ] Implement evaluation metrics (mAP@0.75-0.95)

---

**Status:** ✅ Completed and Tested  
**Date:** 2026-02-05  
**Ready for:** Local testing with real data
