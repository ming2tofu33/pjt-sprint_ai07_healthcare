"""
Stage 5: Kaggle Submission Script

Generates submission CSV file for Kaggle competition.

Features:
- Test image inference
- YOLO index → COCO category_id mapping
- Top-4 predictions per image
- NMS and confidence thresholding
- CSV validation
- Optional TTA (Test Time Augmentation)

Usage:
    # Basic submission
    python scripts/5_submission.py \\
        --checkpoint runs/exp001_*/checkpoints/best.pt \\
        --test_images data/raw/test_images \\
        --category_mapping data/coco_data/category_mapping.json \\
        --output_dir submissions

    # With custom thresholds
    python scripts/5_submission.py \\
        --checkpoint runs/exp001_*/checkpoints/best.pt \\
        --test_images data/raw/test_images \\
        --category_mapping data/coco_data/category_mapping.json \\
        --conf_threshold 0.25 \\
        --iou_nms 0.45 \\
        --max_det 4

    # With TTA
    python scripts/5_submission.py \\
        --checkpoint runs/exp001_*/checkpoints/best.pt \\
        --test_images data/raw/test_images \\
        --category_mapping data/coco_data/category_mapping.json \\
        --tta
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from src.utils import set_seed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Kaggle submission CSV from trained model"
    )
    
    # Model
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pt file)'
    )
    
    # Data
    parser.add_argument(
        '--test_images',
        type=str,
        default='data/raw/test_images',
        help='Directory containing test images (default: data/raw/test_images)'
    )
    parser.add_argument(
        '--category_mapping',
        type=str,
        default='data/coco_data/category_mapping.json',
        help='Path to category mapping JSON (default: data/coco_data/category_mapping.json)'
    )
    
    # Inference parameters
    parser.add_argument(
        '--conf_threshold',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--iou_nms',
        type=float,
        default=0.45,
        help='IoU threshold for NMS (default: 0.45)'
    )
    parser.add_argument(
        '--max_det',
        type=int,
        default=4,
        help='Maximum detections per image (default: 4)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size (default: 640)'
    )
    
    # TTA (Test Time Augmentation)
    parser.add_argument(
        '--tta',
        action='store_true',
        help='Enable Test Time Augmentation'
    )
    
    # Output
    parser.add_argument(
        '--output_dir',
        type=str,
        default='submissions',
        help='Output directory for submission files (default: submissions)'
    )
    parser.add_argument(
        '--output_name',
        type=str,
        default=None,
        help='Output CSV filename (default: auto-generated with timestamp)'
    )
    
    # Other
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress'
    )
    
    return parser.parse_args()


def extract_image_id_from_filename(filename: str) -> int:
    """Extract image_id from test image filename.
    
    Examples:
        test_001.png -> 1
        test_042.png -> 42
        test_842.png -> 842
    
    Args:
        filename: Image filename
        
    Returns:
        Image ID as integer
    """
    # Remove extension
    stem = Path(filename).stem
    
    # Extract number (assumes format: test_XXX or similar)
    # Try to find all digits
    import re
    numbers = re.findall(r'\d+', stem)
    
    if numbers:
        # Take the last number found (usually the ID)
        return int(numbers[-1])
    else:
        # Fallback: hash filename to get unique ID
        return hash(stem) % 100000


def run_inference_on_test_images(
    model,
    test_images_dir: Path,
    conf_threshold: float = 0.25,
    iou_nms: float = 0.45,
    max_det: int = 4,
    imgsz: int = 640,
    use_tta: bool = False,
    verbose: bool = False
) -> Dict[str, List[Dict]]:
    """Run inference on all test images.
    
    Args:
        model: YOLO model
        test_images_dir: Directory containing test images
        conf_threshold: Confidence threshold
        iou_nms: IoU threshold for NMS
        max_det: Maximum detections per image
        imgsz: Input image size
        use_tta: Enable Test Time Augmentation
        verbose: Print detailed progress
        
    Returns:
        Dictionary mapping image_filename to list of predictions
    """
    print(f"\n{'='*80}")
    print("RUNNING INFERENCE ON TEST IMAGES")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    print(f"  Confidence threshold: {conf_threshold}")
    print(f"  NMS IoU threshold: {iou_nms}")
    print(f"  Max detections per image: {max_det}")
    print(f"  Image size: {imgsz}")
    print(f"  TTA enabled: {use_tta}")
    
    # Get all test images
    image_extensions = ['.png', '.jpg', '.jpeg']
    test_images = []
    for ext in image_extensions:
        test_images.extend(list(test_images_dir.glob(f"*{ext}")))
    
    test_images = sorted(test_images)
    print(f"\nFound {len(test_images)} test images")
    
    if len(test_images) == 0:
        raise ValueError(f"No test images found in {test_images_dir}")
    
    # Run inference
    predictions_by_image = {}
    
    print(f"\nRunning inference...")
    for idx, image_path in enumerate(test_images):
        if verbose and (idx + 1) % 100 == 0:
            print(f"  Progress: {idx + 1}/{len(test_images)} images...")
        
        # Run prediction
        results = model.predict(
            source=str(image_path),
            conf=conf_threshold,
            iou=iou_nms,
            max_det=max_det,
            imgsz=imgsz,
            augment=use_tta,
            verbose=False
        )
        
        # Extract predictions
        image_predictions = []
        
        if len(results) > 0:
            result = results[0]  # Single image
            
            # Get boxes, scores, and class IDs
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Get bbox in xyxy format and convert to xywh
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    
                    # Convert to xywh (COCO format)
                    x = float(x1)
                    y = float(y1)
                    w = float(x2 - x1)
                    h = float(y2 - y1)
                    
                    # Get confidence and class
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    image_predictions.append({
                        'bbox': [x, y, w, h],
                        'score': conf,
                        'class_id': cls  # This is YOLO class index (0-based)
                    })
        
        # Sort by score and keep top max_det
        image_predictions = sorted(image_predictions, key=lambda x: x['score'], reverse=True)
        image_predictions = image_predictions[:max_det]
        
        predictions_by_image[image_path.name] = image_predictions
    
    print(f"\n✓ Inference completed")
    print(f"  Total predictions: {sum(len(preds) for preds in predictions_by_image.values())}")
    print(f"  Average per image: {sum(len(preds) for preds in predictions_by_image.values()) / len(test_images):.2f}")
    
    return predictions_by_image


def create_submission_csv(
    predictions_by_image: Dict[str, List[Dict]],
    yolo_to_coco_mapping: Dict[int, int],
    output_path: str,
    validate: bool = True
) -> pd.DataFrame:
    """Create Kaggle submission CSV from predictions.
    
    Args:
        predictions_by_image: Dictionary of predictions per image
        yolo_to_coco_mapping: Mapping from YOLO class index to COCO category_id
        output_path: Path to save CSV
        validate: Whether to validate submission format
        
    Returns:
        Submission DataFrame
    """
    print(f"\n{'='*80}")
    print("CREATING SUBMISSION CSV")
    print(f"{'='*80}")
    
    # Prepare submission rows
    submission_rows = []
    annotation_id = 1
    
    for image_filename, predictions in sorted(predictions_by_image.items()):
        # Extract image_id from filename
        image_id = extract_image_id_from_filename(image_filename)
        
        # Add predictions for this image
        for pred in predictions:
            # Convert YOLO class index to COCO category_id
            yolo_class = pred['class_id']
            
            if yolo_class not in yolo_to_coco_mapping:
                print(f"  Warning: Unknown YOLO class {yolo_class} in {image_filename}, skipping")
                continue
            
            coco_category_id = yolo_to_coco_mapping[yolo_class]
            
            # Get bbox (already in absolute coordinates)
            x, y, w, h = pred['bbox']
            
            # Create submission row
            submission_rows.append({
                'annotation_id': annotation_id,
                'image_id': image_id,
                'category_id': coco_category_id,
                'bbox_x': x,
                'bbox_y': y,
                'bbox_w': w,
                'bbox_h': h,
                'score': pred['score']
            })
            
            annotation_id += 1
    
    # Create DataFrame
    submission_df = pd.DataFrame(submission_rows)
    
    print(f"\n✓ Created submission with {len(submission_df)} annotations")
    print(f"  Unique images: {submission_df['image_id'].nunique()}")
    print(f"  Average detections per image: {len(submission_df) / submission_df['image_id'].nunique():.2f}")
    
    # Validate submission
    if validate:
        print(f"\nValidating submission format...")
        
        # Check required columns
        required_columns = ['annotation_id', 'image_id', 'category_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score']
        missing_columns = set(required_columns) - set(submission_df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for duplicate annotation_ids
        if submission_df['annotation_id'].duplicated().any():
            raise ValueError("Found duplicate annotation_ids")
        
        # Check bbox validity
        if (submission_df['bbox_w'] <= 0).any() or (submission_df['bbox_h'] <= 0).any():
            print("  Warning: Found invalid bbox dimensions (w or h <= 0)")
        
        if (submission_df['bbox_x'] < 0).any() or (submission_df['bbox_y'] < 0).any():
            print("  Warning: Found negative bbox coordinates")
        
        # Check max detections per image
        detections_per_image = submission_df.groupby('image_id').size()
        if (detections_per_image > 4).any():
            print(f"  Warning: Some images have > 4 detections")
            print(f"    Max detections: {detections_per_image.max()}")
        
        # Check score range
        if (submission_df['score'] < 0).any() or (submission_df['score'] > 1).any():
            print("  Warning: Found scores outside [0, 1] range")
        
        print(f"  ✓ Validation completed")
    
    # Save CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    submission_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved submission to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")
    
    return submission_df


def print_submission_summary(submission_df: pd.DataFrame) -> None:
    """Print summary of submission."""
    print(f"\n{'='*80}")
    print("SUBMISSION SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nOverall Statistics:")
    print(f"  Total annotations: {len(submission_df)}")
    print(f"  Unique images: {submission_df['image_id'].nunique()}")
    print(f"  Unique categories: {submission_df['category_id'].nunique()}")
    
    print(f"\nDetections per image:")
    detections_per_image = submission_df.groupby('image_id').size()
    print(f"  Min: {detections_per_image.min()}")
    print(f"  Max: {detections_per_image.max()}")
    print(f"  Mean: {detections_per_image.mean():.2f}")
    print(f"  Median: {detections_per_image.median():.1f}")
    
    print(f"\nScore distribution:")
    print(f"  Min: {submission_df['score'].min():.4f}")
    print(f"  Max: {submission_df['score'].max():.4f}")
    print(f"  Mean: {submission_df['score'].mean():.4f}")
    print(f"  Median: {submission_df['score'].median():.4f}")
    
    print(f"\nTop 10 categories by frequency:")
    top_categories = submission_df['category_id'].value_counts().head(10)
    for cat_id, count in top_categories.items():
        pct = count / len(submission_df) * 100
        print(f"  Category {cat_id}: {count} ({pct:.1f}%)")


def main():
    """Main submission generation function."""
    args = parse_args()
    
    if not ULTRALYTICS_AVAILABLE:
        print("Error: Ultralytics YOLO not installed")
        print("Install with: pip install ultralytics")
        sys.exit(1)
    
    # Set seed
    set_seed(args.seed)
    
    print(f"\n{'='*80}")
    print("STAGE 5: KAGGLE SUBMISSION GENERATION")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Test images: {args.test_images}")
    print(f"  Category mapping: {args.category_mapping}")
    print(f"  Output directory: {args.output_dir}")
    
    # Validate paths
    checkpoint_path = Path(args.checkpoint)
    test_images_dir = Path(args.test_images)
    category_mapping_path = Path(args.category_mapping)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not test_images_dir.exists():
        raise FileNotFoundError(f"Test images directory not found: {test_images_dir}")
    if not category_mapping_path.exists():
        raise FileNotFoundError(f"Category mapping not found: {category_mapping_path}")
    
    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    model = YOLO(str(checkpoint_path))
    print(f"  ✓ Model loaded")
    
    # Load category mapping
    print(f"\nLoading category mapping from {category_mapping_path}...")
    with open(category_mapping_path, 'r') as f:
        mapping = json.load(f)
    
    # Create YOLO index -> COCO category_id mapping
    yolo_to_coco = {int(k): v for k, v in mapping['yolo_to_coco'].items()}
    num_classes = mapping['num_classes']
    
    print(f"  Number of classes: {num_classes}")
    print(f"  YOLO index range: 0-{num_classes-1}")
    print(f"  COCO category_id range: {min(yolo_to_coco.values())}-{max(yolo_to_coco.values())}")
    
    # Run inference
    predictions_by_image = run_inference_on_test_images(
        model=model,
        test_images_dir=test_images_dir,
        conf_threshold=args.conf_threshold,
        iou_nms=args.iou_nms,
        max_det=args.max_det,
        imgsz=args.imgsz,
        use_tta=args.tta,
        verbose=args.verbose
    )
    
    # Generate output filename
    if args.output_name:
        output_filename = args.output_name
        if not output_filename.endswith('.csv'):
            output_filename += '.csv'
    else:
        # Auto-generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = checkpoint_path.parent.parent.name  # Get experiment name from path
        output_filename = f"submission_{exp_name}_{timestamp}.csv"
    
    output_path = Path(args.output_dir) / output_filename
    
    # Create submission CSV
    submission_df = create_submission_csv(
        predictions_by_image=predictions_by_image,
        yolo_to_coco_mapping=yolo_to_coco,
        output_path=str(output_path),
        validate=True
    )
    
    # Print summary
    print_submission_summary(submission_df)
    
    print(f"\n{'='*80}")
    print("✓ SUBMISSION GENERATION COMPLETED")
    print(f"{'='*80}")
    print(f"\nSubmission file: {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Review submission file: {output_path}")
    print(f"  2. Upload to Kaggle competition")
    print(f"  3. Check Public Leaderboard score")
    print(f"\nGood luck! 🚀")


if __name__ == "__main__":
    main()
