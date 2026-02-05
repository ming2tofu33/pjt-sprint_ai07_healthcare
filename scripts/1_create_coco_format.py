"""
Stage 1: COCO Format Creation Script

Merges multiple JSON annotation files into a single COCO format JSON.

Features:
- Merge 763 JSON files from 114 directories into single COCO JSON
- Create category mapping (YOLO index <-> COCO category_id)
- Validate COCO format
- Generate dataset statistics
- Optional class filtering

Usage:
    # Basic merge
    python scripts/1_create_coco_format.py \\
        --train_images data/raw/train_images \\
        --train_annotations data/raw/train_annotations \\
        --output_dir data/coco_data

    # With class filtering
    python scripts/1_create_coco_format.py \\
        --train_images data/raw/train_images \\
        --train_annotations data/raw/train_annotations \\
        --output_dir data/coco_data \\
        --include_classes 1,2,3,5,7,11
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.coco_utils import (
    create_category_mapping,
    save_category_mapping,
    validate_coco_format,
)
from src.utils import set_seed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create COCO format dataset from multiple JSON files"
    )
    
    # Input paths
    parser.add_argument(
        '--train_images',
        type=str,
        required=True,
        help='Path to train images directory'
    )
    parser.add_argument(
        '--train_annotations',
        type=str,
        required=True,
        help='Path to train annotations directory (contains subdirectories with JSON files)'
    )
    
    # Output paths
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/coco_data',
        help='Output directory for COCO JSON and mappings (default: data/coco_data)'
    )
    parser.add_argument(
        '--output_name',
        type=str,
        default='merged_coco.json',
        help='Output COCO JSON filename (default: merged_coco.json)'
    )
    
    # Class filtering
    parser.add_argument(
        '--include_classes',
        type=str,
        default=None,
        help='Comma-separated list of category IDs to include (e.g., "1,2,3,5,7")'
    )
    parser.add_argument(
        '--exclude_classes',
        type=str,
        default=None,
        help='Comma-separated list of category IDs to exclude (e.g., "10,15,20")'
    )
    
    # Validation
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation after creating COCO JSON'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed statistics'
    )
    
    return parser.parse_args()


def find_all_json_files(annotations_dir: Path) -> List[Path]:
    """Find all JSON files in annotations directory and subdirectories.
    
    Args:
        annotations_dir: Root annotations directory
        
    Returns:
        List of JSON file paths
    """
    json_files = list(annotations_dir.rglob("*.json"))
    return sorted(json_files)


def load_single_annotation(json_path: Path) -> Dict:
    """Load a single annotation JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Annotation data dictionary
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_image_filename_from_json_path(json_path: Path) -> str:
    """Extract image filename from JSON path.
    
    Example:
        data/raw/train_annotations/K-001900-016548-019607-029451_json/K-001900/K-001900-016548-019607-029451_0_2_0_2_70_000_200.json
        -> K-001900-016548-019607-029451_0_2_0_2_70_000_200.png
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Image filename (with .png extension)
    """
    # JSON filename without extension
    json_stem = json_path.stem
    # Assume image has same name with .png extension
    return f"{json_stem}.png"


def merge_annotations_to_coco(
    train_images_dir: Path,
    train_annotations_dir: Path,
    include_classes: Set[int] = None,
    exclude_classes: Set[int] = None,
    verbose: bool = False
) -> Dict:
    """Merge multiple annotation JSON files into single COCO format.
    
    Args:
        train_images_dir: Directory containing training images
        train_annotations_dir: Directory containing annotation JSON files
        include_classes: Set of category IDs to include (None = all)
        exclude_classes: Set of category IDs to exclude (None = none)
        verbose: Print progress messages
        
    Returns:
        COCO format dictionary
    """
    print(f"\n{'='*80}")
    print("MERGING ANNOTATIONS TO COCO FORMAT")
    print(f"{'='*80}")
    
    # Find all JSON files
    print(f"\nScanning for JSON files in {train_annotations_dir}...")
    json_files = find_all_json_files(train_annotations_dir)
    print(f"  Found {len(json_files)} JSON files")
    
    # Initialize COCO structure
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    # Track IDs and categories
    image_id_counter = 1
    annotation_id_counter = 1
    image_filename_to_id = {}
    category_ids_seen = set()
    category_id_to_name = {}
    
    # Statistics
    stats = {
        'total_json_files': len(json_files),
        'processed_images': 0,
        'processed_annotations': 0,
        'skipped_images': 0,
        'skipped_annotations': 0,
        'invalid_bbox_count': 0
    }
    
    print(f"\nProcessing JSON files...")
    for idx, json_path in enumerate(json_files):
        if verbose and (idx + 1) % 100 == 0:
            print(f"  Progress: {idx + 1}/{len(json_files)} files...")
        
        try:
            # Load annotation
            ann_data = load_single_annotation(json_path)
            
            # Extract image filename
            image_filename = extract_image_filename_from_json_path(json_path)
            image_path = train_images_dir / image_filename
            
            # Check if image exists
            if not image_path.exists():
                if verbose:
                    print(f"  Warning: Image not found: {image_path}")
                stats['skipped_images'] += 1
                continue
            
            # Get image dimensions (if available in annotation, otherwise read from image)
            if 'image' in ann_data and 'width' in ann_data['image'] and 'height' in ann_data['image']:
                image_width = ann_data['image']['width']
                image_height = ann_data['image']['height']
            else:
                # Read image to get dimensions
                from PIL import Image
                with Image.open(image_path) as img:
                    image_width, image_height = img.size
            
            # Add image to COCO (only if not already added)
            if image_filename not in image_filename_to_id:
                coco_data['images'].append({
                    'id': image_id_counter,
                    'file_name': image_filename,
                    'width': image_width,
                    'height': image_height
                })
                image_filename_to_id[image_filename] = image_id_counter
                current_image_id = image_id_counter
                image_id_counter += 1
                stats['processed_images'] += 1
            else:
                current_image_id = image_filename_to_id[image_filename]
            
            # Process annotations
            if 'annotations' in ann_data:
                for ann in ann_data['annotations']:
                    # Extract annotation fields
                    category_id = ann.get('category_id') or ann.get('dl_idx')
                    bbox = ann.get('bbox')
                    
                    if category_id is None or bbox is None:
                        stats['skipped_annotations'] += 1
                        continue
                    
                    # Apply class filtering
                    if include_classes and category_id not in include_classes:
                        stats['skipped_annotations'] += 1
                        continue
                    if exclude_classes and category_id in exclude_classes:
                        stats['skipped_annotations'] += 1
                        continue
                    
                    # Validate bbox format
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        stats['invalid_bbox_count'] += 1
                        stats['skipped_annotations'] += 1
                        continue
                    
                    x, y, w, h = bbox
                    
                    # Validate bbox values
                    if w <= 0 or h <= 0:
                        stats['invalid_bbox_count'] += 1
                        stats['skipped_annotations'] += 1
                        continue
                    
                    # Clip bbox to image boundaries
                    if x < 0:
                        w += x
                        x = 0
                    if y < 0:
                        h += y
                        y = 0
                    if x + w > image_width:
                        w = image_width - x
                    if y + h > image_height:
                        h = image_height - y
                    
                    # Skip if bbox is completely outside image
                    if w <= 0 or h <= 0:
                        stats['invalid_bbox_count'] += 1
                        stats['skipped_annotations'] += 1
                        continue
                    
                    # Track category
                    category_ids_seen.add(category_id)
                    if category_id not in category_id_to_name:
                        category_name = ann.get('category_name') or ann.get('dl_name') or f"class_{category_id}"
                        category_id_to_name[category_id] = category_name
                    
                    # Add annotation
                    coco_data['annotations'].append({
                        'id': annotation_id_counter,
                        'image_id': current_image_id,
                        'category_id': category_id,
                        'bbox': [x, y, w, h],
                        'area': w * h,
                        'iscrowd': 0
                    })
                    annotation_id_counter += 1
                    stats['processed_annotations'] += 1
        
        except Exception as e:
            if verbose:
                print(f"  Error processing {json_path}: {e}")
            stats['skipped_images'] += 1
    
    # Add categories to COCO
    for category_id in sorted(category_ids_seen):
        coco_data['categories'].append({
            'id': category_id,
            'name': category_id_to_name[category_id],
            'supercategory': 'pill'
        })
    
    # Print statistics
    print(f"\n{'='*80}")
    print("MERGE STATISTICS")
    print(f"{'='*80}")
    print(f"\nInput:")
    print(f"  Total JSON files: {stats['total_json_files']}")
    print(f"\nOutput:")
    print(f"  Images: {len(coco_data['images'])}")
    print(f"  Annotations: {len(coco_data['annotations'])}")
    print(f"  Categories: {len(coco_data['categories'])}")
    print(f"\nSkipped:")
    print(f"  Images: {stats['skipped_images']}")
    print(f"  Annotations: {stats['skipped_annotations']}")
    print(f"  Invalid bboxes: {stats['invalid_bbox_count']}")
    
    return coco_data


def print_dataset_statistics(coco_data: Dict) -> None:
    """Print detailed dataset statistics.
    
    Args:
        coco_data: COCO format dictionary
    """
    print(f"\n{'='*80}")
    print("DATASET STATISTICS")
    print(f"{'='*80}")
    
    # Basic counts
    print(f"\nBasic counts:")
    print(f"  Images: {len(coco_data['images'])}")
    print(f"  Annotations: {len(coco_data['annotations'])}")
    print(f"  Categories: {len(coco_data['categories'])}")
    
    # Objects per image
    image_object_counts = Counter()
    for ann in coco_data['annotations']:
        image_object_counts[ann['image_id']] += 1
    
    object_count_dist = Counter(image_object_counts.values())
    
    print(f"\nObjects per image:")
    for count in sorted(object_count_dist.keys()):
        num_images = object_count_dist[count]
        pct = num_images / len(coco_data['images']) * 100
        print(f"  {count} objects: {num_images} images ({pct:.1f}%)")
    
    # Class distribution
    class_counts = Counter()
    for ann in coco_data['annotations']:
        class_counts[ann['category_id']] += 1
    
    print(f"\nTop 10 classes by frequency:")
    for category_id, count in class_counts.most_common(10):
        pct = count / len(coco_data['annotations']) * 100
        cat_name = next((cat['name'] for cat in coco_data['categories'] if cat['id'] == category_id), 'unknown')
        print(f"  Class {category_id} ({cat_name}): {count} annotations ({pct:.1f}%)")
    
    # Bbox size statistics
    bbox_areas = [ann['area'] for ann in coco_data['annotations']]
    print(f"\nBbox area statistics:")
    print(f"  Min: {min(bbox_areas):.1f}")
    print(f"  Max: {max(bbox_areas):.1f}")
    print(f"  Mean: {sum(bbox_areas)/len(bbox_areas):.1f}")
    print(f"  Median: {sorted(bbox_areas)[len(bbox_areas)//2]:.1f}")


def main():
    """Main function."""
    args = parse_args()
    
    print(f"\n{'='*80}")
    print("STAGE 1: COCO FORMAT CREATION")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    print(f"  Train images: {args.train_images}")
    print(f"  Train annotations: {args.train_annotations}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Output name: {args.output_name}")
    
    # Parse class filtering
    include_classes = None
    exclude_classes = None
    
    if args.include_classes:
        include_classes = set(int(x.strip()) for x in args.include_classes.split(','))
        print(f"  Include classes: {sorted(include_classes)}")
    
    if args.exclude_classes:
        exclude_classes = set(int(x.strip()) for x in args.exclude_classes.split(','))
        print(f"  Exclude classes: {sorted(exclude_classes)}")
    
    # Convert paths
    train_images_dir = Path(args.train_images)
    train_annotations_dir = Path(args.train_annotations)
    output_dir = Path(args.output_dir)
    
    # Validate input paths
    if not train_images_dir.exists():
        raise FileNotFoundError(f"Train images directory not found: {train_images_dir}")
    if not train_annotations_dir.exists():
        raise FileNotFoundError(f"Train annotations directory not found: {train_annotations_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge annotations to COCO format
    coco_data = merge_annotations_to_coco(
        train_images_dir,
        train_annotations_dir,
        include_classes=include_classes,
        exclude_classes=exclude_classes,
        verbose=args.verbose
    )
    
    # Save COCO JSON
    coco_json_path = output_dir / args.output_name
    print(f"\nSaving COCO JSON to {coco_json_path}...")
    with open(coco_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {coco_json_path} ({coco_json_path.stat().st_size / 1024 / 1024:.2f} MB)")
    
    # Create and save category mapping
    print(f"\nCreating category mapping...")
    coco_to_yolo, yolo_to_coco, yolo_to_name = create_category_mapping(coco_data)
    
    mapping_path = output_dir / "category_mapping.json"
    save_category_mapping(yolo_to_coco, yolo_to_name, str(mapping_path))
    print(f"  Saved: {mapping_path}")
    print(f"  Number of classes: {len(yolo_to_coco)}")
    
    # Validate COCO format
    if args.validate:
        print(f"\nValidating COCO format...")
        is_valid, errors = validate_coco_format(coco_data)
        if is_valid:
            print(f"  ✓ COCO format is valid")
        else:
            print(f"  ✗ COCO format has errors:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"    - {error}")
            if len(errors) > 10:
                print(f"    ... and {len(errors) - 10} more errors")
    
    # Print statistics
    if args.verbose:
        print_dataset_statistics(coco_data)
    
    print(f"\n{'='*80}")
    print("✓ STAGE 1 COMPLETED")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
