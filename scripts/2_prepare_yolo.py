"""
Stage 2: YOLO Dataset Preparation Script

Converts COCO format dataset to YOLO format and creates data.yaml.

Features:
- COCO to YOLO format conversion
- Split-based dataset creation
- Automatic data.yaml generation
- Image symlinking (fast) or copying (portable)

Usage:
    python scripts/2_prepare_yolo.py \\
        --coco_dir data/coco_data \\
        --images_dir data/raw/train_images \\
        --splits_dir data/splits \\
        --output_dir data/yolo_data
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.coco_utils import load_coco_json
from src.data.split_utils import load_split_info
from src.data.yolo_dataset import prepare_yolo_dataset_from_coco
from src.utils import set_seed


def create_split_coco_json(
    coco_data: dict,
    image_ids: list,
    output_path: str
) -> str:
    """Create a COCO JSON file for a specific split.
    
    Args:
        coco_data: Full COCO dataset
        image_ids: List of image IDs for this split
        output_path: Path to save split COCO JSON
        
    Returns:
        Path to created COCO JSON file
    """
    image_ids_set = set(image_ids)
    
    # Filter images
    split_images = [img for img in coco_data['images'] if img['id'] in image_ids_set]
    
    # Filter annotations
    split_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in image_ids_set]
    
    # Keep all categories
    split_coco = {
        'images': split_images,
        'annotations': split_annotations,
        'categories': coco_data['categories']
    }
    
    # Save split COCO JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(split_coco, f, indent=2)
    
    return str(output_path)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare YOLO dataset from COCO format"
    )
    
    # Input paths
    parser.add_argument(
        '--coco_dir',
        type=str,
        default='data/coco_data',
        help='Directory containing merged_coco.json and category_mapping.json'
    )
    parser.add_argument(
        '--images_dir',
        type=str,
        default='data/raw/train_images',
        help='Directory containing training images'
    )
    parser.add_argument(
        '--splits_dir',
        type=str,
        default='data/splits',
        help='Directory containing split_info.json'
    )
    
    # Output path
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/yolo_data',
        help='Output directory for YOLO dataset (default: data/yolo_data)'
    )
    
    # Options
    parser.add_argument(
        '--copy_images',
        action='store_true',
        help='Copy images instead of creating symlinks (slower but portable)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    print(f"\n{'='*80}")
    print("STAGE 2: YOLO DATASET PREPARATION")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    print(f"  COCO directory: {args.coco_dir}")
    print(f"  Images directory: {args.images_dir}")
    print(f"  Splits directory: {args.splits_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Copy images: {args.copy_images}")
    print(f"  Seed: {args.seed}")
    
    # Convert paths
    coco_dir = Path(args.coco_dir)
    images_dir = Path(args.images_dir)
    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    
    # Validate input paths
    coco_json_path = coco_dir / "merged_coco.json"
    category_mapping_path = coco_dir / "category_mapping.json"
    split_info_path = splits_dir / "split_info.json"
    
    if not coco_json_path.exists():
        raise FileNotFoundError(f"COCO JSON not found: {coco_json_path}")
    if not category_mapping_path.exists():
        raise FileNotFoundError(f"Category mapping not found: {category_mapping_path}")
    if not split_info_path.exists():
        raise FileNotFoundError(f"Split info not found: {split_info_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Load COCO data
    print(f"\nLoading COCO data from {coco_json_path}...")
    coco_data = load_coco_json(str(coco_json_path))
    print(f"  Total images: {len(coco_data['images'])}")
    print(f"  Total annotations: {len(coco_data['annotations'])}")
    print(f"  Total categories: {len(coco_data['categories'])}")
    
    # Load split info
    print(f"\nLoading split info from {split_info_path}...")
    train_ids, val_ids = load_split_info(str(split_info_path))
    print(f"  Train images: {len(train_ids)}")
    print(f"  Val images: {len(val_ids)}")
    
    # Create split COCO JSONs
    print(f"\n{'='*80}")
    print("CREATING SPLIT COCO JSONs")
    print(f"{'='*80}")
    
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    train_coco_path = create_split_coco_json(
        coco_data,
        train_ids,
        str(temp_dir / "train_coco.json")
    )
    print(f"\n✓ Created train COCO JSON: {train_coco_path}")
    
    val_coco_path = create_split_coco_json(
        coco_data,
        val_ids,
        str(temp_dir / "val_coco.json")
    )
    print(f"✓ Created val COCO JSON: {val_coco_path}")
    
    # Prepare YOLO dataset
    data_yaml_path = prepare_yolo_dataset_from_coco(
        coco_train_json=train_coco_path,
        coco_val_json=val_coco_path,
        train_images_dir=str(images_dir),
        val_images_dir=str(images_dir),  # Same images dir for both splits
        output_dir=str(output_dir),
        category_mapping_path=str(category_mapping_path),
        copy_images=args.copy_images
    )
    
    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_dir)
    print(f"\n✓ Cleaned up temporary files")
    
    print(f"\n{'='*80}")
    print("✓ STAGE 2 COMPLETED")
    print(f"{'='*80}")
    print(f"\nYOLO dataset ready at: {output_dir}")
    print(f"Data YAML: {data_yaml_path}")
    print(f"\nNext step:")
    print(f"  python scripts/3_train.py --config configs/experiments/exp001_baseline.yaml")


if __name__ == "__main__":
    main()
