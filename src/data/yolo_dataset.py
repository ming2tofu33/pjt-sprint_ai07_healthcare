"""
YOLO dataset utilities for converting COCO format to YOLO format.

YOLO format requirements:
- One .txt file per image with same name
- Each line: class_index x_center y_center width height (normalized 0-1)
- data.yaml file with train/val paths and class names
"""

import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from .coco_utils import load_coco_json


def convert_bbox_coco_to_yolo(
    bbox_coco: List[float],
    image_width: int,
    image_height: int
) -> Tuple[float, float, float, float]:
    """Convert COCO bbox format to YOLO format.
    
    COCO format: [x, y, width, height] (absolute coordinates)
    YOLO format: [x_center, y_center, width, height] (normalized 0-1)
    
    Args:
        bbox_coco: COCO bbox [x, y, w, h]
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        YOLO bbox (x_center, y_center, width, height) normalized
    """
    x, y, w, h = bbox_coco
    
    # Convert to center coordinates
    x_center = x + w / 2
    y_center = y + h / 2
    
    # Normalize by image dimensions
    x_center_norm = x_center / image_width
    y_center_norm = y_center / image_height
    w_norm = w / image_width
    h_norm = h / image_height
    
    # Clip to [0, 1] range
    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))
    
    return x_center_norm, y_center_norm, w_norm, h_norm


def create_yolo_dataset(
    coco_json_path: str,
    images_dir: str,
    output_dir: str,
    split_name: str = "train",
    copy_images: bool = False,
    coco_to_yolo_mapping: Dict[int, int] = None
) -> str:
    """Convert COCO dataset to YOLO format.
    
    Creates directory structure:
    output_dir/
    ├── images/
    │   └── train/  (or val/)
    │       ├── image1.png
    │       └── image2.png
    └── labels/
        └── train/  (or val/)
            ├── image1.txt
            └── image2.txt
    
    Args:
        coco_json_path: Path to COCO JSON file
        images_dir: Directory containing images
        output_dir: Output directory for YOLO dataset
        split_name: Split name (train, val, test)
        copy_images: If True, copy images; if False, create symlinks
        coco_to_yolo_mapping: Dict mapping COCO category_id to YOLO class index
        
    Returns:
        Path to created dataset directory
    """
    output_dir = Path(output_dir)
    images_dir = Path(images_dir)
    
    # Create directory structure
    images_out_dir = output_dir / "images" / split_name
    labels_out_dir = output_dir / "labels" / split_name
    
    images_out_dir.mkdir(parents=True, exist_ok=True)
    labels_out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load COCO data
    print(f"\nConverting COCO to YOLO format for {split_name} split...")
    coco_data = load_coco_json(coco_json_path)
    
    # Create category mapping if not provided
    if coco_to_yolo_mapping is None:
        categories = sorted(coco_data['categories'], key=lambda x: x['id'])
        coco_to_yolo_mapping = {cat['id']: idx for idx, cat in enumerate(categories)}
    
    # Create image_id to info mapping
    image_id_to_info = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image_id
    image_id_to_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_id_to_annotations:
            image_id_to_annotations[img_id] = []
        image_id_to_annotations[img_id].append(ann)
    
    # Process each image
    processed_count = 0
    skipped_count = 0
    
    for img_id, img_info in image_id_to_info.items():
        image_filename = img_info['file_name']
        image_path = images_dir / image_filename
        
        # Check if image exists
        if not image_path.exists():
            print(f"  Warning: Image not found: {image_path}")
            skipped_count += 1
            continue
        
        # Get image dimensions
        image_width = img_info['width']
        image_height = img_info['height']
        
        # Copy or symlink image
        dest_image_path = images_out_dir / image_filename
        if copy_images:
            if not dest_image_path.exists():
                shutil.copy2(image_path, dest_image_path)
        else:
            # Create relative symlink
            if not dest_image_path.exists():
                rel_path = Path("../../../") / images_dir.relative_to(Path.cwd()) / image_filename
                dest_image_path.symlink_to(rel_path)
        
        # Create label file
        label_filename = Path(image_filename).stem + ".txt"
        label_path = labels_out_dir / label_filename
        
        # Get annotations for this image
        annotations = image_id_to_annotations.get(img_id, [])
        
        # Write YOLO format labels
        with open(label_path, 'w') as f:
            for ann in annotations:
                # Get YOLO class index
                coco_category_id = ann['category_id']
                if coco_category_id not in coco_to_yolo_mapping:
                    print(f"  Warning: Unknown category_id {coco_category_id} in image {image_filename}")
                    continue
                
                yolo_class_idx = coco_to_yolo_mapping[coco_category_id]
                
                # Convert bbox
                bbox_coco = ann['bbox']
                x_center, y_center, width, height = convert_bbox_coco_to_yolo(
                    bbox_coco, image_width, image_height
                )
                
                # Write YOLO format line
                f.write(f"{yolo_class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        processed_count += 1
    
    print(f"  Processed: {processed_count} images")
    print(f"  Skipped: {skipped_count} images")
    print(f"  Images: {images_out_dir}")
    print(f"  Labels: {labels_out_dir}")
    
    return str(output_dir)


def create_yolo_data_yaml(
    output_dir: str,
    train_path: str,
    val_path: str,
    class_names: List[str],
    num_classes: int
) -> str:
    """Create YOLO data.yaml configuration file.
    
    Args:
        output_dir: Output directory
        train_path: Path to training images (relative or absolute)
        val_path: Path to validation images (relative or absolute)
        class_names: List of class names (ordered by YOLO index)
        num_classes: Number of classes
        
    Returns:
        Path to created data.yaml file
    """
    output_dir = Path(output_dir)
    
    # Create data.yaml content
    data_yaml = {
        'path': str(output_dir.absolute()),  # Dataset root
        'train': train_path,  # Path to train images (relative to path)
        'val': val_path,      # Path to val images (relative to path)
        'nc': num_classes,    # Number of classes
        'names': class_names  # Class names
    }
    
    # Save data.yaml
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ Created data.yaml: {yaml_path}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Train path: {train_path}")
    print(f"  Val path: {val_path}")
    
    return str(yaml_path)


def prepare_yolo_dataset_from_coco(
    coco_train_json: str,
    coco_val_json: str,
    train_images_dir: str,
    val_images_dir: str,
    output_dir: str,
    category_mapping_path: str,
    copy_images: bool = False
) -> str:
    """Prepare complete YOLO dataset from COCO format.
    
    This is a convenience function that:
    1. Converts train and val COCO datasets to YOLO format
    2. Creates data.yaml file
    
    Args:
        coco_train_json: Path to train COCO JSON
        coco_val_json: Path to val COCO JSON
        train_images_dir: Directory containing train images
        val_images_dir: Directory containing val images
        output_dir: Output directory for YOLO dataset
        category_mapping_path: Path to category_mapping.json
        copy_images: If True, copy images; if False, create symlinks
        
    Returns:
        Path to data.yaml file
    """
    import json
    
    print(f"\n{'='*80}")
    print("PREPARING YOLO DATASET FROM COCO")
    print(f"{'='*80}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load category mapping
    with open(category_mapping_path, 'r') as f:
        mapping = json.load(f)
    
    # Create coco_to_yolo mapping
    coco_to_yolo = {}
    for yolo_idx_str, coco_id in mapping['yolo_to_coco'].items():
        yolo_idx = int(yolo_idx_str)
        coco_to_yolo[coco_id] = yolo_idx
    
    # Get class names (ordered by YOLO index)
    num_classes = mapping['num_classes']
    class_names = [mapping['yolo_to_name'][str(i)] for i in range(num_classes)]
    
    # Convert train set
    print(f"\n{'='*80}")
    print("CONVERTING TRAIN SET")
    print(f"{'='*80}")
    create_yolo_dataset(
        coco_json_path=coco_train_json,
        images_dir=train_images_dir,
        output_dir=str(output_dir),
        split_name="train",
        copy_images=copy_images,
        coco_to_yolo_mapping=coco_to_yolo
    )
    
    # Convert val set
    print(f"\n{'='*80}")
    print("CONVERTING VAL SET")
    print(f"{'='*80}")
    create_yolo_dataset(
        coco_json_path=coco_val_json,
        images_dir=val_images_dir,
        output_dir=str(output_dir),
        split_name="val",
        copy_images=copy_images,
        coco_to_yolo_mapping=coco_to_yolo
    )
    
    # Create data.yaml
    print(f"\n{'='*80}")
    print("CREATING DATA.YAML")
    print(f"{'='*80}")
    data_yaml_path = create_yolo_data_yaml(
        output_dir=str(output_dir),
        train_path="images/train",  # Relative to output_dir
        val_path="images/val",      # Relative to output_dir
        class_names=class_names,
        num_classes=num_classes
    )
    
    print(f"\n{'='*80}")
    print("✓ YOLO DATASET PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nDataset structure:")
    print(f"  {output_dir}/")
    print(f"  ├── data.yaml")
    print(f"  ├── images/")
    print(f"  │   ├── train/")
    print(f"  │   └── val/")
    print(f"  └── labels/")
    print(f"      ├── train/")
    print(f"      └── val/")
    
    return data_yaml_path
