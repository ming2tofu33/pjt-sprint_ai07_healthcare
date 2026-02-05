"""
COCO format utilities for pill detection project.

Handles:
- COCO JSON loading and saving
- COCO format validation
- Category ID mapping (YOLO index <-> COCO category_id)
- Visualization
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_coco_json(json_path: str) -> Dict[str, Any]:
    """Load COCO format JSON file.
    
    Args:
        json_path: Path to COCO JSON file
        
    Returns:
        COCO format dictionary
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    return coco_data


def save_coco_json(coco_data: Dict[str, Any], save_path: str) -> None:
    """Save COCO format JSON file.
    
    Args:
        coco_data: COCO format dictionary
        save_path: Path to save JSON file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)


def validate_coco_format(coco_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate COCO format data.
    
    Args:
        coco_data: COCO format dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required keys
    required_keys = ['images', 'annotations', 'categories']
    for key in required_keys:
        if key not in coco_data:
            errors.append(f"Missing required key: {key}")
    
    if errors:
        return False, errors
    
    # Validate images
    image_ids = set()
    for img in coco_data['images']:
        if 'id' not in img:
            errors.append(f"Image missing 'id': {img}")
        if 'file_name' not in img:
            errors.append(f"Image missing 'file_name': {img}")
        if 'id' in img:
            if img['id'] in image_ids:
                errors.append(f"Duplicate image_id: {img['id']}")
            image_ids.add(img['id'])
    
    # Validate categories
    category_ids = set()
    for cat in coco_data['categories']:
        if 'id' not in cat:
            errors.append(f"Category missing 'id': {cat}")
        if 'name' not in cat:
            errors.append(f"Category missing 'name': {cat}")
        if 'id' in cat:
            if cat['id'] in category_ids:
                errors.append(f"Duplicate category_id: {cat['id']}")
            category_ids.add(cat['id'])
    
    # Validate annotations
    for idx, ann in enumerate(coco_data['annotations']):
        if 'id' not in ann:
            errors.append(f"Annotation {idx} missing 'id'")
        if 'image_id' not in ann:
            errors.append(f"Annotation {idx} missing 'image_id'")
        if 'category_id' not in ann:
            errors.append(f"Annotation {idx} missing 'category_id'")
        if 'bbox' not in ann:
            errors.append(f"Annotation {idx} missing 'bbox'")
        
        # Validate bbox format
        if 'bbox' in ann:
            bbox = ann['bbox']
            if not isinstance(bbox, list) or len(bbox) != 4:
                errors.append(f"Annotation {idx} has invalid bbox format: {bbox}")
            else:
                x, y, w, h = bbox
                if w <= 0 or h <= 0:
                    errors.append(f"Annotation {idx} has invalid bbox dimensions: w={w}, h={h}")
                if x < 0 or y < 0:
                    errors.append(f"Annotation {idx} has negative bbox coordinates: x={x}, y={y}")
        
        # Check if image_id and category_id exist
        if 'image_id' in ann and ann['image_id'] not in image_ids:
            errors.append(f"Annotation {idx} references non-existent image_id: {ann['image_id']}")
        if 'category_id' in ann and ann['category_id'] not in category_ids:
            errors.append(f"Annotation {idx} references non-existent category_id: {ann['category_id']}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def create_category_mapping(coco_data: Dict[str, Any]) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, str]]:
    """Create mapping between COCO category_id and YOLO class index.
    
    IMPORTANT: YOLO uses 0-indexed classes (0, 1, 2, ..., N-1)
               COCO category_id can be arbitrary (1, 11, 24, 69, ...)
    
    For submission, we need to convert YOLO predictions back to original COCO category_id.
    
    Args:
        coco_data: COCO format dictionary
        
    Returns:
        Tuple of (coco_to_yolo, yolo_to_coco, yolo_to_name)
        - coco_to_yolo: {coco_category_id: yolo_class_index}
        - yolo_to_coco: {yolo_class_index: coco_category_id}
        - yolo_to_name: {yolo_class_index: class_name}
    """
    categories = sorted(coco_data['categories'], key=lambda x: x['id'])
    
    coco_to_yolo = {}
    yolo_to_coco = {}
    yolo_to_name = {}
    
    for yolo_idx, cat in enumerate(categories):
        coco_id = cat['id']
        class_name = cat['name']
        
        coco_to_yolo[coco_id] = yolo_idx
        yolo_to_coco[yolo_idx] = coco_id
        yolo_to_name[yolo_idx] = class_name
    
    return coco_to_yolo, yolo_to_coco, yolo_to_name


def save_category_mapping(
    yolo_to_coco: Dict[int, int],
    yolo_to_name: Dict[int, str],
    save_path: str
) -> None:
    """Save category mapping to JSON file.
    
    Args:
        yolo_to_coco: {yolo_class_index: coco_category_id}
        yolo_to_name: {yolo_class_index: class_name}
        save_path: Path to save JSON file
    """
    mapping = {
        'yolo_to_coco': {str(k): v for k, v in yolo_to_coco.items()},
        'yolo_to_name': {str(k): v for k, v in yolo_to_name.items()},
        'num_classes': len(yolo_to_coco)
    }
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def load_category_mapping(mapping_path: str) -> Tuple[Dict[int, int], Dict[int, str]]:
    """Load category mapping from JSON file.
    
    Args:
        mapping_path: Path to mapping JSON file
        
    Returns:
        Tuple of (yolo_to_coco, yolo_to_name)
    """
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    
    yolo_to_coco = {int(k): v for k, v in mapping['yolo_to_coco'].items()}
    yolo_to_name = {int(k): v for k, v in mapping['yolo_to_name'].items()}
    
    return yolo_to_coco, yolo_to_name


def visualize_coco_sample(
    coco_data: Dict[str, Any],
    image_dir: str,
    sample_idx: int = 0,
    save_path: Optional[str] = None
) -> Optional[Image.Image]:
    """Visualize a sample from COCO dataset.
    
    Args:
        coco_data: COCO format dictionary
        image_dir: Directory containing images
        sample_idx: Index of image to visualize
        save_path: Path to save visualization (optional)
        
    Returns:
        PIL Image if save_path is None, else None
    """
    image_dir = Path(image_dir)
    
    # Get image info
    if sample_idx >= len(coco_data['images']):
        raise ValueError(f"sample_idx {sample_idx} out of range (max: {len(coco_data['images'])-1})")
    
    img_info = coco_data['images'][sample_idx]
    img_path = image_dir / img_info['file_name']
    
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    # Load image
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Get annotations for this image
    image_id = img_info['id']
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
    
    # Create category mapping for labels
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Draw bounding boxes
    for ann in annotations:
        bbox = ann['bbox']  # [x, y, w, h]
        x, y, w, h = bbox
        
        # Draw rectangle
        draw.rectangle(
            [(x, y), (x + w, y + h)],
            outline='red',
            width=3
        )
        
        # Draw label
        category_name = cat_id_to_name.get(ann['category_id'], 'unknown')
        label = f"{category_name}"
        
        # Draw text background
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Get text bounding box
        bbox_text = draw.textbbox((x, y - 20), label, font=font)
        draw.rectangle(bbox_text, fill='red')
        draw.text((x, y - 20), label, fill='white', font=font)
    
    # Save or return
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path)
        return None
    else:
        return img


class COCODataset:
    """COCO dataset wrapper for easy access."""
    
    def __init__(self, coco_json_path: str, image_dir: str):
        """Initialize COCO dataset.
        
        Args:
            coco_json_path: Path to COCO JSON file
            image_dir: Directory containing images
        """
        self.coco_data = load_coco_json(coco_json_path)
        self.image_dir = Path(image_dir)
        
        # Validate format
        is_valid, errors = validate_coco_format(self.coco_data)
        if not is_valid:
            raise ValueError(f"Invalid COCO format: {errors[:5]}")  # Show first 5 errors
        
        # Create mappings
        self.coco_to_yolo, self.yolo_to_coco, self.yolo_to_name = create_category_mapping(self.coco_data)
        
        # Create lookup dictionaries
        self.image_id_to_info = {img['id']: img for img in self.coco_data['images']}
        self.category_id_to_name = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        # Group annotations by image_id
        self.image_id_to_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)
    
    def __len__(self) -> int:
        """Return number of images."""
        return len(self.coco_data['images'])
    
    def get_image_info(self, idx: int) -> Dict[str, Any]:
        """Get image information by index."""
        return self.coco_data['images'][idx]
    
    def get_annotations(self, image_id: int) -> List[Dict[str, Any]]:
        """Get all annotations for an image."""
        return self.image_id_to_annotations.get(image_id, [])
    
    def get_num_classes(self) -> int:
        """Get number of classes."""
        return len(self.coco_data['categories'])
    
    def get_class_names(self) -> List[str]:
        """Get list of class names (ordered by YOLO index)."""
        return [self.yolo_to_name[i] for i in range(self.get_num_classes())]
