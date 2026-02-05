"""
Data splitting utilities for train/val split and K-Fold cross-validation.

Supports:
- Stratified split by object count
- Stratified split by class distribution
- K-Fold cross-validation with stratification
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold


def calculate_image_object_counts(coco_data: Dict) -> Dict[int, int]:
    """Calculate number of objects per image.
    
    Args:
        coco_data: COCO format dictionary
        
    Returns:
        Dictionary mapping image_id to object count
    """
    image_object_counts = defaultdict(int)
    
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        image_object_counts[image_id] += 1
    
    return dict(image_object_counts)


def calculate_image_class_distribution(coco_data: Dict) -> Dict[int, List[int]]:
    """Calculate class distribution per image.
    
    Args:
        coco_data: COCO format dictionary
        
    Returns:
        Dictionary mapping image_id to list of category_ids
    """
    image_classes = defaultdict(set)
    
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        image_classes[image_id].add(category_id)
    
    # Convert sets to sorted lists
    return {img_id: sorted(list(classes)) for img_id, classes in image_classes.items()}


def stratified_split_by_object_count(
    coco_data: Dict,
    train_ratio: float = 0.8,
    seed: int = 42,
    min_objects_per_split: int = 1
) -> Tuple[List[int], List[int]]:
    """Split images into train/val by stratifying on object count.
    
    This ensures both splits have similar distribution of images with
    1, 2, 3, or 4 objects.
    
    Args:
        coco_data: COCO format dictionary
        train_ratio: Ratio of training images (0-1)
        seed: Random seed for reproducibility
        min_objects_per_split: Minimum objects per image to include
        
    Returns:
        Tuple of (train_image_ids, val_image_ids)
    """
    np.random.seed(seed)
    
    # Calculate object counts
    image_object_counts = calculate_image_object_counts(coco_data)
    
    # Filter images by minimum object count
    valid_image_ids = [
        img_id for img_id, count in image_object_counts.items()
        if count >= min_objects_per_split
    ]
    
    # Create stratification groups (1, 2, 3, 4+ objects)
    stratify_labels = []
    for img_id in valid_image_ids:
        count = image_object_counts[img_id]
        # Cap at 4 to create groups: 1, 2, 3, 4+
        label = min(count, 4)
        stratify_labels.append(label)
    
    # Convert to numpy arrays
    image_ids_array = np.array(valid_image_ids)
    stratify_labels_array = np.array(stratify_labels)
    
    # Perform stratified split
    from sklearn.model_selection import train_test_split
    
    train_ids, val_ids = train_test_split(
        image_ids_array,
        train_size=train_ratio,
        stratify=stratify_labels_array,
        random_state=seed
    )
    
    return train_ids.tolist(), val_ids.tolist()


def stratified_split_by_class(
    coco_data: Dict,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """Split images into train/val by stratifying on dominant class.
    
    For multi-label images, uses the most frequent class as stratification label.
    
    Args:
        coco_data: COCO format dictionary
        train_ratio: Ratio of training images (0-1)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_image_ids, val_image_ids)
    """
    np.random.seed(seed)
    
    # Get class distribution per image
    image_classes = calculate_image_class_distribution(coco_data)
    
    # For each image, use the first class as stratification label
    # (For single-class images, this is the only class)
    image_ids = []
    stratify_labels = []
    
    for img_id, classes in image_classes.items():
        if classes:  # Has at least one class
            image_ids.append(img_id)
            stratify_labels.append(classes[0])  # Use first class
    
    # Convert to numpy arrays
    image_ids_array = np.array(image_ids)
    stratify_labels_array = np.array(stratify_labels)
    
    # Perform stratified split
    from sklearn.model_selection import train_test_split
    
    train_ids, val_ids = train_test_split(
        image_ids_array,
        train_size=train_ratio,
        stratify=stratify_labels_array,
        random_state=seed
    )
    
    return train_ids.tolist(), val_ids.tolist()


def kfold_split(
    coco_data: Dict,
    n_folds: int = 5,
    seed: int = 42,
    stratify_by: str = "object_count"
) -> List[Tuple[List[int], List[int]]]:
    """Create K-Fold splits with stratification.
    
    Args:
        coco_data: COCO format dictionary
        n_folds: Number of folds
        seed: Random seed for reproducibility
        stratify_by: Stratification method ('object_count' or 'class')
        
    Returns:
        List of (train_image_ids, val_image_ids) tuples for each fold
    """
    np.random.seed(seed)
    
    # Get stratification labels
    if stratify_by == "object_count":
        image_object_counts = calculate_image_object_counts(coco_data)
        image_ids = list(image_object_counts.keys())
        stratify_labels = [min(image_object_counts[img_id], 4) for img_id in image_ids]
    elif stratify_by == "class":
        image_classes = calculate_image_class_distribution(coco_data)
        image_ids = list(image_classes.keys())
        stratify_labels = [classes[0] if classes else 0 for classes in image_classes.values()]
    else:
        raise ValueError(f"Unknown stratify_by: {stratify_by}")
    
    # Convert to numpy arrays
    image_ids_array = np.array(image_ids)
    stratify_labels_array = np.array(stratify_labels)
    
    # Create K-Fold splits
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    folds = []
    for train_idx, val_idx in skf.split(image_ids_array, stratify_labels_array):
        train_ids = image_ids_array[train_idx].tolist()
        val_ids = image_ids_array[val_idx].tolist()
        folds.append((train_ids, val_ids))
    
    return folds


def save_split_info(
    train_ids: List[int],
    val_ids: List[int],
    save_path: str,
    metadata: Dict = None
) -> None:
    """Save train/val split information to JSON file.
    
    Args:
        train_ids: List of training image IDs
        val_ids: List of validation image IDs
        save_path: Path to save JSON file
        metadata: Additional metadata to save (optional)
    """
    split_info = {
        'train_ids': train_ids,
        'val_ids': val_ids,
        'num_train': len(train_ids),
        'num_val': len(val_ids),
        'train_ratio': len(train_ids) / (len(train_ids) + len(val_ids)),
    }
    
    if metadata:
        split_info['metadata'] = metadata
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2)


def load_split_info(split_path: str) -> Tuple[List[int], List[int]]:
    """Load train/val split information from JSON file.
    
    Args:
        split_path: Path to split JSON file
        
    Returns:
        Tuple of (train_ids, val_ids)
    """
    with open(split_path, 'r', encoding='utf-8') as f:
        split_info = json.load(f)
    
    return split_info['train_ids'], split_info['val_ids']


def print_split_statistics(
    coco_data: Dict,
    train_ids: List[int],
    val_ids: List[int]
) -> None:
    """Print statistics about train/val split.
    
    Args:
        coco_data: COCO format dictionary
        train_ids: List of training image IDs
        val_ids: List of validation image IDs
    """
    # Calculate object counts
    image_object_counts = calculate_image_object_counts(coco_data)
    
    # Calculate class distribution
    image_classes = calculate_image_class_distribution(coco_data)
    
    # Print split sizes
    print(f"\n{'='*80}")
    print("SPLIT STATISTICS")
    print(f"{'='*80}")
    print(f"\nTotal images: {len(train_ids) + len(val_ids)}")
    print(f"Train images: {len(train_ids)} ({len(train_ids)/(len(train_ids)+len(val_ids))*100:.1f}%)")
    print(f"Val images:   {len(val_ids)} ({len(val_ids)/(len(train_ids)+len(val_ids))*100:.1f}%)")
    
    # Object count distribution
    print(f"\n{'='*80}")
    print("OBJECT COUNT DISTRIBUTION")
    print(f"{'='*80}")
    
    train_counts = [image_object_counts.get(img_id, 0) for img_id in train_ids]
    val_counts = [image_object_counts.get(img_id, 0) for img_id in val_ids]
    
    train_count_dist = Counter(train_counts)
    val_count_dist = Counter(val_counts)
    
    print(f"\n{'Objects':<10} {'Train':<15} {'Val':<15}")
    print(f"{'-'*40}")
    for count in sorted(set(train_counts + val_counts)):
        train_num = train_count_dist.get(count, 0)
        val_num = val_count_dist.get(count, 0)
        train_pct = train_num / len(train_ids) * 100 if train_ids else 0
        val_pct = val_num / len(val_ids) * 100 if val_ids else 0
        print(f"{count:<10} {train_num:>5} ({train_pct:>5.1f}%)  {val_num:>5} ({val_pct:>5.1f}%)")
    
    # Class distribution
    train_class_counts = Counter()
    val_class_counts = Counter()
    
    for ann in coco_data['annotations']:
        if ann['image_id'] in train_ids:
            train_class_counts[ann['category_id']] += 1
        elif ann['image_id'] in val_ids:
            val_class_counts[ann['category_id']] += 1
    
    print(f"\n{'='*80}")
    print("CLASS DISTRIBUTION (Top 10)")
    print(f"{'='*80}")
    
    all_class_ids = set(train_class_counts.keys()) | set(val_class_counts.keys())
    class_stats = []
    
    for class_id in all_class_ids:
        train_num = train_class_counts.get(class_id, 0)
        val_num = val_class_counts.get(class_id, 0)
        total = train_num + val_num
        class_stats.append((class_id, total, train_num, val_num))
    
    # Sort by total count
    class_stats.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Class ID':<12} {'Total':<10} {'Train':<15} {'Val':<15}")
    print(f"{'-'*52}")
    for class_id, total, train_num, val_num in class_stats[:10]:
        train_pct = train_num / total * 100 if total > 0 else 0
        val_pct = val_num / total * 100 if total > 0 else 0
        print(f"{class_id:<12} {total:<10} {train_num:>5} ({train_pct:>5.1f}%)  {val_num:>5} ({val_pct:>5.1f}%)")
    
    if len(class_stats) > 10:
        print(f"\n... and {len(class_stats) - 10} more classes")
