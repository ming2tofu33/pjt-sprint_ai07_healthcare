"""
Evaluation metrics for object detection.

Implements:
- mAP (mean Average Precision) at different IoU thresholds
- mAP@[0.75:0.95] (primary metric for Kaggle competition)
- Per-class AP
- Precision, Recall, F1
- Confusion matrix
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two boxes.
    
    Args:
        box1: [x, y, w, h] format
        box2: [x, y, w, h] format
        
    Returns:
        IoU value (0-1)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to (x1, y1, x2, y2) format
    box1_x2 = x1 + w1
    box1_y2 = y1 + h1
    box2_x2 = x2 + w2
    box2_y2 = y2 + h2
    
    # Calculate intersection
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    # Calculate union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    # Calculate IoU
    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    return iou


def calculate_ap_at_iou(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    class_id: int = None
) -> float:
    """Calculate Average Precision (AP) at a specific IoU threshold.
    
    Args:
        predictions: List of prediction dicts with 'bbox', 'score', 'class_id', 'image_id'
        ground_truths: List of ground truth dicts with 'bbox', 'class_id', 'image_id'
        iou_threshold: IoU threshold for matching
        class_id: Class ID to evaluate (None = all classes)
        
    Returns:
        AP value
    """
    # Filter by class if specified
    if class_id is not None:
        predictions = [p for p in predictions if p['class_id'] == class_id]
        ground_truths = [gt for gt in ground_truths if gt['class_id'] == class_id]
    
    if len(ground_truths) == 0:
        return 0.0
    
    if len(predictions) == 0:
        return 0.0
    
    # Sort predictions by score (descending)
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
    
    # Group ground truths by image_id
    gt_by_image = defaultdict(list)
    for gt in ground_truths:
        gt_by_image[gt['image_id']].append(gt)
    
    # Track which ground truths have been matched
    gt_matched = {i: False for i in range(len(ground_truths))}
    gt_id_map = {id(gt): i for i, gt in enumerate(ground_truths)}
    
    # Calculate TP and FP for each prediction
    tp = []
    fp = []
    
    for pred in predictions:
        image_id = pred['image_id']
        pred_bbox = pred['bbox']
        pred_class = pred['class_id']
        
        # Get ground truths for this image
        image_gts = gt_by_image.get(image_id, [])
        
        # Find best matching ground truth
        best_iou = 0
        best_gt_idx = -1
        
        for gt in image_gts:
            if gt['class_id'] != pred_class:
                continue
            
            iou = calculate_iou(pred_bbox, gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_id_map[id(gt)]
        
        # Check if match is valid
        if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
            tp.append(1)
            fp.append(0)
            gt_matched[best_gt_idx] = True
        else:
            tp.append(0)
            fp.append(1)
    
    # Calculate precision and recall at each threshold
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / len(ground_truths)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # Calculate AP using 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    
    return ap


def calculate_map_at_iou(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: int = None
) -> Tuple[float, Dict[int, float]]:
    """Calculate mean Average Precision (mAP) at a specific IoU threshold.
    
    Args:
        predictions: List of predictions
        ground_truths: List of ground truths
        iou_threshold: IoU threshold
        num_classes: Number of classes (if None, infer from data)
        
    Returns:
        Tuple of (mAP, per_class_ap_dict)
    """
    # Get all unique class IDs
    if num_classes is None:
        all_classes = set()
        for gt in ground_truths:
            all_classes.add(gt['class_id'])
        class_ids = sorted(list(all_classes))
    else:
        class_ids = list(range(num_classes))
    
    # Calculate AP for each class
    per_class_ap = {}
    for class_id in class_ids:
        ap = calculate_ap_at_iou(
            predictions,
            ground_truths,
            iou_threshold=iou_threshold,
            class_id=class_id
        )
        per_class_ap[class_id] = ap
    
    # Calculate mAP
    if len(per_class_ap) > 0:
        map_value = np.mean(list(per_class_ap.values()))
    else:
        map_value = 0.0
    
    return map_value, per_class_ap


def calculate_map_range(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_thresholds: List[float] = [0.75, 0.80, 0.85, 0.90, 0.95],
    num_classes: int = None
) -> Tuple[float, Dict[float, float], Dict[float, Dict[int, float]]]:
    """Calculate mAP over a range of IoU thresholds.
    
    This is the primary metric for the Kaggle competition: mAP@[0.75:0.95]
    
    Args:
        predictions: List of predictions
        ground_truths: List of ground truths
        iou_thresholds: List of IoU thresholds to evaluate
        num_classes: Number of classes
        
    Returns:
        Tuple of (average_mAP, map_per_iou, per_class_ap_per_iou)
    """
    map_per_iou = {}
    per_class_ap_per_iou = {}
    
    for iou_threshold in iou_thresholds:
        map_value, per_class_ap = calculate_map_at_iou(
            predictions,
            ground_truths,
            iou_threshold=iou_threshold,
            num_classes=num_classes
        )
        map_per_iou[iou_threshold] = map_value
        per_class_ap_per_iou[iou_threshold] = per_class_ap
    
    # Calculate average mAP
    average_map = np.mean(list(map_per_iou.values()))
    
    return average_map, map_per_iou, per_class_ap_per_iou


def compute_confusion_matrix(
    predictions: List[Dict],
    ground_truths: List[Dict],
    num_classes: int,
    iou_threshold: float = 0.5
) -> np.ndarray:
    """Compute confusion matrix for object detection.
    
    Args:
        predictions: List of predictions
        ground_truths: List of ground truths
        num_classes: Number of classes
        iou_threshold: IoU threshold for matching
        
    Returns:
        Confusion matrix of shape (num_classes+1, num_classes+1)
        Last row/column is for background (unmatched predictions/gts)
    """
    # Initialize confusion matrix (include background class)
    conf_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)
    
    # Group by image
    gt_by_image = defaultdict(list)
    pred_by_image = defaultdict(list)
    
    for gt in ground_truths:
        gt_by_image[gt['image_id']].append(gt)
    
    for pred in predictions:
        pred_by_image[pred['image_id']].append(pred)
    
    # Get all image IDs
    all_image_ids = set(list(gt_by_image.keys()) + list(pred_by_image.keys()))
    
    # Process each image
    for image_id in all_image_ids:
        image_gts = gt_by_image.get(image_id, [])
        image_preds = pred_by_image.get(image_id, [])
        
        # Track matched ground truths
        gt_matched = [False] * len(image_gts)
        
        # Match predictions to ground truths
        for pred in image_preds:
            pred_bbox = pred['bbox']
            pred_class = pred['class_id']
            
            # Find best matching ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(image_gts):
                iou = calculate_iou(pred_bbox, gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Update confusion matrix
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                gt_class = image_gts[best_gt_idx]['class_id']
                conf_matrix[gt_class, pred_class] += 1
                gt_matched[best_gt_idx] = True
            else:
                # False positive (predicted background as class)
                conf_matrix[num_classes, pred_class] += 1
        
        # Count unmatched ground truths as false negatives
        for gt_idx, gt in enumerate(image_gts):
            if not gt_matched[gt_idx]:
                gt_class = gt['class_id']
                # False negative (missed detection, predicted as background)
                conf_matrix[gt_class, num_classes] += 1
    
    return conf_matrix


def evaluate_detections(
    predictions: List[Dict],
    ground_truths: List[Dict],
    num_classes: int,
    iou_thresholds: List[float] = [0.75, 0.80, 0.85, 0.90, 0.95]
) -> Dict:
    """Comprehensive evaluation of object detection predictions.
    
    Args:
        predictions: List of predictions
        ground_truths: List of ground truths
        num_classes: Number of classes
        iou_thresholds: IoU thresholds for mAP calculation
        
    Returns:
        Dictionary containing all metrics
    """
    # Calculate mAP over range
    average_map, map_per_iou, per_class_ap_per_iou = calculate_map_range(
        predictions,
        ground_truths,
        iou_thresholds=iou_thresholds,
        num_classes=num_classes
    )
    
    # Calculate additional metrics at IoU=0.5 and 0.75
    map_50, per_class_ap_50 = calculate_map_at_iou(
        predictions, ground_truths, iou_threshold=0.5, num_classes=num_classes
    )
    map_75, per_class_ap_75 = calculate_map_at_iou(
        predictions, ground_truths, iou_threshold=0.75, num_classes=num_classes
    )
    
    # Compute confusion matrix
    conf_matrix = compute_confusion_matrix(
        predictions, ground_truths, num_classes=num_classes, iou_threshold=0.5
    )
    
    # Compile results
    results = {
        # Primary metric (Kaggle competition)
        'mAP_0.75-0.95': average_map,
        
        # Secondary metrics (reference)
        'mAP_0.50': map_50,
        'mAP_0.75': map_75,
        
        # Per-IoU mAP
        'mAP_per_iou': map_per_iou,
        
        # Per-class AP at different thresholds
        'per_class_ap_0.50': per_class_ap_50,
        'per_class_ap_0.75': per_class_ap_75,
        'per_class_ap_per_iou': per_class_ap_per_iou,
        
        # Confusion matrix
        'confusion_matrix': conf_matrix,
        
        # Counts
        'num_predictions': len(predictions),
        'num_ground_truths': len(ground_truths),
        'num_classes': num_classes,
    }
    
    return results
