"""
Visualization utilities for evaluation results.

Includes:
- Prediction visualization on images
- PR curves
- Confusion matrix heatmaps
- Class distribution plots
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def visualize_predictions(
    image_path: str,
    predictions: List[Dict],
    ground_truths: List[Dict] = None,
    class_names: List[str] = None,
    save_path: Optional[str] = None,
    conf_threshold: float = 0.25
) -> Optional[Image.Image]:
    """Visualize predictions on an image.
    
    Args:
        image_path: Path to image file
        predictions: List of prediction dicts with 'bbox', 'score', 'class_id'
        ground_truths: Optional list of ground truth dicts
        class_names: List of class names
        save_path: Path to save visualization (optional)
        conf_threshold: Confidence threshold for display
        
    Returns:
        PIL Image if save_path is None, else None
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw ground truths (green boxes)
    if ground_truths:
        for gt in ground_truths:
            bbox = gt['bbox']
            x, y, w, h = bbox
            
            # Draw rectangle
            draw.rectangle(
                [(x, y), (x + w, y + h)],
                outline='green',
                width=3
            )
            
            # Draw label
            if class_names and gt['class_id'] < len(class_names):
                label = f"GT: {class_names[gt['class_id']]}"
            else:
                label = f"GT: {gt['class_id']}"
            
            # Draw text background
            bbox_text = draw.textbbox((x, y - 20), label, font=font_small)
            draw.rectangle(bbox_text, fill='green')
            draw.text((x, y - 20), label, fill='white', font=font_small)
    
    # Draw predictions (red boxes)
    for pred in predictions:
        if pred['score'] < conf_threshold:
            continue
        
        bbox = pred['bbox']
        x, y, w, h = bbox
        
        # Draw rectangle
        draw.rectangle(
            [(x, y), (x + w, y + h)],
            outline='red',
            width=3
        )
        
        # Draw label
        if class_names and pred['class_id'] < len(class_names):
            label = f"{class_names[pred['class_id']]} {pred['score']:.2f}"
        else:
            label = f"{pred['class_id']} {pred['score']:.2f}"
        
        # Draw text background
        bbox_text = draw.textbbox((x, y + h + 2), label, font=font_small)
        draw.rectangle(bbox_text, fill='red')
        draw.text((x, y + h + 2), label, fill='white', font=font_small)
    
    # Save or return
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path)
        return None
    else:
        return img


def plot_pr_curve(
    precisions: List[float],
    recalls: List[float],
    class_name: str = None,
    save_path: Optional[str] = None
) -> None:
    """Plot Precision-Recall curve.
    
    Args:
        precisions: List of precision values
        recalls: List of recall values
        class_name: Name of class (for title)
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, linewidth=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    
    if class_name:
        plt.title(f'Precision-Recall Curve - {class_name}', fontsize=14)
    else:
        plt.title('Precision-Recall Curve', fontsize=14)
    
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    # Calculate AP (area under curve)
    ap = np.trapz(precisions, recalls)
    plt.text(0.5, 0.05, f'AP = {ap:.4f}', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: List[str] = None,
    save_path: Optional[str] = None,
    normalize: bool = False
) -> None:
    """Plot confusion matrix as heatmap.
    
    Args:
        conf_matrix: Confusion matrix of shape (num_classes+1, num_classes+1)
        class_names: List of class names
        save_path: Path to save plot
        normalize: Whether to normalize by row (true class)
    """
    num_classes = conf_matrix.shape[0] - 1
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Normalize if requested
    if normalize:
        conf_matrix_display = conf_matrix.astype('float') / (conf_matrix.sum(axis=1, keepdims=True) + 1e-10)
    else:
        conf_matrix_display = conf_matrix
    
    # Create heatmap
    im = ax.imshow(conf_matrix_display, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    if class_names:
        tick_labels = class_names + ['Background']
    else:
        tick_labels = [str(i) for i in range(num_classes)] + ['Background']
    
    ax.set_xticks(np.arange(num_classes + 1))
    ax.set_yticks(np.arange(num_classes + 1))
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.set_yticklabels(tick_labels)
    
    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)
    
    if normalize:
        ax.set_title('Normalized Confusion Matrix', fontsize=14)
    else:
        ax.set_title('Confusion Matrix', fontsize=14)
    
    # Add text annotations
    thresh = conf_matrix_display.max() / 2.
    for i in range(num_classes + 1):
        for j in range(num_classes + 1):
            value = conf_matrix_display[i, j]
            if normalize:
                text = f'{value:.2f}'
            else:
                text = f'{int(conf_matrix[i, j])}'
            
            ax.text(j, i, text,
                   ha="center", va="center",
                   color="white" if value > thresh else "black",
                   fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_map_per_iou(
    map_per_iou: Dict[float, float],
    save_path: Optional[str] = None
) -> None:
    """Plot mAP across different IoU thresholds.
    
    Args:
        map_per_iou: Dictionary mapping IoU threshold to mAP
        save_path: Path to save plot
    """
    iou_thresholds = sorted(map_per_iou.keys())
    map_values = [map_per_iou[iou] for iou in iou_thresholds]
    
    plt.figure(figsize=(10, 6))
    plt.plot(iou_thresholds, map_values, marker='o', linewidth=2, markersize=8)
    plt.xlabel('IoU Threshold', fontsize=12)
    plt.ylabel('mAP', fontsize=12)
    plt.title('mAP across IoU Thresholds', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim([min(iou_thresholds) - 0.05, max(iou_thresholds) + 0.05])
    plt.ylim([0, 1])
    
    # Add value labels
    for iou, map_val in zip(iou_thresholds, map_values):
        plt.text(iou, map_val + 0.02, f'{map_val:.3f}', 
                ha='center', fontsize=10)
    
    # Add average line
    avg_map = np.mean(map_values)
    plt.axhline(y=avg_map, color='r', linestyle='--', linewidth=1.5, 
                label=f'Average mAP = {avg_map:.4f}')
    plt.legend(fontsize=11)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_per_class_ap(
    per_class_ap: Dict[int, float],
    class_names: List[str] = None,
    save_path: Optional[str] = None,
    top_k: int = 20
) -> None:
    """Plot per-class Average Precision.
    
    Args:
        per_class_ap: Dictionary mapping class_id to AP
        class_names: List of class names
        save_path: Path to save plot
        top_k: Number of top classes to show
    """
    # Sort classes by AP
    sorted_classes = sorted(per_class_ap.items(), key=lambda x: x[1], reverse=True)
    
    # Take top K
    if len(sorted_classes) > top_k:
        sorted_classes = sorted_classes[:top_k]
    
    class_ids = [c[0] for c in sorted_classes]
    ap_values = [c[1] for c in sorted_classes]
    
    # Create labels
    if class_names:
        labels = [class_names[cid] if cid < len(class_names) else str(cid) 
                 for cid in class_ids]
    else:
        labels = [str(cid) for cid in class_ids]
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, ap_values, color='steelblue')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()  # Highest AP at top
    ax.set_xlabel('Average Precision', fontsize=12)
    ax.set_ylabel('Class', fontsize=12)
    ax.set_title(f'Top {len(labels)} Classes by AP', fontsize=14)
    ax.set_xlim([0, 1])
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(ap_values):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_evaluation_report(
    results: Dict,
    class_names: List[str] = None,
    output_dir: str = "evaluation_results"
) -> None:
    """Create comprehensive evaluation report with all plots.
    
    Args:
        results: Results dictionary from evaluate_detections()
        class_names: List of class names
        output_dir: Output directory for plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating evaluation report in {output_dir}...")
    
    # Plot confusion matrix
    print("  - Creating confusion matrix plot...")
    plot_confusion_matrix(
        results['confusion_matrix'],
        class_names=class_names,
        save_path=output_dir / "confusion_matrix.png",
        normalize=True
    )
    
    # Plot mAP per IoU
    print("  - Creating mAP per IoU plot...")
    plot_map_per_iou(
        results['mAP_per_iou'],
        save_path=output_dir / "map_per_iou.png"
    )
    
    # Plot per-class AP at IoU=0.5
    print("  - Creating per-class AP plot...")
    plot_per_class_ap(
        results['per_class_ap_0.50'],
        class_names=class_names,
        save_path=output_dir / "per_class_ap.png",
        top_k=20
    )
    
    print(f"\n✓ Evaluation report saved to {output_dir}")
