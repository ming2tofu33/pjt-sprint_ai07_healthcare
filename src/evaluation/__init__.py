"""Evaluation modules for model assessment."""

from .metrics import (
    calculate_map_at_iou,
    calculate_map_range,
    compute_confusion_matrix,
    evaluate_detections,
)
from .visualizer import (
    plot_confusion_matrix,
    plot_pr_curve,
    visualize_predictions,
)

__all__ = [
    # Metrics
    'calculate_map_at_iou',
    'calculate_map_range',
    'compute_confusion_matrix',
    'evaluate_detections',
    # Visualizer
    'plot_confusion_matrix',
    'plot_pr_curve',
    'visualize_predictions',
]
