"""Data processing modules for YOLO pill detection project."""

from .coco_utils import (
    COCODataset,
    load_coco_json,
    save_coco_json,
    validate_coco_format,
    visualize_coco_sample,
)
from .split_utils import (
    stratified_split_by_object_count,
    kfold_split,
    save_split_info,
    load_split_info,
)
from .yolo_dataset import (
    convert_bbox_coco_to_yolo,
    create_yolo_dataset,
    create_yolo_data_yaml,
    prepare_yolo_dataset_from_coco,
)

__all__ = [
    # COCO utilities
    'COCODataset',
    'load_coco_json',
    'save_coco_json',
    'validate_coco_format',
    'visualize_coco_sample',
    # Split utilities
    'stratified_split_by_object_count',
    'kfold_split',
    'save_split_info',
    'load_split_info',
    # YOLO dataset
    'convert_bbox_coco_to_yolo',
    'create_yolo_dataset',
    'create_yolo_data_yaml',
    'prepare_yolo_dataset_from_coco',
]
