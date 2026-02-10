"""Inference postprocess utilities.

Responsibilities:
- Top-K selection per image
- YOLO class index -> original category_id mapping
- xywh (absolute) conversion pass-through for submission rows

Note:
- annotation_id is intentionally NOT created here.
- annotation_id is assigned in write_submission() as global 1..N sequence.
"""
from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)


def postprocess_detections(
    detections: list[dict],
    idx2id: dict[int, int] | dict[str, int],
    *,
    max_det_per_image: int = 4,
) -> list[dict]:
    """Convert batch detection results to submission row dicts.

    Returned row schema:
    {
        "image_id": int,
        "category_id": int,
        "bbox_x": float,
        "bbox_y": float,
        "bbox_w": float,
        "bbox_h": float,
        "score": float,
    }
    """
    idx2id_norm: dict[int, int] = {}
    for key, value in idx2id.items():
        idx2id_norm[int(key)] = int(value)

    rows: list[dict] = []
    total_truncated = 0
    unmapped_classes: set[int] = set()

    for det in detections:
        image_stem = det.get("image_stem", "")
        boxes = det.get("boxes", [])

        image_id = _parse_image_id(image_stem)
        sorted_boxes = sorted(boxes, key=lambda b: b["conf"], reverse=True)

        if len(sorted_boxes) > max_det_per_image:
            total_truncated += len(sorted_boxes) - max_det_per_image
            sorted_boxes = sorted_boxes[:max_det_per_image]

        for box in sorted_boxes:
            class_idx = box["class_idx"]
            xywh = box["xywh"]

            category_id = idx2id_norm.get(class_idx)
            if category_id is None:
                unmapped_classes.add(class_idx)
                category_id = class_idx

            rows.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox_x": round(xywh[0], 2),
                    "bbox_y": round(xywh[1], 2),
                    "bbox_w": round(xywh[2], 2),
                    "bbox_h": round(xywh[3], 2),
                    "score": round(box["conf"], 6),
                }
            )

    if total_truncated > 0:
        logger.info(
            "Top-%d rule applied: %d detections truncated",
            max_det_per_image,
            total_truncated,
        )
    if unmapped_classes:
        logger.warning(
            "Unmapped class indices found: %s (fallback: category_id=class_idx)",
            sorted(unmapped_classes),
        )

    logger.info("Postprocess completed | images=%d | rows=%d", len(detections), len(rows))
    return rows


def _parse_image_id(stem: str) -> int:
    """Parse numeric image_id from filename stem, fallback to hash."""
    try:
        return int(stem)
    except ValueError:
        logger.warning("image_stem '%s' is not numeric. Using hash fallback.", stem)
        return abs(hash(stem)) % (10**9)
