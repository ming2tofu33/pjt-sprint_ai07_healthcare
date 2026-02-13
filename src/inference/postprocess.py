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
    min_conf: float = 0.0,
    class_min_conf_by_category: dict[int, float] | None = None,
    keep_category_ids: set[int] | None = None,
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
    filtered_by_min_conf = 0
    filtered_by_category = 0
    class_min_conf_by_category = class_min_conf_by_category or {}

    for det in detections:
        image_stem = det.get("image_stem", "")
        boxes = det.get("boxes", [])

        image_id = _parse_image_id(image_stem)
        sorted_boxes = sorted(boxes, key=lambda b: b["conf"], reverse=True)
        kept_count = 0

        for i, box in enumerate(sorted_boxes):
            class_idx = box["class_idx"]
            xywh = box["xywh"]
            score = float(box["conf"])

            category_id = idx2id_norm.get(class_idx)
            if category_id is None:
                unmapped_classes.add(class_idx)
                category_id = class_idx

            if keep_category_ids is not None and category_id not in keep_category_ids:
                filtered_by_category += 1
                continue

            effective_min_conf = class_min_conf_by_category.get(category_id, min_conf)
            if score < effective_min_conf:
                filtered_by_min_conf += 1
                continue

            rows.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox_x": round(xywh[0], 2),
                    "bbox_y": round(xywh[1], 2),
                    "bbox_w": round(xywh[2], 2),
                    "bbox_h": round(xywh[3], 2),
                    "score": round(score, 6),
                }
            )
            kept_count += 1
            if kept_count >= max_det_per_image:
                total_truncated += max(0, len(sorted_boxes) - (i + 1))
                break

    if total_truncated > 0:
        logger.info(
            "Top-%d rule applied: %d detections truncated",
            max_det_per_image,
            total_truncated,
        )
    if filtered_by_min_conf > 0:
        logger.info("Detections filtered by min_conf: %d", filtered_by_min_conf)
    if filtered_by_category > 0:
        logger.info("Detections filtered by keep_category_ids: %d", filtered_by_category)
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
