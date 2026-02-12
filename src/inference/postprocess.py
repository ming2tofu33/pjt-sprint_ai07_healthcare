"""Inference postprocess utilities.

Responsibilities:
- Top-K selection per image
- YOLO class index -> original category_id mapping
- Shared submission filter logic for single/ensemble pipelines
"""
from __future__ import annotations

from collections import defaultdict

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
    """Convert model detections to row dicts and apply shared submission filters."""
    idx2id_norm: dict[int, int] = {int(k): int(v) for k, v in idx2id.items()}

    rows: list[dict] = []
    unmapped_classes: set[int] = set()
    for det in detections:
        image_stem = str(det.get("image_stem", ""))
        image_id = parse_image_id(image_stem)

        for box in det.get("boxes", []):
            class_idx = int(box["class_idx"])
            xywh = box["xywh"]
            score = float(box["conf"])

            category_id = idx2id_norm.get(class_idx)
            if category_id is None:
                unmapped_classes.add(class_idx)
                category_id = class_idx

            rows.append(
                {
                    "image_id": image_id,
                    "category_id": int(category_id),
                    "bbox_x": round(float(xywh[0]), 2),
                    "bbox_y": round(float(xywh[1]), 2),
                    "bbox_w": round(float(xywh[2]), 2),
                    "bbox_h": round(float(xywh[3]), 2),
                    "score": round(score, 6),
                }
            )

    if unmapped_classes:
        logger.warning(
            "Unmapped class indices found: %s (fallback: category_id=class_idx)",
            sorted(unmapped_classes),
        )

    filtered_rows = apply_submission_filters_and_topk(
        rows,
        max_det_per_image=max_det_per_image,
        min_conf=min_conf,
        class_min_conf_by_category=class_min_conf_by_category,
        keep_category_ids=keep_category_ids,
    )
    logger.info("Postprocess completed | images=%d | rows=%d", len(detections), len(filtered_rows))
    return filtered_rows


def apply_submission_filters_and_topk(
    rows: list[dict],
    *,
    max_det_per_image: int = 4,
    min_conf: float = 0.0,
    class_min_conf_by_category: dict[int, float] | None = None,
    keep_category_ids: set[int] | None = None,
) -> list[dict]:
    """Apply category/conf filters and keep top-K boxes per image."""
    class_min_conf_by_category = class_min_conf_by_category or {}

    filtered: list[dict] = []
    filtered_by_min_conf = 0
    filtered_by_category = 0

    for row in rows:
        category_id = int(row["category_id"])
        score = float(row["score"])

        if keep_category_ids is not None and category_id not in keep_category_ids:
            filtered_by_category += 1
            continue

        effective_min_conf = float(class_min_conf_by_category.get(category_id, min_conf))
        if score < effective_min_conf:
            filtered_by_min_conf += 1
            continue

        normalized = dict(row)
        normalized["image_id"] = int(row["image_id"])
        normalized["category_id"] = category_id
        normalized["score"] = round(score, 6)
        normalized["bbox_x"] = round(float(row["bbox_x"]), 2)
        normalized["bbox_y"] = round(float(row["bbox_y"]), 2)
        normalized["bbox_w"] = round(float(row["bbox_w"]), 2)
        normalized["bbox_h"] = round(float(row["bbox_h"]), 2)
        filtered.append(normalized)

    by_image: dict[int, list[dict]] = defaultdict(list)
    for row in filtered:
        by_image[int(row["image_id"])].append(row)

    output: list[dict] = []
    total_truncated = 0
    for image_id, image_rows in by_image.items():
        sorted_rows = sorted(image_rows, key=lambda r: float(r["score"]), reverse=True)
        kept = sorted_rows[:max_det_per_image]
        total_truncated += max(0, len(sorted_rows) - len(kept))
        output.extend(kept)

    if total_truncated > 0:
        logger.info("Top-%d rule applied: %d detections truncated", max_det_per_image, total_truncated)
    if filtered_by_min_conf > 0:
        logger.info("Detections filtered by min_conf: %d", filtered_by_min_conf)
    if filtered_by_category > 0:
        logger.info("Detections filtered by keep_category_ids: %d", filtered_by_category)

    return output


def parse_image_id(stem: str) -> int:
    """Parse numeric image_id from filename stem, fallback to hash."""
    try:
        return int(stem)
    except ValueError:
        logger.warning("image_stem '%s' is not numeric. Using hash fallback.", stem)
        return abs(hash(stem)) % (10**9)
