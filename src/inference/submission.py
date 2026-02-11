"""제출 CSV 생성 및 형식 검증 유틸리티."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

_COLUMNS = [
    "annotation_id",
    "image_id",
    "category_id",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "score",
]


def write_submission(
    rows: list[dict],
    out_path: Path | str,
) -> Path:
    """후처리 결과 rows를 제출 CSV로 저장한다."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_rows = sorted(rows, key=lambda r: (int(r["image_id"]), -float(r["score"])))

    rows_for_csv: list[dict[str, Any]] = []
    for annotation_id, row in enumerate(sorted_rows, start=1):
        rows_for_csv.append(
            {
                "annotation_id": annotation_id,
                "image_id": int(row["image_id"]),
                "category_id": int(row["category_id"]),
                "bbox_x": float(row["bbox_x"]),
                "bbox_y": float(row["bbox_y"]),
                "bbox_w": float(row["bbox_w"]),
                "bbox_h": float(row["bbox_h"]),
                "score": float(row["score"]),
            }
        )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows_for_csv)

    logger.info("제출 CSV 저장 완료 | %s | rows=%d", out_path, len(rows_for_csv))
    return out_path


def validate_submission(
    csv_path: Path | str,
    *,
    max_det_per_image: int = 4,
    valid_category_ids: set[int] | None = None,
) -> dict[str, Any]:
    """제출 CSV를 검증하고 report dict를 반환한다."""
    csv_path = Path(csv_path)
    errors: list[str] = []
    warnings: list[str] = []

    if not csv_path.exists():
        return {
            "valid": False,
            "total_rows": 0,
            "n_images": 0,
            "errors": [f"파일이 존재하지 않습니다: {csv_path}"],
            "warnings": [],
        }

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != _COLUMNS:
            errors.append(f"컬럼 불일치: 기대={_COLUMNS}, 실제={reader.fieldnames}")
        rows = list(reader)

    total_rows = len(rows)
    image_counter: Counter[int] = Counter()
    annotation_ids_seen: set[int] = set()

    for i, row in enumerate(rows, 1):
        line_label = f"row {i}"

        try:
            annotation_id = int(row.get("annotation_id", ""))
        except (ValueError, TypeError):
            errors.append(f"{line_label}: annotation_id가 정수가 아닙니다: {row.get('annotation_id')}")
            continue

        if annotation_id <= 0:
            errors.append(f"{line_label}: annotation_id는 1 이상이어야 합니다: {annotation_id}")
        if annotation_id in annotation_ids_seen:
            errors.append(f"{line_label}: annotation_id 중복: {annotation_id}")
        else:
            annotation_ids_seen.add(annotation_id)

        try:
            image_id = int(row.get("image_id", ""))
        except (ValueError, TypeError):
            errors.append(f"{line_label}: image_id가 정수가 아닙니다: {row.get('image_id')}")
            continue
        image_counter[image_id] += 1

        try:
            cat_id = int(row.get("category_id", ""))
        except (ValueError, TypeError):
            errors.append(f"{line_label}: category_id가 정수가 아닙니다: {row.get('category_id')}")
            continue
        if valid_category_ids is not None and cat_id not in valid_category_ids:
            warnings.append(f"{line_label}: category_id={cat_id} 가 유효 범위를 벗어났습니다")

        try:
            bbox_x = float(row.get("bbox_x", 0))
            bbox_y = float(row.get("bbox_y", 0))
            bbox_w = float(row.get("bbox_w", 0))
            bbox_h = float(row.get("bbox_h", 0))
        except (ValueError, TypeError):
            errors.append(f"{line_label}: bbox 값이 숫자가 아닙니다")
            continue

        if bbox_w <= 0 or bbox_h <= 0:
            warnings.append(f"{line_label}: bbox 너비/높이가 0 이하입니다. w={bbox_w}, h={bbox_h}")
        if bbox_x < 0 or bbox_y < 0:
            warnings.append(f"{line_label}: bbox 좌표가 음수입니다. x={bbox_x}, y={bbox_y}")

        try:
            score = float(row.get("score", 0))
        except (ValueError, TypeError):
            errors.append(f"{line_label}: score가 숫자가 아닙니다")
            continue
        if score < 0 or score > 1:
            warnings.append(f"{line_label}: score={score} 가 [0,1] 범위를 벗어났습니다")

    over_limit = {img: cnt for img, cnt in image_counter.items() if cnt > max_det_per_image}
    if over_limit:
        for img, cnt in sorted(over_limit.items()):
            errors.append(f"image_id={img}: rows={cnt} > max_det_per_image={max_det_per_image}")

    report = {
        "valid": len(errors) == 0,
        "total_rows": total_rows,
        "n_images": len(image_counter),
        "errors": errors,
        "warnings": warnings,
    }

    status = "PASS" if report["valid"] else "FAIL"
    logger.info(
        "제출 검증 %s | rows=%d | images=%d | errors=%d | warnings=%d",
        status,
        report["total_rows"],
        report["n_images"],
        len(errors),
        len(warnings),
    )
    if errors:
        for e in errors[:10]:
            logger.error("  [ERR] %s", e)
        if len(errors) > 10:
            logger.error("  ... 외 %d건", len(errors) - 10)
    if warnings:
        for w in warnings[:10]:
            logger.warning("  [WARN] %s", w)
        if len(warnings) > 10:
            logger.warning("  ... 외 %d건", len(warnings) - 10)

    return report


def write_submission_manifest(
    report: dict,
    *,
    run_dir: Path,
    conf: float,
    n_test_images: int,
    csv_path: Path,
    debug_report: dict | None = None,
    output_path: Path | None = None,
) -> Path:
    """제출 매니페스트(submission_manifest.json)를 저장한다."""
    from datetime import datetime

    manifest = {
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "csv_path": str(csv_path),
        "conf_threshold": conf,
        "n_test_images": n_test_images,
        "total_predictions": report.get("total_rows", 0),
        "n_images_with_predictions": report.get("n_images", 0),
        "validation_passed": report.get("valid", False),
        "n_errors": len(report.get("errors", [])),
        "n_warnings": len(report.get("warnings", [])),
    }
    if debug_report is not None:
        manifest["debug_enabled"] = bool(debug_report.get("debug_enabled", False))
        manifest["debug_output_dir"] = str(debug_report.get("debug_output_dir", ""))
        manifest["debug_sample_requested"] = int(debug_report.get("debug_sample_requested", 0))
        manifest["debug_sample_saved"] = int(debug_report.get("debug_sample_saved", 0))
        if debug_report.get("debug_skipped_reason"):
            manifest["debug_skipped_reason"] = str(debug_report.get("debug_skipped_reason"))

    out = output_path if output_path is not None else (run_dir / "submission_manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info("submission_manifest.json 저장 | %s", out)
    return out
