"""src.inference.submission — 제출 CSV 생성 및 규칙 검증.

후처리된 detection 행을 CSV 파일로 저장하고,
대회 제출 규칙을 검증한다.

사용 예시::

    from src.inference.submission import write_submission, validate_submission
    csv_path = write_submission(rows, out_path)
    report = validate_submission(csv_path, max_det_per_image=4)
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 제출 CSV 컬럼 순서
_COLUMNS = [
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
    """후처리된 행 리스트를 제출 CSV 로 저장한다.

    Parameters
    ----------
    rows : list[dict]
        ``postprocess.postprocess_detections()`` 의 출력.
    out_path : Path | str
        저장할 CSV 경로.

    Returns
    -------
    Path
        저장된 CSV 경로.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # image_id 오름차순 → score 내림차순 정렬
    sorted_rows = sorted(rows, key=lambda r: (r["image_id"], -r["score"]))

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(sorted_rows)

    logger.info("제출 CSV 저장 완료 | %s | 행=%d", out_path, len(sorted_rows))
    return out_path


def validate_submission(
    csv_path: Path | str,
    *,
    max_det_per_image: int = 4,
    valid_category_ids: set[int] | None = None,
) -> dict[str, Any]:
    """제출 CSV 를 검증하고 결과 report dict 를 반환한다.

    검증 항목:
    1. 컬럼 헤더 일치
    2. 이미지당 행 수 ≤ ``max_det_per_image``
    3. ``category_id`` 가 ``valid_category_ids`` 범위 내 (제공 시)
    4. ``bbox_x, bbox_y, bbox_w, bbox_h`` 가 양수
    5. ``score`` 가 [0, 1] 범위
    6. ``image_id`` 가 정수

    Parameters
    ----------
    csv_path : Path | str
        검증할 CSV 경로.
    max_det_per_image : int
        이미지당 허용 최대 행 수.
    valid_category_ids : set[int], optional
        유효한 category_id 집합.

    Returns
    -------
    dict
        ``{valid: bool, total_rows: int, n_images: int, errors: list[str], warnings: list[str]}``
    """
    csv_path = Path(csv_path)
    errors: list[str] = []
    warnings: list[str] = []

    if not csv_path.exists():
        return {"valid": False, "total_rows": 0, "n_images": 0,
                "errors": [f"파일이 존재하지 않습니다: {csv_path}"], "warnings": []}

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # 1) 헤더 검증
        if reader.fieldnames != _COLUMNS:
            errors.append(
                f"컬럼 불일치: 기대={_COLUMNS}, 실제={reader.fieldnames}"
            )

        rows = list(reader)

    total_rows = len(rows)
    image_counter: Counter = Counter()

    for i, row in enumerate(rows, 1):
        line_label = f"행 {i}"

        # image_id 정수 검증
        try:
            image_id = int(row.get("image_id", ""))
        except (ValueError, TypeError):
            errors.append(f"{line_label}: image_id 가 정수가 아닙니다: {row.get('image_id')}")
            continue

        image_counter[image_id] += 1

        # category_id 검증
        try:
            cat_id = int(row.get("category_id", ""))
        except (ValueError, TypeError):
            errors.append(f"{line_label}: category_id 가 정수가 아닙니다: {row.get('category_id')}")
            continue

        if valid_category_ids is not None and cat_id not in valid_category_ids:
            warnings.append(f"{line_label}: category_id={cat_id} 가 유효 범위 밖")

        # bbox 검증
        try:
            bbox_x = float(row.get("bbox_x", 0))
            bbox_y = float(row.get("bbox_y", 0))
            bbox_w = float(row.get("bbox_w", 0))
            bbox_h = float(row.get("bbox_h", 0))
        except (ValueError, TypeError):
            errors.append(f"{line_label}: bbox 값이 숫자가 아닙니다")
            continue

        if bbox_w <= 0 or bbox_h <= 0:
            warnings.append(f"{line_label}: bbox 폭/높이가 0 이하: w={bbox_w}, h={bbox_h}")
        if bbox_x < 0 or bbox_y < 0:
            warnings.append(f"{line_label}: bbox 좌표가 음수: x={bbox_x}, y={bbox_y}")

        # score 검증
        try:
            score = float(row.get("score", 0))
        except (ValueError, TypeError):
            errors.append(f"{line_label}: score 가 숫자가 아닙니다")
            continue

        if score < 0 or score > 1:
            warnings.append(f"{line_label}: score={score} 가 [0,1] 범위 밖")

    # 2) 이미지당 행 수 검증
    over_limit = {img: cnt for img, cnt in image_counter.items()
                  if cnt > max_det_per_image}
    if over_limit:
        for img, cnt in sorted(over_limit.items()):
            errors.append(
                f"image_id={img}: 행 수={cnt} > max_det_per_image={max_det_per_image}"
            )

    n_images = len(image_counter)
    is_valid = len(errors) == 0

    report = {
        "valid": is_valid,
        "total_rows": total_rows,
        "n_images": n_images,
        "errors": errors,
        "warnings": warnings,
    }

    # 로그 출력
    status = "PASS" if is_valid else "FAIL"
    logger.info("제출 검증 %s | rows=%d | images=%d | errors=%d | warnings=%d",
                status, total_rows, n_images, len(errors), len(warnings))

    if errors:
        for e in errors[:10]:
            logger.error("  [ERR] %s", e)
        if len(errors) > 10:
            logger.error("  ... 외 %d 건", len(errors) - 10)

    if warnings:
        for w in warnings[:10]:
            logger.warning("  [WARN] %s", w)
        if len(warnings) > 10:
            logger.warning("  ... 외 %d 건", len(warnings) - 10)

    return report


def write_submission_manifest(
    report: dict,
    *,
    run_dir: Path,
    conf: float,
    n_test_images: int,
    csv_path: Path,
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

    out = run_dir / "submission_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info("submission_manifest.json 저장 | %s", out)
    return out
