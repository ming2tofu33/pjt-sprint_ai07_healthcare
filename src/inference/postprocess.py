"""src.inference.postprocess — 후처리: Top-K 선택, bbox 변환, class→category_id 매핑.

추론 결과를 제출 형식에 맞게 변환한다.

핵심 규칙:
- 이미지당 최대 Top-K(기본 4) detection 만 유지 (대회 규칙)
- YOLO class index → 원본 COCO category_id 변환 (``idx2id`` 사용)
- bbox: xyxy 절대 좌표 → xywh 절대 좌표 (좌상단 + 폭/높이)

사용 예시::

    from src.inference.postprocess import postprocess_detections
    rows = postprocess_detections(detections, idx2id, max_det_per_image=4)
"""
from __future__ import annotations

from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


def postprocess_detections(
    detections: list[dict],
    idx2id: dict[int, int] | dict[str, int],
    *,
    max_det_per_image: int = 4,
) -> list[dict]:
    """추론 결과를 제출용 행(row) 리스트로 변환한다.

    Parameters
    ----------
    detections : list[dict]
        ``predictor.batch_predict()`` 의 출력.
        각 dict 에 ``image_stem``, ``boxes`` 키가 있어야 한다.
    idx2id : dict
        YOLO class index (int) → 원본 COCO category_id (int) 매핑.
        키가 str 인 경우 int 로 변환한다.
    max_det_per_image : int
        이미지당 유지할 최대 detection 수. confidence 내림차순 정렬 후
        상위 K개만 유지한다.

    Returns
    -------
    list[dict]
        제출 CSV 에 바로 쓸 수 있는 행 리스트::

            {
                "image_id": int,          # 파일명 stem 의 정수
                "category_id": int,       # 원본 COCO category_id
                "bbox_x": float,          # 좌상단 x
                "bbox_y": float,          # 좌상단 y
                "bbox_w": float,          # 폭
                "bbox_h": float,          # 높이
                "score": float,           # confidence
            }
    """
    # idx2id 의 키를 int 로 정규화
    idx2id_norm: dict[int, int] = {}
    for k, v in idx2id.items():
        idx2id_norm[int(k)] = int(v)

    rows: list[dict] = []
    total_truncated = 0
    unmapped_classes: set[int] = set()

    for det in detections:
        image_stem = det.get("image_stem", "")
        boxes = det.get("boxes", [])

        # image_id: 파일명 stem 에서 정수 추출
        image_id = _parse_image_id(image_stem)

        # confidence 내림차순 정렬
        sorted_boxes = sorted(boxes, key=lambda b: b["conf"], reverse=True)

        # Top-K 선택
        if len(sorted_boxes) > max_det_per_image:
            total_truncated += len(sorted_boxes) - max_det_per_image
            sorted_boxes = sorted_boxes[:max_det_per_image]

        for box in sorted_boxes:
            class_idx = box["class_idx"]
            xywh = box["xywh"]  # [x, y, w, h] 절대 좌표

            # class index → category_id 매핑
            category_id = idx2id_norm.get(class_idx)
            if category_id is None:
                unmapped_classes.add(class_idx)
                # fallback: class_idx 를 그대로 사용 (경고 출력)
                category_id = class_idx

            rows.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox_x": round(xywh[0], 2),
                "bbox_y": round(xywh[1], 2),
                "bbox_w": round(xywh[2], 2),
                "bbox_h": round(xywh[3], 2),
                "score": round(box["conf"], 6),
            })

    if total_truncated > 0:
        logger.info("Top-%d 규칙 적용: %d 개 detection 제거됨",
                     max_det_per_image, total_truncated)
    if unmapped_classes:
        logger.warning("매핑되지 않은 class index: %s (category_id = class_idx 로 대체)",
                        sorted(unmapped_classes))

    logger.info("후처리 완료 | 이미지=%d | 최종 행=%d", len(detections), len(rows))
    return rows


def _parse_image_id(stem: str) -> int:
    """파일명 stem 에서 정수 image_id 를 추출한다.

    - 순수 숫자 stem: int(stem) → e.g. "123" → 123
    - 그 외: 해시 기반 정수 (예외 상황용)
    """
    try:
        return int(stem)
    except ValueError:
        # 숫자가 아닌 stem → 파일명 해시 사용 (비표준이지만 안전)
        logger.warning("image_stem '%s' 이 정수가 아닙니다. hash 사용.", stem)
        return abs(hash(stem)) % (10 ** 9)
