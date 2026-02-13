"""src.inference.predictor — 배치 추론 오케스트레이터.

테스트 이미지 디렉터리에 대해 YOLO 추론을 실행하고,
이미지별 detection 결과를 표준화된 dict 리스트로 반환한다.

사용 예시::

    from src.inference.predictor import batch_predict
    detections = batch_predict(
        weights_path=Path("runs/exp/weights/best.pt"),
        source=Path("data/raw/test_images"),
        conf=0.25, iou=0.5, max_det=300, device=0, imgsz=640,
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models.detector import PillDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


def batch_predict(
    weights_path: Path,
    source: Path | str,
    *,
    conf: float = 0.25,
    iou: float = 0.5,
    max_det: int = 300,
    device: Any = None,
    imgsz: int = 640,
    verbose: bool = False,
    augment: bool = False,
) -> list[dict]:
    """배치 추론을 실행하고 이미지별 detection dict 리스트를 반환한다.

    Parameters
    ----------
    weights_path : Path
        학습 완료 가중치 (.pt).
    source : Path | str
        추론 대상 이미지 디렉터리 또는 단일 이미지 경로.
    conf : float
        confidence threshold.
    iou : float
        NMS IoU threshold.
    max_det : int
        이미지당 최대 detection 수 (Ultralytics 전달).
    device : Any
        GPU 디바이스 (0, "cpu", etc.).
    imgsz : int
        추론 이미지 크기.
    verbose : bool
        Ultralytics verbose 출력.
    augment : bool
        True 이면 TTA(Test-Time Augmentation)를 적용한다.

    Returns
    -------
    list[dict]
        각 dict 는 이미지 1장의 detection 결과::

            {
                "image_path": str,          # 원본 이미지 경로
                "image_stem": str,          # 파일명 stem (확장자 제외)
                "orig_shape": (H, W),       # 원본 이미지 해상도
                "boxes": [                  # detection 리스트
                    {
                        "class_idx": int,   # YOLO class index
                        "conf": float,      # confidence
                        "xyxy": [x1,y1,x2,y2],  # 절대 픽셀 좌표
                        "xywh": [x,y,w,h],      # 절대 픽셀 좌표 (좌상단 + wh)
                    },
                    ...
                ]
            }
    """
    weights_path = Path(weights_path)
    source = Path(source)

    if not weights_path.exists():
        raise FileNotFoundError(f"가중치 파일이 존재하지 않습니다: {weights_path}")
    if not source.exists():
        raise FileNotFoundError(f"추론 대상이 존재하지 않습니다: {source}")

    logger.info("배치 추론 시작 | weights=%s | source=%s", weights_path, source)
    logger.info("  conf=%.3f | iou=%.3f | max_det=%d | imgsz=%d",
                conf, iou, max_det, imgsz)

    # 모델 로드
    detector = PillDetector.from_weights(weights_path)

    # 추론 실행
    raw_results = detector.predict(
        source=source,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
        imgsz=imgsz,
        save=False,
        verbose=verbose,
        augment=augment,
    )

    # 결과 파싱
    detections: list[dict] = []
    for result in raw_results:
        image_path = Path(result.path) if hasattr(result, "path") else Path("unknown")
        orig_shape = tuple(result.orig_img.shape[:2]) if hasattr(result, "orig_img") else (0, 0)

        boxes_list: list[dict] = []
        if result.boxes is not None and len(result.boxes) > 0:
            # xyxy: 절대 좌표 (x1, y1, x2, y2)
            xyxy_tensor = result.boxes.xyxy.cpu()
            conf_tensor = result.boxes.conf.cpu()
            cls_tensor = result.boxes.cls.cpu()

            for i in range(len(result.boxes)):
                x1, y1, x2, y2 = xyxy_tensor[i].tolist()
                box_conf = float(conf_tensor[i])
                class_idx = int(cls_tensor[i])

                # xyxy → xywh (좌상단 + 폭/높이)
                bx = x1
                by = y1
                bw = x2 - x1
                bh = y2 - y1

                boxes_list.append({
                    "class_idx": class_idx,
                    "conf": box_conf,
                    "xyxy": [x1, y1, x2, y2],
                    "xywh": [bx, by, bw, bh],
                })

        detections.append({
            "image_path": str(image_path),
            "image_stem": image_path.stem,
            "orig_shape": orig_shape,
            "boxes": boxes_list,
        })

    total_boxes = sum(len(d["boxes"]) for d in detections)
    logger.info("추론 완료 | 이미지=%d | 총 detection=%d", len(detections), total_boxes)

    return detections
