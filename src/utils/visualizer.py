"""탐지 품질 디버깅을 위한 시각화 유틸리티.

주요 기능:
- 예측 bbox 시각화 (`draw_predictions`)
- GT vs Pred 나란히 비교 (`draw_comparison`)
- 배치 결과 이미지 저장 (`visualize_batch`)
- 클래스별 지표 바 차트 (`plot_class_ap`)
- Ultralytics `results.csv` 기반 학습 곡선 시각화 (`plot_training_curves`)
- 클래스별 지표 로딩/선택 헬퍼
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

_cv2 = None
_pil_modules = None
_pil_import_tried = False
_unicode_font_path: str | None = None
_unicode_font_path_resolved = False
_unicode_font_cache: dict[int, Any] = {}
_unicode_font_warned = False


def _get_cv2():
    """OpenCV를 지연 import한다."""
    global _cv2
    if _cv2 is None:
        try:
            import cv2
            _cv2 = cv2
        except ImportError as exc:
            raise ImportError(
                "cv2가 설치되어 있지 않습니다. 다음 명령으로 설치하세요: "
                "pip install opencv-python-headless"
            ) from exc
    return _cv2


def _load_image(image: np.ndarray | Path | str) -> np.ndarray:
    """ndarray 또는 경로에서 이미지를 읽어 수정 가능한 복사본을 반환한다."""
    cv2 = _get_cv2()
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"이미지를 불러오지 못했습니다: {image}")
        return img
    return image.copy()


def _generate_palette(n: int) -> list[tuple[int, int, int]]:
    """`n`개의 구분 가능한 BGR 색상을 생성한다."""
    palette: list[tuple[int, int, int]] = []
    for i in range(n):
        hue = int(180 * i / max(n, 1))
        h = hue / 30.0
        sector = int(h)
        frac = h - sector
        v = 255
        s = 200
        p = int(v * (1 - s / 255.0))
        q = int(v * (1 - s / 255.0 * frac))
        t = int(v * (1 - s / 255.0 * (1 - frac)))
        if sector == 0:
            r, g, b = v, t, p
        elif sector == 1:
            r, g, b = q, v, p
        elif sector == 2:
            r, g, b = p, v, t
        elif sector == 3:
            r, g, b = p, q, v
        elif sector == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        palette.append((b, g, r))
    return palette


_DEFAULT_PALETTE = _generate_palette(80)


def _get_color(class_idx: int) -> tuple[int, int, int]:
    """클래스 인덱스 기반 BGR 색상을 반환한다."""
    return _DEFAULT_PALETTE[class_idx % len(_DEFAULT_PALETTE)]


def _contains_non_ascii(text: str) -> bool:
    """텍스트에 비 ASCII 문자가 포함되어 있는지 확인한다."""
    return any(ord(ch) > 127 for ch in str(text))


def _get_pil_modules():
    """Pillow 모듈을 지연 import 한다."""
    global _pil_modules, _pil_import_tried
    if _pil_modules is not None:
        return _pil_modules
    if _pil_import_tried:
        return None

    _pil_import_tried = True
    try:
        from PIL import Image, ImageDraw, ImageFont

        _pil_modules = (Image, ImageDraw, ImageFont)
        return _pil_modules
    except ImportError:
        return None


def _resolve_unicode_font_path() -> str | None:
    """한글/유니코드 렌더링 가능한 폰트 경로를 탐색한다."""
    global _unicode_font_path, _unicode_font_path_resolved
    if _unicode_font_path_resolved:
        return _unicode_font_path

    _unicode_font_path_resolved = True
    env_path = os.environ.get("VISUALIZER_FONT_PATH", "").strip()

    candidates: list[str] = []
    if env_path:
        candidates.append(env_path)

    candidates.extend(
        [
            r"C:\Windows\Fonts\malgun.ttf",  # 맑은 고딕
            r"C:\Windows\Fonts\malgunbd.ttf",
            r"C:\Windows\Fonts\gulim.ttc",
            r"C:\Windows\Fonts\batang.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        ]
    )

    for path in candidates:
        if path and Path(path).exists():
            _unicode_font_path = path
            return _unicode_font_path

    return None


def _font_px_from_scale(font_scale: float) -> int:
    """OpenCV font_scale 값을 PIL 폰트 px 크기로 근사 변환한다."""
    return max(12, int(round(18 * max(0.3, float(font_scale)))))


def _get_unicode_font(font_scale: float):
    """font_scale에 맞는 유니코드 폰트를 반환한다."""
    global _unicode_font_warned

    modules = _get_pil_modules()
    if modules is None:
        if not _unicode_font_warned:
            logger.warning("Pillow 미설치로 유니코드 텍스트 렌더링을 비활성화합니다.")
            _unicode_font_warned = True
        return None

    _, _, ImageFont = modules
    font_path = _resolve_unicode_font_path()
    if not font_path:
        if not _unicode_font_warned:
            logger.warning(
                "유니코드 폰트를 찾지 못해 OpenCV 텍스트 렌더링으로 폴백합니다. "
                "환경변수 VISUALIZER_FONT_PATH로 폰트 경로를 지정할 수 있습니다."
            )
            _unicode_font_warned = True
        return None

    size = _font_px_from_scale(font_scale)
    cached = _unicode_font_cache.get(size)
    if cached is not None:
        return cached

    try:
        font = ImageFont.truetype(font_path, size=size)
    except Exception as exc:  # noqa: BLE001
        if not _unicode_font_warned:
            logger.warning("유니코드 폰트 로딩 실패(%s): %s", type(exc).__name__, exc)
            _unicode_font_warned = True
        return None

    _unicode_font_cache[size] = font
    return font


def _get_text_size(text: str, font_scale: float, thickness: int = 1) -> tuple[int, int]:
    """텍스트 크기를 반환한다. 유니코드는 PIL 기반 측정을 우선한다."""
    if _contains_non_ascii(text):
        modules = _get_pil_modules()
        font = _get_unicode_font(font_scale)
        if modules is not None and font is not None:
            Image, ImageDraw, _ = modules
            canvas = Image.new("RGB", (1, 1), (0, 0, 0))
            draw = ImageDraw.Draw(canvas)
            left, top, right, bottom = draw.textbbox((0, 0), str(text), font=font)
            return max(1, int(right - left)), max(1, int(bottom - top))

    cv2 = _get_cv2()
    (tw, th), _ = cv2.getTextSize(str(text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    return int(tw), int(th)


def _ensure_pil_state(img: np.ndarray, pil_state: dict[str, Any]) -> bool:
    """필요 시 PIL 그리기 컨텍스트를 초기화한다."""
    modules = _get_pil_modules()
    if modules is None:
        return False

    if pil_state.get("draw") is not None and pil_state.get("pil_img") is not None:
        return True

    cv2 = _get_cv2()
    Image, ImageDraw, _ = modules
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil_state["pil_img"] = pil_img
    pil_state["draw"] = ImageDraw.Draw(pil_img)
    return True


def _flush_pil_state(img: np.ndarray, pil_state: dict[str, Any]) -> None:
    """PIL에서 수정한 캔버스를 OpenCV ndarray에 반영한다."""
    if not pil_state:
        return
    pil_img = pil_state.get("pil_img")
    if pil_img is None:
        return

    cv2 = _get_cv2()
    img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _draw_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    *,
    font_scale: float = 0.5,
    thickness: int = 1,
    line_type: int | None = None,
    pil_state: dict[str, Any] | None = None,
) -> None:
    """텍스트를 그린다. 유니코드는 PIL 렌더링을 우선 사용한다."""
    cv2 = _get_cv2()

    text = str(text)
    x, baseline_y = int(org[0]), int(org[1])
    if not _contains_non_ascii(text):
        cv2.putText(
            img,
            text,
            (x, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA if line_type is None else line_type,
        )
        return

    font = _get_unicode_font(font_scale)
    if font is None:
        cv2.putText(
            img,
            text,
            (x, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA if line_type is None else line_type,
        )
        return

    if pil_state is None:
        pil_state = {}
    if not _ensure_pil_state(img, pil_state):
        cv2.putText(
            img,
            text,
            (x, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA if line_type is None else line_type,
        )
        return

    _, text_h = _get_text_size(text, font_scale, thickness)
    top_y = max(0, baseline_y - text_h)
    draw = pil_state["draw"]
    draw.text((x, top_y), text, fill=(int(color[2]), int(color[1]), int(color[0])), font=font)


def _to_xyxy(box: dict) -> tuple[float, float, float, float]:
    """box dict에서 `(x1, y1, x2, y2)` 좌표를 추출한다."""
    xyxy = box.get("xyxy", [0, 0, 0, 0])
    if len(xyxy) != 4:
        return (0.0, 0.0, 0.0, 0.0)
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return (x1, y1, x2, y2)


def _bbox_iou_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """`xyxy` 형식 두 박스의 IoU를 계산한다."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    aw = max(0.0, ax2 - ax1)
    ah = max(0.0, ay2 - ay1)
    bw = max(0.0, bx2 - bx1)
    bh = max(0.0, by2 - by1)

    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _match_pred_to_gt(
    gt_boxes: list[dict],
    pred_boxes: list[dict],
    class_aware: bool = True,
) -> list[dict]:
    """탐욕적(greedy) 1:1 방식으로 Pred를 GT에 매칭한다.

    - Pred는 confidence 내림차순으로 처리한다.
    - GT는 최대 1회만 매칭된다.

    반환값은 `pred_boxes`와 같은 인덱스 순서를 갖는다:
    `[{"pred_idx": int, "gt_idx": int | None, "iou": float}, ...]`
    """
    matches = [{"pred_idx": i, "gt_idx": None, "iou": 0.0} for i in range(len(pred_boxes))]
    if not gt_boxes or not pred_boxes:
        return matches

    sorted_pred_indices = sorted(
        range(len(pred_boxes)),
        key=lambda i: float(pred_boxes[i].get("conf", 0.0)),
        reverse=True,
    )

    used_gt: set[int] = set()
    gt_xyxy = [_to_xyxy(g) for g in gt_boxes]

    for pred_idx in sorted_pred_indices:
        pred = pred_boxes[pred_idx]
        pred_cls = int(pred.get("class_idx", -1))
        pred_xyxy = _to_xyxy(pred)

        best_gt_idx: int | None = None
        best_iou = 0.0

        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in used_gt:
                continue
            if class_aware and int(gt.get("class_idx", -1)) != pred_cls:
                continue

            iou = _bbox_iou_xyxy(pred_xyxy, gt_xyxy[gt_idx])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None:
            used_gt.add(best_gt_idx)
            matches[pred_idx] = {
                "pred_idx": pred_idx,
                "gt_idx": best_gt_idx,
                "iou": float(best_iou),
            }

    return matches


def draw_predictions(
    image: np.ndarray | Path | str,
    boxes: list[dict],
    *,
    idx2name: dict[int, str] | None = None,
    thickness: int = 2,
    font_scale: float = 0.5,
    show_score: bool = True,
    top_k: int | None = None,
    highlight_overflow: bool = False,
    show_rank: bool = False,
    overflow_color: tuple[int, int, int] = (160, 160, 160),
    overflow_thickness: int = 1,
) -> np.ndarray:
    """이미지 위에 예측 박스를 그린다.

    Top-K 디버깅 옵션:
    - confidence 기준 rank 부여
    - `top_k` 초과 박스 스타일 분리
    - 라벨에 rank prefix 표시
    """
    cv2 = _get_cv2()
    img = _load_image(image)
    pil_state: dict[str, Any] = {}

    if top_k is not None and top_k <= 0:
        raise ValueError("top_k는 양의 정수이거나 None이어야 합니다")

    # 랭킹 관련 옵션을 쓰지 않으면 기존 순서 동작을 유지한다.
    use_ranking = top_k is not None or highlight_overflow or show_rank
    if use_ranking:
        ordered = sorted(boxes, key=lambda b: float(b.get("conf", 0.0)), reverse=True)
    else:
        ordered = list(boxes)

    for rank, box in enumerate(ordered, start=1):
        class_idx = int(box.get("class_idx", 0))
        conf = float(box.get("conf", 0.0))
        x1f, y1f, x2f, y2f = _to_xyxy(box)
        x1, y1, x2, y2 = int(x1f), int(y1f), int(x2f), int(y2f)

        is_overflow = bool(top_k is not None and rank > top_k)
        if highlight_overflow and is_overflow:
            color = overflow_color
            box_thickness = overflow_thickness
        else:
            color = _get_color(class_idx)
            box_thickness = thickness

        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)

        if idx2name and class_idx in idx2name:
            name = idx2name[class_idx]
        else:
            name = str(class_idx)

        label_parts: list[str] = []
        if show_rank:
            label_parts.append(f"#{rank}")
        label_parts.append(name)
        if show_score:
            label_parts.append(f"{conf:.2f}")
        label = " ".join(label_parts)

        tw, th = _get_text_size(label, font_scale, 1)
        label_y1 = max(y1 - th - 6, 0)
        cv2.rectangle(img, (x1, label_y1), (x1 + tw + 4, y1), color, -1)
        _draw_text(
            img,
            label,
            (x1 + 2, y1 - 4),
            (255, 255, 255),
            font_scale=font_scale,
            thickness=1,
            line_type=cv2.LINE_AA,
            pil_state=pil_state,
        )

    _flush_pil_state(img, pil_state)
    return img


def draw_comparison(
    image: np.ndarray | Path | str,
    gt_boxes: list[dict],
    pred_boxes: list[dict],
    *,
    idx2name: dict[int, str] | None = None,
    font_scale: float = 0.5,
    show_iou: bool = False,
    iou_threshold: float = 0.75,
    class_aware_match: bool = True,
    show_iou_summary: bool = True,
) -> np.ndarray:
    """GT/Pred를 좌우로 나란히 배치한 비교 이미지를 생성한다.

    - 왼쪽 패널: GT 박스(초록)
    - 오른쪽 패널: Pred 박스(빨강), 필요 시 IoU 오버레이 표시
    """
    cv2 = _get_cv2()
    img = _load_image(image)

    gt_img = img.copy()
    pred_img = img.copy()
    gt_pil_state: dict[str, Any] = {}
    pred_pil_state: dict[str, Any] = {}

    # 왼쪽: GT
    for box in gt_boxes:
        x1f, y1f, x2f, y2f = _to_xyxy(box)
        x1, y1, x2, y2 = int(x1f), int(y1f), int(x2f), int(y2f)
        cv2.rectangle(gt_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cls = int(box.get("class_idx", 0))
        name = idx2name.get(cls, str(cls)) if idx2name else str(cls)
        _draw_text(
            gt_img,
            f"GT:{name}",
            (x1, max(y1 - 4, 12)),
            (0, 255, 0),
            font_scale=font_scale,
            thickness=1,
            line_type=cv2.LINE_AA,
            pil_state=gt_pil_state,
        )

    # 선택적 IoU 매칭
    match_by_pred_idx: dict[int, dict] = {}
    if show_iou or show_iou_summary:
        for m in _match_pred_to_gt(gt_boxes, pred_boxes, class_aware=class_aware_match):
            match_by_pred_idx[int(m["pred_idx"])] = m

    # 오른쪽: Pred
    for pred_idx, box in enumerate(pred_boxes):
        x1f, y1f, x2f, y2f = _to_xyxy(box)
        x1, y1, x2, y2 = int(x1f), int(y1f), int(x2f), int(y2f)
        conf = float(box.get("conf", 0.0))
        cls = int(box.get("class_idx", 0))
        name = idx2name.get(cls, str(cls)) if idx2name else str(cls)

        # Pred 패널은 일관성을 위해 박스 외곽선을 항상 빨강으로 유지한다.
        cv2.rectangle(pred_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        label = f"P:{name} {conf:.2f}"
        text_color = (0, 0, 255)
        if show_iou:
            iou = float(match_by_pred_idx.get(pred_idx, {"iou": 0.0})["iou"])
            label = f"{label} IoU:{iou:.2f}"
            if iou >= iou_threshold:
                text_color = (0, 255, 0)
            elif iou >= 0.75:
                text_color = (0, 255, 255)
            else:
                text_color = (0, 0, 255)

        _draw_text(
            pred_img,
            label,
            (x1, max(y1 - 4, 12)),
            text_color,
            font_scale=font_scale,
            thickness=1,
            line_type=cv2.LINE_AA,
            pil_state=pred_pil_state,
        )

    _draw_text(
        gt_img,
        "정답 GT",
        (10, 25),
        (0, 255, 0),
        font_scale=0.7,
        thickness=2,
        line_type=cv2.LINE_AA,
        pil_state=gt_pil_state,
    )
    _draw_text(
        pred_img,
        "예측 Pred",
        (10, 25),
        (0, 0, 255),
        font_scale=0.7,
        thickness=2,
        line_type=cv2.LINE_AA,
        pil_state=pred_pil_state,
    )

    if show_iou and show_iou_summary:
        matched = [m for m in match_by_pred_idx.values() if m.get("gt_idx") is not None]
        passed = sum(1 for m in matched if float(m["iou"]) >= iou_threshold)
        total_pred = len(pred_boxes)
        unmatched_gt = len(gt_boxes) - len({int(m["gt_idx"]) for m in matched})

        _draw_text(
            pred_img,
            f"IoU>={iou_threshold:.2f}: {passed}/{total_pred} 예측",
            (10, 50),
            (255, 255, 255),
            font_scale=0.55,
            thickness=1,
            line_type=cv2.LINE_AA,
            pil_state=pred_pil_state,
        )
        _draw_text(
            pred_img,
            f"미매칭 GT: {max(0, unmatched_gt)}",
            (10, 70),
            (255, 255, 255),
            font_scale=0.55,
            thickness=1,
            line_type=cv2.LINE_AA,
            pil_state=pred_pil_state,
        )

    _flush_pil_state(gt_img, gt_pil_state)
    _flush_pil_state(pred_img, pred_pil_state)
    return np.hstack([gt_img, pred_img])


def visualize_batch(
    detections: list[dict],
    output_dir: Path | str,
    *,
    idx2name: dict[int, str] | None = None,
    max_images: int = 50,
    show_score: bool = True,
    top_k: int | None = None,
    highlight_overflow: bool = False,
    show_rank: bool = False,
) -> list[Path]:
    """배치 예측 결과를 시각화해 파일로 저장한다."""
    cv2 = _get_cv2()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for det in detections[:max_images]:
        image_path = det.get("image_path", "")
        image_stem = det.get("image_stem", "unknown")
        boxes = det.get("boxes", [])

        if not Path(image_path).exists():
            logger.warning("이미지 파일이 없어 건너뜁니다: %s", image_path)
            continue

        vis_img = draw_predictions(
            image_path,
            boxes,
            idx2name=idx2name,
            show_score=show_score,
            top_k=top_k,
            highlight_overflow=highlight_overflow,
            show_rank=show_rank,
        )

        out_path = output_dir / f"{image_stem}_pred.jpg"
        cv2.imwrite(str(out_path), vis_img)
        saved.append(out_path)

    logger.info(
        "시각화 저장 완료 | %d/%d 이미지 | %s",
        len(saved),
        len(detections),
        output_dir,
    )
    return saved


def plot_class_ap(
    class_ap50: list[float],
    output_path: Path | str,
    *,
    class_names: list[str] | None = None,
    title: str = "클래스별 AP50",
    top_k: int | None = None,
    order: str = "desc",
    metric_label: str = "AP",
) -> Path:
    """클래스별 지표를 수평 바 차트로 그린다.

    Parameters
    ----------
    class_ap50 : list[float]
        클래스별 지표 리스트 (`index = class_idx`).
    order : str
        top-k 적용 전 정렬 순서: `"desc"` 또는 `"asc"`.
    metric_label : str
        X축 레이블 (예: `AP50`, `mAP75_95`).
    """
    if order not in {"asc", "desc"}:
        raise ValueError("order는 'asc' 또는 'desc' 중 하나여야 합니다")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib가 설치되어 있지 않아 차트 생성을 건너뜁니다.")
        return Path(output_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(class_ap50)
    if class_names and len(class_names) == n:
        names = class_names
    else:
        names = [f"cls_{i}" for i in range(n)]

    pairs = sorted(
        zip(class_ap50, names),
        key=lambda x: float(x[0]),
        reverse=(order == "desc"),
    )
    if top_k is not None:
        pairs = pairs[:top_k]

    aps = [float(p[0]) for p in pairs]
    labels = [str(p[1]) for p in pairs]

    fig, ax = plt.subplots(figsize=(10, max(4, len(pairs) * 0.35)))
    y_pos = range(len(pairs))
    colors = ["#2ecc71" if ap >= 0.75 else "#e74c3c" for ap in aps]

    ax.barh(y_pos, aps, color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(metric_label)
    ax.set_title(title)
    ax.invert_yaxis()

    for i, ap in enumerate(aps):
        ax.text(ap + 0.01, i, f"{ap:.3f}", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close(fig)

    logger.info("클래스 지표 차트 저장 완료 | %s", output_path)
    return output_path


def plot_training_curves(
    results_csv: Path | str,
    output_path: Path | str,
    *,
    title: str = "학습 곡선",
) -> Path:
    """Ultralytics `results.csv`에서 학습 곡선을 시각화한다."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib가 설치되어 있지 않아 차트 생성을 건너뜁니다.")
        return Path(output_path)

    results_csv = Path(results_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not results_csv.exists():
        logger.warning("results.csv 파일이 존재하지 않습니다: %s", results_csv)
        return output_path

    with results_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        logger.warning("results.csv 파일이 비어 있습니다: %s", results_csv)
        return output_path

    columns = {col.strip(): col for col in rows[0].keys()}
    epochs_list = list(range(1, len(rows) + 1))

    def _get_col(name_part: str) -> list[float]:
        for clean, raw in columns.items():
            if name_part in clean:
                try:
                    return [float(r[raw]) for r in rows]
                except (ValueError, KeyError):
                    return []
        return []

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14)

    ax = axes[0, 0]
    for loss_name, color in [
        ("train/box_loss", "blue"),
        ("train/cls_loss", "orange"),
        ("train/dfl_loss", "green"),
    ]:
        vals = _get_col(loss_name)
        if vals:
            short_name = loss_name.split("/")[-1]
            ax.plot(epochs_list[:len(vals)], vals, label=short_name, color=color, linewidth=1)
    ax.set_title("Train Loss")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for loss_name, color in [
        ("val/box_loss", "blue"),
        ("val/cls_loss", "orange"),
        ("val/dfl_loss", "green"),
    ]:
        vals = _get_col(loss_name)
        if vals:
            short_name = loss_name.split("/")[-1]
            ax.plot(epochs_list[:len(vals)], vals, label=short_name, color=color, linewidth=1)
    ax.set_title("Val Loss")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    map50 = _get_col("mAP50(B)")
    map50_95 = _get_col("mAP50-95(B)")
    map75_95 = _get_col("mAP75-95")  # 만약 Custom CSV에 추가했다면

    if map50:
        pure_map50 = _get_col("metrics/mAP50(B)")
        if not pure_map50:
            pure_map50 = map50
        ax.plot(epochs_list[:len(pure_map50)], pure_map50, label="mAP50", color="blue", linewidth=1.5)
    if map50_95:
        ax.plot(epochs_list[:len(map50_95)], map50_95, label="mAP50-95", color="red", linewidth=1.5)
    if map75_95:
        ax.plot(epochs_list[:len(map75_95)], map75_95, label="mAP75-95", color="green", linewidth=2.0)

    ax.set_title("mAP (Note: 50-95 is Standard)")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    lr_vals = _get_col("lr/pg0")
    if lr_vals:
        ax.plot(epochs_list[:len(lr_vals)], lr_vals, label="lr/pg0", color="purple", linewidth=1)
    lr_vals2 = _get_col("lr/pg1")
    if lr_vals2:
        ax.plot(epochs_list[:len(lr_vals2)], lr_vals2, label="lr/pg1", color="teal", linewidth=1)
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close(fig)

    logger.info("학습 곡선 차트 저장 완료 | %s", output_path)
    return output_path


def load_idx2name(label_map_path: Path | str) -> dict[int, str]:
    """`label_map_full.json`에서 idx->name 매핑을 로드한다."""
    label_map_path = Path(label_map_path)
    with label_map_path.open("r", encoding="utf-8") as f:
        lm = json.load(f)

    names = lm.get("names", [])
    return {i: n for i, n in enumerate(names)}


def load_per_class_metric(
    metrics_path: Path | str,
    key: str = "eval_per_class_mAP75_95",
) -> list[float]:
    """`metrics.json`에서 클래스별 지표 리스트를 로드한다."""
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics 파일이 존재하지 않습니다: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    if key not in metrics:
        raise KeyError(f"지표 키를 찾을 수 없습니다: {key}")

    scores = metrics[key]
    if not isinstance(scores, list):
        raise TypeError(
            f"지표 '{key}'는 리스트여야 합니다: {type(scores).__name__} 타입 입력"
        )

    return [float(v) for v in scores]


def select_classes_by_metric(
    scores: list[float],
    class_names: list[str] | None = None,
    k: int = 5,
    order: str = "asc",
) -> list[dict]:
    """지표 점수 기준으로 상/하위 클래스를 선택한다.

    Returns:
        [{"class_idx": int, "class_name": str, "score": float}, ...]
    """
    if order not in {"asc", "desc"}:
        raise ValueError("order는 'asc' 또는 'desc' 중 하나여야 합니다")
    if k <= 0:
        return []

    items: list[tuple[int, str, float]] = []
    for idx, score in enumerate(scores):
        if class_names and idx < len(class_names):
            name = class_names[idx]
        else:
            name = f"cls_{idx}"
        items.append((idx, name, float(score)))

    items_sorted = sorted(items, key=lambda x: x[2], reverse=(order == "desc"))[:k]
    return [
        {"class_idx": idx, "class_name": name, "score": score}
        for idx, name, score in items_sorted
    ]
