from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image

from .io_utils import scan_image_files


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _xywh_to_xyxy(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    return x, y, x + w, y + h


def _bbox_iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = a_area + b_area - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def _clip_xyxy(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[int, int, int, int]:
    x1i = int(max(0, min(w, round(x1))))
    y1i = int(max(0, min(h, round(y1))))
    x2i = int(max(0, min(w, round(x2))))
    y2i = int(max(0, min(h, round(y2))))
    if x2i < x1i:
        x1i, x2i = x2i, x1i
    if y2i < y1i:
        y1i, y2i = y2i, y1i
    return x1i, y1i, x2i, y2i


def _build_source_image_index(config: dict, base_dir: Path, resolve_path_fn: Callable[[str, Path], Path]) -> dict[str, dict[str, Path]]:
    paths_cfg = config.get("paths", {})
    train_images_dir = resolve_path_fn(paths_cfg["train_images_dir"], base_dir)
    train_index, _ = scan_image_files(train_images_dir, recursive=True)

    external_index: dict[str, Path] = {}
    ext_cfg = config.get("external_data", {})
    for src in ext_cfg.get("sources", []):
        img_dir = resolve_path_fn(src["images_dir"], base_dir)
        recursive = bool(src.get("recursive", True))
        idx, _ = scan_image_files(img_dir, recursive=recursive)
        for k, p in idx.items():
            external_index.setdefault(k, p)

    return {"train": train_index, "external": external_index}


def _iter_rows_by_image(records: list[dict], target_sources: set[str]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        source = str(r.get("source", "")).strip().lower()
        if source not in target_sources:
            continue
        fn = str(r.get("file_name", "")).strip()
        if not fn:
            continue
        grouped.setdefault((source, fn), []).append(r)
    return grouped


def _extract_component_from_roi(
    roi_rgb: np.ndarray,
    label_bbox_roi: tuple[int, int, int, int],
    min_component_area_px: int,
) -> tuple[Optional[np.ndarray], Optional[tuple[int, int, int, int]], float]:
    """
    ROI에서 컨투어 기반 후보를 찾고, label bbox와 IoU가 가장 큰 컴포넌트를 반환한다.
    반환: (component_mask, component_bbox_xyxy_in_roi, iou_with_label_bbox)
    """
    if roi_rgb.size == 0:
        return None, None, 0.0

    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    sat_mask = np.where(sat > 22, 255, 0).astype(np.uint8)

    candidates = [otsu_bin, otsu_inv, cv2.bitwise_or(otsu_inv, sat_mask)]
    kernel = np.ones((3, 3), np.uint8)

    best_mask: Optional[np.ndarray] = None
    best_bbox: Optional[tuple[int, int, int, int]] = None
    best_iou = 0.0

    for raw in candidates:
        mask = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = float(w * h)
            if area < float(min_component_area_px):
                continue
            bbox = (x, y, x + w, y + h)
            iou = _bbox_iou_xyxy(
                (float(label_bbox_roi[0]), float(label_bbox_roi[1]), float(label_bbox_roi[2]), float(label_bbox_roi[3])),
                (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            )
            if iou > best_iou:
                cmask = np.zeros(mask.shape[:2], dtype=np.uint8)
                cv2.drawContours(cmask, [cnt], contourIdx=-1, color=255, thickness=-1)
                best_mask = cmask
                best_bbox = bbox
                best_iou = iou

    return best_mask, best_bbox, float(best_iou)


@dataclass
class PixelAuditResult:
    filtered_records: list[dict]
    audit_rows: list[dict]


def run_pixel_overlap_audit(
    *,
    records: list[dict],
    config: dict,
    logs: dict[str, list[dict]],
    audit_logs: dict[str, list[dict]],
    base_dir: Path,
    resolve_path_fn: Callable[[str, Path], Path],
) -> PixelAuditResult:
    qa_cfg = config.get("quality_audit", {}).get("pixel_overlap", {})
    if not bool(qa_cfg.get("enabled", False)):
        return PixelAuditResult(filtered_records=records, audit_rows=[])

    action = str(qa_cfg.get("action", "audit_only")).strip().lower()
    sources_raw = qa_cfg.get("sources", ["train", "external"])
    if not isinstance(sources_raw, list) or not sources_raw:
        sources_raw = ["train", "external"]
    target_sources = {str(x).strip().lower() for x in sources_raw if str(x).strip()}

    min_coverage = float(qa_cfg.get("min_coverage", 0.85))
    min_bbox_iou = float(qa_cfg.get("min_bbox_iou", 0.75))
    min_component_area_px = int(qa_cfg.get("min_component_area_px", 60))
    roi_expand_ratio = float(qa_cfg.get("roi_expand_ratio", 0.25))
    log_every_images = int(qa_cfg.get("log_every_images", 500))
    audit_file = str(qa_cfg.get("audit_file", "audit_pixel_overlap.csv"))
    audit_logs.setdefault(audit_file, [])

    source_image_index = _build_source_image_index(config, base_dir, resolve_path_fn)
    grouped = _iter_rows_by_image(records, target_sources)
    total_images = len(grouped)
    t0 = perf_counter()
    print(
        f"[INFO] quality[pixel] start: images={total_images}, action={action}, "
        f"min_coverage={min_coverage}, min_bbox_iou={min_bbox_iou}",
        flush=True,
    )

    rows: list[dict] = []
    keep_flags = {id(r): True for r in records}

    for idx, ((source, file_name), image_rows) in enumerate(grouped.items(), start=1):
        if log_every_images > 0 and (idx == 1 or idx % log_every_images == 0):
            dt = perf_counter() - t0
            print(
                f"[INFO] quality[pixel] progress: {idx}/{total_images} ({dt:.1f}s)",
                flush=True,
            )
        image_path = source_image_index.get(source, {}).get(file_name.lower())
        if image_path is None:
            for r in image_rows:
                row = {
                    "source": source,
                    "file_name": file_name,
                    "source_json": str(r.get("source_json", "")),
                    "category_id": r.get("category_id", ""),
                    "coverage": 0.0,
                    "bbox_iou": 0.0,
                    "status": "image_not_found",
                    "is_suspect": 1,
                }
                rows.append(row)
                if action in {"exclude", "drop"}:
                    keep_flags[id(r)] = False
            continue

        try:
            with Image.open(image_path) as im:
                rgb = np.array(im.convert("RGB"))
        except Exception:
            for r in image_rows:
                row = {
                    "source": source,
                    "file_name": file_name,
                    "source_json": str(r.get("source_json", "")),
                    "category_id": r.get("category_id", ""),
                    "coverage": 0.0,
                    "bbox_iou": 0.0,
                    "status": "image_open_failed",
                    "is_suspect": 1,
                }
                rows.append(row)
                if action in {"exclude", "drop"}:
                    keep_flags[id(r)] = False
            continue

        h, w = rgb.shape[:2]
        for r in image_rows:
            x = _safe_float(r.get("bbox_x", 0.0))
            y = _safe_float(r.get("bbox_y", 0.0))
            bw = _safe_float(r.get("bbox_w", 0.0))
            bh = _safe_float(r.get("bbox_h", 0.0))
            lx1, ly1, lx2, ly2 = _xywh_to_xyxy(x, y, bw, bh)

            mx = bw * roi_expand_ratio
            my = bh * roi_expand_ratio
            rx1, ry1, rx2, ry2 = _clip_xyxy(lx1 - mx, ly1 - my, lx2 + mx, ly2 + my, w, h)
            if rx2 - rx1 <= 1 or ry2 - ry1 <= 1:
                coverage = 0.0
                iou = 0.0
                status = "invalid_roi"
                is_suspect = 1
            else:
                roi = rgb[ry1:ry2, rx1:rx2]
                label_roi = _clip_xyxy(lx1 - rx1, ly1 - ry1, lx2 - rx1, ly2 - ry1, roi.shape[1], roi.shape[0])
                comp_mask, comp_bbox_roi, _ = _extract_component_from_roi(roi, label_roi, min_component_area_px)
                if comp_mask is None or comp_bbox_roi is None:
                    coverage = 0.0
                    iou = 0.0
                    status = "no_component"
                    is_suspect = 1
                else:
                    cbx1, cby1, cbx2, cby2 = comp_bbox_roi
                    comp_bbox_img = (cbx1 + rx1, cby1 + ry1, cbx2 + rx1, cby2 + ry1)
                    iou = _bbox_iou_xyxy((lx1, ly1, lx2, ly2), tuple(float(v) for v in comp_bbox_img))

                    comp_pixels = int(np.count_nonzero(comp_mask > 0))
                    if comp_pixels <= 0:
                        coverage = 0.0
                    else:
                        bx1, by1, bx2, by2 = label_roi
                        in_bbox = np.zeros_like(comp_mask, dtype=np.uint8)
                        if bx2 > bx1 and by2 > by1:
                            in_bbox[by1:by2, bx1:bx2] = 1
                        in_pixels = int(np.count_nonzero((comp_mask > 0) & (in_bbox > 0)))
                        coverage = float(in_pixels / max(1, comp_pixels))

                    is_suspect = int((coverage < min_coverage) or (iou < min_bbox_iou))
                    status = "ok" if is_suspect == 0 else "low_overlap"

            row = {
                "source": source,
                "file_name": file_name,
                "source_json": str(r.get("source_json", "")),
                "category_id": r.get("category_id", ""),
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": bw,
                "bbox_h": bh,
                "coverage": round(float(coverage), 6),
                "bbox_iou": round(float(iou), 6),
                "status": status,
                "is_suspect": int(is_suspect),
                "threshold_coverage": min_coverage,
                "threshold_iou": min_bbox_iou,
            }
            rows.append(row)

            if is_suspect and action in {"exclude", "drop"}:
                keep_flags[id(r)] = False
                excluded = {
                    "source": source,
                    "file_name": file_name,
                    "source_json": str(r.get("source_json", "")),
                    "reason_code": "pixel_overlap_suspect",
                    "detail": f"coverage={row['coverage']}, iou={row['bbox_iou']}",
                }
                logs["excluded_rows"].append(excluded)
                if source == "external" and "excluded_rows_external" in logs:
                    logs["excluded_rows_external"].append(excluded)

    dt = perf_counter() - t0
    suspect_rows = sum(1 for r in rows if int(r.get("is_suspect", 0)) == 1)
    print(
        f"[INFO] quality[pixel] done: rows={len(rows)}, suspect_rows={suspect_rows} ({dt:.1f}s)",
        flush=True,
    )
    audit_logs[audit_file].extend(rows)
    filtered = [r for r in records if keep_flags.get(id(r), True)]
    return PixelAuditResult(filtered_records=filtered, audit_rows=rows)


def _detect_boxes_cv_contour(
    rgb: np.ndarray,
    *,
    min_pred_area_ratio: float,
    max_pred_area_ratio: float,
    min_aspect: float,
    max_aspect: float,
) -> list[tuple[float, float, float, float]]:
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(blur, 60, 140)
    mask = cv2.bitwise_or(otsu_inv, edges)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[tuple[float, float, float, float]] = []
    img_area = float(max(1, w * h))
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = float(bw * bh) / img_area
        if area < min_pred_area_ratio or area > max_pred_area_ratio:
            continue
        aspect = float(bw / max(1.0, float(bh)))
        if aspect < min_aspect or aspect > max_aspect:
            continue
        boxes.append((float(x), float(y), float(x + bw), float(y + bh)))
    return boxes


@dataclass
class AuxAuditResult:
    filtered_records: list[dict]
    audit_rows: list[dict]


def run_aux_detector_audit(
    *,
    records: list[dict],
    config: dict,
    logs: dict[str, list[dict]],
    audit_logs: dict[str, list[dict]],
    base_dir: Path,
    resolve_path_fn: Callable[[str, Path], Path],
) -> AuxAuditResult:
    qa_cfg = config.get("quality_audit", {}).get("auxiliary_detector", {})
    if not bool(qa_cfg.get("enabled", False)):
        return AuxAuditResult(filtered_records=records, audit_rows=[])

    action = str(qa_cfg.get("action", "audit_only")).strip().lower()
    detector = str(qa_cfg.get("detector", "cv_contour")).strip().lower()
    sources_raw = qa_cfg.get("sources", ["train", "external"])
    if not isinstance(sources_raw, list) or not sources_raw:
        sources_raw = ["train", "external"]
    target_sources = {str(x).strip().lower() for x in sources_raw if str(x).strip()}

    min_match_iou = float(qa_cfg.get("min_match_iou", 0.75))
    min_pred_area_ratio = float(qa_cfg.get("min_pred_area_ratio", 0.001))
    max_pred_area_ratio = float(qa_cfg.get("max_pred_area_ratio", 0.5))
    min_aspect = float(qa_cfg.get("min_aspect", 0.2))
    max_aspect = float(qa_cfg.get("max_aspect", 5.0))
    log_every_images = int(qa_cfg.get("log_every_images", 500))
    audit_file = str(qa_cfg.get("audit_file", "audit_aux_detector_iou.csv"))
    audit_logs.setdefault(audit_file, [])

    yolo_model = None
    if detector == "yolo":
        weights = str(qa_cfg.get("yolo_weights", "")).strip()
        if weights:
            try:
                from ultralytics import YOLO  # type: ignore

                wpath = resolve_path_fn(weights, base_dir)
                if wpath.exists():
                    yolo_model = YOLO(str(wpath))
                else:
                    detector = "cv_contour"
            except Exception:
                detector = "cv_contour"
        else:
            detector = "cv_contour"

    source_image_index = _build_source_image_index(config, base_dir, resolve_path_fn)
    grouped = _iter_rows_by_image(records, target_sources)
    total_images = len(grouped)
    t0 = perf_counter()
    print(
        f"[INFO] quality[aux] start: images={total_images}, action={action}, detector={detector}, "
        f"min_match_iou={min_match_iou}",
        flush=True,
    )
    rows: list[dict] = []
    keep_flags = {id(r): True for r in records}

    for idx, ((source, file_name), image_rows) in enumerate(grouped.items(), start=1):
        if log_every_images > 0 and (idx == 1 or idx % log_every_images == 0):
            dt = perf_counter() - t0
            print(
                f"[INFO] quality[aux] progress: {idx}/{total_images} ({dt:.1f}s)",
                flush=True,
            )
        image_path = source_image_index.get(source, {}).get(file_name.lower())
        if image_path is None:
            for r in image_rows:
                row = {
                    "source": source,
                    "file_name": file_name,
                    "source_json": str(r.get("source_json", "")),
                    "category_id": r.get("category_id", ""),
                    "detector": detector,
                    "pred_count": 0,
                    "max_iou": 0.0,
                    "status": "image_not_found",
                    "is_suspect": 1,
                }
                rows.append(row)
                if action in {"exclude", "drop"}:
                    keep_flags[id(r)] = False
            continue

        try:
            with Image.open(image_path) as im:
                rgb = np.array(im.convert("RGB"))
        except Exception:
            for r in image_rows:
                row = {
                    "source": source,
                    "file_name": file_name,
                    "source_json": str(r.get("source_json", "")),
                    "category_id": r.get("category_id", ""),
                    "detector": detector,
                    "pred_count": 0,
                    "max_iou": 0.0,
                    "status": "image_open_failed",
                    "is_suspect": 1,
                }
                rows.append(row)
                if action in {"exclude", "drop"}:
                    keep_flags[id(r)] = False
            continue

        pred_boxes: list[tuple[float, float, float, float]] = []
        if detector == "yolo" and yolo_model is not None:
            try:
                conf = float(qa_cfg.get("yolo_conf", 0.05))
                iou_thr = float(qa_cfg.get("yolo_iou", 0.5))
                imgsz = int(qa_cfg.get("yolo_imgsz", 960))
                pred = yolo_model.predict(rgb, conf=conf, iou=iou_thr, imgsz=imgsz, verbose=False, max_det=32)
                if pred and pred[0].boxes is not None:
                    arr = pred[0].boxes.xyxy.cpu().numpy()
                    for b in arr:
                        pred_boxes.append((float(b[0]), float(b[1]), float(b[2]), float(b[3])))
            except Exception:
                pred_boxes = []
        if detector != "yolo" or yolo_model is None:
            pred_boxes = _detect_boxes_cv_contour(
                rgb,
                min_pred_area_ratio=min_pred_area_ratio,
                max_pred_area_ratio=max_pred_area_ratio,
                min_aspect=min_aspect,
                max_aspect=max_aspect,
            )

        for r in image_rows:
            x = _safe_float(r.get("bbox_x", 0.0))
            y = _safe_float(r.get("bbox_y", 0.0))
            bw = _safe_float(r.get("bbox_w", 0.0))
            bh = _safe_float(r.get("bbox_h", 0.0))
            label_xyxy = _xywh_to_xyxy(x, y, bw, bh)
            best_iou = 0.0
            for pb in pred_boxes:
                iou = _bbox_iou_xyxy(label_xyxy, pb)
                if iou > best_iou:
                    best_iou = iou

            is_suspect = int(best_iou < min_match_iou)
            status = "ok" if is_suspect == 0 else "low_iou"
            row = {
                "source": source,
                "file_name": file_name,
                "source_json": str(r.get("source_json", "")),
                "category_id": r.get("category_id", ""),
                "detector": detector,
                "pred_count": len(pred_boxes),
                "max_iou": round(float(best_iou), 6),
                "status": status,
                "is_suspect": is_suspect,
                "threshold_iou": min_match_iou,
            }
            rows.append(row)

            if is_suspect and action in {"exclude", "drop"}:
                keep_flags[id(r)] = False
                excluded = {
                    "source": source,
                    "file_name": file_name,
                    "source_json": str(r.get("source_json", "")),
                    "reason_code": "aux_detector_low_iou_suspect",
                    "detail": f"max_iou={row['max_iou']}",
                }
                logs["excluded_rows"].append(excluded)
                if source == "external" and "excluded_rows_external" in logs:
                    logs["excluded_rows_external"].append(excluded)

    dt = perf_counter() - t0
    suspect_rows = sum(1 for r in rows if int(r.get("is_suspect", 0)) == 1)
    print(
        f"[INFO] quality[aux] done: rows={len(rows)}, suspect_rows={suspect_rows} ({dt:.1f}s)",
        flush=True,
    )
    audit_logs[audit_file].extend(rows)
    filtered = [r for r in records if keep_flags.get(id(r), True)]
    return AuxAuditResult(filtered_records=filtered, audit_rows=rows)
