from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Tuple

from PIL import Image


def _cast_bbox_value(value: Any, cast_type: str) -> Optional[float]:
    try:
        if cast_type == "int":
            return float(int(float(value)))
        return float(value)
    except Exception:
        return None


def _clip_bbox_xywh(x: float, y: float, w: float, h: float, width: float, height: float) -> Tuple[float, float, float, float]:
    x2 = x + w
    y2 = y + h
    x1c = min(max(x, 0.0), width)
    y1c = min(max(y, 0.0), height)
    x2c = min(max(x2, 0.0), width)
    y2c = min(max(y2, 0.0), height)
    return x1c, y1c, x2c - x1c, y2c - y1c


def normalize_record(
    *,
    source: str,
    data: dict,
    source_json: Path,
    image_index: dict[str, Path],
    image_duplicates: dict[str, list[Path]],
    image_size_cache: Optional[dict[str, tuple[int, int]]] = None,
    config: dict,
    mapping_id: Optional[dict[str, int]] = None,
    mapping_name: Optional[dict[str, str]] = None,
    logs: dict[str, list[dict]],
    audit: dict[str, list[dict]],
    external_cfg: Optional[dict] = None,
) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Normalize one JSON into a clean row.

    Input contract:
    - `data` is original parsed JSON dict (mutated in-place when normalization applies).
    - `image_index`/`image_duplicates` are filename-lowercase based lookup maps.
    - `config`/`external_cfg` determine validation, cast, OOB, and category-sync behavior.
    - `logs` is append-only and must include keys:
      `excluded_rows`, `fixes_bbox`, `excluded_rows_external`, `fixes_bbox_external`.
    - `audit` is append-only and keyed by audit file names (e.g. `audit_bad_bbox.csv`).

    Output contract:
    - Success: `(record_dict, normalized_json_dict)` where `record_dict` matches df_clean row schema.
    - Failure: `(None, None)` after appending exactly the same excluded/fix/audit rows
      (field names, reason_code strings, and payload formats are unchanged).
    """

    def _log_excluded(row: dict) -> None:
        logs["excluded_rows"].append(row)
        if source == "external" and "excluded_rows_external" in logs:
            logs["excluded_rows_external"].append(row)

    def _log_fix(row: dict) -> None:
        logs["fixes_bbox"].append(row)
        if source == "external" and "fixes_bbox_external" in logs:
            logs["fixes_bbox_external"].append(row)

    label_contract = config.get("label_contract", {})
    enforce_split = bool(label_contract.get("split_coco_json_one_object", True))

    images = data.get("images") or []
    annotations = data.get("annotations") or []
    categories = data.get("categories") or []

    if source == "external" and external_cfg:
        enforce_cfg = external_cfg.get("alignment", {}).get("enforce_singletons", {})
        exp_images = int(enforce_cfg.get("images_len", 1))
        exp_anns = int(enforce_cfg.get("annotations_len", 1))
        exp_cats = int(enforce_cfg.get("categories_len", 1))
        if len(images) != exp_images or len(annotations) != exp_anns or len(categories) != exp_cats:
            _log_excluded(
                {
                    "source": source,
                    "file_name": "",
                    "source_json": str(source_json),
                    "reason_code": "invalid_structure",
                    "detail": f"images={len(images)}, annotations={len(annotations)}, categories={len(categories)}",
                }
            )
            audit["audit_missing_labels.csv"].append({"source_json": str(source_json), "reason": "invalid_structure"})
            return None, None
    elif enforce_split:
        if len(images) != 1 or len(annotations) != 1 or len(categories) != 1:
            _log_excluded(
                {
                    "source": source,
                    "file_name": "",
                    "source_json": str(source_json),
                    "reason_code": "invalid_structure",
                    "detail": f"images={len(images)}, annotations={len(annotations)}, categories={len(categories)}",
                }
            )
            audit["audit_missing_labels.csv"].append({"source_json": str(source_json), "reason": "invalid_structure"})
            return None, None

    img0 = images[0]
    ann0 = annotations[0]
    cat0 = categories[0]

    file_name = img0.get("file_name")
    if not isinstance(file_name, str) or file_name.strip() == "":
        _log_excluded(
            {
                "source": source,
                "file_name": "",
                "source_json": str(source_json),
                "reason_code": "missing_file_name",
                "detail": "",
            }
        )
        audit["audit_missing_labels.csv"].append({"source_json": str(source_json), "reason": "missing_file_name"})
        return None, None
    file_name = file_name.strip()
    file_key = file_name.lower()

    # Ambiguous image names in same source
    if file_key in image_duplicates:
        _log_excluded(
            {
                "source": source,
                "file_name": file_name,
                "source_json": str(source_json),
                "reason_code": "ambiguous_image_name",
                "detail": f"dup_paths={len(image_duplicates[file_key])}",
            }
        )
        audit["audit_missing_images.csv"].append({"file_name": file_name, "reason": "ambiguous_name"})
        return None, None

    image_path = image_index.get(file_key)
    image_meta_cfg = (external_cfg or {}).get("alignment", {}).get("image_meta", {}) if source == "external" else {}
    integrity_cfg = config.get("integrity", {}) if source == "train" else {}
    verify_image = bool(image_meta_cfg.get("verify_image_exists", False) or integrity_cfg.get("drop_if_image_missing", False))

    if verify_image and image_path is None:
        _log_excluded(
            {
                "source": source,
                "file_name": file_name,
                "source_json": str(source_json),
                "reason_code": "image_missing",
                "detail": "",
            }
        )
        audit["audit_missing_images.csv"].append({"file_name": file_name, "reason": "not_found"})
        return None, None

    width = img0.get("width")
    height = img0.get("height")

    # If external alignment requires overwrite, use actual image size
    overwrite_wh = bool(image_meta_cfg.get("overwrite_width_height_from_image", False))
    if image_path is not None and (overwrite_wh or width is None or height is None):
        try:
            cache_key = str(image_path)
            if image_size_cache is not None and cache_key in image_size_cache:
                width, height = image_size_cache[cache_key]
            else:
                with Image.open(image_path) as im:
                    width, height = im.size
                if image_size_cache is not None:
                    image_size_cache[cache_key] = (int(width), int(height))
        except Exception:
            _log_excluded(
                {
                    "source": source,
                    "file_name": file_name,
                    "source_json": str(source_json),
                    "reason_code": "image_open_failed",
                    "detail": "",
                }
            )
            audit["audit_missing_images.csv"].append({"file_name": file_name, "reason": "open_failed"})
            return None, None
        if overwrite_wh:
            img0["width"] = width
            img0["height"] = height

    if width is None or height is None:
        _log_excluded(
            {
                "source": source,
                "file_name": file_name,
                "source_json": str(source_json),
                "reason_code": "missing_image_size",
                "detail": "",
            }
        )
        audit["audit_missing_images.csv"].append({"file_name": file_name, "reason": "missing_size"})
        return None, None

    try:
        width = float(width)
        height = float(height)
    except Exception:
        _log_excluded(
            {
                "source": source,
                "file_name": file_name,
                "source_json": str(source_json),
                "reason_code": "invalid_image_size",
                "detail": f"width={width}, height={height}",
            }
        )
        return None, None

    # External category mapping (dl_idx -> canonical id)
    if source == "external" and external_cfg:
        mapping_cfg = external_cfg.get("category_id_mapping", {})
        if mapping_cfg.get("enabled", False):
            key = mapping_cfg.get("mapping_key", "dl_idx")
            dl_idx = img0.get(key)
            dl_idx = "" if dl_idx is None else str(dl_idx).strip()
            mapped_id = None
            if dl_idx != "" and mapping_id is not None and dl_idx in mapping_id:
               mapped_id = mapping_id[dl_idx]
            else:
                # YAML 스위치에 따른 fallback 처리
                on_unmapped = str(mapping_cfg.get("on_unmapped", "exclude")).lower()
                if on_unmapped == "use_dl_idx" and dl_idx.isdigit():
                    mapped_id = int(dl_idx)

            if mapped_id is None:
                _log_excluded(
                    {
                        "source": source,
                        "file_name": file_name,
                        "source_json": str(source_json),
                        "reason_code": "unmapped_dl_idx",
                        "detail": f"{key}={dl_idx}",
                    }
                )
                audit["audit_unmapped_external.csv"].append(
                    {"file_name": file_name, "source_json": str(source_json), "dl_idx": dl_idx}
                )

                return None, None
            ann0["category_id"] = mapped_id
            if external_cfg.get("alignment", {}).get("categories_sync", {}).get(
                "set_categories_id_from_annotation", True
            ):
                cat0["id"] = mapped_id
            if external_cfg.get("alignment", {}).get("categories_sync", {}).get(
                "set_categories_name_from_train", True
            ):
                if mapping_name and dl_idx in mapping_name:
                    cat0["name"] = mapping_name[dl_idx]

    # bbox validation / casting
    validation_cfg = config.get("validation", {})
    bbox_cfg = (
        external_cfg.get("alignment", {}).get("bbox", {}) if source == "external" and external_cfg else validation_cfg
    )

    bbox = ann0.get("bbox")
    if bbox is None and bbox_cfg.get("drop_missing_bbox", True):
        _log_excluded(
            {
                "source": source,
                "file_name": file_name,
                "source_json": str(source_json),
                "reason_code": "missing_bbox",
                "detail": "",
            }
        )
        audit["audit_bad_bbox.csv"].append({"file_name": file_name, "reason": "missing_bbox"})
        return None, None

    if not isinstance(bbox, list) or len(bbox) != 4:
        if bbox_cfg.get("drop_invalid_bbox_len", True):
            _log_excluded(
                {
                    "source": source,
                    "file_name": file_name,
                    "source_json": str(source_json),
                    "reason_code": "invalid_bbox_len",
                    "detail": f"bbox={bbox}",
                }
            )
            audit["audit_bad_bbox.csv"].append({"file_name": file_name, "reason": "invalid_bbox_len"})
            return None, None
        return None, None

    cast_type = bbox_cfg.get("cast", validation_cfg.get("bbox_cast", "float"))
    casted = []
    for v in bbox:
        fv = _cast_bbox_value(v, cast_type)
        if fv is None:
            if bbox_cfg.get("drop_non_numeric_bbox", True):
                _log_excluded(
                    {
                        "source": source,
                        "file_name": file_name,
                        "source_json": str(source_json),
                        "reason_code": "non_numeric_bbox",
                        "detail": f"bbox={bbox}",
                    }
                )
                audit["audit_bad_bbox.csv"].append({"file_name": file_name, "reason": "non_numeric_bbox"})
                return None, None
            return None, None
        casted.append(float(fv))

    x, y, w, h = casted
    if bbox_cfg.get("drop_non_positive_wh", True) and (w <= 0 or h <= 0):
        _log_excluded(
            {
                "source": source,
                "file_name": file_name,
                "source_json": str(source_json),
                "reason_code": "non_positive_wh",
                "detail": f"bbox={bbox}",
            }
        )
        audit["audit_bad_bbox.csv"].append({"file_name": file_name, "reason": "non_positive_wh"})
        return None, None

    # OOB handling
    oob_cfg = (external_cfg.get("alignment", {}).get("oob", {}) if source == "external" and external_cfg else config.get("oob", {}))
    is_oob = x < 0 or y < 0 or (x + w) > width or (y + h) > height
    if is_oob:
        policy = oob_cfg.get("policy", "clip")
        if policy == "exclude":
            _log_excluded(
                {
                    "source": source,
                    "file_name": file_name,
                    "source_json": str(source_json),
                    "reason_code": "oob_excluded",
                    "detail": f"bbox={casted}",
                }
            )
            audit["audit_bad_bbox.csv"].append({"file_name": file_name, "reason": "oob_excluded"})
            return None, None
        if policy == "clip":
            new_x, new_y, new_w, new_h = _clip_bbox_xywh(x, y, w, h, width, height)
            if oob_cfg.get("drop_if_clipped_to_zero", True) and (new_w <= 0 or new_h <= 0):
                _log_excluded(
                    {
                        "source": source,
                        "file_name": file_name,
                        "source_json": str(source_json),
                        "reason_code": "oob_clipped_to_zero",
                        "detail": f"bbox={casted}",
                    }
                )
                audit["audit_bad_bbox.csv"].append({"file_name": file_name, "reason": "oob_clipped_to_zero"})
                return None, None
            _log_fix(
                {
                    "source": source,
                    "file_name": file_name,
                    "source_json": str(source_json),
                    "old_bbox": json.dumps([x, y, w, h]),
                    "new_bbox": json.dumps([new_x, new_y, new_w, new_h]),
                    "reason_code": "oob_clip",
                }
            )
            casted = [new_x, new_y, new_w, new_h]
            x, y, w, h = casted

    ann0["bbox"] = casted
    if "area" not in ann0:
        ann0["area"] = float(w * h)

    cat_id = ann0.get("category_id")
    try:
        cat_id = int(cat_id)
    except Exception:
        _log_excluded(
            {
                "source": source,
                "file_name": file_name,
                "source_json": str(source_json),
                "reason_code": "invalid_category_id",
                "detail": f"category_id={cat_id}",
            }
        )
        return None, None

    # Keep categories in sync when requested
    if source == "external" and external_cfg:
        if external_cfg.get("alignment", {}).get("categories_sync", {}).get("set_categories_id_from_annotation", True):
            cat0["id"] = cat_id

    bbox_area = float(w * h)
    bbox_area_ratio = bbox_area / (width * height) if width > 0 and height > 0 else 0.0
    bbox_aspect = (w / h) if h > 0 else 0.0

    record = {
        "file_name": file_name,
        "source_json": str(source_json),
        "width": width,
        "height": height,
        "category_id": cat_id,
        "bbox": json.dumps([x, y, w, h]),
        "bbox_x": x,
        "bbox_y": y,
        "bbox_w": w,
        "bbox_h": h,
        "bbox_area": bbox_area,
        "bbox_area_ratio": bbox_area_ratio,
        "bbox_aspect": bbox_aspect,
        "is_oob": bool(is_oob),
        "is_bad_bbox": False,
        "is_exact_dup": False,
        "source": source,
    }

    return record, data
