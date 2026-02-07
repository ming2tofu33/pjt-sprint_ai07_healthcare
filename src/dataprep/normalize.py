from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional, Tuple

from PIL import Image


def _cast_bbox_value(value: Any, cast_type: str) -> Optional[float]:
    """bbox 원소 1개를 숫자로 캐스팅한다. 실패하면 None을 반환한다."""
    try:
        if cast_type == "int":
            return float(int(float(value)))
        return float(value)
    except Exception:
        return None


def _clip_bbox_xywh(x: float, y: float, w: float, h: float, width: float, height: float) -> Tuple[float, float, float, float]:
    """xywh 박스를 이미지 경계(0~width/height) 안으로 잘라낸다."""
    x2 = x + w
    y2 = y + h
    x1c = min(max(x, 0.0), width)
    y1c = min(max(y, 0.0), height)
    x2c = min(max(x2, 0.0), width)
    y2c = min(max(y2, 0.0), height)
    return x1c, y1c, x2c - x1c, y2c - y1c


def _extract_k_code_id(value: Any) -> Optional[str]:
    """문자열에서 K-###### 패턴을 찾아 숫자 ID 문자열로 반환한다."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    m = re.search(r"K-(\d{6})", text)
    if not m:
        return None
    return str(int(m.group(1)))


def _safe_int(value: Any) -> Optional[int]:
    """정수 변환 보조 함수. 변환 불가 시 None."""
    try:
        return int(value)
    except Exception:
        return None


def _normalize_file_name_set(values: Any) -> set[str]:
    """수동 제외 파일명 리스트를 소문자 set으로 정규화한다."""
    if not isinstance(values, list):
        return set()
    out: set[str] = set()
    for item in values:
        if not isinstance(item, str):
            continue
        key = item.strip().lower()
        if key:
            out.add(key)
    return out


def _manual_bbox_index(rule: dict) -> Optional[int]:
    """수동 bbox 수정 룰에서 수정할 좌표 인덱스(x/y/w/h)를 해석한다."""
    idx = rule.get("index")
    if idx is not None:
        try:
            i = int(idx)
            if 0 <= i <= 3:
                return i
        except Exception:
            return None
        return None

    field = str(rule.get("field", "")).strip().lower()
    if field in {"x", "bbox_x"}:
        return 0
    if field in {"y", "bbox_y"}:
        return 1
    if field in {"w", "bbox_w", "width"}:
        return 2
    if field in {"h", "bbox_h", "height"}:
        return 3
    return None


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
    단일 JSON 라벨을 df_clean 1행 레코드로 정규화한다.

    입력 계약:
    - `data`는 파싱된 원본 JSON이며, 정규화 과정에서 in-place로 일부 값이 보정될 수 있다.
    - `image_index`/`image_duplicates`는 파일명 소문자 기준 이미지 조회 맵이다.
    - `config`/`external_cfg`는 검증, 캐스팅, OOB 처리, 카테고리 동기화 정책을 결정한다.
    - `logs`는 append-only 구조이며 다음 키를 포함해야 한다.
      `excluded_rows`, `fixes_bbox`, `excluded_rows_external`, `fixes_bbox_external`
    - `audit`는 append-only 구조이며 audit 파일명 키(`audit_bad_bbox.csv` 등)를 사용한다.

    출력 계약:
    - 성공: `(record_dict, normalized_json_dict)` 반환 (`record_dict`는 df_clean 스키마)
    - 실패: `(None, None)` 반환. 동시에 제외/수정/audit 로그를 누락 없이 append한다.
    """

    def _log_excluded(row: dict) -> None:
        # 공통 제외 로그 + 외부 데이터 전용 제외 로그를 함께 유지한다.
        logs["excluded_rows"].append(row)
        if source == "external" and "excluded_rows_external" in logs:
            logs["excluded_rows_external"].append(row)

    def _log_fix(row: dict) -> None:
        # 공통 수정 로그 + 외부 데이터 전용 수정 로그를 함께 유지한다.
        logs["fixes_bbox"].append(row)
        if source == "external" and "fixes_bbox_external" in logs:
            logs["fixes_bbox_external"].append(row)

    label_contract = config.get("label_contract", {})
    enforce_split = bool(label_contract.get("split_coco_json_one_object", True))

    images = data.get("images") or []
    annotations = data.get("annotations") or []
    categories = data.get("categories") or []

    # 1) JSON 구조 검증
    # train은 split-coco(1 image/1 ann/1 cat) 계약을 기본으로 강제하고,
    # external은 external_cfg의 singleton 계약을 따르게 한다.
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

    # 2) 수동 제외 파일 적용
    # 이미 라벨 불완전으로 알려진 파일은 early return으로 학습 데이터에서 제외한다.
    manual_cfg = config.get("manual_overrides", {})
    excluded_file_names = _normalize_file_name_set(manual_cfg.get("exclude_file_names", []))
    if file_key in excluded_file_names:
        reason_code = str(manual_cfg.get("exclude_reason_code", "manual_excluded_file_name"))
        detail = str(manual_cfg.get("exclude_reason_detail", "known_incomplete_labels"))
        _log_excluded(
            {
                "source": source,
                "file_name": file_name,
                "source_json": str(source_json),
                "reason_code": reason_code,
                "detail": detail,
            }
        )
        audit["audit_missing_labels.csv"].append({"source_json": str(source_json), "reason": reason_code})
        return None, None

    # Ambiguous image names in same source
    # 동일 소스 내 같은 파일명이 여러 경로에 존재하면 이미지-라벨 매핑이 불안정하므로 제외.
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

    # 3) 이미지 존재성 검증
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

    # 4) 이미지 크기 동기화
    # 외부 데이터의 width/height 신뢰도가 낮을 수 있어, 설정 시 실제 이미지 크기로 덮어쓴다.
    # 반복 열기 비용을 줄이기 위해 image_size_cache를 사용한다.
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

    # 5) 외부 카테고리 ID 정렬
    # train 기준 canonical category_id로 강제 정렬하여 클래스 체계를 통일한다.
    if source == "external" and external_cfg:
        mapping_cfg = external_cfg.get("category_id_mapping", {})
        if mapping_cfg.get("enabled", False):
            key = mapping_cfg.get("mapping_key", "dl_idx")
            dl_idx = img0.get(key)
            dl_idx = "" if dl_idx is None else str(dl_idx).strip()
            mapped_id = None
            mapped_lookup_key = dl_idx
            fallback_key = None

            # 외부 dl_idx가 어긋난 케이스를 대비해 K-code 필드에서 fallback key를 먼저 탐색한다.
            fallback_fields = mapping_cfg.get("fallback_fields", ["dl_mapping_code", "drug_N"])
            if not isinstance(fallback_fields, list):
                fallback_fields = ["dl_mapping_code", "drug_N"]
            for field in fallback_fields:
                if not isinstance(field, str):
                    continue
                candidate = _extract_k_code_id(img0.get(field))
                if candidate is not None:
                    fallback_key = candidate
                    break

            if dl_idx != "" and mapping_id is not None and dl_idx in mapping_id:
                mapped_id = mapping_id[dl_idx]
                mapped_lookup_key = dl_idx
            elif fallback_key is not None and mapping_id is not None and fallback_key in mapping_id:
                mapped_id = mapping_id[fallback_key]
                mapped_lookup_key = fallback_key
            else:
                on_unmapped = str(mapping_cfg.get("on_unmapped", "exclude")).lower()
                if on_unmapped == "use_dl_idx":
                    if fallback_key is not None and fallback_key.isdigit():
                        mapped_id = int(fallback_key)
                        mapped_lookup_key = fallback_key
                    elif dl_idx.isdigit():
                        mapped_id = int(dl_idx)
                        mapped_lookup_key = dl_idx

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

            # 매핑 성공 시 annotation/category를 함께 맞춰 JSON 내부 정합성을 유지한다.
            ann0["category_id"] = mapped_id
            if external_cfg.get("alignment", {}).get("categories_sync", {}).get(
                "set_categories_id_from_annotation", True
            ):
                cat0["id"] = mapped_id
            if external_cfg.get("alignment", {}).get("categories_sync", {}).get(
                "set_categories_name_from_train", True
            ):
                if mapping_name and mapped_lookup_key in mapping_name:
                    cat0["name"] = mapping_name[mapped_lookup_key]

    # 6) bbox 검증 및 캐스팅
    # 형식/길이/수치 변환 실패를 단계적으로 제거하여 하류 학습 에러를 방지한다.
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

    # 7) 수동 bbox 오탈자 수정
    # 알려진 라벨 오탈자는 룰 기반으로 1회 보정하고 fixes 로그에 남긴다.
    manual_bbox_fixes = manual_cfg.get("bbox_value_fixes", [])
    if isinstance(manual_bbox_fixes, list) and manual_bbox_fixes:
        ann_cat_id_int = _safe_int(ann0.get("category_id"))
        for rule in manual_bbox_fixes:
            if not isinstance(rule, dict):
                continue

            rule_source = str(rule.get("source", "")).strip().lower()
            if rule_source and rule_source != source:
                continue

            rule_file_name = str(rule.get("file_name", "")).strip().lower()
            if rule_file_name and rule_file_name != file_key:
                continue

            if "category_id" in rule:
                rule_cat = _safe_int(rule.get("category_id"))
                if rule_cat is None or ann_cat_id_int != rule_cat:
                    continue

            idx = _manual_bbox_index(rule)
            if idx is None:
                continue

            new_value = _cast_bbox_value(rule.get("new"), cast_type)
            if new_value is None:
                continue

            old_value = _cast_bbox_value(rule.get("old"), cast_type)
            if old_value is not None and float(casted[idx]) != float(old_value):
                continue

            old_bbox = [float(v) for v in casted]
            casted[idx] = float(new_value)
            _log_fix(
                {
                    "source": source,
                    "file_name": file_name,
                    "source_json": str(source_json),
                    "old_bbox": json.dumps(old_bbox),
                    "new_bbox": json.dumps(casted),
                    "reason_code": str(rule.get("reason_code", "manual_bbox_fix")),
                }
            )
            break

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

    # 8) OOB(out-of-bounds) 정책 적용
    # - exclude: 경계를 벗어난 박스는 제거
    # - clip: 경계로 자른 뒤 면적이 0이면 제거
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

    # category 동기화 옵션이 켜져 있으면 categories.id도 annotation 기준으로 맞춘다.
    if source == "external" and external_cfg:
        if external_cfg.get("alignment", {}).get("categories_sync", {}).get("set_categories_id_from_annotation", True):
            cat0["id"] = cat_id

    bbox_area = float(w * h)
    bbox_area_ratio = bbox_area / (width * height) if width > 0 and height > 0 else 0.0
    bbox_aspect = (w / h) if h > 0 else 0.0

    # 9) df_clean 스키마로 레코드 조립
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
