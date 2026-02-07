from __future__ import annotations

from pathlib import Path
from typing import Optional


def _safe_float(value) -> float | None:
    """부동소수 변환 보조 함수. 실패 시 None."""
    try:
        return float(value)
    except Exception:
        return None


def _bbox_iou_xywh(a: dict, b: dict) -> float:
    """xywh 박스 2개의 IoU를 계산한다. 입력이 비정상이면 0.0."""
    ax = _safe_float(a.get("bbox_x"))
    ay = _safe_float(a.get("bbox_y"))
    aw = _safe_float(a.get("bbox_w"))
    ah = _safe_float(a.get("bbox_h"))
    bx = _safe_float(b.get("bbox_x"))
    by = _safe_float(b.get("bbox_y"))
    bw = _safe_float(b.get("bbox_w"))
    bh = _safe_float(b.get("bbox_h"))

    if None in (ax, ay, aw, ah, bx, by, bw, bh):
        return 0.0
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    union = (aw * ah) + (bw * bh) - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def add_train_dedup_key(record: dict, dedup_key_fields: list, train_dedup_keys: set[tuple]) -> None:
    """train 레코드의 exact dedup 키를 set에 추가한다."""
    if dedup_key_fields:
        train_dedup_keys.add(tuple(record.get(k) for k in dedup_key_fields))


def should_keep_external_record_after_dedup(
    record: dict,
    *,
    dedup_against_train: bool,
    dedup_key_fields: list,
    train_dedup_keys: set[tuple],
    logs: dict[str, list[dict]],
    audit_logs: dict[str, list[dict]],
) -> bool:
    """external 레코드가 train과 exact 중복이면 제외하고 False를 반환한다."""
    if dedup_against_train and dedup_key_fields:
        key = tuple(record.get(k) for k in dedup_key_fields)
        if key in train_dedup_keys:
            logs["excluded_rows"].append(
                {
                    "source": "external",
                    "file_name": record["file_name"],
                    "source_json": record["source_json"],
                    "reason_code": "dedup_against_train",
                    "detail": "",
                }
            )
            if "audit_dup_exact.csv" in audit_logs:
                audit_logs["audit_dup_exact.csv"].append({"file_name": record["file_name"], "reason": "dedup_against_train"})
            return False
    return True


def dedup_exact_records(records: list[dict], dedup_cfg: dict, logs: dict[str, list[dict]], audit_logs: dict[str, list[dict]]) -> list[dict]:
    """exact key 중복 제거. 선행 레코드를 유지하고 후행 중복을 제외한다."""
    if dedup_cfg.get("enabled", False):
        key_fields = dedup_cfg.get("key", [])
        seen: set[tuple] = set()
        deduped: list[dict] = []
        for r in records:
            key = tuple(r.get(k) for k in key_fields)
            if key in seen:
                logs["excluded_rows"].append(
                    {
                        "source": r["source"],
                        "file_name": r["file_name"],
                        "source_json": r["source_json"],
                        "reason_code": "dedup_exact",
                        "detail": "",
                    }
                )
                if "audit_dup_exact.csv" in audit_logs:
                    audit_logs["audit_dup_exact.csv"].append({"file_name": r["file_name"], "reason": "dedup_exact"})
                continue
            seen.add(key)
            r["is_exact_dup"] = False
            deduped.append(r)
        return deduped
    return records


def dedup_iou_records(records: list[dict], iou_cfg: dict, logs: dict[str, list[dict]], audit_logs: dict[str, list[dict]]) -> list[dict]:
    """
    파일 단위(옵션: 카테고리/소스까지)로 IoU 중복을 검사한다.
    action이 exclude 계열이면 후행 박스를 제거하고, 아니면 감사 로그만 남긴다.
    """
    if not iou_cfg.get("enabled", False):
        return records

    threshold = float(iou_cfg.get("threshold", 0.9))
    action = str(iou_cfg.get("action", "audit_only")).lower()
    same_category_only = bool(iou_cfg.get("same_category_only", True))
    same_source_only = bool(iou_cfg.get("same_source_only", True))

    kept: list[dict] = []
    grouped_kept: dict[tuple, list[dict]] = {}

    for r in records:
        group_key_parts = [r.get("file_name")]
        if same_category_only:
            group_key_parts.append(r.get("category_id"))
        if same_source_only:
            group_key_parts.append(r.get("source"))
        group_key = tuple(group_key_parts)

        candidates = grouped_kept.get(group_key, [])
        overlap = None
        for k in candidates:
            iou = _bbox_iou_xywh(r, k)
            if iou >= threshold:
                overlap = (k, iou)
                if "audit_iou_pairs.csv" in audit_logs:
                    audit_logs["audit_iou_pairs.csv"].append(
                        {
                            "file_name": r.get("file_name", ""),
                            "category_id": r.get("category_id", ""),
                            "source_json_a": k.get("source_json", ""),
                            "source_json_b": r.get("source_json", ""),
                            "iou": round(float(iou), 6),
                            "reason": "iou_overlap",
                        }
                    )
                break

        if overlap is not None and action in {"exclude", "drop", "exclude_later", "drop_later"}:
            _, overlap_iou = overlap
            logs["excluded_rows"].append(
                {
                    "source": r["source"],
                    "file_name": r["file_name"],
                    "source_json": r["source_json"],
                    "reason_code": "dedup_iou_overlap",
                    "detail": f"iou={overlap_iou:.6f}, threshold={threshold}",
                }
            )
            if "audit_dup_near.csv" in audit_logs:
                audit_logs["audit_dup_near.csv"].append({"file_name": r["file_name"], "reason": "dedup_iou_overlap"})
            continue

        grouped_kept.setdefault(group_key, []).append(r)
        kept.append(r)

    return kept


def _is_copy_like_name(name: str, patterns: list[str]) -> bool:
    low = str(name).lower()
    return any(str(p).lower() in low for p in patterns if str(p).strip())


def filter_external_copy_json_records(
    records: list[dict],
    config: dict,
    logs: dict[str, list[dict]],
    audit_logs: dict[str, list[dict]],
) -> list[dict]:
    """
    external(source=external) 레코드 중 source_json 파일명이 copy 패턴이면 제외한다.
    """
    ext_cfg = config.get("external_data", {})
    qf_cfg = ext_cfg.get("quality_filters", {})
    copy_cfg = qf_cfg.get("exclude_copy_json", {})
    enabled = bool(copy_cfg.get("enabled", True))
    if not enabled:
        return records

    patterns = copy_cfg.get("patterns", [" copy", "(copy", "_copy"])
    if not isinstance(patterns, list) or not patterns:
        patterns = [" copy", "(copy", "_copy"]

    filtered: list[dict] = []
    seen_for_audit: set[tuple[str, str]] = set()
    for r in records:
        if r.get("source") != "external":
            filtered.append(r)
            continue

        source_json = str(r.get("source_json", ""))
        source_name = Path(source_json).name
        if not _is_copy_like_name(source_name, patterns):
            filtered.append(r)
            continue

        excluded_row = {
            "source": "external",
            "file_name": str(r.get("file_name", "")),
            "source_json": source_json,
            "reason_code": "external_copy_json_excluded",
            "detail": f"source_json_name={source_name}",
        }
        logs["excluded_rows"].append(excluded_row)
        if "excluded_rows_external" in logs:
            logs["excluded_rows_external"].append(excluded_row)

        if "audit_external_copy_json.csv" in audit_logs:
            k = (str(r.get("file_name", "")), source_json)
            if k not in seen_for_audit:
                audit_logs["audit_external_copy_json.csv"].append(
                    {
                        "file_name": str(r.get("file_name", "")),
                        "source_json": source_json,
                        "reason": "external_copy_json_excluded",
                    }
                )
                seen_for_audit.add(k)
        elif "audit_missing_labels.csv" in audit_logs:
            audit_logs["audit_missing_labels.csv"].append(
                {"source_json": source_json, "reason": "external_copy_json_excluded"}
            )

    return filtered


def filter_external_bbox_category_conflicts(
    records: list[dict],
    config: dict,
    logs: dict[str, list[dict]],
    audit_logs: dict[str, list[dict]],
) -> list[dict]:
    """
    external 데이터에서 동일 bbox(file_name+bbox)에 서로 다른 category_id가 붙은 충돌 그룹을 제외한다.
    """
    ext_cfg = config.get("external_data", {})
    qf_cfg = ext_cfg.get("quality_filters", {})
    conflict_cfg = qf_cfg.get("exclude_bbox_conflict", {})
    enabled = bool(conflict_cfg.get("enabled", True))
    if not enabled:
        return records

    round_digits = int(conflict_cfg.get("bbox_round_digits", 4))

    bbox_to_cids: dict[tuple, set[int]] = {}
    bbox_to_rows: dict[tuple, list[dict]] = {}

    for r in records:
        if r.get("source") != "external":
            continue
        try:
            key = (
                str(r.get("file_name", "")),
                round(float(r.get("bbox_x", 0.0)), round_digits),
                round(float(r.get("bbox_y", 0.0)), round_digits),
                round(float(r.get("bbox_w", 0.0)), round_digits),
                round(float(r.get("bbox_h", 0.0)), round_digits),
            )
            cid = int(r.get("category_id"))
        except Exception:
            continue

        bbox_to_cids.setdefault(key, set()).add(cid)
        bbox_to_rows.setdefault(key, []).append(r)

    conflict_keys = {k for k, cid_set in bbox_to_cids.items() if len(cid_set) > 1}
    if not conflict_keys:
        return records

    conflict_row_ids = {id(r) for k in conflict_keys for r in bbox_to_rows.get(k, [])}
    filtered: list[dict] = []
    for r in records:
        if id(r) not in conflict_row_ids:
            filtered.append(r)
            continue

        fn = str(r.get("file_name", ""))
        source_json = str(r.get("source_json", ""))
        try:
            key = (
                fn,
                round(float(r.get("bbox_x", 0.0)), round_digits),
                round(float(r.get("bbox_y", 0.0)), round_digits),
                round(float(r.get("bbox_w", 0.0)), round_digits),
                round(float(r.get("bbox_h", 0.0)), round_digits),
            )
            cid_set = sorted(bbox_to_cids.get(key, set()))
        except Exception:
            cid_set = []

        excluded_row = {
            "source": "external",
            "file_name": fn,
            "source_json": source_json,
            "reason_code": "external_bbox_category_conflict",
            "detail": f"conflict_category_ids={cid_set}",
        }
        logs["excluded_rows"].append(excluded_row)
        if "excluded_rows_external" in logs:
            logs["excluded_rows_external"].append(excluded_row)

    if "audit_external_bbox_conflict.csv" in audit_logs:
        for k in sorted(conflict_keys):
            fn, x, y, w, h = k
            cid_set = sorted(bbox_to_cids.get(k, set()))
            audit_logs["audit_external_bbox_conflict.csv"].append(
                {
                    "file_name": fn,
                    "bbox_x": x,
                    "bbox_y": y,
                    "bbox_w": w,
                    "bbox_h": h,
                    "category_ids": ";".join(str(x) for x in cid_set),
                    "entry_count": len(bbox_to_rows.get(k, [])),
                    "reason": "external_bbox_category_conflict",
                }
            )
    elif "audit_missing_labels.csv" in audit_logs:
        for k in sorted(conflict_keys):
            rows = bbox_to_rows.get(k, [])
            source_json = str(rows[0].get("source_json", "")) if rows else ""
            audit_logs["audit_missing_labels.csv"].append(
                {"source_json": source_json, "reason": "external_bbox_category_conflict"}
            )

    return filtered


def filter_external_images_with_any_bad_bbox(
    records: list[dict],
    config: dict,
    logs: dict[str, list[dict]],
    audit_logs: dict[str, list[dict]],
) -> list[dict]:
    """
    external 데이터에서 bbox 관련 불량(reason_code)이 1건이라도 발생한 파일은
    해당 이미지의 남아있는 모든 external 레코드를 통째로 제외한다.
    """
    ext_cfg = config.get("external_data", {})
    qf_cfg = ext_cfg.get("quality_filters", {})
    strict_cfg = qf_cfg.get("exclude_entire_image_if_any_bbox_bad", {})
    enabled = bool(strict_cfg.get("enabled", False))
    if not enabled:
        return records

    default_bbox_reasons = [
        "missing_bbox",
        "invalid_bbox_len",
        "non_numeric_bbox",
        "non_positive_wh",
        "oob_excluded",
        "oob_clipped_to_zero",
    ]
    reason_codes_raw = strict_cfg.get("reason_codes", default_bbox_reasons)
    if not isinstance(reason_codes_raw, list) or not reason_codes_raw:
        reason_codes_raw = default_bbox_reasons
    bad_reason_codes = {str(x).strip() for x in reason_codes_raw if str(x).strip()}
    if not bad_reason_codes:
        return records

    bad_reasons_by_file: dict[str, set[str]] = {}
    for row in logs.get("excluded_rows_external", []):
        fn = str(row.get("file_name", "")).strip()
        code = str(row.get("reason_code", "")).strip()
        if not fn or code not in bad_reason_codes:
            continue
        bad_reasons_by_file.setdefault(fn, set()).add(code)

    if not bad_reasons_by_file:
        return records

    filtered: list[dict] = []
    seen_file_for_audit: set[str] = set()
    for r in records:
        if r.get("source") != "external":
            filtered.append(r)
            continue

        fn = str(r.get("file_name", "")).strip()
        if fn not in bad_reasons_by_file:
            filtered.append(r)
            continue

        reason_list = sorted(bad_reasons_by_file.get(fn, set()))
        excluded_row = {
            "source": "external",
            "file_name": fn,
            "source_json": str(r.get("source_json", "")),
            "reason_code": "external_image_has_bad_bbox_record",
            "detail": f"bad_bbox_reasons={reason_list}",
        }
        logs["excluded_rows"].append(excluded_row)
        if "excluded_rows_external" in logs:
            logs["excluded_rows_external"].append(excluded_row)

        if "audit_bad_bbox.csv" in audit_logs and fn not in seen_file_for_audit:
            audit_logs["audit_bad_bbox.csv"].append(
                {
                    "file_name": fn,
                    "reason": "exclude_entire_image_if_any_bbox_bad",
                }
            )
            seen_file_for_audit.add(fn)

    return filtered


def _expected_code_count_from_file_name(file_name: str) -> Optional[int]:
    """
    파일명 앞부분(`K-000001-000002-..._`)에서 약 코드 개수를 추출한다.
    예: K-000250-000573-002483-012778_... -> 4
    """
    if not isinstance(file_name, str) or not file_name.strip():
        return None
    head = file_name.split("_", 1)[0]
    parts = head.split("-")
    if not parts:
        return None
    count = 0
    for token in parts[1:]:
        t = token.strip()
        if len(t) == 6 and t.isdigit():
            count += 1
    return count if count > 0 else None


def filter_expected4_actual3(
    records: list[dict],
    config: dict,
    logs: dict[str, list[dict]],
    audit_logs: dict[str, list[dict]],
) -> list[dict]:
    """
    파일명 약코드 4개인데 실제 객체 수가 3개인 이미지를 통째로 제외한다.
    기본적으로 train/external 모두에 적용한다.
    """
    ext_cfg = config.get("external_data", {})
    qf_cfg = ext_cfg.get("quality_filters", {})
    rule_cfg = qf_cfg.get("exclude_expected_4_codes_with_3_objects", {})
    enabled = bool(rule_cfg.get("enabled", False))
    if not enabled:
        return records

    sources_raw = rule_cfg.get("sources", ["train", "external"])
    if not isinstance(sources_raw, list) or not sources_raw:
        sources_raw = ["train", "external"]
    target_sources = {str(x).strip().lower() for x in sources_raw if str(x).strip()}
    if not target_sources:
        return records

    per_file_count: dict[tuple[str, str], int] = {}
    for r in records:
        source = str(r.get("source", "")).strip().lower()
        if source not in target_sources:
            continue
        fn = str(r.get("file_name", ""))
        key = (source, fn)
        per_file_count[key] = per_file_count.get(key, 0) + 1

    target_keys: set[tuple[str, str]] = set()
    for (source, fn), count in per_file_count.items():
        expected = _expected_code_count_from_file_name(fn)
        if expected == 4 and count == 3:
            target_keys.add((source, fn))

    if not target_keys:
        return records

    filtered: list[dict] = []
    seen_for_audit: set[tuple[str, str]] = set()
    for r in records:
        source = str(r.get("source", "")).strip().lower()
        fn = str(r.get("file_name", ""))
        key = (source, fn)
        if key not in target_keys:
            filtered.append(r)
            continue

        excluded_row = {
            "source": source,
            "file_name": fn,
            "source_json": str(r.get("source_json", "")),
            "reason_code": "expected4_actual3_excluded",
            "detail": "expected_from_file_name=4, objects_in_df=3",
        }
        logs["excluded_rows"].append(excluded_row)
        if source == "external" and "excluded_rows_external" in logs:
            logs["excluded_rows_external"].append(excluded_row)

        if "audit_external_expected4_actual3.csv" in audit_logs and key not in seen_for_audit:
            audit_logs["audit_external_expected4_actual3.csv"].append(
                {
                    "source": source,
                    "file_name": fn,
                    "expected_code_count": 4,
                    "object_count": 3,
                    "reason": "expected4_actual3_excluded",
                }
            )
            seen_for_audit.add(key)
        elif "audit_missing_labels.csv" in audit_logs and key not in seen_for_audit:
            audit_logs["audit_missing_labels.csv"].append(
                {
                    "source_json": str(r.get("source_json", "")),
                    "reason": "expected4_actual3_excluded",
                }
            )
            seen_for_audit.add(key)

    return filtered


def filter_object_count(
    records: list[dict],
    config: dict,
    logs: dict[str, list[dict]],
    audit_logs: dict[str, list[dict]],
) -> list[dict]:
    """
    이미지당 객체 수 규칙을 적용해 불량 이미지를 제거한다.
    """
    heuristic_cfg = config.get("missing_label", {}).get("heuristic", {})
    if not bool(heuristic_cfg.get("enabled", False)):
        return records

    min_objects = heuristic_cfg.get("min_objects_per_image")
    max_objects = heuristic_cfg.get("max_objects_per_image")
    valid_object_counts = heuristic_cfg.get("valid_object_counts")

    allowed_set: Optional[set[int]] = None
    if isinstance(valid_object_counts, list) and valid_object_counts:
        parsed: set[int] = set()
        for x in valid_object_counts:
            try:
                parsed.add(int(x))
            except Exception:
                continue
        if parsed:
            allowed_set = parsed

    if min_objects is None and max_objects is None and allowed_set is None:
        return records

    min_int = int(min_objects) if min_objects is not None else None
    max_int = int(max_objects) if max_objects is not None else None

    per_image_count: dict[str, int] = {}
    for r in records:
        fn = str(r.get("file_name", ""))
        per_image_count[fn] = per_image_count.get(fn, 0) + 1

    def _is_invalid(count: int) -> bool:
        if allowed_set is not None:
            return count not in allowed_set
        if min_int is not None and count < min_int:
            return True
        if max_int is not None and count > max_int:
            return True
        return False

    invalid_files = {fn for fn, c in per_image_count.items() if _is_invalid(c)}
    if not invalid_files:
        return records

    filtered: list[dict] = []
    seen_invalid_for_audit: set[str] = set()
    seen_invalid_for_missing_labels: set[str] = set()
    for r in records:
        fn = str(r.get("file_name", ""))
        if fn not in invalid_files:
            filtered.append(r)
            continue

        count = per_image_count.get(fn, 0)
        detail_parts = [f"objects={count}"]
        if allowed_set is not None:
            detail_parts.append(f"allowed={sorted(allowed_set)}")
        else:
            detail_parts.append(f"min={min_int}")
            detail_parts.append(f"max={max_int}")
        detail = ", ".join(detail_parts)

        excluded_row = {
            "source": r.get("source", ""),
            "file_name": fn,
            "source_json": r.get("source_json", ""),
            "reason_code": "invalid_object_count_per_image",
            "detail": detail,
        }
        logs["excluded_rows"].append(excluded_row)
        if r.get("source") == "external" and "excluded_rows_external" in logs:
            logs["excluded_rows_external"].append(excluded_row)

        if "audit_invalid_object_count.csv" in audit_logs and fn not in seen_invalid_for_audit:
            audit_logs["audit_invalid_object_count.csv"].append(
                {
                    "file_name": fn,
                    "source": r.get("source", ""),
                    "object_count": count,
                    "detail": detail,
                }
            )
            seen_invalid_for_audit.add(fn)
        elif "audit_missing_labels.csv" in audit_logs and fn not in seen_invalid_for_missing_labels:
            audit_logs["audit_missing_labels.csv"].append(
                {
                    "source_json": r.get("source_json", ""),
                    "reason": "invalid_object_count_per_image",
                }
            )
            seen_invalid_for_missing_labels.add(fn)

    return filtered
