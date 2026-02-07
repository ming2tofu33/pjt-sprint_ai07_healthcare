from __future__ import annotations


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
