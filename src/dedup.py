from __future__ import annotations

def add_train_dedup_key(record: dict, dedup_key_fields: list, train_dedup_keys: set[tuple]) -> None:
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
