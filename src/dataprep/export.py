from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable


def write_audit_files(metadata_dir: Path, audit_logs: dict[str, list[dict]]) -> None:
    """audit 로그를 파일별로 저장한다. 빈 로그도 파일을 생성한다."""
    for audit_name, rows in audit_logs.items():
        with (metadata_dir / audit_name).open("w", newline="", encoding="utf-8") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            else:
                f.write("")


def write_unmapped_external_audit_log(
    *,
    mapping_cfg: dict,
    base_dir: Path,
    audit_logs: dict[str, list[dict]],
    resolve_path_fn: Callable[[str, Path], Path],
) -> None:
    """외부 데이터 unmapped audit 로그를 별도 경로에 저장한다."""
    unmapped_out = mapping_cfg.get("unmapped_log_out")
    if unmapped_out:
        out_path = resolve_path_fn(unmapped_out, base_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = audit_logs.get("audit_unmapped_external.csv", [])
        with out_path.open("w", newline="", encoding="utf-8") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            else:
                f.write("")


def write_outputs(
    *,
    records: list[dict],
    logs: dict[str, list[dict]],
    audit_logs: dict[str, list[dict]],
    mapping_rows: list[dict],
    processed_dir: Path,
    metadata_dir: Path,
    outputs_cfg: dict,
    config: dict,
    base_dir: Path,
    resolve_path_fn: Callable[[str, Path], Path],
) -> None:
    """
    전처리 산출물을 디스크에 저장한다.
    - df_clean / excluded / fixes
    - audit 파일
    - external 전용 로그(설정 시)
    - category 매핑 테이블(설정 시)
    """
    df_name = outputs_cfg.get("df_clean_name", "df_clean.csv")
    excluded_name = outputs_cfg.get("excluded_rows_name", "excluded_rows.csv")
    fixes_name = outputs_cfg.get("fixes_bbox_name", "fixes_bbox.csv")

    # 1) 학습/분석 기준 테이블(df_clean)
    df_path = processed_dir / df_name
    df_path.parent.mkdir(parents=True, exist_ok=True)
    df_fields = [
        "file_name",
        "source_json",
        "width",
        "height",
        "group_id",
        "category_id",
        "bbox",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "bbox_area",
        "bbox_area_ratio",
        "bbox_aspect",
        "is_oob",
        "is_bad_bbox",
        "is_exact_dup",
        "source",
    ]
    with df_path.open("w", newline="", encoding=config.get("format", {}).get("csv_encoding", "utf-8")) as f:
        writer = csv.DictWriter(f, fieldnames=df_fields)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    # 2) 제외 로그(왜 버렸는지 추적용)
    with (metadata_dir / excluded_name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "file_name", "source_json", "reason_code", "detail"])
        writer.writeheader()
        writer.writerows(logs["excluded_rows"])

    # 3) bbox 수정 로그(원본/수정값 추적용)
    with (metadata_dir / fixes_name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["source", "file_name", "source_json", "old_bbox", "new_bbox", "reason_code"]
        )
        writer.writeheader()
        writer.writerows(logs["fixes_bbox"])

    write_audit_files(metadata_dir, audit_logs)

    # 4) 외부 데이터 전용 로그(옵션)
    ext_cfg = config.get("external_data", {}).get("alignment", {}).get("oob", {})
    fixes_ext_out = ext_cfg.get("fixes_log_out")
    excluded_ext_out = ext_cfg.get("excluded_log_out")
    if fixes_ext_out:
        path = resolve_path_fn(fixes_ext_out, base_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["source", "file_name", "source_json", "old_bbox", "new_bbox", "reason_code"]
            )
            writer.writeheader()
            writer.writerows(logs["fixes_bbox_external"])
    if excluded_ext_out:
        path = resolve_path_fn(excluded_ext_out, base_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["source", "file_name", "source_json", "reason_code", "detail"])
            writer.writeheader()
            writer.writerows(logs["excluded_rows_external"])

    # 5) category 매핑 산출물(옵션)
    mapping_cfg = config.get("external_data", {}).get("category_id_mapping", {})
    if mapping_cfg.get("save_mapping_table", False):
        out_path = resolve_path_fn(mapping_cfg.get("mapping_table_out", "data/metadata/category_mapping.csv"), base_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["dl_idx", "canonical_category_id", "name"])
            writer.writeheader()
            writer.writerows(mapping_rows)

    write_unmapped_external_audit_log(
        mapping_cfg=mapping_cfg,
        base_dir=base_dir,
        audit_logs=audit_logs,
        resolve_path_fn=resolve_path_fn,
    )
