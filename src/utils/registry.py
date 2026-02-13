"""src.utils.registry — 실험 레지스트리(runs/_registry.csv) 관리.

사용:
    from src.utils.registry import append_run, load_registry
    append_run(registry_path, run_name="exp_20260209_120000", ...)
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


_COLUMNS = [
    "run_name",
    "created_at",
    "model",
    "epochs",
    "imgsz",
    "best_map50",
    "best_map50_95",
    "best_map75_95",      # 대회 평가 지표: mAP@[0.75:0.95]
    "weights_path",
    "config_path",
    "notes",
]


def _ensure_header(path: Path) -> None:
    """파일이 없거나 비었으면 헤더를 작성하고, 구버전 헤더이면 마이그레이션한다."""
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(_COLUMNS)
        return

    # 기존 파일의 헤더와 _COLUMNS 비교 → 누락 컬럼이 있으면 마이그레이션
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_fields = reader.fieldnames or []
        rows = list(reader)

    missing = [c for c in _COLUMNS if c not in existing_fields]
    if not missing:
        return  # 스키마 일치, 마이그레이션 불필요

    # 구버전 행에 누락 컬럼을 빈 값으로 채워서 다시 쓰기
    for row in rows:
        for col in missing:
            row.setdefault(col, "")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def load_registry(path: Path) -> list[dict[str, str]]:
    """레지스트리 CSV를 읽어 dict 목록으로 반환한다."""
    _ensure_header(path)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def append_run(
    path: Path,
    *,
    run_name: str,
    model: str = "",
    epochs: int = 0,
    imgsz: int = 0,
    best_map50: Optional[float] = None,
    best_map50_95: Optional[float] = None,
    best_map75_95: Optional[float] = None,
    weights_path: str = "",
    config_path: str = "",
    notes: str = "",
) -> None:
    """레지스트리에 실험 행을 추가한다."""
    _ensure_header(path)
    row = {
        "run_name": run_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "epochs": str(epochs),
        "imgsz": str(imgsz),
        "best_map50": f"{best_map50:.4f}" if best_map50 is not None else "",
        "best_map50_95": f"{best_map50_95:.4f}" if best_map50_95 is not None else "",
        "best_map75_95": f"{best_map75_95:.4f}" if best_map75_95 is not None else "",
        "weights_path": weights_path,
        "config_path": config_path,
        "notes": notes,
    }
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        writer.writerow(row)


def update_run(
    path: Path,
    run_name: str,
    **updates: Any,
) -> bool:
    """run_name으로 기존 행을 찾아 필드를 업데이트한다. 성공 시 True.

    ``_COLUMNS`` 에 정의된 필드만 업데이트한다.
    마이그레이션 후에도 row 에 누락 키가 있을 수 있으므로
    ``_COLUMNS`` 기준으로 필터링한다.
    """
    rows = load_registry(path)  # _ensure_header 로 마이그레이션 보장
    updated = False
    valid_keys = set(_COLUMNS)
    for row in rows:
        if row["run_name"] == run_name:
            for k, v in updates.items():
                if k in valid_keys:
                    row[k] = str(v) if v is not None else ""
            updated = True
            break
    if updated:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    return updated
