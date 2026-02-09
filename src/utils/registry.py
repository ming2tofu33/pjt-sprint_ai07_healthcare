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
    "weights_path",
    "config_path",
    "notes",
]


def _ensure_header(path: Path) -> None:
    """파일이 비어 있으면 헤더 행을 작성한다."""
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(_COLUMNS)


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
    """run_name으로 기존 행을 찾아 필드를 업데이트한다. 성공 시 True."""
    rows = load_registry(path)
    updated = False
    for row in rows:
        if row["run_name"] == run_name:
            for k, v in updates.items():
                if k in row:
                    row[k] = str(v) if v is not None else ""
            updated = True
            break
    if updated:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    return updated
