from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Tuple


def scan_json_files(root: Path, recursive: bool = True) -> list[Path]:
    """JSON 파일 목록을 정렬해 반환한다."""
    if recursive:
        return sorted(root.rglob("*.json"))
    return sorted(root.glob("*.json"))


def scan_image_files(root: Path, recursive: bool = True) -> Tuple[dict[str, Path], dict[str, list[Path]]]:
    """
    PNG 파일을 파일명 소문자 기준으로 인덱싱한다.
    동일 파일명이 여러 경로에 있으면 duplicates에 별도 수집한다.
    """
    if recursive:
        paths = list(root.rglob("*.png"))
    else:
        paths = list(root.glob("*.png"))
    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        key = p.name.lower()
        if key in index:
            if key not in duplicates:
                duplicates[key].append(index[key])
            duplicates[key].append(p)
        else:
            index[key] = p
    return index, duplicates


def read_json_with_fallbacks(path: Path) -> Any:
    """UTF-8 계열/CP949 순서로 JSON을 읽고, 실패 시 replace 모드로 마지막 시도한다."""
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            with path.open("r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def parse_one_json(path: Path) -> Tuple[Optional[dict], Optional[str]]:
    """JSON 1개를 안전 파싱한다. 실패 사유 코드를 함께 반환한다."""
    try:
        data = read_json_with_fallbacks(path)
    except Exception as e:
        return None, f"json_read_error:{type(e).__name__}"
    if not isinstance(data, dict):
        return None, "top_level_not_dict"
    return data, None
