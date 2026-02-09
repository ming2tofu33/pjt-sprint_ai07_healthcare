from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Tuple


def scan_json_files(root: Path, recursive: bool = True) -> list[Path]:
    """`root` 하위 JSON 경로를 정렬해 반환한다(기본값: 재귀 탐색)."""
    if recursive:
        return sorted(root.rglob("*.json"))
    return sorted(root.glob("*.json"))


def scan_image_files(root: Path, recursive: bool = True) -> Tuple[dict[str, Path], dict[str, list[Path]]]:
    """
    소문자 파일명을 키로 이미지 인덱스를 구축한다.

    반환값:
    - index: `file_name.lower()` -> 처음 발견된 경로
    - duplicates: JSON->이미지 매핑 충돌을 일으킬 수 있는 중복 파일명 후보
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
    """
    공통 인코딩 fallback 순서로 JSON을 읽는다.

    순서:
    1) utf-8
    2) utf-8-sig
    3) cp949
    4) 최후 수단으로 utf-8(replace 모드)
    """
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            with path.open("r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def parse_one_json(path: Path) -> Tuple[Optional[dict], Optional[str]]:
    """
    JSON 파일 1개를 파싱해 `(data, error_code)` 형태로 반환한다.

    파싱 예외를 호출자에게 바로 던지지 않고 구조화된 오류 코드를 반환해
    상위 파이프라인이 안전하게 로그 기록 후 건너뛸 수 있도록 한다.
    """
    try:
        data = read_json_with_fallbacks(path)
    except Exception as e:
        return None, f"json_read_error:{type(e).__name__}"
    if not isinstance(data, dict):
        return None, "top_level_not_dict"
    return data, None
