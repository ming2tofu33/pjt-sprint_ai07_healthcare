from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - user environment decides
    yaml = None


def resolve_path(path_str: str, repo_root: Path) -> Path:
    """상대 경로를 repo_root 기준 절대 경로로 정규화한다."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    """YAML 파일을 읽어 dict로 반환한다. 루트가 mapping이 아니면 예외."""
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """중첩 dict를 override 우선으로 병합한다."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def find_repo_root_from_script(script_path: Path) -> Optional[Path]:
    """script_path부터 상위로 올라가며 .git 디렉토리를 찾아 repo root를 추정한다."""
    current = script_path.resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def resolve_repo_root(script_path: Path, config_path: Path) -> Path:
    """repo root를 찾지 못하면 config 파일의 부모 경로를 fallback으로 사용한다."""
    repo_root = find_repo_root_from_script(script_path)
    if repo_root is not None:
        return repo_root
    return config_path.resolve().parent


def _resolve_nested_path(mapping: dict[str, Any], key: str, repo_root: Path) -> None:
    """mapping[key]가 문자열 경로일 때 절대 경로 문자열로 치환한다."""
    value = mapping.get(key)
    if isinstance(value, str):
        mapping[key] = str(resolve_path(value, repo_root))


def resolve_paths_in_config(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """
    전처리 설정의 경로 필드를 일괄 절대경로화한다.
    이후 파이프라인에서는 OS/실행 위치와 무관하게 동일 경로를 사용한다.
    """
    resolved = deepcopy(config)

    paths_cfg = resolved.get("paths")
    if isinstance(paths_cfg, dict):
        for key in ("train_images_dir", "train_annotations_dir", "test_images_dir", "processed_dir", "metadata_dir"):
            _resolve_nested_path(paths_cfg, key, repo_root)

    external_cfg = resolved.get("external_data")
    if not isinstance(external_cfg, dict):
        return resolved

    ingest_cfg = external_cfg.get("ingest")
    if isinstance(ingest_cfg, dict):
        for key in ("output_images_dir", "output_annotations_dir"):
            _resolve_nested_path(ingest_cfg, key, repo_root)

    sources_cfg = external_cfg.get("sources")
    if isinstance(sources_cfg, list):
        for source in sources_cfg:
            if isinstance(source, dict):
                _resolve_nested_path(source, "images_dir", repo_root)
                _resolve_nested_path(source, "annotations_dir", repo_root)

    alignment_cfg = external_cfg.get("alignment")
    if isinstance(alignment_cfg, dict):
        oob_cfg = alignment_cfg.get("oob")
        if isinstance(oob_cfg, dict):
            for key in ("fixes_log_out", "excluded_log_out"):
                _resolve_nested_path(oob_cfg, key, repo_root)

    mapping_cfg = external_cfg.get("category_id_mapping")
    if isinstance(mapping_cfg, dict):
        for key in ("mapping_table_out", "unmapped_log_out"):
            _resolve_nested_path(mapping_cfg, key, repo_root)

    return resolved


def load_preprocess_config(config_path: Path, script_path: Path) -> tuple[dict[str, Any], Path, Optional[Path]]:
    """
    전처리 설정 로더:
    1) preprocess.yaml 로드
    2) preprocess.local.yaml(있으면) deep merge
    3) repo root 추정 후 경로 절대화
    """
    config_path = config_path.resolve()
    config = _load_yaml_mapping(config_path)

    local_override_path: Optional[Path] = None
    if config_path.name == "preprocess.yaml":
        # 팀 공용 설정은 유지하고 개인 환경 차이는 local 파일로 덮어쓴다.
        candidate = config_path.with_name("preprocess.local.yaml")
        if candidate.exists():
            local_override_path = candidate
            local_config = _load_yaml_mapping(candidate)
            config = _deep_merge(config, local_config)

    repo_root = resolve_repo_root(script_path, config_path)
    resolved_config = resolve_paths_in_config(config, repo_root)
    return resolved_config, repo_root, local_override_path
