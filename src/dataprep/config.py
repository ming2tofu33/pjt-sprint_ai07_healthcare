from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - user environment decides
    yaml = None


def resolve_path(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def find_repo_root_from_script(script_path: Path) -> Optional[Path]:
    current = script_path.resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def resolve_repo_root(script_path: Path, config_path: Path) -> Path:
    repo_root = find_repo_root_from_script(script_path)
    if repo_root is not None:
        return repo_root
    return config_path.resolve().parent


def _resolve_nested_path(mapping: dict[str, Any], key: str, repo_root: Path) -> None:
    value = mapping.get(key)
    if isinstance(value, str):
        mapping[key] = str(resolve_path(value, repo_root))


def resolve_paths_in_config(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
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
    config_path = config_path.resolve()
    config = _load_yaml_mapping(config_path)

    local_override_path: Optional[Path] = None
    if config_path.name == "preprocess.yaml":
        candidate = config_path.with_name("preprocess.local.yaml")
        if candidate.exists():
            local_override_path = candidate
            local_config = _load_yaml_mapping(candidate)
            config = _deep_merge(config, local_config)

    repo_root = resolve_repo_root(script_path, config_path)
    resolved_config = resolve_paths_in_config(config, repo_root)
    return resolved_config, repo_root, local_override_path
