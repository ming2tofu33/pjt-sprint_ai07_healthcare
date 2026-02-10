from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - user environment decides
    yaml = None


_REQUIRED_EXPERIMENT_PATH_KEYS = (
    "train_images_dir",
    "train_annotations_dir",
    "test_images_dir",
    "processed_dir",
    "metadata_dir",
    "datasets_dir",
    "runs_dir",
    "best_models_dir",
    "submissions_dir",
)


def resolve_path(path_str: str, repo_root: Path) -> Path:
    """상대 경로를 `repo_root` 기준 절대 경로로 변환한다."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    """YAML을 로드하고 루트가 mapping(dict)인지 검증한다."""
    if yaml is None:
        raise RuntimeError("PyYAML이 필요합니다. 다음 명령으로 설치하세요: pip install pyyaml")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"설정 파일 최상위는 매핑(dict)이어야 합니다: {path}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """중첩 딕셔너리를 재귀 병합하며 `override` 값을 우선 적용한다."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def find_repo_root_from_script(script_path: Path) -> Optional[Path]:
    """`script_path`에서 상위로 탐색해 가장 가까운 `.git` 디렉터리를 찾는다."""
    current = script_path.resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def resolve_repo_root(script_path: Path, config_path: Path) -> Path:
    """스크립트 기준으로 repo root를 찾고, 실패 시 config 부모 경로를 사용한다."""
    repo_root = find_repo_root_from_script(script_path)
    if repo_root is not None:
        return repo_root
    return config_path.resolve().parent


def _resolve_nested_path(mapping: dict[str, Any], key: str, repo_root: Path) -> None:
    """`mapping[key]`가 경로 문자열이면 절대 경로 문자열로 정규화한다."""
    value = mapping.get(key)
    if isinstance(value, str):
        mapping[key] = str(resolve_path(value, repo_root))


def resolve_paths_in_config(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """
    preprocess 설정 내부의 알려진 경로 필드를 절대 경로로 정규화한다.

    작업 디렉터리나 실행 OS가 달라도 동일한 파일 IO 동작을 보장하기 위함이다.
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
    전처리 설정을 다음 순서로 로드한다.
    1) preprocess.yaml 로드
    2) preprocess.local.yaml이 있으면 deep-merge 적용
    3) repo root를 결정하고 알려진 경로 필드를 절대 경로로 정규화
    """
    config_path = config_path.resolve()
    config = _load_yaml_mapping(config_path)

    local_override_path: Optional[Path] = None
    if config_path.name == "preprocess.yaml":
        # 공용 기본값은 preprocess.yaml에 유지하고
        # 로컬 환경 차이는 preprocess.local.yaml에서 덮어써 팀 공통 설정 변경을 피한다.
        candidate = config_path.with_name("preprocess.local.yaml")
        if candidate.exists():
            local_override_path = candidate
            local_config = _load_yaml_mapping(candidate)
            config = _deep_merge(config, local_config)

    repo_root = resolve_repo_root(script_path, config_path)
    resolved_config = resolve_paths_in_config(config, repo_root)
    return resolved_config, repo_root, local_override_path


def _normalize_base_refs(base_ref: Any, *, config_path: Path) -> list[str]:
    """`_base_` 값을 정규화해 base 경로 리스트로 반환한다."""
    if base_ref is None:
        return []

    if isinstance(base_ref, str):
        ref = base_ref.strip()
        if not ref:
            raise ValueError(f"_base_ 값이 비어 있습니다: {config_path}")
        return [ref]

    if isinstance(base_ref, list):
        refs: list[str] = []
        for i, item in enumerate(base_ref):
            if not isinstance(item, str):
                raise ValueError(f"_base_ 목록 원소는 문자열이어야 합니다: {config_path} (index={i})")
            ref = item.strip()
            if not ref:
                raise ValueError(f"_base_ 목록에 빈 경로가 있습니다: {config_path} (index={i})")
            refs.append(ref)
        return refs

    raise ValueError(f"_base_ 는 문자열 또는 문자열 목록이어야 합니다: {config_path}")


def _load_experiment_config_recursive(
    config_path: Path,
    *,
    stack: tuple[Path, ...],
) -> dict[str, Any]:
    """실험 config를 `_base_` 체인 끝까지 재귀 병합해 로드한다."""
    config_path = config_path.resolve()
    if config_path in stack:
        chain = " -> ".join(str(p) for p in (*stack, config_path))
        raise ValueError(f"_base_ 순환 상속이 감지되었습니다: {chain}")

    config = _load_yaml_mapping(config_path)
    base_refs = _normalize_base_refs(config.get("_base_"), config_path=config_path)

    local_cfg = deepcopy(config)
    local_cfg.pop("_base_", None)

    merged_base: dict[str, Any] = {}
    for base_ref in base_refs:
        base_path = (config_path.parent / base_ref).resolve()
        if not base_path.exists():
            raise FileNotFoundError(
                f"기본(base) 설정 파일을 찾을 수 없습니다: {base_path} (from: {config_path})"
            )
        base_cfg = _load_experiment_config_recursive(base_path, stack=(*stack, config_path))
        merged_base = _deep_merge(merged_base, base_cfg)

    return _deep_merge(merged_base, local_cfg)


def _validate_experiment_config(config: dict[str, Any], *, config_path: Path) -> None:
    """실험 config 최소 계약(paths + 필수 키)을 검증한다."""
    paths_cfg = config.get("paths")
    if not isinstance(paths_cfg, dict):
        raise ValueError(f"실험 설정에 paths 섹션이 없습니다: {config_path}")

    missing = [
        key
        for key in _REQUIRED_EXPERIMENT_PATH_KEYS
        if not isinstance(paths_cfg.get(key), str) or not str(paths_cfg.get(key)).strip()
    ]
    if missing:
        raise ValueError(
            f"실험 설정 paths 키가 누락되었거나 비어 있습니다 ({config_path}): {', '.join(missing)}"
        )


def load_experiment_config(
    config_path: Path, script_path: Path
) -> tuple[dict[str, Any], Path]:
    """실험 config를 로드한다.

    1) config YAML 로드
    2) ``_base_`` 체인을 재귀적으로 끝까지 병합
    3) 최종 결과에서 ``_base_`` 키 제거 보장
    4) 필수 ``paths`` 계약 검증
    5) repo root 결정
    """
    config_path = config_path.resolve()
    config = _load_experiment_config_recursive(config_path, stack=())
    _validate_experiment_config(config, config_path=config_path)

    repo_root = resolve_repo_root(script_path, config_path)
    return config, repo_root
