# src/ - Core Modules

## 개요

재사용 가능한 핵심 모듈. `scripts/`의 모든 파이프라인 스크립트가 이 모듈을 사용합니다.

---

## 구조

```
src/
├── __init__.py       # Python 패키지 마커
└── utils.py          # 핵심 — 경로, Config 상속/병합, seed, IO
```

> `utils.py` 하나가 전체 파이프라인의 공통 인프라를 담당합니다.
> 학습/추론/데이터 처리는 `scripts/`에서 `ultralytics.YOLO`를 직접 호출합니다.

---

## `utils.py` 모듈 상세

**공통 유틸리티 함수**

| 함수 | 설명 |
|------|------|
| `setup_project_paths()` | 프로젝트 경로 설정 및 폴더 생성 |
| `set_seed()` | 재현성을 위한 seed 고정 |
| `load_config()` | Config 로드 (JSON/YAML, `_base_` 상속 지원) |
| `merge_configs()` | 두 config dict 깊은 병합 (base + override) |
| `save_config()` | Config 저장 (JSON/YAML) |
| `get_dataset_dir()` | YOLO 데이터셋 디렉토리 경로 헬퍼 |
| `get_data_yaml()` | data.yaml 경로 헬퍼 |
| `create_run_manifest()` | 실험 메타데이터 생성 |
| `record_result()` | 결과 기록 (CSV + JSONL) |

**사용 예시**:
```python
from src.utils import setup_project_paths, set_seed, load_config, get_data_yaml

# 경로 설정
paths = setup_project_paths(run_name="exp001", create_dirs=True)

# Seed 고정
set_seed(42, deterministic=True)

# Config 로드 (_base_ 상속 자동 처리)
config = load_config("configs/experiments/exp001_baseline.yaml")

# YOLO data.yaml 경로
data_yaml = get_data_yaml(paths)
```

---

## 모듈 간 관계

```
scripts/ (0~5_*.py)
  ↓ (import)
src/utils.py    ← 전 스크립트가 사용 (경로, config, seed, 기록)
  ↓
ultralytics     ← scripts에서 YOLO() 직접 호출 (학습, 추론, 평가)
```

---

**구현 완료**: 2026-02-06
**담당**: @DM
**상태**: utils.py 리팩토링 완료 (config 상속, flat 구조, 경로 헬퍼)
