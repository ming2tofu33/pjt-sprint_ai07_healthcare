# 01 Data Pipeline

데이터 파이프라인은 STAGE 0~1에서 학습 입력을 고정하고, 누수/불일치를 줄이는 것을 목표로 합니다.

## STAGE 0: split + 정규화 + 감사

엔트리포인트:

- `scripts/0_split_data.py`
- 내부 오케스트레이션: `src/dataprep/output/data_pipeline.py`

주요 처리:

- JSON 로딩/인코딩 fallback (`src/dataprep/setup/io_utils.py`)
- 레코드 정규화/검증/클리핑 (`src/dataprep/process/normalize.py`)
- 중복 및 규칙 필터링 (`src/dataprep/process/dedup.py`)
- 그룹 기반 split (`src/dataprep/process/split.py`)
- 품질 감사 (`src/dataprep/process/quality_audit.py`)

주요 산출물:

- `data/processed/cache/<run_name>/df_clean.csv`
- `data/metadata/splits.csv`
- `data/metadata/preprocess_manifest.json`
- `data/metadata/audit_*.csv`

구조 예약 디렉터리:

- `data/processed/manifests/` (split 목록/manifest 집계)
- `data/processed/annotations/` (변환 라벨 산출물)

## STAGE 1: 학습 포맷 변환

엔트리포인트:

- `scripts/1_preprocess.py`
- 변환 모듈: `src/dataprep/output/export_yolo.py`

주요 처리:

- `df_clean.csv` + `splits.csv` 기반 YOLO 폴더 구성
- 하드링크 우선 복사 전략
- 누락 이미지/클래스 인덱스/라벨 포맷 검증

주요 산출물:

- `data/processed/datasets/<dataset_prefix>_<run_name>/`
- `data/metadata/class_map.csv`

## split/전처리 규칙

- `data/raw/` 데이터는 직접 수정하지 않습니다.
- split은 `group_id` 기반으로 수행해 train/val 누수를 방지합니다.
- bbox 규칙은 설정 파일(`configs/base.yaml`, 실험 override)에서 관리합니다.
- 모든 실행은 `run_name` 기준으로 cache/runs 산출물이 분리됩니다.
