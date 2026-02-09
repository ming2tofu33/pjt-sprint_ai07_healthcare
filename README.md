# Healthcare AI Project

YOLO 기반 의약품(알약) 객체 탐지 프로젝트입니다.  
대회 기간 동안 `재현성`, `실수 방지`, `빠른 실험 반복`을 최우선으로 둡니다.

## 1. 프로젝트 목표

- 의료 이미지 객체 탐지 파이프라인 표준화
- 데이터 정제/중복 제거/품질 감사 자동화
- 안정적인 학습/평가/제출 루프 구축
- 제출 규칙 준수: 이미지당 최대 4개 박스(Top-4), `category_id` 체계 일관성 유지

## 2. 개발 환경

- Python `3.11.9` 권장
- 가상환경 생성 및 의존성 설치:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 3. 표준 디렉토리 구조

아래 구조를 기준으로 운영합니다.

```text
pjt-sprint_ai07_healthcare/
├── .gitignore
├── README.md
├── requirements.txt
├── docs/
│   ├── 00_quickstart.md
│   ├── 01_data_pipeline.md
│   └── 02_experiments.md
├── data/
│   ├── raw/                                 # 원본 데이터 (수정 금지)
│   │   ├── train_images/
│   │   ├── train_annotations/
│   │   ├── test_images/
│   │   └── external/
│   │       └── combined/
│   │           ├── annotations/
│   │           └── images/
│   ├── processed/                           # 파이프라인 산출물
│   │   ├── manifests/                       # STAGE 0 산출물
│   │   ├── annotations/                     # STAGE 1 산출물
│   │   └── cache/
│   └── metadata/
├── configs/
│   ├── base.yaml
│   └── experiments/
│       └── *.yaml
│
├── src/
│   ├── dataprep/
│   │   ├── setup/
│   │   │   └── io_utils.py
│   │   ├── process/
│   │   │   ├── normalize.py
│   │   │   ├── dedup.py
│   │   │   ├── quality_audit.py
│   │   │   └── split.py
│   │   └── output/
│   │       ├── export.py
│   │       ├── manifest.py
│   │       └── data_pipeline.py
│   ├── models/
│   │   ├── detector.py
│   │   └── architectures.py
│   ├── engine/
│   │   ├── trainer.py
│   │   └── validator.py
│   ├── inference/
│   │   ├── predictor.py
│   │   ├── postprocess.py
│   │   └── submission.py
│   └── utils/
│       ├── config_loader.py
│       ├── logger.py
│       ├── visualizer.py
│       └── registry.py
├── scripts/
│   ├── 0_split_data.py
│   ├── 1_preprocess.py
│   ├── 2_train.py
│   ├── 3_evaluate.py
│   ├── 4_submission.py
│   └── run_pipeline.sh
├── runs/
│   ├── exp_YYYYMMDD_HHMMSS/
│   │   ├── config_resolved.yaml
│   │   ├── metrics.json
│   │   ├── weights/
│   │   └── vis/
│   └── _registry.csv
├── artifacts/
│   ├── best_models/
│   └── submissions/
└── playground/
```

## 4. 파이프라인 실행 순서 (STAGE 0~4)

모든 명령은 프로젝트 루트에서 실행합니다.

```bash
RUN_NAME="exp_20260209_120000"
CONFIG="configs/experiments/DM.yaml"

python scripts/0_split_data.py --run-name $RUN_NAME --config $CONFIG
python scripts/1_preprocess.py --run-name $RUN_NAME --config $CONFIG
python scripts/2_train.py --run-name $RUN_NAME --config $CONFIG
python scripts/3_evaluate.py --run-name $RUN_NAME --config $CONFIG
python scripts/4_submission.py --run-name $RUN_NAME --config $CONFIG
```

원커맨드 실행:

```bash
bash scripts/run_pipeline.sh --run-name $RUN_NAME --config $CONFIG
```

## 5. 설정(config) 운영 규칙

- 기본값은 `configs/base.yaml`에 유지
- 실험별 변경값은 `configs/experiments/*.yaml`에만 작성
- 실험 시 `runs/<exp>/config_resolved.yaml`에 최종 병합 설정 저장
- 코드가 아니라 YAML로 실험 조건을 관리

## 6. 재현성 및 운영 원칙

- `data/raw/`는 불변 계층으로 간주하고 직접 수정하지 않음
- split은 seed 고정 + manifest 저장
- 산출물은 파일로 남김: 정제 데이터, 제외 이력, bbox 수정 이력, 감사 로그
- `runs/_registry.csv`에 실험명/점수/체크포인트/날짜 기록

## 7. 대회 기간 우선순위

지금 구현(필수):

- `scripts/0_split_data.py` ~ `scripts/4_submission.py`
- `src/dataprep/*` 전처리 체인
- `src/models/detector.py`
- `src/inference/predictor.py`, `src/inference/postprocess.py`
- `src/utils/config_loader.py`, `src/utils/logger.py`, `src/utils/visualizer.py`

후순위(미루기 권장):

- `src/models/architectures.py` 커스텀 모델 확장
- `src/engine/trainer.py`, `src/engine/validator.py` 자체 재구현
- 로깅/리포팅 고도화(대시보드/대형 HTML 리포트)

## 8. 참고 문서

- `docs/00_quickstart.md`
- `docs/01_data_pipeline.md`
- `docs/02_experiments.md`
- `CLAUDE.md`
