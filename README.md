# pjt-sprint_ai07_healthcare

의료 이미지 객체 탐지(약 객체 검출) 파이프라인 프로젝트입니다.
목표는 재현 가능한 데이터 처리, 안정적인 학습/평가, 제출 산출물의 일관된 관리입니다.

## 환경

- Python: `3.11.9`
- 설치:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 빠른 실행

```bash
RUN_NAME="exp_YYYYMMDD_HHMMSS"
CONFIG="configs/experiments/baseline.yaml"

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

## 프로젝트 구조

```text
pjt-sprint_ai07_healthcare/
├── .gitignore
├── README.md
├── requirements.txt
│
├── docs/
│   ├── 00_quickstart.md
│   ├── 01_data_pipeline.md
│   └── 02_experiments.md
│
├── data/                                  # Git 제외
│   ├── raw/                               # 원본 데이터 (수정 금지)
│   │   ├── train_images/
│   │   ├── train_annotations/
│   │   ├── test_images/
│   │   └── external/
│   │       └── combined/
│   │           ├── annotations/
│   │           └── images/
│   ├── processed/                         # 가공 산출물
│   │   ├── manifests/                     # STAGE 0 split 결과
│   │   ├── annotations/                   # STAGE 1 라벨/변환 산출물
│   │   └── cache/                         # 파이프라인 캐시
│   └── metadata/                          # 감사/메타 로그
│
├── configs/
│   ├── base.yaml
│   └── experiments/
│       └── *.yaml
│
├── src/
│   ├── dataprep/
│   │   ├── __init__.py
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
│
├── scripts/
│   ├── 0_split_data.py
│   ├── 1_preprocess.py
│   ├── 2_train.py
│   ├── 3_evaluate.py
│   ├── 4_submission.py
│   └── run_pipeline.sh
│
├── runs/                                  # Git 제외(레지스트리만 관리)
│   ├── exp_YYYYMMDD_HHMMSS/
│   │   ├── config_resolved.yaml
│   │   ├── metrics.json
│   │   ├── weights/
│   │   └── vis/
│   └── _registry.csv
│
├── artifacts/                             # Git 제외
│   ├── best_models/
│   └── submissions/
│
└── playground/                            # Git 제외
```

참고: 저장소에는 실험 편의를 위한 보조 스크립트/파일이 추가로 존재할 수 있습니다.

## 운영 원칙

- `data/raw/`는 읽기 전용으로 취급합니다.
- 실험별 변경은 `configs/experiments/*.yaml`에서만 관리합니다.
- 실행 시 병합된 최종 설정은 `runs/<exp>/config_resolved.yaml`에 저장합니다.
- 실험 결과는 `runs/_registry.csv`로 중앙 기록합니다.

## 문서

- `docs/00_quickstart.md`
- `docs/01_data_pipeline.md`
- `docs/02_experiments.md`
