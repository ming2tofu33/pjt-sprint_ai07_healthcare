# 00 Quickstart

5분 내 실행을 목표로 한 최소 실행 가이드입니다.

## 1) 환경 준비

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 2) 데이터 위치 확인

아래 경로가 채워져 있어야 합니다.

- `data/raw/train_images/`
- `data/raw/train_annotations/`
- `data/raw/test_images/`

외부 데이터를 쓸 경우:

- `data/raw/external/combined/images/`
- `data/raw/external/combined/annotations/`

## 3) 단일 실험 실행

```bash
RUN_NAME="quick_baseline"
CONFIG="configs/experiments/baseline.yaml"

python scripts/0_split_data.py --run-name $RUN_NAME --config $CONFIG
python scripts/1_preprocess.py --run-name $RUN_NAME --config $CONFIG
python scripts/2_train.py --run-name $RUN_NAME --config $CONFIG
python scripts/3_evaluate.py --run-name $RUN_NAME --config $CONFIG
python scripts/4_submission.py --run-name $RUN_NAME --config $CONFIG
```

## 4) 원커맨드 실행

```bash
bash scripts/run_pipeline.sh --run-name $RUN_NAME --config $CONFIG
```

## 5) 결과 확인

- 실험 결과: `runs/<RUN_NAME>/`
- 제출 CSV: `artifacts/submissions/`
- 베스트 모델: `artifacts/best_models/`
- 레지스트리: `runs/_registry.csv`

## 6) 문제 발생 시

- 설정 병합 결과: `runs/<RUN_NAME>/config_resolved.yaml`
- 데이터 감사 로그: `data/metadata/*.csv`
- 제출 검증 실패 시 `scripts/4_submission.py` 로그 확인
