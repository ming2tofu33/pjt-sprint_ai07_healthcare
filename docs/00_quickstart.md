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

산출물 레이아웃은 `configs/base.yaml`의 `paths.artifact_layout`로 선택합니다.

- `compact`(기본): 학습 산출물이 `runs/<RUN_NAME>/train/`, 제출 디버그가 `runs/<RUN_NAME>/submit/debug/`에 생성
- `legacy`(옵션): 학습 산출물이 `runs/<RUN_NAME>/` 루트에 생성되고, 제출 디버그는 `runs/<RUN_NAME>/submission_debug/`에 생성

학습 로그 모드/스냅샷 기본값:

- `train.log_mode: batch`
- `train.debug_snapshots.enabled: true` (필요 시 실험 YAML에서 끄기)
- 스냅샷을 켜면 `runs/<RUN_NAME>/train/snapshots/`에 저장

## 6) 문제 발생 시

- 설정 병합 결과: `runs/<RUN_NAME>/config_resolved.yaml`
- 데이터 감사 로그: `data/metadata/*.csv`
- 제출 검증 실패 시 `scripts/4_submission.py` 로그 확인

## 7) Ensemble (WBF) 제출

기본값은 단일 모델 제출입니다. 앙상블을 쓰려면 `configs/base.yaml` 또는 실험 YAML에서
`submission.ensemble.enabled: true`로 켜고 `submission.ensemble.runs`를 채우세요.

예시:

```yaml
submission:
  ensemble:
    enabled: true
    method: "wbf"
    strict_category_map: true
    runs:
      - run_name: "yolov8s_best_run"
        weight_tag: "competition_best"
        conf: 0.20
        nms_iou: 0.50
        imgsz: 640
        augment: false
        model_weight: 1.0
      - run_name: "yolo11s_best_run"
        weight_tag: "competition_best"
        conf: 0.20
        nms_iou: 0.50
        imgsz: 1024
        augment: false
        model_weight: 1.0
```

실행:

```bash
python scripts/4_submission.py --run-name ensemble_submit --config <your_config.yaml> --ensemble-enable
```

주의:

- `strict_category_map: true`일 때 run 간 `category_id` 집합이 다르면 즉시 실패합니다.
- 최종 제출은 항상 `Top-4` 규칙을 강제 적용합니다.

### Ensemble Stability Notes

- Ensemble inference runs model-by-model (sequentially) and performs best-effort CUDA cleanup between models (`gc.collect()`, `torch.cuda.empty_cache()`).
- `model_weight` must be a positive finite value. Invalid values (`<=0`, `NaN`, `inf`) fail fast before WBF.
- Submission manifest includes richer ensemble trace fields such as `ensemble_models` and `ensemble_stats`.

### Ops Tips

- If OOM occurs, lower per-model `imgsz` first (for example `1024 -> 896 -> 768`) before changing other settings.
- Tune `model_weight` as relative ratios; the sum does not need to be exactly `1.0`.
