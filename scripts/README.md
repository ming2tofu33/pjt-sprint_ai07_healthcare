# scripts/ - 데이터 파이프라인 및 학습 스크립트

## 📌 개요

재사용 가능한 실험 스크립트 모음입니다. 각 스크립트는 독립적으로 실행 가능하며, `src/utils.py`의 공통 기능을 활용합니다.

---

## 🔄 실행 순서

### Stage 1: 데이터 파이프라인

```bash
# 1. COCO Format 생성 (763개 JSON → 232개 이미지 통합)
python scripts/1_create_coco_format.py

# 2. Train/Val Split (Stratified)
python scripts/0_splitting.py

# 선택: 특정 실험명 지정
python scripts/1_create_coco_format.py --run-name exp_baseline_v1
python scripts/0_splitting.py --run-name exp_baseline_v1
```

### Stage 2: 학습 및 평가

```bash
# 2. YOLO 데이터셋 준비 (COCO → YOLO format)
python scripts/2_prepare_yolo_dataset.py --run-name exp_baseline_v1

# 3. 모델 학습
python scripts/3_train.py --run-name exp_baseline_v1

# 4. 모델 평가
python scripts/4_evaluate.py --run-name exp_baseline_v1

# 5. 제출 파일 생성
python scripts/5_submission.py --run-name exp_baseline_v1
```

---

## 📄 스크립트 상세

### Stage 1: 데이터 파이프라인

#### `1_create_coco_format.py`

**기능**:
- `train_annotations/` 아래 763개 JSON → 232개 이미지 단위 통합
- BBox 클리핑 및 검증 (이미지 경계 밖 제거)
- Category 매핑 생성 (`id2idx`, `idx2id`)
- Class whitelist 적용 (옵션)

**사용법**:
```bash
python scripts/1_create_coco_format.py [--config CONFIG] [--run-name NAME]
```

**옵션**:
- `--config`: Config 파일 경로 (선택, 기본: `runs/<run_name>/config/config.json`)
- `--run-name`: 실험명 (선택, 기본: `exp_YYYYMMDD_HHMMSS`)

**출력 파일**:
```
data/processed/cache/<run_name>/
├── train_merged_coco.json      # 통합 COCO 파일
├── image_id_map.json           # file_name → image_id 매핑
├── category_id_to_name.json    # category_id → name 매핑
├── label_map_full.json         # 전체 클래스 매핑 (id2idx, idx2id)
└── label_map_whitelist.json    # Whitelist 클래스 매핑 (있을 때만)

artifacts/<run_name>/reports/
├── coco_merge_stats.json       # 병합 통계
└── train_only_category_ids.json # Train-only 클래스 (whitelist 있을 때)
```

**Config 설정**:
```json
{
  "data": {
    "class_whitelist": null,  // null=전체 / [1900, 16548, ...]=부분
    "num_classes": 56         // 자동 업데이트됨
  }
}
```

---

#### `0_splitting.py`

**기능**:
- Stratified split (객체 수 기반)
- K-Fold 지원 (옵션, TODO)
- Split 품질 검증 (분포 균등성)
- Train/Val ID 리스트 저장

**사용법**:
```bash
python scripts/0_splitting.py [--config CONFIG] [--run-name NAME] [--kfold]
```

**옵션**:
- `--config`: Config 파일 경로 (선택)
- `--run-name`: 실험명 (선택)
- `--kfold`: K-Fold 모드 (현재 미구현, TODO)
- `--fold-idx`: Fold 인덱스 (K-Fold 모드 시)

**출력 파일**:
```
data/processed/cache/<run_name>/splits/
├── split_train_valid.json  # Split 정보 (image_ids, 분포 등)
├── train_ids.txt           # Train image IDs (한 줄에 하나)
└── valid_ids.txt           # Valid image IDs (한 줄에 하나)
```

**Config 설정**:
```json
{
  "split": {
    "strategy": "stratify_by_num_objects",  // n_objects / signature / hybrid
    "seed": 42,
    "ratios": {"train": 0.8, "valid": 0.2},
    "kfold": {
      "enabled": false,
      "n_splits": 5,
      "fold_idx": 0
    }
  }
}
```

**Stratify 모드**:
- `n_objects`: 이미지당 객체 수 (2/3/4) 기준
- `signature`: 멀티라벨 시그니처 기준 (정밀)
- `hybrid`: 둘 다 사용 (strata가 너무 작으면 n_objects로 fallback)

---

### Stage 2: 학습 파이프라인

#### `2_prepare_yolo_dataset.py`

**기능**:
- COCO → YOLO 포맷 변환
- Train/Val 이미지 + 라벨 복사/심볼릭 링크
- `data.yaml` 생성 (Ultralytics YOLO 필수)
- 데이터셋 검증 (누락/잘못된 라벨 체크)

**사용법**:
```bash
python scripts/2_prepare_yolo_dataset.py [--config CONFIG] [--run-name NAME] [--copy]
```

**옵션**:
- `--config`: Config 파일 경로 (선택)
- `--run-name`: 실험명 (선택)
- `--copy`: 이미지를 심볼릭 링크 대신 복사 (선택)

**출력 파일**:
```
data/processed/datasets/<run_name>_yolo/
├── data.yaml               # YOLO 데이터셋 설정
├── train/
│   ├── images/            # Train 이미지 (symlink or copy)
│   └── labels/            # Train 라벨 (.txt)
├── val/
│   ├── images/            # Val 이미지 (symlink or copy)
│   └── labels/            # Val 라벨 (.txt)
└── convert_manifest.json  # 변환 통계
```

**Config 설정**:
```json
{
  "data": {
    "yolo_dataset_root": "data/processed/datasets"
  }
}
```

---

#### `3_train.py`

**기능**:
- Ultralytics YOLO 학습
- Config의 **모든 학습 파라미터** (augmentation, optimizer, loss weight 등) YOLO에 전달
- 체크포인트 저장 (best/last)
- 학습 로그 기록 (metrics.jsonl, results.csv)

**사용법**:
```bash
python scripts/3_train.py --run-name exp_baseline_v1 --config configs/experiments/exp001_baseline.yaml
```

**옵션**:
- `--run-name`: 실험명 (필수)
- `--config`: 실험 YAML 파일 경로 (선택, 없으면 config.json 또는 기본값)
- `--device`: GPU device (기본: 0)
- `--resume`: 중단된 학습 재개 (선택)

**Config 로드 우선순위**: `--config` > `runs/<run_name>/config/config.json` > 기본값

**출력 파일**:
```
runs/<run_name>/
├── checkpoints/
│   ├── best.pt            # Best 체크포인트
│   └── last.pt            # Last 체크포인트
└── logs/
    └── metrics.jsonl      # 학습 로그
```

**Config 설정** (flat 구조):
```yaml
# configs/experiments/exp001_baseline.yaml
_base_: "../base.yaml"
train:
  model_name: "yolov8s.pt"
  imgsz: 768
  epochs: 80
  batch: 8
  lr0: 0.001
  optimizer: "auto"
  # augmentation
  mosaic: 1.0
  mixup: 0.0
  copy_paste: 0.0
  # loss weights
  box: 7.5
  cls: 0.5
  dfl: 1.5
```

> 모든 `train` 섹션의 값이 YOLO `model.train()`에 전달됩니다.

---

#### `4_evaluate.py`

**기능**:
- 학습된 모델 평가 (Val set)
- mAP@0.75~0.95 계산 (대회 지표)
- mAP@0.5, mAP@0.75 참고용 기록
- Config의 `val` 섹션 (conf, iou, save_json) YOLO val에 전달

**사용법**:
```bash
python scripts/4_evaluate.py --run-name exp_baseline_v1 --config configs/experiments/exp001_baseline.yaml
```

**옵션**:
- `--run-name`: 실험명 (필수)
- `--config`: 실험 YAML 파일 경로 (선택)
- `--ckpt`: 체크포인트 선택 (기본: best, 선택: last)
- `--device`: GPU device (기본: 0)

**출력 파일**:
```
artifacts/<run_name>/reports/
├── eval_results.json      # 평가 결과 (JSON)
└── eval_summary.txt       # 요약 텍스트
```

**출력 지표**:
- `mAP@0.50:0.95` (대회 공식 지표)
- `mAP@0.50` (참고용)
- `mAP@0.75` (참고용)
- Per-class AP (클래스별 성능)

---

#### `5_submission.py`

**기능**:
- Test 이미지 추론
- Top-4 객체 선택 (대회 규칙)
- YOLO 클래스 인덱스 → 원본 COCO category_id 자동 변환
- `submission.csv` 생성 및 검증

**사용법**:
```bash
python scripts/5_submission.py --run-name exp_baseline_v1 --config configs/experiments/exp001_baseline.yaml
```

**옵션**:
- `--run-name`: 실험명 (필수)
- `--config`: 실험 YAML 파일 경로 (선택)
- `--ckpt`: 체크포인트 선택 (기본: best)
- `--conf`: Confidence threshold (기본: config 값)
- `--device`: GPU device (기본: 0)

**출력 파일**:
```
artifacts/<run_name>/submissions/
└── submission.csv         # Kaggle 제출 파일
```

**submission.csv 형식**:
```csv
annotation_id,image_id,category_id,bbox_x,bbox_y,bbox_w,bbox_h,score
1,1,1900,100.5,200.3,50.2,80.1,0.95
2,1,16548,300.2,150.4,60.3,70.5,0.89
```

> category_id는 `label_map_full.json`의 `idx2id` 매핑을 사용하여 원본 COCO ID로 변환됩니다.

**Config 설정**:
```yaml
infer:
  conf_thr: 0.25
  nms_iou_thr: 0.45
  max_det_per_image: 4
```

---

**Stratify 모드**:
- `n_objects`: 이미지당 객체 수 (2/3/4) 기준
- `signature`: 멀티라벨 시그니처 기준 (정밀)
- `hybrid`: 둘 다 사용 (strata가 너무 작으면 n_objects로 fallback)

---

## 🔧 공통 옵션

### 실험명 지정
```bash
# 자동 생성 (exp_YYYYMMDD_HHMMSS)
python scripts/1_create_coco_format.py

# 수동 지정
python scripts/1_create_coco_format.py --run-name exp_baseline_v1
```

### Config 재사용
```bash
# 기존 실험의 config 사용
python scripts/0_splitting.py --config runs/exp_baseline_v1/config/config.json
```

---

## 📊 생성되는 파일 구조

```
pjt-sprint_ai07_healthcare/
├── runs/
│   └── <run_name>/
│       ├── config/
│       │   ├── config.json
│       │   ├── paths_meta.json
│       │   └── env_meta.json
│       ├── checkpoints/     # (Stage 2에서 생성)
│       └── logs/            # (Stage 2에서 생성)
│
├── data/processed/cache/<run_name>/
│   ├── train_merged_coco.json
│   ├── image_id_map.json
│   ├── category_id_to_name.json
│   ├── label_map_full.json
│   ├── label_map_whitelist.json (optional)
│   └── splits/
│       ├── split_train_valid.json
│       ├── train_ids.txt
│       └── valid_ids.txt
│
└── artifacts/<run_name>/
    └── reports/
        ├── coco_merge_stats.json
        └── train_only_category_ids.json (optional)
```

---

## 🐛 Troubleshooting

### Q: `train_merged_coco.json`이 없다는 에러
```
❌ train_merged_coco.json이 없습니다
ℹ️  먼저 scripts/1_create_coco_format.py를 실행하세요.
```
→ **해결**: `scripts/1_create_coco_format.py`를 먼저 실행

### Q: Class whitelist 설정 방법
**방법 1**: Config 파일 수정
```json
{
  "data": {
    "class_whitelist": [1900, 16548, 19607, 29451]  // Test 40개 클래스 ID
  }
}
```

**방법 2**: Config 파일 없이 실행 (기본값 사용)
```bash
python scripts/1_create_coco_format.py  # class_whitelist=null (전체 사용)
```

### Q: Stratify fallback 경고
```
⚠️  Fallback used: hybrid → n_objects
```
→ **정상**: Hybrid 모드가 너무 잘게 나뉘어 n_objects로 자동 전환됨

### Q: 실험명이 너무 길어짐
```bash
# 짧은 이름 권장
python scripts/1_create_coco_format.py --run-name exp_v1
python scripts/0_splitting.py --run-name exp_v1
```

---

## ✅ 실행 예시

### 기본 실험
```bash
# 1. COCO 생성 (전체 클래스)
python scripts/1_create_coco_format.py --run-name exp_baseline

# 2. Split
python scripts/0_splitting.py --run-name exp_baseline

# 확인
cat data/processed/cache/exp_baseline/splits/train_ids.txt | wc -l  # 185
cat data/processed/cache/exp_baseline/splits/valid_ids.txt | wc -l  # 47
```

### Whitelist 실험
```bash
# 1. Config 수정
vi runs/exp_whitelist/config/config.json
# → "class_whitelist": [1900, 16548, 19607, ...]

# 2. COCO 생성
python scripts/1_create_coco_format.py --run-name exp_whitelist

# 3. Split
python scripts/0_splitting.py --run-name exp_whitelist
```

### Config 재사용
```bash
# exp_baseline의 설정을 exp_v2에서 재사용
cp runs/exp_baseline/config/config.json /tmp/my_config.json
# (필요 시 수정)

python scripts/1_create_coco_format.py --config /tmp/my_config.json --run-name exp_v2
python scripts/0_splitting.py --config /tmp/my_config.json --run-name exp_v2
```

---

## 🚀 전체 실행 예시

### 기본 실험 (전체 클래스)

```bash
CONFIG="configs/experiments/exp001_baseline.yaml"

# Stage 1: 데이터 파이프라인
python scripts/1_create_coco_format.py --run-name exp_baseline
python scripts/0_splitting.py --run-name exp_baseline

# Stage 2: 학습 및 평가
python scripts/2_prepare_yolo_dataset.py --run-name exp_baseline
python scripts/3_train.py --run-name exp_baseline --config $CONFIG
python scripts/4_evaluate.py --run-name exp_baseline --config $CONFIG
python scripts/5_submission.py --run-name exp_baseline --config $CONFIG

# 확인
ls -lh artifacts/exp_baseline/submissions/submission.csv
```

### Whitelist 실험 (Test 40개 클래스만)

```bash
CONFIG="configs/experiments/exp002_whitelist.yaml"

# 전체 파이프라인 실행
python scripts/1_create_coco_format.py --config $CONFIG --run-name exp_whitelist
python scripts/0_splitting.py --config $CONFIG --run-name exp_whitelist
python scripts/2_prepare_yolo_dataset.py --config $CONFIG --run-name exp_whitelist
python scripts/3_train.py --config $CONFIG --run-name exp_whitelist
python scripts/4_evaluate.py --config $CONFIG --run-name exp_whitelist
python scripts/5_submission.py --config $CONFIG --run-name exp_whitelist
```

---

## 🚀 다음 단계

### 개선 사항
- [ ] K-Fold split 구현 (`0_splitting.py`)
- [ ] Multi-GPU 지원 (`3_train.py`)
- [ ] TTA (Test-Time Augmentation) 지원 (`5_submission.py`)
- [ ] Config validation (YAML schema)
- [ ] Ensemble 스크립트 (다중 모델 결과 병합)

---

**구현 완료**: 2026-02-06
**담당**: @DM
**상태**: Stage 0~5 완료 ✅ (전 스크립트 --config 플래그 지원)
