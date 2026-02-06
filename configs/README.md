# configs/ - Experiment Configuration Files

## 📌 개요

YAML 기반 실험 설정 관리. 모든 하이퍼파라미터와 실험 메타데이터를 버전 관리합니다.

---

## 📂 구조

```
configs/
├── base.yaml                    # 기본 설정 (모든 실험의 베이스)
└── experiments/
    ├── exp001_baseline.yaml     # Baseline (YOLOv8s, 56 classes)
    ├── exp002_whitelist.yaml    # Test 40 classes only
    ├── exp003_yolov8m.yaml      # Larger model
    ├── exp004_heavy_aug.yaml    # Heavy augmentation
    └── exp005_imgsz1024.yaml    # Higher resolution
```

---

## 🚀 사용법

### 1. Base Config로 실행

```bash
# scripts가 base.yaml을 자동으로 읽음 (기본값)
python scripts/1_create_coco_format.py --run-name exp_test
python scripts/3_train.py --run-name exp_test
```

### 2. 특정 실험 Config로 실행

```bash
# 실험 config를 명시적으로 지정
python scripts/1_create_coco_format.py --config configs/experiments/exp001_baseline.yaml --run-name exp001
python scripts/0_splitting.py --config configs/experiments/exp001_baseline.yaml --run-name exp001
python scripts/2_prepare_yolo_dataset.py --config configs/experiments/exp001_baseline.yaml --run-name exp001
python scripts/3_train.py --config configs/experiments/exp001_baseline.yaml --run-name exp001
python scripts/4_evaluate.py --run-name exp001
python scripts/5_submission.py --run-name exp001
```

### 3. Config 값 Override

```bash
# 실험명을 config의 실험 ID와 매칭하면 자동 연결
python scripts/3_train.py --run-name exp001
# → runs/exp001/config/config.json 생성됨 (YAML → JSON 변환)
```

---

## 📄 실험 Config 작성 가이드

### 기본 템플릿

```yaml
# Experiment XXX: Description
# Brief explanation

# Inherit base config
_base_: "../base.yaml"

# ============================================================
# Experiment Metadata
# ============================================================
experiment:
  id: "expXXX"
  name: "experiment_name"
  description: "What this experiment does"
  author: "@YourName"
  created: "2026-02-05"

# ============================================================
# Data Configuration (Override)
# ============================================================
data:
  class_whitelist: null  # or [1900, 16548, ...]
  num_classes: 56

# ============================================================
# Training Configuration (Override)
# ============================================================
train:
  model_name: "yolov8s.pt"
  imgsz: 768
  epochs: 80
  batch: 8
  lr0: 0.001
  # ... other overrides

# ============================================================
# Notes
# ============================================================
notes: |
  Additional notes and observations.
```

### Override 규칙

1. `_base_` 필드로 base.yaml 상속
2. 변경하고 싶은 필드만 명시 (나머지는 base 값 사용)
3. 중첩 필드도 부분 override 가능 (예: `train.epochs`만 변경)

---

## 📊 실험 목록

### exp001_baseline.yaml
- **목적**: Baseline 성능 측정
- **모델**: YOLOv8s
- **클래스**: 전체 56개
- **특징**: 기본 설정, 특별한 트릭 없음

### exp002_whitelist.yaml
- **목적**: Test set 클래스만 학습
- **모델**: YOLOv8s
- **클래스**: Test 40개 (whitelist 필요)
- **특징**: Class confusion 감소 기대

### exp003_yolov8m.yaml
- **목적**: 모델 용량 증가
- **모델**: YOLOv8m (larger)
- **배치**: 4 (메모리 제약)
- **특징**: Better capacity, 과적합 위험

### exp004_heavy_aug.yaml
- **목적**: 과적합 방지
- **증강**: Mosaic + Mixup + Copy-paste
- **에폭**: 120 (더 많이 필요)
- **특징**: 작은 데이터셋(232)에 적합

### exp005_imgsz1024.yaml
- **목적**: 작은 객체 검출 개선
- **해상도**: 1024 (기본 768)
- **배치**: 4 (메모리 제약)
- **특징**: 알약이 작을 수 있어 고해상도 필요

---

## 🔧 Config 값 설명

### 주요 필드

#### data
- `class_whitelist`: 학습할 클래스 필터 (null=전체)
- `num_classes`: 클래스 개수
- `max_objects_per_image`: 이미지당 최대 객체 (4)

#### split
- `strategy`: 분할 전략 (`stratify_by_num_objects`)
- `ratios`: Train/Val 비율 (0.8/0.2)

#### train
- `model_name`: YOLO 모델 (yolov8n/s/m/l/x)
- `imgsz`: 이미지 크기 (768, 1024, ...)
- `epochs`: 학습 에폭
- `batch`: 배치 크기
- `lr0`: 초기 learning rate
- `augment`: 증강 활성화 여부
- `mosaic/mixup/copy_paste`: 증강 확률

#### infer
- `conf_thr`: Confidence threshold (0.25)
- `nms_iou_thr`: NMS IoU threshold (0.45)
- `max_det_per_image`: 최대 검출 개수 (4)

---

## ✅ 실험 실행 체크리스트

1. **Config 작성**
   ```bash
   cp configs/experiments/exp001_baseline.yaml configs/experiments/exp006_custom.yaml
   # 수정...
   ```

2. **전체 파이프라인 실행**
   ```bash
   EXP_NAME="exp006"
   python scripts/1_create_coco_format.py --config configs/experiments/${EXP_NAME}_custom.yaml --run-name ${EXP_NAME}
   python scripts/0_splitting.py --config configs/experiments/${EXP_NAME}_custom.yaml --run-name ${EXP_NAME}
   python scripts/2_prepare_yolo_dataset.py --config configs/experiments/${EXP_NAME}_custom.yaml --run-name ${EXP_NAME}
   python scripts/3_train.py --config configs/experiments/${EXP_NAME}_custom.yaml --run-name ${EXP_NAME}
   python scripts/4_evaluate.py --run-name ${EXP_NAME}
   python scripts/5_submission.py --run-name ${EXP_NAME}
   ```

3. **결과 확인**
   ```bash
   cat artifacts/${EXP_NAME}/reports/eval_summary.txt
   ls artifacts/${EXP_NAME}/submissions/submission.csv
   ```

---

## 🎯 실험 우선순위 권장

1. **exp001_baseline**: 반드시 먼저 실행 (Baseline)
2. **exp002_whitelist**: Baseline 다음 (클래스 필터링 효과 확인)
3. **exp004_heavy_aug**: 과적합 있으면 시도
4. **exp005_imgsz1024**: mAP 부족하면 시도
5. **exp003_yolov8m**: 시간 여유 있으면 시도

---

**구현 완료**: 2026-02-05  
**담당**: @DM  
**상태**: Stage 3 완료 ✅
