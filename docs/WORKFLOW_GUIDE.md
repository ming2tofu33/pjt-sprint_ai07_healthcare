# 🏥 경구약제 객체 검출 — 프로젝트 워크플로우 가이드

> **Team #4 | Kaggle Pill Detection Competition**
> 타겟 지표: `mAP@[0.75:0.95]` | 모델: YOLO (Ultralytics) | 데이터: 232 Train + 842 Test 이미지

---

## 📌 목차

1. [실행 전 필수 준비물](#1--실행-전-필수-준비물-prerequisites)
2. [전체 파이프라인 흐름도](#2--전체-파이프라인-흐름도)
3. [단계별 상세 실행 가이드](#3--단계별-상세-실행-가이드)
4. [Config 시스템 & 실험 YAML 관리](#4--config-시스템--실험-yaml-관리)
5. [가중치 파일 관리](#5--가중치-파일weights-관리)
6. [평가 결과물 및 해석](#6--평가-결과물-및-해석)
7. [새 실험 시작하기 — Step-by-Step 예시](#7--새-실험-시작하기--step-by-step-예시)
8. [고급: 2단계 학습 & TTA](#8--고급-2단계-학습--tta)
9. [Troubleshooting](#9--troubleshooting)

---

## 1. 🛠 실행 전 필수 준비물 (Prerequisites)

### 1-1. 데이터 디렉토리 구조

> ⚠️ **아래 구조가 정확히 맞아야 파이프라인이 동작합니다.**

```
data/
└── raw/                            ← 원본 데이터 (수정 금지!)
    ├── train_images/               ← 학습 이미지 (232장, .png)
    │   ├── 4.png
    │   ├── 5.png
    │   ├── 6.png
    │   └── ...
    │
    ├── train_annotations/          ← 어노테이션 (114개 폴더, 763개 JSON)
    │   ├── ㅇㅇㅇ정/
    │   │   ├── ㅇㅇㅇ정_1.json
    │   │   ├── ㅇㅇㅇ정_2.json
    │   │   └── ...
    │   ├── ㅁㅁㅁ틴/
    │   │   └── ...
    │   └── ... (114개 약제 폴더)
    │
    └── test_images/                ← 테스트 이미지 (842장, .png)
        ├── 0.png
        ├── 1.png
        └── ...
```

> 💡 **데이터 다운로드**: Kaggle 대회 페이지에서 데이터를 다운받아 위 경로에 배치하세요.

### 1-2. 환경 구축

```bash
# 1. Python 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# 2. 패키지 설치
pip install -r requirements.txt
```

> ✅ **핵심 패키지 버전**

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `ultralytics` | 8.4.12 | YOLO 프레임워크 (YOLO11 포함) |
| `torch` | 2.5.1+cu121 | PyTorch (CUDA 12.1) |
| `pandas` | 3.0.0 | 제출 파일 생성 |
| `PyYAML` | 6.0.3 | Config 파일 로드 |
| `scikit-learn` | 1.8.0 | 데이터 분할 |

### 1-3. GPU 확인

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

> ⚠️ GPU가 없으면 `--device cpu`를 사용할 수 있지만, 학습 시간이 매우 오래 걸립니다.

---

## 2. 🔄 전체 파이프라인 흐름도

### 실행 순서

```
┌─────────────────────────────────────────────────────────────┐
│                    데이터 준비 (1회만)                        │
│                                                             │
│  [0] COCO 생성 ─→ [1] Split ─→ [2] YOLO 변환                 │
│   763 JSON         80/20         images/ + labels/          │
│   → merged COCO    분할           → data.yaml                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    학습 & 평가                               │
│                                                             │
│  [3] Train ──→ [4] Evaluate ──→ [5] Submission              │
│   YOLO 학습       Val mAP         Test 추론                  │
│   → best.pt       확인             → submission.csv          │
└─────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                                   📤 Kaggle 제출
```

> 💡 **스크립트 번호가 실행 순서와 같습니다.** `0 → 1 → 2 → 3 → 4 → 5` 순서로 실행하세요.

### 파이프라인 요약 테이블

| 순서 | 스크립트 | 목적 | 입력 | 출력 | 데이터 형식 |
|:----:|----------|------|------|------|------------|
| 0 | `0_create_coco_format.py` | 어노테이션 통합 | `data/raw/train_annotations/` (763 JSON) | `cache/<run>/train_merged_coco.json` | 개별 JSON → COCO JSON |
| 1 | `1_splitting.py` | Train/Val 분할 | `cache/<run>/train_merged_coco.json` | `cache/<run>/splits/train_ids.txt`, `valid_ids.txt` | COCO JSON → Split 목록 |
| 2 | `2_prepare_yolo_dataset.py` | YOLO 변환 | merged COCO + splits + images | `datasets/pill_od_yolo_<run>/` | COCO → YOLO (normalized xywh) |
| 3 | `3_train.py` | 모델 학습 | `data.yaml` + Config | `runs/<run>/checkpoints/best.pt` | YOLO → 가중치 (.pt) |
| 4 | `4_evaluate.py` | Val 평가 | `best.pt` + `data.yaml` | `reports/eval_results.json` | 가중치 → mAP 메트릭 |
| 5 | `5_submission.py` | 제출 파일 | `best.pt` + test images | `submissions/submission.csv` | 가중치 → Kaggle CSV |

---

## 3. 📋 단계별 상세 실행 가이드

> 💡 모든 명령어는 **프로젝트 루트 디렉토리**에서 실행합니다.

### Stage 0: COCO 포맷 생성 (`0_create_coco_format.py`)

763개의 개별 JSON 어노테이션을 하나의 COCO 포맷 JSON으로 통합합니다.

```bash
python scripts/0_create_coco_format.py --run-name exp004
```

**입력:**
```
data/raw/train_annotations/     ← 114개 폴더, 763개 JSON
data/raw/train_images/          ← 232개 이미지 (참조용)
```

**출력:**
```
data/processed/cache/exp004/
├── train_merged_coco.json      ← 통합 COCO 어노테이션
├── image_id_map.json           ← 이미지 ID 매핑
├── category_id_to_name.json    ← 카테고리 ID → 약제명
├── label_map_full.json         ← YOLO index ↔ 원본 category_id 매핑
└── ...

artifacts/exp004/reports/
└── coco_merge_stats.json       ← 데이터셋 통계
```

> ✅ **확인 포인트**: `train_merged_coco.json`이 생성되었고, 통계에서 56개 클래스가 잡히는지 확인

---

### Stage 1: Train/Val 분할 (`1_splitting.py`)

232장의 이미지를 80:20으로 분할합니다 (이미지 내 객체 수 기준 층화추출).

```bash
python scripts/1_splitting.py --run-name exp004
```

**입력:**
```
data/processed/cache/exp004/train_merged_coco.json
```

**출력:**
```
data/processed/cache/exp004/splits/
├── split_train_valid.json      ← 분할 정보 (재현 가능)
├── train_ids.txt               ← Train 이미지 ID 목록
└── valid_ids.txt               ← Validation 이미지 ID 목록
```

> ✅ **확인 포인트**: Train ~186장, Val ~46장으로 분할되었는지 확인

---

### Stage 3: YOLO 데이터셋 준비 (`2_prepare_yolo_dataset.py`)

COCO 포맷을 YOLO 포맷으로 변환하고, 이미지와 라벨 파일을 정리합니다.

```bash
python scripts/2_prepare_yolo_dataset.py --run-name exp004
```

**입력:**
```
data/processed/cache/exp004/train_merged_coco.json
data/processed/cache/exp004/splits/split_train_valid.json
data/processed/cache/exp004/label_map_full.json
data/raw/train_images/
```

**출력:**
```
data/processed/datasets/pill_od_yolo_exp004/
├── data.yaml                   ← YOLO 데이터셋 설정 (자동 생성)
├── images/
│   ├── train/                  ← Train 이미지 (symlink)
│   └── val/                    ← Val 이미지 (symlink)
└── labels/
    ├── train/                  ← YOLO 라벨 (.txt, normalized xywh)
    └── val/
```

> ✅ **확인 포인트**: `data.yaml`이 생성되었고, `images/train/` 안에 이미지가 있는지 확인

---

### Stage 4: 모델 학습 (`3_train.py`)

YOLO 모델을 학습합니다.

```bash
# 기본 실행 (base.yaml 기본값 사용)
python scripts/3_train.py --run-name exp004

# Config YAML 지정 (권장)
python scripts/3_train.py --run-name exp004 --config configs/experiments/exp004_heavy_aug.yaml

# CLI 인자로 override
python scripts/3_train.py --run-name exp004 --model yolov8m --epochs 100 --batch 4

# 학습 재개 (중단된 경우)
python scripts/3_train.py --run-name exp004 --resume
```

**입력:**
```
data/processed/datasets/pill_od_yolo_exp004/data.yaml
configs/experiments/exp004_heavy_aug.yaml   (선택)
```

**출력:**
```
runs/exp004/
├── checkpoints/
│   ├── best.pt                 ← 최고 성능 가중치 ⭐
│   └── last.pt                 ← 마지막 에폭 가중치
├── config/
│   ├── config.json             ← 실험 설정 스냅샷 (재현용)
│   └── train_meta.json         ← 학습 메타 정보
└── train/                      ← Ultralytics 원본 출력
    ├── weights/
    │   ├── best.pt
    │   └── last.pt
    ├── results.csv             ← 에폭별 메트릭 ⭐
    ├── results.png             ← 학습 곡선 시각화
    ├── confusion_matrix.png    ← 혼동 행렬
    ├── BoxPR_curve.png         ← Precision-Recall 곡선
    ├── BoxF1_curve.png         ← F1 곡선
    ├── labels.jpg              ← 라벨 분포 시각화
    ├── train_batch*.jpg        ← 학습 배치 샘플
    └── val_batch*_pred.jpg     ← 검증 예측 시각화
```

> ✅ **확인 포인트**: `best.pt`가 생성되었고, `results.csv`에서 mAP 추이가 수렴하는지 확인

---

### Stage 5: 모델 평가 (`4_evaluate.py`)

Val 데이터셋으로 정밀 평가를 수행합니다.

```bash
python scripts/4_evaluate.py --run-name exp004 --config configs/experiments/exp004_heavy_aug.yaml
```

**입력:**
```
runs/exp004/checkpoints/best.pt
data/processed/datasets/pill_od_yolo_exp004/data.yaml
```

**출력:**
```
artifacts/exp004/reports/
└── eval_results.json           ← mAP@0.5, mAP@0.75, mAP@[0.5:0.95]

runs/exp004/eval/               ← Ultralytics 평가 출력
├── confusion_matrix.png
├── BoxPR_curve.png
├── BoxP_curve.png
├── BoxR_curve.png
└── BoxF1_curve.png
```

> ✅ **확인 포인트**: 터미널에 출력되는 `mAP@[0.5:0.95]` 값 확인. 이 값이 Kaggle 점수와 가장 유사합니다.

---

### Stage 6: 제출 파일 생성 (`5_submission.py`)

Test 이미지 842장에 대해 추론하고 Kaggle 제출 CSV를 생성합니다.

```bash
# 기본 제출
python scripts/5_submission.py --run-name exp004 --conf 0.25

# Config 지정 + 낮은 Confidence
python scripts/5_submission.py --run-name exp004 --config configs/experiments/exp004_heavy_aug.yaml --conf 0.20

# TTA (Test-Time Augmentation) 적용
python scripts/5_submission.py --run-name exp004 --conf 0.20 --tta
```

**입력:**
```
runs/exp004/checkpoints/best.pt
data/raw/test_images/                       ← 842장
data/processed/cache/exp004/label_map_full.json  ← YOLO idx → category_id 변환용
```

**출력:**
```
artifacts/exp004/submissions/
└── submission.csv              ← Kaggle 제출 파일 ⭐
```

**submission.csv 형식:**

| annotation_id | image_id | category_id | bbox_x | bbox_y | bbox_w | bbox_h | score |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0 | 1900 | 120.5 | 80.3 | 55.2 | 65.1 | 0.95 |
| 2 | 0 | 16548 | 300.2 | 150.4 | 60.3 | 70.5 | 0.89 |
| ... | ... | ... | ... | ... | ... | ... | ... |

> ⚠️ **중요**: `category_id`는 원본 ID (1900, 16548 등)입니다. YOLO 인덱스(0~55)가 아닙니다!

> ✅ **확인 포인트**: `submission.csv`에 `annotation_id`, `image_id`, `category_id` 등 8개 컬럼이 있고, `category_id`가 1900 이상의 값인지 확인

---

## 4. ⚙️ Config 시스템 & 실험 YAML 관리

### Config 상속 구조

```
configs/base.yaml              ← 모든 실험의 기본값 (56개 설정)
    ↑ _base_ 상속
configs/experiments/
├── exp001_baseline.yaml       ← base.yaml + 변경된 값만 작성
├── exp010_yolo11s.yaml        ← model_name만 "yolo11s"로 변경
└── ...
```

> 💡 실험 YAML에는 **base.yaml과 다른 값만** 작성하면 됩니다. 나머지는 자동 상속!

### 새 실험 YAML 만들기

```bash
# 1. 템플릿 복사
cp configs/experiments/_TEMPLATE.yaml configs/experiments/exp004_my_experiment.yaml

# 2. 파일을 열어서 수정
```

**YAML 예시:**
```yaml
_base_: "../base.yaml"        # 반드시 이 줄 유지!

experiment:
  id: "exp004"
  name: "my_experiment"
  description: "무엇을 테스트하는 실험인지 설명"
  author: "@이름"
  created: "2026-02-06"

# base.yaml에서 변경할 값만 아래에 작성
train:
  model_name: "yolo11s"       # 모델 변경
  imgsz: 1024                 # 해상도 변경
  epochs: 100
  batch: 4
```

### 현재 실험 YAML 목록

| 파일 | 모델 | 해상도 | Epochs | 핵심 변경사항 |
|------|------|:------:|:------:|--------------|
| `exp001_baseline.yaml` | YOLOv8s | 768 | 80 | 기본 실험 |
| `exp002_whitelist.yaml` | YOLOv8s | 768 | 80 | 테스트 40클래스만 |
| `exp003_yolov8m.yaml` | YOLOv8m | 768 | 80 | 더 큰 모델 |
| `exp004_heavy_aug.yaml` | YOLOv8s | 768 | 100 | 강한 증강 (mosaic+mixup) |
| `exp005_imgsz1024.yaml` | YOLOv8s | 1024 | 80 | 고해상도 |
| `exp006_high_conf.yaml` | YOLOv8s | 768 | 80 | 높은 conf threshold |
| `exp007_final.yaml` | YOLOv8s | 1024 | 120 | 최종 조합 |
| `exp010_yolo11s.yaml` | **YOLO11s** | 768 | 100 | YOLO11 아키텍처 |
| `exp012_yolo11s_1024.yaml` | **YOLO11s** | 1024 | 100 | YOLO11 + 고해상도 |
| `exp020_stage1.yaml` | **YOLO11s** | 1024 | 150 | 2단계 학습 - 1단계 (강한 증강) |
| `exp020_stage2.yaml` | **YOLO11s** | 1024 | 60 | 2단계 학습 - 2단계 (증강 OFF, lr=5e-5) |

---

## 5. 📦 가중치 파일(Weights) 관리

### 가중치 저장 위치

학습 완료 후 가중치는 **두 곳**에 저장됩니다:

```
runs/<run_name>/
├── checkpoints/                ← 📁 스크립트가 복사한 가중치 (이 경로를 사용!)
│   ├── best.pt                 ← 최고 성능 모델 (~23MB)
│   └── last.pt                 ← 마지막 에폭 모델 (~23MB)
│
└── train/weights/              ← 📁 Ultralytics가 생성한 원본 (참고용)
    ├── best.pt
    └── last.pt
```

> 💡 평가(`4_evaluate.py`)와 제출(`5_submission.py`)은 자동으로 `runs/<run_name>/checkpoints/best.pt`를 참조합니다.

### 팀원 간 가중치 공유

```bash
# 1. 공유할 가중치 파일 위치 확인
ls runs/exp010/checkpoints/best.pt

# 2. 팀원에게 전달 방법
#    - Google Drive / 팀 공유 폴더에 업로드
#    - 파일명 규칙: {실험명}_{모델}_{해상도}_best.pt
#    예: exp010_yolo11s_768_best.pt
```

### 다른 실험의 가중치 사용 (2단계 학습)

```bash
# --ckpt-from 옵션으로 다른 run의 체크포인트 로드
python scripts/3_train.py \
    --run-name exp020_s2 \
    --ckpt-from runs/exp020_s1/checkpoints/best.pt \
    --config configs/experiments/exp020_stage2.yaml
```

> ⚠️ `--ckpt-from`은 **새 학습의 시작점**으로 다른 가중치를 로드합니다. `--resume`은 **같은 실험의 중단된 학습을 이어가는 것**입니다.

---

## 6. 📊 평가 결과물 및 해석

### 자동 생성 시각화 목록

학습 완료 후 `runs/<run_name>/train/` 폴더에 다음 파일들이 자동 생성됩니다:

| 파일 | 설명 | 확인 포인트 |
|------|------|------------|
| `results.png` | 학습 곡선 (loss, mAP 추이) | loss 수렴, mAP 상승 확인 |
| `results.csv` | 에폭별 수치 데이터 | 최고 mAP 에폭 확인 |
| `confusion_matrix.png` | 클래스별 예측 정확도 | 약한 클래스 파악 |
| `confusion_matrix_normalized.png` | 정규화된 혼동 행렬 | 비율 기반 분석 |
| `BoxPR_curve.png` | Precision-Recall 곡선 | 클래스별 AP 비교 |
| `BoxF1_curve.png` | F1 Score 곡선 | 최적 conf threshold 탐색 |
| `BoxP_curve.png` | Precision 곡선 | 오탐(FP) 경향 확인 |
| `BoxR_curve.png` | Recall 곡선 | 미탐(FN) 경향 확인 |
| `labels.jpg` | 라벨 분포 시각화 | 클래스 불균형 확인 |
| `val_batch*_pred.jpg` | Val 예측 시각화 | 실제 검출 품질 눈으로 확인 |
| `val_batch*_labels.jpg` | Val 정답 시각화 | Ground Truth와 비교 |

### results.csv 핵심 컬럼

| 컬럼 | 설명 | 중요도 |
|------|------|:------:|
| `epoch` | 에폭 번호 | - |
| `train/box_loss` | bbox 위치 loss | ⭐ |
| `train/cls_loss` | 분류 loss | ⭐ |
| `train/dfl_loss` | Distribution Focal Loss | ⭐ |
| `metrics/precision(B)` | Precision | ⭐ |
| `metrics/recall(B)` | Recall | ⭐ |
| `metrics/mAP50(B)` | mAP@0.5 | ⭐⭐ |
| `metrics/mAP50-95(B)` | mAP@[0.5:0.95] | ⭐⭐⭐ (대회 지표!) |

### mAP 확인 방법

```bash
# 방법 1: eval_results.json 확인
cat artifacts/exp004/reports/eval_results.json

# 출력 예시:
# {
#   "mAP_50": 0.9723,
#   "mAP_75": 0.9651,
#   "mAP_50_95": 0.9548
# }

# 방법 2: results.csv에서 최고 mAP 에폭 확인
python -c "
import pandas as pd
df = pd.read_csv('runs/exp004/train/results.csv')
df.columns = [c.strip() for c in df.columns]
best = df.loc[df['metrics/mAP50-95(B)'].idxmax()]
print(f'Best Epoch: {int(best[\"epoch\"])}')
print(f'mAP@0.5: {best[\"metrics/mAP50(B)\"]:.4f}')
print(f'mAP@[0.5:0.95]: {best[\"metrics/mAP50-95(B)\"]:.4f}')
"
```

---

## 7. 🚀 새 실험 시작하기 — Step-by-Step 예시

> 아래는 `exp004_heavy_aug` 실험을 처음부터 제출까지 진행하는 전체 흐름입니다.

### Step 1: 실험 YAML 생성

```bash
cp configs/experiments/_TEMPLATE.yaml configs/experiments/exp004_heavy_aug.yaml
```

`exp004_heavy_aug.yaml`을 열어서 수정:

```yaml
_base_: "../base.yaml"

experiment:
  id: "exp004"
  name: "heavy_augmentation"
  description: "강한 증강으로 데이터 부족 보완"
  author: "@나"
  created: "2026-02-06"

train:
  model_name: "yolov8s"
  epochs: 100
  batch: 8
  mosaic: 1.0
  mixup: 0.15
  copy_paste: 0.1
  degrees: 5.0
```

### Step 2: 데이터 준비 (0 → 1 → 2)

```bash
python scripts/0_create_coco_format.py --run-name exp004
python scripts/1_splitting.py --run-name exp004
python scripts/2_prepare_yolo_dataset.py --run-name exp004
```

> ✅ 3개 스크립트 모두 에러 없이 완료되면 데이터 준비 끝!

### Step 3: 학습

```bash
python scripts/3_train.py --run-name exp004 --config configs/experiments/exp004_heavy_aug.yaml
```

> ⏱ YOLOv8s / 768px / 100ep 기준 약 1~2시간 (GPU 성능에 따라 다름)

### Step 4: 평가

```bash
python scripts/4_evaluate.py --run-name exp004 --config configs/experiments/exp004_heavy_aug.yaml
```

출력에서 `mAP@[0.5:0.95]` 값을 확인합니다.

### Step 5: 제출 파일 생성

```bash
python scripts/5_submission.py --run-name exp004 --config configs/experiments/exp004_heavy_aug.yaml --conf 0.25
```

### Step 6: Kaggle 제출

1. `artifacts/exp004/submissions/submission.csv` 파일을 찾습니다
2. [Kaggle 대회 페이지](https://www.kaggle.com/) → Submit Predictions
3. CSV 파일 업로드
4. Public Score 확인!

---

## 8. 🔬 고급: 2단계 학습 & TTA

### 2단계 학습 전략

> 💡 **핵심 아이디어**: 1단계에서 강한 증강으로 일반화 → 2단계에서 증강 OFF + 극저 LR로 bbox 정밀도 미세조정

이 전략은 팀원이 Kaggle 점수 **0.96849**를 달성한 검증된 방법입니다.

**왜 효과적인가?**
- 증강은 데이터 다양성을 높이지만 bbox 좌표를 왜곡시킵니다
- 2단계에서 원본 이미지로만 학습하면 bbox가 더 정확해집니다
- `mAP@[0.75:0.95]`는 bbox 정확도가 점수에 직결됩니다

#### 실행 방법

```bash
# ========= 1단계: 강한 증강으로 일반화 학습 =========
python scripts/0_create_coco_format.py --run-name exp020_s1
python scripts/1_splitting.py --run-name exp020_s1
python scripts/2_prepare_yolo_dataset.py --run-name exp020_s1
python scripts/3_train.py --run-name exp020_s1 --config configs/experiments/exp020_stage1.yaml

# (선택) 1단계 결과 확인
python scripts/4_evaluate.py --run-name exp020_s1 --config configs/experiments/exp020_stage1.yaml
python scripts/5_submission.py --run-name exp020_s1 --conf 0.25

# ========= 2단계: 증강 OFF + bbox 미세조정 =========
python scripts/0_create_coco_format.py --run-name exp020_s2
python scripts/1_splitting.py --run-name exp020_s2
python scripts/2_prepare_yolo_dataset.py --run-name exp020_s2

# ⭐ 핵심: --ckpt-from으로 1단계 best.pt를 불러옵니다!
python scripts/3_train.py \
    --run-name exp020_s2 \
    --ckpt-from runs/exp020_s1/checkpoints/best.pt \
    --config configs/experiments/exp020_stage2.yaml

# 평가 + 제출
python scripts/4_evaluate.py --run-name exp020_s2 --config configs/experiments/exp020_stage2.yaml
python scripts/5_submission.py --run-name exp020_s2 --conf 0.20
```

### TTA (Test-Time Augmentation)

TTA는 추론 시 이미지를 다양한 스케일/반전으로 여러 번 예측한 뒤 결과를 앙상블합니다.

```bash
# --tta 플래그 추가만 하면 됩니다
python scripts/5_submission.py --run-name exp020_s2 --conf 0.20 --tta
```

> ✅ TTA는 학습 없이 추론 시간만 약간 늘어나며, 보통 **1~3% mAP 향상**을 기대할 수 있습니다.

### conf threshold 스윕 (최적 값 탐색)

```bash
# 여러 conf 값으로 제출 파일 생성 → 각각 Kaggle에 제출해서 비교
python scripts/5_submission.py --run-name exp020_s2 --conf 0.15 --tta
python scripts/5_submission.py --run-name exp020_s2 --conf 0.20 --tta
python scripts/5_submission.py --run-name exp020_s2 --conf 0.25 --tta
python scripts/5_submission.py --run-name exp020_s2 --conf 0.30 --tta
```

> 💡 `mAP`는 PR 곡선의 넓이이므로, **conf를 낮추면** Recall이 올라가 mAP가 높아지는 경향이 있습니다. 보통 0.20~0.25가 최적입니다.

---

## 9. 🔧 Troubleshooting

### 자주 발생하는 에러

| 에러 | 원인 | 해결 |
|------|------|------|
| `❌ data.yaml 없음` | 2번 스크립트 미실행 | `scripts/2_prepare_yolo_dataset.py` 먼저 실행 |
| `❌ 체크포인트 없음` | 3번 스크립트 미실행 | `scripts/3_train.py` 먼저 실행 |
| `❌ Label map 없음` | 1번 스크립트 미실행 | `scripts/0_create_coco_format.py` 먼저 실행 |
| `CUDA out of memory` | batch 크기 초과 | `--batch 4` 또는 `--batch 2`로 줄이기 |
| `Unknown class index` | label_map 불일치 | 데이터 준비(0→1→2)를 같은 run-name으로 재실행 |
| `submission.csv에 category_id가 0~55` | 버전 오류 | 최신 `5_submission.py` 사용 확인 (idx2id 변환 포함) |

### 학습이 중단되었을 때

```bash
# --resume 플래그로 이어서 학습
python scripts/3_train.py --run-name exp004 --resume
```

### Run 디렉토리 구조 확인

```bash
# 특정 실험의 출력물 확인
ls runs/exp004/checkpoints/
ls runs/exp004/train/
ls artifacts/exp004/submissions/
```

### 전체 파이프라인 한 번에 실행

```bash
# 한 줄로 전체 파이프라인 실행 (bash)
RUN=exp004 && CONFIG=configs/experiments/exp004_heavy_aug.yaml && \
python scripts/0_create_coco_format.py --run-name $RUN && \
python scripts/1_splitting.py --run-name $RUN && \
python scripts/2_prepare_yolo_dataset.py --run-name $RUN && \
python scripts/3_train.py --run-name $RUN --config $CONFIG && \
python scripts/4_evaluate.py --run-name $RUN --config $CONFIG && \
python scripts/5_submission.py --run-name $RUN --config $CONFIG --conf 0.25
```

```powershell
# PowerShell 버전
$RUN="exp004"; $CONFIG="configs/experiments/exp004_heavy_aug.yaml"
python scripts/0_create_coco_format.py --run-name $RUN; if($?) {
python scripts/1_splitting.py --run-name $RUN }; if($?) {
python scripts/2_prepare_yolo_dataset.py --run-name $RUN }; if($?) {
python scripts/3_train.py --run-name $RUN --config $CONFIG }; if($?) {
python scripts/4_evaluate.py --run-name $RUN --config $CONFIG }; if($?) {
python scripts/5_submission.py --run-name $RUN --config $CONFIG --conf 0.25 }
```

---

> 📝 **문서 최종 업데이트**: 2026-02-06 | Team #4
