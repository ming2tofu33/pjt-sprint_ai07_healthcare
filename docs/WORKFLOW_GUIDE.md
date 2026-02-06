# 🏥 경구약제 객체 검출 — 프로젝트 워크플로우 가이드

> **Team #4 | Kaggle Pill Detection Competition**  
> 타겟 지표: **mAP@[0.75:0.95]** | 모델: Ultralytics YOLO

---

## 📌 목차

1. [실행 전 필수 준비물](#1--실행-전-필수-준비물-prerequisites)
2. [전체 파이프라인 흐름도](#2--전체-파이프라인-흐름도)
3. [단계별 상세 실행 가이드 (Stage 0~5)](#3--단계별-상세-실행-가이드-stage-05)
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
    ├── train_annotations/          ← 어노테이션 (114개 폴더, 763개 JSON)
    └── test_images/                ← 테스트 이미지 (842장, .png)
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

> ✅ **핵심 패키지 버전 (requirements.txt 기준)**

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

### 실행 순서 (Stage 0~5)

```
┌─────────────────────────────────────────────────────────────┐
│                      데이터 준비                              │
│                                                             │
│  [0] COCO 생성 ─→ [1] Split ─→ [2] YOLO 변환                │
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

### 파이프라인 요약 테이블

| Stage | 스크립트 | 목적 | 입력 | 출력 | 데이터 형식 |
|:----:|----------|------|------|------|------------|
| 0 | `0_create_coco_format.py` | 어노테이션 통합 | `data/raw/train_annotations/` | `data/processed/cache/<run>/train_merged_coco.json` + `label_map_full.json` | JSON → COCO JSON |
| 1 | `1_splitting.py` | Train/Val 분할 | `cache/<run>/train_merged_coco.json` | `cache/<run>/splits/` | COCO JSON → Split 목록 |
| 2 | `2_prepare_yolo_dataset.py` | YOLO 변환 | COCO + splits + images | `data/processed/datasets/pill_od_yolo_<run>/` | COCO → YOLO (normalized xywh) |
| 3 | `3_train.py` | 모델 학습 | `data.yaml` + Config | `runs/<run>/checkpoints/best.pt` | YOLO → 가중치 (.pt) |
| 4 | `4_evaluate.py` | Val 평가 | `best.pt` + `data.yaml` | `artifacts/<run>/reports/eval_results.json` | 가중치 → mAP 메트릭 |
| 5 | `5_submission.py` | 제출 파일 | `best.pt` + test images | `artifacts/<run>/submissions/submission.csv` | 가중치 → Kaggle CSV |

---

## 3. 📋 단계별 상세 실행 가이드 (Stage 0~5)

> 💡 모든 명령어는 **프로젝트 루트 디렉토리**에서 실행합니다.

### Stage 0: COCO 포맷 생성 (`0_create_coco_format.py`)

763개의 개별 JSON 어노테이션을 하나의 COCO 포맷 JSON으로 통합합니다.

```bash
python scripts/0_create_coco_format.py --run-name exp004
# (권장) 실험 YAML 사용
python scripts/0_create_coco_format.py --run-name exp004 --config configs/experiments/exp004_heavy_aug.yaml
```

**입력:**
```
data/raw/train_annotations/
data/raw/train_images/
```

**출력:**
```
data/processed/cache/exp004/
├── train_merged_coco.json
├── image_id_map.json
├── category_id_to_name.json
├── label_map_full.json
└── label_map_whitelist.json   (옵션)

artifacts/exp004/reports/
└── coco_merge_stats.json
```

> ✅ **확인 포인트**: `label_map_full.json`의 `num_classes`(전체 56 클래스)가 맞는지 확인

---

### Stage 1: Train/Val 분할 (`1_splitting.py`)

이미지를 80:20으로 분할합니다 (객체 수 기준 stratified).

```bash
python scripts/1_splitting.py --run-name exp004
python scripts/1_splitting.py --run-name exp004 --config configs/experiments/exp004_heavy_aug.yaml
```

**출력:**
```
data/processed/cache/exp004/splits/
├── split_train_valid.json
├── train_ids.txt
└── valid_ids.txt
```

> ✅ **확인 포인트**: Train/Val 이미지 개수가 합쳐서 232인지 확인

---

### Stage 2: YOLO 데이터셋 준비 (`2_prepare_yolo_dataset.py`)

COCO 포맷을 YOLO 포맷으로 변환하고 이미지/라벨을 정리합니다.

```bash
python scripts/2_prepare_yolo_dataset.py --run-name exp004
# (선택) 심볼릭 링크 사용 (Windows는 권한 필요할 수 있음)
python scripts/2_prepare_yolo_dataset.py --run-name exp004 --symlink
```

**출력:**
```
data/processed/datasets/pill_od_yolo_exp004/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

> ✅ **확인 포인트**: `data.yaml` 생성 및 `images/train/`에 이미지가 있는지 확인

---

### Stage 3: 모델 학습 (`3_train.py`)

```bash
python scripts/3_train.py --run-name exp004 --config configs/experiments/exp004_heavy_aug.yaml

# CLI override 예시
python scripts/3_train.py --run-name exp004 --model yolov8m --epochs 100 --batch 4

# 학습 재개
python scripts/3_train.py --run-name exp004 --resume

# 2단계 학습 (다른 run의 체크포인트로 시작)
python scripts/3_train.py --run-name exp020_s2 --ckpt-from runs/exp020_s1/checkpoints/best.pt --config configs/experiments/exp020_stage2.yaml
```

**출력:**
```
runs/exp004/
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── config/
│   ├── config.json
│   └── train_meta.json
└── train/            # Ultralytics 기본 출력 (results.csv, plots 등)
```

---

### Stage 4: 모델 평가 (`4_evaluate.py`)

```bash
python scripts/4_evaluate.py --run-name exp004 --config configs/experiments/exp004_heavy_aug.yaml
```

**출력:**
```
artifacts/exp004/reports/
└── eval_results.json

runs/exp004/eval/     # Ultralytics 평가 출력
```

**평가 지표:**
- `mAP@[0.75:0.95]` (대회 공식)
- `mAP@0.5`, `mAP@0.75` (참고용)

---

### Stage 5: 제출 파일 생성 (`5_submission.py`)

```bash
python scripts/5_submission.py --run-name exp004 --config configs/experiments/exp004_heavy_aug.yaml --conf 0.25

# TTA 적용
python scripts/5_submission.py --run-name exp004 --conf 0.20 --tta
```

**출력:**
```
artifacts/exp004/submissions/
└── submission.csv
```

**submission.csv 형식:**

| annotation_id | image_id | category_id | bbox_x | bbox_y | bbox_w | bbox_h | score |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0 | 1900 | 120.5 | 80.3 | 55.2 | 65.1 | 0.95 |

> ⚠️ **중요**
> - `category_id`는 **원본 COCO ID**입니다 (YOLO 인덱스가 아님)
> - `bbox_*`는 **절대 픽셀 좌표의 xywh**입니다 (정규화 아님)

---

## 4. ⚙️ Config 시스템 & 실험 YAML 관리

### Config 상속 구조

```
configs/base.yaml              ← 기본값
    ↑ _base_ 상속
configs/experiments/*.yaml     ← 변경값만 override
```

**YAML 예시:**
```yaml
_base_: "../base.yaml"

experiment:
  id: "exp004"
  name: "heavy_augmentation"
  created: "2026-02-06"

train:
  model_name: "yolov8s"
  epochs: 100
  batch: 8
```

### 현재 실험 YAML 목록

| 파일 | 모델 | 해상도 | Epochs | 핵심 변경사항 |
|------|------|:------:|:------:|--------------|
| `exp001_baseline.yaml` | YOLOv8s | 768 | 80 | 기본 실험 |
| `exp002_whitelist.yaml` | YOLOv8s | 768 | 80 | 테스트 40클래스만 |
| `exp003_yolov8m.yaml` | YOLOv8m | 768 | 80 | 더 큰 모델 |
| `exp004_heavy_aug.yaml` | YOLOv8s | 768 | 100 | 강한 증강 |
| `exp005_imgsz1024.yaml` | YOLOv8s | 1024 | 80 | 고해상도 |
| `exp006_high_conf.yaml` | YOLOv8s | 768 | 80 | 높은 conf |
| `exp007_final.yaml` | YOLOv8s | 1024 | 120 | 최종 조합 |
| `exp010_yolo11s.yaml` | **YOLO11s** | 768 | 100 | YOLO11 아키텍처 |
| `exp012_yolo11s_1024.yaml` | **YOLO11s** | 1024 | 100 | YOLO11 + 고해상도 |
| `exp020_stage1.yaml` | **YOLO11s** | 1024 | 150 | 2단계 학습 1단계 |
| `exp020_stage2.yaml` | **YOLO11s** | 1024 | 60 | 2단계 학습 2단계 |

---

## 5. 📦 가중치 파일(Weights) 관리

학습 완료 후 가중치는 **두 곳**에 저장됩니다:

```
runs/<run_name>/
├── checkpoints/                ← 스크립트가 복사한 가중치 (이 경로 사용)
│   ├── best.pt
│   └── last.pt
└── train/weights/              ← Ultralytics 원본 출력
    ├── best.pt
    └── last.pt
```

> 평가(`4_evaluate.py`)와 제출(`5_submission.py`)은 `runs/<run_name>/checkpoints/best.pt`를 참조합니다.

---

## 6. 📊 평가 결과물 및 해석

### 평가 결과 확인

```bash
cat artifacts/exp004/reports/eval_results.json
```

**예시 출력:**
```
{
  "mAP_50": 0.9723,
  "mAP_75": 0.9651,
  "mAP_50_95": 0.9548
}
```

### Ultralytics 학습 로그 확인

- `runs/<run>/train/results.csv`에서 `metrics/mAP50-95(B)` 컬럼을 확인
- 최고값 에폭이 실질적인 최고 성능 구간입니다

---

## 7. 🚀 새 실험 시작하기 — Step-by-Step 예시

### Step 1: 실험 YAML 생성

```bash
cp configs/experiments/_TEMPLATE.yaml configs/experiments/exp004_heavy_aug.yaml
```

### Step 2: 데이터 준비 (Stage 0 → 1 → 2)

```bash
python scripts/0_create_coco_format.py --run-name exp004
python scripts/1_splitting.py --run-name exp004
python scripts/2_prepare_yolo_dataset.py --run-name exp004
```

### Step 3: 학습 (Stage 3)

```bash
python scripts/3_train.py --run-name exp004 --config configs/experiments/exp004_heavy_aug.yaml
```

### Step 4: 평가 (Stage 4)

```bash
python scripts/4_evaluate.py --run-name exp004 --config configs/experiments/exp004_heavy_aug.yaml
```

### Step 5: 제출 (Stage 5)

```bash
python scripts/5_submission.py --run-name exp004 --config configs/experiments/exp004_heavy_aug.yaml --conf 0.25
```

---

## 8. 🔬 고급: 2단계 학습 & TTA

### 2단계 학습 전략

> 1단계에서 강한 증강으로 일반화 → 2단계에서 증강 OFF + 극저 LR로 bbox 정밀도 미세조정

#### 실행 방법

```bash
# ========= 1단계 =========
python scripts/0_create_coco_format.py --run-name exp020_s1
python scripts/1_splitting.py --run-name exp020_s1
python scripts/2_prepare_yolo_dataset.py --run-name exp020_s1
python scripts/3_train.py --run-name exp020_s1 --config configs/experiments/exp020_stage1.yaml

# (선택) 1단계 평가/제출
python scripts/4_evaluate.py --run-name exp020_s1 --config configs/experiments/exp020_stage1.yaml
python scripts/5_submission.py --run-name exp020_s1 --config configs/experiments/exp020_stage1.yaml --conf 0.25

# ========= 2단계 =========
python scripts/0_create_coco_format.py --run-name exp020_s2
python scripts/1_splitting.py --run-name exp020_s2
python scripts/2_prepare_yolo_dataset.py --run-name exp020_s2

# ⭐ 핵심: --ckpt-from으로 1단계 best.pt 로드
python scripts/3_train.py \
    --run-name exp020_s2 \
    --ckpt-from runs/exp020_s1/checkpoints/best.pt \
    --config configs/experiments/exp020_stage2.yaml

# 평가 + 제출
python scripts/4_evaluate.py --run-name exp020_s2 --config configs/experiments/exp020_stage2.yaml
python scripts/5_submission.py --run-name exp020_s2 --config configs/experiments/exp020_stage2.yaml --conf 0.20
```

### TTA (Test-Time Augmentation)

```bash
python scripts/5_submission.py --run-name exp020_s2 --conf 0.20 --tta
```

---

## 9. 🔧 Troubleshooting

| 에러 | 원인 | 해결 |
|------|------|------|
| `❌ data.yaml 없음` | Stage 2 미실행 | `scripts/2_prepare_yolo_dataset.py` 실행 |
| `❌ 체크포인트 없음` | Stage 3 미실행 | `scripts/3_train.py` 실행 |
| `❌ Label map 없음` | Stage 0 미실행 | `scripts/0_create_coco_format.py` 실행 |
| `CUDA out of memory` | batch 크기 초과 | `--batch 4` 또는 `--batch 2`로 줄이기 |
| `Unknown class index` | label_map 불일치 | 동일 run-name으로 Stage 0~2 재실행 |
| `category_id가 0~55` | 매핑 미적용 | 최신 `label_map_full.json` 확인 |

### Run 디렉토리 구조 확인

```bash
ls runs/exp004/checkpoints/
ls runs/exp004/train/
ls artifacts/exp004/submissions/
```

---

> 📝 **문서 최종 업데이트**: 2026-02-06 | Team #4
