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

### Stage 2: 학습 및 평가 (TODO)

```bash
# 3. 모델 학습
python scripts/3_train.py --run-name exp_baseline_v1

# 4. 모델 평가
python scripts/4_evaluate.py --run-name exp_baseline_v1

# 5. 제출 파일 생성
python scripts/5_submission.py --run-name exp_baseline_v1
```

---

## 📄 스크립트 상세

### `1_create_coco_format.py`

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

### `0_splitting.py`

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

## 🚀 다음 단계 (TODO)

### Stage 2: 학습 및 평가
- [ ] `3_train.py` - YOLO 학습 스크립트
- [ ] `4_evaluate.py` - mAP 평가 스크립트
- [ ] `5_submission.py` - Kaggle 제출 파일 생성

### 개선 사항
- [ ] K-Fold split 구현 (`0_splitting.py`)
- [ ] Multi-GPU 지원 (`3_train.py`)
- [ ] TTA (Test-Time Augmentation) 지원 (`5_submission.py`)
- [ ] Config validation (YAML schema)

---

**구현 완료**: 2026-02-05  
**담당**: @DM  
**상태**: Stage 1 완료, Stage 2 대기
