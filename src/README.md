# src/ - Core Modules

## 📌 개요

재사용 가능한 핵심 모듈. scripts는 이 모듈들을 사용하여 구현됩니다.

---

## 📂 구조

```
src/
├── utils.py          # ✅ 핵심 — 경로, Config 상속/병합, seed, 경로 헬퍼, 결과 기록
├── data_loader.py    # [DEPRECATED] 데이터 로딩 및 전처리
├── model.py          # [DEPRECATED] YOLO 모델 래퍼
├── trainer.py        # [DEPRECATED] 학습 프로세스 관리
└── inference.py      # [DEPRECATED] 추론 및 결과 처리
```

> **NOTE**: `utils.py`만 현재 파이프라인(scripts/)에서 활발히 사용됩니다.
> 나머지 4개 모듈은 참고용으로 보존되며, 실제 학습/추론은 scripts에서 ultralytics.YOLO를 직접 호출합니다.

---

## 📄 모듈 상세

### `utils.py` (핵심 모듈)
**공통 유틸리티 함수**

- `setup_project_paths()`: 프로젝트 경로 설정 및 폴더 생성
- `set_seed()`: 재현성을 위한 seed 고정
- `load_config()`: Config 로드 (JSON/YAML, `_base_` 상속 지원)
- `merge_configs()`: 두 config dict 깊은 병합 (base + override)
- `get_dataset_dir()`: YOLO 데이터셋 디렉토리 경로 헬퍼
- `get_data_yaml()`: data.yaml 경로 헬퍼
- `save_config()`: Config 저장 (JSON/YAML)
- `create_run_manifest()`: 실험 메타데이터 생성
- `record_result()`: 결과 기록 (CSV + JSONL)

**사용 예시**:
```python
from src.utils import setup_project_paths, set_seed, load_config, get_data_yaml

# 경로 설정
paths = setup_project_paths(run_name="exp001", create_dirs=True)

# Seed 고정
set_seed(42, deterministic=True)

# Config 로드 (_base_ 상속 자동 처리)
config = load_config("configs/experiments/exp001_baseline.yaml")

# YOLO data.yaml 경로
data_yaml = get_data_yaml(paths)
```

---

### `data_loader.py` [DEPRECATED]
**데이터셋 로딩 및 전처리** (현재 파이프라인에서 미사용)

#### COCODataset
COCO 포맷 데이터셋 로더 (PyTorch Dataset)

```python
from src.data_loader import COCODataset, load_split_ids

# Split IDs 로드
train_ids = load_split_ids("data/processed/cache/exp001/splits/train_ids.txt")

# Dataset 생성
dataset = COCODataset(
    coco_json_path="data/processed/cache/exp001/train_merged_coco.json",
    image_root="data/raw/train_images",
    split_ids=train_ids,
)

# 사용
image, target = dataset[0]
print(target["boxes"], target["labels"])
```

#### YOLODatasetWrapper
YOLO 데이터셋 래퍼 (data.yaml 기반)

```python
from src.data_loader import YOLODatasetWrapper

wrapper = YOLODatasetWrapper("data/processed/datasets/exp001_yolo/data.yaml")
print(wrapper.get_num_classes())  # 56
print(wrapper.get_class_names())  # ['class1', 'class2', ...]
```

---

### `model.py` [DEPRECATED]
**YOLO 모델 래퍼** (현재 파이프라인에서 미사용)

#### YOLOModel
Ultralytics YOLO 모델 관리

```python
from src.model import YOLOModel

# 모델 생성
model = YOLOModel(model_name="yolov8s.pt", device="0")

# 학습
results = model.train(
    data_yaml="data/processed/datasets/exp001_yolo/data.yaml",
    epochs=80,
    imgsz=768,
    batch=8,
)

# 추론
results = model.predict(source="data/raw/test_images/", conf=0.25)

# 평가
val_results = model.validate()
```

---

### `trainer.py` [DEPRECATED]
**학습 프로세스 관리** (현재 파이프라인에서 미사용)

#### Trainer
Config 기반 학습 실행

```python
from src.trainer import Trainer

# Trainer 생성
trainer = Trainer(
    run_name="exp001",
    config="configs/experiments/exp001_baseline.yaml",
    device="0",
)

# 학습 실행
results = trainer.train(
    data_yaml="data/processed/datasets/exp001_yolo/data.yaml",
)

# 평가
eval_results = trainer.evaluate(split="val")

# 체크포인트 로드
trainer.load_checkpoint("runs/exp001/checkpoints/best.pt")
```

**주요 기능**:
- Config 기반 자동 설정
- 재현성 보장 (seed, deterministic)
- Run manifest 생성
- 결과 자동 기록

---

### `inference.py` [DEPRECATED]
**추론 및 결과 처리** (현재 파이프라인에서 미사용, 5_submission.py가 직접 처리)

#### Inferencer
추론 실행 및 제출 파일 생성

```python
from src.inference import Inferencer

# Inferencer 생성
inferencer = Inferencer(
    checkpoint_path="runs/exp001/checkpoints/best.pt",
    device="0",
)

# 추론 실행
results = inferencer.predict(
    source="data/raw/test_images/",
    conf=0.25,
    iou=0.45,
)

# 제출 파일 생성
inferencer.create_submission_csv(
    results=results,
    output_path="artifacts/exp001/submissions/submission.csv",
    top_k=4,  # Top-4 only
)

# 검증
validation = inferencer.validate_submission_csv(
    "artifacts/exp001/submissions/submission.csv"
)
print(validation["valid"])  # True/False
```

**주요 기능**:
- Top-K 필터링
- submission.csv 생성
- 제출 파일 검증
- DataFrame 변환

---

## 🔗 모듈 간 관계

```
scripts/ (0~5_*.py)
  ↓ (직접 사용)
src/
  └── utils.py          ← 전 스크립트가 사용 (경로, config, seed, 기록)

[DEPRECATED - 참고용]
  ├── data_loader.py    ← COCODataset, YOLO wrapper
  ├── model.py          ← YOLOModel (Ultralytics 래퍼)
  ├── trainer.py        ← utils + model (학습 실행)
  └── inference.py      ← model (추론 실행)
```

---

## ✅ 전체 워크플로우 예시

```python
# 1. 경로 설정
from src.utils import setup_project_paths, set_seed, load_config

paths = setup_project_paths("exp001", create_dirs=True)
set_seed(42)
config = load_config("configs/experiments/exp001_baseline.yaml")

# 2. 데이터 로딩
from src.data_loader import YOLODatasetWrapper

yolo_data = YOLODatasetWrapper(paths["DATA"] / "datasets/exp001_yolo/data.yaml")
print(f"Classes: {yolo_data.get_num_classes()}")

# 3. 학습
from src.trainer import Trainer

trainer = Trainer(run_name="exp001", config=config)
trainer.train(data_yaml=yolo_data.get_data_yaml_path())

# 4. 평가
eval_results = trainer.evaluate(split="val")

# 5. 추론
from src.inference import Inferencer

inferencer = Inferencer(checkpoint_path=paths["CKPT"] / "best.pt")
results = inferencer.predict_and_filter_top_k(
    source=paths["TEST_IMAGES"],
    top_k=4,
    conf=0.25,
)

# 6. 제출 파일 생성
inferencer.create_submission_csv(
    results=results,
    output_path=paths["SUBMISSIONS"] / "submission.csv",
    top_k=4,
)
```

---

## 🧪 테스트

각 모듈은 독립적으로 테스트 가능:

```bash
# utils.py 테스트
python src/utils.py

# data_loader.py 테스트
python src/data_loader.py

# model.py 테스트
python src/model.py

# trainer.py 테스트
python src/trainer.py

# inference.py 테스트
python src/inference.py
```

---

## 📊 코드 통계

- **utils.py**: 19KB, 600+ lines
- **data_loader.py**: 8.4KB, 280+ lines
- **model.py**: 9.1KB, 300+ lines
- **trainer.py**: 10KB, 330+ lines
- **inference.py**: 9.9KB, 320+ lines

**총 코드량**: ~56KB, 1,830+ lines

---

## 🔧 확장 가능성

### 향후 추가 가능한 기능

1. **data_loader.py**
   - Custom augmentation pipeline
   - Multi-scale training support
   - Cached dataset (faster loading)

2. **model.py**
   - Ensemble 모델 지원
   - TTA (Test-Time Augmentation)
   - Model pruning/quantization

3. **trainer.py**
   - Multi-GPU 학습 (DDP)
   - Mixed precision training (AMP)
   - Learning rate finder

4. **inference.py**
   - Batch inference optimization
   - Visualization tools
   - Uncertainty estimation

---

**구현 완료**: 2026-02-06
**담당**: @DM
**상태**: utils.py 리팩토링 완료 ✅ (config 상속, flat 구조, 경로 헬퍼)
