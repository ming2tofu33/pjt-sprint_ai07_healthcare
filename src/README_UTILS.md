# src/utils.py 사용 가이드

## 📌 개요

`src/utils.py`는 프로젝트 전반에서 사용되는 **공통 유틸리티 함수 모음**입니다.

### 주요 기능

1. **경로 관리** (`setup_project_paths`)
   - 프로젝트 디렉터리 구조 자동 생성 및 검증
   - 실험별 runs/ 및 artifacts/ 폴더 관리

2. **재현성 설정** (`set_seed`)
   - Random seed 고정 (Python, NumPy, PyTorch)
   - CUDA deterministic 모드 설정
   - 환경 정보 수집 (패키지 버전, GPU 정보 등)

3. **실험 관리** (`get_default_config`, `save_config`, `load_config`)
   - Config 파일 생성/저장/로드 (JSON/YAML 지원)
   - Run manifest 생성 (Git 정보 포함)
   - 실험 레지스트리 관리 (`runs/_registry.csv`)

4. **결과 기록** (`record_result`)
   - 실험 결과를 `results.csv` 및 `results.jsonl`에 자동 기록
   - 메트릭, 하이퍼파라미터, 경로 정보 통합 관리

---

## 🚀 사용 예시

### 1. 기본 사용법

```python
from pathlib import Path
from src.utils import (
    setup_project_paths,
    set_seed,
    get_default_config,
    save_config,
)

# 1) 경로 설정
paths = setup_project_paths(
    run_name="exp_baseline_v1",  # 실험명 (None이면 자동 생성)
    root=Path(__file__).parent,  # 프로젝트 루트
    create_dirs=True,            # 폴더 자동 생성
    check_input_exists=True,     # 데이터 폴더 검증
)

print(f"RUN_DIR: {paths['RUN_DIR']}")
print(f"CKPT: {paths['CKPT']}")
print(f"LOGS: {paths['LOGS']}")

# 2) Seed 고정
env_meta = set_seed(seed=42, deterministic=True)
save_json(paths["CONFIG"] / "env_meta.json", env_meta)

# 3) Config 생성
config = get_default_config(
    run_name=paths["RUN_NAME"],
    paths=paths,
    seed=42,
)

# Config 커스터마이징
config["train"]["model"]["imgsz"] = 960  # 해상도 변경
config["train"]["hyperparams"]["epochs"] = 100
config["data"]["class_whitelist"] = [1900, 16548, 19607]  # 특정 클래스만

# Config 저장
save_config(config, paths["CONFIG"] / "config.json")
```

---

### 2. 실험 결과 기록

```python
from src.utils import record_result

# 학습/평가 후
metrics = {
    "mAP_75_95": 0.4523,
    "mAP_50": 0.6891,
    "mAP_75": 0.5234,
    "precision": 0.7123,
    "recall": 0.6845,
}

record_result(
    results_csv=paths["REPORTS"] / "results.csv",
    results_jsonl=paths["REPORTS"] / "results.jsonl",
    run_name=paths["RUN_NAME"],
    result_name="baseline_v1",  # 결과 별칭
    stage="val",                # val / oof / public_lb / private_lb
    config=config,
    metrics=metrics,
    paths=paths,
    notes="YOLOv8s 768px baseline",
    submission_path=paths["SUBMISSIONS"] / "submission_v1.csv",
)
```

---

### 3. Config 변경 패턴 (실험 변화)

```python
# 기본 Config 로드
config = load_config(Path("runs/exp_baseline_v1/config/config.json"))

# 실험 변형 1: 해상도 증가
config["train"]["model"]["imgsz"] = 960
config["notes"] = "해상도 960px 실험"

# 실험 변형 2: Class whitelist 적용
config["data"]["class_whitelist"] = [1900, 16548, 19607, 29451]
config["notes"] = "Test 클래스 40개만 사용"

# 실험 변형 3: Augmentation 끄기
config["train"]["augment"]["mosaic"] = False
config["train"]["augment"]["mixup"] = False
config["notes"] = "Augmentation 최소화"

# 새 실험으로 저장
new_paths = setup_project_paths(run_name="exp_imgsz960_v1")
save_config(config, new_paths["CONFIG"] / "config.json")
```

---

## 📂 생성되는 폴더 구조

`setup_project_paths()` 실행 시 자동 생성:

```
pjt-sprint_ai07_healthcare/
├── runs/
│   ├── exp_baseline_v1/
│   │   ├── checkpoints/          # 모델 체크포인트
│   │   ├── logs/                 # 학습 로그
│   │   │   ├── metrics.jsonl
│   │   │   └── events.jsonl
│   │   └── config/               # 실험 설정
│   │       ├── config.json
│   │       ├── paths_meta.json
│   │       ├── env_meta.json
│   │       └── run_manifest.json
│   └── _registry.csv             # 전체 실험 목록
│
├── artifacts/
│   └── exp_baseline_v1/
│       ├── submissions/          # 제출 파일
│       ├── plots/                # 시각화
│       └── reports/              # 평가 리포트
│           ├── results.csv
│           └── results.jsonl
│
└── data/
    └── processed/
        └── cache/
            └── exp_baseline_v1/  # 실험별 캐시
```

---

## ⚙️ Config 구조 (기본값)

```json
{
  "project": {
    "name": "ai07_pill_od",
    "run_name": "exp_YYYYMMDD_HHMMSS"
  },
  "reproducibility": {
    "seed": 42,
    "deterministic": true
  },
  "data": {
    "format": "coco_json_multi",
    "max_objects_per_image": 4,
    "num_classes": null,           # 자동 추출
    "class_whitelist": null         # null=전체 / [id1,id2,...]=부분
  },
  "split": {
    "strategy": "stratify_by_num_objects",
    "seed": 42,
    "ratios": {"train": 0.8, "valid": 0.2},
    "kfold": {"enabled": false, "n_splits": 5, "fold_idx": 0}
  },
  "train": {
    "framework": "ultralytics_yolo",
    "model": {
      "name": "yolov8s",           # yolov8n/s/m/l/x
      "imgsz": 768,                # 640 / 768 / 960
      "pretrained": true
    },
    "hyperparams": {
      "epochs": 80,
      "batch": 8,
      "lr0": null,                 # null=YOLO 기본값
      "weight_decay": null,
      "workers": 4
    },
    "augment": {
      "enabled": true,
      "mosaic": true,
      "mixup": false,
      "hsv": true,
      "flip": true
    }
  },
  "infer": {
    "conf_thr": 0.001,             # 낮게 설정 후 후처리로 조정
    "nms_iou_thr": 0.5,
    "max_det_per_image": 4         # 대회 규칙
  },
  "postprocess": {
    "strategy": "topk_by_score",
    "topk": 4,
    "classwise_threshold": null,   # {1900: 0.3, 16548: 0.25, ...}
    "clip_boxes": true
  }
}
```

---

## 🔧 주요 함수 API

### `setup_project_paths()`
```python
def setup_project_paths(
    run_name: Optional[str] = None,
    root: Optional[Path] = None,
    create_dirs: bool = True,
    check_input_exists: bool = True,
) -> Dict[str, Path]:
```

**반환값**:
- `ROOT`, `DATA_ROOT`, `RUN_NAME`
- `TRAIN_IMAGES`, `TRAIN_ANN_DIR`, `TEST_IMAGES`
- `RUNS`, `ARTIFACTS`, `RUN_DIR`, `ART_DIR`
- `CKPT`, `LOGS`, `CONFIG`, `SUBMISSIONS`, `PLOTS`, `REPORTS`, `CACHE`

---

### `set_seed()`
```python
def set_seed(
    seed: int = 42,
    deterministic: bool = True
) -> Dict[str, Any]:
```

**반환값**:
- `timestamp`, `seed`, `deterministic`
- `python`: version, executable
- `platform`: system, release, machine
- `packages`: numpy, torch, ultralytics 등 버전
- `torch`: CUDA 정보

---

### `record_result()`
```python
def record_result(
    results_csv: Path,
    results_jsonl: Path,
    run_name: str,
    result_name: str,      # "baseline_v1"
    stage: str,            # "val" / "public_lb"
    config: Dict[str, Any],
    metrics: Optional[Dict[str, Any]] = None,
    paths: Optional[Dict[str, Path]] = None,
    notes: str = "",
    submission_path: Optional[Path] = None,
):
```

**자동 기록 항목**:
- CSV: 주요 Config + Metrics (mAP, Precision, Recall 등)
- JSONL: 전체 Config + Metrics (상세 보존)

---

## 📊 실험 레지스트리 활용

`runs/_registry.csv` 파일에 모든 실험이 자동 기록됩니다:

```bash
# 전체 실험 목록 확인
cat runs/_registry.csv

# 특정 날짜 실험만 필터링
grep "20260204" runs/_registry.csv
```

---

## ✅ 체크리스트

**새 실험 시작 시**:
1. [ ] `setup_project_paths()`로 폴더 구조 생성
2. [ ] `set_seed()`로 재현성 확보 + env_meta.json 저장
3. [ ] `get_default_config()` 또는 기존 config 로드
4. [ ] Config 커스터마이징 (모델/하이퍼파라미터)
5. [ ] `save_config()`로 config.json 저장
6. [ ] `create_run_manifest()`로 Git 정보 스냅샷

**실험 종료 시**:
1. [ ] `record_result()`로 결과 기록
2. [ ] Checkpoint 복사 (`best.pt` → `CKPT/`)
3. [ ] Submission 파일 저장 (`SUBMISSIONS/`)
4. [ ] 시각화/리포트 저장 (`PLOTS/`, `REPORTS/`)

---

## 💡 Tips

1. **환경별 분기**:
   ```python
   # Colab vs 로컬 자동 감지
   if Path("/content").exists():
       root = Path("/content/drive/MyDrive/healthcare_project")
   else:
       root = Path.cwd()
   ```

2. **Config override**:
   ```python
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument("--imgsz", type=int, default=768)
   args = parser.parse_args()
   
   config["train"]["model"]["imgsz"] = args.imgsz
   ```

3. **실험 비교**:
   ```python
   import pandas as pd
   df = pd.read_csv("artifacts/exp_baseline_v1/reports/results.csv")
   print(df.sort_values("mAP_75_95", ascending=False).head(10))
   ```

---

## 🐛 Troubleshooting

**Q: `FileNotFoundError: 필수 INPUT 폴더가 없습니다`**
- 데이터가 실제로 없는 경우: `check_input_exists=False` 옵션 사용
- 경로가 잘못된 경우: `root` 인자로 명시적 경로 지정

**Q: CUDA 관련 에러 (`deterministic=True` 시)**
- 일부 YOLO 연산이 deterministic 모드 미지원
- `deterministic=False`로 변경하거나 환경변수 수정

**Q: Config가 너무 길어짐**
- YAML 형식 사용 (읽기 쉬움): `save_config(config, "config.yaml")`
- 실험별 변경 사항만 기록하는 diff 방식 고려

---

**구현 완료**: 2026-02-05  
**다음 단계**: Stage 1 (데이터 분할 및 COCO 변환 스크립트)
