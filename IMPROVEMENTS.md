# 코드 개선 사항 (2026-02-05)

## ✨ 주요 개선 사항

### 1. YAML Config 병합 로직 구현
- **기능**: `_base_` 키워드를 사용한 config 상속 지원
- **예시**:
  ```yaml
  # configs/experiments/exp001_baseline.yaml
  _base_: "../base.yaml"  # base config 상속
  
  train:
    epochs: 100  # base.yaml의 값을 override
  ```
- **구현**: `src/utils.py` - `load_yaml_with_inheritance()` 함수
- **장점**: 
  - 코드 중복 제거
  - 실험마다 필요한 부분만 override 가능
  - 계층적 config 관리

### 2. 하드코딩 제거 및 설정 파일 기반 관리
**개선 전**:
```python
dataset_root = paths["PROC_ROOT"] / "datasets" / f"pill_od_yolo_{paths['RUN_NAME']}"
```

**개선 후**:
```python
dataset_prefix = config.get("data", {}).get("dataset_prefix", "pill_od_yolo")
dataset_root = paths["PROC_ROOT"] / "datasets" / f"{dataset_prefix}_{paths['RUN_NAME']}"
```

**적용 파일**:
- `scripts/2_prepare_yolo_dataset.py`
- `scripts/3_train.py`
- `scripts/4_evaluate.py`

**장점**:
- 프로젝트 설정을 config에서 중앙 관리
- 다른 프로젝트에 쉽게 재사용 가능

### 3. Error Handling 강화
**추가된 함수들** (`src/utils.py`):

```python
# 의존성 체크
ensure_dependencies(required_packages=None, exit_on_missing=True)

# 데이터 존재 확인
check_data_exists(paths, required_keys=None)

# 프로젝트 기본값 가져오기
get_project_defaults()
```

**특징**:
- 실행 전 필수 패키지 확인
- 데이터 디렉토리 존재 확인
- 친절한 에러 메시지 및 해결 방법 제시

### 4. CLI 인자 개선
**개선 전** (각 스크립트 개별 실행):
```bash
python scripts/1_create_coco_format.py --run-name exp001
python scripts/0_splitting.py --run-name exp001
python scripts/2_prepare_yolo_dataset.py --run-name exp001
python scripts/3_train.py --run-name exp001 --epochs 100
```

**개선 후** (통합 파이프라인 러너):
```bash
# 전체 파이프라인 한 번에 실행
python scripts/run_pipeline.py \
  --config configs/experiments/exp001_baseline.yaml \
  --epochs 100 --batch 16
```

**장점**:
- 단일 명령으로 전체 실험 실행
- CLI 인자로 config override 가능
- 특정 단계만 선택 실행 가능

### 5. 전체 파이프라인 통합 스크립트
**새로운 파일**: `scripts/run_pipeline.py`

**주요 기능**:
1. **Config 기반 실행**: YAML config로 실험 설정 관리
2. **유연한 stage 선택**: `--stages 1,2,3` 으로 특정 단계만 실행
3. **CLI override**: 명령행에서 주요 파라미터 변경 가능
4. **에러 처리**: 각 단계별 에러 핸들링 및 로깅
5. **사전 조건 체크**: 의존성 및 데이터 확인

**사용 예시**:
```bash
# 전체 파이프라인 실행
python scripts/run_pipeline.py --config configs/experiments/exp001_baseline.yaml

# 특정 단계만 실행 (COCO 생성 + Split)
python scripts/run_pipeline.py --config configs/experiments/exp001_baseline.yaml --stages 1,2

# CLI로 파라미터 override
python scripts/run_pipeline.py \
  --config configs/experiments/exp001_baseline.yaml \
  --epochs 100 \
  --batch 16 \
  --model yolov8m \
  --device 0
```

### 6. 테스트 환경 지원
**새로운 파일**: `scripts/create_dummy_data.py`

**기능**:
- 실제 데이터 없이 파이프라인 테스트 가능
- 더미 이미지 및 annotation 자동 생성
- 원본 데이터 구조와 동일한 형식

**사용법**:
```bash
python scripts/create_dummy_data.py --n-train 10 --n-test 5 --n-cat 5
```

## 📊 테스트 결과

### Stage 1-3 성공 확인
```
✅ SUCCESS | Stage 1: COCO Format 생성
✅ SUCCESS | Stage 2: Train/Val Split  
✅ SUCCESS | Stage 3: YOLO Dataset 준비
```

**생성된 파일들**:
- `data/processed/cache/test_pipeline_v1/train_merged_coco.json`
- `data/processed/cache/test_pipeline_v1/splits/split_train_valid.json`
- `data/processed/datasets/pill_od_yolo_test_pipeline_v1/data.yaml`

### Config 병합 테스트
```
✅ Config 로드 성공!
- Project name: ai07_pill_od
- Train epochs: 80 (base.yaml에서)
- Train batch: 8 (exp001_baseline.yaml에서 override)
- Model name: yolov8s.pt
- Data classes: 56
```

## 🔧 Config 구조 개선

### base.yaml 구조 (기존)
```yaml
train:
  model_name: "yolov8s.pt"
  imgsz: 768
  epochs: 80
  batch: 8
  ...
```

### 실험 config 예시
```yaml
# configs/experiments/exp002_larger.yaml
_base_: "../base.yaml"

experiment:
  id: "exp002"
  name: "larger_model"
  description: "YOLOv8m with more epochs"

train:
  model_name: "yolov8m.pt"  # base를 override
  epochs: 150               # base를 override
  # batch: 8 은 base.yaml 값 사용
```

## 🚀 팀원들을 위한 사용 가이드

### 1. 새로운 실험 시작하기

#### Step 1: Experiment Config 생성
```bash
# configs/experiments/ 에 새 파일 생성
cp configs/experiments/exp001_baseline.yaml configs/experiments/exp005_my_exp.yaml
```

#### Step 2: Config 수정
```yaml
_base_: "../base.yaml"

experiment:
  id: "exp005"
  name: "my_experiment"
  description: "실험 설명"
  author: "@YourName"

# 변경하고 싶은 부분만 작성
train:
  epochs: 150
  batch: 16
```

#### Step 3: 실행
```bash
# 방법 1: 통합 파이프라인 (권장)
python scripts/run_pipeline.py --config configs/experiments/exp005_my_exp.yaml

# 방법 2: CLI로 빠른 테스트
python scripts/run_pipeline.py \
  --config configs/experiments/exp005_my_exp.yaml \
  --epochs 10 \
  --batch 4 \
  --device cpu
```

### 2. 개별 단계 실행

```bash
# Stage 1-2만 (데이터 준비)
python scripts/run_pipeline.py --config <config> --stages 1,2

# Stage 4만 (학습) - 데이터가 이미 준비된 경우
python scripts/run_pipeline.py --config <config> --stages 4 --run-name existing_run

# Stage 5-6만 (평가 + 제출)
python scripts/run_pipeline.py --config <config> --stages 5,6 --run-name existing_run
```

### 3. 일반적인 워크플로우

```bash
# 1. 더미 데이터로 먼저 테스트 (선택)
python scripts/create_dummy_data.py --n-train 10

# 2. 전체 파이프라인 dry-run (epochs 적게)
python scripts/run_pipeline.py \
  --config configs/experiments/exp001_baseline.yaml \
  --epochs 2 \
  --device cpu

# 3. 실제 데이터로 실험
python scripts/run_pipeline.py \
  --config configs/experiments/exp001_baseline.yaml \
  --device 0
```

## 📝 남은 작업

### Stage 4-6 (학습/평가/제출)
- **현재 상태**: Config 호환성 개선 완료
- **필요사항**: 
  - `ultralytics` 설치 필요
  - GPU 환경 권장 (CPU로도 가능하나 매우 느림)
- **실행 조건**: 
  - Stage 1-3이 성공적으로 완료되어야 함
  - data.yaml 파일이 생성되어 있어야 함

### 추가 개선 가능 사항
1. W&B 통합 (실험 트래킹)
2. K-Fold Cross Validation 구현
3. Ensemble 지원
4. TTA (Test Time Augmentation)

## 🎯 주요 이점

1. **재사용성**: 다른 프로젝트에 쉽게 적용 가능
2. **유지보수성**: 중앙화된 config 관리
3. **확장성**: 새로운 실험 추가가 간단함
4. **팀 협업**: 일관된 실험 방식
5. **에러 방지**: 사전 체크 및 명확한 에러 메시지

## 📦 파일 구조

```
pjt-sprint_ai07_healthcare/
├── scripts/
│   ├── run_pipeline.py          # ⭐ NEW: 통합 파이프라인 실행
│   ├── create_dummy_data.py     # ⭐ NEW: 테스트용 더미 데이터 생성
│   ├── 0_splitting.py           # ✅ UPDATED: Config 호환성 개선
│   ├── 1_create_coco_format.py  # ✅ UPDATED
│   ├── 2_prepare_yolo_dataset.py # ✅ UPDATED: 하드코딩 제거
│   ├── 3_train.py               # ✅ UPDATED: Config 접근 개선
│   ├── 4_evaluate.py            # ✅ UPDATED
│   └── 5_submission.py          # OK
│
├── src/
│   └── utils.py                 # ✅ UPDATED: 새 함수들 추가
│       - load_yaml_with_inheritance()  # ⭐ NEW
│       - deep_merge_dict()             # ⭐ NEW
│       - ensure_dependencies()         # ⭐ NEW
│       - check_data_exists()           # ⭐ NEW
│       - get_project_defaults()        # ⭐ NEW
│
├── configs/
│   ├── base.yaml                # OK: 기본 설정
│   └── experiments/
│       ├── exp001_baseline.yaml # OK: _base_ 상속 사용
│       ├── exp002_whitelist.yaml
│       └── ...
│
└── IMPROVEMENTS.md              # ⭐ NEW: 이 문서
```

## 🤝 팀 협업 가이드

### 실험 수행 시
1. **브랜치**: `feat/<your-name>` 또는 `exp/<exp-name>`
2. **Config**: `configs/experiments/exp_<id>_<name>.yaml` 생성
3. **실행**: `run_pipeline.py` 사용
4. **결과 공유**: `runs/<run_name>/` 폴더 확인

### Git Workflow (기존 규칙 준수)
- 코드 수정 후 즉시 commit
- PR 생성 전 remote와 sync
- Conflict 발생 시 remote 우선
- Commit squash 후 PR 생성

---

**작성일**: 2026-02-05  
**작성자**: AI Assistant (feat/DM 브랜치 개선)  
**테스트 환경**: Python 3.12, Dummy Data (10 train images)
