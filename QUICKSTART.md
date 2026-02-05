# 🚀 빠른 시작 가이드

## 필수 사항

### 1. Python 환경
```bash
python --version  # Python 3.8 이상
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

## 사용 방법

### Option 1: 통합 파이프라인 실행 (권장) ⭐

#### 기본 실행
```bash
python scripts/run_pipeline.py --config configs/experiments/exp001_baseline.yaml
```

#### CLI로 파라미터 변경
```bash
python scripts/run_pipeline.py \
  --config configs/experiments/exp001_baseline.yaml \
  --epochs 100 \
  --batch 16 \
  --device 0
```

#### 특정 단계만 실행
```bash
# Stage 1-3만 (데이터 준비)
python scripts/run_pipeline.py \
  --config configs/experiments/exp001_baseline.yaml \
  --stages 1,2,3

# Stage 4만 (학습)
python scripts/run_pipeline.py \
  --config configs/experiments/exp001_baseline.yaml \
  --stages 4 \
  --run-name existing_run

# Stage 5-6만 (평가 + 제출)
python scripts/run_pipeline.py \
  --config configs/experiments/exp001_baseline.yaml \
  --stages 5,6 \
  --run-name existing_run
```

### Option 2: 개별 스크립트 실행

```bash
# Stage 1: COCO Format 생성
python scripts/1_create_coco_format.py --run-name my_exp

# Stage 2: Train/Val Split
python scripts/0_splitting.py --run-name my_exp

# Stage 3: YOLO Dataset 준비
python scripts/2_prepare_yolo_dataset.py --run-name my_exp

# Stage 4: 모델 학습
python scripts/3_train.py --run-name my_exp --epochs 80 --batch 8 --device 0

# Stage 5: 모델 평가
python scripts/4_evaluate.py --run-name my_exp --ckpt best --device 0

# Stage 6: 제출 파일 생성
python scripts/5_submission.py --run-name my_exp --ckpt best --device 0
```

## 새로운 실험 시작하기

### 1. Config 파일 복사
```bash
cp configs/experiments/exp001_baseline.yaml configs/experiments/exp_my_test.yaml
```

### 2. Config 수정
```yaml
# configs/experiments/exp_my_test.yaml
_base_: "../base.yaml"  # base config 상속

experiment:
  id: "exp_my"
  name: "my_test"
  description: "실험 설명"
  author: "@YourName"

# 변경하고 싶은 부분만 작성
train:
  epochs: 150
  batch: 16
```

### 3. 실행
```bash
python scripts/run_pipeline.py --config configs/experiments/exp_my_test.yaml
```

## 테스트 (데이터 없이)

더미 데이터로 파이프라인 테스트:

```bash
# 1. 더미 데이터 생성
python scripts/create_dummy_data.py --n-train 10 --n-test 5

# 2. 파이프라인 테스트 (짧은 epoch)
python scripts/run_pipeline.py \
  --config configs/experiments/exp001_baseline.yaml \
  --epochs 2 \
  --batch 2 \
  --device cpu
```

## 파일 구조

```
pjt-sprint_ai07_healthcare/
├── scripts/
│   ├── run_pipeline.py          # ⭐ 통합 파이프라인 실행
│   ├── create_dummy_data.py     # 테스트용 더미 데이터 생성
│   └── [0-5]_*.py              # 개별 단계 스크립트
│
├── configs/
│   ├── base.yaml               # 기본 설정
│   └── experiments/            # 실험별 config
│       ├── exp001_baseline.yaml
│       └── ...
│
├── data/
│   ├── raw/                    # 원본 데이터 (Git 제외)
│   │   ├── train_images/
│   │   ├── train_annotations/
│   │   └── test_images/
│   └── processed/              # 처리된 데이터 (Git 제외)
│
├── runs/                       # 실험 결과 (Git 제외)
│   └── <run_name>/
│       ├── checkpoints/
│       ├── config/
│       └── train/
│
└── artifacts/                  # 최종 산출물 (Git 제외)
    └── <run_name>/
        ├── submissions/
        ├── plots/
        └── reports/
```

## 도움말

더 자세한 정보는:
- `IMPROVEMENTS.md`: 개선 사항 상세 설명
- `README.md`: 프로젝트 전체 문서
- `scripts/run_pipeline.py --help`: CLI 도움말

## 문제 해결

### Import 에러
```bash
# src 모듈이 인식 안 될 때
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
```

### 의존성 에러
```bash
pip install -r requirements.txt
```

### 데이터 없음 에러
```bash
# 테스트용 더미 데이터 생성
python scripts/create_dummy_data.py
```

## 주요 인자 설명

| 인자 | 설명 | 예시 |
|------|------|------|
| `--config` | Config YAML 파일 경로 | `configs/experiments/exp001_baseline.yaml` |
| `--run-name` | 실험명 (자동 생성 가능) | `exp001_baseline` |
| `--stages` | 실행할 단계 (쉼표 구분) | `1,2,3` |
| `--epochs` | 학습 epoch 수 | `100` |
| `--batch` | Batch size | `16` |
| `--model` | YOLO 모델 | `yolov8s` / `yolov8m` |
| `--device` | GPU 디바이스 | `0` / `cpu` |
| `--conf` | Confidence threshold | `0.25` |
| `--skip-check` | 사전 체크 건너뛰기 | flag |

---

**TIP**: 처음 실행할 때는 `--epochs 2 --device cpu`로 빠르게 테스트해보세요!
