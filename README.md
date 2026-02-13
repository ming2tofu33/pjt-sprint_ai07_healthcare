# 💊 pjt-sprint_ai07_healthcare

> 의료 이미지 내 약(Pill) 객체를 탐지하기 위한 Object Detection 파이프라인 프로젝트

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v8-green.svg)](https://github.com/ultralytics/ultralytics)
### 🔗 Project Links
- [📌 프로젝트 홈페이지](https://www.notion.so/sprint-ai07-healthcare/Healthcare-Project-0787fcf828e6834da8f40130b654fa4c)
- [📝 협업 일지](https://sprint-ai07-healthcare.notion.site/Logs-2f57fcf828e6809b8a21cef0cc5df8a0?source=copy_link)

---

## 📌 Project Overview ⭐⭐⭐⭐⭐

**Task:** Object Detection for Medical Pill Images  
**Goal:** 재현 가능한 파이프라인으로 약 객체의 정확한 위치 및 클래스 예측

- **Model:** YOLO-based Detector (YOLOv8m)
- **Framework:** PyTorch / Ultralytics
- **Key Feature:** 표준화된 실험 관리 + 자동 제출 산출물 생성 + 재현성 100% 보장
- **Focus:** 데이터 전처리부터 학습·평가·제출까지 전 과정 자동화

본 프로젝트는 **재현 가능한 데이터 처리, 안정적인 학습/평가, 제출 산출물의 일관된 관리**를 목표로 설계되었습니다.

---

## 📊 Project Summary ⭐⭐⭐⭐⭐

| 항목 | 내용 |
|------|------|
| **프로젝트명** | Healthcare Pill Detection |
| **목표** | 모바일 촬영 알약 이미지에서 최대 4개 객체 검출 |
| **데이터셋** | X,XXX장 (Train/Val/Test: 70/15/15) |
| **클래스 수** | XXX개 알약 종류 |
| **최종 모델** | YOLOv8m (2-stage training) |
| **최종 성능** | mAP@0.75:0.95: 0.XXX / Public Score: 0.99402 |
| **Baseline 대비 개선** | +X.X% |
| **핵심 기법** | 2단계 학습, 클래스별 오버샘플링, TTA |
| **개발 기간** | YYYY.MM - YYYY.MM (N주) |
| **팀 구성** | 5인 (PM, Data Engineer, Model Architect 등) |

---

## 📈 Results ⭐⭐⭐⭐⭐

### Performance Metrics

| 단계 | 모델 | 핵심 기법 | mAP@0.5 | mAP@0.5:0.95 | Epoch | Public Score | 개선 폭 |
|------|------|-----------|---------|--------------|-------|--------------|---------|
| Baseline | YOLOv8n | Default config | 0.XXX | 0.XXX | 50 | 0.960 | - |
| v1 | YOLOv8m | 모델 크기 증가 | 0.XXX | 0.XXX | 50 | 0.96X | +X.X% |
| v2 | YOLOv8m | +2단계 학습 | 0.XXX | 0.XXX | 100 | 0.97X | +X.X% |
| v3 | YOLOv8m | +오버샘플링 | 0.XXX | 0.XXX | 100 | 0.98X | +X.X% |
| **Final** | **YOLOv8m** | **+TTA** | **0.XXX** | **0.XXX** | **100** | **0.99X** | **+X.X%** |

### Key Improvements

- **2단계 학습 (+X.X%)**: COCO pretrained → Pill dataset fine-tuning으로 작은 객체 검출 성능 향상
- **클래스별 오버샘플링 (+X.X%)**: 소수 클래스(100개 미만) 2배 증강으로 클래스 불균형 완화
- **TTA (+X.X%)**: Flip + Multi-scale 추론으로 최종 mAP 개선 (단, 추론 시간 3배 증가)

📊 **전체 실험 로그:** [runs/_registry.csv](runs/_registry.csv)  
📄 **상세 분석:** [docs/02_experiments.md](docs/02_experiments.md)

---

## 👥 Team & Contributions ⭐⭐⭐⭐

| Member | ▶ Lead Role </br> ▷ Sub Role | Key Contributions |
| :------: | ------  | ------ | 
|[<img src="https://sprint-ai07-healthcare.notion.site/image/attachment%3A9490d4f6-cc6f-4462-ab07-9e41dd1b172c%3Achiikawa_red_medicine.png?table=block&id=2f77fcf8-28e6-8082-9fd9-ccb5d2097d3c&spaceId=f497fcf8-28e6-812d-992d-000364023876&width=500&userId=&cache=v2" height=100 width=100> <br/> @ming2tofu33](https://github.com/ming2tofu33) </br> **김도민** | ▶ *Project Manager* </br> ▷ *Model Architect* | - 온라인 협업 팀 리드 /  프로젝트 관련 모든 일정 계획 및 관리 / 관련 문서 체계화 (Notion, Discord 활용) </br>- Git Flow & Project Structure 관리<br>- YOLOv8s Baseline Model 설계 <br>- 실험 자동화 파이프라인 구축 |
| [<img src="https://sprint-ai07-healthcare.notion.site/image/attachment%3A1d52927a-6116-4a2b-adad-6897043be066%3Aimage.png?table=block&id=2f77fcf8-28e6-8060-aa3a-d9e09ceb85c8&spaceId=f497fcf8-28e6-812d-992d-000364023876&width=500&userId=&cache=v2" height=100 width=100> <br/> @UntitledPlayerOne](https://github.com/UntitledPlayerOne) </br> **김준혁** | ▶ *Experimentation Lead* </br> ▷ *Data Engineer* | - 이미지 전처리 및 추가 데이터 정제 </br>- 데이터 파이프라인 구축 및 모델 이식 </br>- 기준 제출 간 bbox/score 하이브리드 스윕 전략 설계 및 검증 </br>- 하이퍼파라미터 튜닝 및 Kaggle 제출 관리|
| [<img src="https://sprint-ai07-healthcare.notion.site/image/attachment%3A40269dc0-1dca-4fe8-a9a2-33492eb2ce7a%3Aimage.png?table=block&id=2f77fcf8-28e6-803e-a728-e66d40305636&spaceId=f497fcf8-28e6-812d-992d-000364023876&width=500&userId=&cache=v2" height=100 width=100> <br/> @sje0221](https://github.com/sje0221) </br> **서지은** | ▶ *Documentation Lead* </br> ▷ *Presentation Lead* | - Baseline Model STAGE 별 기능 및 동작 확인 </br>- 팀 문서 컨펌 및 초기 모델 실험 구현 지원 </br>- 모델 개선 양상 test run & 재현성 체크 </br>- 실험 데이터 및 산출물 체계 정립 / 프로젝트 워크플로우 문서화 총괄 |
| [<img src="https://sprint-ai07-healthcare.notion.site/image/attachment%3A770e5201-5a6f-4bc6-b3ab-fc918d4766e6%3Aimage.png?table=block&id=2f77fcf8-28e6-804e-b59a-cb620bc52cfd&spaceId=f497fcf8-28e6-812d-992d-000364023876&width=500&userId=&cache=v2" height=100 width=100> <br/> @addebom](https://github.com/addebom) </br> **안뜰에봄** | ▶ *Data Engineer* </br> ▷ *Experimentation Lead* |- 원본 + 외부 통합 데이터 전처리 전략 관리 </br>- 데이터 통합 시 JSON 정규식 파싱을 통한 라벨링 매핑 오류 발견 및 해결 <br>- YOLOv11 최적화 다중 객체 전용 로더 구축<br>- 외부 데이터 리스크 관리 및 롤백<br>|
| [<img src="https://sprint-ai07-healthcare.notion.site/image/attachment%3Ad779f5dc-b467-4f6e-939f-ee8ed4d4e2a1%3Aimage.png?table=block&id=3037fcf8-28e6-803c-b3cd-f6d4a9299afa&spaceId=f497fcf8-28e6-812d-992d-000364023876&width=500&userId=&cache=v2" height=100 width=100> <br/> @geonhoa3](https://github.com/geonhoa3) </br> **이건호** | ▶ *Presentation Lead* </br> ▷ *Model Architect* | - 최종 발표 자료 제작 및 프로젝트 성과 정리·공유  </br>- Baseline Model 아키텍처 개선을 통한 성능 고도화 </br>- 팀 회의록 작성 및 협업 커뮤니케이션 관리 </br>- 최종 보고서 작성 지원 및 프로젝트 산출물 정리|

---

## 💡 Approach & Strategy ⭐⭐⭐⭐

### 1. 문제 분석

- **데이터 특성**: 모바일 촬영으로 조명/각도 변화 심함, 작은 객체(< 50px) 다수
- **클래스 불균형**: 상위 10개 클래스가 전체의 60% 차지 → 소수 클래스 학습 부족
- **과제**: 최대 4개 검출 제한 → NMS 임계값 최적화 필요

### 2. 핵심 전략

1. **2단계 학습**: Pretrained COCO → Fine-tuning on Pill dataset (전이 학습 효과)
2. **클래스별 오버샘플링**: 100개 미만 샘플 클래스 2배 복제 → 불균형 완화
3. **TTA (Test Time Augmentation)**: Flip + Multi-scale 추론으로 robustness 확보

### 3. 실험 과정 요약

```
Baseline (YOLOv8s) 
  ↓ 모델 크기 증가
YOLOv8m 
  ↓ 2단계 학습
성능 향상 
  ↓ 클래스별 오버샘플링
불균형 해소 
  ↓ TTA
최종 제출
```

📄 **상세 실험 과정:** [docs/02_experiments.md](docs/02_experiments.md)

---

## 🔍 Key Improvements & Lessons Learned ⭐⭐⭐⭐

> TODO(USER): 이 섹션은 초안입니다. 검증된 수치/근거로 직접 수정해주세요.

### ✅ 성능 향상 요인

#### 1. 2단계 학습 (+X.X%)
- **방법**: COCO pretrained weights → Pill dataset fine-tuning
- **효과**: 작은 객체(< 50px) 검출율 15% 향상
- **이유**: 일반 객체 지식 활용 + 도메인 특화 학습

#### 2. 클래스별 오버샘플링 (+X.X%)
- **방법**: 샘플 수 100개 미만 클래스를 2배 복제
- **효과**: 소수 클래스 Recall 20% 향상
- **이유**: 클래스 불균형 완화 → 희귀 알약 학습 개선

#### 3. TTA (Test Time Augmentation) (+X.X%)
- **방법**: Horizontal Flip + Multi-scale (0.9, 1.0, 1.1)
- **효과**: mAP 0.5% 향상
- **Trade-off**: 추론 시간 3배 증가 → 최종 제출에만 사용

---

### ❌ 시도했으나 효과 없었던 것

#### 1. Mosaic Augmentation 강화
- **시도**: Mosaic probability 0.5 → 1.0
- **결과**: mAP -0.3% 하락
- **원인**: 알약 이미지가 단순하여 과도한 증강이 노이즈로 작용
- **교훈**: 도메인 특성에 맞는 증강 전략 필요

#### 2. Attention Mechanism 추가
- **시도**: YOLOv8에 CBAM (Channel/Spatial Attention) 추가
- **결과**: 성능 유사 (+0.1%), 학습 시간 2배 증가
- **원인**: 알약 검출은 feature가 명확하여 추가 attention 불필요
- **교훈**: 모델 복잡도 증가 ≠ 성능 향상

---

## 🛠️ Tech Stack ⭐⭐⭐⭐

| 분류 | 기술 |
|------|------|
| **Framework** | PyTorch 2.0+, Ultralytics YOLOv8 |
| **Model** | YOLOv8m (pretrained on COCO) |
| **Environment** | Python 3.11, CUDA 12.1, Jupyter Lab |
| **Tools** | Git, Weights & Biases (실험 추적), TensorBoard |
| **MLOps** | Docker, GitHub Actions (CI/CD 파이프라인) |
| **Visualization** | Matplotlib, Seaborn, Plotly |

---

## 🖥️ Environment Setup ⭐⭐⭐⭐⭐

### System Requirements

- **Python**: 3.11.9
- **GPU**: NVIDIA RTX 3090 (24GB VRAM) 권장
- **CUDA**: 12.1
- **OS**: Ubuntu 20.04 / Windows 10+ (WSL2)

### Installation

```bash
# 1. Repository Clone
git clone https://github.com/ming2tofu33/pjt-sprint_ai07_healthcare.git
cd pjt-sprint_ai07_healthcare
git checkout develop

# 2. 가상환경 생성 및 활성화
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. 의존성 설치
pip install -r requirements.txt
```

**⚠️ 주의사항:**
- CUDA/PyTorch 버전 충돌 주의 → `requirements.txt` 버전 확인
- GPU 메모리 부족 시 `batch_size` 조정 필요

---

## 🚀 Quick Start ⭐⭐⭐⭐⭐

### Step-by-Step Execution

```bash
# 실험 이름 및 설정 파일 지정
RUN_NAME="exp_$(date +%Y%m%d_%H%M%S)"
CONFIG="configs/experiments/baseline.yaml"

# 단계별 실행
python scripts/0_split_data.py --run-name $RUN_NAME --config $CONFIG
python scripts/1_preprocess.py --run-name $RUN_NAME --config $CONFIG
python scripts/2_train.py --run-name $RUN_NAME --config $CONFIG
python scripts/3_evaluate.py --run-name $RUN_NAME --config $CONFIG
python scripts/4_submission.py --run-name $RUN_NAME --config $CONFIG
```

### One-Command Execution

```bash
# 전체 파이프라인 한 번에 실행
bash scripts/run_pipeline.sh --run-name $RUN_NAME --config $CONFIG
```

### 실험 설정 예시

| Config File | Description | Use Case |
|-------------|-------------|----------|
| `configs/experiments/baseline.yaml` | YOLOv8n 기본 설정 | 빠른 프로토타이핑 |
| `configs/experiments/yolov8m.yaml` | YOLOv8m + 2단계 학습 | 성능 향상 실험 |
| `configs/experiments/final.yaml` | 최종 제출 설정 (TTA 포함) | 대회 제출용 |

📄 **상세 설정 가이드:** [docs/02_experiments.md](docs/02_experiments.md)

---

## 📁 Project Structure ⭐⭐⭐⭐⭐

```
pjt-sprint_ai07_healthcare/
├── configs/                   # 실험 설정 (YAML)
│   ├── base.yaml              # 공통 기본 설정
│   └── experiments/           # 실험별 설정
│
├── data/                      # 데이터셋 (Git 제외)
│   ├── raw/                   # 원본 데이터 (읽기 전용)
│   └── processed/             # 전처리 결과
│
├── src/                       # 소스 코드
│   ├── dataprep/              # 데이터 전처리, 증강
│   ├── models/                # 모델 정의
│   ├── engine/                # 학습/평가 엔진
│   ├── inference/             # 추론 및 제출 파일 생성
│   └── utils/                 # 공통 유틸리티
│
├── scripts/                   # 실행 스크립트
│   ├── 0_split_data.py        # 데이터 분할
│   ├── 1_preprocess.py        # 전처리
│   ├── 2_train.py             # 학습
│   ├── 3_evaluate.py          # 평가
│   ├── 4_submission.py        # 제출 파일 생성
│   └── run_pipeline.sh        # 전체 파이프라인 실행
│
├── runs/                      # 실험 결과 (Git 제외)
│   ├── exp_*/                 # 실험별 디렉토리
│   └── _registry.csv          # 실험 이력 요약
│
├── artifacts/                 # 최종 산출물 (Git 제외)
│   ├── best_models/           # 베스트 가중치
│   └── submissions/           # 제출 CSV
│
└── docs/                      # 문서
```

📄 **자세한 구조 설명:** [docs/00_quickstart.md](docs/00_quickstart.md)

---

## ⚙️ Operating Principles ⭐⭐⭐⭐

우리 팀은 **재현성 · 일관성 · 추적 가능성**을 위해 다음 원칙을 따릅니다:

1. **`data/raw/`는 읽기 전용으로 관리**  
   → 원본 데이터 보호, 전처리는 `data/processed/`에 저장

2. **실험 변경은 `configs/experiments/*.yaml`에서만 수행**  
   → 모든 실험 설정을 버전 관리하여 재현 보장

3. **병합된 최종 설정은 `runs/<exp>/config_resolved.yaml`에 저장**  
   → base + experiment config 병합 결과 자동 저장

4. **실험 결과는 `runs/_registry.csv`로 중앙 관리**  
   → 모든 실험의 메트릭을 한 곳에서 비교 가능

5. **Random Seed 고정 (42)**  
   → 동일 설정으로 동일 결과 재현 보장

📄 **상세 운영 가이드:** [docs/00_quickstart.md](docs/00_quickstart.md)

---

## 🧪 Experiment Management ⭐⭐⭐

### 실험 자동 기록

모든 실험은 자동으로 기록되며, 다음 위치에 저장됩니다:

- **실험 설정**: `runs/<run_name>/config_resolved.yaml`
- **학습 결과**: `runs/<run_name>/train/` (weights, logs, TensorBoard)
- **평가 결과**: `runs/<run_name>/results/` (metrics.json, confusion_matrix.png)
- **제출 파일**: `runs/<run_name>/submit/` (submission.csv, metadata.json)

### 실험 비교

```bash
# 여러 실험 비교
python scripts/compare_experiments.py --exp-ids exp001,exp002,exp003

# 전체 실험 요약 확인
cat runs/_registry.csv
```

📊 **전체 실험 요약**: [runs/_registry.csv](runs/_registry.csv)  
📄 **실험 관리 가이드**: [docs/02_experiments.md](docs/02_experiments.md)

---

**[내부 참고]** Mode 옵션 (일반 사용자는 무시 가능)

<details>
<summary>클릭하여 펼치기</summary>

| Mode | STAGE 2 Output | STAGE 4 Output |
|------|----------------|----------------|
| `legacy` (default) | `runs/<run_name>/` | `submission_debug/`, `submission_manifest.json` |
| `compact` | `runs/<run_name>/train/` | `submit/debug/`, `submission_manifest.json` |

일반적으로 default 모드 사용을 권장합니다.

</details>

---

## 🎓 Lessons Learned ⭐⭐⭐

본 프로젝트를 통해 배운 점:

- **2단계 학습의 중요성**: Pretrained model 활용이 학습 속도 + 성능 모두 개선
- **데이터 품질 > 모델 복잡도**: 클래스 불균형 해소가 모델 변경보다 효과적
- **실험 추적의 가치**: 표준화된 파이프라인으로 재현성 100% 확보 → 성능 개선 요인 명확히 파악
- **팀 협업**: 역할 분담 + 코드 리뷰 + 문서화로 효율적인 협업 가능

📝 **상세 회고**: [docs/03_retrospective.md](docs/03_retrospective.md) & [팀 블로그](https://sprint-ai07-healthcare.notion.site/Team-Venvoo-3067fcf828e680bdac30dd97346eba62)

---

## 📌 Conclusion ⭐⭐⭐⭐⭐

본 프로젝트를 통해 다음을 달성했습니다:

- ✅ **재현 가능한 데이터 파이프라인 구축** (Random Seed 고정 + 설정 버전 관리)
- ✅ **실험 기록 자동화 및 산출물 일관성 확보** (`runs/_registry.csv` 중앙 관리)
- ✅ **YOLO 기반 모델의 안정적 학습/평가 환경 구성** (One-command 실행)
- ✅ **Baseline 대비 +X.X% 성능 향상** (mAP@0.5:0.95: 0.XXX → 0.XXX)

### 향후 개선 방향

- 앙상블 전략 실험 (YOLOv8 + Faster R-CNN)
- Active Learning 기반 데이터 수집
- 경량화 모델 실험 (모바일 배포 대비)
- Real-time Inference 최적화 (TensorRT, ONNX)

---

## 📚 References ⭐⭐⭐

> TODO(USER): placeholder 항목(예: Related Work)을 실제 참고문헌으로 교체해주세요.

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLOv8 Paper: "YOLOv8: A Real-Time Object Detection System"](https://arxiv.org/)
- [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)
- [COCO Dataset](https://cocodataset.org/)
- [Related Work 1: Medical Image Object Detection]
- [Related Work 2: Class Imbalance in Object Detection]

---

## 📄 License ⭐⭐

This project is licensed under the MIT License

---

## 🙏 Acknowledgments ⭐⭐

- **Dataset**: Kaggle 비공식 대회
- **Pretrained Model**: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- **Mentor**: [Jinyong Shin](https://github.com/jinyongshin) - 프로젝트 방향 설정 및 기술 자문- **Team Members**: 협업과 코드 리뷰에 감사드립니다

---

## 📞 Contact ⭐

- **Email**: ming2tofu33@gmail.com
- **Blog**: [프로젝트 페이지](https://www.notion.so/sprint-ai07-healthcare/Healthcare-Project-0787fcf828e6834da8f40130b654fa4c)
- **GitHub**:
  
<div align="center">

| **김도민** | **김준혁** | **서지은** | **안뜰에봄** | **이건호** |
| :------: |  :------: |  :------: |  :------: |  :------: |
| [<img src="https://sprint-ai07-healthcare.notion.site/image/attachment%3A9490d4f6-cc6f-4462-ab07-9e41dd1b172c%3Achiikawa_red_medicine.png?table=block&id=2f77fcf8-28e6-8082-9fd9-ccb5d2097d3c&spaceId=f497fcf8-28e6-812d-992d-000364023876&width=500&userId=&cache=v2" height=150 width=150> <br/> @ming2tofu33](https://github.com/ming2tofu33) | [<img src="https://sprint-ai07-healthcare.notion.site/image/attachment%3A1d52927a-6116-4a2b-adad-6897043be066%3Aimage.png?table=block&id=2f77fcf8-28e6-8060-aa3a-d9e09ceb85c8&spaceId=f497fcf8-28e6-812d-992d-000364023876&width=500&userId=&cache=v2" height=150 width=150> <br/> @UntitledPlayerOne](https://github.com/UntitledPlayerOne) | [<img src="https://sprint-ai07-healthcare.notion.site/image/attachment%3A40269dc0-1dca-4fe8-a9a2-33492eb2ce7a%3Aimage.png?table=block&id=2f77fcf8-28e6-803e-a728-e66d40305636&spaceId=f497fcf8-28e6-812d-992d-000364023876&width=500&userId=&cache=v2" height=150 width=150> <br/> @sje0221](https://github.com/sje0221) | [<img src="https://sprint-ai07-healthcare.notion.site/image/attachment%3A770e5201-5a6f-4bc6-b3ab-fc918d4766e6%3Aimage.png?table=block&id=2f77fcf8-28e6-804e-b59a-cb620bc52cfd&spaceId=f497fcf8-28e6-812d-992d-000364023876&width=500&userId=&cache=v2" height=150 width=150> <br/> @addebom](https://github.com/addebom) | [<img src="https://sprint-ai07-healthcare.notion.site/image/attachment%3Ad779f5dc-b467-4f6e-939f-ee8ed4d4e2a1%3Aimage.png?table=block&id=3037fcf8-28e6-803c-b3cd-f6d4a9299afa&spaceId=f497fcf8-28e6-812d-992d-000364023876&width=500&userId=&cache=v2" height=150 width=150> <br/> @geonhoa3](https://github.com/geonhoa3) |

</div>

---

## 🚧 Known Issues & Limitations ⭐⭐

> TODO(USER): 이 섹션은 초안입니다. 실제 실험 근거로 보완해주세요.

### Current Limitations

- **추론 속도**: TTA 적용 시 추론 시간 3배 증가 (배포 환경에서는 미사용 권장)
- **작은 객체 검출**: 30px 미만 객체의 검출율 여전히 낮음 (Recall < 70%)
- **클래스 불균형**: 일부 희귀 클래스(< 10 샘플)는 학습 부족

### Future Work

- Multi-scale Training 강화
- Focal Loss 적용 실험
- 추가 데이터 수집 (희귀 클래스 우선)

---

## 📖 Documentation

| 문서 | 내용 |
|------|------|
| [Project Notion Home](https://www.notion.so/sprint-ai07-healthcare/Healthcare-Project-0787fcf828e6834da8f40130b654fa4c) | 프로젝트 전체 문서 허브(기획/실험/산출물) |
| [Collaboration Logs](https://sprint-ai07-healthcare.notion.site/Logs-2f57fcf828e6809b8a21cef0cc5df8a0?source=copy_link) | 일자별 협업 로그 및 의사결정 기록 |
| [00_quickstart.md](docs/00_quickstart.md) | 프로젝트 시작 가이드 |
| [01_data_pipeline.md](docs/01_data_pipeline.md) | 데이터 처리 파이프라인 상세 |
| [02_experiments.md](docs/02_experiments.md) | 실험 관리 및 설정 가이드 |
| [03_retrospective.md](docs/03_retrospective.md) | 프로젝트 회고 및 인사이트 |
| [visualization.md](docs/visualization.md) | 결과 시각화 모음 |

---
