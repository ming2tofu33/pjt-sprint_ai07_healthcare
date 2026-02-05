# 🏠 로컬 환경 실험 가이드

이 가이드는 로컬 컴퓨터에서 YOLOv8 베이스라인 실험을 시작하는 방법을 단계별로 설명합니다.

---

## 📋 **사전 준비 체크리스트**

- [ ] Python 3.8 이상 설치
- [ ] Git 설치
- [ ] CUDA 지원 GPU (권장, CPU도 가능하지만 느림)
- [ ] 최소 10GB 여유 공간
- [ ] 데이터 파일 준비 (232 train images, 763 annotations, 842 test images)

---

## 🚀 **1단계: 코드 가져오기**

```bash
# 1. 레포지토리 클론
git clone https://github.com/ming2tofu33/pjt-sprint_ai07_healthcare.git
cd pjt-sprint_ai07_healthcare

# 2. feat/DM-refactor 브랜치로 전환
git checkout feat/DM-refactor

# 3. 최신 코드 가져오기
git pull origin feat/DM-refactor

# 4. 프로젝트 구조 확인
ls -la
```

---

## 🔧 **2단계: Python 환경 설정**

### **방법 A: Conda (권장)**

```bash
# 1. 새 환경 생성
conda create -n pill_detection python=3.10 -y
conda activate pill_detection

# 2. PyTorch 설치 (CUDA 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. 나머지 패키지 설치
pip install ultralytics pandas numpy PyYAML scikit-learn matplotlib seaborn albumentations

# 4. 설치 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### **방법 B: venv**

```bash
# 1. 가상환경 생성
python -m venv venv

# 2. 활성화
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. 패키지 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics pandas numpy PyYAML scikit-learn matplotlib seaborn albumentations

# 4. 설치 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### **방법 C: pip만 사용**

```bash
# 1. PyTorch 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. 나머지 패키지 설치
pip install ultralytics pandas numpy PyYAML scikit-learn matplotlib seaborn albumentations
```

---

## 📁 **3단계: 데이터 설정**

### **옵션 A: 자동 설정 스크립트 사용 (추천)**

```bash
# 1. 스크립트 실행
bash setup_local_data.sh

# 2. 프롬프트가 나오면 데이터가 있는 경로 입력
# 예: /home/user/downloads/pill_data
# 또는: ../my_data

# 3. 자동으로 심볼릭 링크 생성됨
```

### **옵션 B: 수동 설정**

```bash
# 1. 데이터 디렉토리 생성
mkdir -p data/raw

# 2. 데이터가 다른 위치에 있다면 심볼릭 링크 생성
ln -s /path/to/your/train_images data/raw/train_images
ln -s /path/to/your/train_annotations data/raw/train_annotations
ln -s /path/to/your/test_images data/raw/test_images

# 3. 데이터 확인
ls -la data/raw/
```

### **옵션 C: 데이터 복사 (충분한 공간이 있다면)**

```bash
# 데이터를 프로젝트 폴더로 복사
cp -r /path/to/your/train_images data/raw/
cp -r /path/to/your/train_annotations data/raw/
cp -r /path/to/your/test_images data/raw/
```

### **데이터 구조 확인**

```bash
# 예상되는 구조:
tree data/raw/ -L 2

# 출력 예시:
# data/raw/
# ├── train_images/          (232 files)
# │   ├── K-001900-016548-019607-029451_0_2_0_2_70_000_200.png
# │   └── ...
# ├── train_annotations/     (114 folders, 763 JSON files)
# │   ├── K-001900-016548-019607-029451_json/
# │   │   ├── K-001900/
# │   │   │   └── K-001900-016548-019607-029451_0_2_0_2_70_000_200.json
# │   └── ...
# └── test_images/           (842 files)
#     ├── test_001.png
#     └── ...

# 파일 개수 확인
echo "Train images: $(find data/raw/train_images -type f | wc -l)"
echo "Train annotations: $(find data/raw/train_annotations -name "*.json" | wc -l)"
echo "Test images: $(find data/raw/test_images -type f | wc -l)"

# 예상 출력:
# Train images: 232
# Train annotations: 763
# Test images: 842
```

---

## 🎯 **4단계: 빠른 테스트 (5분)**

전체 파이프라인을 작은 데이터로 빠르게 테스트해봅니다.

```bash
# 1. COCO 포맷 생성 (1분)
python scripts/1_create_coco_format.py \
    --train_images data/raw/train_images \
    --train_annotations data/raw/train_annotations \
    --output_dir data/coco_data \
    --validate

# 출력 확인:
# ✅ Merged COCO JSON saved to: data/coco_data/merged_coco.json
# ✅ Category mapping saved to: data/coco_data/category_mapping.json

# 2. 데이터 분할 (30초)
python scripts/0_splitting.py \
    --coco_json data/coco_data/merged_coco.json \
    --output_dir data/splits \
    --train_ratio 0.8 \
    --stratify_by object_count

# 출력 확인:
# ✅ Split info saved to: data/splits/split_info.json

# 3. YOLO 데이터셋 준비 (1분)
python scripts/2_prepare_yolo.py \
    --coco_dir data/coco_data \
    --images_dir data/raw/train_images \
    --splits_dir data/splits \
    --output_dir data/yolo_data \
    --symlink

# 출력 확인:
# ✅ YOLO dataset created at: data/yolo_data
# ✅ data.yaml saved

# 4. 설정 파일 확인
cat data/yolo_data/data.yaml

# 5. 간단한 테스트 학습 (1 epoch, 2분)
python scripts/3_train.py \
    --config configs/experiments/exp001_baseline.yaml \
    --epochs 1 \
    --batch_size 8

# 출력 확인:
# ✅ Training completed
# ✅ Best model saved to: runs/exp001_*/checkpoints/best.pt
```

**✅ 테스트가 성공하면 본격적인 실험을 시작할 수 있습니다!**

---

## 🏃 **5단계: 실제 실험 시작**

### **실험 1: YOLOv8n 베이스라인 (1-2시간)**

```bash
# 1. 베이스라인 학습 (50 epochs)
python scripts/3_train.py \
    --config configs/experiments/exp001_baseline.yaml

# 2. 학습 모니터링 (새 터미널에서)
tail -f runs/exp001_*/logs/exp001.log

# 3. 학습 완료 후 평가
python scripts/4_evaluate.py \
    --checkpoint runs/exp001_*/checkpoints/best.pt \
    --data_yaml data/yolo_data/data.yaml

# 4. Kaggle 제출 파일 생성
python scripts/5_submission.py \
    --checkpoint runs/exp001_*/checkpoints/best.pt \
    --test_images data/raw/test_images \
    --category_mapping data/coco_data/category_mapping.json

# 5. 생성된 CSV 확인
ls -lh submissions/
head -20 submissions/submission_exp001_*.csv
```

### **실험 2: YOLOv8s 확장 (2-4시간)**

```bash
# 더 큰 모델로 학습
python scripts/3_train.py \
    --config configs/experiments/exp002_yolov8s_extended.yaml

# 평가 및 제출
python scripts/4_evaluate.py \
    --checkpoint runs/exp002_*/checkpoints/best.pt \
    --data_yaml data/yolo_data/data.yaml

python scripts/5_submission.py \
    --checkpoint runs/exp002_*/checkpoints/best.pt \
    --test_images data/raw/test_images \
    --category_mapping data/coco_data/category_mapping.json
```

### **실험 3: 고해상도 학습 (4-8시간)**

```bash
# 1280 이미지 크기로 학습
python scripts/3_train.py \
    --config configs/experiments/exp003_yolov8m_highres.yaml
```

---

## 📊 **6단계: 결과 확인 및 분석**

### **학습 결과 확인**

```bash
# 1. 실험 디렉토리 확인
ls -la runs/

# 2. 로그 확인
cat runs/exp001_*/logs/exp001.log | tail -50

# 3. 평가 결과 확인
cat evaluation_results/summary.txt

# 4. 제출 파일 확인
cat submissions/submission_exp001_*.csv | head -20
```

### **시각화 (선택사항)**

```python
# Jupyter Notebook 또는 Python 스크립트에서
import pandas as pd
import matplotlib.pyplot as plt

# 제출 파일 분석
df = pd.read_csv('submissions/submission_exp001_*.csv')
print(f"Total detections: {len(df)}")
print(f"Unique images: {df['image_id'].nunique()}")
print(f"Avg detections per image: {len(df) / df['image_id'].nunique():.2f}")

# 점수 분포
df['score'].hist(bins=50)
plt.xlabel('Confidence Score')
plt.ylabel('Count')
plt.title('Detection Score Distribution')
plt.show()
```

---

## 🔧 **커스텀 실험하기**

### **자신만의 실험 config 만들기**

```bash
# 1. 기존 config 복사
cp configs/experiments/exp001_baseline.yaml configs/experiments/exp_mytest.yaml

# 2. config 수정 (텍스트 에디터로)
vim configs/experiments/exp_mytest.yaml
# 또는
code configs/experiments/exp_mytest.yaml

# 3. 수정 예시:
# experiment:
#   name: "mytest"
#   model_variant: "yolov8s"  # n → s로 변경
#   epochs: 100               # 50 → 100으로 증가
#   description: "My custom experiment"

# 4. 실험 실행
python scripts/3_train.py --config configs/experiments/exp_mytest.yaml
```

### **하이퍼파라미터 튜닝**

```bash
# CLI로 바로 오버라이드
python scripts/3_train.py \
    --config configs/experiments/exp001_baseline.yaml \
    --epochs 150 \
    --batch_size 32 \
    --lr 0.0005 \
    --image_size 1280
```

### **제출 파일 튜닝**

```bash
# Confidence threshold 조정
python scripts/5_submission.py \
    --checkpoint runs/exp001_*/checkpoints/best.pt \
    --test_images data/raw/test_images \
    --category_mapping data/coco_data/category_mapping.json \
    --conf_threshold 0.15  # 기본값: 0.25

# NMS threshold 조정
python scripts/5_submission.py \
    --checkpoint runs/exp001_*/checkpoints/best.pt \
    --test_images data/raw/test_images \
    --category_mapping data/coco_data/category_mapping.json \
    --iou_nms 0.40  # 기본값: 0.45

# TTA (Test Time Augmentation) 적용
python scripts/5_submission.py \
    --checkpoint runs/exp001_*/checkpoints/best.pt \
    --test_images data/raw/test_images \
    --category_mapping data/coco_data/category_mapping.json \
    --tta  # 점수 향상, 시간 4배 증가
```

---

## 🐛 **문제 해결 (Troubleshooting)**

### **문제 1: CUDA out of memory**

```bash
# 해결책: batch_size 줄이기
python scripts/3_train.py \
    --config configs/experiments/exp001_baseline.yaml \
    --batch_size 8  # 기본값 16에서 8로 감소

# 또는 image_size 줄이기
python scripts/3_train.py \
    --config configs/experiments/exp001_baseline.yaml \
    --image_size 640  # 1280에서 640으로 감소
```

### **문제 2: 데이터를 찾을 수 없음**

```bash
# 데이터 경로 확인
ls -la data/raw/train_images
ls -la data/raw/train_annotations
ls -la data/raw/test_images

# 파일 개수 확인
find data/raw/train_images -type f | wc -l  # 232 예상
find data/raw/train_annotations -name "*.json" | wc -l  # 763 예상
find data/raw/test_images -type f | wc -l  # 842 예상

# 심볼릭 링크가 깨졌다면 다시 생성
rm -rf data/raw/train_images data/raw/train_annotations data/raw/test_images
bash setup_local_data.sh
```

### **문제 3: 학습이 수렴하지 않음**

```bash
# 1. Learning rate 감소
python scripts/3_train.py \
    --config configs/experiments/exp001_baseline.yaml \
    --lr 0.0005  # 기본값 0.001에서 감소

# 2. Warmup epochs 증가
python scripts/3_train.py \
    --config configs/experiments/exp001_baseline.yaml \
    --warmup_epochs 5

# 3. 데이터 검증
python scripts/1_create_coco_format.py \
    --train_images data/raw/train_images \
    --train_annotations data/raw/train_annotations \
    --output_dir data/coco_data \
    --validate \
    --verbose
```

### **문제 4: 제출 파일 형식 오류**

```bash
# 제출 파일 검증
python -c "
import pandas as pd

# CSV 로드
df = pd.read_csv('submissions/submission_exp001_*.csv')

# 기본 정보
print(f'Total rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Unique images: {df[\"image_id\"].nunique()}')

# 검증
print('\\nValidation:')
print(f'  Duplicate annotation_ids: {df[\"annotation_id\"].duplicated().sum()}')
print(f'  Negative bbox values: {(df[[\"bbox_x\",\"bbox_y\",\"bbox_w\",\"bbox_h\"]] < 0).sum().sum()}')
print(f'  Max detections per image: {df.groupby(\"image_id\").size().max()}')
print(f'  Score range: [{df[\"score\"].min():.3f}, {df[\"score\"].max():.3f}]')

# 샘플
print('\\nFirst 5 rows:')
print(df.head())
"
```

---

## 📝 **체크리스트: 첫 제출 전**

실제로 Kaggle에 제출하기 전 확인사항:

- [ ] 모델이 수렴했는가? (validation loss가 안정화됨)
- [ ] mAP@0.75-0.95 > 0.30 달성했는가?
- [ ] 제출 CSV가 생성되었는가?
- [ ] CSV validation 통과했는가? (위의 검증 스크립트 실행)
- [ ] category_mapping.json 사용되었는가?
- [ ] bbox 형식이 절대 좌표인가? (정규화되지 않음)
- [ ] image_id가 올바르게 추출되었는가?
- [ ] 총 842개 test 이미지 처리되었는가?
- [ ] 이미지당 평균 2-3개 detection이 있는가?

**모두 체크했다면 Kaggle에 제출하세요! 🚀**

---

## 🎯 **다음 단계**

### **점수 향상을 위한 아이디어**

1. **모델 크기 증가**
   - YOLOv8n → YOLOv8s → YOLOv8m

2. **이미지 크기 증가**
   - 640 → 1280 (작은 객체 검출 향상)

3. **학습 epoch 증가**
   - 50 → 100 → 150 epochs

4. **Threshold 튜닝**
   - confidence threshold: 0.15, 0.20, 0.25, 0.30, 0.35
   - NMS threshold: 0.40, 0.45, 0.50

5. **Test Time Augmentation**
   - `--tta` 플래그 사용

6. **앙상블**
   - 여러 모델 학습 후 예측 결합

---

## 💡 **유용한 명령어 모음**

```bash
# GPU 사용률 모니터링
watch -n 1 nvidia-smi

# 학습 로그 실시간 확인
tail -f runs/exp001_*/logs/exp001.log

# 디스크 용량 확인
du -sh data/* runs/*

# 특정 실험 결과만 보기
ls -lh runs/exp001_*

# 모든 제출 파일 보기
ls -lh submissions/

# 가장 최근 실험 찾기
ls -lt runs/ | head -5
```

---

## 📚 **참고 문서**

- **전체 가이드**: `docs/IMPLEMENTATION_COMPLETE.md`
- **프로젝트 상태**: `PROJECT_STATUS.md`
- **Phase 1**: `docs/PHASE1_COMPLETE.md`
- **Phase 2**: `docs/PHASE2_COMPLETE.md`
- **Phase 3**: `docs/PHASE3_COMPLETE.md`
- **Phase 4 & 5**: `docs/PHASE4_5_COMPLETE.md`

---

## 🤝 **도움이 필요하면**

1. 문서 확인: `docs/` 폴더
2. 로그 확인: `runs/exp00X_*/logs/`
3. Config 확인: `config_snapshot.yaml`
4. GitHub Issues에 질문 올리기

---

**행운을 빕니다! 🏆**

좋은 점수 받으세요! 💪
