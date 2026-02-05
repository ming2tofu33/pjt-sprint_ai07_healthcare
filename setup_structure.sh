#!/bin/bash

# 1. 핵심 디렉토리 생성
mkdir -p data/raw/{train_images,train_annotations,test_images}
mkdir -p data/splits
mkdir -p data/coco_data/meta
mkdir -p configs/experiments
mkdir -p src/{data,models,training,evaluation,inference,utils}
mkdir -p scripts
mkdir -p notebooks
mkdir -p runs
mkdir -p artifacts/{best_models,submissions}
mkdir -p docs

# 2. .gitkeep 생성 (Git이 빈 폴더를 인식하고 구조를 유지하게 함)
touch data/raw/.gitkeep
touch data/splits/.gitkeep
touch data/coco_data/.gitkeep
touch runs/.gitkeep
touch artifacts/.gitkeep
touch notebooks/.gitkeep

# 3. src 파일들 (MVP 기반 평면 구조)
touch src/__init__.py
touch src/{data_loader,model,train_loop,infer,utils}.py

# 4. scripts 파일들 (이미지 흐름 반영)
touch scripts/{0_splitting,1_create_coco_format,3_train,4_evaluate,5_submission}.py

# 5. configs
touch configs/base.yaml
touch configs/experiments/exp001_baseline.yaml

# 6. 문서 및 환경 설정
touch docs/{SETUP,WORKFLOW}.md
touch requirements.txt
touch README.md
touch runs/_registry.csv

# 4. .gitignore 자동 생성
cat <<EOF > .gitignore
# ============================================
# 1. Project Specific (Data & Stage-wise)
# ============================================

# [STAGE 0~1] 원본 및 전처리 데이터 제외
# 부모 폴더를 제외하되 특정 파일 예외 처리를 위해 /* 패턴 사용
data/*
!data/coco_data/

# [STAGE 3~4] 실험 결과물
runs/*
!runs/.gitkeep
!runs/_registry.csv

# [STAGE 5] 최종 산출물
artifacts/
!artifacts/.gitkeep
!artifacts/best_models/.gitkeep
!submissions/.gitkeep


# ============================================

# 모델 가중치 (대용량 바이너리)
*.pt
*.pth
*.onnx
*.weights

# ============================================
# 2. Python & Development
# ============================================
__pycache__/
*.py[cod]
venv/
env/
.venv/
dist/
build/
*.egg-info/

# ============================================
# 3. Jupyter Notebook & IDEs
# ============================================
.ipynb_checkpoints/
.vscode/
.idea/
.DS_Store
Thumbs.db

# ============================================
# 4. ML Tools & Logs (Ultralytics / Tracking)
# ============================================
yolo_settings.json
.ultralytics/
wandb/
mlruns/
lightning_logs/
*.log
nohup.out

# ============================================
# 5. Exception (공유 권장 파일)
# ============================================
*.csv
!runs/_registry.csv
!data/coco_data/meta/*.json
EOF

echo "✅ [Health Eat] MLOps 표준 구조 및 .gitignore 세팅 완료!"
echo "🚀 'git add .'를 통해 빈 폴더 구조를 먼저 커밋하세요."