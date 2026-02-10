set -euo pipefail

# ============================================
# setup_structure.sh
# - Sprint AI07 Healthcare Object Detection 프로젝트 스캐폴딩 스크립트
# - 현재 최종 구조(모듈화된 dataprep 및 실험 설정 포함)를 반영
# ============================================

ROOT="pjt-sprint_ai07_healthcare"

# --- 헬퍼 함수 ---
say() { printf "%s\n" "$*"; }

mkfile() {
  local f="$1"
  mkdir -p "$(dirname "$f")"
  [ -f "$f" ] || : > "$f"
}

# --- 0) 루트 디렉토리 생성 및 이동 ---
mkdir -p "$ROOT"
cd "$ROOT"

# --- 1) 디렉토리 구조 생성 ---
mkdir -p \
  .claude \
  artifacts/best_models \
  artifacts/submissions \
  configs/experiments \
  data/metadata \
  data/processed/cache \
  data/processed/datasets \
  data/raw/train_images \
  data/raw/train_annotations \
  data/raw/test_images \
  data/raw/external/combined/annotations \
  data/raw/external/combined/images \
  docs \
  playground \
  runs \
  scripts \
  src/dataprep/output \
  src/dataprep/process \
  src/dataprep/setup \
  src/engine \
  src/inference \
  src/models \
  src/utils

# --- 2) .gitkeep 파일 생성 (빈 디렉토리 추적용) ---
find data artifacts configs/experiments playground runs -type d -empty -exec touch {}/.gitkeep \;

# --- 3) 프로젝트 루트 파일 ---
mkfile ".gitignore"
mkfile "CLAUDE.md"
mkfile "README.md"
mkfile "requirements.txt"
mkfile "setup_structure.sh"

# --- 4) 설정 및 문서 파일 ---
mkfile "configs/base.yaml"
mkfile "configs/experiments/baseline.yaml"
mkfile "configs/experiments/smoke_test.yaml"
mkfile "docs/00_quickstart.md"
mkfile "docs/01_data_pipeline.md"
mkfile "docs/02_experiments.md"
mkfile "runs/_registry.csv"

# --- 5) 실행 스크립트 (Entrypoints) ---
mkfile "scripts/0_split_data.py"
mkfile "scripts/1_preprocess.py"
mkfile "scripts/2_train.py"
mkfile "scripts/3_evaluate.py"
mkfile "scripts/4_submission.py"
mkfile "scripts/run_pipeline.sh"
chmod +x scripts/run_pipeline.sh || true

# --- 6) 소스 코드 (src/) ---
# __init__.py 파일들
find src -type d -exec touch {}/__init__.py \;

# src/dataprep (Facade & Utils)
mkfile "src/dataprep/config.py"
mkfile "src/dataprep/dedup.py"
mkfile "src/dataprep/export.py"
mkfile "src/dataprep/io_utils.py"
mkfile "src/dataprep/manifest.py"
mkfile "src/dataprep/normalize.py"
mkfile "src/dataprep/quality_audit.py"
mkfile "src/dataprep/split.py"

# src/dataprep/output
mkfile "src/dataprep/output/data_pipeline.py"
mkfile "src/dataprep/output/export_yolo.py"
mkfile "src/dataprep/output/export.py"
mkfile "src/dataprep/output/manifest.py"

# src/dataprep/process
mkfile "src/dataprep/process/build_dataset.py"
mkfile "src/dataprep/process/dedup.py"
mkfile "src/dataprep/process/io_utils.py"
mkfile "src/dataprep/process/normalize.py"
mkfile "src/dataprep/process/quality_audit.py"
mkfile "src/dataprep/process/split.py"

# src/dataprep/setup
mkfile "src/dataprep/setup/io_utils.py"

# 기타 모듈
mkfile "src/engine/trainer.py"
mkfile "src/engine/validator.py"
mkfile "src/inference/postprocess.py"
mkfile "src/inference/predictor.py"
mkfile "src/inference/submission.py"
mkfile "src/models/architectures.py"
mkfile "src/models/detector.py"
mkfile "src/utils/config_loader.py"
mkfile "src/utils/logger.py"
mkfile "src/utils/registry.py"
mkfile "src/utils/visualizer.py"

# --- 7) 기본 .gitignore 내용 작성 ---
cat <<'EOF' > .gitignore
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
playground/
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


# ============================================
# Claude / AI coding assistants (local-only)
# ============================================
.claude/
**/.claude/
claude*.log
**/claude*.log
**/*claude*cache*
EOF

say "✅ 최종 구조를 반영한 프로젝트 스캐폴딩이 준비되었습니다: $ROOT"
