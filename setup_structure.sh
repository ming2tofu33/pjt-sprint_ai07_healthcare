#!/usr/bin/env bash
set -euo pipefail

# setup_structure.sh
# Create a clean project skeleton for pjt-sprint_ai07_healthcare.
# Usage:
#   bash setup_structure.sh [target_root]

ROOT="${1:-pjt-sprint_ai07_healthcare}"

say() {
  printf "%s\n" "$*"
}

mkfile() {
  local f="$1"
  mkdir -p "$(dirname "$f")"
  [[ -f "$f" ]] || : > "$f"
}

mkdir -p "$ROOT"
cd "$ROOT"

# 1) Directory layout
mkdir -p \
  artifacts/best_models \
  artifacts/submissions \
  configs/experiments \
  data/raw/train_images \
  data/raw/train_annotations \
  data/raw/test_images \
  data/raw/external/combined/annotations \
  data/raw/external/combined/images \
  data/processed/manifests \
  data/processed/annotations \
  data/processed/cache \
  data/processed/datasets \
  data/metadata \
  docs \
  playground \
  runs \
  scripts \
  src/dataprep/setup \
  src/dataprep/process \
  src/dataprep/output \
  src/models \
  src/engine \
  src/inference \
  src/utils

# 2) Placeholder files
mkfile data/raw/.gitkeep
mkfile data/processed/.gitkeep
mkfile data/processed/manifests/.gitkeep
mkfile data/processed/annotations/.gitkeep
mkfile data/metadata/.gitkeep
mkfile artifacts/.gitkeep
mkfile artifacts/best_models/.gitkeep
mkfile artifacts/submissions/.gitkeep
mkfile runs/.gitkeep
mkfile runs/_registry.csv
mkfile playground/.gitkeep

# 3) Root files
mkfile .gitignore
mkfile README.md
mkfile requirements.txt

# 4) Docs
mkfile docs/00_quickstart.md
mkfile docs/01_data_pipeline.md
mkfile docs/02_experiments.md

# 5) Configs
mkfile configs/base.yaml
mkfile configs/experiments/baseline.yaml

# 6) Entrypoint scripts
mkfile scripts/0_split_data.py
mkfile scripts/1_preprocess.py
mkfile scripts/2_train.py
mkfile scripts/3_evaluate.py
mkfile scripts/4_submission.py
mkfile scripts/run_pipeline.sh
chmod +x scripts/run_pipeline.sh || true

# 7) Source package files
find src -type d -exec sh -c '[[ -f "$1/__init__.py" ]] || : > "$1/__init__.py"' _ {} \;

mkfile src/dataprep/setup/io_utils.py
mkfile src/dataprep/process/normalize.py
mkfile src/dataprep/process/dedup.py
mkfile src/dataprep/process/quality_audit.py
mkfile src/dataprep/process/split.py
mkfile src/dataprep/output/export.py
mkfile src/dataprep/output/manifest.py
mkfile src/dataprep/output/data_pipeline.py

mkfile src/models/detector.py
mkfile src/models/architectures.py
mkfile src/engine/trainer.py
mkfile src/engine/validator.py
mkfile src/inference/predictor.py
mkfile src/inference/postprocess.py
mkfile src/inference/submission.py
mkfile src/utils/config_loader.py
mkfile src/utils/logger.py
mkfile src/utils/visualizer.py
mkfile src/utils/registry.py

# 8) Default .gitignore
cat <<'EOF' > .gitignore
# ============================================================
# Project outputs and large assets
# ============================================================

data/**
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/processed/manifests/.gitkeep
!data/processed/annotations/.gitkeep
!data/metadata/.gitkeep

runs/**
!runs/.gitkeep
!runs/_registry.csv

artifacts/**
!artifacts/.gitkeep
!artifacts/best_models/.gitkeep
!artifacts/submissions/.gitkeep

playground/**
!playground/.gitkeep

*.pt
*.pth
*.onnx
*.weights

__pycache__/
*.py[cod]
venv/
.venv/
env/
dist/
build/
*.egg-info/

.ipynb_checkpoints/
.vscode/
.idea/
.DS_Store
Thumbs.db

yolo_settings.json
.ultralytics/
wandb/
mlruns/
lightning_logs/
*.log
nohup.out

.claude/
**/.claude/
claude*.log
**/claude*.log
**/*claude*cache*
EOF

say "[OK] Project skeleton prepared at: $ROOT"