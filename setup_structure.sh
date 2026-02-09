#!/usr/bin/env bash
set -euo pipefail

# ============================================
# setup_structure.sh
# - Sprint AI07 Healthcare Object Detection project scaffold
# - Creates directories, placeholder files, and .gitignore
# ============================================

ROOT="pjt-sprint_ai07_healthcare"

# --- helpers ---
say() { printf "%s\n" "$*"; }

mkfile() {
  local f="$1"
  mkdir -p "$(dirname "$f")"
  [ -f "$f" ] || : > "$f"
}

# --- 0) root ---
mkdir -p "$ROOT"
cd "$ROOT"

# --- 1) core directories ---
mkdir -p \
  docs \
  data/raw/train_images \
  data/raw/train_annotations \
  data/raw/test_images \
  data/raw/external/combined/annotations \
  data/raw/external/combined/images \
  data/processed/manifests \
  data/processed/annotations \
  data/processed/cache \
  data/metadata \
  configs/experiments \
  src/dataprep/setup \
  src/dataprep/process \
  src/dataprep/output \
  src/models \
  src/engine \
  src/inference \
  src/utils \
  scripts \
  runs \
  artifacts/best_models \
  artifacts/submissions \
  playground

# --- 2) .gitkeep (keep empty dirs tracked) ---
mkfile "data/raw/.gitkeep"
mkfile "data/raw/train_images/.gitkeep"
mkfile "data/raw/train_annotations/.gitkeep"
mkfile "data/raw/test_images/.gitkeep"
mkfile "data/raw/external/.gitkeep"
mkfile "data/raw/external/combined/.gitkeep"
mkfile "data/raw/external/combined/annotations/.gitkeep"
mkfile "data/raw/external/combined/images/.gitkeep"

mkfile "data/processed/.gitkeep"
mkfile "data/processed/manifests/.gitkeep"
mkfile "data/processed/annotations/.gitkeep"
mkfile "data/processed/cache/.gitkeep"
mkfile "data/metadata/.gitkeep"

mkfile "configs/experiments/.gitkeep"
mkfile "runs/.gitkeep"
mkfile "artifacts/.gitkeep"
mkfile "artifacts/best_models/.gitkeep"
mkfile "artifacts/submissions/.gitkeep"
mkfile "playground/.gitkeep"

# --- 3) python package init files ---
mkfile "src/__init__.py"
mkfile "src/dataprep/__init__.py"
mkfile "src/dataprep/setup/__init__.py"
mkfile "src/dataprep/process/__init__.py"
mkfile "src/dataprep/output/__init__.py"
mkfile "src/models/__init__.py"
mkfile "src/engine/__init__.py"
mkfile "src/inference/__init__.py"
mkfile "src/utils/__init__.py"

# --- 4) src module stubs (touch only; content intentionally blank) ---
mkfile "src/dataprep/setup/io_utils.py"

mkfile "src/dataprep/process/normalize.py"
mkfile "src/dataprep/process/dedup.py"
mkfile "src/dataprep/process/quality_audit.py"
mkfile "src/dataprep/process/split.py"

mkfile "src/dataprep/output/export.py"
mkfile "src/dataprep/output/manifest.py"
mkfile "src/dataprep/output/data_pipeline.py"

mkfile "src/models/detector.py"
mkfile "src/models/architectures.py"

mkfile "src/engine/trainer.py"
mkfile "src/engine/validator.py"

mkfile "src/inference/predictor.py"
mkfile "src/inference/postprocess.py"
mkfile "src/inference/submission.py"

mkfile "src/utils/config_loader.py"
mkfile "src/utils/logger.py"
mkfile "src/utils/visualizer.py"
mkfile "src/utils/registry.py"

# --- 5) scripts (entrypoints) ---
mkfile "scripts/0_split_data.py"
mkfile "scripts/1_preprocess.py"
mkfile "scripts/2_train.py"
mkfile "scripts/3_evaluate.py"
mkfile "scripts/4_submission.py"
mkfile "scripts/run_pipeline.sh"

# make pipeline script executable (safe even if empty)
chmod +x "scripts/run_pipeline.sh" || true

# --- 6) configs + docs + misc ---
mkfile "configs/base.yaml"

mkfile "docs/00_quickstart.md"
mkfile "docs/01_data_pipeline.md"
mkfile "docs/02_experiments.md"

mkfile "README.md"
mkfile "requirements.txt"
mkfile "runs/_registry.csv"

# --- 7) .gitignore ---
cat <<'EOF' > .gitignore
# ============================================
# 0. OS / IDE / Python artifacts
# ============================================
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.vscode/
.idea/
.DS_Store
Thumbs.db

# venv / build
venv/
env/
.venv/
dist/
build/
*.egg-info/

# ============================================
# 1. Data (exclude by default; keep structure with .gitkeep)
# ============================================
data/**
!data/**/.gitkeep

# If you decide to version some metadata artifacts, whitelist explicitly here:
# !data/metadata/**/*.json

# ============================================
# 2. Runs / Artifacts (exclude; keep registry + placeholders)
# ============================================
runs/**
!runs/.gitkeep
!runs/_registry.csv

artifacts/**
!artifacts/.gitkeep
!artifacts/**/.gitkeep

# playground is personal
playground/**
!playground/.gitkeep

# ============================================
# 3. Model weights / large binaries
# ============================================
*.pt
*.pth
*.onnx
*.weights

# ============================================
# 4. ML tool logs / trackers
# ============================================
wandb/
mlruns/
lightning_logs/
.ultralytics/
yolo_settings.json
*.log
nohup.out
EOF

say "✅ Project scaffold created at: $(pwd)"
say "🚀 Next:"
say "  1) Put Kaggle data into data/raw/{train_images,train_annotations,test_images}"
say "  2) git add . && git commit -m \"init scaffold\""