#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="exp010"
CONFIG="configs/experiments/exp010_yolo11s.yaml"

step() {
  echo "$1"
  shift
  "$@"
}

step "[exp010] Stage 0: COCO format" \
  python scripts/0_create_coco_format.py --config "$CONFIG" --run-name "$RUN_NAME"

step "[exp010] Stage 1: Train/Val split" \
  python scripts/1_splitting.py --config "$CONFIG" --run-name "$RUN_NAME"

step "[exp010] Stage 2: Prepare YOLO dataset" \
  python scripts/2_prepare_yolo_dataset.py --run-name "$RUN_NAME" --symlink

step "[exp010] Stage 3: Train" \
  python scripts/3_train.py --config "$CONFIG" --run-name "$RUN_NAME"

step "[exp010] Stage 4: Evaluate" \
  python scripts/4_evaluate.py --config "$CONFIG" --run-name "$RUN_NAME"

step "[exp010] Stage 5: Submission" \
  python scripts/5_submission.py --config "$CONFIG" --run-name "$RUN_NAME"
