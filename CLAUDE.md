# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## 1. Project Intent

Healthcare pill object detection project using YOLO (Ultralytics).

Primary goals:

- build a reproducible training/inference pipeline
- prevent data leakage and submission mistakes
- run experiments quickly with clear logs and artifacts

Competition-facing constraints:

- keep `category_id` mapping consistent across the pipeline
- submit at most Top-4 detections per image
- treat `data/raw/` as immutable

## 2. Standard Directory Baseline

Use this structure as the canonical baseline:

```text
pjt-sprint_ai07_healthcare/
├── .gitignore
├── README.md
├── requirements.txt
├── docs/
│   ├── 00_quickstart.md
│   ├── 01_data_pipeline.md
│   └── 02_experiments.md
├── data/
│   ├── raw/
│   │   ├── train_images/
│   │   ├── train_annotations/
│   │   ├── test_images/
│   │   └── external/combined/{annotations,images}
│   ├── processed/
│   │   ├── manifests/
│   │   ├── annotations/
│   │   └── cache/
│   └── metadata/
├── configs/
│   ├── base.yaml
│   └── experiments/
├── src/
│   ├── dataprep/{setup,process,output}
│   ├── models/
│   ├── engine/
│   ├── inference/
│   └── utils/
├── scripts/
│   ├── 0_split_data.py
│   ├── 1_preprocess.py
│   ├── 2_train.py
│   ├── 3_evaluate.py
│   ├── 4_submission.py
│   └── run_pipeline.sh
├── runs/
│   ├── exp_YYYYMMDD_HHMMSS/
│   └── _registry.csv
├── artifacts/
│   ├── best_models/
│   └── submissions/
└── playground/
```

## 3. Pipeline Stages (Canonical)

Run from repository root:

```bash
RUN_NAME="exp_20260209_120000"
CONFIG="configs/experiments/*.yaml"

python scripts/0_split_data.py --run-name $RUN_NAME --config $CONFIG
python scripts/1_preprocess.py --run-name $RUN_NAME --config $CONFIG
python scripts/2_train.py --run-name $RUN_NAME --config $CONFIG
python scripts/3_evaluate.py --run-name $RUN_NAME --config $CONFIG
python scripts/4_submission.py --run-name $RUN_NAME --config $CONFIG
```

Optional one-command flow:

```bash
bash scripts/run_pipeline.sh --run-name $RUN_NAME --config $CONFIG
```

## 4. Module Responsibilities

- `src/dataprep/setup/io_utils.py`: robust file scanning/loading
- `src/dataprep/process/normalize.py`: schema/bbox normalization
- `src/dataprep/process/dedup.py`: dedup and rule-based filtering
- `src/dataprep/process/quality_audit.py`: image-based audit checks
- `src/dataprep/process/split.py`: group-based split for leakage prevention
- `src/dataprep/output/export.py`: output files for cleaned data and logs
- `src/dataprep/output/export_yolo.py`: conversion to YOLO format (Stage 1)
- `src/dataprep/output/manifest.py`: reproducibility metadata snapshot
- `src/dataprep/output/data_pipeline.py`: orchestration for dataprep flow
- `src/models/detector.py`: model wrapper for training/inference
- `src/inference/predictor.py`: batch prediction entry
- `src/inference/postprocess.py`: threshold/NMS/Top-4 rule handling
- `src/inference/submission.py`: final CSV generation and checks
- `src/utils/config_loader.py`: base+override config merge
- `src/utils/logger.py`: run logs and metrics output
- `src/utils/visualizer.py`: failure and prediction visualization
- `src/utils/registry.py`: `runs/_registry.csv` tracking

## 5. Config Rules

- keep common defaults in `configs/base.yaml`
- keep experiment overrides in `configs/experiments/*.yaml`
- persist merged config per run as `runs/<exp>/config_resolved.yaml`
- manage experiment conditions in YAML, not hardcoded Python

## 6. Reproducibility Rules

- do not modify files under `data/raw/`
- keep split deterministic (fixed seed) and persist manifests
- persist cleaned outputs and audit logs for traceability
- update `runs/_registry.csv` with experiment metadata

## 7. Priority During Competition

Implement now (core):

- `scripts/0_split_data.py` ~ `scripts/4_submission.py`
- `src/dataprep/*` pipeline
- `src/models/detector.py`
- `src/inference/predictor.py`, `src/inference/postprocess.py`
- `src/utils/config_loader.py`, `src/utils/logger.py`, `src/utils/visualizer.py`

Defer if schedule is tight:

- `src/models/architectures.py` (custom architectures)
- `src/engine/trainer.py`, `src/engine/validator.py` (full custom loops)
- advanced logging/dashboard automation

## 8. References

- `README.md`
- `docs/00_quickstart.md`
- `docs/01_data_pipeline.md`
- `docs/02_experiments.md`

