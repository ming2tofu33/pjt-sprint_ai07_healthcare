# CLAUDE.md

This file is the repo-specific working guide for Claude Code.
It reflects the current implementation on this branch.

## 1. Project Scope

Healthcare pill object detection pipeline using Ultralytics YOLO.

Core constraints:

- Keep `category_id` mapping consistent end-to-end.
- Keep submission format strict (`annotation_id,image_id,category_id,bbox_x,bbox_y,bbox_w,bbox_h,score`).
- Keep max detections per image at Top-4 by default.
- Treat `data/raw/` as immutable source data.

## 2. Current Pipeline Entry Points

Main scripts:

- `scripts/0_split_data.py`
- `scripts/1_preprocess.py`
- `scripts/2_train.py`
- `scripts/3_evaluate.py`
- `scripts/4_submission.py`
- `scripts/run_pipeline.sh` (wrapper)

Standard run:

```bash
RUN_NAME="exp_YYYYMMDD_HHMMSS"
CONFIG="configs/experiments/<name>.yaml"

python scripts/0_split_data.py --run-name "$RUN_NAME" --config "$CONFIG"
python scripts/1_preprocess.py --run-name "$RUN_NAME" --config "$CONFIG"
python scripts/2_train.py --run-name "$RUN_NAME" --config "$CONFIG"
python scripts/3_evaluate.py --run-name "$RUN_NAME" --config "$CONFIG"
python scripts/4_submission.py --run-name "$RUN_NAME" --config "$CONFIG"
```

## 3. Config System (Actual Behavior)

- Experiment configs are loaded by `src/utils/config_loader.py`.
- `_base_` chains are supported recursively with deep-merge.
- Required path keys are validated (`train_images_dir`, `train_annotations_dir`, `test_images_dir`, `processed_dir`, `metadata_dir`, `datasets_dir`, `runs_dir`, `best_models_dir`, `submissions_dir`).
- Runtime precedence is generally:
  - CLI argument override
  - experiment yaml
  - base yaml (`configs/base.yaml`)

Important defaults in `configs/base.yaml`:

- `paths.artifact_layout: "compact"`
- `manual_overrides.exclude_file_names_file: "configs/resources/exclude_4444_208.txt"`
- `external_data.enabled: true`
- `yolo_convert.verify_labels: true`
- `train.competition_select.enabled: true`

## 4. Stage Contracts

### Stage 0: Clean + Split + COCO/Label Map

Script: `scripts/0_split_data.py`

Does:

- Loads config and merges manual exclude names from `exclude_file_names_file`.
- Runs dataprep orchestrator `src/dataprep/output/data_pipeline.py`.
- Patches `processed_dir` to `.../cache/<run_name>` during execution.
- Writes split artifacts and mapping metadata.

Key outputs:

- `data/processed/cache/<run_name>/df_clean.csv` (name configurable)
- `data/processed/cache/<run_name>/train_merged_coco.json`
- `data/processed/cache/<run_name>/label_map_full.json`
- `data/processed/cache/<run_name>/category_id_to_name.json`
- `data/processed/cache/<run_name>/image_id_map.json`
- `data/processed/cache/<run_name>/splits/split_train_valid.json`
- `data/processed/cache/<run_name>/splits/train_ids.txt`
- `data/processed/cache/<run_name>/splits/valid_ids.txt`
- `data/metadata/splits.csv`
- `data/metadata/preprocess_manifest.json` + audit csv files

### Stage 1: YOLO Dataset Export

Script: `scripts/1_preprocess.py`

Does:

- Reads Stage 0 outputs (`df_clean.csv`, `splits.csv`).
- Calls `src/dataprep/output/export_yolo.py::run_export`.
- Uses hardlink-first (`link_mode`) with copy fallback.
- Writes `class_map.csv`.
- Optionally verifies labels (`verify_labels`).

Key outputs:

- `data/processed/datasets/<dataset_prefix>_<run_name>/data.yaml`
- `.../images/train`, `.../images/val`, `.../labels/train`, `.../labels/val`
- `data/metadata/class_map.csv`
- `convert_manifest.json` in dataset dir

Note:

- No offline minority augmentation pipeline is currently wired in this branch.
- No balanced train-manifest auto-builder is currently wired in this branch.

### Stage 2: Train

Script: `scripts/2_train.py`

Does:

- Resolves `data.yaml`:
  - CLI `--data/--data-yaml` first
  - otherwise `data/processed/datasets/<dataset_prefix>_<run_name>/data.yaml`
- Supports many CLI train overrides (`--epochs`, `--imgsz`, `--batch`, `--optimizer`, `--mosaic`, etc.).
- Supports class filtering:
  - `train.classes` (class index list), or
  - `train.target_category_ids` (mapped through `data.yaml names`)
- Supports resume modes:
  - `--resume`: strict, requires `runs/<run_name>/weights/last.pt`
  - `--auto-resume`: resumes only if `last.pt` exists
- Trains through `src/models/detector.py::PillDetector.train`.
- Copies best weights and writes metrics.
- Runs `competition_select` (default on): compares `best.pt` vs `last.pt` by `mAP75_95` and writes `competition_best.pt`.

Key outputs:

- `runs/<run_name>/config_resolved.yaml`
- `runs/<run_name>/metrics.json`
- `runs/<run_name>/weights/best.pt`
- `runs/<run_name>/weights/last.pt` (if produced)
- `runs/<run_name>/weights/competition_best.pt` (if competition select enabled and candidates exist)
- `runs/<run_name>/competition_best.json`
- `artifacts/best_models/<run_name>_best.pt`
- `artifacts/best_models/<run_name>_competition_best.pt` (if selected)
- `runs/_registry.csv` updated

Artifact layout behavior:

- `compact`: train artifacts under `runs/<run_name>/train/` and `results.csv` shortcut copied to `runs/<run_name>/results.csv`
- `legacy`: train artifacts under `runs/<run_name>/`

### Stage 3: Evaluate

Script: `scripts/3_evaluate.py`

Does:

- Selects weight priority: `competition_best.pt` -> `best.pt`.
- Validates on the Stage 1 `data.yaml`.
- Writes eval metrics into existing `metrics.json` with `eval_` prefix keys.
- Writes eval artifacts to `runs/<run_name>/eval/val`.
- Updates `runs/_registry.csv`.

### Stage 4: Submission

Script: `scripts/4_submission.py`

Does:

- Selects weight priority: CLI `--weights` > `competition_best.pt` > `best.pt`.
- Loads class mapping from:
  - CLI `--class-map`, or
  - Stage 0 `label_map_full.json`
- Runs inference via `src/inference/predictor.py::batch_predict`.
- Postprocesses via `src/inference/postprocess.py`:
  - Top-K per image (`max_det_per_image`)
  - global `min_conf`
  - per-class `class_min_conf_csv` override
  - category whitelist (`keep_category_ids` + file)
- Writes CSV and validates submission format/rules.
- Writes submission manifest.

Key outputs:

- default csv: `artifacts/submissions/<run_name>_conf<conf>.csv`
- manifest:
  - `compact`: `runs/<run_name>/submit/submission_manifest.json` (+ shortcut `runs/<run_name>/submission_manifest.json`)
  - `legacy`: `runs/<run_name>/submission_manifest.json`
- debug images:
  - `compact`: `runs/<run_name>/submit/debug/`
  - `legacy`: `runs/<run_name>/submission_debug/`

## 5. Important Modules

- `src/dataprep/output/data_pipeline.py`: Stage 0 orchestration
- `src/dataprep/output/export.py`: cleaned/audit output writers
- `src/dataprep/output/export_yolo.py`: Stage 1 conversion and label verification
- `src/models/detector.py`: train/val/predict wrapper and metric extraction
- `src/inference/predictor.py`: batch inference adapter
- `src/inference/postprocess.py`: Top-K and confidence/category filtering
- `src/inference/submission.py`: CSV writer, validator, manifest writer
- `src/utils/registry.py`: `runs/_registry.csv` append/update

## 6. Known Caveats (Current Code)

- `scripts/run_pipeline.sh` references `TRAIN_DATA_YAML_ARG` in Stage 2 command, but this array is not defined in the script.
- `train.rect` is not currently forwarded to Ultralytics in `src/models/detector.py` (`_DIRECT_KEYS` has no `rect`).
- Base config enables external data ingestion by default (`external_data.enabled: true`), so experiment YAML should override explicitly when needed.

## 7. Working Rules for Assistants

- Do not rewrite raw dataset under `data/raw/`.
- Prefer config-driven changes (`configs/base.yaml` + `configs/experiments/*.yaml`) over hardcoded constants.
- Keep outputs reproducible and traceable (manifest/metrics/registry updates must remain intact).
