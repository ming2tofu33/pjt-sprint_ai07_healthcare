# build_dataset Refactor Contract (Structure Only)

## Scope
- Target: `scripts/build_dataset.py` modularization/refactor.
- Allowed: function split/move, module/file reorganization, import cleanup, internal naming cleanup.
- Forbidden: algorithm change, logic/policy change, default value change, schema/column/order/location change, new behavior.

## 1) Current Pipeline Stages (Must Stay Equivalent)
1. Scan
- Load config and resolve paths.
- Scan train JSON (`*.json`) and train images (`*.png`, with duplicate-name index).
- If external is enabled, scan each external source JSON/images the same way.
- Parse JSON with encoding fallbacks (`utf-8` -> `utf-8-sig` -> `cp949` -> replace mode).

2. Validation
- Validate JSON parse/top-level structure.
- Enforce singleton structure (`images/annotations/categories`) per config.
- Validate `file_name`, image existence/open/size, bbox presence/length/numeric/positive width-height, and `category_id`.
- Apply external safety check for banned path/file patterns.

3. Normalization
- Train/external to common record schema.
- External `dl_idx` mapping to canonical `category_id` (with config fallback behavior).
- Width/height overwrite from real image (when configured).
- OOB handling (`clip` or `exclude`) and bbox/stat fields calculation.
- Append exclusion/fix/audit logs during normalization.

4. Merge
- Append normalized train/external rows into one `records` list.
- Optional external normalized artifact copy (`normalize_and_copy`) to processed outputs.

5. Dedup
- External-vs-train dedup (if enabled).
- Global exact dedup using configured key fields.

6. Split
- Derive `group_id` from `file_name` (regex + fallback).
- Group-wise train/val split with `random_seed` and `train_ratio`.

7. Export
- Write `df_clean.csv`, `excluded_rows.csv`, `fixes_bbox.csv`, `splits.csv`.
- Write configured audit CSV files.
- Write external-specific logs/mapping outputs when configured.

8. Manifest
- Write `preprocess_manifest.json` with config hash and summary counts.

## 2) Non-Negotiable Contracts

### 2.1 Input Contract (YAML)
- `configs/preprocess_v1_1.yaml` schema is immutable.
- Key names and meanings must not change.
- Top-level contract keys to keep:  
`version`, `random_seed`, `paths`, `outputs`, `label_contract`, `validation`, `oob`, `missing_label`, `dedup`, `integrity`, `split`, `audit`, `external_data`, `format`.

### 2.2 Required Output Files (Names + Locations + Formats)
- `df_clean.csv`: `paths.processed_dir / outputs.df_clean_name` (default `data/processed/df_clean.csv`)
- `excluded_rows.csv`: `paths.metadata_dir / outputs.excluded_rows_name` (default `data/metadata/excluded_rows.csv`)
- `fixes_bbox.csv`: `paths.metadata_dir / outputs.fixes_bbox_name` (default `data/metadata/fixes_bbox.csv`)
- `splits.csv`: `paths.metadata_dir / split.splits_name` (default `data/metadata/splits.csv`)
- `preprocess_manifest.json`: `paths.metadata_dir / outputs.manifest_name` (default `data/metadata/preprocess_manifest.json`)
- Audit files: every name in `audit.files` is created under `paths.metadata_dir` (even if empty).

### 2.3 Column/Schema Contracts (Must Stay Identical)
- `df_clean.csv` columns (exact order):
`file_name, source_json, width, height, group_id, category_id, bbox, bbox_x, bbox_y, bbox_w, bbox_h, bbox_area, bbox_area_ratio, bbox_aspect, is_oob, is_bad_bbox, is_exact_dup, source`
- `excluded_rows.csv` columns (exact order):
`source, file_name, source_json, reason_code, detail`
- `fixes_bbox.csv` columns (exact order):
`source, file_name, source_json, old_bbox, new_bbox, reason_code`
- `splits.csv` columns (exact order):
`group_id, split`
- `preprocess_manifest.json` keys:
`created_at`, `git_commit`, `config_path`, `config_sha256`, `summary`
- `summary` keys:
`total_rows`, `train_rows`, `external_rows`, `excluded_rows`, `fixes_bbox`, `unique_images`, `objects_per_image_dist`

### 2.4 Additional Output Contracts (When Configured)
- Audit CSV formatting rule:
- If audit file has rows, header columns are `sorted(rows[0].keys())`.
- If audit file has no rows, file is created as empty text (no header row).
- External OOB logs:
`external_data.alignment.oob.fixes_log_out`,
`external_data.alignment.oob.excluded_log_out`
- Category mapping output:
`external_data.category_id_mapping.mapping_table_out` (if `save_mapping_table: true`, columns: `dl_idx, canonical_category_id, name`)
- Unmapped external log:
`external_data.category_id_mapping.unmapped_log_out`

### 2.5 Encoding/Write Behavior Contracts
- `df_clean.csv` encoding = `format.csv_encoding` (default `utf-8`).
- Other CSV files are written as UTF-8.
- `preprocess_manifest.json` is UTF-8, `ensure_ascii=False`, `indent=2`.
- Relative output paths are resolved from current working directory via existing path resolution behavior.

### 2.6 Raw Data Immutability Contract
- `data/raw/**` is read-only.
- No create/update/delete/rename/move under `data/raw`.
- All writes must remain under processed/metadata or explicit configured output paths outside raw.

## 3) Refactor Equivalence Checklist

Use same config file and same raw dataset for pre/post refactor runs.

1. File existence/location
- [ ] Required 5 outputs exist in exact expected directories.
- [ ] All configured audit files exist under `metadata_dir`.

2. `df_clean.csv` equivalence
- [ ] Total row count identical.
- [ ] `source` distribution identical (`train`/`external`).
- [ ] `is_oob=True` count identical.
- [ ] Unique `file_name` count identical.
- [ ] `category_id` distribution identical.
- [ ] `group_id` count/set identical.
- [ ] `bbox_x/y/w/h` numeric values identical.

3. Log/audit equivalence
- [ ] `excluded_rows.csv` row count identical.
- [ ] `excluded_rows.csv` `reason_code` distribution identical.
- [ ] `fixes_bbox.csv` row count identical.
- [ ] `fixes_bbox.csv` `reason_code` distribution identical.
- [ ] Each audit CSV row count identical (including empty-file behavior).

4. Split equivalence
- [ ] `splits.csv` row count identical.
- [ ] Train/val split counts identical.
- [ ] `group_id -> split` mapping identical.

5. Manifest equivalence
- [ ] `summary` numeric values identical.
- [ ] `config_sha256` identical.
- [ ] `created_at` is expected to differ by run time.
- [ ] `git_commit` may differ only if HEAD changed.

6. Raw data safety
- [ ] No modified files under `data/raw/**` after run.

## 4) Acceptance Rule
- Refactor is accepted only if all checklist items pass.
- Any mismatch means rollback/fix refactor structure only; do not adjust algorithm/policy to "fit" new outputs.
