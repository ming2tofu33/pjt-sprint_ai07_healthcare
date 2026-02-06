from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter
from typing import Iterable, Optional, Tuple

from src.config import resolve_path
from src.dedup import add_train_dedup_key, dedup_exact_records, should_keep_external_record_after_dedup
from src.export import write_outputs
from src.io_utils import parse_one_json, scan_image_files, scan_json_files
from src.manifest import write_manifest
from src.normalize import normalize_record
from src.split import add_group_id, make_group_split, write_splits


def build_train_mapping(
    train_json_paths: Iterable[Path], mapping_key: str
) -> Tuple[dict[str, int], dict[str, str], list[dict[str, str]]]:
    """
    Build dl_idx -> canonical category_id/name from train annotations.
    Returns (id_map, name_map, suspect_rows).
    """
    id_map: dict[str, int] = {}
    name_map: dict[str, str] = {}
    suspect_rows: list[dict[str, str]] = []

    for p in train_json_paths:
        data, err = parse_one_json(p)
        if err or data is None:
            continue
        images = data.get("images") or []
        annotations = data.get("annotations") or []
        categories = data.get("categories") or []
        if len(images) != 1 or len(annotations) != 1 or len(categories) != 1:
            continue

        img0 = images[0]
        ann0 = annotations[0]
        cat0 = categories[0]

        dl_idx = img0.get(mapping_key)
        if dl_idx is None:
            continue
        dl_idx = str(dl_idx).strip()
        if dl_idx == "":
            continue

        cat_id = ann0.get("category_id")
        if not isinstance(cat_id, int):
            try:
                cat_id = int(str(cat_id))
            except Exception:
                continue

        prev = id_map.get(dl_idx)
        if prev is not None and prev != cat_id:
            suspect_rows.append(
                {
                    "source_json": str(p),
                    "dl_idx": dl_idx,
                    "prev_category_id": str(prev),
                    "new_category_id": str(cat_id),
                    "reason": "inconsistent_category_id",
                }
            )
            continue
        id_map[dl_idx] = cat_id

        name = cat0.get("name")
        if isinstance(name, str) and name.strip():
            name_map.setdefault(dl_idx, name)

    return id_map, name_map, suspect_rows


def build_df_clean(
    config: dict, config_path: Path, *, repo_root: Path, quiet: bool = False, log_every: int = 500
) -> dict:
    base_dir = repo_root
    paths_cfg = config.get("paths", {})
    outputs_cfg = config.get("outputs", {})

    train_images_dir = resolve_path(paths_cfg["train_images_dir"], base_dir)
    train_ann_dir = resolve_path(paths_cfg["train_annotations_dir"], base_dir)
    processed_dir = resolve_path(paths_cfg["processed_dir"], base_dir)
    metadata_dir = resolve_path(paths_cfg["metadata_dir"], base_dir)

    metadata_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Audit logs (always create keys, even if empty)
    audit_files = config.get("audit", {}).get("files", [])
    audit_logs = {name: [] for name in audit_files}
    audit_logs.setdefault("audit_unmapped_external.csv", [])

    logs: dict[str, list[dict]] = {
        "excluded_rows": [],
        "fixes_bbox": [],
        "excluded_rows_external": [],
        "fixes_bbox_external": [],
    }

    # Train mapping from dl_idx -> category_id/name
    train_json_paths = scan_json_files(train_ann_dir, recursive=True)
    mapping_key = config.get("external_data", {}).get("category_id_mapping", {}).get("mapping_key", "dl_idx")
    train_map_id, train_map_name, suspect_rows = build_train_mapping(train_json_paths, mapping_key)
    if "audit_suspect_files.csv" in audit_logs and suspect_rows:
        audit_logs["audit_suspect_files.csv"].extend(suspect_rows)

    # Index train images
    train_image_index, train_image_dups = scan_image_files(train_images_dir, recursive=True)

    if not quiet:
        print(f"[INFO] train: {len(train_json_paths)} jsons, {len(train_image_index)} images", flush=True)

    records: list[dict] = []
    train_dedup_keys: set[tuple] = set()
    train_image_size_cache: dict[str, tuple[int, int]] = {}

    # Process train
    t0 = perf_counter()
    for i, p in enumerate(train_json_paths, start=1):
        if not quiet and log_every > 0 and (i == 1 or i % log_every == 0):
            dt = perf_counter() - t0
            print(f"[INFO] train progress: {i}/{len(train_json_paths)} ({dt:.1f}s)", flush=True)
        data, err = parse_one_json(p)
        if err or data is None:
            logs["excluded_rows"].append(
                {"source": "train", "file_name": "", "source_json": str(p), "reason_code": err or "parse_failed", "detail": ""}
            )
            if "audit_missing_labels.csv" in audit_logs:
                audit_logs["audit_missing_labels.csv"].append({"source_json": str(p), "reason": err})
            continue

        record, _ = normalize_record(
            source="train",
            data=data,
            source_json=p,
            image_index=train_image_index,
            image_duplicates=train_image_dups,
            image_size_cache=train_image_size_cache,
            config=config,
            mapping_id=None,
            mapping_name=None,
            logs=logs,
            audit=audit_logs,
        )
        if record is None:
            continue
        records.append(record)

        # build dedup key set for external comparison
        dedup_key_fields = config.get("dedup", {}).get("exact", {}).get("key", [])
        add_train_dedup_key(record, dedup_key_fields, train_dedup_keys)

    # External data
    external_cfg = config.get("external_data", {})
    if external_cfg.get("enabled", False):
        banned_patterns = external_cfg.get("banned_patterns", [])
        sources = external_cfg.get("sources", [])

        ingest_cfg = external_cfg.get("ingest", {})
        normalize_and_copy = bool(ingest_cfg.get("normalize_and_copy", False))
        out_ext_images = resolve_path(ingest_cfg.get("output_images_dir", "data/processed/external_normalized/images"), base_dir)
        out_ext_anns = resolve_path(ingest_cfg.get("output_annotations_dir", "data/processed/external_normalized/annotations"), base_dir)
        overwrite_outputs = bool(ingest_cfg.get("overwrite_outputs", False))

        dedup_cfg = external_cfg.get("dedup", {})
        dedup_against_train = bool(dedup_cfg.get("dedup_against_train", False))

        for src in sources:
            name = src.get("name", "external")
            img_dir = resolve_path(src["images_dir"], base_dir)
            ann_dir = resolve_path(src["annotations_dir"], base_dir)
            recursive = bool(src.get("recursive", True))

            # Safety: banned patterns in paths
            for root in (img_dir, ann_dir):
                for dirpath, _, filenames in os.walk(root):
                    if any(pat.lower() in dirpath.lower() for pat in banned_patterns):
                        raise RuntimeError(f"Banned pattern found in path: {dirpath}")
                    for fn in filenames:
                        full = os.path.join(dirpath, fn)
                        if any(pat.lower() in full.lower() for pat in banned_patterns):
                            raise RuntimeError(f"Banned pattern found in file: {full}")

            ext_image_index, ext_image_dups = scan_image_files(img_dir, recursive=recursive)
            ext_json_paths = scan_json_files(ann_dir, recursive=recursive)

            if not quiet:
                print(f"[INFO] external[{name}]: {len(ext_json_paths)} jsons, {len(ext_image_index)} images", flush=True)

            ext_image_size_cache: dict[str, tuple[int, int]] = {}
            t1 = perf_counter()
            for j, p in enumerate(ext_json_paths, start=1):
                if not quiet and log_every > 0 and (j == 1 or j % log_every == 0):
                    dt = perf_counter() - t1
                    print(f"[INFO] external[{name}] progress: {j}/{len(ext_json_paths)} ({dt:.1f}s)", flush=True)
                data, err = parse_one_json(p)
                if err or data is None:
                    logs["excluded_rows"].append(
                        {
                            "source": "external",
                            "file_name": "",
                            "source_json": str(p),
                            "reason_code": err or "parse_failed",
                            "detail": "",
                        }
                    )
                    if "audit_missing_labels.csv" in audit_logs:
                        audit_logs["audit_missing_labels.csv"].append({"source_json": str(p), "reason": err})
                    continue

                record, normalized_json = normalize_record(
                    source="external",
                    data=data,
                    source_json=p,
                    image_index=ext_image_index,
                    image_duplicates=ext_image_dups,
                    image_size_cache=ext_image_size_cache,
                    config=config,
                    mapping_id=train_map_id,
                    mapping_name=train_map_name,
                    logs=logs,
                    audit=audit_logs,
                    external_cfg=external_cfg,
                )
                if record is None:
                    continue

                # External dedup against train
                dedup_key_fields = dedup_cfg.get("exact", {}).get("key", [])
                keep_record = should_keep_external_record_after_dedup(
                    record,
                    dedup_against_train=dedup_against_train,
                    dedup_key_fields=dedup_key_fields,
                    train_dedup_keys=train_dedup_keys,
                    logs=logs,
                    audit_logs=audit_logs,
                )
                if not keep_record:
                    continue

                records.append(record)

                # Optional: normalize & copy external artifacts
                if normalize_and_copy and normalized_json is not None:
                    try:
                        rel = p.relative_to(ann_dir)
                    except Exception:
                        rel = Path(p.name)
                    out_ann_path = out_ext_anns / name / rel
                    out_img_path = out_ext_images / name / record["file_name"]

                    out_ann_path.parent.mkdir(parents=True, exist_ok=True)
                    out_img_path.parent.mkdir(parents=True, exist_ok=True)

                    if overwrite_outputs or not out_ann_path.exists():
                        with out_ann_path.open("w", encoding="utf-8") as f:
                            json.dump(normalized_json, f, ensure_ascii=False, indent=2)
                    if overwrite_outputs or not out_img_path.exists():
                        src_img = ext_image_index.get(record["file_name"].lower())
                        if src_img is not None:
                            out_img_path.write_bytes(src_img.read_bytes())

    # Dedup exact (global)
    dedup_cfg = config.get("dedup", {}).get("exact", {})
    records = dedup_exact_records(records, dedup_cfg, logs, audit_logs)

    # Add group_id
    split_cfg = config.get("split", {})
    add_group_id(records, split_cfg)

    # mapping table rows
    mapping_rows = []
    for dl_idx, cat_id in train_map_id.items():
        mapping_rows.append(
            {
                "dl_idx": dl_idx,
                "canonical_category_id": cat_id,
                "name": train_map_name.get(dl_idx, ""),
            }
        )

    return {
        "records": records,
        "logs": logs,
        "audit_logs": audit_logs,
        "paths": {"processed_dir": processed_dir, "metadata_dir": metadata_dir},
        "outputs": outputs_cfg,
        "config_path": config_path,
        "repo_root": repo_root,
        "config": config,
        "mapping_rows": mapping_rows,
    }


def run(
    config: dict, *, config_path: Path, repo_root: Path, quiet: bool = False, log_every: int = 500
) -> dict:
    result = build_df_clean(config, config_path, repo_root=repo_root, quiet=quiet, log_every=log_every)

    records = result["records"]
    logs = result["logs"]
    audit_logs = result["audit_logs"]
    mapping_rows = result["mapping_rows"]
    processed_dir = result["paths"]["processed_dir"]
    metadata_dir = result["paths"]["metadata_dir"]
    outputs_cfg = result["outputs"]
    base_dir = result["repo_root"]

    write_outputs(
        records=records,
        logs=logs,
        audit_logs=audit_logs,
        mapping_rows=mapping_rows,
        processed_dir=processed_dir,
        metadata_dir=metadata_dir,
        outputs_cfg=outputs_cfg,
        config=config,
        base_dir=base_dir,
        resolve_path_fn=resolve_path,
    )

    splits = make_group_split(records, config.get("split", {}), config.get("random_seed", 42))
    splits_name = config.get("split", {}).get("splits_name", "splits.csv")
    write_splits(metadata_dir, splits_name, splits)

    write_manifest(
        records=records,
        logs=logs,
        config_path=result["config_path"],
        metadata_dir=metadata_dir,
        outputs_cfg=outputs_cfg,
        repo_root=base_dir,
    )

    return result
