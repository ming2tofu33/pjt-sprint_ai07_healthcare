from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter
from typing import Iterable, Tuple

from src.dataprep.config import resolve_path
from src.dataprep.dedup import (
    add_train_dedup_key,
    dedup_exact_records,
    dedup_iou_records,
    filter_expected4_actual3,
    filter_external_bbox_category_conflicts,
    filter_external_copy_json_records,
    filter_external_images_with_any_bad_bbox,
    filter_object_count,
    should_keep_external_record_after_dedup,
)
from src.dataprep.export import write_outputs
from src.dataprep.io_utils import parse_one_json, scan_image_files, scan_json_files
from src.dataprep.manifest import write_manifest
from src.dataprep.normalize import normalize_record
from src.dataprep.quality_audit import run_aux_detector_audit, run_pixel_overlap_audit
from src.dataprep.split import add_group_id, make_group_split, write_splits


def build_train_mapping(
    train_json_paths: Iterable[Path], mapping_key: str
) -> Tuple[dict[str, int], dict[str, str], list[dict[str, str]]]:
    """
    train annotation??湲곗??쇰줈 dl_idx -> canonical category_id/name 留ㅽ븨??留뚮뱺??
    諛섑솚媛? (id_map, name_map, suspect_rows)
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
    """
    ?먯쿇 JSON/?대?吏瑜??뺢퇋?뷀빐 df_clean ?덉퐫??由ъ뒪?몃? 援ъ텞?쒕떎.
    ???⑥닔??'?섏쭛/?뺢퇋??以묐났?쒓굅/遺꾪븷以鍮?源뚯? ?대떦?섍퀬, ?ㅼ젣 ?뚯씪 ??μ? run()?먯꽌 泥섎━?쒕떎.
    """
    base_dir = repo_root
    paths_cfg = config.get("paths", {})
    outputs_cfg = config.get("outputs", {})

    train_images_dir = resolve_path(paths_cfg["train_images_dir"], base_dir)
    train_ann_dir = resolve_path(paths_cfg["train_annotations_dir"], base_dir)
    processed_dir = resolve_path(paths_cfg["processed_dir"], base_dir)
    metadata_dir = resolve_path(paths_cfg["metadata_dir"], base_dir)

    metadata_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Audit 濡쒓렇??鍮?寃곌낵?쇰룄 ?뚯씪 ?ㅽ궎留덈? ?덉젙?곸쑝濡?留뚮뱾 ???덇쾶 ?ㅻ? 誘몃━ 以鍮꾪븳??
    audit_files = config.get("audit", {}).get("files", [])
    audit_logs = {name: [] for name in audit_files}
    audit_logs.setdefault("audit_unmapped_external.csv", [])

    logs: dict[str, list[dict]] = {
        "excluded_rows": [],
        "fixes_bbox": [],
        "excluded_rows_external": [],
        "fixes_bbox_external": [],
    }

    # ?몃? ?곗씠???뺣젹???꾩슂??train 湲곗? 留ㅽ븨 ?뚯씠釉?援ъ텞
    train_json_paths = scan_json_files(train_ann_dir, recursive=True)
    mapping_key = config.get("external_data", {}).get("category_id_mapping", {}).get("mapping_key", "dl_idx")
    train_map_id, train_map_name, suspect_rows = build_train_mapping(train_json_paths, mapping_key)
    if "audit_suspect_files.csv" in audit_logs and suspect_rows:
        audit_logs["audit_suspect_files.csv"].extend(suspect_rows)

    # ?뚯씪紐?lower) -> ?대?吏 寃쎈줈 ?몃뜳?? 以묐났 ?뚯씪紐?紐⑸줉
    train_image_index, train_image_dups = scan_image_files(train_images_dir, recursive=True)

    if not quiet:
        print(f"[INFO] train: {len(train_json_paths)} jsons, {len(train_image_index)} images", flush=True)

    records: list[dict] = []
    train_dedup_keys: set[tuple] = set()
    train_image_size_cache: dict[str, tuple[int, int]] = {}

    # 1) Train 레코드 수집/정규화
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

        # external dedup 鍮꾧탳???ㅻ? train 湲곗??쇰줈 ?꾩쟻
        dedup_key_fields = config.get("dedup", {}).get("exact", {}).get("key", [])
        add_train_dedup_key(record, dedup_key_fields, train_dedup_keys)

    # 2) External ?덉퐫???섏쭛/?뺢퇋??(?듭뀡)
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

            # 湲덉? ?곗씠?곗뀑(議고빀 ?곗씠?? 寃쎈줈/?뚯씪紐낆쓣 ?ъ쟾 李⑤떒?쒕떎.
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

                # train怨?exact key媛 ?숈씪??external row???뺤콉???곕씪 ?쒖쇅
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

                # ?꾩슂 ???뺢퇋?붾맂 external ?곗텧臾쇱쓣 蹂꾨룄 ?대뜑??????ы쁽/媛먯궗 紐⑹쟻)
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

    # 3) ?꾩껜(train+external) 湲곗? 以묐났 ?쒓굅
    # 2-1) ?몃? copy.json ?⑦꽩 ?덉퐫???쒖쇅
    records = filter_external_copy_json_records(records, config, logs, audit_logs)

    # 2-2) ?몃? ?숈씪 bbox-?ㅼ쨷移댄뀒怨좊━ 異⑸룎 ?쒖쇅
    records = filter_external_bbox_category_conflicts(records, config, logs, audit_logs)

    # 2-3) ?몃? bbox 遺덈웾 ?대젰???덈뒗 ?대?吏???덉퐫???꾩껜 ?쒖쇅
    records = filter_external_images_with_any_bad_bbox(records, config, logs, audit_logs)

    dedup_root_cfg = config.get("dedup", {})
    exact_cfg = dedup_root_cfg.get("exact", {})
    iou_cfg = dedup_root_cfg.get("iou", {})
    records = dedup_exact_records(records, exact_cfg, logs, audit_logs)
    records = dedup_iou_records(records, iou_cfg, logs, audit_logs)

    # 3-1) 媛앹껜 ??洹쒖튃 ?곸슜: ?대?吏??媛앹껜媛 ?덉슜 踰붿쐞(?? 3~4)媛 ?꾨땲硫??쒓굅
    records = filter_object_count(records, config, logs, audit_logs)

    # 3-2) external: ?뚯씪紐??쎌퐫??4媛쒖씤??媛앹껜 3媛쒖씤 ?대?吏 ?쒓굅
    records = filter_expected4_actual3(records, config, logs, audit_logs)

    # 3-3) ?쎌? 湲곕컲 bbox ?덉쭏 媛먯궗(coverage / bbox IoU)
    pixel_audit = run_pixel_overlap_audit(
        records=records,
        config=config,
        logs=logs,
        audit_logs=audit_logs,
        base_dir=base_dir,
        resolve_path_fn=resolve_path,
    )
    records = pixel_audit.filtered_records

    # 3-4) 蹂댁“ detector 湲곕컲 援먯감寃利?max IoU)
    aux_audit = run_aux_detector_audit(
        records=records,
        config=config,
        logs=logs,
        audit_logs=audit_logs,
        base_dir=base_dir,
        resolve_path_fn=resolve_path,
    )
    records = aux_audit.filtered_records

    # 4) 누수 방지 split을 위한 group_id 부여
    split_cfg = config.get("split", {})
    add_group_id(records, split_cfg)

    # 5) 留ㅽ븨 ?뚯씠釉?異쒕젰????議곕┰
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
    """
    ?꾩쿂由??뚯씠?꾨씪???ㅽ뻾 ?⑥닔:
    - build_df_clean(): 硫붾え由???寃곌낵 ?앹꽦
    - write_outputs(): csv/audit ???    - make_group_split()/write_splits(): train/val 遺꾪븷 ???    - write_manifest(): ?ㅽ뻾 ?붿빟 ???    """
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


