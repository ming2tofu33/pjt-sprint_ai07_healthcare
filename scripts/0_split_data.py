"""STAGE 0: 데이터 정제 + 분할 + COCO 재조립 + label_map 생성.

사용법::

    python scripts/0_split_data.py --run-name exp_20260209_120000 \\
        --config configs/experiments/baseline.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# ── 프로젝트 루트를 sys.path 에 추가 ────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config_loader import load_experiment_config, load_preprocess_config
from src.utils.logger import get_logger
from src.dataprep.output.data_pipeline import run as run_pipeline

logger = get_logger(__name__)


def _read_exclude_file_names(path: Path) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        file_name = Path(text).name
        key = file_name.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(file_name)
    return names


def _apply_manual_exclude_file_from_config(config: dict, repo_root: Path) -> None:
    manual_cfg = config.setdefault("manual_overrides", {})
    src = manual_cfg.get("exclude_file_names_file")
    if not isinstance(src, str) or not src.strip():
        return

    src_path = Path(src)
    if not src_path.is_absolute():
        src_path = (repo_root / src_path).resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"exclude_file_names_file not found: {src_path}")

    from_file = _read_exclude_file_names(src_path)
    existing = manual_cfg.get("exclude_file_names", [])
    if not isinstance(existing, list):
        existing = []

    merged: list[str] = []
    seen: set[str] = set()
    for file_name in [*existing, *from_file]:
        if not isinstance(file_name, str):
            continue
        text = file_name.strip()
        if not text:
            continue
        key = Path(text).name.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(Path(text).name)

    manual_cfg["exclude_file_names"] = merged
    logger.info(
        "manual exclude list loaded | file=%s | from_file=%d | merged=%d",
        src_path,
        len(from_file),
        len(merged),
    )


# ─────────────────────────────────────────────
#  COCO 재조립
# ─────────────────────────────────────────────

def _build_coco_json(
    records: list[dict],
    *,
    train_ann_dir: Path,
    run_name: str,
) -> dict:
    """df_clean records -> COCO Object Detection JSON.

    Returns
    -------
    dict
        ``{info, images, annotations, categories}`` 구조.
    """
    # 1) 이미지 고유 목록 (file_name 기준 정렬)
    file_names = sorted({r["file_name"] for r in records})
    fname_to_id: dict[str, int] = {}
    images: list[dict] = []
    for idx, fn in enumerate(file_names, start=1):
        fname_to_id[fn] = idx
        # records 중 동일 file_name 첫 번째에서 width/height 추출
        sample = next(r for r in records if r["file_name"] == fn)
        images.append({
            "id": idx,
            "file_name": fn,
            "width": int(sample.get("width", 0)),
            "height": int(sample.get("height", 0)),
        })

    # 2) annotation 목록
    annotations: list[dict] = []
    ann_id = 0
    for r in records:
        ann_id += 1
        cat_id = int(r["category_id"])
        bx = float(r["bbox_x"])
        by = float(r["bbox_y"])
        bw = float(r["bbox_w"])
        bh = float(r["bbox_h"])
        annotations.append({
            "id": ann_id,
            "image_id": fname_to_id[r["file_name"]],
            "category_id": cat_id,
            "bbox": [bx, by, bw, bh],
            "area": round(bw * bh, 1),
            "iscrowd": 0,
            "ignore": 0,
            "segmentation": [],
        })

    # 3) category 목록 (고유 category_id 정렬)
    cat_id_to_name: dict[int, str] = {}
    for r in records:
        cid = int(r["category_id"])
        # category_name 이 있으면 사용, 없으면 빈 문자열
        cname = r.get("category_name", "")
        if cid not in cat_id_to_name:
            cat_id_to_name[cid] = cname

    categories: list[dict] = []
    for cid in sorted(cat_id_to_name.keys()):
        categories.append({
            "id": cid,
            "name": cat_id_to_name[cid],
            "supercategory": "pill",
        })

    info = {
        "description": "AI07 pill OD - merged coco (train)",
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "source": str(train_ann_dir),
    }

    return {
        "info": info,
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


# ─────────────────────────────────────────────
#  label_map 생성
# ─────────────────────────────────────────────

def _build_label_map(coco: dict) -> dict:
    """COCO JSON 의 categories 로부터 label_map_full 을 구축한다.

    Returns
    -------
    dict
        ``{category_ids, id2idx, idx2id, names, num_classes}``
    """
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    category_ids = [c["id"] for c in cats]
    names = [c.get("name", "") for c in cats]

    id2idx: dict[str, int] = {}
    idx2id: dict[str, int] = {}
    for idx, cid in enumerate(category_ids):
        id2idx[str(cid)] = idx
        idx2id[str(idx)] = cid

    return {
        "category_ids": category_ids,
        "id2idx": id2idx,
        "idx2id": idx2id,
        "names": names,
        "num_classes": len(category_ids),
    }


def _build_category_id_to_name(coco: dict) -> dict[str, str]:
    """category_id (str) -> name 매핑."""
    return {str(c["id"]): c.get("name", "") for c in coco["categories"]}


def _build_image_id_map(coco: dict) -> dict[str, int]:
    """file_name -> image_id 매핑."""
    return {img["file_name"]: img["id"] for img in coco["images"]}


# ─────────────────────────────────────────────
#  split_train_valid.json 생성
# ─────────────────────────────────────────────

def _build_split_json(
    coco: dict,
    records: list[dict],
    splits_rows: list[dict],
    *,
    seed: int,
    train_ratio: float,
    run_name: str,
) -> dict:
    """group 기반 split 결과를 image_id 리스트로 변환.

    Parameters
    ----------
    coco : dict
        COCO JSON (images 목록 필요).
    records : list[dict]
        df_clean records (file_name, group_id 필요).
    splits_rows : list[dict]
        ``make_group_split()`` 결과 (group_id, split).
    """
    fname_to_imgid = {img["file_name"]: img["id"] for img in coco["images"]}
    group_to_split = {row["group_id"]: row["split"] for row in splits_rows}

    # 이미지당 annotation 수 집계 (stratify 확인용)
    img_obj_count = Counter(r["file_name"] for r in records)

    train_ids: list[int] = []
    valid_ids: list[int] = []

    for fn, imgid in fname_to_imgid.items():
        # record 에서 file_name 에 해당하는 group_id 추출
        rec = next((r for r in records if r["file_name"] == fn), None)
        if rec is None:
            continue
        gid = rec.get("group_id", fn)
        split = group_to_split.get(gid, "train")
        if split == "train":
            train_ids.append(imgid)
        else:
            valid_ids.append(imgid)

    # n_objects 기준 분포 집계
    def _obj_dist(ids: list[int]) -> dict[int, int]:
        id_to_fn = {img["id"]: img["file_name"] for img in coco["images"]}
        counts = [img_obj_count.get(id_to_fn.get(i, ""), 0) for i in ids]
        return dict(sorted(Counter(counts).items()))

    total = len(train_ids) + len(valid_ids)
    return {
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "run_name": run_name,
        "seed": seed,
        "train_ratio": train_ratio,
        "valid_ratio": round(1.0 - train_ratio, 4),
        "stratify_mode": "n_objects",
        "fallback_used": False,
        "counts": {
            "total": total,
            "train": len(train_ids),
            "valid": len(valid_ids),
        },
        "train_image_ids": train_ids,
        "valid_image_ids": valid_ids,
    }


# ─────────────────────────────────────────────
#  메인
# ─────────────────────────────────────────────

def _try_enrich_category_names(records: list[dict], coco_categories: list[dict]) -> None:
    """records 에 category_name 이 누락된 경우 mapping_rows/categories 에서 보충."""
    id_to_name = {c["id"]: c.get("name", "") for c in coco_categories}
    for r in records:
        if not r.get("category_name"):
            r["category_name"] = id_to_name.get(int(r["category_id"]), "")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="STAGE 0: 데이터 정제 + 분할")
    parser.add_argument("--run-name", required=True, help="실험 이름 (예: exp_20260209_120000)")
    parser.add_argument("--config", required=True, help="실험 config YAML 경로")
    parser.add_argument("--verbose", action="store_true", help="상세 로그 출력")
    args = parser.parse_args(argv)

    run_name: str = args.run_name
    config_path = Path(args.config).resolve()
    script_path = Path(__file__).resolve()

    logger.info("STAGE 0 시작 | run_name=%s | config=%s", run_name, config_path)

    # ── 1) config 로드 ──────────────────────────────────────
    # experiment config 로드 (base + experiment merge)
    exp_config, repo_root = load_experiment_config(config_path, script_path)

    # data_pipeline.run() 은 preprocess 설정 형식을 기대하므로
    # experiment config 그대로 전달한다 (paths, dedup 등 공통 키가 동일).
    config = exp_config
    try:
        _apply_manual_exclude_file_from_config(config, repo_root)
    except FileNotFoundError as exc:
        logger.error("manual exclude file load failed: %s", exc)
        sys.exit(1)

    # ── 2) 출력 디렉터리 준비 ────────────────────────────────
    paths_cfg = config.get("paths", {})
    processed_base = Path(paths_cfg.get("processed_dir", "data/processed/cache"))
    if not processed_base.is_absolute():
        processed_base = (repo_root / processed_base).resolve()

    cache_dir = processed_base / run_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = cache_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    metadata_dir_str = paths_cfg.get("metadata_dir", "data/metadata")
    metadata_dir = Path(metadata_dir_str)
    if not metadata_dir.is_absolute():
        metadata_dir = (repo_root / metadata_dir).resolve()
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # ── 3) data_pipeline.run() 실행 (정제/필터/split) ────────
    logger.info("data_pipeline.run() 시작 ...")

    # data_pipeline.run() 은 config["paths"]["processed_dir"] / config["paths"]["metadata_dir"]
    # 로 직접 출력한다. cache/<run_name> 하위로 저장되게 경로를 임시 패치한다.
    patched_config = _patch_output_paths(config, cache_dir, metadata_dir)

    pipeline_result = run_pipeline(
        patched_config,
        config_path=config_path,
        repo_root=repo_root,
        quiet=not args.verbose,
    )

    records: list[dict] = pipeline_result["records"]
    mapping_rows: list[dict] = pipeline_result["mapping_rows"]
    logger.info("정제 완료 | records=%d | mapping_rows=%d", len(records), len(mapping_rows))

    # ── 4) mapping_rows 에서 category name 보충 ─────────────
    id_to_name_from_mapping = {
        int(row["canonical_category_id"]): row.get("name", "")
        for row in mapping_rows
        if row.get("canonical_category_id")
    }
    for r in records:
        if not r.get("category_name"):
            r["category_name"] = id_to_name_from_mapping.get(int(r["category_id"]), "")

    # ── 5) train_merged_coco.json ────────────────────────────
    train_ann_dir_str = paths_cfg.get("train_annotations_dir", "data/raw/train_annotations")
    train_ann_dir = Path(train_ann_dir_str)
    if not train_ann_dir.is_absolute():
        train_ann_dir = (repo_root / train_ann_dir).resolve()

    coco = _build_coco_json(records, train_ann_dir=train_ann_dir, run_name=run_name)
    coco_path = cache_dir / "train_merged_coco.json"
    with coco_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    logger.info("train_merged_coco.json 저장 | images=%d | annotations=%d | categories=%d",
                len(coco["images"]), len(coco["annotations"]), len(coco["categories"]))

    # ── 6) label_map_full.json ───────────────────────────────
    label_map = _build_label_map(coco)
    lm_path = cache_dir / "label_map_full.json"
    with lm_path.open("w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    logger.info("label_map_full.json 저장 | num_classes=%d", label_map["num_classes"])

    # ── 7) category_id_to_name.json ──────────────────────────
    cat_name_map = _build_category_id_to_name(coco)
    cn_path = cache_dir / "category_id_to_name.json"
    with cn_path.open("w", encoding="utf-8") as f:
        json.dump(cat_name_map, f, ensure_ascii=False, indent=2)

    # ── 8) image_id_map.json ─────────────────────────────────
    img_id_map = _build_image_id_map(coco)
    im_path = cache_dir / "image_id_map.json"
    with im_path.open("w", encoding="utf-8") as f:
        json.dump(img_id_map, f, ensure_ascii=False, indent=2)

    # ── 9) split_train_valid.json + train_ids.txt / valid_ids.txt ─
    seed = config.get("random_seed", 42)
    train_ratio = config.get("split", {}).get("train_ratio", 0.8)

    # data_pipeline.run() 이 이미 splits 를 metadata_dir 에 저장했으므로
    # 그 결과를 읽어 split_train_valid.json 으로 재구성한다.
    splits_csv_name = config.get("split", {}).get("splits_name", "splits.csv")
    splits_csv_path = metadata_dir / splits_csv_name

    splits_rows: list[dict] = []
    if splits_csv_path.exists():
        import csv
        with splits_csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            splits_rows = list(reader)

    split_json = _build_split_json(
        coco, records, splits_rows,
        seed=seed, train_ratio=train_ratio, run_name=run_name,
    )

    stv_path = splits_dir / "split_train_valid.json"
    with stv_path.open("w", encoding="utf-8") as f:
        json.dump(split_json, f, ensure_ascii=False, indent=2)

    # train_ids.txt / valid_ids.txt
    with (splits_dir / "train_ids.txt").open("w", encoding="utf-8") as f:
        for tid in split_json["train_image_ids"]:
            f.write(f"{tid}\n")
    with (splits_dir / "valid_ids.txt").open("w", encoding="utf-8") as f:
        for vid in split_json["valid_image_ids"]:
            f.write(f"{vid}\n")

    logger.info("split 저장 | train=%d | valid=%d",
                split_json["counts"]["train"], split_json["counts"]["valid"])

    # ── 10) 요약 출력 ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 0 완료")
    logger.info("  run_name     : %s", run_name)
    logger.info("  cache_dir    : %s", cache_dir)
    logger.info("  images       : %d", len(coco["images"]))
    logger.info("  annotations  : %d", len(coco["annotations"]))
    logger.info("  categories   : %d (num_classes=%d)", len(coco["categories"]), label_map["num_classes"])
    logger.info("  train/valid  : %d / %d", split_json["counts"]["train"], split_json["counts"]["valid"])
    logger.info("=" * 60)


def _patch_output_paths(config: dict, cache_dir: Path, metadata_dir: Path) -> dict:
    """data_pipeline.run() 의 산출물이 cache/<run_name>/ 하위로 저장되도록 경로를 패치한다.

    원본 config 은 변경하지 않고 복사본을 반환한다.
    """
    from copy import deepcopy
    patched = deepcopy(config)
    patched.setdefault("paths", {})
    patched["paths"]["processed_dir"] = str(cache_dir)
    patched["paths"]["metadata_dir"] = str(metadata_dir)
    return patched


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[INT] interrupted by user (Ctrl+C).", file=sys.stderr)
        raise SystemExit(130)
