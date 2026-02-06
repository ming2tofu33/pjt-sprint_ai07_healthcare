#!/usr/bin/env python3
"""
COCO Format 생성 스크립트

기능:
1. train_annotations/ 아래 763개 JSON → 232개 이미지 단위 통합
2. BBox 클리핑 및 검증 (이미지 경계 밖 제거)
3. Category 매핑 생성 (id2idx, idx2id)
4. Class whitelist 적용 (옵션)
5. Image-level 통계 생성

사용법:
    python scripts/0_create_coco_format.py [--config CONFIG_PATH] [--run-name RUN_NAME]
    
    예시:
    python scripts/0_create_coco_format.py
    python scripts/0_create_coco_format.py --run-name exp_baseline_v1
    python scripts/0_create_coco_format.py --config runs/exp_test/config/config.json
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import json

# src 모듈 import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import (
    setup_project_paths,
    load_config,
    save_config,
    save_json,
    print_section,
)


def clip_bbox_xywh(bbox, W, H):
    """BBox를 이미지 경계로 클리핑"""
    x, y, w, h = map(float, bbox)
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    
    # Clip
    cx1 = max(0.0, min(x1, float(W)))
    cy1 = max(0.0, min(y1, float(H)))
    cx2 = max(0.0, min(x2, float(W)))
    cy2 = max(0.0, min(y2, float(H)))
    
    nw = cx2 - cx1
    nh = cy2 - cy1
    clipped = (cx1 != x1) or (cy1 != y1) or (cx2 != x2) or (cy2 != y2)
    valid = (nw > 0.0) and (nh > 0.0)
    
    return [cx1, cy1, nw, nh], clipped, valid


def to_int_stem(file_name):
    """파일명의 stem을 int로 변환 (실패 시 None)"""
    try:
        return int(Path(file_name).stem)
    except Exception:
        return None


def merge_coco_annotations(
    ann_root: Path,
    use_file_stem_as_image_id: bool = True,
):
    """
    train_annotations/ 아래 JSON들을 이미지 단위로 통합
    
    Returns:
        dict: merged COCO format
        dict: image_id_map (file_name -> image_id)
        dict: category_id_to_name
        dict: stats (clipped, invalid, skipped 등)
    """
    json_files = sorted(ann_root.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"train_annotations 아래에서 json을 찾지 못했습니다: {ann_root}")
    
    # Collectors
    img_meta_by_name = {}
    anns_by_name = defaultdict(list)
    cat_id_to_name = {}
    
    seen_json = 0
    skipped_ann = 0
    clipped_ann = 0
    invalid_after_clip = 0
    
    print(f"[INFO] 총 {len(json_files)}개 JSON 파일 처리 중...")
    
    for fp in json_files:
        seen_json += 1
        if seen_json % 100 == 0:
            print(f"  - 진행: {seen_json}/{len(json_files)}")
        
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        images = data.get("images", [])
        anns = data.get("annotations", [])
        cats = data.get("categories", [])
        
        if not images or not anns or not cats:
            skipped_ann += 1
            continue
        
        im = images[0]
        a = anns[0]
        c = cats[0]
        
        file_name = str(im.get("file_name"))
        W = im.get("width", None)
        H = im.get("height", None)
        
        # Categories map
        cid = c.get("id", None)
        cname = c.get("name", None)
        if cid is not None and cname is not None:
            cat_id_to_name[int(cid)] = str(cname)
        
        # 이미지 메타 (처음 본 것 기준)
        if file_name not in img_meta_by_name:
            img_meta_by_name[file_name] = {
                "file_name": file_name,
                "width": W,
                "height": H,
            }
        
        bbox = a.get("bbox", None)
        category_id = a.get("category_id", None)
        
        if bbox is None or (not isinstance(bbox, list)) or len(bbox) != 4:
            skipped_ann += 1
            continue
        if category_id is None:
            skipped_ann += 1
            continue
        
        # BBox clip
        clipped = False
        valid = True
        new_bbox = list(map(float, bbox))
        if isinstance(W, (int, float)) and isinstance(H, (int, float)):
            new_bbox, clipped, valid = clip_bbox_xywh(bbox, W, H)
            if clipped:
                clipped_ann += 1
            if not valid:
                invalid_after_clip += 1
                continue
        
        anns_by_name[file_name].append({
            "category_id": int(category_id),
            "bbox": new_bbox,
            "iscrowd": int(a.get("iscrowd", 0) or 0),
            "ignore": int(a.get("ignore", 0) or 0),
            "area": float(a.get("area", new_bbox[2] * new_bbox[3])),
            "segmentation": a.get("segmentation", []),
        })
    
    # Build merged COCO
    file_names = sorted(img_meta_by_name.keys())
    image_id_map = {}
    images_out = []
    annotations_out = []
    
    next_img_id = 1
    next_ann_id = 1
    
    for fn in file_names:
        meta = img_meta_by_name[fn]
        W, H = meta.get("width"), meta.get("height")
        
        img_id = None
        if use_file_stem_as_image_id:
            img_id = to_int_stem(fn)
        if img_id is None:
            img_id = next_img_id
            next_img_id += 1
        
        image_id_map[fn] = img_id
        
        images_out.append({
            "id": img_id,
            "file_name": fn,
            "width": W,
            "height": H,
        })
        
        for ann in anns_by_name.get(fn, []):
            x, y, w, h = ann["bbox"]
            annotations_out.append({
                "id": next_ann_id,
                "image_id": img_id,
                "category_id": ann["category_id"],
                "bbox": [x, y, w, h],
                "area": float(ann.get("area", w * h)),
                "iscrowd": int(ann.get("iscrowd", 0)),
                "ignore": int(ann.get("ignore", 0)),
                "segmentation": ann.get("segmentation", []),
            })
            next_ann_id += 1
    
    # Categories
    categories_out = []
    for cid in sorted(cat_id_to_name.keys()):
        categories_out.append({
            "id": int(cid),
            "name": cat_id_to_name[cid],
            "supercategory": "pill",
        })
    
    merged = {
        "info": {
            "description": "AI07 pill OD - merged coco (train)",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": str(ann_root.resolve()),
        },
        "images": images_out,
        "annotations": annotations_out,
        "categories": categories_out,
    }
    
    stats = {
        "n_json_files": seen_json,
        "n_unique_images": len(images_out),
        "n_annotations": len(annotations_out),
        "n_categories": len(categories_out),
        "skipped_ann": skipped_ann,
        "clipped_ann": clipped_ann,
        "invalid_after_clip": invalid_after_clip,
    }
    
    return merged, image_id_map, cat_id_to_name, stats


def build_label_mapping(cat_id_to_name, class_whitelist=None):
    """
    Category ID -> Contiguous Index 매핑 생성
    
    Args:
        cat_id_to_name: {category_id: name} dict
        class_whitelist: None (전체) 또는 [id1, id2, ...] (부분)
    
    Returns:
        dict: label_map_full (전체 매핑)
        dict: label_map_whitelist (whitelist 매핑, whitelist 있을 때만)
        list: train_only_ids (whitelist에 없는 id, whitelist 있을 때만)
    """
    cat_ids_sorted = sorted(cat_id_to_name.keys())
    
    def _build_map(category_ids):
        id2idx = {cid: i for i, cid in enumerate(category_ids)}
        idx2id = {i: cid for cid, i in id2idx.items()}
        names = [cat_id_to_name[cid] for cid in category_ids]
        return {
            "category_ids": category_ids,
            "id2idx": id2idx,
            "idx2id": idx2id,
            "names": names,
            "num_classes": len(category_ids),
        }
    
    label_map_full = _build_map(cat_ids_sorted)
    
    label_map_whitelist = None
    train_only_ids = []
    
    if class_whitelist:
        # whitelist에 있지만 train에 없는 id 제외
        whitelist_in_train = [cid for cid in class_whitelist if cid in cat_id_to_name]
        label_map_whitelist = _build_map(whitelist_in_train)
        train_only_ids = [cid for cid in cat_ids_sorted if cid not in set(whitelist_in_train)]
    
    return label_map_full, label_map_whitelist, train_only_ids


def compute_statistics(merged_coco):
    """통계 계산 (객체 수 분포 등)"""
    anns = merged_coco["annotations"]
    
    # 이미지당 객체 수
    obj_per_img = Counter()
    for a in anns:
        obj_per_img[a["image_id"]] += 1
    
    dist = Counter(obj_per_img.values())
    max_objs = max(obj_per_img.values()) if obj_per_img else 0
    gt4 = sum(1 for v in obj_per_img.values() if v > 4)
    
    # 클래스 분포
    cat_counts = Counter(a["category_id"] for a in anns)
    
    return {
        "objects_per_image_dist": dict(dist),
        "max_objects_per_image": max_objs,
        "n_images_gt4": gt4,
        "category_counts": dict(cat_counts),
    }


def main():
    parser = argparse.ArgumentParser(description="COCO Format 생성")
    parser.add_argument("--config", type=str, help="Config 파일 경로 (선택)")
    parser.add_argument("--run-name", type=str, help="실험명 (선택, 자동 생성)")
    args = parser.parse_args()
    
    print_section("Stage 0: COCO Format 생성")
    
    # 1) 경로 설정
    print("\n[1] 경로 설정...")
    paths = setup_project_paths(
        run_name=args.run_name,
        root=Path(__file__).parent.parent,
        create_dirs=True,
        check_input_exists=True,
    )
    print(f"  ✅ RUN_NAME: {paths['RUN_NAME']}")
    print(f"  ✅ CACHE: {paths['CACHE']}")
    
    # 2) Config 로드 또는 생성
    print("\n[2] Config 로드...")
    if args.config:
        config = load_config(Path(args.config))
        print(f"  ✅ Config 로드: {args.config}")
    else:
        config_path = paths["CONFIG"] / "config.json"
        if config_path.exists():
            config = load_config(config_path)
            print(f"  ✅ 기존 Config 사용: {config_path}")
        else:
            from utils import get_default_config
            config = get_default_config(paths["RUN_NAME"], paths)
            save_config(config, config_path)
            print(f"  ✅ 기본 Config 생성: {config_path}")
    
    # 3) COCO 병합
    print("\n[3] COCO 병합 중...")
    ann_root = paths["TRAIN_ANN_DIR"]
    
    merged, image_id_map, cat_id_to_name, stats = merge_coco_annotations(
        ann_root=ann_root,
        use_file_stem_as_image_id=True,
    )
    
    print(f"  ✅ JSON 파일: {stats['n_json_files']}")
    print(f"  ✅ 이미지: {stats['n_unique_images']}")
    print(f"  ✅ Annotations: {stats['n_annotations']}")
    print(f"  ✅ Categories: {stats['n_categories']}")
    print(f"  ⚠️  Clipped BBox: {stats['clipped_ann']}")
    print(f"  ⚠️  Invalid (drop): {stats['invalid_after_clip']}")
    
    # 4) Label 매핑
    print("\n[4] Label 매핑 생성...")
    class_whitelist = config["data"].get("class_whitelist", None)
    
    label_map_full, label_map_whitelist, train_only_ids = build_label_mapping(
        cat_id_to_name,
        class_whitelist,
    )
    
    print(f"  ✅ Full classes: {label_map_full['num_classes']}")
    if label_map_whitelist:
        print(f"  ✅ Whitelist classes: {label_map_whitelist['num_classes']}")
        print(f"  ⚠️  Train-only classes: {len(train_only_ids)}")
    else:
        print(f"  ℹ️  Whitelist 없음 (전체 클래스 사용)")
    
    # 5) 통계 계산
    print("\n[5] 통계 계산...")
    statistics = compute_statistics(merged)
    print(f"  ✅ 객체 수 분포: {statistics['objects_per_image_dist']}")
    print(f"  ✅ Max objects/image: {statistics['max_objects_per_image']}")
    print(f"  ⚠️  GT4 images: {statistics['n_images_gt4']}")
    
    # 6) 저장
    print("\n[6] 파일 저장...")
    cache_dir = paths["CACHE"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Merged COCO
    out_coco = cache_dir / "train_merged_coco.json"
    save_json(out_coco, merged)
    print(f"  ✅ {out_coco.relative_to(paths['ROOT'])}")
    
    # Image ID map
    out_map = cache_dir / "image_id_map.json"
    save_json(out_map, image_id_map)
    print(f"  ✅ {out_map.relative_to(paths['ROOT'])}")
    
    # Category ID to name
    out_cat = cache_dir / "category_id_to_name.json"
    save_json(out_cat, cat_id_to_name)
    print(f"  ✅ {out_cat.relative_to(paths['ROOT'])}")
    
    # Label map (full)
    out_label_full = cache_dir / "label_map_full.json"
    save_json(out_label_full, label_map_full)
    print(f"  ✅ {out_label_full.relative_to(paths['ROOT'])}")
    
    # Label map (whitelist, if exists)
    if label_map_whitelist:
        out_label_wl = cache_dir / "label_map_whitelist.json"
        save_json(out_label_wl, {
            **label_map_whitelist,
            "whitelist_source": "config",
        })
        print(f"  ✅ {out_label_wl.relative_to(paths['ROOT'])}")
        
        # Train-only IDs
        out_train_only = paths["REPORTS"] / "train_only_category_ids.json"
        save_json(out_train_only, {
            "train_only_category_ids": train_only_ids,
            "n_train_only": len(train_only_ids),
        })
        print(f"  ✅ {out_train_only.relative_to(paths['ROOT'])}")
    
    # Statistics
    out_stats = paths["REPORTS"] / "coco_merge_stats.json"
    save_json(out_stats, {
        "merge_stats": stats,
        "statistics": statistics,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })
    print(f"  ✅ {out_stats.relative_to(paths['ROOT'])}")
    
    # 7) Config 업데이트
    print("\n[7] Config 업데이트...")
    config["data"]["num_classes"] = label_map_full["num_classes"]
    if not config["data"].get("class_whitelist"):
        config["data"]["class_whitelist"] = None
    
    config_path = paths["CONFIG"] / "config.json"
    save_config(config, config_path)
    print(f"  ✅ {config_path.relative_to(paths['ROOT'])}")
    
    print_section("✅ COCO Format 생성 완료")
    print(f"\n다음 단계:")
    print(f"  python scripts/1_splitting.py --run-name {paths['RUN_NAME']}")


if __name__ == "__main__":
    main()
