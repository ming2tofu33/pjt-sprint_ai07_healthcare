#!/usr/bin/env python3
"""
Train/Val 데이터 분할 스크립트

기능:
1. Stratified split (객체 수 기반)
2. K-Fold 지원 (옵션)
3. Split 품질 검증 (분포 균등성)
4. Train/Val ID 리스트 저장

사용법:
    python scripts/1_splitting.py [--run-name RUN_NAME] [--config CONFIG_PATH]
    
    예시:
    python scripts/1_splitting.py
    python scripts/1_splitting.py --run-name exp_baseline_v1
    python scripts/1_splitting.py --config runs/exp_test/config/config.json --kfold
"""

import sys
import argparse
import random
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


def build_image_table_from_coco(merged_coco_path):
    """
    Merged COCO에서 image-level 테이블 생성
    
    Returns:
        list: 각 이미지의 메타 정보 [{image_id, n_objects, category_ids, ...}, ...]
    """
    with open(merged_coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    
    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    
    # image_id -> annotations
    anns_by_img = defaultdict(list)
    for a in anns:
        if not isinstance(a, dict):
            continue
        iid = a.get("image_id", None)
        if iid is None:
            continue
        anns_by_img[int(iid)].append(a)
    
    # Build image table
    rows = []
    for im in images:
        if not isinstance(im, dict):
            continue
        iid = im.get("id", None)
        if iid is None:
            continue
        
        iid = int(iid)
        alist = anns_by_img.get(iid, [])
        
        cat_ids = [int(a["category_id"]) for a in alist if a.get("category_id") is not None]
        uniq_cat_ids = sorted(set(cat_ids))
        n_obj = len(alist)
        n_labels = len(uniq_cat_ids)
        
        # Label signature (for stratification)
        sig = ",".join(map(str, uniq_cat_ids))
        
        rows.append({
            "image_id": iid,
            "file_name": im.get("file_name"),
            "width": im.get("width"),
            "height": im.get("height"),
            "n_objects": n_obj,
            "n_unique_labels": n_labels,
            "category_ids": uniq_cat_ids,
            "label_signature": sig,
        })
    
    return rows


def stratified_split(
    rows,
    train_ratio=0.8,
    seed=42,
    stratify_mode="n_objects",
    min_per_stratum=2,
):
    """
    Stratified split
    
    Args:
        rows: Image table (list of dicts)
        train_ratio: Train 비율 (0~1)
        seed: Random seed
        stratify_mode: "n_objects" (객체 수) / "signature" (멀티라벨) / "hybrid"
        min_per_stratum: Hybrid 모드에서 strata 최소 샘플 수 (이보다 작으면 fallback)
    
    Returns:
        list: train_ids
        list: valid_ids
        str: actual_stratify_mode (fallback 여부 반영)
        bool: fallback_used
    """
    def make_key(r):
        nobj = int(r["n_objects"])
        sig = r.get("label_signature", "")
        if stratify_mode == "n_objects":
            return f"nobj={nobj}"
        if stratify_mode == "signature":
            return f"sig={sig}"
        # hybrid
        return f"nobj={nobj}|sig={sig}"
    
    keys = [make_key(r) for r in rows]
    key_counts = Counter(keys)
    
    # Fallback 판단 (hybrid가 너무 잘게 나뉘면 n_objects로)
    fallback_used = False
    actual_mode = stratify_mode
    
    if stratify_mode == "hybrid":
        n_singletons = sum(1 for k, c in key_counts.items() if c < min_per_stratum)
        if n_singletons > 0:
            actual_mode = "n_objects"
            fallback_used = True
            keys = [make_key(r) for r in rows]
            key_counts = Counter(keys)
    
    # Group by key
    rng = random.Random(seed)
    by_key = defaultdict(list)
    for r, k in zip(rows, keys):
        by_key[k].append(r)
    
    # Shuffle within each stratum
    for k in by_key:
        rng.shuffle(by_key[k])
    
    train_ids = []
    valid_ids = []
    
    for k, items in by_key.items():
        n = len(items)
        n_train = int(round(n * train_ratio))
        
        # 최소 1개는 각 셋에 배정 (가능하면)
        if n >= 2:
            n_train = min(max(1, n_train), n - 1)
        else:
            n_train = n  # 1개면 train으로
        
        train_part = items[:n_train]
        valid_part = items[n_train:]
        
        train_ids.extend([int(x["image_id"]) for x in train_part])
        valid_ids.extend([int(x["image_id"]) for x in valid_part])
    
    # Global shuffle
    rng.shuffle(train_ids)
    rng.shuffle(valid_ids)
    
    return train_ids, valid_ids, actual_mode, fallback_used


def compute_distribution(image_ids, image_table):
    """분포 계산 (검증용)"""
    id_to_row = {int(r["image_id"]): r for r in image_table}
    
    dist_obj = Counter()
    dist_labels = Counter()
    
    for iid in image_ids:
        r = id_to_row.get(int(iid))
        if r is None:
            continue
        dist_obj[int(r["n_objects"])] += 1
        dist_labels[int(r["n_unique_labels"])] += 1
    
    return {
        "n_objects": dict(dist_obj),
        "n_unique_labels": dict(dist_labels),
    }


def main():
    parser = argparse.ArgumentParser(description="Train/Val 데이터 분할")
    parser.add_argument("--config", type=str, help="Config 파일 경로 (선택)")
    parser.add_argument("--run-name", type=str, help="실험명 (선택)")
    parser.add_argument("--kfold", action="store_true", help="K-Fold 모드 활성화")
    parser.add_argument("--fold-idx", type=int, default=0, help="Fold 인덱스 (K-Fold 모드 시)")
    args = parser.parse_args()
    
    print_section("Stage 1: Train/Val Split")
    
    # 1) 경로 설정
    print("\n[1] 경로 설정...")
    paths = setup_project_paths(
        run_name=args.run_name,
        root=Path(__file__).parent.parent,
        create_dirs=True,
        check_input_exists=False,  # COCO 파일만 있으면 됨
    )
    print(f"  ✅ RUN_NAME: {paths['RUN_NAME']}")
    print(f"  ✅ CACHE: {paths['CACHE']}")
    
    # 2) Config 로드
    print("\n[2] Config 로드...")
    config_path = paths["CONFIG"] / "config.json"
    if args.config:
        config = load_config(Path(args.config))
        print(f"  ✅ Config 로드: {args.config}")
    elif config_path.exists():
        config = load_config(config_path)
        print(f"  ✅ 기존 Config 사용: {config_path}")
    else:
        from utils import get_default_config
        config = get_default_config(paths["RUN_NAME"], paths)
        save_config(config, config_path)
        print(f"  ✅ 기본 Config 생성: {config_path}")
    
    # 3) Merged COCO 로드
    print("\n[3] Merged COCO 로드...")
    merged_coco_path = paths["CACHE"] / "train_merged_coco.json"
    if not merged_coco_path.exists():
        print(f"  ❌ train_merged_coco.json이 없습니다: {merged_coco_path}")
        print(f"  ℹ️  먼저 scripts/0_create_coco_format.py를 실행하세요.")
        sys.exit(1)
    
    print(f"  ✅ {merged_coco_path.relative_to(paths['ROOT'])}")
    
    # 4) Image table 생성
    print("\n[4] Image table 생성...")
    image_table = build_image_table_from_coco(merged_coco_path)
    print(f"  ✅ 총 이미지: {len(image_table)}")
    
    # 5) Split 설정
    print("\n[5] Split 설정...")
    split_config = config.get("split", {})
    seed = split_config.get("seed", 42)
    ratios = split_config.get("ratios", {"train": 0.8, "valid": 0.2})
    train_ratio = float(ratios.get("train", 0.8))
    valid_ratio = float(ratios.get("valid", 0.2))
    
    # 합이 1이 아니면 보정
    s = train_ratio + valid_ratio
    if abs(s - 1.0) > 1e-6:
        valid_ratio = 1.0 - train_ratio
    
    stratify_mode = split_config.get("strategy", "stratify_by_num_objects")
    # 표준 이름 매핑
    if "num_objects" in stratify_mode.lower():
        stratify_mode = "n_objects"
    elif "signature" in stratify_mode.lower():
        stratify_mode = "signature"
    else:
        stratify_mode = "n_objects"  # 기본값
    
    print(f"  ✅ Seed: {seed}")
    print(f"  ✅ Ratios: train={train_ratio:.2f}, valid={valid_ratio:.2f}")
    print(f"  ✅ Stratify mode: {stratify_mode}")
    
    # K-Fold 모드
    kfold_enabled = args.kfold or split_config.get("kfold", {}).get("enabled", False)
    if kfold_enabled:
        print(f"  ⚠️  K-Fold 모드는 현재 미구현 (TODO)")
        print(f"  ℹ️  단일 split으로 진행합니다.")
    
    # 6) Split 실행
    print("\n[6] Split 실행...")
    train_ids, valid_ids, actual_mode, fallback_used = stratified_split(
        rows=image_table,
        train_ratio=train_ratio,
        seed=seed,
        stratify_mode=stratify_mode,
        min_per_stratum=2,
    )
    
    print(f"  ✅ Train: {len(train_ids)}")
    print(f"  ✅ Valid: {len(valid_ids)}")
    if fallback_used:
        print(f"  ⚠️  Fallback used: {stratify_mode} → {actual_mode}")
    
    # 7) 분포 검증
    print("\n[7] 분포 검증...")
    train_dist = compute_distribution(train_ids, image_table)
    valid_dist = compute_distribution(valid_ids, image_table)
    
    print(f"  Train n_objects: {train_dist['n_objects']}")
    print(f"  Valid n_objects: {valid_dist['n_objects']}")
    
    # 8) 저장
    print("\n[8] 파일 저장...")
    split_dir = paths["CACHE"] / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    
    split_obj = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": paths["RUN_NAME"],
        "seed": seed,
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "stratify_mode": actual_mode,
        "fallback_used": fallback_used,
        "counts": {
            "total": len(image_table),
            "train": len(train_ids),
            "valid": len(valid_ids),
        },
        "train_image_ids": train_ids,
        "valid_image_ids": valid_ids,
        "distribution": {
            "train": train_dist,
            "valid": valid_dist,
        },
    }
    
    # JSON
    out_split_json = split_dir / "split_train_valid.json"
    save_json(out_split_json, split_obj)
    print(f"  ✅ {out_split_json.relative_to(paths['ROOT'])}")
    
    # TXT (간편용)
    out_train_txt = split_dir / "train_ids.txt"
    out_valid_txt = split_dir / "valid_ids.txt"
    out_train_txt.write_text("\n".join(map(str, train_ids)) + "\n", encoding="utf-8")
    out_valid_txt.write_text("\n".join(map(str, valid_ids)) + "\n", encoding="utf-8")
    print(f"  ✅ {out_train_txt.relative_to(paths['ROOT'])}")
    print(f"  ✅ {out_valid_txt.relative_to(paths['ROOT'])}")
    
    # 9) Config 업데이트
    print("\n[9] Config 업데이트...")
    config["split"]["seed"] = seed
    config["split"]["ratios"] = {"train": train_ratio, "valid": valid_ratio}
    config["split"]["strategy"] = actual_mode
    save_config(config, config_path)
    print(f"  ✅ {config_path.relative_to(paths['ROOT'])}")
    
    print_section("✅ Split 완료")
    print(f"\n다음 단계:")
    print(f"  python scripts/3_train.py --run-name {paths['RUN_NAME']}")


if __name__ == "__main__":
    main()
