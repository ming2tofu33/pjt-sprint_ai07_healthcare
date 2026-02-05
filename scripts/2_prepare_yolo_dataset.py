#!/usr/bin/env python3
"""
YOLO Dataset 준비 스크립트

기능:
1. COCO Format → YOLO Format 변환
2. Train/Val 이미지 및 라벨 복사/심볼릭링크
3. data.yaml 생성 (Ultralytics 학습용)
4. Label 포맷 검증

사용법:
    python scripts/2_prepare_yolo_dataset.py [--run-name RUN_NAME] [--symlink]
    
    예시:
    python scripts/2_prepare_yolo_dataset.py
    python scripts/2_prepare_yolo_dataset.py --run-name exp_baseline_v1
    python scripts/2_prepare_yolo_dataset.py --run-name exp_v1 --symlink
"""

import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json

# src 모듈 import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import (
    setup_project_paths,
    load_config,
    save_json,
    print_section,
)


def xywh_to_yolo(bbox_xywh, W, H):
    """COCO bbox [x,y,w,h] → YOLO [xc,yc,w,h] (normalized)"""
    x, y, w, h = map(float, bbox_xywh)
    
    # Center
    xc = (x + w / 2.0) / float(W)
    yc = (y + h / 2.0) / float(H)
    ww = w / float(W)
    hh = h / float(H)
    
    # Clamp to [0, 1]
    def clamp01(v):
        return max(0.0, min(1.0, v))
    
    return clamp01(xc), clamp01(yc), clamp01(ww), clamp01(hh)


def copy_or_symlink(src, dst, use_symlink=False):
    """파일 복사 또는 심볼릭 링크"""
    if dst.exists():
        return
    
    if use_symlink:
        try:
            dst.symlink_to(src.resolve())
            return
        except Exception:
            # Symlink 실패 시 copy로 fallback
            pass
    
    shutil.copy2(src, dst)


def check_label_format(label_dir, nc, max_report=10):
    """
    YOLO 라벨 포맷 검증
    
    Returns:
        list: 오류 샘플 [(filename, line, reason), ...]
        int: 전체 오류 개수
    """
    bad = []
    for p in sorted(label_dir.glob("*.txt")):
        txt = p.read_text(encoding="utf-8").strip()
        if not txt:
            continue
        
        for ln in txt.splitlines():
            parts = ln.strip().split()
            if len(parts) != 5:
                bad.append((p.name, ln, "len!=5"))
                continue
            
            try:
                c = int(parts[0])
                vals = list(map(float, parts[1:]))
            except Exception:
                bad.append((p.name, ln, "parse_error"))
                continue
            
            if not (0 <= c < nc):
                bad.append((p.name, ln, f"class_out_of_range (0-{nc-1})"))
                continue
            
            if any(v < 0 or v > 1 for v in vals):
                bad.append((p.name, ln, "val_out_of_[0,1]"))
                continue
    
    return bad[:max_report], len(bad)


def main():
    parser = argparse.ArgumentParser(description="YOLO Dataset 준비")
    parser.add_argument("--run-name", type=str, help="실험명")
    parser.add_argument("--symlink", action="store_true", help="이미지를 복사 대신 심볼릭 링크 사용")
    args = parser.parse_args()
    
    print_section("Stage 2-1: YOLO Dataset 준비")
    
    # 1) 경로 설정
    print("\n[1] 경로 설정...")
    paths = setup_project_paths(
        run_name=args.run_name,
        root=Path(__file__).parent.parent,
        create_dirs=True,
        check_input_exists=True,
    )
    print(f"  ✅ RUN_NAME: {paths['RUN_NAME']}")
    
    # 2) 필수 파일 확인
    print("\n[2] 필수 파일 확인...")
    merged_coco = paths["CACHE"] / "train_merged_coco.json"
    label_map = paths["CACHE"] / "label_map_full.json"
    split_json = paths["CACHE"] / "splits" / "split_train_valid.json"
    
    missing = []
    if not merged_coco.exists():
        missing.append("train_merged_coco.json")
    if not label_map.exists():
        missing.append("label_map_full.json")
    if not split_json.exists():
        missing.append("split_train_valid.json")
    
    if missing:
        print(f"  ❌ 필수 파일 없음: {', '.join(missing)}")
        print(f"  ℹ️  먼저 scripts/1_create_coco_format.py와 scripts/0_splitting.py를 실행하세요.")
        sys.exit(1)
    
    print(f"  ✅ 모든 필수 파일 존재")
    
    # 3) 데이터 로드
    print("\n[3] 데이터 로드...")
    with open(merged_coco, "r", encoding="utf-8") as f:
        coco = json.load(f)
    with open(label_map, "r", encoding="utf-8") as f:
        lm = json.load(f)
    with open(split_json, "r", encoding="utf-8") as f:
        split = json.load(f)
    
    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    
    # Category mapping
    id2idx = {int(k): int(v) for k, v in lm.get("id2idx", {}).items()}
    names = lm.get("names", [])
    nc = int(lm.get("num_classes", len(id2idx)))
    
    train_ids = set(int(x) for x in split.get("train_image_ids", []))
    val_ids = set(int(x) for x in split.get("valid_image_ids", []))
    
    print(f"  ✅ 이미지: {len(images)}")
    print(f"  ✅ Annotations: {len(anns)}")
    print(f"  ✅ Classes: {nc}")
    print(f"  ✅ Train: {len(train_ids)}, Val: {len(val_ids)}")
    
    # 4) Image/Annotation 인덱싱
    print("\n[4] 데이터 인덱싱...")
    img_by_id = {int(im["id"]): im for im in images if isinstance(im, dict) and im.get("id") is not None}
    
    anns_by_img = defaultdict(list)
    for a in anns:
        if not isinstance(a, dict):
            continue
        iid = a.get("image_id", None)
        if iid is None:
            continue
        anns_by_img[int(iid)].append(a)
    
    # 5) YOLO 데이터셋 디렉터리 생성
    print("\n[5] YOLO 데이터셋 디렉터리 생성...")
    dataset_root = paths["PROC_ROOT"] / "datasets" / f"pill_od_yolo_{paths['RUN_NAME']}"
    
    img_train_dir = dataset_root / "images" / "train"
    img_val_dir = dataset_root / "images" / "val"
    lbl_train_dir = dataset_root / "labels" / "train"
    lbl_val_dir = dataset_root / "labels" / "val"
    
    for p in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        p.mkdir(parents=True, exist_ok=True)
    
    print(f"  ✅ {dataset_root.relative_to(paths['ROOT'])}")
    
    # 6) 이미지 복사 및 라벨 변환
    print("\n[6] 이미지/라벨 변환 중...")
    use_symlink = args.symlink
    train_img_src = paths["TRAIN_IMAGES"]
    
    stats = {
        "n_train_images": 0,
        "n_val_images": 0,
        "n_train_labels": 0,
        "n_val_labels": 0,
        "n_train_objects": 0,
        "n_val_objects": 0,
        "n_missing_images": 0,
        "n_skipped_boxes": 0,
        "n_empty_labels": 0,
    }
    
    def process_split(image_ids, img_dir, lbl_dir, split_name):
        for iid in image_ids:
            im = img_by_id.get(int(iid))
            if im is None:
                continue
            
            file_name = str(im.get("file_name"))
            W = im.get("width", None)
            H = im.get("height", None)
            
            if W is None or H is None:
                stats["n_skipped_boxes"] += len(anns_by_img.get(int(iid), []))
                continue
            
            src_img = train_img_src / Path(file_name).name
            if not src_img.exists():
                stats["n_missing_images"] += 1
                continue
            
            dst_img = img_dir / Path(file_name).name
            copy_or_symlink(src_img, dst_img, use_symlink)
            
            # Label file
            label_path = lbl_dir / (Path(file_name).stem + ".txt")
            lines = []
            
            for a in anns_by_img.get(int(iid), []):
                cid = a.get("category_id", None)
                bbox = a.get("bbox", None)
                
                if cid is None or bbox is None or not isinstance(bbox, list) or len(bbox) != 4:
                    stats["n_skipped_boxes"] += 1
                    continue
                
                cid = int(cid)
                if cid not in id2idx:
                    stats["n_skipped_boxes"] += 1
                    continue
                
                cls = id2idx[cid]
                x, y, w, h = map(float, bbox)
                
                if w <= 0 or h <= 0:
                    stats["n_skipped_boxes"] += 1
                    continue
                
                xc, yc, ww, hh = xywh_to_yolo([x, y, w, h], W, H)
                
                if ww <= 0 or hh <= 0:
                    stats["n_skipped_boxes"] += 1
                    continue
                
                lines.append(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
            
            if not lines:
                stats["n_empty_labels"] += 1
                label_path.write_text("", encoding="utf-8")
            else:
                label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            
            # Stats
            if split_name == "train":
                stats["n_train_images"] += 1
                stats["n_train_labels"] += 1
                stats["n_train_objects"] += len(lines)
            else:
                stats["n_val_images"] += 1
                stats["n_val_labels"] += 1
                stats["n_val_objects"] += len(lines)
    
    process_split(train_ids, img_train_dir, lbl_train_dir, "train")
    process_split(val_ids, img_val_dir, lbl_val_dir, "val")
    
    print(f"  ✅ Train: {stats['n_train_images']} images, {stats['n_train_objects']} objects")
    print(f"  ✅ Val: {stats['n_val_images']} images, {stats['n_val_objects']} objects")
    if stats["n_missing_images"] > 0:
        print(f"  ⚠️  Missing images: {stats['n_missing_images']}")
    if stats["n_skipped_boxes"] > 0:
        print(f"  ⚠️  Skipped boxes: {stats['n_skipped_boxes']}")
    
    # 7) data.yaml 생성
    print("\n[7] data.yaml 생성...")
    data_yaml = dataset_root / "data.yaml"
    yaml_lines = [
        f"path: {dataset_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {nc}",
        "names:",
    ]
    
    for i, n in enumerate(names):
        n_safe = str(n).replace('"', '\\"')
        yaml_lines.append(f'  {i}: "{n_safe}"')
    
    data_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    print(f"  ✅ {data_yaml.relative_to(paths['ROOT'])}")
    
    # 8) Label 포맷 검증
    print("\n[8] Label 포맷 검증...")
    bad_train, n_bad_train = check_label_format(lbl_train_dir, nc)
    bad_val, n_bad_val = check_label_format(lbl_val_dir, nc)
    
    print(f"  ✅ Train bad lines: {n_bad_train}")
    print(f"  ✅ Val bad lines: {n_bad_val}")
    
    if n_bad_train > 0:
        print(f"  ⚠️  Train 오류 샘플:")
        for fname, line, reason in bad_train[:3]:
            print(f"     - {fname}: {reason}")
    
    if n_bad_val > 0:
        print(f"  ⚠️  Val 오류 샘플:")
        for fname, line, reason in bad_val[:3]:
            print(f"     - {fname}: {reason}")
    
    # 9) Manifest 저장
    print("\n[9] Manifest 저장...")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "data_yaml": str(data_yaml),
        "copy_mode": "symlink" if use_symlink else "copy",
        "merged_coco": str(merged_coco),
        "split_json": str(split_json),
        "label_map": str(label_map),
        "stats": stats,
        "sanity": {
            "n_bad_train_lines": n_bad_train,
            "n_bad_val_lines": n_bad_val,
            "bad_train_samples": bad_train,
            "bad_val_samples": bad_val,
        },
    }
    
    manifest_path = dataset_root / "convert_manifest.json"
    save_json(manifest_path, manifest)
    print(f"  ✅ {manifest_path.relative_to(paths['ROOT'])}")
    
    print_section("✅ YOLO Dataset 준비 완료")
    print(f"\ndata.yaml 경로:")
    print(f"  {data_yaml}")
    print(f"\n다음 단계:")
    print(f"  python scripts/3_train.py --run-name {paths['RUN_NAME']}")


if __name__ == "__main__":
    main()
