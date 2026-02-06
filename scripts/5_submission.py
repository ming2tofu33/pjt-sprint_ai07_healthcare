#!/usr/bin/env python3
"""
Kaggle 제출 파일 생성 스크립트 (Placeholder)

기능:
1. Test 이미지에 대한 추론 실행
2. Top-4 객체 선택 (대회 규칙)
3. submission.csv 생성
4. 제출 파일 검증

사용법:
    python scripts/5_submission.py --run-name RUN_NAME
    
    예시:
    python scripts/5_submission.py --run-name exp_baseline_v1
    python scripts/5_submission.py --run-name exp_v1 --ckpt best --conf 0.25
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import json

# src 모듈 import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import (
    setup_project_paths,
    load_config,
    print_section,
)


def main():
    parser = argparse.ArgumentParser(description="제출 파일 생성")
    parser.add_argument("--run-name", type=str, required=True, help="실험명")
    parser.add_argument("--ckpt", type=str, default="best", choices=["best", "last"], help="체크포인트")
    parser.add_argument("--conf", type=float, help="Confidence threshold (기본: config 값)")
    parser.add_argument("--device", type=str, default="0", help="GPU device")
    args = parser.parse_args()
    
    print_section("Stage 2-4: 제출 파일 생성")
    
    # 1) 경로 설정
    print("\n[1] 경로 설정...")
    paths = setup_project_paths(
        run_name=args.run_name,
        root=Path(__file__).parent.parent,
        create_dirs=True,
        check_input_exists=True,
    )
    print(f"  ✅ RUN_NAME: {paths['RUN_NAME']}")
    
    # 2) Config 로드
    print("\n[2] Config 로드...")
    config_path = paths["CONFIG"] / "config.json"
    config = load_config(config_path)
    
    conf_thr = args.conf or config["infer"]["conf_thr"]
    nms_iou = config["infer"]["nms_iou_thr"]
    max_det = config["infer"]["max_det_per_image"]
    
    print(f"  ✅ Conf threshold: {conf_thr}")
    print(f"  ✅ NMS IoU: {nms_iou}")
    print(f"  ✅ Max det/image: {max_det}")
    
    # 3) 체크포인트 확인
    print("\n[3] 체크포인트 확인...")
    ckpt_path = paths["CKPT"] / f"{args.ckpt}.pt"
    
    if not ckpt_path.exists():
        print(f"  ❌ 체크포인트 없음: {ckpt_path}")
        sys.exit(1)
    
    print(f"  ✅ {ckpt_path.relative_to(paths['ROOT'])}")
    
    # 4) Test 이미지 확인
    print("\n[4] Test 이미지 확인...")
    test_images = list(paths["TEST_IMAGES"].glob("*.png"))
    
    if not test_images:
        print(f"  ❌ Test 이미지 없음: {paths['TEST_IMAGES']}")
        sys.exit(1)
    
    print(f"  ✅ Test 이미지: {len(test_images)}")
    
    # 5) Label map 로드 (YOLO idx → 원본 category_id 변환용)
    print("\n[5] Label map 로드...")
    cache_dir = paths["DATA_ROOT"] / "processed" / "cache" / args.run_name
    
    # label_map_whitelist 또는 label_map_full 찾기
    label_map_path = cache_dir / "label_map_whitelist.json"
    if not label_map_path.exists():
        label_map_path = cache_dir / "label_map_full.json"
    
    if not label_map_path.exists():
        print(f"  ❌ Label map 없음: {label_map_path}")
        print(f"  ℹ️  먼저 scripts/1_create_coco_format.py를 실행하세요.")
        sys.exit(1)
    
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    
    # idx2id: YOLO 인덱스 → 원본 category_id
    # label_map은 {original_category_id: yolo_idx, ...} 구조
    # 숫자 키만 필터링하고 뒤집어서 {yolo_idx: original_category_id} 만들기
    idx2id = {}
    for k, v in label_map.items():
        # 숫자 키만 처리 (메타데이터 키 무시)
        if k.isdigit():
            idx2id[v] = int(k)
    
    print(f"  ✅ Label map 로드: {label_map_path.name}")
    print(f"  ✅ 클래스 개수: {len(idx2id)}")
    
    # 6) 추론 실행
    print("\n[6] 추론 실행...")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ❌ ultralytics 패키지가 설치되지 않았습니다.")
        sys.exit(1)
    
    model = YOLO(str(ckpt_path))
    
    print(f"  🚀 추론 중... ({len(test_images)} images)")
    
    # 추론 실행
    results = model.predict(
        source=str(paths["TEST_IMAGES"]),
        conf=conf_thr,
        iou=nms_iou,
        max_det=max_det,
        device=args.device,
        save=False,
        verbose=False,
    )
    
    print(f"  ✅ 추론 완료!")
    
    # 7) submission.csv 생성 (YOLO idx → 원본 category_id 변환)
    print("\n[7] submission.csv 생성...")
    rows = []
    annotation_id = 1
    
    for result in results:
        image_path = Path(result.path)
        image_id = int(image_path.stem)  # 파일명이 숫자라고 가정
        
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        
        # Top-4 선택 (score 높은 순)
        scores = boxes.conf.cpu().numpy()
        top_indices = scores.argsort()[::-1][:max_det]
        
        for idx in top_indices:
            cls_idx = int(boxes.cls[idx].item())  # YOLO 인덱스 (0~55)
            score = float(boxes.conf[idx].item())
            xyxy = boxes.xyxy[idx].cpu().numpy()
            
            # ⭐ YOLO 인덱스 → 원본 category_id 변환
            if cls_idx not in idx2id:
                print(f"  ⚠️  Unknown class index: {cls_idx} (이미지 {image_id})")
                continue
            category_id = idx2id[cls_idx]  # 원본 dl_idx (1900, 16548 등)
            
            # xyxy → xywh
            x1, y1, x2, y2 = xyxy
            x = float(x1)
            y = float(y1)
            w = float(x2 - x1)
            h = float(y2 - y1)
            
            rows.append({
                "annotation_id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,  # ⭐ 원본 category_id 사용
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "score": score,
            })
            annotation_id += 1
    
    # DataFrame 생성
    df = pd.DataFrame(rows)
    
    if df.empty:
        print(f"  ⚠️  예측 결과 없음!")
        df = pd.DataFrame(columns=["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"])
    
    print(f"  ✅ 총 예측 객체: {len(df)}")
    print(f"  ✅ 이미지당 평균: {len(df)/len(test_images):.2f}")
    
    # category_id 분포 확인
    if not df.empty:
        unique_cats = df["category_id"].unique()
        print(f"  ✅ 예측된 카테고리 개수: {len(unique_cats)}")
        print(f"  ℹ️  카테고리 ID 샘플: {sorted(unique_cats)[:10]}")
    
    # 8) 제출 파일 저장
    print("\n[8] 제출 파일 저장...")
    submission_path = paths["SUBMISSIONS"] / "submission.csv"
    df.to_csv(submission_path, index=False)
    print(f"  ✅ {submission_path.relative_to(paths['ROOT'])}")
    
    # 9) 검증
    print("\n[9] 제출 파일 검증...")
    required_cols = ["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    if missing_cols:
        print(f"  ❌ 필수 컬럼 누락: {missing_cols}")
    else:
        print(f"  ✅ 컬럼 검증 통과")
    
    # NaN 체크
    if df.isnull().any().any():
        print(f"  ⚠️  NaN 값 존재!")
    else:
        print(f"  ✅ NaN 없음")
    
    # 음수 좌표 체크
    if (df[["bbox_x", "bbox_y", "bbox_w", "bbox_h"]] < 0).any().any():
        print(f"  ⚠️  음수 bbox 존재!")
    else:
        print(f"  ✅ bbox 유효")
    
    print_section("✅ 제출 파일 생성 완료")
    print(f"\n제출 파일:")
    print(f"  {submission_path}")
    print(f"\n통계:")
    print(f"  - 총 객체: {len(df)}")
    print(f"  - 총 이미지: {len(test_images)}")
    print(f"  - 이미지당 평균: {len(df)/len(test_images):.2f}")


if __name__ == "__main__":
    main()
