#!/usr/bin/env python3
"""
테스트용 더미 데이터 생성 스크립트

실제 데이터 없이 파이프라인을 테스트할 수 있도록 최소한의 더미 데이터를 생성합니다.
"""

import json
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# src 모듈 import
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import print_section


def create_dummy_data(root: Path, n_train_images: int = 10, n_test_images: int = 5, n_categories: int = 5):
    """
    더미 데이터 생성
    
    Args:
        root: 프로젝트 루트
        n_train_images: Train 이미지 수
        n_test_images: Test 이미지 수
        n_categories: 카테고리 수
    """
    print_section("더미 데이터 생성")
    
    data_root = root / "data" / "raw"
    train_img_dir = data_root / "train_images"
    train_ann_dir = data_root / "train_annotations"
    test_img_dir = data_root / "test_images"
    
    # 디렉토리 생성
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_ann_dir.mkdir(parents=True, exist_ok=True)
    test_img_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[INFO] 생성 위치: {data_root}")
    print(f"[INFO] Train images: {n_train_images}")
    print(f"[INFO] Test images: {n_test_images}")
    print(f"[INFO] Categories: {n_categories}")
    
    # 카테고리 ID (실제 데이터와 유사하게 큰 숫자 사용)
    category_ids = [1900 + i * 100 for i in range(n_categories)]
    category_names = [f"pill_class_{cid}" for cid in category_ids]
    
    print(f"\n[1] Train 이미지 및 Annotation 생성...")
    for i in range(1, n_train_images + 1):
        # 이미지 생성 (768x768 랜덤 컬러)
        img = Image.fromarray(np.random.randint(0, 255, (768, 768, 3), dtype=np.uint8))
        img_path = train_img_dir / f"{i}.png"
        img.save(img_path)
        
        # Annotation 생성 (이미지당 1~4개 객체)
        n_objects = np.random.randint(1, 5)
        
        for obj_idx in range(n_objects):
            # 각 객체마다 별도의 JSON 파일 생성 (원본 데이터 구조와 동일)
            ann_dir = train_ann_dir / f"class_{category_ids[obj_idx % n_categories]}"
            ann_dir.mkdir(parents=True, exist_ok=True)
            
            # 랜덤 bbox
            x = np.random.randint(50, 400)
            y = np.random.randint(50, 400)
            w = np.random.randint(100, 300)
            h = np.random.randint(100, 300)
            
            coco_ann = {
                "images": [{
                    "id": i,
                    "file_name": f"{i}.png",
                    "width": 768,
                    "height": 768,
                    "dl_idx": str(category_ids[obj_idx % n_categories]),
                }],
                "annotations": [{
                    "id": 1,
                    "image_id": i,
                    "category_id": category_ids[obj_idx % n_categories],
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": float(w * h),
                    "iscrowd": 0,
                    "ignore": 0,
                    "segmentation": [],
                }],
                "categories": [{
                    "id": category_ids[obj_idx % n_categories],
                    "name": category_names[obj_idx % n_categories],
                    "supercategory": "pill",
                }],
                "type": "instances",
            }
            
            ann_path = ann_dir / f"train_{i}_{obj_idx}.json"
            with open(ann_path, "w", encoding="utf-8") as f:
                json.dump(coco_ann, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ Train: {n_train_images} images, {sum(1 for _ in train_ann_dir.rglob('*.json'))} annotations")
    
    print(f"\n[2] Test 이미지 생성...")
    for i in range(1, n_test_images + 1):
        img = Image.fromarray(np.random.randint(0, 255, (768, 768, 3), dtype=np.uint8))
        img_path = test_img_dir / f"{1000 + i}.png"  # Test는 1001부터 시작
        img.save(img_path)
    
    print(f"  ✅ Test: {n_test_images} images")
    
    # README 생성
    readme_path = data_root / "README_DUMMY.md"
    readme_content = f"""# 더미 데이터

이 데이터는 테스트용으로 자동 생성되었습니다.

## 생성 정보
- Train 이미지: {n_train_images}개
- Test 이미지: {n_test_images}개
- 카테고리: {n_categories}개
- 이미지 크기: 768x768
- 생성 스크립트: scripts/create_dummy_data.py

## 실제 데이터 사용 시
1. `data/raw/` 디렉토리의 더미 데이터 삭제
2. 실제 데이터를 `data/raw/` 에 배치:
   - train_images/
   - train_annotations/
   - test_images/

## 주의
이 데이터로 학습한 모델은 의미 없는 결과를 생성합니다.
실제 제출용 모델은 반드시 실제 데이터로 학습하세요.
"""
    readme_path.write_text(readme_content, encoding="utf-8")
    
    print_section("✅ 더미 데이터 생성 완료")
    print(f"\n디렉토리:")
    print(f"  - {train_img_dir}")
    print(f"  - {train_ann_dir}")
    print(f"  - {test_img_dir}")
    print(f"\n⚠️  이 데이터는 테스트용입니다. 실제 제출 시 실제 데이터를 사용하세요!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="테스트용 더미 데이터 생성")
    parser.add_argument("--n-train", type=int, default=10, help="Train 이미지 수")
    parser.add_argument("--n-test", type=int, default=5, help="Test 이미지 수")
    parser.add_argument("--n-cat", type=int, default=5, help="카테고리 수")
    args = parser.parse_args()
    
    root = Path(__file__).parent.parent
    
    # 확인
    data_raw = root / "data" / "raw"
    if data_raw.exists() and list(data_raw.glob("train_images/*.png")):
        response = input(f"\n⚠️  {data_raw}에 이미 데이터가 있습니다. 덮어쓰시겠습니까? (yes/no): ")
        if response.lower() != "yes":
            print("[INFO] 취소됨")
            return
    
    create_dummy_data(root, args.n_train, args.n_test, args.n_cat)


if __name__ == "__main__":
    main()
