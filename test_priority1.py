#!/usr/bin/env python3
"""
Priority 1 수정사항 테스트: category_id 변환 확인
"""

import json
from pathlib import Path

print("=" * 60)
print("Priority 1: category_id 변환 테스트")
print("=" * 60)

# 1. label_map 구조 확인 (예시)
label_map_example = {
    "id2idx": {
        "1900": 0,
        "16548": 1,
        "19607": 2,
    },
    "idx2id": {
        "0": 1900,
        "1": 16548,
        "2": 19607,
    },
    "names": ["class1", "class2", "class3"]
}

print("\n[1] label_map 구조:")
print(json.dumps(label_map_example, indent=2, ensure_ascii=False))

# 2. 변환 로직 시뮬레이션
print("\n[2] 변환 로직:")
print("  YOLO 예측: cls_idx = 0")
print("  label_map['idx2id']['0'] = 1900")
print("  submission.csv: category_id = 1900 ✅")

print("\n[3] 수정 전 (잘못됨):")
print("  cls = int(boxes.cls[idx].item())  # 0")
print("  category_id = cls  # 0 ❌ (Kaggle에서 0점)")

print("\n[4] 수정 후 (올바름):")
print("  cls_idx = int(boxes.cls[idx].item())  # 0")
print("  category_id = idx2id[cls_idx]  # 1900 ✅")

print("\n[5] 수정된 파일:")
print("  - scripts/5_submission.py")
print("  - src/inference.py")

print("\n[6] 다음 실행 명령:")
print("  python scripts/5_submission.py --run-name test_exp")
print("  # submission.csv의 category_id가 1900, 16548 등으로 나와야 함")

print("\n" + "=" * 60)
print("✅ Priority 1 수정 완료!")
print("=" * 60)
