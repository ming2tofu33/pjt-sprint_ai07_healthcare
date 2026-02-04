import os

# 현재 위치: ROOT/configs
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 프로젝트 루트: ROOT
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

CONFIG = {
    # [1. 경로 설정] - 이미지와 JSON이 아무리 깊이 있어도 SEARCH_ROOT 아래에만 있으면 다 찾습니다.
    "AIHUB_JSON": os.path.join(PROJECT_ROOT, "data", "raw", "ex_train_annotations"),
    "KAGGLE_JSON": os.path.join(PROJECT_ROOT, "data", "raw", "train_annotations"),
    "SEARCH_ROOT": os.path.join(PROJECT_ROOT, "data", "raw"),
    "PROCESSED_DIR": os.path.join(PROJECT_ROOT, "data", "processed"),
    "FINAL_CSV": os.path.join(PROJECT_ROOT, "data", "processed", "final_golden_dataset_v2.csv"),
    "YOLO_ROOT": os.path.join(PROJECT_ROOT, "data", "yolo_format"),

    # [2. 분할 및 증강 설정]
    "SPLIT_RATIO": 0.2,
    "AUG_TARGET_ID": 114, 
    "AUG_GOAL_COUNT": 300,

    # [3. 증강 ON/OFF 스위치]
    "AUG_GEOMETRIC_ON": True,   # 회전/반전 (필수)
    "AUG_COLOR_ON": True,       # 밝기/대비/색상
    "AUG_BLUR_ON": False,       # 흐림 효과
}