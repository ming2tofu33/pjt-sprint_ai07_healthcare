import os

# 현재 위치: ROOT/configs
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 프로젝트 루트: ROOT
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

CONFIG = {
    # --------------------------------------------------------------------------
    # [1. 원본 데이터 경로] - 팀 표준(external/combined) 구조로 동기화
    # --------------------------------------------------------------------------
    
    # Kaggle 데이터 (기존 유지)
    "KAGGLE_JSON": os.path.join(PROJECT_ROOT, "data", "raw", "train_annotations"),
    "KAGGLE_IMG": os.path.join(PROJECT_ROOT, "data", "raw", "train_images"),
    
    # AIHub 외부 데이터 (팀원 표준 경로로 수정 완료)
    "AIHUB_JSON": os.path.join(PROJECT_ROOT, "data", "raw", "external", "combined", "annotations"),
    "AIHUB_IMG": os.path.join(PROJECT_ROOT, "data", "raw", "external", "combined", "images"),

    # --------------------------------------------------------------------------
    # [2. 처리 및 출력 경로]
    # --------------------------------------------------------------------------
    "PROCESSED_DIR": os.path.join(PROJECT_ROOT, "data", "processed"),
    "FINAL_CSV": os.path.join(PROJECT_ROOT, "data", "processed", "final_golden_dataset_v2.csv"),
    "YOLO_ROOT": os.path.join(PROJECT_ROOT, "data", "yolo_format"),

    # [3. 분할 및 증강 설정]
    "SPLIT_RATIO": 0.2,
    "AUG_TARGET_ID": 114, 
    "AUG_GOAL_COUNT": 300,

    # [4. 증강 ON/OFF 스위치]
    "AUG_GEOMETRIC_ON": False,
    "AUG_COLOR_ON": False,
    "AUG_BLUR_ON": False,

    # --------------------------------------------------------------------------
    # [로직 제어 스위치] - ON/OFF를 여기서 결정합니다. / 아직 사용하지 않음
    # --------------------------------------------------------------------------
    #"SWITCH_MY_COLLECTION": True,   # AB전용 데이터 수집 (현재 OFF)
    #"SWITCH_MY_CLEANING": False,     # AB전용 정제/YOLO변환 (현재 OFF)
    #"SWITCH_MY_AUGMENTATION": True,  # AB전용 증강 (이건 좋으니까 ON!)
}