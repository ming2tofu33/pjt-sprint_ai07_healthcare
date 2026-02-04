import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

CONFIG = {
    # 1. 경로 설정
    "RAW_CSV": os.path.join(PROJECT_ROOT, "data", "processed", "final_golden_dataset_v2.csv"),
    "SEARCH_ROOT": os.path.join(PROJECT_ROOT, "data", "raw"),
    "OUTPUT_ROOT": os.path.join(PROJECT_ROOT, "data", "yolo_format"),
    "SPLIT_RATIO": 0.2,
    
    # =====================================================
    # 🧪 [실험실] 증강(Augmentation) 마스터 컨트롤
    # =====================================================
    "USE_AUGMENTATION": False,
    "AUG_TARGET_ID": 114,
    "AUG_COUNT": 300,
    
    # 1. 기하학적 변환 (모양)
    "AUG_ROTATE_LIMIT": 30,    # 회전 각도 (±30도)
    "AUG_ROTATE_PROB": 0.7,    # 회전 적용 확률
    "AUG_FLIP_PROB": 0.5,      # 좌우 반전 확률
    
    # 2. 색상/조명 변환 (빛)
    "AUG_BRIGHT_LIMIT": 0.2,   # 밝기/대비 조절 범위
    "AUG_BRIGHT_PROB": 0.5,    # 밝기 적용 확률
    
    # 3. 색조 변환 (색감)
    "AUG_HUE_LIMIT": 20,       # 색조(Hue)
    "AUG_SAT_LIMIT": 30,       # 채도(Saturation)
    "AUG_VAL_LIMIT": 20,       # 명도(Value)
    "AUG_HSV_PROB": 0.5,       # 색조 적용 확률
    
    # 4. 픽셀 노이즈 (센서)
    "AUG_RGB_SHIFT": 15,       # RGB 값 이동 범위
    "AUG_RGB_PROB": 0.5        # RGB Shift 확률
}