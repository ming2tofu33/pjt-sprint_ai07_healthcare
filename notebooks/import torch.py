import torch

# 1. CUDA 사용 가능 여부 확인
if torch.cuda.is_available():
    print("✅ CUDA(GPU)를 사용할 수 있습니다.")
    print(f"   - GPU 모델명: {torch.cuda.get_device_name(0)}")
    print(f"   - CUDA 버전: {torch.version.cuda}")
else:
    print("❌ CUDA를 사용할 수 없습니다. (CPU로 실행됩니다)")

# 2. 디바이스 설정 및 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\\n▶ 최종 설정된 Device: {device}")