import sys

print("✅ Notebook kernel python:", sys.executable)

# 1) pip 최신화 (설치 꼬임 방지)
!"{sys.executable}" -m pip install -U pip setuptools wheel

# 2) Ultralytics 설치
# (일반적으로 이 한 줄이면 끝)
!"{sys.executable}" -m pip install -U ultralytics

# 3) 설치 확인
import ultralytics
from ultralytics import YOLO
import torch

print("✅ ultralytics:", ultralytics.__version__)
print("✅ torch      :", torch.__version__)
print("✅ cuda avail :", torch.cuda.is_available())
