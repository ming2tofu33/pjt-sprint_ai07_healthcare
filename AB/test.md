내일 task 1.깃, 깃허브 협업
□ 풀
□ 푸시
□ PR
□ 머지(마스터)
□ 충돌상황

2. 프로젝트 앱개발
   □ 주제 및 방향성 정하기
   → 앱: 경량화 필수
   → 약을 왜 사진찍어서 정보를 알고싶은가?
   → 비지니스로 어떻게 구조화 시킬수있나

□ 데이터셋 구성(외부데이터 가져오기 등) 논의 및 전처리 전략 기획
-> 제안: 전이학습 / 데이터를 더 추가로 가져오지 않아도 괜찮지 않을까요?

3. 환경설정
   개발 언어 및 가상환경
   □ Python: 3.14.x (최신 3.15 버전 대비 패키지 호환성 및 안정성이 우수함)
   □ Venv / conda / Git bash (사용 확인 후 선택 필요)

필수 라이브러리 리스트
a. 데이터 핸들링
□ numpy : 대규모 다차원 배열 및 고성능 수치 계산을 위한 필수 도구
□ pandas : 데이터프레임 기반의 정량적 데이터 분석 및 처리를 담당
□ sklearn (scikit-learn) : 학습/검증 데이터 분할(Train/Test Split) 및 기초 통계 분석에 사용합니다.

b. 이미지 처리 및 컴퓨터 비전
□ OpenCV (opencv-python) : 알약 이미지의 전처리(Resize, Grayscale, Noise 제거 등)를 수행합니다.
□ Torchvision : Pytorch와 연동되는 이미지 변환(Transform) 및 유명 모델 아키텍처를 제공합니다.
□ Pillow : 기초적인 이미지 파일 열기 및 조작을 지원합니다.

c. 딥러닝 프레임워크
□ PyTorch : 프로젝트의 핵심 AI 모델 학습 엔진입니다.

d. 시각화 리포팅
□ Matplotlibt : 데이터 분포 및 학습 결고(Loss, Accuracy)를 그래프로 시각화합니다.
□ Seaborn : Matplotlib 기반의 고수준 인터페이스로, 더 정교한 통계 시각화를 제공합니다.
□ Koreanize-matplotlib : 그래프 내 한글 깨짐 현상을 방지하는 [패치] 라이브러리

e. 유틸리티 및 인터페이스
□ Tqdm : 대용량 데이터 처리 및 학습 시 진행 상태를 프로그레스 바로 보여줍니다.
□ Gradio : 완성된 모델을 웹 환경에서 즉시 테스트할 수 있는 GUI 데모 페이지를 구축합니다.

```
import numpy as np
import pandas as pd
import cv2  # OpenCV
import torch
import torchvision
from PIL import Image  # Pillow
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
from tqdm import tqdm
import gradio as gr
from sklearn.model_[selection] import [train_test_split] # sklearn 예시
```

---

협업 가이드라인
Git 브랜치 전략 : feature/기능이름 형식을 준수하여 작업 선을 명확히 관리
커밋 컨벤션 : 커밋 메시지 말머리에 `[add]`, `[fix]`, `[update]` 등을 사용하여 수정 내역 직관적으로 파악
