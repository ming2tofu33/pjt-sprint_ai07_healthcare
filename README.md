# 💊 Healthcare AI Project - Team #4

> AI 엔지니어링 팀이 되어, 알약 이미지 인식 모델을 개발하는 프로젝트입니다.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v8-green.svg)](https://github.com/ultralytics/ultralytics)

---

## 🎯 프로젝트 개요

사용자가 모바일 앱으로 촬영한 알약 사진에서 **최대 4개 알약의 이름(클래스)과 위치(Bounding Box)** 를 자동으로 검출하는 Object Detection 모델을 개발합니다.

- **기간**: 2026.01.27 ~ 2026.02.13 (3주)
- **플랫폼**: Kaggle Private Competition
- **목표**: mAP@0.5 기준 0.50 이상 달성

---

## 👥 Team Members

| Name | Role | Sub Role |
|------|------|----------|
| 김도민 | Project Manager | Model Architect |
| 안뜰에봄 | Data Engineer | Project Manager |
| 서지은 | Model Architect | Data Engineer |
| 김준혁 | Experimentation Lead | FE & Presentation |
| 이건호 | FE & Presentation | Model Architect |

---

## 🛠️ Tech Stack

### Core
- **Python** 3.8+
- **PyTorch** 2.0+
- **OpenCV** 4.x

### Models
- **YOLO v8** (Main)
- **YOLO v11** (Latest)
- **Faster R-CNN** (Comparison)

### Tools
- **Kaggle** - Competition Platform
- **Google Colab** - GPU Training
- **W&B** - Experiment Tracking
- **Notion** - Project Management
- **GitHub** - Version Control

### Additional
- **Grad-CAM** - XAI Visualization
- **Frontend MVP** - Demo UI

---

## 📂 Project Structure

```
pjt-sprint_ai07_healthcare/
├── data/                   # 데이터셋 (gitignore)
├── notebooks/              # EDA & 실험 노트북
├── src/                    # 소스 코드
│   ├── models/            # 모델 아키텍처
│   ├── preprocessing/     # 전처리 파이프라인
│   ├── training/          # 학습 스크립트
│   └── utils/             # 유틸리티 함수
├── experiments/           # 실험 기록
├── submission/            # Kaggle 제출 파일
├── docs/                  # 문서 & 회의록
├── requirements.txt       # 패키지 의존성
└── README.md
```

---

## 🚀 Quick Start

### 1. 환경 설정

```bash
# Repository 클론
git clone https://github.com/ming2tofu33/pjt-sprint_ai07_healthcare.git
cd pjt-sprint_ai07_healthcare

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

---

## 📝 Documentation

- 📅 [프로젝트 일정](https://sprint-ai07-healthcare.notion.site/Schedule-2f57fcf828e680fd8f7ac7f1c02d0f22)
- 📓 [협업 일지](https://sprint-ai07-healthcare.notion.site/Logs-2f57fcf828e6809b8a21cef0cc5df8a0)
- 🧪 [실험 기록](https://sprint-ai07-healthcare.notion.site/Test-Record-2f57fcf828e680e08441f2acbaae6732)
- 🚨 [Risk & Issue](https://sprint-ai07-healthcare.notion.site/Risk-Issue-2f57fcf828e6803eac36f3592d742de0)
- 📐 [Project Charter](https://sprint-ai07-healthcare.notion.site/Project-Charter-2f57fcf828e680d0ad57fd6d3fec727a)
- 🏠 [프로젝트 홈](https://sprint-ai07-healthcare.notion.site/Healthcare-Project-0787fcf828e6834da8f40130b654fa4c)


- 🚀 Kaggle **상위 30%**

---

## 📜 License

This project is for educational purposes as part of Code-it Sprint AI Bootcamp.

---

## 📞 Contact

- **Team Notion**: [Healthcare Project](https://sprint-ai07-healthcare.notion.site)
- **GitHub**: [pjt-sprint_ai07_healthcare](https://github.com/ming2tofu33/pjt-sprint_ai07_healthcare)

---

<div align="center">
  <sub>Built with ❤️ by Team #4</sub>
</div>