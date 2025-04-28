<h1 align="center">🎮 LoL Highlight Auto-Generator 🤖✨</h1>

<p align="center">
  <strong>SKT AI Fellowship 2기 최우수 프로젝트 🏆 리그 오브 레전드 경기의 하이라이트를 자동으로 생성하는 AI 모델입니다.</strong>
</p>

![final model](https://github.com/user-attachments/assets/77e17b02-bc93-4721-85f3-e1437744d0c9)

<br/>

## 🌟 프로젝트 소개 (Overview)

7시간짜리 풀경기 영상에서 **AI가 알아서** 핵심 전투와 멋진 플레이만 쏙쏙 골라 **5분짜리 하이라이트**를 만들어줍니다. LoL 전문가가 아니어도, 편집 기술만 있다면 누구나 손쉽게 고퀄리티 하이라이트 영상을 제작할 수 있습니다.

## 🎯 핵심 목표 (Project Goal)

* **자동 하이라이트 생성:** LoL 경기 영상에서 중요한 순간(전투, 주요 플레이)을 자동으로 감지하고 추출합니다.
* **편집 효율 극대화:** LoL 비전문가도 쉽게 하이라이트를 만들 수 있도록, '하이라이트 구간 감지' 과정을 자동화합니다.
* **시청 경험 향상:** 짧고 임팩트 있는 요약 영상을 제공하여 시청자 만족도를 높입니다.

## ✨ 주요 기능 (Features)

* **AI 기반 장면 중요도 분석:** Attention 메커니즘을 활용하여 영상 프레임별 중요도를 정확하게 예측합니다.
* **영상 + 음성 멀티모달 분석:** 비디오 영상뿐만 아니라 해설자의 음성 톤 변화까지 분석하여 하이라이트 감지 정확도를 높였습니다.
* **정교한 프레임 매칭:** 풀 영상과 하이라이트 영상 간의 정확한 프레임 매칭으로 자동 레이블링 및 성능 평가 기반 마련.
* **효율적인 세그먼트 선택:** KTS(Kernel Temporal Segmentation)와 Knapsack 알고리즘으로 최적의 하이라이트 구간 조합.

## 🔧 기술 스택 및 핵심 개념 (Tech Stack & Concepts)

* **언어:** Python
* **주요 라이브러리:** PyTorch, OpenCV, Librosa (음성 처리), Scikit-learn
* **모델 아키텍처:**
    * VASNet (Video Summarization with Attention) 기반
    * GoogleNet, EfficientNetB7 (Feature Extraction)
    * Self-Attention Network
* **핵심 기술:**
    * **Video Summarization:** 영상 요약 기술
    * **Frame Matching:** 영상 프레임 정합 (논문 기반 알고리즘 구현)
    * **Feature Extraction:** 영상(GoogleNet) 및 음성(MFCC) 특징 추출
    * **Temporal Segmentation:** KTS (Kernel Temporal Segmentation)
    * **Optimization:** Knapsack Algorithm, SMBO (Hyperparameter Tuning)
    * **Preprocessing:** VGGNet 기반 게임/비게임 화면 분류

## 💡 주요 도전 과제 및 해결 (Key Challenges & Solutions)
* **챌린지 1: 데이터 확보 및 정제**: 흩어져 있는 풀 영상/하이라이트 영상 수집 및 7시간 영상에서 불필요한 (광고, 대기 화면 등) 부분 제거.
    * **솔루션**: 트위치/유튜브 크롤링, VGGNet 기반 In-Game Classifier 학습 및 적용.
* **챌린지 2: 정확한 자동 레이블링**: 기존 Frame Matching 기법(SIFT 등)의 노이즈 취약성.
    * **솔루션**: 관련 SOTA 논문 기반의 Robust Frame Matching 알고리즘 직접 구현 및 적용.
* **챌린지 3: 모델 성능 개선**: 기본 영상 Feature만으로는 부족한 하이라이트 감지 성능.
    * **솔루션**: 해설자 음성(Audio) Feature(MFCC)를 추출하고, 실험을 통해 최적의 방식으로 영상 Feature와 결합하여 모델 성능 향상.
 
## 📈 결과 (Results)
자체 평가 결과, AI가 생성한 하이라이트는 전문가가 직접 편집한 하이라이트와 매우 유사한 높은 품질을 보여주었습니다.

## ⚙️ 프로젝트 구조 (Project Structure)
```bash
lol_highlight_ai/
├── preprocessing/         # 데이터 전처리 (다운로드, 게임 분류, 레이블링, Feature 추출)
│   ├── downloader/
│   ├── ingame_classifier/
│   ├── labeling/
│   └── h5/                # HDF5 데이터 관리
├── model/                 # 메인 모델 (VASNet 기반) 및 관련 모듈
│   ├── data/              # 데이터셋 및 Split 관리
│   ├── datasets/          # 데이터 로더
│   ├── evaluation/        # 모델 평가 및 하이라이트 생성
│   ├── model/             # 모델 아키텍처 정의
│   ├── splits/            # 데이터 분할 정보 (JSON)
│   ├── main.py            # 모델 학습/테스트 실행
│   └── config.py          # 설정 파일
├── README.md              # 바로 여기! 👋
└── ... (기타 설정 파일, 요구사항 등)
```
