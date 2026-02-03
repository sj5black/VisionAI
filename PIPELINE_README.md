# VisionAI Pipeline - 동물 행동 예측 시스템

경량화된 5단계 AI 파이프라인으로 동물의 행동을 분석하고 예측합니다.

## 📋 파이프라인 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    VisionAI Pipeline                        │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Step 1-2   │    │   Step 3     │    │   Step 4-5   │
│              │    │              │    │              │
│  Detection   │───▶│   Emotion    │───▶│  Temporal    │
│  + Keypoint  │    │   Analysis   │    │  + Predict   │
│              │    │              │    │              │
│   YOLOv8n    │    │ MobileNetV3  │    │ Rule + LSTM  │
└──────────────┘    └──────────────┘    └──────────────┘
```

### 🎯 각 단계 상세

#### Step 1-2: Object + Keypoint Detection (YOLOv8n)
- **모델**: YOLOv8n (6.3 MB) - 가장 경량 모델
- **기능**:
  - 개/고양이 등 동물 탐지 및 위치 파악
  - 신체 부위 keypoint 탐지 (17개 포인트)
- **입력**: RGB 이미지
- **출력**: 객체 위치 (bbox), 클래스, keypoints

#### Step 3: Emotion & Pose Analysis (MobileNetV3-Small)
- **모델**: MobileNetV3-Small (2.5 MB) + 멀티태스크 헤드
- **기능**:
  - **표정 분석**: relaxed, alert, fearful, aggressive, playful
  - **자세 분석**: sitting, standing, lying, running, jumping
  - 통합 상태 판단
- **입력**: 탐지된 객체 영역 (cropped)
- **출력**: 감정, 자세, 통합 상태

#### Step 4: Temporal Action Recognition (규칙 기반)
- **방법**: 시간 축 특징 집계 + 규칙 기반 휴리스틱
- **기능**:
  - 여러 프레임의 감정/자세 변화 추적
  - 움직임 강도 계산
  - 행동 인식: resting, walking, running, playing, eating, grooming, hunting, alert_scan
- **입력**: 시간 순서 특징 시퀀스
- **출력**: 현재 행동, 지속 시간, 움직임 강도

#### Step 5: Behavior Prediction (규칙 기반 + LSTM)
- **모델**: 경량 LSTM (옵션)
- **기능**:
  - 과거 행동 패턴 분석
  - 다음 행동 예측 (5초 후)
  - 대안 행동 제시
- **입력**: 행동 시퀀스
- **출력**: 예측 행동, 신뢰도, 대안

## 🚀 설치

### 1. 의존성 설치

```bash
cd /home/teddy/VisionAI

# 파이프라인 전용 requirements
pip install -r pipeline_requirements.txt
```

### 2. YOLOv8 모델 다운로드

첫 실행 시 자동으로 다운로드됩니다 (~9 MB):
- `yolov8n.pt` (object detection)
- `yolov8n-pose.pt` (keypoint detection)

## 📖 사용법

### CLI로 이미지 분석

```bash
# 기본 사용
python run_pipeline.py --image dog.jpg --output result.jpg

# 디바이스 지정
python run_pipeline.py --image cat.jpg --output result.jpg --device cuda

# 신뢰도 조정
python run_pipeline.py --image pet.jpg --conf 0.7
```

### CLI로 비디오 분석

```bash
# 비디오 분석 (5 FPS 샘플링)
python run_pipeline.py --video cat_video.mp4 --output result.mp4 --fps 5

# 빠른 샘플링 (1 FPS)
python run_pipeline.py --video dog_video.mp4 --output result.mp4 --fps 1

# 시각화 없이 JSON만 저장
python run_pipeline.py --video video.mp4 --output result.json --no-visualize
```

### Python 코드에서 사용

```python
from visionai_pipeline import VisionAIPipeline
import numpy as np
from PIL import Image

# 파이프라인 초기화
pipeline = VisionAIPipeline(device='cuda')

# 단일 이미지 분석
image = np.array(Image.open('dog.jpg'))
result = pipeline.process_image(image)

print(f"탐지: {len(result.detections)}개")
print(f"감정: {result.emotions}")
print(f"행동: {result.action}")
print(f"예측: {result.prediction}")

# 시각화
vis_image = pipeline.visualize(image, result)
```

### 비디오 스트림 처리

```python
import cv2
from visionai_pipeline import VisionAIPipeline

pipeline = VisionAIPipeline(device='cuda')
cap = cv2.VideoCapture('video.mp4')

frame_idx = 0
while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break
    
    # BGR -> RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # 타임스탬프
    timestamp = frame_idx / 30.0  # 30 FPS 가정
    
    # 분석
    result = pipeline.process_frame(frame_rgb, timestamp)
    
    # 결과 출력
    if result.prediction:
        print(f"다음 행동 예측: {result.prediction['predicted_action']}")
    
    frame_idx += 1

cap.release()
```

## ⚙️ 경량화 옵션

필요에 따라 일부 단계를 비활성화하여 더 빠르게 실행:

```bash
# 감정 분석만 비활성화
python run_pipeline.py --image dog.jpg --no-emotion

# 시간 축 분석 비활성화 (단일 이미지에 적합)
python run_pipeline.py --image dog.jpg --no-temporal --no-prediction

# 최소 모드 (탐지만)
python run_pipeline.py --image dog.jpg --no-emotion --no-temporal --no-prediction
```

Python에서:

```python
# 최소 구성 (탐지만)
pipeline = VisionAIPipeline(
    device='cuda',
    enable_emotion=False,
    enable_temporal=False,
    enable_prediction=False
)

# 감정 분석까지만
pipeline = VisionAIPipeline(
    device='cuda',
    enable_temporal=False,
    enable_prediction=False
)
```

## 📊 출력 형식

### JSON 구조

```json
{
  "detections": [
    {
      "class_id": 16,
      "class_name": "dog",
      "confidence": 0.92,
      "bbox": [100, 150, 400, 500],
      "has_keypoints": true
    }
  ],
  "emotions": [
    {
      "class_name": "dog",
      "emotion": "playful",
      "emotion_confidence": 0.85,
      "pose": "running",
      "pose_confidence": 0.91,
      "combined_state": "playing"
    }
  ],
  "action": {
    "action": "playing",
    "confidence": 0.7,
    "duration": 2.5,
    "motion_intensity": 0.8
  },
  "prediction": {
    "predicted_action": "resting",
    "confidence": 0.65,
    "time_horizon": 5.0,
    "alternative_actions": [
      ["walking", 0.25],
      ["playing", 0.10]
    ]
  },
  "timestamp": 1234567890.123,
  "processing_time": 0.15
}
```

## 🎓 모델 학습 (선택)

현재는 규칙 기반으로 동작하지만, 데이터가 있다면 학습 가능:

### Step 3: 감정 분석 모델 학습

```python
from visionai_pipeline.emotion import EmotionAnalyzer, EmotionClassifier
import torch
from torch.utils.data import DataLoader

# 모델 초기화
model = EmotionClassifier(num_emotions=5, num_poses=5)

# 학습 루프 (예시)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for images, emotion_labels, pose_labels in train_loader:
        optimizer.zero_grad()
        
        emotion_logits, pose_logits = model(images)
        
        loss = (criterion(emotion_logits, emotion_labels) + 
                criterion(pose_logits, pose_labels))
        
        loss.backward()
        optimizer.step()

# 저장
torch.save(model.state_dict(), 'emotion_model.pth')

# 사용
pipeline = VisionAIPipeline(emotion_model_path='emotion_model.pth')
```

## 🔧 성능 최적화

### GPU 사용

```python
# CUDA
pipeline = VisionAIPipeline(device='cuda')

# Apple Silicon (MPS)
pipeline = VisionAIPipeline(device='mps')
```

### 배치 처리

여러 이미지를 효율적으로 처리하려면 YOLOv8의 배치 기능 활용:

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model(image_list, batch=8)  # 배치 크기 8
```

### 모델 크기 vs 정확도

| 모델 | 크기 | 속도 | 정확도 |
|------|------|------|--------|
| YOLOv8n | 6 MB | 매우 빠름 | 양호 |
| YOLOv8s | 22 MB | 빠름 | 좋음 |
| YOLOv8m | 52 MB | 중간 | 매우 좋음 |

현재 파이프라인은 YOLOv8n을 사용하여 **경량화**를 우선시합니다.

## 📁 프로젝트 구조

```
VisionAI/
├── visionai_pipeline/           # 파이프라인 모듈
│   ├── __init__.py
│   ├── pipeline.py              # 통합 파이프라인
│   ├── detection.py             # Step 1-2: 객체+키포인트 탐지
│   ├── emotion.py               # Step 3: 감정 분석
│   ├── temporal.py              # Step 4: 시간 축 행동 인식
│   └── predictor.py             # Step 5: 행동 예측
│
├── run_pipeline.py              # CLI 인터페이스
├── pipeline_requirements.txt    # 의존성
├── PIPELINE_README.md          # 이 문서
│
└── visionai_resnet/            # 기존 ResNet 기반 (호환성 유지)
    └── ...
```

## 🐛 문제 해결

### YOLOv8 설치 오류

```bash
# ultralytics 재설치
pip uninstall ultralytics -y
pip install ultralytics --no-cache-dir
```

### CUDA Out of Memory

```bash
# CPU 사용
python run_pipeline.py --image dog.jpg --device cpu

# 또는 배치 크기 줄이기
```

### 모델 다운로드 실패

수동 다운로드:
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt
```

## 🎯 다음 단계

1. **데이터 수집**: 동물 행동 데이터셋 수집
2. **모델 학습**: 감정/행동 분류기 학습
3. **Fine-tuning**: 특정 동물 종에 맞게 조정
4. **배포**: 웹 API 또는 모바일 앱으로 배포

## 📝 라이선스

이 프로젝트는 기존 VisionAI 프로젝트의 확장입니다.

## 🤝 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.
