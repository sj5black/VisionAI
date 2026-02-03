# VisionAI Pipeline - 빠른 시작 가이드

## 🚀 5분 안에 시작하기

### 1단계: 설치 (2분)

```bash
cd /home/teddy/VisionAI

# 파이프라인 의존성 설치
pip install -r pipeline_requirements.txt
```

### 2단계: 테스트 (1분)

```bash
# 설치 확인
python test_pipeline.py
```

첫 실행 시 YOLOv8 모델이 자동으로 다운로드됩니다 (~9 MB).

### 3단계: 실행! (2분)

#### 📸 이미지 분석

```bash
python run_pipeline.py --image your_dog.jpg --output result.jpg
```

#### 🎥 비디오 분석

```bash
python run_pipeline.py --video your_video.mp4 --output result.mp4 --fps 5
```

#### 🌐 웹 API 실행

```bash
python webapp/pipeline_api.py
```

브라우저에서 `http://localhost:8002` 접속

---

## 📊 출력 예시

### 이미지 분석 결과 (JSON)

```json
{
  "detections": [
    {
      "class_name": "dog",
      "confidence": 0.92,
      "bbox": [100, 150, 400, 500]
    }
  ],
  "emotions": [
    {
      "emotion": "playful",
      "emotion_confidence": 0.85,
      "pose": "running",
      "combined_state": "playing"
    }
  ],
  "action": {
    "action": "playing",
    "confidence": 0.7
  },
  "prediction": {
    "predicted_action": "resting",
    "confidence": 0.65
  }
}
```

---

## ⚙️ 옵션

### 경량 모드 (빠름)

```bash
python run_pipeline.py --image dog.jpg --no-temporal --no-prediction
```

### GPU 사용

```bash
python run_pipeline.py --image dog.jpg --device cuda
```

### CPU만 사용

```bash
python run_pipeline.py --image dog.jpg --device cpu
```

---

## 🐍 Python 코드에서 사용

```python
from visionai_pipeline import VisionAIPipeline
import numpy as np
from PIL import Image

# 초기화
pipeline = VisionAIPipeline(device='auto')

# 이미지 로드
image = np.array(Image.open('dog.jpg'))

# 분석
result = pipeline.process_image(image)

# 결과 출력
print(f"탐지: {result.detections}")
print(f"감정: {result.emotions}")
print(f"예측: {result.prediction}")
```

---

## 📚 더 알아보기

- **전체 문서**: [PIPELINE_README.md](PIPELINE_README.md)
- **예제 코드**: [examples/quick_start.py](examples/quick_start.py)
- **벤치마크**: [examples/benchmark.py](examples/benchmark.py)

---

## ❓ 문제 해결

### "ultralytics 없음" 오류

```bash
pip install ultralytics
```

### CUDA Out of Memory

```bash
python run_pipeline.py --image dog.jpg --device cpu
```

### 모델 다운로드 실패

수동 다운로드:
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt
```

---

## ✅ 체크리스트

- [ ] Python 3.8+ 설치됨
- [ ] `pip install -r pipeline_requirements.txt` 실행
- [ ] `python test_pipeline.py` 통과
- [ ] 첫 이미지 분석 성공!

**완료되었으면 프로덕션 사용 준비 완료입니다!** 🎉
