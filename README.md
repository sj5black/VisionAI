# VisionAI - 사람 표정·자세 분석 및 행동 예측 시스템

**사람의 표정, 자세를 분석하고 이후 행동을 예측하는 경량화된 AI 파이프라인**

## 🆕 VisionAI Pipeline (NEW!)

5단계 AI 파이프라인으로 사람의 표정·자세·행동을 분석하고 다음 행동을 예측:

1. **객체 탐지** - YOLOv8n, 사람(person) 탐지
2. **키포인트 탐지** - 신체 부위 17개 포인트 (YOLOv8-pose)
3. **표정/자세 분석** - OpenCLIP 또는 Swin (감정·자세)
4. **행동 인식** - 시간 흐름 기반
5. **행동 예측** - 이후 행동 예측

**총 모델 크기**: ~9-10 MB (경량)

### 🚀 빠른 시작

```bash
# 설치
pip install -r pipeline_requirements.txt

# 이미지 분석 (사람 표정·자세·행동 예측)
python run_pipeline.py --image person.jpg --output result.jpg

# 비디오 분석
python run_pipeline.py --video video.mp4 --output result.mp4 --fps 5
```

**자세한 내용**: [QUICKSTART.md](QUICKSTART.md) | [PIPELINE_README.md](PIPELINE_README.md)

---

## ResNet Image Analyzer (기존 기능)

`ResNet.md`에 있는 `Block`, `CustomResNet` 구조를 참고해 **ResNet 기반 이미지 분석(추론)** 로직을 실행 가능한 형태로 정리했습니다.

## 설치

```bash
cd /home/teddy/VisionAI
pip install -r requirements.txt
```

## 실행

### 1) torchvision(pretrained ImageNet) ResNet으로 분석 (권장)

```bash
python analyze_resnet.py /path/to/image.jpg --backend torchvision --arch resnet50 --topk 5
```

폴더 통째로:

```bash
python analyze_resnet.py /path/to/images_dir --backend torchvision --arch resnet50 --topk 5
```

feature 벡터까지 (avgpool 출력):

```bash
python analyze_resnet.py /path/to/image.jpg --backend torchvision --arch resnet50 --feature
```

### 2) `ResNet.md` 구조(CustomResNet-18)로 분석

> 커스텀 모델은 기본적으로 랜덤 초기화라 “의미 있는 분류”를 하려면 학습된 체크포인트가 필요합니다.

```bash
python analyze_resnet.py /path/to/image.jpg --backend custom --custom-num-classes 10 --topk 5
```

학습된 체크포인트 로드:

```bash
python analyze_resnet.py /path/to/image.jpg --backend custom --checkpoint /path/to/model.pth --custom-num-classes 10
```

## 객체 탐지(Object Detection): 이미지 안의 “객체 종류” 뽑기

분류(ResNet top-k)는 이미지 전체에 대한 라벨이지만, **객체 탐지**는 이미지 안의 여러 객체를 찾아서
`(라벨, 점수, 바운딩박스)`를 반환합니다. (ResNet50 백본 탐지 모델 사용)

```bash
python detect_objects.py /path/to/image.jpg --model fasterrcnn_resnet50_fpn_v2 --threshold 0.5
```

폴더 통째로:

```bash
python detect_objects.py /path/to/images_dir --model fasterrcnn_resnet50_fpn_v2 --threshold 0.5
```

박스가 그려진 결과 이미지 저장:

```bash
python detect_objects.py /path/to/image.jpg --save-vis ./outputs --threshold 0.5
```

## 출력 형식

기본 출력은 JSON이며, 각 이미지에 대해 `topk` 예측(클래스 id/라벨/확률)을 제공합니다.

## 코드 위치

- `visionai_resnet/models.py`: `Block`, `CustomResNet` (ResNet.md 기반)
- `visionai_resnet/analyzer.py`: 전처리 + 추론 + Top-K + (옵션) feature 추출
- `visionai_resnet/detector.py`: 객체 탐지(라벨/점수/박스) + (옵션) 시각화 저장
- `analyze_resnet.py`: CLI 엔트리포인트
- `detect_objects.py`: 객체 탐지 CLI 엔트리포인트

## 웹사이트 기능: 이미지 업로드 → 사람 탐지·표정·자세·행동 예측

**VisionAI Pipeline**: 이미지에서 **사람(person)** 을 탐지하고, 표정·자세를 분석한 뒤 **이후 행동을 예측**합니다.

(기존 ResNet 웹 모드에서는 이미지 안의 여러 객체를 탐지해 보여주며, 동물일 경우 동물 전용 행동/표정 추정을 추가로 사용할 수 있습니다.)

### 실행 방법

```bash
cd /home/teddy/VisionAI
conda activate vision
# (필요 시) 웹 의존성만 설치:
python -m pip install fastapi uvicorn python-multipart jinja2
# (선택) 표정/자세 분석(Pipeline) 사용:
python -m pip install open_clip_torch
uvicorn webapp.main:app --host 0.0.0.0 --port 8001
```

브라우저에서 `http://localhost:8001` 접속 후 이미지를 업로드하면,
탐지된 `object_types`(라벨 목록)과 박스가 그려진 결과 이미지를 확인할 수 있습니다.

### 옵션

- `VISIONAI_DEVICE`: 강제로 디바이스 지정 (예: `cpu`, `cuda`, `cuda:0`)
- `VISIONAI_ENABLE_ANIMAL_INSIGHTS`: `0`으로 설정 시 표정/자세 분석 비활성화

```bash
VISIONAI_DEVICE=cpu uvicorn webapp.main:app --host 0.0.0.0 --port 8001
```

---

## 📚 문서

### 파이프라인 (NEW)
- [QUICKSTART.md](QUICKSTART.md) - 5분 빠른 시작
- [PIPELINE_README.md](PIPELINE_README.md) - 전체 파이프라인 사용법
- [ARCHITECTURE.md](ARCHITECTURE.md) - 시스템 아키텍처
- [MODEL_COMPARISON.md](MODEL_COMPARISON.md) - 모델 선택 근거
- [PIPELINE_SUMMARY.md](PIPELINE_SUMMARY.md) - 완성 요약

### ResNet (기존)
- [ResNet.md](ResNet.md) - ResNet 구조 설명

---

## 🎯 주요 기능

### VisionAI Pipeline
- ✅ 객체 탐지 (개/고양이)
- ✅ 신체 부위 키포인트
- ✅ 감정 분석 (relaxed, alert, fearful, aggressive, playful)
- ✅ 자세 분석 (sitting, standing, lying, running, jumping)
- ✅ 행동 인식 (resting, walking, running, playing, etc.)
- ✅ 다음 행동 예측
- ✅ 실시간 처리 (25-30 FPS)

### ResNet Analyzer
- ✅ 이미지 분류 (ImageNet top-k)
- ✅ 객체 탐지 (Faster R-CNN)
- ✅ Feature extraction

---

## 🔧 예제

### Python에서 사용

```python
from visionai_pipeline import VisionAIPipeline
import numpy as np
from PIL import Image

# 파이프라인 초기화
pipeline = VisionAIPipeline(device='cuda')

# 이미지 로드
image = np.array(Image.open('dog.jpg'))

# 분석
result = pipeline.process_image(image)

# 결과
print(f"감정: {result.emotions[0]['emotion']}")
print(f"자세: {result.emotions[0]['pose']}")
print(f"행동: {result.action['action']}")
print(f"예측: {result.prediction['predicted_action']}")

# 시각화
vis_image = pipeline.visualize(image, result)
```

### CLI로 사용

```bash
# 이미지 분석
python run_pipeline.py --image dog.jpg --output result.jpg

# 비디오 분석 (5 FPS 샘플링)
python run_pipeline.py --video cat_video.mp4 --output result.mp4 --fps 5

# 경량 모드 (탐지만)
python run_pipeline.py --image dog.jpg --no-emotion --no-temporal --no-prediction

# GPU 지정
python run_pipeline.py --image dog.jpg --device cuda
```

### 웹 API

```bash
# API 서버 시작
python webapp/pipeline_api.py

# 브라우저에서 접속
# http://localhost:8002
```

---

## 📊 성능

| 구성 | 모델 크기 | FPS (GPU) | 정확도 |
|------|----------|-----------|--------|
| VisionAI Pipeline | ~10 MB | 25-30 | 중상 |
| 경량 모드 | ~9 MB | 30-50 | 중 |
| ResNet Analyzer | ~150 MB | 20-30 | 높음 |

---

## 🎓 프로젝트 구조

```
VisionAI/
├── visionai_pipeline/        # 파이프라인 모듈
├── visionai_resnet/          # ResNet 모듈
├── webapp/                   # 웹 인터페이스
├── examples/                 # 예제 스크립트
├── run_pipeline.py           # CLI
└── test_pipeline.py          # 테스트
```
