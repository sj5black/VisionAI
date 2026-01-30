# VisionAI - ResNet Image Analyzer

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

## 웹사이트 기능: 이미지 업로드 → 포함 객체/동물 보여주기

이미지 “분류(top-k)” 대신, 이미지 안의 여러 객체를 찾아주는 **객체 탐지(Object Detection)** 를 웹에서 바로 쓸 수 있게 구성했습니다.

## (추가) 동물일 경우: 행동/표정 → 다음 행동/상태(추정)

탐지된 객체가 `dog/cat/...` 같은 동물로 판단되면, 바운딩박스 영역을 크롭해서
**행동(예: sitting, running)** 과 **표정/정서(예: relaxed, fearful)** 를 추가로 “추정”하고,
그 조합으로 **다음 행동 패턴/상태(추정)** 를 반환합니다.

> 주의: 이 기능은 OpenCLIP 기반 **zero-shot 추정**이라 정확하지 않을 수 있으며,
> **수의학적/의학적 진단이 아닙니다.**

### 실행 방법

```bash
cd /home/teddy/VisionAI
conda activate vision
# (필요 시) 웹 의존성만 설치:
python -m pip install fastapi uvicorn python-multipart jinja2
# (선택) 동물 행동/표정 분석 기능까지 사용:
python -m pip install open_clip_torch
uvicorn webapp.main:app --host 0.0.0.0 --port 8001
```

브라우저에서 `http://localhost:8001` 접속 후 이미지를 업로드하면,
탐지된 `object_types`(라벨 목록)과 박스가 그려진 결과 이미지를 확인할 수 있습니다.

### 옵션

- `VISIONAI_DEVICE`: 강제로 디바이스 지정 (예: `cpu`, `cuda`, `cuda:0`)
- `VISIONAI_ENABLE_ANIMAL_INSIGHTS`: `0`으로 설정 시 동물 행동/표정 분석 비활성화

```bash
VISIONAI_DEVICE=cpu uvicorn webapp.main:app --host 0.0.0.0 --port 8001
```
