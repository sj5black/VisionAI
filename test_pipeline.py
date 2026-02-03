#!/usr/bin/env python3
"""
VisionAI Pipeline 간단 테스트

설치 확인 및 기본 동작 테스트
"""

import sys
import numpy as np
from pathlib import Path

print("=" * 60)
print("VisionAI Pipeline 테스트")
print("=" * 60)

# 1. 의존성 체크
print("\n[1/5] 의존성 체크...")
required_packages = [
    ('torch', 'PyTorch'),
    ('torchvision', 'TorchVision'),
    ('PIL', 'Pillow'),
    ('numpy', 'NumPy'),
    ('cv2', 'OpenCV')
]

missing = []
for package, name in required_packages:
    try:
        __import__(package)
        print(f"  ✓ {name}")
    except ImportError:
        print(f"  ✗ {name} (없음)")
        missing.append(name)

if missing:
    print(f"\n⚠ 누락된 패키지: {', '.join(missing)}")
    print("설치: pip install -r pipeline_requirements.txt")
    sys.exit(1)

# 2. YOLO 체크
print("\n[2/5] YOLO 패키지 체크...")
try:
    from ultralytics import YOLO
    print("  ✓ ultralytics")
except ImportError:
    print("  ✗ ultralytics (없음)")
    print("설치: pip install ultralytics")
    sys.exit(1)

# 3. 파이프라인 모듈 체크
print("\n[3/5] 파이프라인 모듈 체크...")
try:
    from visionai_pipeline import VisionAIPipeline
    from visionai_pipeline.detection import ObjectDetector
    from visionai_pipeline.emotion import EmotionAnalyzer
    from visionai_pipeline.temporal import TemporalAnalyzer
    from visionai_pipeline.predictor import BehaviorPredictor
    print("  ✓ 모든 모듈 임포트 성공")
except ImportError as e:
    print(f"  ✗ 모듈 임포트 실패: {e}")
    sys.exit(1)

# 4. 디바이스 체크
print("\n[4/5] 디바이스 체크...")
import torch

if torch.cuda.is_available():
    device = "cuda"
    print(f"  ✓ CUDA 사용 가능 ({torch.cuda.get_device_name(0)})")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
    print("  ✓ MPS (Apple Silicon) 사용 가능")
else:
    device = "cpu"
    print("  ✓ CPU 사용 (GPU 없음)")

# 5. 파이프라인 초기화 테스트
print("\n[5/5] 파이프라인 초기화 테스트...")
print("(첫 실행 시 YOLOv8 모델을 다운로드합니다. 약 9MB)")
print()

try:
    # 경량 모드로 테스트
    pipeline = VisionAIPipeline(
        device=device,
        enable_emotion=True,
        enable_temporal=False,
        enable_prediction=False
    )
    print("\n  ✓ 파이프라인 초기화 성공!")
except Exception as e:
    print(f"\n  ✗ 파이프라인 초기화 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 6. 더미 이미지로 테스트
print("\n[6/6] 더미 이미지 테스트...")
try:
    # 640x480 랜덤 이미지
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    result = pipeline.process_image(dummy_image)
    
    print(f"  ✓ 이미지 처리 성공!")
    print(f"    - 처리 시간: {result.processing_time:.3f}초")
    print(f"    - 탐지 객체: {len(result.detections)}개")
    print(f"    - 감정 분석: {len(result.emotions)}개")
    
except Exception as e:
    print(f"  ✗ 이미지 처리 실패: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 성공!
print("\n" + "=" * 60)
print("✓ 모든 테스트 통과!")
print("=" * 60)
print("\n다음 단계:")
print("  1. 실제 이미지로 테스트:")
print("     python run_pipeline.py --image your_image.jpg --output result.jpg")
print()
print("  2. 비디오로 테스트:")
print("     python run_pipeline.py --video your_video.mp4 --output result.mp4")
print()
print("  3. 예제 실행:")
print("     python examples/quick_start.py")
print()
print("  4. 웹 API 실행:")
print("     python webapp/pipeline_api.py")
print()
print("자세한 내용은 PIPELINE_README.md를 참고하세요.")
print("=" * 60)
