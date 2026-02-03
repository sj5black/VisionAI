#!/usr/bin/env python3
"""
VisionAI Pipeline Benchmark

각 단계별 성능 벤치마크
"""

import sys
from pathlib import Path
import time
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from visionai_pipeline import VisionAIPipeline
from visionai_pipeline.detection import ObjectDetector
from visionai_pipeline.emotion import EmotionAnalyzer


def create_dummy_image(size=(640, 480)):
    """더미 이미지 생성"""
    return np.random.randint(0, 255, (*size, 3), dtype=np.uint8)


def benchmark_detection(num_runs=100):
    """객체 탐지 벤치마크"""
    print("\n" + "=" * 60)
    print("Step 1-2: Object + Keypoint Detection (YOLOv8n)")
    print("=" * 60)
    
    detector = ObjectDetector(device='auto')
    image = create_dummy_image()
    
    # Warm-up
    for _ in range(5):
        detector.detect(image)
    
    # Benchmark
    times = []
    for i in range(num_runs):
        start = time.time()
        detections = detector.detect(image)
        elapsed = time.time() - start
        times.append(elapsed)
        
        if (i + 1) % 10 == 0:
            print(f"  진행: {i+1}/{num_runs}", end='\r')
    
    times = np.array(times) * 1000  # ms로 변환
    
    print(f"\n결과 (n={num_runs}):")
    print(f"  평균: {times.mean():.2f} ms")
    print(f"  중앙값: {np.median(times):.2f} ms")
    print(f"  최소: {times.min():.2f} ms")
    print(f"  최대: {times.max():.2f} ms")
    print(f"  FPS: {1000/times.mean():.1f}")


def benchmark_emotion(num_runs=100):
    """감정 분석 벤치마크"""
    print("\n" + "=" * 60)
    print("Step 3: Emotion & Pose Analysis (MobileNetV3)")
    print("=" * 60)
    
    analyzer = EmotionAnalyzer(device='auto')
    image = create_dummy_image()
    bbox = (100, 100, 400, 400)  # 더미 bbox
    
    # Warm-up
    for _ in range(5):
        analyzer.analyze(image, bbox)
    
    # Benchmark
    times = []
    for i in range(num_runs):
        start = time.time()
        result = analyzer.analyze(image, bbox)
        elapsed = time.time() - start
        times.append(elapsed)
        
        if (i + 1) % 10 == 0:
            print(f"  진행: {i+1}/{num_runs}", end='\r')
    
    times = np.array(times) * 1000
    
    print(f"\n결과 (n={num_runs}):")
    print(f"  평균: {times.mean():.2f} ms")
    print(f"  중앙값: {np.median(times):.2f} ms")
    print(f"  최소: {times.min():.2f} ms")
    print(f"  최대: {times.max():.2f} ms")
    print(f"  FPS: {1000/times.mean():.1f}")


def benchmark_pipeline(num_runs=50):
    """전체 파이프라인 벤치마크"""
    print("\n" + "=" * 60)
    print("Full Pipeline (All Steps)")
    print("=" * 60)
    
    pipeline = VisionAIPipeline(
        device='auto',
        enable_emotion=True,
        enable_temporal=True,
        enable_prediction=True
    )
    
    image = create_dummy_image()
    
    # Warm-up
    for i in range(5):
        pipeline.process_image(image)
    
    # Benchmark
    times = []
    for i in range(num_runs):
        start = time.time()
        result = pipeline.process_image(image)
        elapsed = time.time() - start
        times.append(elapsed)
        
        if (i + 1) % 5 == 0:
            print(f"  진행: {i+1}/{num_runs}", end='\r')
    
    times = np.array(times) * 1000
    
    print(f"\n결과 (n={num_runs}):")
    print(f"  평균: {times.mean():.2f} ms")
    print(f"  중앙값: {np.median(times):.2f} ms")
    print(f"  최소: {times.min():.2f} ms")
    print(f"  최대: {times.max():.2f} ms")
    print(f"  FPS: {1000/times.mean():.1f}")


def benchmark_lightweight():
    """경량 모드 벤치마크"""
    print("\n" + "=" * 60)
    print("Lightweight Mode (Detection + Emotion only)")
    print("=" * 60)
    
    pipeline = VisionAIPipeline(
        device='auto',
        enable_emotion=True,
        enable_temporal=False,
        enable_prediction=False
    )
    
    image = create_dummy_image()
    
    # Warm-up
    for _ in range(5):
        pipeline.process_image(image)
    
    # Benchmark
    num_runs = 50
    times = []
    for i in range(num_runs):
        start = time.time()
        result = pipeline.process_image(image)
        elapsed = time.time() - start
        times.append(elapsed)
        
        if (i + 1) % 5 == 0:
            print(f"  진행: {i+1}/{num_runs}", end='\r')
    
    times = np.array(times) * 1000
    
    print(f"\n결과 (n={num_runs}):")
    print(f"  평균: {times.mean():.2f} ms")
    print(f"  중앙값: {np.median(times):.2f} ms")
    print(f"  최소: {times.min():.2f} ms")
    print(f"  최대: {times.max():.2f} ms")
    print(f"  FPS: {1000/times.mean():.1f}")


def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║            VisionAI Pipeline - Benchmark                   ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    import torch
    print(f"\n시스템 정보:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA 버전: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n벤치마크를 시작합니다...")
    print("(첫 실행 시 모델 다운로드가 진행될 수 있습니다)\n")
    
    # 각 단계별 벤치마크
    benchmark_detection(num_runs=100)
    benchmark_emotion(num_runs=100)
    
    # 파이프라인 벤치마크
    benchmark_lightweight()
    benchmark_pipeline(num_runs=50)
    
    print("\n" + "=" * 60)
    print("벤치마크 완료!")
    print("=" * 60)


if __name__ == '__main__':
    main()
