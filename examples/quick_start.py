#!/usr/bin/env python3
"""
VisionAI Pipeline Quick Start Example

간단한 사용 예제
"""

import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from visionai_pipeline import VisionAIPipeline
import numpy as np
from PIL import Image
import json


def example_single_image():
    """단일 이미지 분석 예제"""
    print("=" * 60)
    print("예제 1: 단일 이미지 분석")
    print("=" * 60)
    
    # 파이프라인 초기화
    pipeline = VisionAIPipeline(
        device='auto',
        enable_emotion=True,
        enable_temporal=False,  # 단일 이미지라 불필요
        enable_prediction=False
    )
    
    # 테스트 이미지 경로 (사용자가 제공)
    image_path = input("\n이미지 경로를 입력하세요 (Enter로 건너뛰기): ").strip()
    
    if not image_path or not Path(image_path).exists():
        print("⚠ 유효한 이미지 경로를 제공하지 않았습니다. 예제를 건너뜁니다.")
        return
    
    # 이미지 로드
    image = np.array(Image.open(image_path).convert('RGB'))
    
    # 분석
    result = pipeline.process_image(image)
    
    # 결과 출력
    print("\n분석 결과:")
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    
    # 간단한 요약
    print("\n요약:")
    print(f"- 탐지된 객체: {len(result.detections)}개")
    for i, det in enumerate(result.detections):
        print(f"  [{i+1}] {det['class_name']} (신뢰도: {det['confidence']:.2%})")
    
    if result.emotions:
        print(f"\n- 감정/자세:")
        for i, emo in enumerate(result.emotions):
            print(f"  [{i+1}] {emo['emotion']} / {emo['pose']} → {emo['combined_state']}")


def example_video_stream():
    """비디오 스트림 분석 예제"""
    print("\n" + "=" * 60)
    print("예제 2: 비디오 스트림 분석 (웹캠 또는 파일)")
    print("=" * 60)
    
    # 파이프라인 초기화 (전체 기능)
    pipeline = VisionAIPipeline(
        device='auto',
        enable_emotion=True,
        enable_temporal=True,
        enable_prediction=True
    )
    
    # 입력 소스
    source = input("\n비디오 경로 (또는 '0'으로 웹캠 사용, Enter로 건너뛰기): ").strip()
    
    if not source:
        print("⚠ 비디오 소스를 제공하지 않았습니다. 예제를 건너뜁니다.")
        return
    
    try:
        import cv2
    except ImportError:
        print("⚠ opencv-python이 필요합니다: pip install opencv-python")
        return
    
    # 비디오 캡처
    cap_source = 0 if source == '0' else source
    cap = cv2.VideoCapture(cap_source)
    
    if not cap.isOpened():
        print(f"⚠ 비디오 열기 실패: {source}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    
    print("\n분석 중... (q를 눌러 종료)")
    
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            
            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # 타임스탬프
            timestamp = frame_idx / fps
            
            # 분석 (5 프레임마다)
            if frame_idx % 5 == 0:
                result = pipeline.process_frame(frame_rgb, timestamp)
                
                # 시각화
                vis_rgb = pipeline.visualize(frame_rgb, result)
                vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
                
                cv2.imshow('VisionAI Pipeline', vis_bgr)
                
                # 콘솔에도 출력
                if result.prediction:
                    pred = result.prediction
                    print(f"\r[{timestamp:.1f}s] 다음 행동: {pred['predicted_action']} "
                          f"({pred['confidence']:.2%})    ", end='')
            else:
                cv2.imshow('VisionAI Pipeline', frame_bgr)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_idx += 1
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n\n✓ 비디오 분석 완료")


def example_custom_config():
    """커스텀 설정 예제"""
    print("\n" + "=" * 60)
    print("예제 3: 커스텀 설정으로 파이프라인 구성")
    print("=" * 60)
    
    # 경량 모드 (탐지 + 감정만)
    print("\n[경량 모드] 탐지 + 감정 분석만")
    pipeline_light = VisionAIPipeline(
        device='cpu',  # CPU만 사용
        enable_emotion=True,
        enable_temporal=False,
        enable_prediction=False
    )
    print("✓ 경량 파이프라인 초기화 완료")
    
    # 전체 모드
    print("\n[전체 모드] 모든 기능 활성화")
    pipeline_full = VisionAIPipeline(
        device='auto',  # GPU 자동 감지
        enable_emotion=True,
        enable_temporal=True,
        enable_prediction=True
    )
    print("✓ 전체 파이프라인 초기화 완료")
    
    print("\n사용 시나리오:")
    print("- 경량 모드: 실시간 처리, 모바일, 저사양 환경")
    print("- 전체 모드: 고정밀 분석, 연구, 오프라인 처리")


def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║        VisionAI Pipeline - Quick Start Examples           ║
║                                                            ║
║  동물의 감정, 자세, 행동을 분석하고 예측합니다            ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # 예제 선택
    print("\n실행할 예제를 선택하세요:")
    print("1. 단일 이미지 분석")
    print("2. 비디오 스트림 분석")
    print("3. 커스텀 설정 (데모만)")
    print("4. 전체 실행")
    
    choice = input("\n선택 (1-4): ").strip()
    
    if choice == '1':
        example_single_image()
    elif choice == '2':
        example_video_stream()
    elif choice == '3':
        example_custom_config()
    elif choice == '4':
        example_single_image()
        example_video_stream()
        example_custom_config()
    else:
        print("유효하지 않은 선택입니다.")
    
    print("\n" + "=" * 60)
    print("예제 완료! 더 많은 정보는 PIPELINE_README.md를 참고하세요.")
    print("=" * 60)


if __name__ == '__main__':
    main()
