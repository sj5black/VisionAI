#!/usr/bin/env python3
"""
VisionAI Pipeline CLI

단일 이미지 또는 비디오에 대한 5단계 파이프라인 실행
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

from visionai_pipeline import VisionAIPipeline


def load_image(image_path: str) -> np.ndarray:
    """이미지 로드"""
    img = Image.open(image_path).convert('RGB')
    return np.array(img)


def process_image(
    pipeline: VisionAIPipeline,
    image_path: str,
    output_path: str = None,
    visualize: bool = True
):
    """단일 이미지 처리"""
    print(f"\n처리 중: {image_path}")
    
    # 이미지 로드
    image = load_image(image_path)
    
    # 파이프라인 실행
    result = pipeline.process_image(image)
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("분석 결과")
    print("=" * 60)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    
    # 시각화
    if visualize:
        vis_image = pipeline.visualize(image, result)
        
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # RGB -> BGR for cv2
            vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_file), vis_bgr)
            print(f"\n✓ 시각화 결과 저장: {output_file}")
        else:
            # 화면 표시
            cv2.imshow('VisionAI Result', cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            print("\n이미지를 표시합니다. 아무 키나 눌러 종료...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def process_video(
    pipeline: VisionAIPipeline,
    video_path: str,
    output_path: str = None,
    sample_fps: float = 5.0,
    visualize: bool = True
):
    """비디오 처리"""
    print(f"\n처리 중: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"비디오 열기 실패: {video_path}")
    
    # 비디오 정보
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"비디오 정보: {width}x{height}, {video_fps:.2f} FPS, {total_frames} 프레임")
    print(f"샘플링: {sample_fps:.2f} FPS로 분석")
    
    # 출력 비디오 writer
    writer = None
    if output_path and visualize:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(output_file),
            fourcc,
            sample_fps,
            (width, height)
        )
    
    # 샘플링 간격
    frame_interval = int(video_fps / sample_fps) if sample_fps < video_fps else 1
    
    frame_idx = 0
    results = []
    
    pipeline.reset()  # 파이프라인 초기화
    
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            
            # 샘플링
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue
            
            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # 타임스탬프 계산
            timestamp = frame_idx / video_fps
            
            # 파이프라인 실행
            result = pipeline.process_frame(frame_rgb, timestamp)
            results.append(result.to_dict())
            
            # 진행 상황
            progress = (frame_idx + 1) / total_frames * 100
            print(f"\r진행: {progress:.1f}% ({frame_idx+1}/{total_frames})", end='')
            
            # 시각화
            if visualize:
                vis_image = pipeline.visualize(frame_rgb, result)
                vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
                
                if writer:
                    writer.write(vis_bgr)
                else:
                    cv2.imshow('VisionAI Video', vis_bgr)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n사용자가 중단했습니다.")
                        break
            
            frame_idx += 1
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    
    print(f"\n✓ 총 {len(results)} 프레임 분석 완료")
    
    # 결과 JSON 저장
    if output_path:
        json_path = Path(output_path).with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ 결과 JSON 저장: {json_path}")
    
    # 요약 통계
    print("\n" + "=" * 60)
    print("분석 요약")
    print("=" * 60)
    
    # 탐지된 동물 수
    total_detections = sum(len(r['detections']) for r in results)
    print(f"총 탐지: {total_detections}개")
    
    # 감정 분포
    if results and results[0]['emotions']:
        emotions = [e['emotion'] for r in results for e in r['emotions']]
        emotion_counts = {}
        for e in emotions:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        print(f"\n감정 분포:")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
            print(f"  {emotion}: {count}")
    
    # 행동 분포
    actions = [r['action']['action'] for r in results if r['action']]
    if actions:
        action_counts = {}
        for a in actions:
            action_counts[a] = action_counts.get(a, 0) + 1
        print(f"\n행동 분포:")
        for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            print(f"  {action}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="VisionAI Pipeline - 동물 행동 예측",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 단일 이미지 분석
  python run_pipeline.py --image dog.jpg --output result.jpg
  
  # 비디오 분석 (5 FPS 샘플링)
  python run_pipeline.py --video cat_video.mp4 --output result.mp4 --fps 5
  
  # 경량화 모드 (감정/시간축/예측 비활성화)
  python run_pipeline.py --image dog.jpg --no-emotion --no-temporal --no-prediction
        """
    )
    
    # 입력
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str, help='입력 이미지 경로')
    input_group.add_argument('--video', type=str, help='입력 비디오 경로')
    
    # 출력
    parser.add_argument('--output', type=str, help='출력 파일 경로')
    parser.add_argument('--no-visualize', action='store_true', help='시각화 비활성화')
    
    # 파이프라인 옵션
    parser.add_argument('--device', type=str, default='auto',
                        help='디바이스 (auto/cpu/cuda/mps)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='탐지 신뢰도 임계값 (기본: 0.5)')
    parser.add_argument('--fps', type=float, default=5.0,
                        help='비디오 샘플링 FPS (기본: 5.0)')
    
    # 모듈 활성화/비활성화
    parser.add_argument('--no-emotion', action='store_true',
                        help='감정 분석 비활성화')
    parser.add_argument('--no-temporal', action='store_true',
                        help='시간 축 행동 인식 비활성화')
    parser.add_argument('--no-prediction', action='store_true',
                        help='행동 예측 비활성화')
    
    # 모델 경로
    parser.add_argument('--emotion-model', type=str,
                        help='감정 분석 모델 경로')
    parser.add_argument('--temporal-model', type=str,
                        help='시간 축 모델 경로')
    parser.add_argument('--prediction-model', type=str,
                        help='예측 모델 경로')
    
    args = parser.parse_args()
    
    # 파이프라인 초기화
    pipeline = VisionAIPipeline(
        device=args.device,
        enable_emotion=not args.no_emotion,
        enable_temporal=not args.no_temporal,
        enable_prediction=not args.no_prediction,
        emotion_model_path=args.emotion_model,
        temporal_model_path=args.temporal_model,
        prediction_model_path=args.prediction_model
    )
    
    # 처리
    if args.image:
        process_image(
            pipeline,
            args.image,
            output_path=args.output,
            visualize=not args.no_visualize
        )
    elif args.video:
        process_video(
            pipeline,
            args.video,
            output_path=args.output,
            sample_fps=args.fps,
            visualize=not args.no_visualize
        )


if __name__ == '__main__':
    main()
