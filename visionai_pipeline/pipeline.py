"""
VisionAI 통합 파이프라인 - 사람 표정·자세 분석 및 행동 예측

5단계:
1. Object Detection (YOLOv8-pose, 사람 전용)
2. Keypoint Detection (YOLOv8-pose) - 신체 키포인트
3. Emotion/Pose Analysis - 표정·자세 (OpenFace 2.0 AU)
4. Temporal Action Recognition - 시간축 행동 인식
5. Behavior Prediction - 이후 행동 예측
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import os
import time
import numpy as np
from PIL import Image

from .detection import ObjectDetector, Detection
from .emotion import EmotionAnalyzer, EmotionResult
from .temporal import TemporalAnalyzer, ActionResult
from .predictor import BehaviorPredictor, PredictionResult


@dataclass
class PipelineResult:
    """파이프라인 전체 결과"""
    # Step 1: Detection
    detections: List[Dict[str, Any]]
    
    # Step 2 & 3: Emotion/Pose per detection
    emotions: List[Dict[str, Any]]
    
    # Step 4: Temporal action
    action: Optional[Dict[str, Any]]
    
    # Step 5: Behavior prediction
    prediction: Optional[Dict[str, Any]]
    
    # 메타데이터
    timestamp: float
    processing_time: float
    emotion_backend: Optional[str] = None  # 감정/자세 분석 백엔드 표시명 (OpenFace 2.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'detections': self.detections,
            'emotions': self.emotions,
            'action': self.action,
            'prediction': self.prediction,
            'timestamp': self.timestamp,
            'processing_time': self.processing_time,
            'emotion_backend': self.emotion_backend,
        }


class VisionAIPipeline:
    """
    사람 표정·자세 분석 및 이후 행동 예측 파이프라인.
    
    Usage:
        pipeline = VisionAIPipeline(device='cuda')
        
        # 단일 이미지 분석
        result = pipeline.process_image(image)
        
        # 비디오 스트림 분석
        for frame in video:
            result = pipeline.process_frame(frame, timestamp)
    """
    
    def __init__(
        self,
        device: str = "auto",
        enable_emotion: bool = True,
        enable_temporal: bool = True,
        enable_prediction: bool = True,
        emotion_model_path: Optional[str] = None,
        temporal_model_path: Optional[str] = None,
        prediction_model_path: Optional[str] = None,
        emotion_backend: Optional[str] = None
    ):
        """
        Args:
            device: 디바이스 ('auto', 'cpu', 'cuda', 'mps')
            enable_emotion: 감정 분석 활성화
            enable_temporal: 시간 축 행동 인식 활성화
            enable_prediction: 행동 예측 활성화
            emotion_model_path: 감정 분석 모델 경로
            temporal_model_path: 시간 축 모델 경로
            prediction_model_path: 예측 모델 경로
            emotion_backend: (호환용, 무시) 이제 OpenFace 2.0만 사용
        """
        print("=" * 60)
        print("VisionAI Pipeline 초기화 중...")
        print("=" * 60)
        
        self.device = device
        self.enable_emotion = enable_emotion
        self.enable_temporal = enable_temporal
        self.enable_prediction = enable_prediction
        
        # Step 1 & 2: Object Detection + Keypoint Detection
        print("\n[1/4] 객체 탐지 모델 로딩 (YOLOv8n-pose, 사람 전용)...")
        self.detector = ObjectDetector(device=device)
        
        # Step 3: Emotion & Pose Analysis (OpenFace 2.0 AU 기반)
        if enable_emotion:
            print("\n[2/4] 표정/자세 분석 모델 로딩 (OpenFace 2.0, Action Units)...")
            self.emotion_analyzer = EmotionAnalyzer(
                model_path=emotion_model_path,
                device=device,
            )
        else:
            self.emotion_analyzer = None
        
        # Step 4: Temporal Action Recognition
        if enable_temporal:
            print("\n[3/4] 시간 축 행동 인식 초기화...")
            self.temporal_analyzer = TemporalAnalyzer(
                model_path=temporal_model_path,
                device=device
            )
        else:
            self.temporal_analyzer = None
        
        # Step 5: Behavior Prediction
        if enable_prediction:
            print("\n[4/4] 행동 예측 모델 초기화...")
            self.predictor = BehaviorPredictor(
                model_path=prediction_model_path,
                device=device
            )
        else:
            self.predictor = None
        
        print("\n" + "=" * 60)
        print("✓ VisionAI Pipeline 초기화 완료!")
        print("=" * 60)
    
    def process_image(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.5
    ) -> PipelineResult:
        """
        단일 이미지 처리 (timestamp는 현재 시각 사용)
        
        Args:
            image: RGB 이미지 (H, W, 3)
            conf_threshold: 탐지 신뢰도 임계값
            
        Returns:
            파이프라인 결과
        """
        timestamp = time.time()
        return self.process_frame(image, timestamp, conf_threshold)
    
    def process_frame(
        self,
        image: np.ndarray,
        timestamp: float,
        conf_threshold: float = 0.5
    ) -> PipelineResult:
        """
        비디오 프레임 처리
        
        Args:
            image: RGB 이미지 (H, W, 3)
            timestamp: 프레임 타임스탬프 (초)
            conf_threshold: 탐지 신뢰도 임계값
            
        Returns:
            파이프라인 결과
        """
        start_time = time.time()
        
        # Step 1 & 2: Object Detection + Keypoint Detection (사람 탐지)
        detections = self.detector.detect_persons(image, conf_threshold)
        
        detection_dicts = []
        emotion_dicts = []
        
        for det in detections:
            det_dict = {
                'class_id': det.class_id,
                'class_name': det.class_name,
                'confidence': det.confidence,
                'bbox': det.bbox,
                'has_keypoints': det.keypoints is not None
            }
            detection_dicts.append(det_dict)
            
            # Step 3: Emotion & Pose Analysis (OpenFace 2.0 AU)
            if self.enable_emotion and self.emotion_analyzer:
                emotion_result = self.emotion_analyzer.analyze(
                    image, det.bbox, species="person"
                )
                emotion_dict = {
                    'class_name': det.class_name,
                    'emotion': emotion_result.emotion,
                    'emotion_confidence': emotion_result.emotion_confidence,
                    'pose': emotion_result.pose,
                    'pose_confidence': emotion_result.pose_confidence,
                    'combined_state': emotion_result.combined_state
                }
                emotion_dicts.append(emotion_dict)
                
                # Step 4: Temporal 분석을 위한 특징 추가
                if self.enable_temporal and self.temporal_analyzer:
                    self.temporal_analyzer.add_frame(
                        timestamp=timestamp,
                        emotion=emotion_result.emotion,
                        pose=emotion_result.pose,
                        keypoints=det.keypoints,
                        bbox=det.bbox
                    )
        
        # Step 4: Temporal Action Recognition
        action_dict = None
        if self.enable_temporal and self.temporal_analyzer:
            action_result = self.temporal_analyzer.analyze()
            if action_result:
                action_dict = {
                    'action': action_result.action,
                    'confidence': action_result.confidence,
                    'duration': action_result.duration,
                    'motion_intensity': action_result.motion_intensity
                }
                
                # Step 5: Behavior Prediction
                if self.enable_prediction and self.predictor:
                    self.predictor.add_action(action_result.action)
        
        # Step 5: Behavior Prediction
        prediction_dict = None
        if self.enable_prediction and self.predictor:
            prediction_result = self.predictor.predict()
            if prediction_result:
                prediction_dict = {
                    'predicted_action': prediction_result.predicted_action,
                    'confidence': prediction_result.confidence,
                    'time_horizon': prediction_result.time_horizon,
                    'alternative_actions': prediction_result.alternative_actions
                }
        
        processing_time = time.time() - start_time
        emotion_backend = None
        if self.emotion_analyzer is not None:
            emotion_backend = getattr(
                self.emotion_analyzer, 'emotion_backend_name',
                getattr(self.emotion_analyzer, '_emotion_backend_name', None)
            )
        
        return PipelineResult(
            detections=detection_dicts,
            emotions=emotion_dicts,
            action=action_dict,
            prediction=prediction_dict,
            timestamp=timestamp,
            processing_time=processing_time,
            emotion_backend=emotion_backend,
        )
    
    def visualize(
        self,
        image: np.ndarray,
        result: PipelineResult
    ) -> np.ndarray:
        """
        결과 시각화
        
        Args:
            image: 원본 이미지
            result: 파이프라인 결과
            
        Returns:
            시각화된 이미지
        """
        """
        NOTE:
        - 웹 서버 환경에서 opencv-python(cv2)이 없을 수 있어, 시각화는 PIL 기반으로 구현합니다.
        - 입력/출력은 RGB numpy array를 유지합니다.
        """

        from PIL import Image as PILImage
        from PIL import ImageDraw

        img = PILImage.fromarray(image.copy())
        draw = ImageDraw.Draw(img)

        # Detection + Emotion 시각화
        for i, det_dict in enumerate(result.detections):
            x1, y1, x2, y2 = det_dict["bbox"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 바운딩 박스
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)

            # 라벨 구성
            label_parts = [str(det_dict.get("class_name", ""))]
            if i < len(result.emotions):
                emo = result.emotions[i]
                label_parts.append(f"{emo.get('emotion', '-')}/{emo.get('pose', '-')}")
            label = " | ".join([p for p in label_parts if p])

            if label:
                # 텍스트 박스 계산
                try:
                    bbox = draw.textbbox((0, 0), label)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                except Exception:
                    # Pillow 구버전 fallback
                    text_w, text_h = draw.textsize(label)

                pad = 3
                tx1 = x1
                ty1 = max(0, y1 - text_h - pad * 2)
                tx2 = x1 + text_w + pad * 2
                ty2 = ty1 + text_h + pad * 2

                # 배경 + 텍스트
                draw.rectangle([tx1, ty1, tx2, ty2], fill=(0, 255, 0))
                draw.text((tx1 + pad, ty1 + pad), label, fill=(0, 0, 0))

        # 상단 오버레이 (Action / Prediction)
        overlay_y = 8
        overlay_lines: List[str] = []
        if result.action:
            overlay_lines.append(
                f"Action: {result.action.get('action', '-')}"
                f" ({float(result.action.get('confidence', 0.0)):.2f})"
            )
        if result.prediction:
            overlay_lines.append(
                f"Next: {result.prediction.get('predicted_action', '-')}"
                f" ({float(result.prediction.get('confidence', 0.0)):.2f})"
            )

        if overlay_lines:
            text = "  ".join(overlay_lines)
            try:
                bbox = draw.textbbox((0, 0), text)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except Exception:
                text_w, text_h = draw.textsize(text)

            pad = 4
            draw.rectangle(
                [8, overlay_y, 8 + text_w + pad * 2, overlay_y + text_h + pad * 2],
                fill=(0, 0, 0),
            )
            draw.text((8 + pad, overlay_y + pad), text, fill=(255, 255, 255))

        return np.array(img)
    
    def reset(self):
        """파이프라인 상태 초기화 (새 비디오 시작 시)"""
        if self.temporal_analyzer:
            self.temporal_analyzer.reset()
        if self.predictor:
            self.predictor.reset()
