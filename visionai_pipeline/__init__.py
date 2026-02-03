"""
VisionAI Pipeline - 경량화된 동물 행동 예측 파이프라인

5단계 파이프라인:
1. Object Detection (YOLOv8)
2. Keypoint Detection (YOLOv8-pose)
3. Emotion/Pose Analysis (경량 분류기)
4. Temporal Action Recognition (Temporal pooling)
5. Behavior Prediction (LSTM)
"""

from .pipeline import VisionAIPipeline
from .detection import ObjectDetector
from .emotion import EmotionAnalyzer
from .temporal import TemporalAnalyzer
from .predictor import BehaviorPredictor

__all__ = [
    'VisionAIPipeline',
    'ObjectDetector',
    'EmotionAnalyzer',
    'TemporalAnalyzer',
    'BehaviorPredictor',
]
