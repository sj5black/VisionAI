"""
DeepFace 기반 이미지 표정 분석

- 이미지 전용 (영상은 pyfaceau 사용)
- 7가지 기본 감정: angry, disgust, fear, happy, sad, surprise, neutral
- VisionAI EXPRESSION_LABELS로 매핑
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .emotion_categories import get_state_from_emotion_pose


@dataclass
class DeepFaceResult:
    """DeepFace 단일 이미지 결과"""
    success: bool
    expression: str
    expression_confidence: float
    pose: str  # deepface는 pose 미지원, 기본값 "front"
    pose_confidence: float
    combined_state: str
    emotion_scores: Optional[Dict[str, float]] = None


def _deepface_to_expression(emotion_dict: Dict[str, float]) -> Tuple[str, float]:
    """
    DeepFace 감정 → VisionAI EXPRESSION_LABELS 매핑.
    
    DeepFace 감정: angry, disgust, fear, happy, sad, surprise, neutral
    VisionAI 표정: neutral, real_smile, fake_smile, focused, surprised, sad, displeased, attention
    """
    if not emotion_dict:
        return "neutral", 0.5
    
    # 가장 높은 확률의 감정 찾기
    dominant_emotion = max(emotion_dict, key=emotion_dict.get)
    confidence = emotion_dict[dominant_emotion] / 100.0  # DeepFace는 0-100 스케일
    
    # VisionAI 라벨로 매핑
    mapping = {
        "happy": "real_smile",
        "neutral": "neutral",
        "sad": "sad",
        "surprise": "surprised",
        "angry": "displeased",
        "disgust": "displeased",
        "fear": "attention",  # 긴장/주의
    }
    
    expression = mapping.get(dominant_emotion, "neutral")
    return expression, min(1.0, confidence)


def analyze_image(
    image: np.ndarray,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> DeepFaceResult:
    """
    단일 이미지에 대해 DeepFace 표정 분석.
    
    Args:
        image: RGB (H, W, 3)
        bbox: (x1, y1, x2, y2) 얼굴 bbox. None이면 전체 이미지에서 얼굴 자동 탐지.
    """
    try:
        from deepface import DeepFace
    except ImportError:
        if not getattr(analyze_image, "_warned_deepface", False):
            print("⚠ 이미지 표정 분석을 위해 deepface가 필요합니다: pip install deepface")
            analyze_image._warned_deepface = True
        return DeepFaceResult(
            success=False,
            expression="neutral",
            expression_confidence=0.0,
            pose="front",
            pose_confidence=0.8,
            combined_state="neutral",
            emotion_scores=None,
        )
    
    # ROI 추출 (bbox가 주어진 경우)
    if bbox is not None:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        h, w = image.shape[:2]
        # 패딩 추가 (얼굴 전체가 들어오도록)
        pad = 0.15
        x1 = max(0, x1 - int((x2 - x1) * pad))
        y1 = max(0, y1 - int((y2 - y1) * pad))
        x2 = min(w, x2 + int((x2 - x1) * pad))
        y2 = min(h, y2 + int((y2 - y1) * pad))
        roi = image[y1:y2, x1:x2]
    else:
        roi = image
    
    if roi.size == 0:
        return DeepFaceResult(
            success=False,
            expression="neutral",
            expression_confidence=0.0,
            pose="front",
            pose_confidence=0.8,
            combined_state="neutral",
        )
    
    try:
        # DeepFace 분석 (감정만)
        # enforce_detection=False: 얼굴이 없어도 에러 대신 전체 이미지 분석
        result = DeepFace.analyze(
            img_path=roi,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",  # 빠른 탐지
            silent=True,
        )
        
        # result는 list[dict] 형태 (여러 얼굴 가능)
        if isinstance(result, list):
            result = result[0]  # 첫 번째 얼굴 사용
        
        emotion_scores = result.get("emotion", {})
        
        if not emotion_scores:
            return DeepFaceResult(
                success=False,
                expression="neutral",
                expression_confidence=0.0,
                pose="front",
                pose_confidence=0.8,
                combined_state="neutral",
            )
        
        expression, expr_conf = _deepface_to_expression(emotion_scores)
        
        # pose는 deepface가 지원하지 않으므로 기본값
        pose_label = "front"
        pose_conf = 0.8
        
        combined_state = get_state_from_emotion_pose(expression, pose_label)
        
        return DeepFaceResult(
            success=True,
            expression=expression,
            expression_confidence=expr_conf,
            pose=pose_label,
            pose_confidence=pose_conf,
            combined_state=combined_state,
            emotion_scores=emotion_scores,
        )
    
    except Exception as e:
        if not getattr(analyze_image, "_warned_analyze", False):
            print(f"⚠ DeepFace 분석 실패: {e}")
            analyze_image._warned_analyze = True
        return DeepFaceResult(
            success=False,
            expression="neutral",
            expression_confidence=0.0,
            pose="front",
            pose_confidence=0.8,
            combined_state="neutral",
        )
