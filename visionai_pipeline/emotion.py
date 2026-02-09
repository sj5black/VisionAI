"""
Step 3: 표정·행동 분석 — 다중 백엔드 지원

지원 백엔드:
- openclip: OpenCLIP zero-shot (ViT-B-32) — 동물 표정/자세 (기존 방식)
- deepface: DeepFace — 7가지 기본 감정 (이미지 전용)
- pyfaceau: OpenFace 2.0 — AU + head pose (영상 전용, 이미지는 임시 비디오 변환)
"""

from __future__ import annotations
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class EmotionResult:
    """감정/자세 분석 결과 (파이프라인 호환)"""
    emotion: str
    emotion_confidence: float
    pose: str
    pose_confidence: float
    combined_state: str


# 각 백엔드별 라벨
from .emotion_categories import EXPRESSION_LABELS, POSE_LABELS
from .openclip_analyzer import EMOTION_CLASSES_OPENCLIP, POSE_CLASSES_OPENCLIP


class EmotionAnalyzer:
    """
    다중 백엔드 지원 표정·행동 분석기.
    
    지원 백엔드:
    - openclip: 기존 OpenCLIP zero-shot 방식
    - deepface: DeepFace 감정 분석
    - pyfaceau: OpenFace 2.0 AU 분석
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        use_swin_when_no_model: bool = False,
        use_openclip_when_no_model: bool = False,
        emotion_backend: str = "openclip",  # openclip | deepface | pyfaceau
    ):
        """
        Args:
            emotion_backend: 사용할 백엔드 ("openclip", "deepface", "pyfaceau")
        """
        self.device = device
        self.backend = (emotion_backend or "openclip").lower()
        
        if self.backend == "openclip":
            self._emotion_backend_name = "OpenCLIP (기존 방식)"
            self.EMOTION_CLASSES = EMOTION_CLASSES_OPENCLIP
            self.POSE_CLASSES = POSE_CLASSES_OPENCLIP
            print("✓ 감정 분석: OpenCLIP (Vision-Language, 16 emotions + 18 poses)")
        elif self.backend == "deepface":
            self._emotion_backend_name = "DeepFace (이미지)"
            self.EMOTION_CLASSES = EXPRESSION_LABELS
            self.POSE_CLASSES = POSE_LABELS
            print("✓ 감정 분석: DeepFace (7가지 기본 감정)")
        elif self.backend == "pyfaceau":
            self._emotion_backend_name = "OpenFace 2.0 (AU + pose)"
            self.EMOTION_CLASSES = EXPRESSION_LABELS
            self.POSE_CLASSES = POSE_LABELS
            print("✓ 감정 분석: OpenFace 2.0 (Action Units + head pose)")
        else:
            # 기본값
            self.backend = "openclip"
            self._emotion_backend_name = "OpenCLIP (기본)"
            self.EMOTION_CLASSES = EMOTION_CLASSES_OPENCLIP
            self.POSE_CLASSES = POSE_CLASSES_OPENCLIP
            print("✓ 감정 분석: OpenCLIP (기본)")

    @property
    def emotion_backend_name(self) -> str:
        return self._emotion_backend_name

    def analyze(
        self,
        image: np.ndarray,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        species: Optional[str] = None,
    ) -> EmotionResult:
        """
        이미지(및 선택적 bbox)에서 표정·자세 분석.
        백엔드에 따라 다른 분석 방법 사용.
        """
        if self.backend == "openclip":
            from .openclip_analyzer import analyze_image_openclip
            result = analyze_image_openclip(image, bbox, device=self.device, species=species)
            return EmotionResult(
                emotion=result.expression,
                emotion_confidence=result.expression_confidence,
                pose=result.pose,
                pose_confidence=result.pose_confidence,
                combined_state=result.combined_state,
            )
        
        elif self.backend == "deepface":
            from .deepface_analyzer import analyze_image
            result = analyze_image(image, bbox)
            return EmotionResult(
                emotion=result.expression,
                emotion_confidence=result.expression_confidence,
                pose=result.pose,
                pose_confidence=result.pose_confidence,
                combined_state=result.combined_state,
            )
        
        elif self.backend == "pyfaceau":
            from .openface_analyzer import analyze_frame
            result = analyze_frame(image, bbox)
            return EmotionResult(
                emotion=result.expression,
                emotion_confidence=result.expression_confidence,
                pose=result.pose,
                pose_confidence=result.pose_confidence,
                combined_state=result.combined_state,
            )
        
        else:
            # fallback
            return EmotionResult(
                emotion="neutral",
                emotion_confidence=0.0,
                pose="sitting",
                pose_confidence=0.0,
                combined_state="neutral",
            )

    def save_model(self, save_path: str) -> None:
        """사전 학습 모델만 사용하므로 저장 없음."""
        print(f"⚠ {self.backend} 모드에서는 저장할 학습 가중치가 없습니다.")
