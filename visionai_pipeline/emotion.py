"""
Step 3: 표정·행동 분석 — OpenFace 2.0 (Action Units) 기반

- OpenFace 2.0 호환: pyfaceau 사용 (얼굴 위치, head pose, gaze, AU intensity/presence)
- AU 조합 → 표정: 진짜 웃음(AU12+AU6), 가짜 웃음(AU12 only), 집중(AU4+AU7), 놀람(AU1+AU2) 등
- 추가 학습 없음
"""

from __future__ import annotations
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .openface_analyzer import analyze_frame as _openface_analyze


@dataclass
class EmotionResult:
    """감정/자세 분석 결과 (파이프라인 호환)"""
    emotion: str
    emotion_confidence: float
    pose: str
    pose_confidence: float
    combined_state: str


# 파이프라인/API에서 참조하는 라벨 목록 (OpenFace AU 기반)
EXPRESSION_LABELS = [
    "neutral", "real_smile", "fake_smile", "focused", "surprised",
    "sad", "displeased", "attention",
]
POSE_LABELS = ["front", "looking_down", "looking_up", "looking_side"]


class EmotionAnalyzer:
    """
    OpenFace 2.0 (pyfaceau) 기반 표정·행동 분석기.
    AU intensity → 표정 라벨, head pose → 자세 라벨.
    """

    EMOTION_CLASSES = EXPRESSION_LABELS
    POSE_CLASSES = POSE_LABELS

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        use_swin_when_no_model: bool = False,
        use_openclip_when_no_model: bool = False,
    ):
        """
        OpenFace만 사용. model_path / use_swin / use_openclip 인자는 호환용으로 무시.
        """
        self._emotion_backend_name = "OpenFace 2.0 (AU)"
        print("✓ 감정 분석: OpenFace 2.0 (Action Units) — 진짜/가짜 웃음, 집중, 놀람 등")

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
        OpenFace 2.0 AU → 표정, head pose → 자세.
        """
        out = _openface_analyze(image, bbox=bbox)
        return EmotionResult(
            emotion=out.expression,
            emotion_confidence=out.expression_confidence,
            pose=out.pose,
            pose_confidence=out.pose_confidence,
            combined_state=out.combined_state,
        )

    def save_model(self, save_path: str) -> None:
        """OpenFace는 사전 학습 모델만 사용하므로 저장 없음."""
        print("⚠ OpenFace 모드에서는 저장할 학습 가중치가 없습니다.")
