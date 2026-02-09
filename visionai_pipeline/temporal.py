"""
Step 4: Temporal Action Recognition (시간 흐름 기반 행동 인식)

경량화 전략:
- 여러 프레임의 특징을 시간 축으로 집계
- 간단한 Temporal pooling + 1D Conv
- Heavy한 Video Transformer 대신 효율적인 방법 사용
"""

from __future__ import annotations
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import deque
import torch

from visionai_pipeline.emotion_categories import get_emotion_group, get_pose_group
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class TemporalFeature:
    """시간 축 특징"""
    timestamp: float
    emotion: str
    pose: str
    keypoints: Optional[np.ndarray]
    bbox: tuple[float, float, float, float]


@dataclass
class ActionResult:
    """행동 인식 결과"""
    action: str
    confidence: float
    duration: float  # 행동 지속 시간 (초)
    motion_intensity: float  # 움직임 강도 (0-1)


class TemporalActionRecognizer(nn.Module):
    """
    경량 시간 축 행동 인식 모델
    
    입력: 시간 축으로 정렬된 특징 벡터 시퀀스
    출력: 행동 클래스
    """
    
    def __init__(
        self,
        feature_dim: int = 64,
        num_actions: int = 8,
        temporal_window: int = 16
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.temporal_window = temporal_window
        
        # 1D Temporal Convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(feature_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 분류 헤드
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_actions)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, feature_dim) - 시간 축 특징
            
        Returns:
            logits: (B, num_actions)
        """
        # (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)
        
        # Temporal convolution
        x = self.temporal_conv(x)  # (B, 128, 1)
        x = x.squeeze(-1)  # (B, 128)
        
        # 분류
        logits = self.classifier(x)
        return logits


class TemporalAnalyzer:
    """
    시간 흐름 기반 행동 분석기
    
    여러 프레임의 정보를 시간 축으로 집계하여 행동 인식
    """
    
    # 행동 클래스 정의
    ACTION_CLASSES = [
        'resting',      # 휴식
        'eating',       # 먹기
        'walking',      # 걷기
        'running',      # 달리기
        'playing',      # 놀기
        'grooming',     # 그루밍
        'hunting',      # 사냥 자세
        'alert_scan'    # 경계하며 둘러보기
    ]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        temporal_window: int = 16,
        fps: float = 5.0
    ):
        """
        Args:
            model_path: 학습된 모델 경로
            device: 디바이스
            temporal_window: 분석할 프레임 수
            fps: 프레임 레이트 (초당 프레임)
        """
        self.device = self._get_device(device)
        self.temporal_window = temporal_window
        self.fps = fps
        
        # 모델 초기화
        self.model = TemporalActionRecognizer(
            feature_dim=64,
            num_actions=len(self.ACTION_CLASSES),
            temporal_window=temporal_window
        ).to(self.device)
        
        if model_path:
            self._load_model(model_path)
        else:
            print("⚠ 학습된 시간 축 모델이 없어 규칙 기반 분석 사용")
        
        self.model.eval()
        
        # 특징 버퍼 (시간 순서대로 저장)
        self.feature_buffer: deque[TemporalFeature] = deque(maxlen=temporal_window)
    
    def _get_device(self, device: str) -> torch.device:
        """디바이스 자동 선택"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def _load_model(self, model_path: str):
        """모델 로드"""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"✓ 시간 축 행동 인식 모델 로드 완료: {model_path}")
        except Exception as e:
            print(f"⚠ 모델 로드 실패: {e}")
    
    def add_frame(
        self,
        timestamp: float,
        emotion: str,
        pose: str,
        keypoints: Optional[np.ndarray],
        bbox: tuple[float, float, float, float]
    ):
        """
        프레임 정보 추가
        
        Args:
            timestamp: 타임스탬프 (초)
            emotion: 감정
            pose: 자세
            keypoints: 키포인트
            bbox: 바운딩 박스
        """
        feature = TemporalFeature(
            timestamp=timestamp,
            emotion=emotion,
            pose=pose,
            keypoints=keypoints,
            bbox=bbox
        )
        self.feature_buffer.append(feature)
    
    def analyze(self) -> Optional[ActionResult]:
        """
        현재 버퍼의 시간 축 정보로 행동 인식
        
        Returns:
            행동 인식 결과 (버퍼가 충분하지 않으면 None)
        """
        if len(self.feature_buffer) < 3:
            return None
        
        # 규칙 기반 행동 인식 (모델이 없을 때)
        return self._rule_based_recognition()
    
    def _rule_based_recognition(self) -> ActionResult:
        """
        규칙 기반 행동 인식
        
        간단한 휴리스틱으로 행동 판단
        """
        features = list(self.feature_buffer)
        
        # 움직임 강도 계산 (bbox 변화량)
        motion_intensity = self._compute_motion_intensity(features)
        
        # 자세 시퀀스 분석
        poses = [f.pose for f in features]
        pose_counts = {pose: poses.count(pose) for pose in set(poses)}
        dominant_pose = max(pose_counts, key=pose_counts.get)
        
        # 감정 시퀀스 분석
        emotions = [f.emotion for f in features]
        emotion_counts = {emo: emotions.count(emo) for emo in set(emotions)}
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        
        # 행동 추론
        action = self._infer_action(
            dominant_pose,
            dominant_emotion,
            motion_intensity
        )
        
        # 지속 시간 계산
        duration = features[-1].timestamp - features[0].timestamp
        
        return ActionResult(
            action=action,
            confidence=0.7,  # 규칙 기반이라 고정 신뢰도
            duration=duration,
            motion_intensity=motion_intensity
        )
    
    def _compute_motion_intensity(self, features: List[TemporalFeature]) -> float:
        """
        움직임 강도 계산
        
        바운딩 박스 중심의 이동 거리로 계산
        """
        if len(features) < 2:
            return 0.0
        
        movements = []
        for i in range(1, len(features)):
            prev_bbox = features[i-1].bbox
            curr_bbox = features[i].bbox
            
            # 중심 좌표
            prev_cx = (prev_bbox[0] + prev_bbox[2]) / 2
            prev_cy = (prev_bbox[1] + prev_bbox[3]) / 2
            curr_cx = (curr_bbox[0] + curr_bbox[2]) / 2
            curr_cy = (curr_bbox[1] + curr_bbox[3]) / 2
            
            # 이동 거리
            dist = np.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)
            movements.append(dist)
        
        # 평균 이동 거리 정규화 (0-1)
        avg_movement = np.mean(movements)
        normalized = min(avg_movement / 50.0, 1.0)  # 50픽셀 = 1.0
        
        return normalized
    
    def _infer_action(
        self,
        dominant_pose: str,
        dominant_emotion: str,
        motion_intensity: float
    ) -> str:
        """
        자세, 감정, 움직임 강도로 행동 추론.
        OpenFace 2.0 AU 기반 표정/자세는 emotion_categories의 그룹 매핑으로 처리.
        """
        emo = get_emotion_group(dominant_emotion)
        pos = get_pose_group(dominant_pose)

        # 먹기 행동: eating/drinking 자세
        if pos == "eating":
            return 'eating'

        # 움직임이 거의 없음
        if motion_intensity < 0.1:
            if pos == 'lying':
                return 'resting'
            if pos in ('sitting', 'standing'):
                return 'resting' if emo == 'relaxed' else 'alert_scan'

        # 중간 움직임
        if motion_intensity < 0.5:
            if emo == 'playful':
                return 'playing'
            if emo == 'alert':
                return 'alert_scan'
            if pos == 'walking':
                return 'walking'
            return 'walking'

        # 빠른 움직임
        if emo == 'fearful':
            return 'running'
        if emo == 'playful':
            return 'playing'
        if emo == 'aggressive' or dominant_pose == 'stalking':
            return 'hunting'
        return 'running'
    
    def reset(self):
        """버퍼 초기화"""
        self.feature_buffer.clear()
