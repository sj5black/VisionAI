"""
Step 5: Behavior Prediction (이후 행동 예측)

경량 LSTM 기반 시퀀스 예측:
- 과거 행동/상태 시퀀스로 다음 행동 예측
- 간단한 규칙 기반 폴백
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import torch
import torch.nn as nn
import numpy as np


@dataclass
class PredictionResult:
    """행동 예측 결과"""
    predicted_action: str
    confidence: float
    time_horizon: float  # 예측 시간 범위 (초)
    alternative_actions: List[Tuple[str, float]]  # (행동, 확률)


class BehaviorPredictorModel(nn.Module):
    """
    경량 LSTM 기반 행동 예측 모델
    """
    
    def __init__(
        self,
        action_vocab_size: int = 8,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 2
    ):
        super().__init__()
        
        # 행동 임베딩
        self.action_embedding = nn.Embedding(action_vocab_size, embedding_dim)
        
        # LSTM 인코더
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 예측 헤드
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, action_vocab_size)
        )
    
    def forward(
        self,
        action_sequence: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            action_sequence: (B, T) - 행동 ID 시퀀스
            
        Returns:
            logits: (B, action_vocab_size) - 다음 행동 예측
        """
        # 임베딩
        embedded = self.action_embedding(action_sequence)  # (B, T, E)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(embedded)  # lstm_out: (B, T, H)
        
        # 마지막 hidden state 사용
        last_hidden = lstm_out[:, -1, :]  # (B, H)
        
        # 예측
        logits = self.predictor(last_hidden)  # (B, vocab_size)
        
        return logits


class BehaviorPredictor:
    """
    행동 예측기
    
    과거 행동 시퀀스를 기반으로 다음 행동 예측
    """
    
    # 행동 클래스 (temporal.py와 동일)
    ACTION_CLASSES = [
        'resting',
        'eating',
        'walking',
        'running',
        'playing',
        'grooming',
        'hunting',
        'alert_scan'
    ]
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        sequence_length: int = 8,
        time_horizon: float = 5.0
    ):
        """
        Args:
            model_path: 학습된 모델 경로
            device: 디바이스
            sequence_length: 예측에 사용할 과거 시퀀스 길이
            time_horizon: 예측 시간 범위 (초)
        """
        self.device = self._get_device(device)
        self.sequence_length = sequence_length
        self.time_horizon = time_horizon
        
        # 행동 -> ID 매핑
        self.action_to_id = {action: i for i, action in enumerate(self.ACTION_CLASSES)}
        self.id_to_action = {i: action for action, i in self.action_to_id.items()}
        
        # 모델 초기화
        self.model = BehaviorPredictorModel(
            action_vocab_size=len(self.ACTION_CLASSES),
            embedding_dim=32,
            hidden_dim=64,
            num_layers=2
        ).to(self.device)
        
        if model_path:
            self._load_model(model_path)
        else:
            print("⚠ 학습된 예측 모델이 없어 규칙 기반 예측 사용")
        
        self.model.eval()
        
        # 행동 히스토리
        self.action_history: deque[str] = deque(maxlen=sequence_length)
    
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
            print(f"✓ 행동 예측 모델 로드 완료: {model_path}")
        except Exception as e:
            print(f"⚠ 모델 로드 실패: {e}")
    
    def add_action(self, action: str):
        """
        관찰된 행동 추가
        
        Args:
            action: 행동 이름
        """
        if action in self.action_to_id:
            self.action_history.append(action)
    
    def predict(self) -> Optional[PredictionResult]:
        """
        다음 행동 예측
        
        Returns:
            예측 결과 (히스토리가 충분하지 않으면 None)
        """
        if len(self.action_history) < 2:
            return None
        
        # 규칙 기반 예측 (모델이 없을 때)
        return self._rule_based_prediction()
    
    def _rule_based_prediction(self) -> PredictionResult:
        """
        규칙 기반 행동 예측
        
        간단한 전이 규칙과 통계 기반
        """
        actions = list(self.action_history)
        current_action = actions[-1]
        
        # 행동 전이 규칙 (확률)
        transition_rules = {
            'resting': {
                'resting': 0.6,
                'walking': 0.2,
                'grooming': 0.1,
                'alert_scan': 0.1
            },
            'walking': {
                'walking': 0.4,
                'resting': 0.2,
                'running': 0.1,
                'eating': 0.15,
                'alert_scan': 0.15
            },
            'running': {
                'running': 0.5,
                'walking': 0.3,
                'resting': 0.1,
                'playing': 0.1
            },
            'playing': {
                'playing': 0.5,
                'running': 0.2,
                'walking': 0.2,
                'resting': 0.1
            },
            'eating': {
                'eating': 0.5,
                'grooming': 0.2,
                'resting': 0.2,
                'walking': 0.1
            },
            'grooming': {
                'grooming': 0.5,
                'resting': 0.3,
                'walking': 0.2
            },
            'hunting': {
                'hunting': 0.4,
                'running': 0.3,
                'alert_scan': 0.2,
                'walking': 0.1
            },
            'alert_scan': {
                'alert_scan': 0.4,
                'walking': 0.2,
                'running': 0.2,
                'resting': 0.1,
                'hunting': 0.1
            }
        }
        
        # 현재 행동의 전이 확률
        transitions = transition_rules.get(current_action, {})
        
        if not transitions:
            # 기본 전이: 현재 행동 유지
            transitions = {current_action: 0.7, 'walking': 0.2, 'resting': 0.1}
        
        # 최근 패턴 고려 (반복 감지)
        if len(actions) >= 3:
            recent_pattern = actions[-3:]
            if len(set(recent_pattern)) == 1:
                # 같은 행동 반복 중 -> 변화 가능성 증가
                for action in transitions:
                    if action != current_action:
                        transitions[action] *= 1.5
                # 정규화
                total = sum(transitions.values())
                transitions = {k: v/total for k, v in transitions.items()}
        
        # 예측 결과
        sorted_actions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
        predicted_action = sorted_actions[0][0]
        confidence = sorted_actions[0][1]
        
        return PredictionResult(
            predicted_action=predicted_action,
            confidence=confidence,
            time_horizon=self.time_horizon,
            alternative_actions=sorted_actions[1:4]  # 상위 3개 대안
        )
    
    def _model_based_prediction(self) -> PredictionResult:
        """
        모델 기반 행동 예측
        
        학습된 LSTM 모델 사용
        """
        # 행동 시퀀스를 ID로 변환
        action_ids = [
            self.action_to_id[action]
            for action in self.action_history
        ]
        
        # 패딩 (sequence_length에 맞춤)
        if len(action_ids) < self.sequence_length:
            action_ids = [0] * (self.sequence_length - len(action_ids)) + action_ids
        else:
            action_ids = action_ids[-self.sequence_length:]
        
        # 텐서로 변환
        input_tensor = torch.tensor([action_ids], dtype=torch.long).to(self.device)
        
        # 예측
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]
        
        # 결과 파싱
        probs_np = probs.cpu().numpy()
        sorted_indices = np.argsort(probs_np)[::-1]
        
        predicted_action = self.id_to_action[sorted_indices[0]]
        confidence = probs_np[sorted_indices[0]]
        
        alternative_actions = [
            (self.id_to_action[idx], probs_np[idx])
            for idx in sorted_indices[1:4]
        ]
        
        return PredictionResult(
            predicted_action=predicted_action,
            confidence=float(confidence),
            time_horizon=self.time_horizon,
            alternative_actions=alternative_actions
        )
    
    def reset(self):
        """히스토리 초기화"""
        self.action_history.clear()
