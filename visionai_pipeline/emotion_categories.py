"""
VisionAI Pipeline - OpenFace 2.0 Action Unit 기반 표정·행동 해석

AU 조합 → 표정 라벨 (추가 학습 없음, OpenFace 2.0 출력 기반)
- AU12 only → 가짜 웃음
- AU12 + AU6 → 진짜 웃음
- AU4 + AU7 → 집중
- AU1 + AU2 → 놀람
- 기타 AU 조합으로 확장
"""

from typing import Dict, List, Tuple

# OpenFace 2.0 AU 기반 표정 라벨 (파이프라인 출력용)
EXPRESSION_LABELS: List[str] = [
    "neutral",       # 중립
    "real_smile",    # 진짜 웃음 (AU12+AU6)
    "fake_smile",    # 가짜 웃음 (AU12 only)
    "focused",       # 집중 (AU4+AU7)
    "surprised",     # 놀람 (AU1+AU2)
    "sad",           # 슬픔/불만 (AU15, AU17)
    "displeased",    # 찡그림/불쾌 (AU4)
    "attention",     # 주의 (AU5, AU26 등)
]

# Head pose / gaze 기반 자세 라벨 (간단)
POSE_LABELS: List[str] = [
    "front",
    "looking_down",
    "looking_up",
    "looking_side",
]

# Temporal 행동 추론용: 표정 → 5개 그룹
_EXPRESSION_TO_GROUP: Dict[str, str] = {
    "neutral": "relaxed",
    "real_smile": "playful",
    "fake_smile": "playful",
    "focused": "alert",
    "surprised": "alert",
    "sad": "relaxed",
    "displeased": "aggressive",
    "attention": "alert",
}


def get_emotion_group(emotion: str) -> str:
    """표정 라벨을 temporal 행동 추론용 5개 그룹 중 하나로 반환."""
    e = (emotion or "").strip().lower()
    return _EXPRESSION_TO_GROUP.get(e, "relaxed")


def get_pose_group(pose: str) -> str:
    """자세 라벨을 temporal용 그룹(lying/sitting/standing 등)으로 반환."""
    p = (pose or "").strip().lower()
    if p in ("looking_down",):
        return "sitting"
    if p in ("looking_up", "attention"):
        return "standing"
    return "standing"


def get_state_from_emotion_pose(emotion: str, pose: str) -> str:
    """감정+자세 → 통합 상태 (간단 매핑)."""
    e = (emotion or "").strip().lower()
    p = (pose or "").strip().lower()
    if not e and not p:
        return "neutral"
    state_map: Dict[Tuple[str, str], str] = {
        ("real_smile", "front"): "happy",
        ("fake_smile", "front"): "smiling",
        ("focused", "front"): "focused",
        ("surprised", "front"): "surprised",
        ("sad", "front"): "sad",
        ("displeased", "front"): "displeased",
        ("neutral", "front"): "neutral",
        ("attention", "front"): "attentive",
    }
    return state_map.get((e, p), e or p or "neutral")
