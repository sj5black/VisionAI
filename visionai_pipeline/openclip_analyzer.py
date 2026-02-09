"""
OpenCLIP 기반 zero-shot 표정/자세 분석 (기존 방식)

- 동물 표정·자세 분석 (16가지 감정 + 18가지 자세)
- Vision-Language 모델로 시맨틱 이해
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from .emotion_categories import get_state_from_emotion_pose


# 기존 감정/자세 카테고리
EMOTION_CLASSES_OPENCLIP = [
    'relaxed', 'happy', 'content', 'curious', 'alert', 'excited', 'playful',
    'sleepy', 'bored', 'fearful', 'anxious', 'stressed', 'nervous',
    'aggressive', 'dominant', 'submissive', 'affectionate'
]
POSE_CLASSES_OPENCLIP = [
    'sitting', 'standing', 'lying', 'running', 'jumping', 'walking',
    'crouching', 'stretching', 'sleeping', 'eating', 'drinking',
    'sniffing', 'grooming', 'playing', 'begging', 'hiding',
    'rolling', 'stalking'
]


@dataclass
class OpenCLIPResult:
    """OpenCLIP 분석 결과"""
    success: bool
    expression: str
    expression_confidence: float
    pose: str
    pose_confidence: float
    combined_state: str


class OpenCLIPEmotionBackend:
    """
    OpenCLIP 기반 zero-shot 표정/자세 분석.
    사전학습된 vision-language 모델로 동물 표정·자세를 시맨틱하게 추정.
    """
    EMOTION_CLASSES = EMOTION_CLASSES_OPENCLIP
    POSE_CLASSES = POSE_CLASSES_OPENCLIP

    def __init__(self, device: torch.device, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        try:
            import open_clip  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "표정 분석을 위해 open_clip_torch가 필요합니다. "
                "설치: pip install open_clip_torch"
            ) from e
        self._open_clip = open_clip
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.to(device)
        self.model.eval()
        self._text_cache: Dict[Tuple[str, str], torch.Tensor] = {}

    def _prompts(self, species: str, kind: str) -> List[str]:
        sp = (species or "person").lower().replace("_", " ")
        if kind == "emotion":
            return [f"a photo of a {sp} that looks {e}" for e in self.EMOTION_CLASSES]
        if kind == "pose":
            return [f"a photo of a {sp} that is {p}" for p in self.POSE_CLASSES]
        return []

    def _encode_text(self, prompts: List[str]) -> torch.Tensor:
        tokens = self._open_clip.tokenize(prompts).to(self.device)
        features = self.model.encode_text(tokens)
        return features / features.norm(dim=-1, keepdim=True)

    def _get_text_features(self, species: str, kind: str) -> torch.Tensor:
        key = (species, kind)
        if key not in self._text_cache:
            prompts = self._prompts(species, kind)
            self._text_cache[key] = self._encode_text(prompts)
        return self._text_cache[key]

    @torch.inference_mode()
    def analyze(
        self,
        image: np.ndarray,
        bbox: Optional[Tuple[float, float, float, float]],
        species: Optional[str] = None
    ) -> OpenCLIPResult:
        if bbox is not None:
            x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            roi = image[y1:y2, x1:x2]
        else:
            roi = image
        
        if roi.size == 0:
            return OpenCLIPResult(
                success=False,
                expression="neutral",
                expression_confidence=0.0,
                pose="sitting",
                pose_confidence=0.0,
                combined_state="neutral"
            )
        
        pil_image = Image.fromarray(roi).convert("RGB")
        img_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        img_features = self.model.encode_image(img_tensor)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        img_features = img_features.squeeze(0)

        species = (species or "person").strip() or "person"
        emo_text = self._get_text_features(species, "emotion")
        pose_text = self._get_text_features(species, "pose")

        emo_logits = (img_features @ emo_text.T) * 100.0
        pose_logits = (img_features @ pose_text.T) * 100.0
        emo_probs = F.softmax(emo_logits, dim=-1)
        pose_probs = F.softmax(pose_logits, dim=-1)

        emo_idx = int(emo_probs.argmax().item())
        pose_idx = int(pose_probs.argmax().item())
        emotion = self.EMOTION_CLASSES[emo_idx]
        pose = self.POSE_CLASSES[pose_idx]
        emotion_conf = float(emo_probs[emo_idx].item())
        pose_conf = float(pose_probs[pose_idx].item())

        combined_state = get_state_from_emotion_pose(emotion, pose)
        return OpenCLIPResult(
            success=True,
            expression=emotion,
            expression_confidence=emotion_conf,
            pose=pose,
            pose_confidence=pose_conf,
            combined_state=combined_state
        )


def analyze_image_openclip(
    image: np.ndarray,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    device: str = "auto",
    species: Optional[str] = None
) -> OpenCLIPResult:
    """
    OpenCLIP으로 이미지 분석 (캐싱된 인스턴스 사용)
    """
    if not hasattr(analyze_image_openclip, "_backend"):
        import torch
        if device == "auto":
            if torch.cuda.is_available():
                dev = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                dev = torch.device("mps")
            else:
                dev = torch.device("cpu")
        else:
            dev = torch.device(device)
        
        analyze_image_openclip._backend = OpenCLIPEmotionBackend(dev)
        print(f"✓ OpenCLIP 백엔드 초기화 완료 (device: {dev})")
    
    return analyze_image_openclip._backend.analyze(image, bbox, species)
