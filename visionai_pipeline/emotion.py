"""
Step 3: Emotion & Pose Analysis (표정 + 자세 해석)

고급 분석 옵션:
- 기본(학습 모델 없음): OpenCLIP zero-shot (ViT-B-32) — 시맨틱 이해로 정확한 표정/자세 추정
- 학습 모델 있음: EfficientNet-B2 백본 + 커스텀 분류 헤드
- 감정 16종: relaxed, happy, content, curious, alert, excited, playful, sleepy, bored,
  fearful, anxious, stressed, nervous, aggressive, dominant, submissive, affectionate
- 자세 18종: sitting, standing, lying, running, jumping, walking, crouching, stretching,
  sleeping, eating, drinking, sniffing, grooming, playing, begging, hiding, rolling, stalking
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
import numpy as np
from PIL import Image


@dataclass
class EmotionResult:
    """감정/자세 분석 결과"""
    emotion: str
    emotion_confidence: float
    pose: str
    pose_confidence: float
    combined_state: str  # 통합 상태


class EmotionClassifier(nn.Module):
    """
    고급 감정/자세 분류기 (학습된 체크포인트 사용 시)
    EfficientNet-B2 백본 + 멀티태스크 헤드
    """

    def __init__(
        self,
        num_emotions: int = 5,
        num_poses: int = 5
    ):
        super().__init__()
        backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        feature_dim = 1408  # EfficientNet-B2 classifier input

        self.emotion_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_emotions)
        )
        self.pose_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_poses)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features(x)
        emotion_logits = self.emotion_head(features)
        pose_logits = self.pose_head(features)
        return emotion_logits, pose_logits


def _get_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


# 확장된 감정/자세 카테고리 (OpenCLIP zero-shot에 적합한 범주)
EMOTION_CLASSES = [
    'relaxed', 'happy', 'content', 'curious', 'alert', 'excited', 'playful',
    'sleepy', 'bored', 'fearful', 'anxious', 'stressed', 'nervous',
    'aggressive', 'dominant', 'submissive', 'affectionate'
]
POSE_CLASSES = [
    'sitting', 'standing', 'lying', 'running', 'jumping', 'walking',
    'crouching', 'stretching', 'sleeping', 'eating', 'drinking',
    'sniffing', 'grooming', 'playing', 'begging', 'hiding',
    'rolling', 'stalking'
]


class _OpenCLIPEmotionBackend:
    """
    OpenCLIP 기반 zero-shot 표정/자세 분석.
    사전학습된 vision-language 모델로 동물 표정·자세를 시맨틱하게 추정.
    """
    EMOTION_CLASSES = EMOTION_CLASSES
    POSE_CLASSES = POSE_CLASSES

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
        sp = (species or "animal").lower().replace("_", " ")
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
    ) -> EmotionResult:
        if bbox is not None:
            x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            roi = image[y1:y2, x1:x2]
        else:
            roi = image
        pil_image = Image.fromarray(roi).convert("RGB")
        img_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        img_features = self.model.encode_image(img_tensor)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        img_features = img_features.squeeze(0)

        species = (species or "animal").strip() or "animal"
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

        combined_state = _determine_combined_state(emotion, pose)
        return EmotionResult(
            emotion=emotion,
            emotion_confidence=emotion_conf,
            pose=pose,
            pose_confidence=pose_conf,
            combined_state=combined_state
        )


def _determine_combined_state(emotion: str, pose: str) -> str:
    """감정+자세 조합으로 통합 상태 생성. 주요 조합은 의미 있는 라벨로 매핑."""
    state_map = {
        ('relaxed', 'lying'): 'resting',
        ('relaxed', 'sitting'): 'calm',
        ('relaxed', 'sleeping'): 'resting',
        ('happy', 'sitting'): 'content',
        ('happy', 'standing'): 'content',
        ('content', 'lying'): 'resting',
        ('content', 'sitting'): 'calm',
        ('curious', 'standing'): 'vigilant',
        ('curious', 'sitting'): 'attentive',
        ('alert', 'standing'): 'vigilant',
        ('alert', 'sitting'): 'attentive',
        ('alert', 'crouching'): 'stalking',
        ('excited', 'running'): 'playing',
        ('excited', 'jumping'): 'excited',
        ('playful', 'running'): 'playing',
        ('playful', 'jumping'): 'excited',
        ('playful', 'standing'): 'playing',
        ('sleepy', 'lying'): 'resting',
        ('sleepy', 'sleeping'): 'resting',
        ('bored', 'lying'): 'resting',
        ('fearful', 'lying'): 'hiding',
        ('fearful', 'running'): 'fleeing',
        ('fearful', 'crouching'): 'hiding',
        ('anxious', 'standing'): 'vigilant',
        ('stressed', 'running'): 'fleeing',
        ('nervous', 'crouching'): 'hiding',
        ('aggressive', 'standing'): 'threatening',
        ('aggressive', 'crouching'): 'stalking',
        ('dominant', 'standing'): 'threatening',
        ('submissive', 'lying'): 'submissive',
        ('affectionate', 'sitting'): 'calm',
    }
    return state_map.get((emotion, pose), f"{emotion}_{pose}")


class EmotionAnalyzer:
    """
    표정 및 자세 분석기.
    - 학습 모델 없음: OpenCLIP zero-shot (고급 시맨틱 분석)
    - 학습 모델 있음: EfficientNet-B2 기반 분류기
    """

    EMOTION_CLASSES = EMOTION_CLASSES
    POSE_CLASSES = POSE_CLASSES

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        use_openclip_when_no_model: bool = True
    ):
        """
        Args:
            model_path: 학습된 모델 경로. None이면 OpenCLIP zero-shot 사용(권장).
            device: 'auto', 'cpu', 'cuda', 'mps'
            use_openclip_when_no_model: model_path가 None일 때 OpenCLIP 사용 여부.
        """
        self.device = _get_device(device)
        self._use_clip = False
        self._clip_backend: Optional[_OpenCLIPEmotionBackend] = None
        self.model: Optional[EmotionClassifier] = None
        self.transform = transforms.Compose([
            transforms.Resize((260, 260)),  # EfficientNet-B2 권장
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if model_path:
            self.model = EmotionClassifier(
                num_emotions=len(self.EMOTION_CLASSES),
                num_poses=len(self.POSE_CLASSES)
            ).to(self.device)
            self._load_model(model_path)
            self.model.eval()
            print("✓ 감정 분석: 학습된 EfficientNet-B2 모델 사용")
        elif use_openclip_when_no_model:
            try:
                self._clip_backend = _OpenCLIPEmotionBackend(self.device)
                self._use_clip = True
                print("✓ 감정 분석: OpenCLIP zero-shot (ViT-B-32) 사용 — 표정/자세 정확도 향상")
            except Exception as e:
                print(f"⚠ OpenCLIP 로드 실패 ({e}), 랜덤 초기화 분류기로 대체 (정확도 낮음)")
                self._use_clip = False
                self._init_fallback_classifier()
        else:
            self._init_fallback_classifier()

        if self.model is not None:
            self.model.eval()

    def _init_fallback_classifier(self):
        """OpenCLIP 미사용 시 호환용 경량 분류기 (랜덤 초기화)."""
        from torchvision.models import mobilenet_v3_small
        feature_dim = 576
        backbone = mobilenet_v3_small(weights='DEFAULT')
        features = nn.Sequential(*list(backbone.children())[:-1])

        class _FallbackClassifier(nn.Module):
            def __init__(self_):
                super().__init__()
                self_.features = features
                self_.emotion_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                    nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(256, len(EmotionAnalyzer.EMOTION_CLASSES))
                )
                self_.pose_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                    nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(256, len(EmotionAnalyzer.POSE_CLASSES))
                )
            def forward(self_, x):
                f = self_.features(x)
                return self_.emotion_head(f), self_.pose_head(f)

        self.model = _FallbackClassifier().to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("⚠ 학습된 모델 없음, 랜덤 초기화 사용 (정확도 낮음). open_clip_torch 설치 권장.")

    def _load_model(self, model_path: str):
        try:
            state = torch.load(model_path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.model.load_state_dict(state, strict=False)
            print(f"✓ 감정 모델 로드: {model_path}")
        except Exception as e:
            print(f"⚠ 감정 모델 로드 실패: {e}")

    def analyze(
        self,
        image: np.ndarray,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        species: Optional[str] = None
    ) -> EmotionResult:
        """
        이미지에서 표정 및 자세 분석.

        Args:
            image: RGB 이미지 (H, W, 3)
            bbox: 관심 영역 (x1, y1, x2, y2), None이면 전체
            species: 동물 종(예: 'dog', 'cat'). OpenCLIP 사용 시 프롬프트에 반영되어 정확도 향상
        """
        if self._use_clip and self._clip_backend is not None:
            return self._clip_backend.analyze(image, bbox, species)

        if bbox is not None:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            roi = image[y1:y2, x1:x2]
        else:
            roi = image
        pil_image = Image.fromarray(roi)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emotion_logits, pose_logits = self.model(input_tensor)
            emotion_probs = F.softmax(emotion_logits, dim=1)
            pose_probs = F.softmax(pose_logits, dim=1)

        emotion_idx = emotion_probs.argmax(dim=1).item()
        pose_idx = pose_probs.argmax(dim=1).item()
        emotion = self.EMOTION_CLASSES[emotion_idx]
        pose = self.POSE_CLASSES[pose_idx]
        emotion_conf = emotion_probs[0, emotion_idx].item()
        pose_conf = pose_probs[0, pose_idx].item()
        combined_state = _determine_combined_state(emotion, pose)

        return EmotionResult(
            emotion=emotion,
            emotion_confidence=emotion_conf,
            pose=pose,
            pose_confidence=pose_conf,
            combined_state=combined_state
        )

    def save_model(self, save_path: str):
        """학습된 분류기만 저장 (OpenCLIP 백엔드는 저장하지 않음)."""
        if self.model is not None and not self._use_clip:
            torch.save(self.model.state_dict(), save_path)
            print(f"✓ 모델 저장: {save_path}")
        else:
            print("⚠ OpenCLIP 모드에서는 저장할 학습 가중치가 없습니다.")
