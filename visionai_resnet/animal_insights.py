from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image


# COCO animal-ish labels (as returned by torchvision detection weights).
COCO_ANIMALS = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}


DEFAULT_BEHAVIORS: Sequence[str] = (
    "sleeping",
    "lying down",
    "sitting",
    "standing",
    "walking",
    "running",
    "jumping",
    "playing",
    "eating",
    "drinking",
    "sniffing",
)

DEFAULT_EXPRESSIONS: Sequence[str] = (
    "relaxed",
    "happy",
    "curious",
    "alert",
    "fearful",
    "stressed",
    "aggressive",
    "sad",
)


def _default_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _clamp_box_xyxy(box: Tuple[float, float, float, float], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w))
    y2 = max(0, min(int(round(y2)), h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


@dataclass(frozen=True)
class AnimalInsights:
    species: str
    behavior: str
    behavior_confidence: float
    expression: str
    expression_confidence: float
    estimated_state: str
    predicted_next_actions: List[str]
    disclaimer: str


class AnimalBehaviorExpressionAnalyzer:
    """
    Zero-shot animal behavior/expression analyzer using OpenCLIP.

    Important:
    - This is a heuristic, zero-shot estimate.
    - Not a veterinary/medical diagnosis.
    """

    def __init__(
        self,
        *,
        device: Optional[str] = None,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        behaviors: Sequence[str] = DEFAULT_BEHAVIORS,
        expressions: Sequence[str] = DEFAULT_EXPRESSIONS,
    ) -> None:
        self.device = _default_device(device)
        self.model_name = model_name
        self.pretrained = pretrained
        self.behaviors = list(behaviors)
        self.expressions = list(expressions)

        try:
            import open_clip  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "open_clip_torch is required for animal behavior/expression analysis. "
                "Install it with: pip install open_clip_torch"
            ) from e

        self._open_clip = open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.to(self.device)
        self.model.eval()

        # Cache text embeddings by species and kind ("behavior"/"expression")
        self._text_cache: Dict[Tuple[str, str], torch.Tensor] = {}

    def _build_prompts(self, *, species: str, kind: str, labels: Sequence[str]) -> List[str]:
        sp = species.replace("_", " ")
        if kind == "behavior":
            # Keep prompts simple (works reasonably for many animals).
            return [f"a photo of a {sp} that is {b}" for b in labels]
        if kind == "expression":
            return [f"a photo of a {sp} that looks {e}" for e in labels]
        raise ValueError("kind must be 'behavior' or 'expression'")

    @torch.inference_mode()
    def _encode_text(self, prompts: List[str]) -> torch.Tensor:
        tokens = self._open_clip.tokenize(prompts)
        tokens = tokens.to(self.device)
        text_features = self.model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    @torch.inference_mode()
    def _encode_image(self, crop: Image.Image) -> torch.Tensor:
        x = self.preprocess(crop).unsqueeze(0).to(self.device)
        img_features = self.model.encode_image(x)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        return img_features[0]

    def _get_text_features(self, *, species: str, kind: str) -> torch.Tensor:
        key = (species, kind)
        cached = self._text_cache.get(key)
        if cached is not None:
            return cached

        labels = self.behaviors if kind == "behavior" else self.expressions
        prompts = self._build_prompts(species=species, kind=kind, labels=labels)
        feats = self._encode_text(prompts)
        self._text_cache[key] = feats
        return feats

    def _estimate_state(self, expression: str) -> str:
        mapping = {
            "relaxed": "calm/comfortable",
            "happy": "positive/engaged",
            "curious": "interested/attentive",
            "alert": "high attention/arousal",
            "fearful": "anxious/avoidant",
            "stressed": "tense/overstimulated",
            "aggressive": "defensive/territorial",
            "sad": "low-energy/withdrawn",
        }
        return mapping.get(expression, "unknown")

    def _predict_next_actions(self, *, behavior: str, expression: str, species: str) -> List[str]:
        # Small rule-based layer to turn the estimates into something actionable.
        next_actions: List[str] = []

        if behavior in {"sleeping", "lying down"}:
            next_actions += ["continue resting", "change posture", "wake up if disturbed"]
        elif behavior in {"sitting", "standing"}:
            next_actions += ["observe surroundings", "approach a stimulus", "move away if threatened"]
        elif behavior in {"walking", "running", "jumping"}:
            next_actions += ["keep moving", "chase/avoid something", "slow down and rest"]
        elif behavior == "playing":
            next_actions += ["seek interaction", "chase toys/objects", "pause for rest/water"]
        elif behavior in {"eating", "drinking"}:
            next_actions += ["continue feeding", "search for more food/water", "move to a calmer spot"]
        elif behavior == "sniffing":
            next_actions += ["explore environment", "track a scent", "change direction"]

        if expression in {"fearful", "stressed"}:
            next_actions = ["increase distance", "freeze/hesitate", "seek a safe person/place"]
        elif expression == "aggressive":
            next_actions = ["guard territory", "warn (growl/hiss)", "keep distance from others"]
        elif expression in {"curious", "alert"}:
            next_actions = list(dict.fromkeys(next_actions + ["look toward stimulus", "approach cautiously"]))
        elif expression in {"relaxed", "happy"}:
            next_actions = list(dict.fromkeys(next_actions + ["remain calm", "engage gently"]))

        # Minor species flavor
        if species == "cat" and "warn (growl/hiss)" in next_actions:
            next_actions = ["warn (hiss)", "keep distance from others", "guard territory"]

        # Dedup, keep order, cap.
        uniq: List[str] = []
        for a in next_actions:
            if a not in uniq:
                uniq.append(a)
        return uniq[:5]

    @torch.inference_mode()
    def analyze_crop(self, crop: Image.Image, *, species: str) -> AnimalInsights:
        species = species.lower().strip()
        if not species:
            species = "animal"

        img_feat = self._encode_image(crop)

        beh_text = self._get_text_features(species=species, kind="behavior")
        exp_text = self._get_text_features(species=species, kind="expression")

        beh_logits = (img_feat @ beh_text.T) * 100.0
        exp_logits = (img_feat @ exp_text.T) * 100.0

        beh_probs = torch.softmax(beh_logits, dim=-1)
        exp_probs = torch.softmax(exp_logits, dim=-1)

        beh_idx = int(torch.argmax(beh_probs).item())
        exp_idx = int(torch.argmax(exp_probs).item())

        behavior = self.behaviors[beh_idx]
        expression = self.expressions[exp_idx]
        behavior_conf = float(beh_probs[beh_idx].item())
        expression_conf = float(exp_probs[exp_idx].item())

        estimated_state = self._estimate_state(expression)
        predicted_next = self._predict_next_actions(behavior=behavior, expression=expression, species=species)

        return AnimalInsights(
            species=species,
            behavior=behavior,
            behavior_confidence=behavior_conf,
            expression=expression,
            expression_confidence=expression_conf,
            estimated_state=estimated_state,
            predicted_next_actions=predicted_next,
            disclaimer=(
                "Zero-shot 추정 결과입니다. 사진/각도/조명에 따라 크게 틀릴 수 있으며 "
                "수의학적/의학적 진단이 아닙니다."
            ),
        )

    def crop_from_image(
        self,
        image: Image.Image,
        *,
        box_xyxy: Tuple[float, float, float, float],
        pad_ratio: float = 0.06,
    ) -> Image.Image:
        w, h = image.size
        x1, y1, x2, y2 = box_xyxy

        # Add a small padding to include head/body context.
        pad_x = (x2 - x1) * pad_ratio
        pad_y = (y2 - y1) * pad_ratio
        padded = (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)

        cx1, cy1, cx2, cy2 = _clamp_box_xyxy(padded, w, h)
        return image.crop((cx1, cy1, cx2, cy2)).convert("RGB")

