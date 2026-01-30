from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .models import CustomResNet, custom_resnet18


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Prediction:
    class_id: int
    label: str
    probability: float


@dataclass(frozen=True)
class ImageAnalysis:
    image_path: str
    topk: List[Prediction]
    feature: Optional[List[float]] = None


def _default_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _read_labels_file(path: Path) -> List[str]:
    labels: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            labels.append(s)
    return labels


def _pil_open_rgb(image_path: Path) -> Image.Image:
    with Image.open(image_path) as img:
        return img.convert("RGB")


class ResNetImageAnalyzer:
    """
    Image analyzer using ResNet.

    Two modes:
    - torchvision pretrained resnet (recommended for immediate "analysis")
    - custom_resnet18 (architecture from ResNet.md) with optional checkpoint loading
    """

    def __init__(
        self,
        *,
        backend: str = "torchvision",
        arch: str = "resnet50",
        device: Optional[str] = None,
        labels: Optional[Sequence[str]] = None,
        labels_file: Optional[str] = None,
        custom_num_classes: int = 10,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        self.device = _default_device(device)

        if labels is not None and labels_file is not None:
            raise ValueError("Provide either labels or labels_file, not both")

        if labels_file is not None:
            labels = _read_labels_file(Path(labels_file))

        self.labels: Optional[List[str]] = list(labels) if labels is not None else None

        self.model, self.preprocess = self._load_model_and_preprocess(
            backend=backend,
            arch=arch,
            custom_num_classes=custom_num_classes,
            checkpoint_path=checkpoint_path,
        )
        self.model.to(self.device)
        self.model.eval()

        self._feature_hook_handle = None
        self._last_feature: Optional[torch.Tensor] = None
        self._install_feature_hook()

    def _load_model_and_preprocess(
        self,
        *,
        backend: str,
        arch: str,
        custom_num_classes: int,
        checkpoint_path: Optional[str],
    ) -> Tuple[nn.Module, Any]:
        backend = backend.lower().strip()
        arch = arch.lower().strip()

        if backend == "custom":
            model: nn.Module = custom_resnet18(num_classes=custom_num_classes)
            if checkpoint_path:
                ckpt = torch.load(checkpoint_path, map_location="cpu")
                state_dict = ckpt.get("state_dict", ckpt)
                model.load_state_dict(state_dict, strict=False)

            # CIFAR-style stem; for general images we still resize/crop to 224 by default.
            preprocess = self._fallback_preprocess()
            return model, preprocess

        if backend != "torchvision":
            raise ValueError("backend must be 'torchvision' or 'custom'")

        # torchvision path (pretrained ImageNet weights if available)
        try:
            from torchvision import models, transforms  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "torchvision is required for backend='torchvision'. Install torch+torchvision."
            ) from e

        weights = None
        categories: Optional[List[str]] = None

        # Try to use the modern Weights API when available.
        try:
            if arch == "resnet18":
                from torchvision.models import ResNet18_Weights  # type: ignore

                weights = ResNet18_Weights.DEFAULT
                model = models.resnet18(weights=weights)
            elif arch == "resnet34":
                from torchvision.models import ResNet34_Weights  # type: ignore

                weights = ResNet34_Weights.DEFAULT
                model = models.resnet34(weights=weights)
            elif arch == "resnet50":
                from torchvision.models import ResNet50_Weights  # type: ignore

                weights = ResNet50_Weights.DEFAULT
                model = models.resnet50(weights=weights)
            elif arch == "resnet101":
                from torchvision.models import ResNet101_Weights  # type: ignore

                weights = ResNet101_Weights.DEFAULT
                model = models.resnet101(weights=weights)
            elif arch == "resnet152":
                from torchvision.models import ResNet152_Weights  # type: ignore

                weights = ResNet152_Weights.DEFAULT
                model = models.resnet152(weights=weights)
            else:
                raise ValueError(
                    "arch must be one of: resnet18,resnet34,resnet50,resnet101,resnet152"
                )
        except Exception:
            # Fallback for older torchvision (pretrained=True path)
            if not hasattr(models, arch):
                raise ValueError(
                    "arch must be one of: resnet18,resnet34,resnet50,resnet101,resnet152"
                )
            fn = getattr(models, arch)
            model = fn(pretrained=True)  # type: ignore[arg-type]

        # Prefer weights-provided transforms & categories, but keep a safe fallback.
        preprocess = self._fallback_preprocess()
        if weights is not None:
            try:
                preprocess = weights.transforms()
            except Exception:
                preprocess = self._fallback_preprocess()
            try:
                meta = getattr(weights, "meta", None)
                if isinstance(meta, dict) and isinstance(meta.get("categories"), list):
                    categories = list(meta["categories"])
            except Exception:
                categories = None

        if self.labels is None and categories is not None:
            self.labels = categories

        return model, preprocess

    def _fallback_preprocess(self) -> Any:
        try:
            from torchvision import transforms  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "torchvision is required for preprocessing. Install torchvision."
            ) from e

        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def _install_feature_hook(self) -> None:
        """
        Save the pooled feature vector (before fc) for each forward pass.
        Works for torchvision ResNet and this repo's CustomResNet (both have avgpool).
        """

        if not hasattr(self.model, "avgpool"):
            return

        def hook(_module: nn.Module, _inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
            self._last_feature = out.detach()

        self._feature_hook_handle = getattr(self.model, "avgpool").register_forward_hook(hook)

    def close(self) -> None:
        if self._feature_hook_handle is not None:
            self._feature_hook_handle.remove()
            self._feature_hook_handle = None

    def _label_for(self, class_id: int) -> str:
        if self.labels and 0 <= class_id < len(self.labels):
            return self.labels[class_id]
        return f"class_{class_id}"

    @torch.inference_mode()
    def analyze_image(self, image_path: str, *, topk: int = 5, return_feature: bool = False) -> ImageAnalysis:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(str(path))

        img = _pil_open_rgb(path)
        x = self.preprocess(img).unsqueeze(0).to(self.device)

        self._last_feature = None
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0]

        k = min(topk, probs.numel())
        values, indices = torch.topk(probs, k=k)

        preds: List[Prediction] = []
        for p, idx in zip(values.tolist(), indices.tolist()):
            preds.append(Prediction(class_id=int(idx), label=self._label_for(int(idx)), probability=float(p)))

        feature: Optional[List[float]] = None
        if return_feature and self._last_feature is not None:
            feat = self._last_feature
            # avgpool output is typically (N,C,1,1); flatten to (C,)
            feat = torch.flatten(feat, 1)[0].to("cpu")
            feature = [float(v) for v in feat.tolist()]

        return ImageAnalysis(image_path=str(path), topk=preds, feature=feature)

    def analyze_images(
        self,
        image_paths: Iterable[str],
        *,
        topk: int = 5,
        return_feature: bool = False,
    ) -> List[ImageAnalysis]:
        results: List[ImageAnalysis] = []
        for p in image_paths:
            results.append(self.analyze_image(p, topk=topk, return_feature=return_feature))
        return results

    def analyze_dir(
        self,
        dir_path: str,
        *,
        topk: int = 5,
        return_feature: bool = False,
        exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    ) -> List[ImageAnalysis]:
        d = Path(dir_path)
        if not d.exists():
            raise FileNotFoundError(str(d))
        paths = [str(p) for p in sorted(d.rglob("*")) if p.suffix.lower() in set(exts)]
        return self.analyze_images(paths, topk=topk, return_feature=return_feature)

    @staticmethod
    def to_json(results: List[ImageAnalysis]) -> str:
        def pred_to_dict(p: Prediction) -> Dict[str, Any]:
            return {"class_id": p.class_id, "label": p.label, "probability": p.probability}

        payload = []
        for r in results:
            payload.append(
                {
                    "image_path": r.image_path,
                    "topk": [pred_to_dict(p) for p in r.topk],
                    "feature": r.feature,
                }
            )
        return json.dumps(payload, ensure_ascii=False, indent=2)

