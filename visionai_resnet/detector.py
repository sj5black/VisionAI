from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class DetectedObject:
    label: str
    score: float
    box_xyxy: Tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionResult:
    image_path: str
    objects: List[DetectedObject]
    object_types: List[str]


def _default_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pil_open_rgb(image_path: Path) -> Image.Image:
    with Image.open(image_path) as img:
        return img.convert("RGB")


def _candidate_search_dirs(p: Path) -> List[Path]:
    """
    A small set of common directories to search for typos/wrong paths.
    Keeps it conservative (no wide repo scan).
    """

    dirs: List[Path] = []
    if p.parent:
        dirs.append(p.parent)
    cwd = Path.cwd()
    dirs.append(cwd)
    dirs.append(cwd / "images")
    dirs.append(Path.home() / "images")

    # Deduplicate while preserving order, and keep only existing dirs
    uniq: List[Path] = []
    for d in dirs:
        if d.exists() and d.is_dir() and d not in uniq:
            uniq.append(d)
    return uniq


def _suggest_existing_paths(p: Path) -> List[Path]:
    """
    Suggest existing paths based on basename similarity.
    """

    target_name = p.name
    candidates: List[Path] = []
    for d in _candidate_search_dirs(p):
        try:
            for child in d.iterdir():
                if child.is_file() and child.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    candidates.append(child)
        except Exception:
            continue

    # Prefer exact basename matches first
    exact = [c for c in candidates if c.name == target_name]
    if exact:
        return exact[:5]

    names = [c.name for c in candidates]
    close = difflib.get_close_matches(target_name, names, n=5, cutoff=0.6)
    out: List[Path] = []
    for nm in close:
        for c in candidates:
            if c.name == nm:
                out.append(c)
                break
    return out


class ResNetObjectDetector:
    """
    Object detection using torchvision detection models with ResNet backbones.

    - fasterrcnn_resnet50_fpn_v2 (default): good accuracy, slower
    - retinanet_resnet50_fpn_v2: faster, a bit lower accuracy

    Outputs (COCO-style) bounding boxes + class labels + scores.
    """

    def __init__(
        self,
        *,
        model_name: str = "fasterrcnn_resnet50_fpn_v2",
        device: Optional[str] = None,
        labels: Optional[Sequence[str]] = None,
        labels_file: Optional[str] = None,
    ) -> None:
        if labels is not None and labels_file is not None:
            raise ValueError("Provide either labels or labels_file, not both")

        self.device = _default_device(device)
        self.labels: Optional[List[str]] = list(labels) if labels is not None else None
        if labels_file is not None:
            self.labels = self._read_labels_file(Path(labels_file))

        self.model, self.preprocess = self._load_model_and_preprocess(model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _read_labels_file(path: Path) -> List[str]:
        labels: List[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s:
                labels.append(s)
        return labels

    def _load_model_and_preprocess(self, model_name: str) -> Tuple[nn.Module, Any]:
        try:
            from torchvision import models, transforms  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("torchvision is required for object detection.") from e

        name = model_name.lower().strip()

        weights = None
        categories: Optional[List[str]] = None

        # Prefer modern Weights API, but keep compatibility with older torchvision.
        if name in ("fasterrcnn_resnet50_fpn_v2", "fasterrcnn"):
            try:
                from torchvision.models.detection import (  # type: ignore
                    FasterRCNN_ResNet50_FPN_V2_Weights,
                )

                weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                model = models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
            except Exception:
                # Fallback to v1
                try:
                    from torchvision.models.detection import (  # type: ignore
                        FasterRCNN_ResNet50_FPN_Weights,
                    )

                    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
                    model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
                except Exception:
                    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        elif name in ("retinanet_resnet50_fpn_v2", "retinanet"):
            try:
                from torchvision.models.detection import (  # type: ignore
                    RetinaNet_ResNet50_FPN_V2_Weights,
                )

                weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
                model = models.detection.retinanet_resnet50_fpn_v2(weights=weights)
            except Exception:
                try:
                    from torchvision.models.detection import (  # type: ignore
                        RetinaNet_ResNet50_FPN_Weights,
                    )

                    weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
                    model = models.detection.retinanet_resnet50_fpn(weights=weights)
                except Exception:
                    model = models.detection.retinanet_resnet50_fpn(pretrained=True)
        else:
            raise ValueError(
                "model_name must be one of: fasterrcnn_resnet50_fpn_v2, retinanet_resnet50_fpn_v2"
            )

        # Detection models expect float tensor in [0,1] (no manual Normalize needed).
        preprocess = transforms.Compose([transforms.ToTensor()])
        if weights is not None:
            try:
                preprocess = weights.transforms()
            except Exception:
                preprocess = transforms.Compose([transforms.ToTensor()])
            try:
                meta = getattr(weights, "meta", None)
                if isinstance(meta, dict) and isinstance(meta.get("categories"), list):
                    categories = list(meta["categories"])
            except Exception:
                categories = None

        if self.labels is None and categories is not None:
            self.labels = categories

        return model, preprocess

    def _label_for(self, class_id: int) -> str:
        # COCO models usually include a background class at index 0 depending on weights;
        # weights.meta['categories'] mapping aligns with the model's label indices.
        if self.labels and 0 <= class_id < len(self.labels):
            return self.labels[class_id]
        return f"class_{class_id}"

    @torch.inference_mode()
    def detect_image(
        self,
        image_path: str,
        *,
        score_threshold: float = 0.5,
        max_detections: int = 100,
    ) -> DetectionResult:
        path = Path(image_path)
        if not path.exists():
            suggestions = _suggest_existing_paths(path)
            hint = ""
            if suggestions:
                hint = " Did you mean one of: " + ", ".join(str(s) for s in suggestions)
            raise FileNotFoundError(f"{path} (not found).{hint}")

        img = _pil_open_rgb(path)
        x = self.preprocess(img).to(self.device)

        outputs = self.model([x])
        out = outputs[0]

        boxes = out.get("boxes")
        labels = out.get("labels")
        scores = out.get("scores")
        if boxes is None or labels is None or scores is None:
            raise RuntimeError("Model output missing boxes/labels/scores")

        objs: List[DetectedObject] = []
        keep = 0
        for box, lab, sc in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
            if sc < score_threshold:
                continue
            objs.append(
                DetectedObject(
                    label=self._label_for(int(lab)),
                    score=float(sc),
                    box_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                )
            )
            keep += 1
            if keep >= max_detections:
                break

        types = sorted({o.label for o in objs})
        return DetectionResult(image_path=str(path), objects=objs, object_types=types)

    def detect_dir(
        self,
        dir_path: str,
        *,
        score_threshold: float = 0.5,
        max_detections: int = 100,
        exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    ) -> List[DetectionResult]:
        d = Path(dir_path)
        if not d.exists():
            raise FileNotFoundError(str(d))
        paths = [p for p in sorted(d.rglob("*")) if p.suffix.lower() in set(exts)]
        return [
            self.detect_image(str(p), score_threshold=score_threshold, max_detections=max_detections)
            for p in paths
        ]

    @staticmethod
    def to_json(results: List[DetectionResult]) -> str:
        def obj_to_dict(o: DetectedObject) -> Dict[str, Any]:
            x1, y1, x2, y2 = o.box_xyxy
            return {"label": o.label, "score": o.score, "box_xyxy": [x1, y1, x2, y2]}

        payload = []
        for r in results:
            payload.append(
                {
                    "image_path": r.image_path,
                    "object_types": r.object_types,
                    "objects": [obj_to_dict(o) for o in r.objects],
                }
            )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @staticmethod
    def draw_boxes(
        image_path: str,
        result: DetectionResult,
        *,
        output_path: str,
        max_labels: int = 30,
    ) -> None:
        img = _pil_open_rgb(Path(image_path))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:  # pragma: no cover
            font = None

        for i, o in enumerate(result.objects):
            if i >= max_labels:
                break
            x1, y1, x2, y2 = o.box_xyxy
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            text = f"{o.label} {o.score:.2f}"
            if font is not None:
                # background for readability
                l, t, r, b = draw.textbbox((0, 0), text, font=font)
                tw, th = (r - l), (b - t)
                draw.rectangle([x1, y1, x1 + tw + 4, y1 + th + 4], fill="red")
                draw.text((x1 + 2, y1 + 2), text, fill="white", font=font)
            else:
                draw.text((x1 + 2, y1 + 2), text, fill="red")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)

