from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from visionai_resnet.detector import ResNetObjectDetector
from visionai_resnet.animal_insights import (
    AnimalBehaviorExpressionAnalyzer,
    COCO_ANIMALS,
)


ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
TEMPLATES_DIR = ROOT / "templates"
STATIC_DIR = ROOT / "static"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="VisionAI - Object Detection")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


_detectors: Dict[str, ResNetObjectDetector] = {}
_detector_lock = threading.Lock()

_animal_analyzer: Optional[AnimalBehaviorExpressionAnalyzer] = None
_animal_lock = threading.Lock()


def _get_detector(model_name: str, device: Optional[str]) -> ResNetObjectDetector:
    key = f"{model_name}||{device or ''}"
    cached = _detectors.get(key)
    if cached is not None:
        return cached
    with _detector_lock:
        cached = _detectors.get(key)
        if cached is None:
            cached = ResNetObjectDetector(model_name=model_name, device=device)
            _detectors[key] = cached
    return cached


def _get_animal_analyzer(device: Optional[str]) -> Optional[AnimalBehaviorExpressionAnalyzer]:
    """
    Lazy-load animal analyzer (OpenCLIP). If dependency/model load fails, return None.
    """

    if os.getenv("VISIONAI_ENABLE_ANIMAL_INSIGHTS", "1").strip() in {"0", "false", "False"}:
        return None

    global _animal_analyzer
    if _animal_analyzer is not None:
        return _animal_analyzer
    with _animal_lock:
        if _animal_analyzer is None:
            try:
                _animal_analyzer = AnimalBehaviorExpressionAnalyzer(device=device)
            except Exception:
                # Keep the API working even if open_clip isn't installed or weights can't load.
                _animal_analyzer = None
    return _animal_analyzer


def _safe_ext(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return ext
    return ".jpg"


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Any:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
        },
    )


@app.post("/api/detect")
def api_detect(
    image: UploadFile = File(...),
    threshold: float = Form(0.5),
    max_detections: int = Form(100),
    model: str = Form("fasterrcnn_resnet50_fpn_v2"),
) -> Dict[str, Any]:
    # Some clients (e.g., curl) may send application/octet-stream for images like .webp.
    # We allow it if the filename extension is an allowed image type.
    if image.content_type is not None and not image.content_type.startswith("image/"):
        ext = Path(image.filename or "").suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            raise HTTPException(status_code=400, detail="Only image uploads are supported.")

    if threshold < 0 or threshold > 1:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1.")
    if max_detections < 1 or max_detections > 300:
        raise HTTPException(status_code=400, detail="max_detections must be between 1 and 300.")
    if model not in {"fasterrcnn_resnet50_fpn_v2", "retinanet_resnet50_fpn_v2", "fasterrcnn", "retinanet"}:
        raise HTTPException(status_code=400, detail="Unsupported model.")

    file_id = uuid4().hex
    ext = _safe_ext(image.filename or "upload.jpg")
    upload_path = UPLOAD_DIR / f"{file_id}{ext}"
    annotated_path = OUTPUT_DIR / f"{file_id}.jpg"

    try:
        with upload_path.open("wb") as f:
            while True:
                chunk = image.file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}") from e

    device = os.getenv("VISIONAI_DEVICE")  # e.g. "cpu", "cuda", "cuda:0"
    detector = _get_detector(model_name=model, device=device)
    animal_analyzer = _get_animal_analyzer(device=device)

    try:
        result = detector.detect_image(
            str(upload_path),
            score_threshold=float(threshold),
            max_detections=int(max_detections),
        )
        detector.draw_boxes(str(upload_path), result, output_path=str(annotated_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}") from e

    pil_img: Optional[Image.Image] = None
    if animal_analyzer is not None:
        try:
            pil_img = Image.open(upload_path).convert("RGB")
        except Exception:
            pil_img = None

    objects: List[Dict[str, Any]] = []
    for o in result.objects:
        x1, y1, x2, y2 = o.box_xyxy
        obj: Dict[str, Any] = {
            "label": o.label,
            "score": o.score,
            "box_xyxy": [x1, y1, x2, y2],
        }

        # Extra analysis for animals (behavior/expression + "likely next actions").
        if animal_analyzer is not None and pil_img is not None and o.label in COCO_ANIMALS:
            try:
                crop = animal_analyzer.crop_from_image(pil_img, box_xyxy=o.box_xyxy)
                ins = animal_analyzer.analyze_crop(crop, species=o.label)
                obj["animal_insights"] = {
                    "species": ins.species,
                    "behavior": ins.behavior,
                    "behavior_confidence": ins.behavior_confidence,
                    "expression": ins.expression,
                    "expression_confidence": ins.expression_confidence,
                    "estimated_state": ins.estimated_state,
                    "predicted_next_actions": ins.predicted_next_actions,
                    "disclaimer": ins.disclaimer,
                }
            except Exception:
                # If one crop fails, do not fail the entire request.
                obj["animal_insights"] = {"error": "analysis_failed"}

        objects.append(obj)

    return {
        "id": file_id,
        "model": model,
        "threshold": threshold,
        "max_detections": max_detections,
        "object_types": result.object_types,
        "objects": objects,
        "animal_insights_enabled": animal_analyzer is not None,
        "original_image_url": f"/files/{file_id}/original",
        "annotated_image_url": f"/files/{file_id}/annotated",
    }


@app.get("/files/{file_id}/original")
def get_original(file_id: str) -> FileResponse:
    # Search for any allowed extension.
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        p = UPLOAD_DIR / f"{file_id}{ext}"
        if p.exists():
            return FileResponse(str(p))
    raise HTTPException(status_code=404, detail="Not found")


@app.get("/files/{file_id}/annotated")
def get_annotated(file_id: str) -> FileResponse:
    p = OUTPUT_DIR / f"{file_id}.jpg"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(p), media_type="image/jpeg")

