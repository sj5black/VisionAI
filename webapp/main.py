from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

from visionai_resnet.detector import ResNetObjectDetector
from visionai_resnet.animal_insights import (
    AnimalBehaviorExpressionAnalyzer,
    COCO_ANIMALS,
)

# 🆕 새로운 VisionAI Pipeline
try:
    from visionai_pipeline import VisionAIPipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    VisionAIPipeline = None


ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
TEMPLATES_DIR = ROOT / "templates"
STATIC_DIR = ROOT / "static"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="VisionAI - Object Detection")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# 라운지(8004) 등 다른 포트에서 /api/emotion-backend 호출 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


_detectors: Dict[str, ResNetObjectDetector] = {}
_detector_lock = threading.Lock()

_animal_analyzer: Optional[AnimalBehaviorExpressionAnalyzer] = None
_animal_lock = threading.Lock()

# 🆕 VisionAI Pipeline (emotion_backend별 캐시: openclip / swin)
_pipeline_cache: Dict[tuple, Any] = {}
_pipeline_lock = threading.Lock()


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


def _get_pipeline(device: Optional[str], emotion_backend: str = "openclip") -> Optional[Any]:
    """
    🆕 Lazy-load VisionAI Pipeline. emotion_backend: "openclip" | "deepface" | "pyfaceau"
    """
    if not PIPELINE_AVAILABLE:
        return None
    backend = (emotion_backend or "openclip").strip().lower()
    if backend not in ("openclip", "deepface", "pyfaceau", "swin"):
        backend = "openclip"
    key = (device or "auto", backend)
    cached = _pipeline_cache.get(key)
    if cached is not None:
        return cached
    with _pipeline_lock:
        cached = _pipeline_cache.get(key)
        if cached is None:
            try:
                cached = VisionAIPipeline(
                    device=device or "auto",
                    enable_emotion=True,
                    enable_temporal=False,
                    enable_prediction=True,
                    emotion_backend=backend,
                )
                _pipeline_cache[key] = cached
            except Exception as e:
                print(f"Failed to load VisionAI Pipeline (backend={backend}): {e}")
                cached = None
    return cached


@app.get("/api/emotion-backend")
def api_emotion_backend(emotion_backend: Optional[str] = None) -> Dict[str, Any]:
    """
    라운지(room) 등 표시용. emotion_backend 미지정 시 openclip 기준.
    """
    backend = (emotion_backend or "openclip").strip().lower() if emotion_backend else "openclip"
    if backend not in ("openclip", "swin"):
        backend = "openclip"
    pipeline = _get_pipeline(os.getenv("VISIONAI_DEVICE"), emotion_backend=backend)
    if pipeline is None or pipeline.emotion_analyzer is None:
        return {"emotion_backend": None}
    name = getattr(
        pipeline.emotion_analyzer,
        "emotion_backend_name",
        getattr(pipeline.emotion_analyzer, "_emotion_backend_name", None),
    )
    return {"emotion_backend": name}


def _safe_ext(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return ext
    return ".jpg"


def _handle_pipeline_detection(
    image: UploadFile,
    threshold: float,
    max_detections: int,
    emotion_backend: str = "openclip",
) -> Dict[str, Any]:
    """
    🆕 VisionAI Pipeline으로 이미지 처리. emotion_backend: openclip | swin
    """
    import numpy as np
    from PIL import Image as PILImage
    
    device = os.getenv("VISIONAI_DEVICE")
    backend = (emotion_backend or "openclip").strip().lower()
    if backend not in ("openclip", "swin"):
        backend = "openclip"
    pipeline = _get_pipeline(device, emotion_backend=backend)
    
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="VisionAI Pipeline is not available. Install: pip install ultralytics"
        )
    
    file_id = uuid4().hex
    ext = _safe_ext(image.filename or "upload.jpg")
    upload_path = UPLOAD_DIR / f"{file_id}{ext}"
    annotated_path = OUTPUT_DIR / f"{file_id}.jpg"
    
    # 이미지 저장
    try:
        with upload_path.open("wb") as f:
            while True:
                chunk = image.file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}") from e
    
    # 이미지 로드
    try:
        pil_img = PILImage.open(upload_path).convert('RGB')
        image_np = np.array(pil_img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load image: {e}") from e
    
    # 파이프라인 실행
    try:
        result = pipeline.process_image(image_np, conf_threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}") from e
    
    # 시각화
    try:
        vis_image = pipeline.visualize(image_np, result)
        PILImage.fromarray(vis_image).convert("RGB").save(
            annotated_path, format="JPEG", quality=90
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visualization failed: {e}") from e
    
    # 결과 변환 (기존 포맷과 호환)
    object_types = list(set(d['class_name'] for d in result.detections))
    objects = []
    
    for i, det in enumerate(result.detections[:max_detections]):
        obj = {
            "label": det['class_name'],
            "score": det['confidence'],
            "box_xyxy": det['bbox'],
        }
        
        # 감정/자세 정보 추가
        if i < len(result.emotions):
            emotion = result.emotions[i]
            obj["pipeline_insights"] = {
                "emotion": emotion['emotion'],
                "emotion_confidence": emotion['emotion_confidence'],
                "pose": emotion['pose'],
                "pose_confidence": emotion['pose_confidence'],
                "combined_state": emotion['combined_state'],
            }
            
            # Temporal action (있으면)
            if result.action:
                obj["pipeline_insights"]["action"] = result.action['action']
                obj["pipeline_insights"]["action_confidence"] = result.action['confidence']
            
            # Prediction (있으면)
            if result.prediction:
                obj["pipeline_insights"]["predicted_action"] = result.prediction['predicted_action']
                obj["pipeline_insights"]["prediction_confidence"] = result.prediction['confidence']
                obj["pipeline_insights"]["alternative_actions"] = result.prediction['alternative_actions']
        
        objects.append(obj)
    
    return {
        "id": file_id,
        "model": "visionai_pipeline",
        "threshold": threshold,
        "max_detections": max_detections,
        "object_types": object_types,
        "objects": objects,
        "pipeline_enabled": True,
        "processing_time": result.processing_time,
        "emotion_backend": getattr(result, "emotion_backend", None),
        "original_image_url": f"/files/{file_id}/original",
        "annotated_image_url": f"/files/{file_id}/annotated",
    }


def _mood_summary_ko(emotion_counts: Dict[str, int], dominant_emotion: str) -> str:
    """감정 빈도 기반 짧은 기분 요약 문장 (한국어)."""
    if not dominant_emotion:
        return "분석된 표정이 없습니다."
    e = (dominant_emotion or "").strip().lower()
    # 감정별 한 줄 요약
    summaries = {
        "happy": "전반적으로 기쁘고 행복한 표정이 많습니다.",
        "relaxed": "편안하고 여유로운 분위기입니다.",
        "content": "만족스럽고 평온한 표정이 주를 이룹니다.",
        "curious": "호기심이나 관심이 느껴지는 표정이 많습니다.",
        "alert": "주의가 집중된, 경쾌한 표정이 보입니다.",
        "excited": "흥분되거나 기대에 찬 표정이 두드러집니다.",
        "playful": "장난스럽고 유쾌한 분위기입니다.",
        "sleepy": "졸리거나 나른한 표정이 많습니다.",
        "bored": "지루하거나 무관심한 표정이 보입니다.",
        "fearful": "불안하거나 두려운 표정이 있습니다.",
        "anxious": "걱정되거나 불안한 표정이 있습니다.",
        "stressed": "스트레스를 받은 듯한 표정이 보입니다.",
        "nervous": "긴장되거나 초조한 표정이 있습니다.",
        "aggressive": "공격적이거나 화가 난 표정이 있습니다.",
        "dominant": "자신감 있거나 주도적인 인상입니다.",
        "submissive": "복종하거나 위축된 표정이 있습니다.",
        "affectionate": "다정하고 애정 어린 표정이 많습니다.",
    }
    return summaries.get(e, f"전반적으로 '{dominant_emotion}'한 표정이 가장 많이 관찰되었습니다.")


def _handle_video_analysis(
    video: UploadFile,
    emotion_backend: str = "openclip",  # 영상 분석 모델 선택
    sample_fps: float = 2.0,
    max_duration_sec: float = 30.0,
    conf_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    짧은 영상을 프레임 단위로 샘플링해 표정·자세를 분석하고, 요약 반환.
    emotion_backend: "openclip" | "deepface" | "pyfaceau"
    """
    if not CV2_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Video analysis requires OpenCV (cv2). Install: pip install opencv-python",
        )
    
    backend = (emotion_backend or "openclip").strip().lower()
    
    # pyfaceau는 영상 파일을 직접 처리
    if backend == "pyfaceau":
        return _handle_video_analysis_pyfaceau(video, sample_fps, max_duration_sec)
    
    # openclip/deepface는 프레임별 처리
    return _handle_video_analysis_pipeline(video, backend, sample_fps, max_duration_sec, conf_threshold)


def _handle_video_analysis_pyfaceau(
    video: UploadFile,
    sample_fps: float = 2.0,
    max_duration_sec: float = 30.0,
) -> Dict[str, Any]:
    """pyfaceau로 영상 전체 분석 (빠른 방식)"""
    try:
        from visionai_pipeline.openface_analyzer import analyze_video, _au_to_expression, _head_pose_to_pose_label
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Video analysis requires pyfaceau. Install: pip install pyfaceau && python -m pyfaceau.download_weights",
        )

    file_id = uuid4().hex
    ext = (Path(video.filename or "").suffix or ".mp4").lower()
    if ext not in {".mp4", ".webm", ".mov", ".avi", ".mkv"}:
        ext = ".mp4"
    upload_path = UPLOAD_DIR / f"{file_id}{ext}"
    try:
        with upload_path.open("wb") as f:
            while True:
                chunk = video.file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}") from e

    # 영상 정보 확인
    cap = cv2.VideoCapture(str(upload_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video file. Unsupported format or corrupted file.")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / video_fps if video_fps > 0 else 0
    cap.release()

    # pyfaceau로 분석 (max_frames 제한)
    max_frames_to_analyze = int(min(duration_sec, max_duration_sec) * video_fps)
    start_wall = time.time()
    
    df = analyze_video(str(upload_path), max_frames=max_frames_to_analyze)
    
    if df is None or df.empty:
        raise HTTPException(status_code=500, detail="Video analysis failed. Check server logs.")
    
    processing_time = time.time() - start_wall
    
    # DataFrame에서 결과 추출
    all_emotions: List[str] = []
    all_poses: List[str] = []
    frame_results: List[Dict[str, Any]] = []
    
    # AU 컬럼 추출
    au_cols = [c for c in df.columns if c.startswith("AU") and c.endswith("_r")]
    
    for idx, row in df.iterrows():
        if not row.get("success", False):
            frame_results.append({
                "timestamp": round(row.get("timestamp", idx / video_fps), 2),
                "emotion": "—",
                "pose": "—",
            })
            continue
        
        # AU → expression
        au_dict = {col: float(row[col]) for col in au_cols if col in row}
        expression, _ = _au_to_expression(au_dict)
        all_emotions.append(expression)
        
        # head pose → pose_label
        if "pose_Tx" in row and "pose_Ty" in row and "pose_Tz" in row:
            head_pose = (float(row["pose_Tx"]), float(row["pose_Ty"]), float(row["pose_Tz"]))
            pose_label, _ = _head_pose_to_pose_label(head_pose)
        else:
            pose_label = "front"
        all_poses.append(pose_label)
        
        frame_results.append({
            "timestamp": round(row.get("timestamp", idx / video_fps), 2),
            "emotion": expression,
            "pose": pose_label,
        })
    
    # 통계 집계
    emotion_counts: Dict[str, int] = {}
    for x in all_emotions:
        if x and x != "—":
            emotion_counts[x] = emotion_counts.get(x, 0) + 1
    pose_counts: Dict[str, int] = {}
    for x in all_poses:
        if x and x != "—":
            pose_counts[x] = pose_counts.get(x, 0) + 1
    dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else None
    dominant_pose = max(pose_counts, key=pose_counts.get) if pose_counts else None
    mood_summary = _mood_summary_ko(emotion_counts, dominant_emotion or "")

    return {
        "video_analysis": True,
        "file_id": file_id,
        "duration_sec": round(duration_sec, 2),
        "frames_analyzed": len(frame_results),
        "processing_time_sec": round(processing_time, 2),
        "emotion_backend": "OpenFace 2.0 (pyfaceau)",
        "summary": {
            "dominant_emotion": dominant_emotion,
            "dominant_pose": dominant_pose,
            "emotion_counts": emotion_counts,
            "pose_counts": pose_counts,
            "mood_summary": mood_summary,
        },
        "frames": frame_results,
        "video_url": f"/files/{file_id}/video",
    }


def _handle_video_analysis_pipeline(
    video: UploadFile,
    emotion_backend: str,
    sample_fps: float = 2.0,
    max_duration_sec: float = 30.0,
    conf_threshold: float = 0.5,
) -> Dict[str, Any]:
    """pipeline.process_frame으로 영상 분석 (OpenCLIP/DeepFace 사용)"""
    if not PIPELINE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="VisionAI Pipeline is not available.",
        )
    
    device = os.getenv("VISIONAI_DEVICE")
    pipeline = _get_pipeline(device, emotion_backend=emotion_backend)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="VisionAI Pipeline is not available.")

    file_id = uuid4().hex
    ext = (Path(video.filename or "").suffix or ".mp4").lower()
    if ext not in {".mp4", ".webm", ".mov", ".avi", ".mkv"}:
        ext = ".mp4"
    upload_path = UPLOAD_DIR / f"{file_id}{ext}"
    try:
        with upload_path.open("wb") as f:
            while True:
                chunk = video.file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}") from e

    cap = cv2.VideoCapture(str(upload_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video file. Unsupported format or corrupted file.")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / video_fps if video_fps > 0 else 0
    cap.release()

    # 샘플링: max_duration_sec 초까지, sample_fps 간격으로
    max_frames_to_read = int(min(duration_sec, max_duration_sec) * video_fps)
    frame_interval = max(1, int(round(video_fps / sample_fps)))
    frames_to_process = []
    cap = cv2.VideoCapture(str(upload_path))
    frame_idx = 0
    while frame_idx < max_frames_to_read:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames_to_process.append((frame_idx / video_fps, frame_rgb))
        frame_idx += 1
    cap.release()

    if not frames_to_process:
        raise HTTPException(status_code=400, detail="No frames could be read from the video.")

    pipeline.reset()
    start_wall = time.time()
    emotion_backend_name = None
    all_emotions: List[str] = []
    all_poses: List[str] = []
    frame_results: List[Dict[str, Any]] = []

    for timestamp, frame_rgb in frames_to_process:
        result = pipeline.process_frame(frame_rgb, timestamp, conf_threshold=conf_threshold)
        if pipeline.emotion_analyzer is not None and emotion_backend_name is None:
            emotion_backend_name = getattr(
                pipeline.emotion_analyzer,
                "emotion_backend_name",
                getattr(pipeline.emotion_analyzer, "_emotion_backend_name", None),
            )
        for emo in result.emotions:
            all_emotions.append(emo.get("emotion") or "—")
            all_poses.append(emo.get("pose") or "—")
        if result.emotions:
            first = result.emotions[0]
            frame_results.append({
                "timestamp": round(timestamp, 2),
                "emotion": first.get("emotion") or "—",
                "pose": first.get("pose") or "—",
            })
        else:
            frame_results.append({"timestamp": round(timestamp, 2), "emotion": "—", "pose": "—"})

    processing_time = time.time() - start_wall
    emotion_counts: Dict[str, int] = {}
    for x in all_emotions:
        if x and x != "—":
            emotion_counts[x] = emotion_counts.get(x, 0) + 1
    pose_counts: Dict[str, int] = {}
    for x in all_poses:
        if x and x != "—":
            pose_counts[x] = pose_counts.get(x, 0) + 1
    dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else None
    dominant_pose = max(pose_counts, key=pose_counts.get) if pose_counts else None
    mood_summary = _mood_summary_ko(emotion_counts, dominant_emotion or "")

    return {
        "video_analysis": True,
        "file_id": file_id,
        "duration_sec": round(duration_sec, 2),
        "frames_analyzed": len(frames_to_process),
        "processing_time_sec": round(processing_time, 2),
        "emotion_backend": emotion_backend_name or emotion_backend,
        "summary": {
            "dominant_emotion": dominant_emotion,
            "dominant_pose": dominant_pose,
            "emotion_counts": emotion_counts,
            "pose_counts": pose_counts,
            "mood_summary": mood_summary,
        },
        "frames": frame_results,
        "video_url": f"/files/{file_id}/video",
    }


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
    use_pipeline: bool = Form(False),  # 🆕 파이프라인 사용 여부
    emotion_backend: str = Form("openclip"),  # 🆕 표정·자세 분석: openclip | deepface | pyfaceau
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
    
    # 🆕 파이프라인 모드
    if use_pipeline or model == "visionai_pipeline":
        return _handle_pipeline_detection(image, threshold, max_detections, emotion_backend=emotion_backend)
    
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


@app.get("/files/{file_id}/video")
def get_video(file_id: str) -> FileResponse:
    """업로드된 영상 파일 반환 (영상 분석용)."""
    for ext in [".mp4", ".webm", ".mov", ".avi", ".mkv"]:
        p = UPLOAD_DIR / f"{file_id}{ext}"
        if p.exists():
            return FileResponse(str(p), media_type="video/mp4" if ext == ".mp4" else "video/webm")
    raise HTTPException(status_code=404, detail="Not found")


@app.post("/api/analyze-video")
def api_analyze_video(
    video: UploadFile = File(...),
    emotion_backend: str = Form("openclip"),
    sample_fps: float = Form(2.0),
    max_duration_sec: float = Form(30.0),
    threshold: float = Form(0.5),
) -> Dict[str, Any]:
    """
    짧은 영상 업로드 후 프레임을 샘플링해 표정·자세를 분석하고 기분 요약을 반환합니다.
    """
    if not CV2_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Video analysis requires OpenCV (cv2). Install: pip install opencv-python",
        )
    if sample_fps < 0.5 or sample_fps > 10:
        raise HTTPException(status_code=400, detail="sample_fps must be between 0.5 and 10.")
    if max_duration_sec < 1 or max_duration_sec > 120:
        raise HTTPException(status_code=400, detail="max_duration_sec must be between 1 and 120.")
    if threshold < 0 or threshold > 1:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1.")
    return _handle_video_analysis(
        video,
        emotion_backend=emotion_backend,
        sample_fps=sample_fps,
        max_duration_sec=max_duration_sec,
        conf_threshold=threshold,
    )

