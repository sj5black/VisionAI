"""
OpenFace 2.0 (pyfaceau) 기반 영상 표정·행동 분석

- 영상 전용 (이미지는 deepface_analyzer 사용)
- 얼굴 위치, head pose, gaze, Action Units (AU) intensity/presence
- AU 조합 → 표정: 진짜 웃음(AU12+AU6), 가짜 웃음(AU12 only), 집중(AU4+AU7), 놀람(AU1+AU2) 등
- 추가 학습 없음, OpenFace 2.0 출력만 사용
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import os
import tempfile
from pathlib import Path

from .emotion_categories import get_state_from_emotion_pose

# OpenCV import (optional, for video writing)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# AU 강도 임계값 (0~5 스케일, OpenFace 2.0)
AU_THRESHOLD = 1.5


@dataclass
class OpenFaceResult:
    """OpenFace 2.0 단일 프레임 결과"""
    success: bool
    expression: str
    expression_confidence: float
    pose: str
    pose_confidence: float
    combined_state: str
    au_intensities: Optional[Dict[str, float]] = None
    head_pose: Optional[Tuple[float, float, float]] = None  # pitch, yaw, roll


def _get_weights_dir() -> Path:
    return Path(os.environ.get("PYFACEAU_WEIGHTS_DIR", os.path.expanduser("~/.pyfaceau/weights")))


def _au_value(au_dict: Optional[Dict[str, float]], key: str) -> float:
    """AU 강도 반환 (키 예: 'AU12_r', '12' 등)."""
    if not au_dict:
        return 0.0
    for k in (key, f"AU{key}_r", f"AU{key.zfill(2)}_r"):
        if k in au_dict:
            return float(au_dict[k])
    return 0.0


def _au_to_expression(au: Dict[str, float]) -> Tuple[str, float]:
    """
    AU 강도 조합 → 표정 라벨.
    - AU12 only (high), AU6 low → 가짜 웃음
    - AU12 + AU6 high → 진짜 웃음
    - AU4 + AU7 high → 집중
    - AU1 + AU2 high → 놀람
    - AU15, AU17 → 슬픔/불만
    - AU4 only → 찡그림/불쾌
    """
    t = AU_THRESHOLD
    au12 = _au_value(au, "12")
    au6 = _au_value(au, "6")
    au4 = _au_value(au, "4")
    au7 = _au_value(au, "7")
    au1 = _au_value(au, "1")
    au2 = _au_value(au, "2")
    au15 = _au_value(au, "15")
    au17 = _au_value(au, "17")

    if au12 >= t and au6 >= t:
        return "real_smile", min(1.0, (au12 + au6) / 10.0)
    if au12 >= t and au6 < t:
        return "fake_smile", min(1.0, au12 / 5.0)
    if au4 >= t and au7 >= t:
        return "focused", min(1.0, (au4 + au7) / 10.0)
    if au1 >= t and au2 >= t:
        return "surprised", min(1.0, (au1 + au2) / 10.0)
    if au15 >= t or au17 >= t:
        return "sad", min(1.0, max(au15, au17) / 5.0)
    if au4 >= t:
        return "displeased", min(1.0, au4 / 5.0)
    if _au_value(au, "5") >= t or _au_value(au, "26") >= t:
        return "attention", 0.7
    return "neutral", 0.5


def _head_pose_to_pose_label(pose: Optional[Tuple[float, float, float]]) -> Tuple[str, float]:
    """pitch, yaw, roll (라디안) → 자세 라벨."""
    if pose is None or len(pose) < 3:
        return "front", 0.5
    pitch, yaw, roll = pose[0], pose[1], pose[2]
    # 간단 규칙: pitch < -0.2 → looking_down, > 0.2 → looking_up, |yaw| > 0.3 → looking_side
    if pitch < -0.2:
        return "looking_down", 0.7
    if pitch > 0.2:
        return "looking_up", 0.7
    if abs(yaw) > 0.3:
        return "looking_side", 0.7
    return "front", 0.8


def analyze_video(video_path: str, max_frames: Optional[int] = None):
    """
    영상 파일을 pyfaceau로 분석하여 DataFrame 반환.
    
    Args:
        video_path: 영상 파일 경로
        max_frames: 최대 처리 프레임 수 (None이면 전체)
    
    Returns:
        pandas.DataFrame: frame, timestamp, success, AU01_r ~ AU45_r, pose_Tx, pose_Ty, pose_Tz 등
    """
    try:
        from pyfaceau import FullPythonAUPipeline
    except ImportError:
        print("⚠ 영상 표정 분석을 위해 pyfaceau가 필요합니다: pip install pyfaceau")
        return None
    
    weights_dir = _get_weights_dir()
    
    try:
        pipeline = FullPythonAUPipeline(
            pdm_file=str(weights_dir / "In-the-wild_aligned_PDM_68.txt"),
            au_models_dir=str(weights_dir / "AU_predictors"),
            triangulation_file=str(weights_dir / "tris_68_full.txt"),
            patch_expert_file=str(weights_dir / "svr_patches_0.25_general.txt"),
        )
        
        df = pipeline.process_video(video_path, max_frames=max_frames)
        return df
    
    except Exception as e:
        print(f"⚠ pyfaceau 영상 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_frame(
    image: np.ndarray,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> OpenFaceResult:
    """
    단일 이미지(또는 얼굴 영역)에 대해 OpenFace 2.0 AU 분석 후 표정·자세 반환.
    
    pyfaceau는 비디오 전용이므로, 이미지를 1프레임 비디오로 변환 후 처리.

    Args:
        image: RGB (H, W, 3)
        bbox: (x1, y1, x2, y2) 사람 bbox. None이면 전체 이미지.
    """
    if not CV2_AVAILABLE:
        if not getattr(analyze_frame, "_warned_cv2", False):
            print("⚠ OpenFace 2.0 분석을 위해 opencv-python이 필요합니다: pip install opencv-python")
            analyze_frame._warned_cv2 = True
        return OpenFaceResult(
            success=False,
            expression="neutral",
            expression_confidence=0.0,
            pose="front",
            pose_confidence=0.0,
            combined_state="neutral",
            au_intensities=None,
            head_pose=None,
        )

    try:
        from pyfaceau import FullPythonAUPipeline
    except ImportError:
        if not getattr(analyze_frame, "_warned_pyfaceau", False):
            print("⚠ OpenFace 2.0 표정 분석을 위해 pyfaceau가 필요합니다: pip install pyfaceau")
            print("  설치 후 가중치 다운로드: python -m pyfaceau.download_weights")
            analyze_frame._warned_pyfaceau = True
        return OpenFaceResult(
            success=False,
            expression="neutral",
            expression_confidence=0.0,
            pose="front",
            pose_confidence=0.0,
            combined_state="neutral",
            au_intensities=None,
            head_pose=None,
        )

    if bbox is not None:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        h, w = image.shape[:2]
        pad = 0.1
        x1 = max(0, x1 - int((x2 - x1) * pad))
        y1 = max(0, y1 - int((y2 - y1) * pad))
        x2 = min(w, x2 + int((x2 - x1) * pad))
        y2 = min(h, y2 + int((y2 - y1) * pad))
        roi = image[y1:y2, x1:x2]
    else:
        roi = image

    if roi.size == 0:
        return OpenFaceResult(
            success=False,
            expression="neutral",
            expression_confidence=0.0,
            pose="front",
            pose_confidence=0.0,
            combined_state="neutral",
        )

    # FullPythonAUPipeline 초기화 (캐싱)
    if not hasattr(analyze_frame, "_pipeline"):
        try:
            weights_dir = _get_weights_dir()
            analyze_frame._pipeline = FullPythonAUPipeline(
                pdm_file=str(weights_dir / "In-the-wild_aligned_PDM_68.txt"),
                au_models_dir=str(weights_dir / "AU_predictors"),
                triangulation_file=str(weights_dir / "tris_68_full.txt"),
                patch_expert_file=str(weights_dir / "svr_patches_0.25_general.txt"),
            )
            if not getattr(analyze_frame, "_init_logged", False):
                print("✓ OpenFace 2.0 FullPythonAUPipeline 초기화 완료")
                analyze_frame._init_logged = True
        except Exception as e:
            if not getattr(analyze_frame, "_warned_init", False):
                print(f"⚠ OpenFace 2.0 초기화 실패: {e}")
                print("  가중치 다운로드: python -m pyfaceau.download_weights")
                analyze_frame._warned_init = True
            return OpenFaceResult(
                success=False,
                expression="neutral",
                expression_confidence=0.0,
                pose="front",
                pose_confidence=0.0,
                combined_state="neutral",
            )

    try:
        # 1. 이미지를 1프레임 비디오로 저장
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
            temp_video_path = tmpfile.name
        
        # RGB → BGR 변환 (opencv는 BGR 사용)
        bgr_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
        h, w = bgr_roi.shape[:2]
        
        # 비디오 writer 생성 (1 frame, 1 fps)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, 1.0, (w, h))
        out.write(bgr_roi)
        out.release()
        
        # 2. pyfaceau로 처리
        df = analyze_frame._pipeline.process_video(temp_video_path, max_frames=1)
        
        # 3. 임시 파일 삭제
        os.unlink(temp_video_path)
        
        # 4. 결과 추출
        if df.empty or not df.iloc[0].get("success", False):
            return OpenFaceResult(
                success=False,
                expression="neutral",
                expression_confidence=0.0,
                pose="front",
                pose_confidence=0.0,
                combined_state="neutral",
            )
        
        row = df.iloc[0]
        
        # AU intensities 추출
        au_cols = [c for c in df.columns if c.startswith("AU") and c.endswith("_r")]
        au_dict = {col: float(row[col]) for col in au_cols if col in row}
        
        # head pose 추출
        if "pose_Tx" in row and "pose_Ty" in row and "pose_Tz" in row:
            head_pose = (float(row["pose_Tx"]), float(row["pose_Ty"]), float(row["pose_Tz"]))
        else:
            head_pose = None
        
        expression, expr_conf = _au_to_expression(au_dict)
        pose_label, pose_conf = _head_pose_to_pose_label(head_pose)
        combined_state = get_state_from_emotion_pose(expression, pose_label)
        
        return OpenFaceResult(
            success=True,
            expression=expression,
            expression_confidence=expr_conf,
            pose=pose_label,
            pose_confidence=pose_conf,
            combined_state=combined_state,
            au_intensities=au_dict,
            head_pose=head_pose,
        )
    
    except Exception as e:
        if not getattr(analyze_frame, "_warned_analyze", False):
            print(f"⚠ OpenFace 2.0 분석 실패: {e}")
            import traceback
            traceback.print_exc()
            analyze_frame._warned_analyze = True
        # 임시 파일 정리
        try:
            if 'temp_video_path' in locals():
                os.unlink(temp_video_path)
        except:
            pass
        return OpenFaceResult(
            success=False,
            expression="neutral",
            expression_confidence=0.0,
            pose="front",
            pose_confidence=0.0,
            combined_state="neutral",
        )
