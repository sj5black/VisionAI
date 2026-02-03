"""
Step 1 & 2: Object Detection + Keypoint Detection using YOLOv8

YOLOv8n을 사용하여 경량화:
- 객체 탐지 (개/고양이 위치)
- Keypoint 탐지 (얼굴/신체 부위)
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import torch
import numpy as np


@dataclass
class Detection:
    """탐지된 객체 정보"""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    keypoints: Optional[np.ndarray] = None  # (N, 3) - x, y, confidence


class ObjectDetector:
    """
    YOLOv8 기반 객체 탐지 + Keypoint 탐지
    
    경량화를 위해 YOLOv8n (nano) 사용
    """
    
    def __init__(self, device: str = "auto"):
        """
        Args:
            device: 'auto', 'cpu', 'cuda', 'mps' 등
        """
        self.device = self._get_device(device)
        self.model = None
        self.pose_model = None
        self._load_models()
        
    def _get_device(self, device: str) -> str:
        """디바이스 자동 선택"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_models(self):
        """YOLOv8 모델 로드"""
        try:
            from ultralytics import YOLO
            # YOLOv8n - 가장 경량 모델 (6.3 MB)
            self.model = YOLO('yolov8n.pt')
            # YOLOv8n-pose - Keypoint 탐지용
            self.pose_model = YOLO('yolov8n-pose.pt')
            
            # 디바이스로 이동
            if self.device != "cpu":
                self.model.to(self.device)
                self.pose_model.to(self.device)
                
            print(f"✓ YOLOv8n 모델 로드 완료 (device: {self.device})")
        except ImportError:
            raise ImportError(
                "ultralytics 패키지가 필요합니다. "
                "설치: pip install ultralytics"
            )
    
    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.5,
        classes: Optional[List[int]] = None
    ) -> List[Detection]:
        """
        이미지에서 객체 탐지 + Keypoint 탐지
        
        Args:
            image: RGB 이미지 (H, W, 3)
            conf_threshold: 신뢰도 임계값
            classes: 탐지할 클래스 ID (None이면 전체, [15,16]=고양이/개)
            
        Returns:
            탐지된 객체 리스트
        """
        # 1단계: 객체 탐지
        results = self.model(image, conf=conf_threshold, classes=classes, verbose=False)
        
        # 2단계: Pose 탐지 (동물 객체에 대해)
        pose_results = self.pose_model(image, conf=conf_threshold, verbose=False)
        
        detections = []
        
        # 객체 탐지 결과 파싱
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
                
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Keypoint 찾기 (같은 영역의 pose)
                keypoints = None
                if len(pose_results) > 0 and pose_results[0].keypoints is not None:
                    kpts = pose_results[0].keypoints.data
                    if i < len(kpts):
                        keypoints = kpts[i].cpu().numpy()  # (17, 3) for COCO format
                
                detections.append(Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    keypoints=keypoints
                ))
        
        return detections
    
    def detect_animals(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.5
    ) -> List[Detection]:
        """
        개/고양이만 탐지 (COCO: 15=cat, 16=dog)
        
        Args:
            image: RGB 이미지
            conf_threshold: 신뢰도 임계값
            
        Returns:
            탐지된 동물 리스트
        """
        # COCO 클래스: 15=cat, 16=dog, 17=horse 등
        animal_classes = [15, 16, 17, 18, 19, 20, 21, 22, 23]  # 주요 동물들
        return self.detect(image, conf_threshold, classes=animal_classes)
    
    def visualize(
        self,
        image: np.ndarray,
        detections: List[Detection]
    ) -> np.ndarray:
        """
        탐지 결과 시각화
        
        Args:
            image: 원본 이미지
            detections: 탐지 결과
            
        Returns:
            시각화된 이미지
        """
        import cv2
        
        vis_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 바운딩 박스
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 라벨
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(
                vis_image, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
            
            # Keypoints
            if det.keypoints is not None:
                for kp in det.keypoints:
                    x, y, conf = kp
                    if conf > 0.5:  # 신뢰도 있는 keypoint만
                        cv2.circle(vis_image, (int(x), int(y)), 3, (255, 0, 0), -1)
        
        return vis_image
