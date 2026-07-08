from typing import List
import numpy as np
import cv2

from ocrbench.core.datatypes import DetectionResult
from .base import LazyDetector

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class YOLOProvider(LazyDetector):
    """YOLOv8/v11/v12 detector provider."""

    def __init__(
        self,
        model_name: str = "yolov8s",
        model_path: str = None,
        confidence_threshold: float = 0.35
    ):
        super().__init__(model_name)
        self.confidence_threshold = confidence_threshold
        
        if model_path is None:
            model_path = f"{model_name}-license-plate"
        self.model_path = model_path

    def initialize(self) -> None:
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed")
        
        self._model = YOLO(self.model_path)

    def _detect_internal(self, image: np.ndarray) -> List[DetectionResult]:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        results = self._model(image, conf=self.confidence_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else None
                
                detections.append(DetectionResult(
                    bbox=(
                        int(xyxy[0]),
                        int(xyxy[1]),
                        int(xyxy[2]),
                        int(xyxy[3])
                    ),
                    confidence=float(confidence),
                    class_id=class_id,
                    label=f"plate_{class_id}" if class_id is not None else "plate"
                ))
        
        return detections

    def shutdown(self) -> None:
        if self._model is not None:
            del self._model
        self._model = None
        self._initialized = False
