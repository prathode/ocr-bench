from typing import List, Dict, Any
from pathlib import Path
import numpy as np
import cv2

from ocrbench.detectors import get_detector
from ocrbench.providers import get_ocr_provider
from ocrbench.metrics import (
    calculate_iou,
    calculate_cer_batch,
    calculate_fps,
    calculate_latency,
)


class ALPRPipeline:
    """Full ALPR pipeline: detect + OCR."""

    def __init__(
        self,
        detector_name: str,
        ocr_provider_name: str,
        confidence_threshold: float = 0.35
    ):
        self.detector = get_detector(detector_name, confidence_threshold=confidence_threshold)
        self.ocr_provider = get_ocr_provider(ocr_provider_name)
        self.confidence_threshold = confidence_threshold

    def process(self, image: np.ndarray) -> Dict[str, Any]:
        detections = self.detector.detect(image)
        
        results = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            plate_img = image[y1:y2, x1:x2] if len(image.shape) == 3 else image[y1:y2, x1:x2]
            
            ocr_result = self.ocr_provider.predict(plate_img)
            results.append({
                "bbox": det.bbox,
                "confidence": det.confidence,
                "text": ocr_result.text,
                "ocr_confidence": ocr_result.confidence,
            })
        
        return {"detections": results}

    def batch_process(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        return [self.process(img) for img in images]
