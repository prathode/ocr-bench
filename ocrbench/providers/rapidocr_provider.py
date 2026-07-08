from typing import List
import numpy as np
import cv2
from pathlib import Path

from ocrbench.core.datatypes import OCRResult
from .base import LazyOCRProvider

try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
except ImportError:
    RAPIDOCR_AVAILABLE = False


class RapidOCRProvider(LazyOCRProvider):
    """RapidOCR provider (ONNX backend)."""

    def __init__(self, model_path: str = None):
        super().__init__("rapidocr")
        self.model_path = model_path
        self._ocr = None

    def initialize(self) -> None:
        if not RAPIDOCR_AVAILABLE:
            raise ImportError("rapidocr-onnxruntime not installed")
        self._ocr = RapidOCR(model_path=self.model_path)

    def _predict_internal(self, image: np.ndarray) -> OCRResult:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        
        result = self._ocr(image)
        
        text = ""
        confidence = None
        bboxes = []
        
        if result and result[0] is not None:
            for item in result[0]:
                if len(item) >= 3:
                    text += item[1]
                    if item[2] is not None:
                        confidence = item[2] if confidence is None else (confidence + item[2]) / 2
                    if item[0] and len(item[0]) >= 4:
                        bbox = item[0]
                        bboxes.append((
                            int(min(p[0] for p in bbox)),
                            int(min(p[1] for p in bbox)),
                            int(max(p[0] for p in bbox)),
                            int(max(p[1] for p in bbox))
                        ))
        
        return OCRResult(
            text=text,
            confidence=confidence,
            bounding_boxes=bboxes
        )

    def shutdown(self) -> None:
        self._ocr = None
        self._initialized = False
