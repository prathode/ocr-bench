from typing import List
import numpy as np
import cv2

from ocrbench.core.datatypes import OCRResult
from .base import LazyOCRProvider

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class EasyOCRProvider(LazyOCRProvider):
    """EasyOCR provider with lazy loading."""

    def __init__(self, lang: List[str] = None, gpu: bool = True):
        super().__init__("easyocr")
        self.lang = lang or ["en"]
        self.gpu = gpu
        self._ocr = None

    def initialize(self) -> None:
        if not EASYOCR_AVAILABLE:
            raise ImportError("easyocr not installed")
        self._ocr = easyocr.Reader(self.lang, gpu=self.gpu)

    def _predict_internal(self, image: np.ndarray) -> OCRResult:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        result = self._ocr.readtext(image)
        
        text = ""
        confidences = []
        bboxes = []
        
        for item in result:
            if len(item) >= 3:
                text += item[1]
                confidences.append(item[2])
                if item[0] and len(item[0]) >= 4:
                    bbox = item[0]
                    bboxes.append((
                        int(min(p[0] for p in bbox)),
                        int(min(p[1] for p in bbox)),
                        int(max(p[0] for p in bbox)),
                        int(max(p[1] for p in bbox))
                    ))
        
        avg_conf = sum(confidences) / len(confidences) if confidences else None
        
        return OCRResult(
            text=text,
            confidence=avg_conf,
            bounding_boxes=bboxes
        )

    def shutdown(self) -> None:
        self._ocr = None
        self._initialized = False
