from typing import List, Tuple
import numpy as np
import cv2

from ocrbench.core.datatypes import OCRResult
from .base import LazyOCRProvider

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False


class PaddleOCRProvider(LazyOCRProvider):
    """PaddleOCR provider with lazy loading."""

    def __init__(self, use_angle_cls: bool = True, lang: str = "en"):
        super().__init__("paddleocr")
        self.use_angle_cls = use_angle_cls
        self.lang = lang
        self._ocr: PaddleOCR = None

    def initialize(self) -> None:
        if not PADDLE_AVAILABLE:
            raise ImportError("paddleocr not installed")
        self._ocr = PaddleOCR(
            use_angle_cls=self.use_angle_cls,
            lang=self.lang,
            show_log=False
        )

    def _predict_internal(self, image: np.ndarray) -> OCRResult:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        
        result = self._ocr.ocr(image, cls=True)
        
        text = ""
        confidence = None
        bboxes = []
        
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    text += line[1][0]
                    if len(line) >= 2 and isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                        confidence = line[1][1] if confidence is None else (confidence + line[1][1]) / 2
                    if len(line) >= 1 and isinstance(line[0], list):
                        bbox_points = line[0]
                        if len(bbox_points) >= 4:
                            x_coords = [p[0] for p in bbox_points]
                            y_coords = [p[1] for p in bbox_points]
                            bboxes.append((
                                int(min(x_coords)),
                                int(min(y_coords)),
                                int(max(x_coords)),
                                int(max(y_coords))
                            ))
        
        return OCRResult(
            text=text,
            confidence=confidence,
            bounding_boxes=bboxes
        )

    def shutdown(self) -> None:
        self._ocr = None
        self._initialized = False
