from typing import List
import numpy as np
import cv2
import pytesseract

from ocrbench.core.datatypes import OCRResult
from .base import LazyOCRProvider


class TesseractProvider(LazyOCRProvider):
    """Tesseract OCR provider."""

    def __init__(self, lang: str = "eng"):
        super().__init__("tesseract")
        self.lang = lang

    def initialize(self) -> None:
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            raise ImportError("Tesseract not installed or not in PATH")

    def _predict_internal(self, image: np.ndarray) -> OCRResult:
        if len(image.shape) == 2:
            pass
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        
        text = pytesseract.image_to_string(image, lang=self.lang)
        
        data = pytesseract.image_to_data(image, lang=self.lang, output_type=pytesseract.Output.DICT)
        
        confidences = [int(c) for c in data["conf"] if c != -1]
        avg_conf = sum(confidences) / len(confidences) / 100 if confidences else None
        
        bboxes = []
        for i in range(len(data["text"])):
            if int(data["conf"][i]) != -1:
                bboxes.append((
                    data["left"][i],
                    data["top"][i],
                    data["left"][i] + data["width"][i],
                    data["top"][i] + data["height"][i]
                ))
        
        return OCRResult(
            text=text.strip(),
            confidence=avg_conf,
            bounding_boxes=bboxes
        )

    def shutdown(self) -> None:
        self._initialized = False
