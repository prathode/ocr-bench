from typing import List
import numpy as np
import cv2
from pathlib import Path

from ocrbench.core.datatypes import OCRResult
from .base import LazyOCRProvider

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False


class TrOCRProvider(LazyOCRProvider):
    """Microsoft TrOCR provider."""

    def __init__(self, model_name: str = "microsoft/trocr-base-printed"):
        super().__init__("trocr")
        self.model_name = model_name
        self._processor = None
        self._model = None

    def initialize(self) -> None:
        if not TROCR_AVAILABLE:
            raise ImportError("transformers or torch not installed")
        self._processor = TrOCRProcessor.from_pretrained(self.model_name)
        self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)

    def _predict_internal(self, image: np.ndarray) -> OCRResult:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        
        from PIL import Image
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        pixel_values = self._processor(pil_img, return_tensors="pt").pixel_values
        
        with torch.no_grad():
            generated_ids = self._model.generate(pixel_values)
        
        text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return OCRResult(
            text=text,
            confidence=None,
            bounding_boxes=[]
        )

    def shutdown(self) -> None:
        self._processor = None
        self._model = None
        self._initialized = False
