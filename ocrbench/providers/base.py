import time
from typing import List
import numpy as np
from pathlib import Path

from ocrbench.core.interfaces import BaseOCRProvider
from ocrbench.core.datatypes import OCRResult


class LazyOCRProvider(BaseOCRProvider):
    """Base provider with lazy initialization."""

    def __init__(self, name: str):
        self.name = name
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy load the model on first predict call."""
        if not self._initialized:
            self.initialize()
            self._initialized = True

    def predict(self, image: np.ndarray) -> OCRResult:
        start = time.time()
        self._ensure_initialized()
        result = self._predict_internal(image)
        result.provider = self.name
        result.processing_time_ms = (time.time() - start) * 1000
        return result

    def batch_predict(self, images: List[np.ndarray]) -> List[OCRResult]:
        self._ensure_initialized()
        return [self.predict(img) for img in images]

    def _predict_internal(self, image: np.ndarray) -> OCRResult:
        """Override in subclass."""
        raise NotImplementedError
