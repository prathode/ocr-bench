import time
from typing import List
import numpy as np
from pathlib import Path

from ocrbench.core.interfaces import BaseDetector
from ocrbench.core.datatypes import DetectionResult


class LazyDetector(BaseDetector):
    """Base detector with lazy initialization."""

    def __init__(self, name: str):
        self.name = name
        self._initialized = False
        self._model = None

    def _ensure_initialized(self) -> None:
        """Lazy load model on first detect call."""
        if not self._initialized:
            self.initialize()
            self._initialized = True

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        self._ensure_initialized()
        return self._detect_internal(image)

    def batch_detect(self, images: List[np.ndarray]) -> List[List[DetectionResult]]:
        self._ensure_initialized()
        return [self.detect(img) for img in images]

    def _detect_internal(self, image: np.ndarray) -> List[DetectionResult]:
        """Override in subclass."""
        raise NotImplementedError

    def shutdown(self) -> None:
        self._model = None
        self._initialized = False
