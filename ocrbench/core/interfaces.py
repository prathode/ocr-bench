from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from .datatypes import OCRResult, DetectionResult


class BaseOCRProvider(ABC):
    """Base interface for OCR providers."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the OCR provider (lazy loading)."""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> OCRResult:
        """Run OCR on a single image."""
        pass

    @abstractmethod
    def batch_predict(self, images: List[np.ndarray]) -> List[OCRResult]:
        """Run OCR on multiple images."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Cleanup resources."""
        pass


class BaseDetector(ABC):
    """Base interface for object detectors."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the detector (lazy loading)."""
        pass

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect objects in a single image."""
        pass

    @abstractmethod
    def batch_detect(self, images: List[np.ndarray]) -> List[List[DetectionResult]]:
        """Detect objects in multiple images."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Cleanup resources."""
        pass
