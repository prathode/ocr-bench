from .interfaces import BaseOCRProvider, BaseDetector
from .datatypes import OCRResult, DetectionResult, BenchmarkResult
from .config import get_config, setup_config

__all__ = [
    "BaseOCRProvider",
    "BaseDetector",
    "OCRResult",
    "DetectionResult",
    "BenchmarkResult",
    "get_config",
    "setup_config",
]
