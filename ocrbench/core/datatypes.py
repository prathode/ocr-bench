from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class OCRResult:
    """Normalized OCR result from any provider."""
    text: str
    confidence: Optional[float] = None
    bounding_boxes: Optional[List[Tuple[int, int, int, int]]] = field(default_factory=list)
    provider: str = ""
    processing_time_ms: float = 0.0


@dataclass
class DetectionResult:
    """Detection result from object detector."""
    bbox: Tuple[int, int, int, int]
    confidence: float
    label: Optional[str] = None
    class_id: Optional[int] = None


@dataclass
class BenchmarkResult:
    """Result from a full benchmark run."""
    provider_name: str
    total_images: int
    ocr_metrics: dict
    detection_metrics: Optional[dict] = None
    performance_metrics: dict = field(default_factory=dict)
    resource_metrics: dict = field(default_factory=dict)
    cost_metrics: dict = field(default_factory=dict)
    config: Optional[dict] = None
