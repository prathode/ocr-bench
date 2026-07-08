from typing import Type
from pathlib import Path

from .base import LazyDetector
from .yolo_provider import YOLOProvider
from .rtdetr_provider import RTDETRProvider


DETECTOR_REGISTRY = {
    "yolov8s": YOLOProvider,
    "yolov11s": YOLOProvider,
    "yolov12s": YOLOProvider,
    "rtdetr": RTDETRProvider,
}


def get_detector(
    name: str,
    **kwargs
) -> LazyDetector:
    """Get detector by name."""
    if name not in DETECTOR_REGISTRY:
        raise ValueError(
            f"Unknown detector: {name}. "
            f"Available: {list(DETECTOR_REGISTRY.keys())}"
        )
    return DETECTOR_REGISTRY[name](name=name, **kwargs)
