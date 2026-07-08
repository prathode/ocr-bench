from typing import Optional, Dict, Type
from pathlib import Path

from .base import LazyOCRProvider
from .paddleocr_provider import PaddleOCRProvider
from .easyocr_provider import EasyOCRProvider
from .tesseract_provider import TesseractProvider
from .trocr_provider import TrOCRProvider
from .rapidocr_provider import RapidOCRProvider


OCR_REGISTRY: Dict[str, Type[LazyOCRProvider]] = {
    "paddleocr": PaddleOCRProvider,
    "easyocr": EasyOCRProvider,
    "tesseract": TesseractProvider,
    "trocr": TrOCRProvider,
    "rapidocr": RapidOCRProvider,
}


def get_ocr_provider(
    name: str,
    **kwargs
) -> LazyOCRProvider:
    """Get OCR provider by name."""
    if name not in OCR_REGISTRY:
        raise ValueError(
            f"Unknown OCR provider: {name}. "
            f"Available: {list(OCR_REGISTRY.keys())}"
        )
    return OCR_REGISTRY[name](**kwargs)
