from .base import LazyOCRProvider
from .paddleocr_provider import PaddleOCRProvider
from .easyocr_provider import EasyOCRProvider
from .tesseract_provider import TesseractProvider
from .trocr_provider import TrOCRProvider
from .rapidocr_provider import RapidOCRProvider
from .registry import OCR_REGISTRY, get_ocr_provider

__all__ = [
    "LazyOCRProvider",
    "PaddleOCRProvider",
    "EasyOCRProvider",
    "TesseractProvider",
    "TrOCRProvider",
    "RapidOCRProvider",
    "OCR_REGISTRY",
    "get_ocr_provider",
]
