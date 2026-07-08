from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class BenchmarkRequest(BaseModel):
    providers: List[str] = ["paddleocr"]
    dataset: str = "ufpr-alpr"
    mode: str = "ocr-only"
    batch_size: int = 8
    confidence_threshold: float = 0.35


class BenchmarkResponse(BaseModel):
    run_id: str
    results: List[Dict[str, Any]]
    status: str = "completed"


class ProviderInfo(BaseModel):
    name: str
    type: str
    available: bool
    model_size_mb: Optional[float] = None


class TrainingRequest(BaseModel):
    mode: str = "ocr"
    model_name: Optional[str] = "trocr"
    dataset: str = "ufpr-alpr"
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4


class TrainingResponse(BaseModel):
    run_id: str
    results: Dict[str, Any]
    status: str = "completed"
