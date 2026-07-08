from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schemas import BenchmarkRequest, BenchmarkResponse, ProviderInfo, TrainingRequest, TrainingResponse
from ocrbench.providers import OCR_REGISTRY
from ocrbench.detectors import DETECTOR_REGISTRY
from ocrbench.benchmark import run_benchmark
from ocrbench.training import run_training

app = FastAPI(
    title="OCRBench API",
    description="Open-source OCR benchmarking API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/providers", response_model=List[ProviderInfo])
def list_providers():
    providers = []
    for name in OCR_REGISTRY.keys():
        providers.append(ProviderInfo(
            name=name,
            type="ocr",
            available=True
        ))
    for name in DETECTOR_REGISTRY.keys():
        providers.append(ProviderInfo(
            name=name,
            type="detector",
            available=True
        ))
    return providers


@app.post("/benchmark", response_model=BenchmarkResponse)
def run_benchmark_endpoint(request: BenchmarkRequest):
    results = run_benchmark(
        providers=request.providers,
        dataset=request.dataset,
        mode=request.mode,
        batch_size=request.batch_size,
        confidence_threshold=request.confidence_threshold
    )
    return BenchmarkResponse(
        run_id="local",
        results=results,
        status="completed"
    )


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/train", response_model=TrainingResponse)
def run_training_endpoint(request: TrainingRequest):
    results = run_training(
        mode=request.mode,
        model_name=request.model_name,
        dataset=request.dataset,
        epochs=request.epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate
    )
    return TrainingResponse(
        run_id="local",
        results=results,
        status="completed"
    )
