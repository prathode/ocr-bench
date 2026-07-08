# OCRBench

Open-source OCR benchmarking platform for license plate recognition (ALPR).

## Quick Start

```bash
docker compose up
```

Open http://localhost:7860 for the Gradio UI or http://localhost:8000 for the API.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

### Benchmarking

```python
from ocrbench.benchmark import run_benchmark

results = run_benchmark(
    providers=["paddleocr", "trocr", "easyocr"],
    dataset="ufpr-alpr",
    mode="ocr-only"
)
```

### Training
```python
from ocrbench.training import run_training

# Fine-tune TrOCR on custom dataset
results = run_training(
    mode="ocr",
    model_name="trocr",
    dataset="ufpr-alpr",
    epochs=10,
    batch_size=8,
    learning_rate=1e-4
)

# Fine-tune YOLO detector on custom dataset (supports yolov8s, yolov11s, yolov12s, rtdetr)
results = run_training(
    mode="detector",
    model_name="yolov8s",
    dataset="ufpr-alpr",
    epochs=20,
    batch_size=16,
    learning_rate=1e-4
)
```

## Features

- 5 OCR providers: PaddleOCR, EasyOCR, Tesseract, TrOCR, RapidOCR
- 4 detectors: YOLOv8s/v11s/v12s, RT-DETR
- Comprehensive metrics: CER, WER, IoU, mAP, FPS, latency
- MLflow experiment tracking
- Model training/fine-tuning (TrOCR, YOLO)
- 5 deployment profiles for recommendations (Real-Time, High-Accuracy, Low-Cost, Edge, Balanced)

## License

MIT
