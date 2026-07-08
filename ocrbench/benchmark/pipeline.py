from typing import List, Dict, Any
from pathlib import Path
import numpy as np

from ocrbench.providers import get_ocr_provider
from ocrbench.detectors import get_detector
from ocrbench.datasets import get_dataset
from ocrbench.metrics import (
    calculate_cer_batch,
    calculate_wer_batch,
    calculate_exact_match_batch,
    calculate_character_accuracy_batch,
    calculate_precision_recall_f1,
    calculate_fps,
    calculate_latency,
    get_memory_usage,
)
from ocrbench.experiments import MLflowTracker
from ocrbench.reports import ReportExporter


def run_benchmark(
    providers: List[str],
    dataset: str = "ufpr-alpr",
    mode: str = "ocr-only",
    batch_size: int = 8,
    confidence_threshold: float = 0.35
) -> List[Dict[str, Any]]:
    """Run OCR benchmark on specified providers."""
    runner = BenchmarkRunner(
        providers=providers,
        dataset_name=dataset,
        mode=mode,
        batch_size=batch_size,
        confidence_threshold=confidence_threshold
    )
    return runner.run()


class BenchmarkRunner:
    """Orchestrate OCR benchmark runs."""

    def __init__(
        self,
        providers: List[str],
        dataset_name: str = "ufpr-alpr",
        mode: str = "ocr-only",
        batch_size: int = 8,
        confidence_threshold: float = 0.35
    ):
        self.providers = providers
        self.dataset_name = dataset_name
        self.mode = mode
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.dataset = get_dataset(dataset_name, subset_size=50)
        self.tracker = MLflowTracker()

    def run(self) -> List[Dict[str, Any]]:
        self.dataset.load()
        images = self.dataset.get_images()
        ground_truths = self.dataset.get_ground_truth()

        results = []

        for provider_name in self.providers:
            result = self._run_provider(provider_name, images, ground_truths)
            results.append(result)

        return results

    def _run_provider(
        self,
        provider_name: str,
        images: List[np.ndarray],
        ground_truths: List[str]
    ) -> Dict[str, Any]:
        provider = get_ocr_provider(provider_name)
        
        run_id = self.tracker.start_run(run_name=f"{provider_name}_benchmark")
        
        processing_times = []
        predictions = []

        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            batch_results = provider.batch_predict(batch)
            
            for ocr_result in batch_results:
                processing_times.append(ocr_result.processing_time_ms)
                predictions.append(ocr_result.text)

        ocr_metrics = {
            **calculate_cer_batch(predictions, ground_truths),
            **calculate_wer_batch(predictions, ground_truths),
            **calculate_exact_match_batch(predictions, ground_truths),
            **calculate_character_accuracy_batch(predictions, ground_truths),
            **calculate_precision_recall_f1(predictions, ground_truths),
        }

        performance_metrics = {
            **calculate_latency(processing_times),
            "fps": calculate_fps(processing_times),
        }

        resource_metrics = get_memory_usage()

        self.tracker.log_params({
            "provider": provider_name,
            "dataset": self.dataset_name,
            "mode": self.mode,
            "batch_size": self.batch_size,
        })
        self.tracker.log_metrics(ocr_metrics)
        self.tracker.end_run()

        return {
            "provider_name": provider_name,
            "total_images": len(images),
            "ocr_metrics": ocr_metrics,
            "performance_metrics": performance_metrics,
            "resource_metrics": resource_metrics,
        }